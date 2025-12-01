# -*- coding: utf-8 -*-
"""
Pipeline de Análisis Heurístico de Postura para Gradio.
Consolidación de todas las funciones heurísticas, optimización de velocidad 
(Frame Skipping) y optimización de memoria (No almacena frames BGR).
"""

import cv2
import numpy as np
import pandas as pd
import os
import time
import re
import mediapipe as mp
from numpy.linalg import norm
from typing import List, Optional, Dict, Tuple, Callable
# Importar Gradio si está disponible para el tipo de progreso
try:
    import gradio as gr
except ImportError:
    # Definición mínima para que el script pueda correr sin Gradio instalado
    class DummyProgress:
        def __call__(self, *args, **kwargs):
            pass
    gr = None

# ====================================================================
## ⚙️ CONSTANTES Y CONFIGURACIÓN
# ====================================================================

# Directorios de Salida (Se usarán rutas temporales en Gradio)
OUTPUT_DIR_MP = 'temp_mediapipe'
os.makedirs(OUTPUT_DIR_MP, exist_ok=True)

# ⚡ OPTIMIZACIÓN DE VELOCIDAD: Factor de Salto de Fotogramas
FRAME_SKIP_FACTOR = 2 # Procesar 1 de cada 10 frames para la pose

# Parámetros de Análisis
FPS = 30
THICKNESS = 2
RADIUS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Mapeo de Índices de MediaPipe (33 Landmarks) ---
MP_JOINTS = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'NOSE': 0
}

# --- UMBRALES HEURÍSTICOS (Consolidación de los valores de tu script) ---
UMBRAL_VELOCIDAD_CUANTIL = 90
UMBRAL_VELOCIDAD_ESTATICA = 0.0005
UMBRAL_AFC_PASIVA_ESTIRADA = 160.0
UMBRAL_AEB_PASIVA_MAX = 15.0
UMBRAL_SM_CERRADA = 0.50
UMBRAL_DCM_CERRADA = 1.1
UMBRAL_PROXIMIDAD_Y_TORSO_RATIO = 0.30
UMBRAL_ALTURA_TORSO_Y_ACTIVA = 0.75
UMBRAL_AEB_ACTIVA = 60.0
UMBRAL_AFC_ACTIVA = 160.0
UMBRAL_AFC_PASIVA_FLEXIBLE = 160.0 
UMBRAL_AEB_PASIVA_FLEXIBLE = 35.0

# --- CONSTANTES DE COLOR
COLOR_ABIERTA = (0, 255, 0)      # Verde
COLOR_PASIVA = (0, 165, 255)     # Naranja/Azul Intermedio
COLOR_CERRADA = (0, 0, 255)      # Rojo
LABEL_COLORS = {
    "Abierta": COLOR_ABIERTA,
    "Pasiva": COLOR_PASIVA,
    "Cerrada": COLOR_CERRADA
}

# Conexiones principales del esqueleto (Para dibujar)
BONE_CONNECTIONS = [
    (11, 13), (13, 15), # Brazo Izquierdo
    (12, 14), (14, 16), # Brazo Derecho
    (11, 12), (11, 23), (12, 24), (23, 24), # Tronco
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8)
]

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ====================================================================
## 1. EXTRACCIÓN Y OPTIMIZACIÓN DE POSE (MediaPipe)
# ====================================================================

def extract_mediapipe_keypoints_optimized(video_path, pose_model, progress: Optional[Callable] = None):
    """
    Procesa el video, aplicando Frame Skipping, y retorna la matriz de pose (Fx33x3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ ERROR: No se pudo abrir el archivo de video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames_landmarks = []
    last_valid_pose = np.zeros((33, 3)) # Inicializar con ceros
    frame_count = 0

    print(f"Iniciando extracción de pose (Skip={FRAME_SKIP_FACTOR})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_pose = None
        
        # ⚡ OPTIMIZACIÓN: Solo procesar el esqueleto si es un frame clave
        if frame_count % FRAME_SKIP_FACTOR == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False 
            results = pose_model.process(image)
            image.flags.writeable = True
            
            frame_landmarks = []
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks.append([landmark.x, landmark.y, landmark.z])
                
            if frame_landmarks:
                 current_pose = np.array(frame_landmarks, dtype=np.float32)
                 last_valid_pose = current_pose # Actualizar la última pose
                 all_frames_landmarks.append(current_pose)
            else:
                 # Si no se detecta NADA, usar la última pose válida o ceros
                 all_frames_landmarks.append(last_valid_pose)

        # Para frames saltados, repetir la última pose válida
        else:
             all_frames_landmarks.append(last_valid_pose)

        frame_count += 1
        
        # Reporte de progreso
        if progress and total_frames > 0 and frame_count % 100 == 0:
            percentage = min(0.70, 0.05 + 0.65 * (frame_count / total_frames)) # Del 5% al 70%
            progress(percentage, desc=f"Paso 1/4: Extrayendo Pose (MediaPipe) - {frame_count} frames")


    cap.release()
    print(f"Extracción de pose completada. Total frames: {frame_count}")

    # Retornar la matriz de pose (Frames x 33 x 3)
    return np.array(all_frames_landmarks)

# ====================================================================
## 2. FUNCIONES DE CÁLCULO HEURÍSTICO (Angulos, Distancias, Clasificación)
# ====================================================================

def calculate_angle(p1, p2, p3):
    """Calcula el ángulo (en grados) entre 3 puntos (X, Y)."""
    p1 = np.array(p1[:2]); p2 = np.array(p2[:2]); p3 = np.array(p3[:2])
    v1 = p1 - p2; v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1); norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 180.0
    cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def calculate_normalized_distance(p_wrist, p_shoulder_l, p_shoulder_r, p_hip_l, p_hip_r):
    """Distancia Muñeca al Centro del Torso (CT), normalizada por longitud del torso."""
    mid_shoulder = (p_shoulder_l + p_shoulder_r) / 2
    mid_hip = (p_hip_l + p_hip_r) / 2
    torso_length = np.linalg.norm(mid_shoulder[:2] - mid_hip[:2])
    center_torso = mid_shoulder
    dist_wrist_to_ct = np.linalg.norm(p_wrist[:2] - center_torso[:2])
    if torso_length < 1e-6: return 0.0
    return dist_wrist_to_ct / torso_length

def classify_hand_posture(frame_pose_3d, joints_map):
    """Clasifica la postura de las manos/brazos (Abierta/Pasiva/Cerrada) con heurísticas."""
    P = {};
    for name, idx in joints_map.items(): P[name] = frame_pose_3d[idx]

    if np.all(frame_pose_3d == 0): return "Cerrada", LABEL_COLORS['Cerrada']

    shoulder_width = np.linalg.norm(P['RIGHT_SHOULDER'][:2] - P['LEFT_SHOULDER'][:2])
    if shoulder_width < 1e-6: shoulder_width = 1e-6
    P_MID_HIP = (P['LEFT_HIP'] + P['RIGHT_HIP']) / 2
    Y_REF_CADERA = P_MID_HIP[1]
    P_MID_SHOULDER = (P['LEFT_SHOULDER'] + P['RIGHT_SHOULDER']) / 2
    Torso_Length = np.linalg.norm(P_MID_SHOULDER[:2] - P_MID_HIP[:2])
    if Torso_Length < 1e-6: Torso_Length = 1e-6
    
    SM = np.linalg.norm(P['RIGHT_WRIST'][:2] - P['LEFT_WRIST'][:2]) / shoulder_width
    AEB_L = calculate_angle(P['LEFT_SHOULDER'].copy(), P['LEFT_SHOULDER'], P['LEFT_ELBOW'])
    AEB_R = calculate_angle(P['RIGHT_SHOULDER'].copy(), P['RIGHT_SHOULDER'], P['RIGHT_ELBOW'])
    AFC_L = calculate_angle(P['LEFT_SHOULDER'], P['LEFT_ELBOW'], P['LEFT_WRIST'])
    AFC_R = calculate_angle(P['RIGHT_SHOULDER'], P['RIGHT_ELBOW'], P['RIGHT_WRIST'])
    DCM_L = calculate_normalized_distance(P['LEFT_WRIST'], P['LEFT_SHOULDER'], P['RIGHT_SHOULDER'], P['LEFT_HIP'], P['RIGHT_HIP'])
    DCM_R = calculate_normalized_distance(P['RIGHT_WRIST'], P['LEFT_SHOULDER'], P['RIGHT_SHOULDER'], P['LEFT_HIP'], P['RIGHT_HIP'])
    
    wrist_height_ratio_L = (P['LEFT_WRIST'][1] - P_MID_SHOULDER[1]) / Torso_Length
    wrist_height_ratio_R = (P['RIGHT_WRIST'][1] - P_MID_SHOULDER[1]) / Torso_Length
    is_wrist_high_L = wrist_height_ratio_L < UMBRAL_ALTURA_TORSO_Y_ACTIVA
    is_wrist_high_R = wrist_height_ratio_R < UMBRAL_ALTURA_TORSO_Y_ACTIVA
    
    diff_Y_L_cadera = abs(P['LEFT_WRIST'][1] - Y_REF_CADERA)
    is_wrist_at_hip_level_torso_L = (diff_Y_L_cadera / Torso_Length) < UMBRAL_PROXIMIDAD_Y_TORSO_RATIO
    diff_Y_R_cadera = abs(P['RIGHT_WRIST'][1] - Y_REF_CADERA)
    is_wrist_at_hip_level_torso_R = (diff_Y_R_cadera / Torso_Length) < UMBRAL_PROXIMIDAD_Y_TORSO_RATIO
    
    is_pasiva_pura_L = (AFC_L > UMBRAL_AFC_PASIVA_ESTIRADA) and (AEB_L < UMBRAL_AEB_PASIVA_MAX)
    is_pasiva_pura_R = (AFC_R > UMBRAL_AFC_PASIVA_ESTIRADA) and (AEB_R < UMBRAL_AEB_PASIVA_MAX)
    is_pasiva_relajada_L = (AFC_L > UMBRAL_AFC_PASIVA_FLEXIBLE) and (AEB_L < UMBRAL_AEB_PASIVA_FLEXIBLE)
    is_pasiva_relajada_R = (AFC_R > UMBRAL_AFC_PASIVA_FLEXIBLE) and (AEB_R < UMBRAL_AEB_PASIVA_FLEXIBLE)

    # CLASIFICACIÓN
    if is_pasiva_pura_L and is_pasiva_pura_R: return "Pasiva", LABEL_COLORS['Pasiva']
    
    is_cerrada_base = (SM < UMBRAL_SM_CERRADA) and (DCM_L < UMBRAL_DCM_CERRADA) and (DCM_R < UMBRAL_DCM_CERRADA)
    if is_cerrada_base and is_wrist_at_hip_level_torso_L and is_wrist_at_hip_level_torso_R:
      return "Cerrada", LABEL_COLORS['Cerrada']

    is_activa_L = (is_wrist_high_L) or (AEB_L < UMBRAL_AEB_ACTIVA) or (AFC_L < UMBRAL_AFC_ACTIVA)
    is_activa_R = (is_wrist_high_R) or (AEB_R < UMBRAL_AEB_ACTIVA) or (AFC_R < UMBRAL_AFC_ACTIVA)
    if is_activa_L or is_activa_R: return "Abierta", LABEL_COLORS['Abierta']

    if is_pasiva_relajada_L and is_pasiva_relajada_R: return "Pasiva", LABEL_COLORS['Pasiva']
    
    return "Pasiva", LABEL_COLORS['Pasiva']

def analyze_body_displacement(pose_data_3d, joints_map):
    """Analiza la velocidad del centro de la cadera para obtener el porcentaje estático."""
    L_HIP, R_HIP = joints_map['LEFT_HIP'], joints_map['RIGHT_HIP']
    L_SHOULDER, R_SHOULDER = joints_map['LEFT_SHOULDER'], joints_map['RIGHT_SHOULDER']

    mid_hip = (pose_data_3d[:, L_HIP] + pose_data_3d[:, R_HIP]) / 2

    mid_shoulder = (pose_data_3d[:, L_SHOULDER] + pose_data_3d[:, R_SHOULDER]) / 2
    torso_lengths = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    valid_frames_mask = torso_lengths > 1e-6
    torso_size_avg = np.mean(torso_lengths[valid_frames_mask])

    if torso_size_avg < 1e-6 or pose_data_3d.shape[0] <= 1:
        return 0.0, "Datos No Válidos", 0.0, 0.0

    mid_hip_valid = mid_hip[valid_frames_mask]
    displacement_per_frame = np.linalg.norm(mid_hip_valid[1:, :2] - mid_hip_valid[:-1, :2], axis=1)
    normalized_velocity = displacement_per_frame / torso_size_avg

    static_intervals = np.sum(normalized_velocity < UMBRAL_VELOCIDAD_ESTATICA)
    total_intervals = len(normalized_velocity)

    static_percentage = ((static_intervals + 1) / (total_intervals + 1)) * 100
    total_normalized_displacement = np.sum(normalized_velocity)

    movement_category = "Estático" if static_percentage > 85 else "Desplazamiento constante"
    
    initial_pos = mid_hip[0]
    final_pos = mid_hip[-1]
    final_position_change = np.linalg.norm(final_pos - initial_pos) / torso_size_avg

    return static_percentage, movement_category, total_normalized_displacement, final_position_change

def analyze_torso_rigidity(pose_data_3d, joints_map):
    """Evalúa la soltura/rigidez midiendo la distancia (longitud) del torso."""
    rigidity_level = "Datos Insuficientes"; variance = 0.0; percent_change = 0.0

    L_SHOULDER, R_SHOULDER = joints_map['LEFT_SHOULDER'], joints_map['RIGHT_SHOULDER']
    L_HIP, R_HIP = joints_map['LEFT_HIP'], joints_map['RIGHT_HIP']

    shoulder_dist = np.linalg.norm(pose_data_3d[:, L_SHOULDER] - pose_data_3d[:, R_SHOULDER], axis=1)
    shoulder_dist_safe = shoulder_dist.copy()
    shoulder_dist_safe[shoulder_dist_safe < 1e-6] = 1e-6

    mid_shoulder = (pose_data_3d[:, L_SHOULDER] + pose_data_3d[:, R_SHOULDER]) / 2
    mid_hip = (pose_data_3d[:, L_HIP] + pose_data_3d[:, R_HIP]) / 2
    torso_length = np.linalg.norm(mid_shoulder - mid_hip, axis=1)

    normalized_torso_length = torso_length / shoulder_dist_safe

    if normalized_torso_length.size > 1:
        variance = np.var(normalized_torso_length)

        if variance < 0.001: rigidity_level = "Alta (Rigidez)"
        elif variance < 0.005: rigidity_level = "Media"
        else: rigidity_level = "Baja (Soltura)"

        initial_length = normalized_torso_length[0]
        final_length = normalized_torso_length[-1]

        percent_change = ((final_length - initial_length) / initial_length) * 100 if initial_length >= 1e-6 else 0.0

    return rigidity_level, variance, percent_change

def analyze_head_and_gaze(pose_data_3d, joints_map):
    """Analiza la estabilidad (STD) y el porcentaje de tiempo de la Orientación Vertical de la cabeza."""
    NOSE = joints_map['NOSE']
    L_SHOULDER, R_SHOULDER = joints_map['LEFT_SHOULDER'], joints_map['RIGHT_SHOULDER']
    L_HIP = joints_map['LEFT_HIP']

    mid_shoulder = (pose_data_3d[:, L_SHOULDER] + pose_data_3d[:, R_SHOULDER]) / 2
    torso_size = np.linalg.norm(mid_shoulder - pose_data_3d[:, L_HIP], axis=1)
    valid_frames_mask = torso_size > 1e-6
    torso_size_avg = np.mean(torso_size[valid_frames_mask])

    if torso_size_avg < 1e-6:
        return "N/A", 0.0, {"ARRIBA": 0, "FRONTAL": 0, "ABAJO": 0}

    nose_positions = pose_data_3d[:, NOSE]
    normalized_nose_movement = nose_positions / torso_size_avg
    std_dev_head_xy = np.std(normalized_nose_movement[:, :2])

    if std_dev_head_xy < 0.05: stability_category = "Fija (Muy Rígida)"
    elif std_dev_head_xy < 0.15: stability_category = "Media (Movimiento Natural)"
    else: stability_category = "Móvil (Activa)"

    nose_y = pose_data_3d[:, NOSE, 1]
    avg_y_position_global = np.mean(pose_data_3d[:, NOSE, 1])

    UMBRAL_ARRIBA = -0.05
    UMBRAL_ABAJO = 0.05

    gaze_counts = {"ARRIBA": 0, "FRONTAL": 0, "ABAJO": 0}

    for y_pos in nose_y:
        y_displacement = y_pos - avg_y_position_global
        if y_displacement < UMBRAL_ARRIBA: gaze_counts["ARRIBA"] += 1
        elif y_displacement > UMBRAL_ABAJO: gaze_counts["ABAJO"] += 1
        else: gaze_counts["FRONTAL"] += 1

    return stability_category, std_dev_head_xy, gaze_counts

def draw_pose_on_frame_final(frame, pose_data_frame, connections, color):
    """Dibuja el esqueleto 2D sobre el frame, asumiendo coordenadas MP normalizadas [0, 1]."""
    H, W, _ = frame.shape

    if pose_data_frame.size != 99: return

    pose_data_frame = pose_data_frame.reshape(33, 3)

    for i in range(len(pose_data_frame)):
        x, y, _ = pose_data_frame[i]
        cx = int(x * W); cy = int(y * H)
        if 0 <= cx < W and 0 <= cy < H: cv2.circle(frame, (cx, cy), RADIUS, color, -1)

    for p1_idx, p2_idx in connections:
        if p1_idx < 33 and p2_idx < 33:
            x1, y1, _ = pose_data_frame[p1_idx]; x2, y2, _ = pose_data_frame[p2_idx]
            cx1 = int(x1 * W); cy1 = int(y1 * H)
            cx2 = int(x2 * W); cy2 = int(y2 * H)

            if 0 <= cx1 < W and 0 <= cy1 < H and 0 <= cx2 < W and 0 <= cy2 < H:
                 cv2.line(frame, (cx1, cy1), (cx2, cy2), color, THICKNESS)

# ====================================================================
## 3. PIPELINE CONSOLIDADO PARA GRADIO
# ====================================================================

def run_full_heuristic_analysis(video_path: str, output_video_path: str, progress: Optional[Callable] = None) -> Tuple[str, str, str]:
    """
    Ejecuta el pipeline completo de análisis heurístico.
    Retorna la ruta del video de salida, el reporte Markdown, y la tabla Markdown de porcentajes.
    """
    
    # 1. Inicializar MediaPipe Pose y Extracción (Paso más lento)
    if progress: progress(0.01, desc="Paso 1/4: Iniciando Extracción de Pose")

    pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False)
    POSE_DATA_3D = extract_mediapipe_keypoints_optimized(video_path, pose_model, progress=progress)
    pose_model.close()
    
    N_TOTAL_FRAMES = POSE_DATA_3D.shape[0]
    if N_TOTAL_FRAMES == 0:
        raise RuntimeError("No se detectaron poses en el video. Asegúrate de que el sujeto esté visible.")


    # 2. Análisis Consolidado y Clasificación Frame-by-Frame
    if progress: progress(0.70, desc="Paso 2/4: Análisis de movimiento y clasificación")
    
    # Análisis Global
    static_percentage, movement_category, total_displacement, final_position_change = analyze_body_displacement(POSE_DATA_3D, MP_JOINTS)
    rigidity_level, rigidity_variance, torso_change = analyze_torso_rigidity(POSE_DATA_3D, MP_JOINTS)
    stability_category, std_dev_head_xy, gaze_counts = analyze_head_and_gaze(POSE_DATA_3D, MP_JOINTS)
    
    # Clasificación Frame-by-Frame
    posture_counts = {"Abierta": 0, "Pasiva": 0, "Cerrada": 0}
    frame_classifications = [] 
    
    for frame_pose_33x3 in POSE_DATA_3D:
        posture_label, current_color = classify_hand_posture(frame_pose_33x3, MP_JOINTS)
        posture_counts[posture_label] += 1
        frame_classifications.append({'label': posture_label, 'color': current_color})


    # 3. Generación del Video de Retroalimentación (Re-lectura para ahorrar RAM)
    if progress: progress(0.85, desc="Paso 3/4: Generación del video de salida")

    cap = cv2.VideoCapture(video_path)
    W_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); FPS_VIDEO = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, FPS_VIDEO, (W_out, H_out))
    
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_index >= N_TOTAL_FRAMES:
            break

        # Obtener datos precalculados
        classification = frame_classifications[frame_index]
        posture_label = classification['label']
        current_color = classification['color']
        frame_pose_33x3 = POSE_DATA_3D[frame_index] # Pose para dibujar

        # DIBUJAR LA POSE
        draw_pose_on_frame_final(frame, frame_pose_33x3.reshape(-1), BONE_CONNECTIONS, current_color)

        # Superponer el Recuadro y el Texto
        cv2.rectangle(frame, (0, H_out - 60), (W_out, H_out), current_color, -1)
        text_line1 = f"T: {frame_index/FPS_VIDEO:.1f}s | POSTURA: {posture_label}"
        cv2.putText(frame, text_line1, (20, H_out - 35), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_index += 1

    cap.release(); out.release()
    print(f"Video de retroalimentación generado: {output_video_path}")

    # 4. Generación de Reporte Final
    if progress: progress(0.95, desc="Paso 4/4: Generación de reporte final")
    
    total_gaze_frames = sum(gaze_counts.values())
    gaze_pct_frontal = (gaze_counts["FRONTAL"] / total_gaze_frames) * 100 if total_gaze_frames > 0 else 0
    gaze_pct_arriba = (gaze_counts["ARRIBA"] / total_gaze_frames) * 100 if total_gaze_frames > 0 else 0
    gaze_pct_abajo = (gaze_counts["ABAJO"] / total_gaze_frames) * 100 if total_gaze_frames > 0 else 0
    
    # Construcción del Reporte Markdown
    report_content = f"""
📋 Análisis Cuantitativo de Expresividad Corporal

Total de Frames Analizados: {N_TOTAL_FRAMES}

---

## 1. ESTABILIDAD Y MOVIMIENTO GENERAL (Tronco y Cabeza) 🏃

| Métrica | Clasificación | Detalle |
|:---|:---|:---|
| Movimiento General | {movement_category} | Desplazamiento total: {total_displacement:.2f} unidades |
| Tiempo Estático | {static_percentage:.2f}% | Umbral Estático: < {UMBRAL_VELOCIDAD_ESTATICA:.4f} |
| Estabilidad Cabeza | {stability_category} | Desviación STD XY: {std_dev_head_xy:.4f} |

---

## 2. ORIENTACIÓN VERTICAL DE LA MIRADA (Foco) 🔭

| Orientación | Porcentaje del Video (%) |
|:---|:---|
| Frontal (Foco Estable) | {gaze_pct_frontal:.2f} |
| Arriba (Introspección) | {gaze_pct_arriba:.2f} |
| Abajo (Leyendo/Notas) | {gaze_pct_abajo:.2f} |

---

## 3. POSTURA DE BRAZOS Y RIGIDEZ 🗣️

| Postura | Porcentaje del Video (%) |
|:---|:---|
| Abierta (Activa) | {posture_counts.get('Abierta', 0) / N_TOTAL_FRAMES * 100:.2f} |
| Pasiva (Intermedia) | {posture_counts.get('Pasiva', 0) / N_TOTAL_FRAMES * 100:.2f} |
| Cerrada (Retraída) | {posture_counts.get('Cerrada', 0) / N_TOTAL_FRAMES * 100:.2f} |
| Rigidez del Torso | {rigidity_level} (Varianza: {rigidity_variance:.5f}) |
"""
    
    # Tabla de Postura (separada)
    df_postura = pd.DataFrame([
        {'Postura': p, 'Porcentaje (%)': (c / N_TOTAL_FRAMES) * 100}
        for p, c in posture_counts.items()
    ])
    posture_table_markdown = df_postura.to_markdown(index=False)
    
    return output_video_path, report_content, posture_table_markdown

# ====================================================================
## 4. FUNCIONES DE GRADIO (Punto de Entrada)
# ====================================================================

def run_analysis_for_gradio(video_path_input: Optional[str], progress=None) -> Tuple[Optional[str], str, str]:
    """
    Función principal llamada por Gradio.
    """
    # Manejar el objeto progress si no se está ejecutando en Gradio
    if gr is None: progress = DummyProgress()

    if video_path_input is None:
        raise gr.Error("Por favor, sube un archivo de video para clasificar.")
        
    # Rutas temporales
    timestamp = int(time.time())
    output_video_path = os.path.join(OUTPUT_DIR_MP, f"feedback_video_{timestamp}.mp4")
    
    try:
        # Llama al pipeline completo
        final_video_path, report_markdown, posture_table_markdown = run_full_heuristic_analysis(
            video_path_input, output_video_path, progress=progress
        )
        
        if final_video_path:
             progress(1.0, desc="✅ Análisis Completado")
             return final_video_path, report_markdown, posture_table_markdown
        
        raise gr.Error("El análisis no pudo generar el video de salida.")
        
    except RuntimeError as e:
        progress(1.0, desc="❌ Error de Ejecución")
        raise gr.Error(f"Error de Ejecución: {e}")
    except Exception as e:
        progress(1.0, desc="❌ Error Inesperado")
        raise gr.Error(f"Ocurrió un error inesperado: {e}")
