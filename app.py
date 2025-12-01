# --- app.py modificado para este nuevo modelo ---

import gradio as gr
import os
import time
from typing import Optional

# 1. Importar la función clave y otras dependencias
from analisis_postura_gradio import run_analysis_for_gradio # <--- ¡NUEVA FUNCIÓN!

# Directorio temporal
OUTPUT_DIR = "temp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Definición de la Interfaz de Gradio ---

iface = gr.Interface(
    fn=run_analysis_for_gradio,
    
    inputs=gr.Video(label="🎥 Sube el video del discurso"),
    
    # 💡 TRES SALIDAS
    outputs=[
        gr.Video(label="✅ Video con Retroalimentación (Postura Dibujada)"),
        gr.Markdown(label="📋 Reporte Consolidado (Movimiento, Mirada, Rigidez)"),
        gr.Markdown(label="📊 Porcentajes de Postura"),
    ],
    
    title="🔬 Análisis Heurístico de Postura SkillTalk",
    description="Analiza ángulos, distancias, y movimiento para generar un reporte cuantitativo del estilo de gesticulación."
)

# 3. Iniciar la interfaz con Timeout alto
iface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 7860))
)
