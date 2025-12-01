import gradio as gr
import os
import time
from typing import Optional, Tuple

# 1. Importar la función clave desde el script de análisis
from analisis_postura_gradio import run_analysis_for_gradio

# Directorio temporal para guardar videos y archivos procesados
OUTPUT_DIR = "temp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def gradio_processor(video_path_input: Optional[str], progress=gr.Progress()) -> Tuple[Optional[str], str, str]:
    """
    Función wrapper que llama al pipeline de análisis heurístico.
    """
    if video_path_input is None:
        raise gr.Error("Por favor, sube un archivo de video para el análisis.")
        
    # Crear una ruta de salida temporal única
    timestamp = int(time.time())
    output_video_path = os.path.join(OUTPUT_DIR, f"feedback_video_{timestamp}.mp4")
    
    try:
        final_video_path, report_markdown, posture_table_markdown = run_analysis_for_gradio(
            video_path_input, 
            output_video_path, 
            progress=progress
        )
        
        return final_video_path, report_markdown, posture_table_markdown
        
    except gr.Error:
        raise
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        raise gr.Error(f"Error en el procesamiento del modelo: {e}. Revisa los logs de Render para más detalles.")


# --- 2. Definición de la Interfaz de Gradio ---

iface = gr.Interface(
    fn=gradio_processor,
    
    # ENTRADA
    inputs=gr.Video(label="🎥 Sube el video del discurso"),
    
    # SALIDAS (TRES SALIDAS)
    outputs=[
        gr.Video(label="✅ Video con Retroalimentación (Postura Dibujada)"),
        gr.Markdown(label="📋 Reporte Consolidado (Movimiento, Mirada, Rigidez)"),
        gr.Markdown(label="📊 Porcentajes de Postura (Tabla)"),
    ],
    
    title="🔬 Análisis Heurístico de Postura SkillTalk",
    description="Analiza la postura y el movimiento (rigidez, gesticulación, mirada) para generar un reporte cuantitativo."
    # 🛑 ALLOW_FLAGGING FUE ELIMINADO PARA COMPATIBILIDAD CON TU VERSIÓN DE GRADIO
)

# 3. Iniciar la interfaz
iface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 7860))
    # server_timeout fue eliminado previamente para evitar errores en algunas versiones de Gradio
)
