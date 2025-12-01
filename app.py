import gradio as gr
import os
import time
from typing import Optional, Tuple

# 1. Importar la función clave desde el script de análisis
from analisis_postura_gradio import run_analysis_for_gradio

# Directorio temporal para guardar videos y archivos procesados
OUTPUT_DIR = "temp_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 🛑 CORRECCIÓN CLAVE: El argumento 'progress' ya NO tiene un valor por defecto.
# Gradio inyecta el valor automáticamente al llamarla.
def gradio_processor(video_path_input: Optional[str], progress) -> Tuple[Optional[str], str, str]:
    """
    Función wrapper que llama al pipeline de análisis heurístico.
    El objeto 'progress' es inyectado por Gradio.
    """
    if video_path_input is None:
        raise gr.Error("Por favor, sube un archivo de video para el análisis.")
        
    # Rutas
    timestamp = int(time.time())
    output_video_path = os.path.join(OUTPUT_DIR, f"feedback_video_{timestamp}.mp4")
    
    try:
        # Llama a la función principal, pasando 'progress' como argumento de palabra clave
        # Esto es correcto porque Gradio lo inyecta como keyword argument en el nivel superior.
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
    description="Analiza la postura y el movimiento (rigidez, gesticulación, mirada) para generar un reporte cuantitativo.",
)

# 3. Iniciar la interfaz
iface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 7860)),
    # server_timeout fue omitido para evitar errores de versión en launch()
)
