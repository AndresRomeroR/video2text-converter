#!/usr/bin/env python3
# file: video2text.py
# Convierte video.mp4 ─> video.txt y video.srt usando Whisper (todo local)

from pathlib import Path
import whisper
import sys
import torch

# --- Configuración rápida ----------------------------------------------------
VIDEO_FILE = Path(__file__).with_name("video6.mp4")   # raíz del proyecto
MODEL_SIZE = "large"       # small | medium | large
LANG       = "es"          # idioma del video
USE_FP16   = True          # True si tu GPU soporta FP16
DEVICE     = "cuda"        # "cuda" o "cpu"
# ------------------------------------------------------------------------------

# --- Validaciones ------------------------------------------------------------
if not VIDEO_FILE.exists():
    sys.exit(f"❌  No se encontró {VIDEO_FILE.name} en la raíz del proyecto.")

if DEVICE == "cuda" and not torch.cuda.is_available():
    print("⚠️  CUDA no está disponible. Cambiando a CPU...")
    DEVICE = "cpu"
    USE_FP16 = False

# --- Cargar modelo -----------------------------------------------------------
print(f"⏳  Cargando modelo Whisper ({MODEL_SIZE}) en {DEVICE}…")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)

# --- Transcripción -----------------------------------------------------------
print(f"🎙️  Transcribiendo {VIDEO_FILE.name} …")
result = model.transcribe(
    str(VIDEO_FILE),
    language=LANG,
    fp16=USE_FP16 and DEVICE == "cuda",
    word_timestamps=False
)

# --- Guardar texto -----------------------------------------------------------
text_file = VIDEO_FILE.with_suffix(".txt")
text_file.write_text(result["text"].strip(), encoding="utf-8")
print(f"✅  Transcripción guardada en {text_file.name}")

# --- Generar archivo SRT -----------------------------------------------------
def srt_timestamp(seconds: float) -> str:
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

srt_lines = []
for idx, seg in enumerate(result["segments"], start=1):
    srt_lines.append(str(idx))
    srt_lines.append(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}")
    srt_lines.append(seg["text"].strip())
    srt_lines.append("")

srt_file = VIDEO_FILE.with_suffix(".srt")
srt_file.write_text("\n".join(srt_lines), encoding="utf-8")
print(f"🎞️  Subtítulos guardados en {srt_file.name}")

print("🏁  Proceso finalizado exitosamente.")
