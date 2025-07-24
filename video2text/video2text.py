#!/usr/bin/env python3
# file: video2text.py
# convierte video.mp4 ─> video.txt y video.srt usando Whisper (todo local)

from pathlib import Path
import whisper
import sys

# --- Configuración rápida ----------------------------------------------------
VIDEO_FILE = Path(__file__).with_name("video6.mp4")   # raíz del proyecto
MODEL_SIZE = "large"          # usa "small" / "medium" / "large" si necesitas
LANG       = "es"            # ajusta si el video no está en español
USE_FP16   = True         # pon True si tienes GPU con soporte FP16
DEVICE   = "cuda"
# -----------------------------------------------------------------------------

if not VIDEO_FILE.exists():
    sys.exit("❌  No se encontró video.mp4 en la raíz del proyecto.")

print("⏳  Cargando modelo Whisper…")
model = whisper.load_model(MODEL_SIZE)

print(f"🎙️  Transcribiendo {VIDEO_FILE.name} …")
result = model.transcribe(
    str(VIDEO_FILE),
    language=LANG,
    fp16=USE_FP16,
    word_timestamps=False,   # pon True si quieres tiempo por palabra
)

# --- Guarda texto plano ------------------------------------------------------
text_file = VIDEO_FILE.with_suffix(".txt")
text_file.write_text(result["text"], encoding="utf-8")
print(f"✅  Transcripción guardada en {text_file}")

# --- Guarda subtítulos SRT ----------------------------------------------------
def srt_timestamp(seconds: float) -> str:
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    ms   = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

srt_lines = []
for idx, seg in enumerate(result["segments"], start=1):
    start, end = seg["start"], seg["end"]
    srt_lines.append(f"{idx}")
    srt_lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
    srt_lines.append(seg["text"].strip())
    srt_lines.append("")  # línea en blanco

srt_file = VIDEO_FILE.with_suffix(".srt")
srt_file.write_text("\n".join(srt_lines), encoding="utf-8")
print(f"🎞️  Subtítulos SRT guardados en {srt_file}")

print("🏁  Terminado.")
