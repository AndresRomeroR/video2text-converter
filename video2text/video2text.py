#!/usr/bin/env python3
# file: video2text.py
# Convierte video.mp4 o video.mkv a video.txt y video.srt usando Whisper (todo local)

from pathlib import Path
import sys

import torch
import whisper

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mkv")

# --- Configuracion rapida -----------------------------------------------------
VIDEO_FILE = Path(__file__).with_name("2026-04-14 08-05-10.mkv")  # raiz del proyecto
MODEL_SIZE = "large"  # small | medium | large
LANG = "es"  # idioma del video
USE_FP16 = True  # True si tu GPU soporta FP16
DEVICE = "cuda"  # "cuda" o "cpu"
# ------------------------------------------------------------------------------


def resolve_video_file(video_file: Path) -> Path:
    suffix = video_file.suffix.lower()

    if not suffix:
        candidates = [video_file.with_suffix(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS]
    elif suffix in SUPPORTED_VIDEO_EXTENSIONS:
        candidates = [video_file] + [
            video_file.with_suffix(ext)
            for ext in SUPPORTED_VIDEO_EXTENSIONS
            if ext != suffix
        ]
    else:
        supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
        sys.exit(
            f"Extension no compatible: {video_file.suffix or '(sin extension)'}. "
            f"Usa una de estas: {supported}."
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(path.name for path in candidates)
    sys.exit(f"No se encontro un video compatible. Probe: {tried}.")


VIDEO_FILE = resolve_video_file(VIDEO_FILE)

if DEVICE == "cuda" and not torch.cuda.is_available():
    print("CUDA no esta disponible. Cambiando a CPU...")
    DEVICE = "cpu"
    USE_FP16 = False

print(f"Cargando modelo Whisper ({MODEL_SIZE}) en {DEVICE}...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)

print(f"Transcribiendo {VIDEO_FILE.name}...")
result = model.transcribe(
    str(VIDEO_FILE),
    language=LANG,
    fp16=USE_FP16 and DEVICE == "cuda",
    word_timestamps=False,
)

text_file = VIDEO_FILE.with_suffix(".txt")
text_file.write_text(result["text"].strip(), encoding="utf-8")
print(f"Transcripcion guardada en {text_file.name}")


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
print(f"Subtitulos guardados en {srt_file.name}")

print("Proceso finalizado exitosamente.")
