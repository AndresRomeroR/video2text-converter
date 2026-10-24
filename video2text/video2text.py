from __future__ import annotations

import ctypes
import gc
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from pathlib import Path
from typing import Callable, Optional


def ensure_standard_streams() -> None:
    """pythonw/PyInstaller windowed dejan stdout/stderr en None.

    tqdm y otras dependencias necesitan objetos con write/flush incluso cuando
    no hay consola. Conservamos los streams reales si se ejecuta en terminal.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


# Antes de importar dependencias que puedan capturar los streams al inicializarse.
ensure_standard_streams()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
from tkinter.scrolledtext import ScrolledText

from transcription_queue import BatchQueue, QueueItem
from output_paths import plan_output_paths

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = None
    TkinterDnD = None


_torch = None
_torch_error: Optional[Exception] = None
_whisper = None
_whisper_error: Optional[Exception] = None


APP_TITLE = "Audio y Video a Texto - Whisper"
WINDOW_SIZE = (1000, 740)
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
SUPPORTED_MEDIA_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS
DEFAULT_MODEL = "turbo"
DEFAULT_LANG = "es"
LANGUAGE_ALIASES = {
    "alemán": "de",
    "aleman": "de",
    "castellano": "es",
    "español": "es",
    "espanol": "es",
    "francés": "fr",
    "frances": "fr",
    "inglés": "en",
    "ingles": "en",
    "italiano": "it",
    "portugués": "pt",
    "portugues": "pt",
}
MODEL_HINTS = {
    "turbo": "Recomendado · rápido y muy preciso",
    "large-v3": "Máxima precisión · mayor uso de memoria",
    "medium": "Buen equilibrio para equipos intermedios",
    "small": "Ligero · menor precisión",
    "base": "Muy ligero · ideal para pruebas",
    "tiny": "El más rápido · precisión básica",
}


def bundled_resource(name: str) -> Path:
    """Resuelve recursos tanto en código fuente como dentro de PyInstaller."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    base_path = Path(bundle_root) if bundle_root else Path(__file__).resolve().parent
    return base_path / name


def find_media_binary(name: str) -> Optional[str]:
    """Localiza FFmpeg/FFprobe incluidos en el EXE o disponibles en PATH."""
    executable_name = f"{name}.exe" if sys.platform == "win32" else name
    packaged_binary = bundled_resource(executable_name)
    if packaged_binary.is_file():
        return str(packaged_binary)
    return shutil.which(name)


def dependency_error(component: str, package: str) -> str:
    if getattr(sys, "frozen", False):
        return (
            f"No se pudo cargar {component} desde el ejecutable. "
            "Vuelve a generar la aplicación con build_exe.py (build_exe.ps1 en Windows); "
            "instalar paquetes con pip no modifica un ejecutable ya creado."
        )
    command = (
        f'& "{sys.executable}"' if sys.platform == "win32" else shlex.quote(sys.executable)
    )
    return (
        f"No se pudo cargar {component}. Reinstala en este entorno con: "
        f'{command} -m pip install --upgrade {package}'
    )


def available_devices(torch_module) -> list[str]:
    """Consulta los backends instalados; ROCm usa la API cuda de PyTorch."""
    devices = []
    backends = (
        ("cuda", getattr(torch_module, "cuda", None)),
        ("xpu", getattr(torch_module, "xpu", None)),
        ("mps", getattr(getattr(torch_module, "backends", None), "mps", None)),
    )
    for name, backend in backends:
        try:
            if backend is not None and backend.is_available():
                devices.append(name)
        except Exception:
            # Un controlador roto no debe impedir el uso de CPU.
            continue
    return devices + ["cpu"]


def resolve_device(torch_module, requested: str, fp16: bool) -> tuple[str, bool]:
    devices = available_devices(torch_module)
    requested = "cuda" if requested == "rocm" else requested
    device = devices[0] if requested == "auto" else requested
    if device not in devices:
        device = "cpu"
    # FP32 en XPU/MPS evita asumir compatibilidad de las operaciones FP16 de Whisper.
    return device, bool(fp16 and device == "cuda")


def device_description(torch_module, device: str) -> str:
    if device == "cuda":
        kind = "AMD / ROCm" if getattr(torch_module.version, "hip", None) else "NVIDIA / CUDA"
        return f"{kind}: {torch_module.cuda.get_device_name(0)}"
    if device == "xpu":
        return f"Intel / XPU: {torch_module.xpu.get_device_name(0)}"
    return "Apple / MPS" if device == "mps" else "CPU"


def torch_build_description(torch_module) -> str:
    """Distingue las capacidades incluidas de la GPU disponible en este equipo."""
    if getattr(torch_module.version, "hip", None):
        return f"ROCm {torch_module.version.hip} + CPU"
    if getattr(torch_module.version, "cuda", None):
        return f"CUDA {torch_module.version.cuda} + CPU"
    if getattr(getattr(torch_module, "xpu", None), "_is_compiled", lambda: False)():
        return "Intel XPU + CPU"
    if getattr(getattr(torch_module.backends, "mps", None), "is_built", lambda: False)():
        return "Apple MPS + CPU"
    return "CPU"


def clear_device_cache(torch_module, device: str) -> None:
    backend = getattr(torch_module, device, None)
    try:
        if backend is not None and hasattr(backend, "empty_cache"):
            backend.empty_cache()
    except Exception:
        pass  # La limpieza de una GPU averiada no debe bloquear el respaldo CPU.


def run_with_cpu_fallback(action, device: str, fp16: bool, release, log):
    """Reintenta una sola vez fuera del bloque except para liberar el traceback GPU."""
    try:
        return action(device, fp16), device
    except (RuntimeError, NotImplementedError) as exc:
        if device == "cpu":
            raise
        log(f"Falló la ejecución en {device}: {exc}")
    release()
    log("Reintentando en CPU con FP32. El modelo y el idioma se conservan.")
    return action("cpu", False), "cpu"


def open_folder(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))
    else:
        command = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([command, str(path)])


def load_torch():
    global _torch, _torch_error
    if _torch is not None:
        return _torch
    if _torch_error is not None:
        raise RuntimeError(
            dependency_error("torch", "torch")
        ) from _torch_error
    try:
        import torch as torch_module
    except Exception as exc:
        _torch_error = exc
        raise RuntimeError(
            dependency_error("torch", "torch")
        ) from exc
    _torch = torch_module
    return _torch


def load_whisper():
    global _whisper, _whisper_error
    if _whisper is not None:
        return _whisper
    if _whisper_error is not None:
        raise RuntimeError(
            dependency_error("Whisper", "openai-whisper")
        ) from _whisper_error
    try:
        import whisper as whisper_module
    except Exception as exc:
        _whisper_error = exc
        raise RuntimeError(
            dependency_error("Whisper", "openai-whisper")
        ) from exc
    _whisper = whisper_module
    return _whisper


def _windows_set_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _parse_drop_files(master: tk.Tk, data: str) -> list[str]:
    if not data:
        return []
    try:
        return [str(item) for item in master.tk.splitlist(data)]
    except tk.TclError:
        return []


def _parse_drop_file(master: tk.Tk, data: str) -> Optional[str]:
    """Compatibilidad con integraciones que esperan un único archivo."""
    items = _parse_drop_files(master, data)
    return items[0] if items else None


def resolve_media_file(media_file: Path) -> Path:
    if not media_file.exists() or not media_file.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {media_file}")

    suffix = media_file.suffix.lower()
    if suffix not in SUPPORTED_MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_MEDIA_EXTENSIONS))
        raise ValueError(
            f"Extensión no compatible: {suffix or '(sin extensión)'}.\n"
            f"Extensiones permitidas: {supported}"
        )
    return media_file


def resolve_video_file(video_file: Path) -> Path:
    """Compatibilidad con integraciones que usaban el nombre anterior."""
    return resolve_media_file(video_file)


def require_ffmpeg() -> str:
    ffmpeg_path = find_media_binary("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "No se encontró FFmpeg. Instálalo y asegúrate de que el comando "
            "'ffmpeg' esté disponible en PowerShell."
        )
    return ffmpeg_path


def require_ffprobe() -> str:
    ffprobe_path = find_media_binary("ffprobe")
    if ffprobe_path is None:
        raise RuntimeError(
            "No se encontró FFprobe. Instala FFmpeg completo y asegúrate de que "
            "el comando 'ffprobe' esté disponible en PowerShell."
        )
    return ffprobe_path


def probe_audio_stream(media_file: Path, ffprobe_path: Optional[str] = None) -> dict:
    """Comprueba que FFmpeg pueda leer al menos una pista de audio."""
    command = [
        ffprobe_path or require_ffprobe(),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index,codec_name,channels,sample_rate",
        "-of",
        "json",
        str(media_file),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"FFprobe agotó el tiempo al analizar el archivo: {media_file.name}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"No se pudo ejecutar FFprobe: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or "FFprobe no pudo leer el archivo."
        raise ValueError(f"Archivo multimedia inválido o dañado: {detail}")

    try:
        streams = json.loads(completed.stdout or "{}").get("streams") or []
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("FFprobe devolvió una respuesta inválida.") from exc

    if not streams:
        raise ValueError("El archivo no contiene una pista de audio decodificable.")
    return streams[0]


def normalize_language(whisper_module, language: str) -> Optional[str]:
    language = language.strip().lower()
    if language in {"", "auto", "automático", "automatico"}:
        return None

    language = LANGUAGE_ALIASES.get(language, language)
    tokenizer = whisper_module.tokenizer
    language = tokenizer.TO_LANGUAGE_CODE.get(language, language)
    if language not in tokenizer.LANGUAGES:
        raise ValueError(
            f"Idioma no reconocido: {language!r}. Usa un código como 'es', 'en' "
            "o escribe 'auto' para detectarlo."
        )
    return language


def srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def write_windows_text(path: Path, content: str) -> None:
    """Escribe UTF-8 con finales CRLF sin dejar un archivo parcial."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as f:
            f.write(normalized.replace("\n", "\r\n"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_transcript_outputs(result: dict) -> tuple[str, str]:
    """Construye TXT y SRT, descartando segmentos vacíos o mal formados."""
    valid_segments: list[tuple[float, float, str]] = []
    for segment in result.get("segments") or []:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(segment["start"]))
            end = max(start, float(segment["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        valid_segments.append((start, end, text))

    transcript = str(result.get("text") or "").strip()
    if not transcript:
        transcript = " ".join(text for _, _, text in valid_segments)

    srt_lines: list[str] = []
    for index, (start, end, text) in enumerate(valid_segments, start=1):
        srt_lines.extend(
            (
                str(index),
                f"{srt_timestamp(start)} --> {srt_timestamp(end)}",
                text,
                "",
            )
        )

    txt_content = f"{transcript}\n" if transcript else ""
    srt_content = "\n".join(srt_lines)
    return txt_content, srt_content


def transcribe_media(
    model,
    media_file: Path,
    language: Optional[str],
    use_fp16: bool,
    initial_prompt: Optional[str],
) -> dict:
    """Ejecuta Whisper sin depender de la interfaz gráfica."""
    return model.transcribe(
        str(media_file),
        language=language,
        fp16=use_fp16,
        initial_prompt=initial_prompt,
        word_timestamps=False,
        verbose=None,
    )


def save_transcript_outputs(
    media_file: Path, result: dict, *, include_extension: bool = False,
    output_paths: Optional[tuple[Path, Path]] = None,
) -> tuple[Path, Path]:
    """Guarda las salidas de una transcripción junto al archivo original."""
    output_base = media_file.name if include_extension else media_file.stem
    txt_file, srt_file = output_paths or (
        media_file.with_name(f"{output_base}.txt"), media_file.with_name(f"{output_base}.srt"))
    txt_content, srt_content = build_transcript_outputs(result)
    write_windows_text(txt_file, txt_content)
    write_windows_text(srt_file, srt_content)
    return txt_file, srt_file


class Video2TextApp(TkinterDnD.Tk if TkinterDnD else tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        # Tk obtiene los DPI del sistema. Las fuentes están en puntos; solo las
        # distancias expresadas en píxeles se multiplican por este factor.
        self._scale = float(self.tk.call("tk", "scaling")) * 72.0 / 96.0

        self.title(APP_TITLE)
        self._apply_window_icon()
        self._apply_theme()

        self._batch = BatchQueue()
        self._output_plan: dict[str, tuple[Path, Path]] = {}
        self._stop_requested = threading.Event()
        self._active_item_id: Optional[str] = None
        self._running = False
        self._closing = False
        self._main_thread_id = threading.get_ident()
        self._ui_queue: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._loaded_model = None
        self._loaded_model_key: Optional[tuple[str, str]] = None

        self.var_model = tk.StringVar(value=DEFAULT_MODEL)
        self.var_lang = tk.StringVar(value=DEFAULT_LANG)
        self.var_device = tk.StringVar(value="auto")
        self.var_fp16 = tk.BooleanVar(value=True)
        self.var_prompt = tk.StringVar()

        self._build_ui()
        self._refresh_summary()
        self._compact = False
        self.bind("<Configure>", self._on_window_resize, add="+")
        self._center_window()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Control-o>", self._shortcut_select)
        self.bind("<Control-Return>", self._shortcut_process)
        self.after(50, self._drain_ui_queue)
        self.after(150, lambda: threading.Thread(
            target=self._detect_acceleration, daemon=True
        ).start())

    def _apply_window_icon(self) -> None:
        icon_path = bundled_resource("totext.ico")
        if not icon_path.is_file():
            return
        try:
            self.iconbitmap(default=str(icon_path))
        except tk.TclError:
            pass

    def _px(self, value: int) -> int:
        return max(1, round(value * self._scale))

    def _apply_theme(self) -> None:
        style = ttk.Style(self)
        for theme in ("vista", "winnative", "xpnative", style.theme_use()):
            try:
                style.theme_use(theme)
                break
            except tk.TclError:
                continue
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family="Segoe UI", size=10)
        style.configure("TButton", padding=(self._px(10), self._px(6)))
        style.configure("TNotebook.Tab", padding=(self._px(16), self._px(7)))
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Hint.TLabel", foreground="#5f6368")
        style.configure("Field.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Drop.TLabel", font=("Segoe UI", 11, "bold"),
                        padding=(self._px(12), self._px(14)), background="#eaf2fb",
                        foreground="#234e78")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"), foreground="#305f8f")
        style.configure("Success.Status.TLabel", foreground="#19733b")
        style.configure("Error.Status.TLabel", foreground="#b42318")
        line_height = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        style.configure("Treeview", rowheight=line_height + self._px(10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _center_window(self) -> None:
        self.update_idletasks()
        left, top = 0, 0
        right, bottom = self.winfo_screenwidth(), self.winfo_screenheight()
        if sys.platform == "win32":
            try:
                from ctypes import wintypes
                area = wintypes.RECT()
                if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(area), 0):
                    left, top, right, bottom = area.left, area.top, area.right, area.bottom
            except (AttributeError, OSError):
                pass
        available_w = max(1, right - left - self._px(24))
        available_h = max(1, bottom - top - self._px(48))
        width = min(self._px(WINDOW_SIZE[0]), available_w)
        height = min(self._px(WINDOW_SIZE[1]), available_h)
        self.minsize(min(self._px(650), available_w), min(self._px(480), available_h))
        self.resizable(True, True)
        self.geometry(f"{width}x{height}+{left + (right-left-width)//2}+{top + (bottom-top-height)//2}")

    def _build_ui(self) -> None:
        pad = self._px(14)
        root = ttk.Frame(self, padding=pad)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, self._px(12)))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Audio y Video a Texto", style="Title.TLabel").grid(
            row=0, column=0, sticky="w")
        self.lbl_acceleration = ttk.Label(header, text="Comprobando aceleración…", style="Hint.TLabel")
        self.lbl_acceleration.grid(row=1, column=0, sticky="ew", pady=(self._px(4), 0))
        header.bind("<Configure>", lambda e: self.lbl_acceleration.config(wraplength=max(1, e.width)))

        self.tabs = ttk.Notebook(root)
        self.tabs.grid(row=1, column=0, sticky="nsew")
        files = ttk.Frame(self.tabs, padding=pad)
        settings = ttk.Frame(self.tabs, padding=pad)
        logs = ttk.Frame(self.tabs, padding=pad)
        self.tabs.add(files, text="Archivos")
        self.tabs.add(settings, text="Configuración")
        self.tabs.add(logs, text="Registro")
        files.columnconfigure(0, weight=1)
        files.rowconfigure(2, weight=1)

        self.lbl_drop = ttk.Label(
            files, text="Arrastra aquí uno o varios audios y videos\no haz clic para añadirlos",
            anchor="center", justify="center", cursor="hand2", style="Drop.TLabel")
        self.lbl_drop.grid(row=0, column=0, sticky="ew")
        self.lbl_drop.bind("<Button-1>", self._on_drop_click)
        files.bind("<Configure>", lambda e: self.lbl_drop.config(wraplength=max(1, e.width - 2*pad)))

        toolbar = ttk.Frame(files)
        toolbar.grid(row=1, column=0, sticky="ew", pady=self._px(10))
        self.btn_select = ttk.Button(toolbar, text="Añadir archivos…", command=self._on_select_file)
        self.btn_select.pack(side="left")
        self.btn_remove = ttk.Button(toolbar, text="Quitar selección", command=self._on_remove)
        self.btn_remove.pack(side="left", padx=self._px(6))
        self.btn_clear_finished = ttk.Button(toolbar, text="Limpiar finalizados", command=self._on_clear_finished)
        self.btn_clear_finished.pack(side="left")

        table = ttk.Frame(files)
        table.grid(row=2, column=0, sticky="nsew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.file_tree = ttk.Treeview(table, columns=("name", "status", "size", "time", "folder"),
                                     show="headings", selectmode="extended", height=7)
        for key, title, width in (("name", "Archivo", 280), ("status", "Estado", 130),
                                  ("size", "Tamaño", 90), ("time", "Tiempo", 85), ("folder", "Carpeta", 260)):
            self.file_tree.heading(key, text=title)
            self.file_tree.column(key, width=self._px(width), minwidth=self._px(70),
                                  stretch=key in ("name", "folder"), anchor="w")
        self.file_tree.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(table, orient="vertical", command=self.file_tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(table, orient="horizontal", command=self.file_tree.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.file_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.file_tree.tag_configure("processing", foreground="#185b9d")
        self.file_tree.tag_configure("completed", foreground="#19733b")
        self.file_tree.tag_configure("error", foreground="#b42318")
        self.file_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.file_tree.bind("<Delete>", lambda _e: self._on_remove())
        self.file_tree.bind("<Double-1>", lambda _e: self._on_open_folder())
        if TkinterDnD is not None:
            for target in (self.lbl_drop, self.file_tree):
                try:
                    target.drop_target_register(DND_FILES)
                    target.dnd_bind("<<Drop>>", self._on_drop_file)
                except tk.TclError:
                    pass

        self.txt_detail = ScrolledText(files, height=3, wrap="word", state="disabled",
                                      font="TkDefaultFont", relief="flat", padx=self._px(6), pady=self._px(4))
        self.txt_detail.grid(row=3, column=0, sticky="ew", pady=(self._px(8), 0))
        self._set_file_detail("Añade archivos para comenzar. Se procesan en el orden del listado.")
        self.btn_open_folder = ttk.Button(files, text="Abrir carpeta del seleccionado", command=self._on_open_folder)
        self.btn_open_folder.grid(row=4, column=0, sticky="w", pady=(self._px(6), 0))

        self._build_settings(settings)
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(1, weight=1)
        ttk.Button(logs, text="Limpiar registro", command=self._clear_console).grid(row=0, column=0, sticky="e")
        self.txt_console = ScrolledText(logs, height=8, wrap="word", state="disabled",
                                       font=("Consolas", 10), background="#111827", foreground="#dbe5f1",
                                       relief="flat", padx=self._px(10), pady=self._px(8))
        self.txt_console.grid(row=1, column=0, sticky="nsew", pady=(self._px(8), 0))

        footer = ttk.Frame(root)
        footer.grid(row=2, column=0, sticky="ew", pady=(self._px(12), 0))
        footer.columnconfigure(0, weight=1)
        self.lbl_summary = ttk.Label(footer, style="Field.TLabel")
        self.lbl_summary.grid(row=0, column=0, sticky="ew", columnspan=2)
        self.lbl_summary.bind("<Configure>", lambda e: self.lbl_summary.config(wraplength=max(1, e.width)))
        self.pbar = ttk.Progressbar(footer, orient="horizontal", mode="determinate", maximum=1)
        self.pbar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=self._px(7))
        status_frame = ttk.Frame(footer, height=tkfont.nametofont("TkDefaultFont").metrics("linespace") * 2)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        status_frame.grid_propagate(False)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        self.lbl_status = ttk.Label(status_frame, text="Listo para añadir archivos", style="Status.TLabel", anchor="w")
        self.lbl_status.grid(row=0, column=0, sticky="nsew")
        self.lbl_status.bind("<Configure>", lambda e: self.lbl_status.config(wraplength=max(1, e.width)))
        controls = ttk.Frame(footer)
        controls.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(self._px(8), 0))
        controls.columnconfigure(0, weight=1)
        self.lbl_config = ttk.Label(controls, style="Hint.TLabel")
        self.lbl_config.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(self._px(6), 0))
        self.lbl_config.bind("<Configure>", lambda e: self.lbl_config.config(wraplength=max(1, e.width)))
        self.btn_retry = ttk.Button(controls, text="Reintentar errores", command=self._on_retry)
        self.btn_retry.grid(row=0, column=1, padx=self._px(6))
        self.btn_stop = ttk.Button(controls, text="Detener después del actual", command=self._on_stop, state="disabled")
        self.btn_stop.grid(row=0, column=2, padx=(0, self._px(6)))
        self.btn_process = ttk.Button(controls, text="Procesar cola", command=self._on_process,
                                      state="disabled", style="Accent.TButton")
        self.btn_process.grid(row=0, column=3)
        for variable in (self.var_model, self.var_lang, self.var_device):
            variable.trace_add("write", lambda *_: self._update_config_summary())
        self._update_config_summary()
        if TkinterDnD is None:
            self._append_console("Arrastrar y soltar requiere tkinterdnd2. Usa Añadir archivos o Ctrl+O.")

    def _build_settings(self, parent) -> None:
        # Dos columnas evitan comprimir los campos a DPI altos. Esta pestaña
        # puede desplazarse sin ocultar el listado ni las acciones de la cola.
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas)
        window_id = canvas.create_window(0, 0, anchor="nw", window=content)
        content.columnconfigure(1, weight=1)
        content.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        def wheel(event):
            canvas.yview_scroll(-int(event.delta / 120), "units")
            return "break"
        canvas.bind("<MouseWheel>", wheel)
        rows = (("Modelo", "Whisper se carga una vez y se reutiliza en la cola."),
                ("Idioma", "Código o nombre: es, español, en o auto."),
                ("Dispositivo", "Auto elige una aceleración disponible; CPU como respaldo."))
        self.cmb_model = ttk.Combobox(content, textvariable=self.var_model, state="readonly", width=16,
                                      values=("turbo", "large-v3", "medium", "small", "base", "tiny"))
        self.ent_lang = ttk.Entry(content, textvariable=self.var_lang)
        self.cmb_device = ttk.Combobox(content, textvariable=self.var_device, state="readonly", width=16,
                                       values=("auto", "cuda", "rocm", "xpu", "mps", "cpu"))
        for index, ((label, hint), widget) in enumerate(zip(rows, (self.cmb_model, self.ent_lang, self.cmb_device))):
            row = index * 3
            ttk.Label(content, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w",
                                                                                    padx=(0, self._px(16)))
            widget.grid(row=row, column=1, sticky="ew")
            hint_label = ttk.Label(content, text=hint, style="Hint.TLabel")
            hint_label.grid(row=row+1, column=0, columnspan=2, sticky="ew", pady=(self._px(4), self._px(14)))
            hint_label.bind("<Configure>", lambda e, w=hint_label: w.config(wraplength=max(1, e.width)))
        self.lbl_model_hint = ttk.Label(content, text=MODEL_HINTS[DEFAULT_MODEL], style="Hint.TLabel")
        self.lbl_model_hint.grid(row=9, column=0, columnspan=2, sticky="ew")
        self.lbl_model_hint.bind("<Configure>", lambda e: self.lbl_model_hint.config(wraplength=max(1, e.width)))
        self.cmb_model.bind("<<ComboboxSelected>>", self._update_model_hint)
        self.chk_fp16 = ttk.Checkbutton(content, text="Usar FP16 en CUDA / ROCm", variable=self.var_fp16)
        self.chk_fp16.grid(row=10, column=0, columnspan=2, sticky="w", pady=self._px(12))
        ttk.Label(content, text="Contexto opcional", style="Field.TLabel").grid(row=11, column=0, columnspan=2, sticky="w")
        self.ent_prompt = ttk.Entry(content, textvariable=self.var_prompt)
        self.ent_prompt.grid(row=12, column=0, columnspan=2, sticky="ew", pady=self._px(6))
        hint = ttk.Label(content, text="Nombres propios, siglas o vocabulario técnico. Se aplica a todos los archivos de la cola.",
                         style="Hint.TLabel")
        hint.grid(row=13, column=0, columnspan=2, sticky="ew")
        hint.bind("<Configure>", lambda e: hint.config(wraplength=max(1, e.width)))
        for widget in (content, *content.winfo_children()):
            if not isinstance(widget, ttk.Combobox):
                widget.bind("<MouseWheel>", wheel)

    def _update_config_summary(self) -> None:
        self.lbl_config.config(text=f"{self.var_model.get()} · {self.var_lang.get() or 'es'} · {self.var_device.get()}   |   Ctrl+O: añadir · Ctrl+Enter: procesar")

    def _on_window_resize(self, event) -> None:
        if event.widget is not self:
            return
        compact = event.height < self._px(650)
        if compact == self._compact:
            return
        self._compact = compact
        for widget in (self.lbl_drop, self.txt_detail, self.btn_open_folder, self.lbl_acceleration):
            widget.grid_remove() if compact else widget.grid()
        self.tabs.tab(0, text="Archivos · arrastra al listado" if compact else "Archivos")

    def _detect_acceleration(self) -> None:
        try:
            torch_module = load_torch()
            included = torch_build_description(torch_module)
            device, _ = resolve_device(torch_module, "auto", True)
            available = device_description(torch_module, device)
            text = f"Versión incluida: {included} · Automático en este equipo: {available}"
        except Exception as exc:
            text = f"No se pudo comprobar la aceleración: {exc}"
        self._ui(lambda: self.lbl_acceleration.config(text=text))
        self._append_console(text)

    def _shortcut_select(self, _event=None) -> str:
        if not self._running:
            self._on_select_file()
        return "break"

    def _shortcut_process(self, _event=None) -> str:
        self._on_process()
        return "break"

    def _on_drop_click(self, _event=None) -> None:
        if not self._running:
            self._on_select_file()

    def _update_model_hint(self, _event=None) -> None:
        model = self.var_model.get()
        self.lbl_model_hint.config(text=MODEL_HINTS.get(model, "Modelo Whisper"))

    def _clear_console(self) -> None:
        self.txt_console.config(state="normal")
        self.txt_console.delete("1.0", "end")
        self.txt_console.config(state="disabled")

    def _ui(self, fn: Callable[[], None]) -> None:
        """Ejecuta cambios de interfaz únicamente desde el hilo de Tkinter."""
        if self._closing:
            return
        if threading.get_ident() == self._main_thread_id:
            fn()
        else:
            self._ui_queue.put(fn)

    def _drain_ui_queue(self) -> None:
        if self._closing:
            return
        while True:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except tk.TclError:
                if not self._closing:
                    raise
        self.after(50, self._drain_ui_queue)

    def _on_close(self) -> None:
        if self._running and not messagebox.askyesno(
            APP_TITLE,
            "Hay una transcripción en curso. Si cierras ahora, el proceso se cancelará.\n\n"
            "¿Deseas cerrar la aplicación?",
            parent=self,
        ):
            return
        self.close()

    def close(self) -> None:
        """Cierra la aplicación e ignora actualizaciones tardías del worker."""
        if self._closing:
            return
        self._closing = True
        try:
            self.pbar.stop()
        except tk.TclError:
            pass
        self.destroy()

    def _append_console(self, message: str) -> None:
        def _append() -> None:
            self.txt_console.config(state="normal")
            self.txt_console.insert("end", f"{message}\n")
            self.txt_console.see("end")
            self.txt_console.config(state="disabled")

        self._ui(_append)

    def _set_busy(self, busy: bool) -> None:
        self._running = busy
        state = "disabled" if busy else "normal"
        self.btn_select.config(state=state)
        self.cmb_model.config(state="disabled" if busy else "readonly")
        self.cmb_device.config(state="disabled" if busy else "readonly")
        for widget in (self.ent_lang, self.ent_prompt, self.chk_fp16, self.lbl_drop):
            widget.config(state=state)
        self.btn_stop.config(state="normal" if busy else "disabled")
        self._refresh_summary()

    def _refresh_summary(self, counts=None) -> None:
        counts = self._batch.summary() if counts is None else counts
        finished = counts["completed"] + counts["error"]
        self.lbl_summary.config(text=(
            f"{counts['total']} archivos   ·   {counts['pending']} pendientes   ·   "
            f"{counts['processing']} en proceso   ·   {counts['completed']} completados   ·   {counts['error']} errores"))
        self.pbar.config(maximum=max(1, counts["total"]), value=finished)
        self.btn_process.config(state="normal" if counts["pending"] and not self._running else "disabled",
                                text=f"Procesar {counts['pending']} pendientes" if counts["pending"] else "Procesar cola")
        self.btn_retry.config(state="normal" if counts["error"] and not self._running else "disabled")
        self.btn_clear_finished.config(state="normal" if finished and not self._running else "disabled")
        self._update_selection_controls()

    def _set_status(self, text: str, *, error: bool = False, success: bool = False) -> None:
        active = next((item for item in self._batch.items if item.id == self._active_item_id), None)
        if active is not None and self._running:
            text = f"{active.path.name} · {text}"
            if self._stop_requested.is_set():
                text += " · Se detendrá al terminar este archivo"
        def _apply() -> None:
            style = "Status.TLabel"
            if error:
                style = "Error.Status.TLabel"
            elif success:
                style = "Success.Status.TLabel"
            self.lbl_status.config(text=text, style=style)
        self._ui(_apply)

    def _on_drop_file(self, event) -> str:
        if self._running:
            return "refuse_drop"
        self._add_files(Path(path) for path in _parse_drop_files(self, getattr(event, "data", "") or ""))
        return "copy"

    def _on_select_file(self) -> None:
        if self._running:
            return
        paths = filedialog.askopenfilenames(
            parent=self, title="Selecciona uno o varios audios y videos",
            filetypes=[
                ("Audio y video compatibles", "*.aac *.aif *.aiff *.flac *.m4a *.mp3 *.oga *.ogg *.opus *.wav *.wma *.mp4 *.mkv *.mov *.avi *.webm"),
                ("Audio", "*.aac *.aif *.aiff *.flac *.m4a *.mp3 *.oga *.ogg *.opus *.wav *.wma"),
                ("Videos", "*.mp4 *.mkv *.mov *.avi *.webm"),
                ("Todos los archivos", "*.*"),
            ],
        )
        self._add_files(Path(path) for path in paths)

    def _add_files(self, paths) -> None:
        if self._running:
            return
        valid, rejected = [], []
        for path in paths:
            try:
                valid.append(resolve_media_file(Path(path).expanduser().resolve()))
            except (OSError, ValueError) as exc:
                rejected.append(f"{Path(path).name}: {exc}")
        added, duplicates = self._batch.add(valid)
        for item in added:
            self._render_item(item)
        if added:
            self.tabs.select(0)
            self.file_tree.see(added[-1].id)
            self._set_status(f"{len(added)} añadidos · {duplicates} duplicados omitidos")
            self._append_console(f"Añadidos {len(added)} archivos a la cola. Duplicados omitidos: {duplicates}.")
        elif duplicates:
            self._set_status(f"{duplicates} archivos ya están en la cola")
        self._refresh_summary()
        if rejected:
            for error in rejected:
                self._append_console(f"No se añadió: {error}")
            preview = "\n".join(rejected[:5])
            if len(rejected) > 5:
                preview += f"\n… y {len(rejected)-5} más; consulta el registro."
            messagebox.showwarning(APP_TITLE, f"Se omitieron {len(rejected)} archivos. Los compatibles se añadieron.\n\n{preview}", parent=self)

    def _render_item(self, item: QueueItem) -> None:
        labels = {"pending": "Pendiente", "processing": "En proceso", "completed": "Completado", "error": "Error"}
        try:
            size = f"{item.path.stat().st_size / (1024**2):.1f} MB"
        except OSError:
            size = "—"
        values = (item.path.name, labels[item.status], size,
                  f"{item.elapsed:.1f} s" if item.status in ("completed", "error") else "—", str(item.path.parent))
        if self.file_tree.exists(item.id):
            self.file_tree.item(item.id, values=values, tags=(item.status,))
        else:
            self.file_tree.insert("", "end", iid=item.id, values=values, tags=(item.status,))

    def _set_file_detail(self, text: str) -> None:
        self.txt_detail.config(state="normal")
        self.txt_detail.delete("1.0", "end")
        self.txt_detail.insert("1.0", text)
        self.txt_detail.config(state="disabled")

    def _on_tree_select(self, _event=None) -> None:
        self._update_selection_controls()
        selected = self.file_tree.selection()
        if not selected:
            self._set_file_detail("Selecciona una fila para ver la ruta y los resultados. Ctrl o Mayús permiten seleccionar varias.")
            return
        item = next((item for item in self._batch.items if item.id == selected[0]), None)
        if item is None:
            return
        detail = str(item.path)
        if item.error:
            detail += f"\nError: {item.error}"
        elif item.outputs:
            detail += "\nGenerados: " + " · ".join(path.name for path in item.outputs)
        self._set_file_detail(detail)

    def _update_selection_controls(self) -> None:
        selected = bool(self.file_tree.selection())
        self.btn_remove.config(state="normal" if selected and not self._running else "disabled")
        self.btn_open_folder.config(state="normal" if selected else "disabled")

    def _remove_items_from_view(self, removed) -> None:
        for item in removed:
            self.file_tree.delete(item.id)
        self._refresh_summary()
        self._on_tree_select()
        if not self._batch.items:
            self._set_status("Listo para añadir archivos")

    def _on_remove(self) -> None:
        if not self._running:
            self._remove_items_from_view(self._batch.remove(self.file_tree.selection()))

    def _on_clear_finished(self) -> None:
        if not self._running:
            self._remove_items_from_view(self._batch.clear_finished())

    def _on_retry(self) -> None:
        if self._running:
            return
        for item in self._batch.items:
            if item.status == "error":
                item.status, item.error, item.elapsed = "pending", "", 0.0
                self._render_item(item)
        self._on_process()

    def _on_open_folder(self) -> None:
        selected = self.file_tree.selection()
        item = next((item for item in self._batch.items if selected and item.id == selected[0]), None)
        if item:
            try:
                open_folder(item.path.parent)
            except Exception as exc:
                messagebox.showerror(APP_TITLE, str(exc), parent=self)

    def _on_stop(self) -> None:
        if self._running:
            self._stop_requested.set()
            self.btn_stop.config(state="disabled")
            self._set_status("Se detendrá la cola al terminar el archivo actual")

    def _on_process(self) -> None:
        if self._running or not self._batch.summary()["pending"]:
            return
        self._stop_requested.clear()
        self._active_item_id = None
        self._set_busy(True)
        self._set_status("Preparando cola")
        self._append_console("Iniciando cola de transcripción…")
        self.tabs.select(0)
        worker = threading.Thread(
            target=self._worker_process,
            args=(self.var_model.get().strip() or DEFAULT_MODEL,
                  self.var_lang.get().strip() or DEFAULT_LANG,
                  self.var_device.get().strip().lower() or "auto", bool(self.var_fp16.get()),
                  self.var_prompt.get().strip() or None),
            daemon=True, name="whisper-transcription-queue")
        worker.start()

    def _on_item_changed(self, item: QueueItem) -> None:
        # Congela el estado antes de pasarlo al hilo Tk: el worker puede avanzar
        # al siguiente archivo antes de que se vacíe la cola de eventos.
        from dataclasses import replace
        snapshot = replace(item)
        counts = self._batch.summary()
        if item.status == "processing":
            self._active_item_id = item.id
        def apply():
            self._render_item(snapshot)
            self._refresh_summary(counts)
            if snapshot.status == "processing":
                self.file_tree.see(snapshot.id)
            if snapshot.id in self.file_tree.selection():
                self._on_tree_select()
        self._ui(apply)

    def _finish_batch(self, started_at: float, setup_error: str = "") -> None:
        self._active_item_id = None
        self._set_busy(False)
        counts = self._batch.summary()
        finished = counts['completed'] + counts['error']
        if setup_error:
            self._set_status(f"Error al preparar la cola: {setup_error}", error=True)
        else:
            prefix = "Cola detenida" if counts["pending"] else "Cola terminada"
            self._set_status(
                f"{prefix} · {finished}/{counts['total']} finalizados · {counts['completed']} completados · {counts['error']} errores",
                error=bool(counts["error"]), success=not counts["pending"] and not counts["error"])
        self._append_console(f"Tiempo de esta ejecución: {time.perf_counter()-started_at:.1f} s")

    def _resolve_device(self, requested_device: str, use_fp16: bool) -> tuple[str, bool]:
        torch_module = load_torch()
        device, use_fp16 = resolve_device(torch_module, requested_device, use_fp16)
        if device == "cpu" and requested_device not in ("auto", "cpu"):
            self._append_console(f"{requested_device} no está disponible. Se usará CPU.")
        return device, use_fp16

    def _release_model(self) -> None:
        device = self._loaded_model_key[1] if self._loaded_model_key else "cpu"
        self._loaded_model = None
        self._loaded_model_key = None
        gc.collect()
        clear_device_cache(load_torch(), device)

    def _get_model(self, whisper_module, model_size: str, device: str):
        model_key = (model_size, device)
        if self._loaded_model is not None and self._loaded_model_key == model_key:
            self._append_console("Reutilizando el modelo cargado en memoria.")
            return self._loaded_model

        if self._loaded_model is not None:
            self._append_console("Liberando el modelo anterior...")
            self._release_model()

        model = whisper_module.load_model(model_size, device=device)
        self._loaded_model = model
        self._loaded_model_key = model_key
        return model

    def _worker_process(
        self, model_size: str, lang: str, requested_device: str,
        use_fp16: bool, initial_prompt: Optional[str],
    ) -> None:
        started_at = time.perf_counter()
        setup_error = ""
        try:
            self._set_status("Cargando dependencias y comprobando configuración")
            whisper_module = load_whisper()
            require_ffmpeg()
            require_ffprobe()
            language = normalize_language(whisper_module, lang)
            device, use_fp16 = self._resolve_device(requested_device, use_fp16)
            self._output_plan = plan_output_paths(self._batch.items)

            def process_item(item):
                nonlocal device, use_fp16
                outputs = self._transcribe_item(item, whisper_module, model_size, language,
                                                device, use_fp16, initial_prompt)
                # Si la GPU requirió respaldo, conserva CPU durante este lote
                # para reutilizar el modelo y evitar otro fallo por archivo.
                if self._loaded_model_key == (model_size, "cpu") and device != "cpu":
                    device, use_fp16 = "cpu", False
                return outputs

            self._batch.run(
                process_item,
                self._on_item_changed,
                lambda: self._stop_requested.is_set() or self._closing,
            )
        except Exception as exc:
            setup_error = str(exc)
            self._append_console(f"Error al preparar la cola: {setup_error}\n{traceback.format_exc()}")
        finally:
            self._ui(lambda: self._finish_batch(started_at, setup_error))

    def _transcribe_item(
        self, item: QueueItem, whisper_module, model_size: str, language: Optional[str],
        device: str, use_fp16: bool, initial_prompt: Optional[str],
    ) -> tuple[Path, Path]:
        started_at = time.perf_counter()
        try:
            self._set_status("Comprobando pista de audio")
            media_file = resolve_media_file(item.path)
            audio_stream = probe_audio_stream(media_file)
            media_kind = "Audio" if media_file.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS else "Video"
            self._append_console(f"\n{media_kind}: {media_file}")
            self._append_console(f"Modelo: {model_size} · Idioma: {language or 'detección automática'} · Dispositivo: {device}")
            self._append_console(
                f"Pista: {audio_stream.get('codec_name', '?')} · "
                f"{audio_stream.get('channels', '?')} canal(es) · {audio_stream.get('sample_rate', '?')} Hz")
            torch_module = load_torch()
            self._append_console(f"Procesador: {device_description(torch_module, device)} · FP16: {'Sí' if use_fp16 else 'No'}")

            def transcribe_on_device(target_device, fp16):
                self._set_status("Cargando modelo")
                model = self._get_model(whisper_module, model_size, target_device)
                self._set_status("Transcribiendo audio")
                with torch_module.inference_mode():
                    return transcribe_media(model, media_file, language, fp16, initial_prompt)

            def release_failed_model():
                self._release_model()
                clear_device_cache(torch_module, device)

            result, device = run_with_cpu_fallback(
                transcribe_on_device, device, use_fp16, release_failed_model, self._append_console)
            self._append_console(f"Dispositivo utilizado: {device}")
            self._set_status("Guardando TXT y SRT")
            outputs = save_transcript_outputs(media_file, result, output_paths=self._output_plan[item.id])
            for output in outputs:
                self._append_console(f"Generado: {output}")
            self._append_console(f"Completado en {time.perf_counter() - started_at:.1f} s")
            return outputs
        except Exception as exc:
            diagnostic = traceback.format_exc()
            message = str(exc)
            if "out of memory" in message.lower():
                message = "No hay memoria suficiente. Prueba un modelo más pequeño, como 'base'."
                try:
                    clear_device_cache(load_torch(), device)
                except Exception:
                    pass
            self._append_console(f"Error en {item.path.name}: {message}\nDetalles técnicos:\n{diagnostic}")
            raise RuntimeError(message) from exc


def check_dependencies() -> dict:
    """Comprueba también los recursos nativos y datos del ejecutable empaquetado."""
    torch_module = load_torch()
    whisper_module = load_whisper()
    if TkinterDnD is None:
        raise RuntimeError(dependency_error("TkDnD", "tkinterdnd2"))
    root = TkinterDnD.Tk()
    root.withdraw()
    try:
        root.drop_target_register(DND_FILES)
        tkdnd_version = root.tk.call("package", "require", "tkdnd")
    finally:
        root.destroy()
    tokenizer = whisper_module.tokenizer.get_tokenizer(multilingual=True, language="es")
    assert tokenizer.decode(tokenizer.encode("Prueba de voz")) == "Prueba de voz"
    whisper_module.log_mel_spectrogram(torch_module.zeros(16000))
    # La descarga usa tqdm aun cuando la transcripcion tiene verbose=None.
    # Esta comprobacion debe funcionar tambien dentro del EXE sin consola.
    with whisper_module.tqdm(total=1, disable=False) as progress:
        progress.update(1)
    for name in ("ffmpeg", "ffprobe"):
        binary = find_media_binary(name)
        if not binary:
            raise RuntimeError(f"No se encontro {name}")
        subprocess.run([binary, "-version"], check=True, capture_output=True, timeout=30)
    return {
        "ok": True,
        "python": sys.version.split()[0],
        "torch": torch_module.__version__,
        "torch_build": torch_build_description(torch_module),
        "whisper": whisper_module.__version__,
        "tkdnd": tkdnd_version,
        "cuda": torch_module.cuda.is_available(),
        "devices": available_devices(torch_module),
        "torch_cuda": getattr(torch_module.version, "cuda", None),
        "torch_rocm": getattr(torch_module.version, "hip", None),
    }


def check_model_download() -> dict:
    """Prueba optativa de descarga sin cache e inferencia en el mismo ejecutable."""
    report = check_dependencies()
    torch_module = load_torch()
    whisper_module = load_whisper()
    with tempfile.TemporaryDirectory(prefix="video2text-check-") as directory:
        model = whisper_module.load_model(
            "tiny", device="cpu", download_root=str(Path(directory) / "models")
        )
        audio_file = Path(directory) / "silence.wav"
        with wave.open(str(audio_file), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * 8000)
        with torch_module.inference_mode():
            result = transcribe_media(model, audio_file, "es", False, None)
        if "text" not in result or "segments" not in result:
            raise RuntimeError("La prueba de inferencia no devolvio una transcripcion.")
    report.update(model="tiny", fresh_download=True, inference=True)
    return report


def main() -> None:
    checks = {
        "--check-dependencies": check_dependencies,
        "--check-model-download": check_model_download,
    }
    if len(sys.argv) == 3 and sys.argv[1] in checks:
        try:
            report = checks[sys.argv[1]]()
        except Exception:
            report = {"ok": False, "error": traceback.format_exc()}
        Path(sys.argv[2]).write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(0 if report["ok"] else 1)
    _windows_set_dpi_awareness()
    app = Video2TextApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        # Ctrl+C es una solicitud de cierre, no un error de la aplicación.
        app.close()


if __name__ == "__main__":
    main()
