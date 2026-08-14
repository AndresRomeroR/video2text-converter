from __future__ import annotations

import ctypes
import gc
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

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
WINDOW_SIZE = (920, 700)
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
MODEL_HINTS = {
    "turbo": "Recomendado · rápido y muy preciso",
    "large-v3": "Máxima precisión · mayor uso de memoria",
    "medium": "Buen equilibrio para equipos intermedios",
    "small": "Ligero · menor precisión",
    "base": "Muy ligero · ideal para pruebas",
    "tiny": "El más rápido · precisión básica",
}


def load_torch():
    global _torch, _torch_error
    if _torch is not None:
        return _torch
    if _torch_error is not None:
        raise RuntimeError(
            "No se pudo cargar torch. Reinstala con: pip install --upgrade torch"
        ) from _torch_error
    try:
        import torch as torch_module
    except Exception as exc:
        _torch_error = exc
        raise RuntimeError(
            "No se pudo cargar torch. Reinstala con: pip install --upgrade torch"
        ) from exc
    _torch = torch_module
    return _torch


def load_whisper():
    global _whisper, _whisper_error
    if _whisper is not None:
        return _whisper
    if _whisper_error is not None:
        raise RuntimeError(
            "No se pudo cargar Whisper. Reinstala con: pip install -U openai-whisper"
        ) from _whisper_error
    try:
        import whisper as whisper_module
    except Exception as exc:
        _whisper_error = exc
        raise RuntimeError(
            "No se pudo cargar Whisper. Reinstala con: pip install -U openai-whisper"
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


def _normalize_tk_scaling_to_96dpi(root: tk.Tk) -> None:
    try:
        dpi = float(root.winfo_fpixels("1i"))
        if dpi > 0:
            root.tk.call("tk", "scaling", 96.0 / 72.0)
    except Exception:
        try:
            root.tk.call("tk", "scaling", 1.0)
        except Exception:
            pass


def _parse_drop_file(master: tk.Tk, data: str) -> Optional[str]:
    if not data:
        return None
    try:
        items = master.tk.splitlist(data)
        if not items:
            return None
        return str(items[0])
    except Exception:
        raw = str(data).strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1].strip()
        return raw or None


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
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "No se encontró FFmpeg. Instálalo y asegúrate de que el comando "
            "'ffmpeg' esté disponible en PowerShell."
        )
    return ffmpeg_path


def normalize_language(whisper_module, language: str) -> Optional[str]:
    language = language.strip().lower()
    if language in {"", "auto", "automático", "automatico"}:
        return None

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


class Video2TextApp(TkinterDnD.Tk if TkinterDnD else tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        _normalize_tk_scaling_to_96dpi(self)

        self.title(APP_TITLE)
        self._apply_theme()

        self._selected_file: Optional[Path] = None
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
        self._center_window()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Control-o>", self._shortcut_select)
        self.bind("<Control-Return>", self._shortcut_process)
        self.after(50, self._drain_ui_queue)

    def _apply_theme(self) -> None:
        style = ttk.Style(self)
        for theme in ("vista", "winnative", "xpnative", style.theme_use()):
            try:
                style.theme_use(theme)
                break
            except Exception:
                continue

        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#5f6368")
        style.configure("Section.TLabelframe", padding=14)
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Field.TLabel", font=("Segoe UI", 9, "bold"))
        style.configure("Hint.TLabel", font=("Segoe UI", 9), foreground="#5f6368")
        style.configure("File.TLabel", font=("Segoe UI", 9), foreground="#394457")
        style.configure("Drop.TLabel", font=("Segoe UI", 11, "bold"), padding=(18, 22))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(18, 8))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"), foreground="#305f8f")
        style.configure("Success.Status.TLabel", foreground="#19733b")
        style.configure("Error.Status.TLabel", foreground="#b42318")

    def _center_window(self) -> None:
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        width = min(WINDOW_SIZE[0], max(760, sw - 80))
        height = min(WINDOW_SIZE[1], max(620, sh - 100))
        self.minsize(760, 620)
        self.resizable(True, True)
        pos_x = (sw // 2) - (width // 2)
        pos_y = (sh // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{max(0, pos_x)}+{max(0, pos_y)}")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=20)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Audio y Video a Texto", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Transcripción local con Whisper · genera TXT y SRT junto al archivo",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(header, text="Ctrl+O  Abrir   ·   Ctrl+Enter  Procesar", style="Hint.TLabel").grid(
            row=0, column=1, rowspan=2, sticky="e"
        )

        file_box = ttk.LabelFrame(
            root, text="1  Selecciona el audio o video", style="Section.TLabelframe"
        )
        file_box.grid(row=1, column=0, sticky="ew")
        file_box.columnconfigure(0, weight=1)

        self.lbl_drop = ttk.Label(
            file_box,
            text="Arrastra el audio o video aquí\no haz clic para buscarlo",
            anchor="center",
            relief="groove",
            justify="center",
            cursor="hand2",
            style="Drop.TLabel",
        )
        self.lbl_drop.grid(row=0, column=0, sticky="ew")
        self.lbl_drop.bind("<Button-1>", self._on_drop_click)

        if TkinterDnD is not None:
            try:
                self.lbl_drop.drop_target_register(DND_FILES)
                self.lbl_drop.dnd_bind("<<Drop>>", self._on_drop_file)
            except Exception:
                pass

        self.lbl_file = ttk.Label(
            file_box,
            text="Ningún archivo seleccionado · audio: MP3, OGG, WAV, M4A, FLAC · video: MP4, MKV, MOV",
            style="File.TLabel",
            anchor="w",
            wraplength=820,
        )
        self.lbl_file.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        options = ttk.LabelFrame(
            root, text="2  Configura la transcripción", style="Section.TLabelframe"
        )
        options.grid(row=2, column=0, sticky="ew", pady=(14, 0))

        options.columnconfigure(0, weight=1)
        options.columnconfigure(1, weight=1)
        options.columnconfigure(2, weight=1)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Modelo", style="Field.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.cmb_model = ttk.Combobox(
            options,
            textvariable=self.var_model,
            state="readonly",
            values=("turbo", "large-v3", "medium", "small", "base", "tiny"),
        )
        self.cmb_model.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(4, 0))
        self.lbl_model_hint = ttk.Label(
            options, text=MODEL_HINTS[DEFAULT_MODEL], style="Hint.TLabel"
        )
        self.lbl_model_hint.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(3, 10))
        self.cmb_model.bind("<<ComboboxSelected>>", self._update_model_hint)

        ttk.Label(options, text="Idioma", style="Field.TLabel").grid(
            row=0, column=1, sticky="w", padx=8
        )
        self.ent_lang = ttk.Entry(options, textvariable=self.var_lang)
        self.ent_lang.grid(row=1, column=1, sticky="ew", padx=8, pady=(4, 0))
        ttk.Label(options, text="Código: es, en o auto", style="Hint.TLabel").grid(
            row=2, column=1, sticky="w", padx=8, pady=(3, 10)
        )

        ttk.Label(options, text="Dispositivo", style="Field.TLabel").grid(
            row=0, column=2, sticky="w", padx=8
        )
        self.cmb_device = ttk.Combobox(
            options,
            textvariable=self.var_device,
            state="readonly",
            values=("auto", "cuda", "cpu"),
        )
        self.cmb_device.grid(row=1, column=2, sticky="ew", padx=8, pady=(4, 0))
        ttk.Label(options, text="Auto usa CUDA si está disponible", style="Hint.TLabel").grid(
            row=2, column=2, sticky="w", padx=8, pady=(3, 10)
        )

        self.chk_fp16 = ttk.Checkbutton(
            options,
            text="Aceleración FP16",
            variable=self.var_fp16,
        )
        self.chk_fp16.grid(row=1, column=3, sticky="w", padx=(8, 0), pady=(4, 0))
        ttk.Label(options, text="Menos memoria y mayor velocidad", style="Hint.TLabel").grid(
            row=2, column=3, sticky="w", padx=(8, 0), pady=(3, 10)
        )

        ttk.Label(options, text="Contexto opcional", style="Field.TLabel").grid(
            row=3, column=0, columnspan=4, sticky="w"
        )
        self.ent_prompt = ttk.Entry(
            options,
            textvariable=self.var_prompt,
        )
        self.ent_prompt.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        ttk.Label(
            options,
            text="Ejemplo: Azure OpenAI, FastAPI, nombres propios o vocabulario técnico",
            style="Hint.TLabel",
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(3, 0))

        actions = ttk.Frame(root)
        actions.grid(row=3, column=0, sticky="ew", pady=14)
        actions.columnconfigure(2, weight=1)

        self.btn_select = ttk.Button(actions, text="Cambiar archivo", command=self._on_select_file)
        self.btn_select.grid(row=0, column=0, sticky="w")

        self.btn_open_folder = ttk.Button(
            actions,
            text="Abrir carpeta del archivo",
            command=self._on_open_folder,
            state="disabled",
        )
        self.btn_open_folder.grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.btn_process = ttk.Button(
            actions,
            text="Transcribir archivo",
            command=self._on_process,
            state="disabled",
            style="Accent.TButton",
        )
        self.btn_process.grid(row=0, column=3, sticky="e")

        console_box = ttk.LabelFrame(root, text="3  Progreso", style="Section.TLabelframe")
        console_box.grid(row=4, column=0, sticky="nsew")
        console_box.columnconfigure(0, weight=1)
        console_box.rowconfigure(2, weight=1)

        progress_header = ttk.Frame(console_box)
        progress_header.grid(row=0, column=0, sticky="ew")
        progress_header.columnconfigure(0, weight=1)
        self.lbl_status = ttk.Label(progress_header, text="Listo", style="Success.Status.TLabel")
        self.lbl_status.grid(row=0, column=0, sticky="w")
        ttk.Button(progress_header, text="Limpiar registro", command=self._clear_console).grid(
            row=0, column=1, sticky="e"
        )

        self.pbar = ttk.Progressbar(console_box, orient="horizontal", mode="indeterminate")
        self.pbar.grid(row=1, column=0, sticky="ew", pady=(10, 10))

        self.txt_console = ScrolledText(console_box, height=12, wrap="word", state="disabled")
        self.txt_console.grid(row=2, column=0, sticky="nsew")
        self.txt_console.configure(
            font=("Cascadia Mono", 9),
            background="#111827",
            foreground="#dbe5f1",
            insertbackground="#ffffff",
            relief="flat",
            padx=10,
            pady=8,
        )

        if TkinterDnD is None:
            self._append_console(
                "Arrastrar y soltar no está disponible porque tkinterdnd2 no está instalado. "
                "Haz clic en la zona de selección para buscar el archivo."
            )

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
        state_normal = "disabled" if busy else "normal"
        self.btn_select.config(state=state_normal)
        self.btn_open_folder.config(
            state=("disabled" if self._selected_file is None else state_normal)
        )
        self.btn_process.config(
            state=("disabled" if busy or self._selected_file is None else "normal")
        )
        self.cmb_model.config(state="disabled" if busy else "readonly")
        self.cmb_device.config(state="disabled" if busy else "readonly")
        self.ent_lang.config(state="disabled" if busy else "normal")
        self.ent_prompt.config(state="disabled" if busy else "normal")
        self.chk_fp16.config(state="disabled" if busy else "normal")

        try:
            self.lbl_drop.config(state=state_normal)
        except Exception:
            pass

        if busy:
            self.pbar.start(12)
        else:
            self.pbar.stop()

    def _set_status(self, text: str) -> None:
        def _apply() -> None:
            style = "Status.TLabel"
            if "error" in text.lower():
                style = "Error.Status.TLabel"
            elif any(word in text.lower() for word in ("listo", "éxito", "terminado")):
                style = "Success.Status.TLabel"
            self.lbl_status.config(text=text, style=style)

        self._ui(_apply)

    def _on_drop_file(self, event) -> None:
        path_str = _parse_drop_file(self, getattr(event, "data", "") or "")
        if not path_str:
            return
        self._set_selected_file(Path(path_str).expanduser())

    def _on_select_file(self) -> None:
        path_str = filedialog.askopenfilename(
            parent=self,
            title="Selecciona el audio o video",
            filetypes=[
                (
                    "Audio y video compatibles",
                    "*.aac *.aif *.aiff *.flac *.m4a *.mp3 *.oga *.ogg *.opus *.wav *.wma "
                    "*.mp4 *.mkv *.mov *.avi *.webm",
                ),
                ("Audio", "*.aac *.aif *.aiff *.flac *.m4a *.mp3 *.oga *.ogg *.opus *.wav *.wma"),
                ("Videos", "*.mp4 *.mkv *.mov *.avi *.webm"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path_str:
            return
        self._set_selected_file(Path(path_str))

    def _set_selected_file(self, path: Path) -> None:
        try:
            path = resolve_media_file(path)
        except Exception as exc:
            messagebox.showwarning(APP_TITLE, str(exc), parent=self)
            return

        self._selected_file = path
        size_mb = path.stat().st_size / (1024**2)
        self.lbl_file.config(text=f"{path.name}  ·  {size_mb:.1f} MB\n{path.parent}")
        self.btn_process.config(state="normal" if not self._running else "disabled")
        self.btn_open_folder.config(state="normal" if not self._running else "disabled")
        self._set_status("Archivo listo para procesar")
        self._append_console(f"Archivo seleccionado: {path}")

    def _on_open_folder(self) -> None:
        if self._selected_file is None:
            return
        target = self._selected_file.parent
        try:
            os.startfile(str(target))
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc), parent=self)

    def _on_process(self) -> None:
        if self._running:
            return
        if self._selected_file is None:
            messagebox.showwarning(
                APP_TITLE, "Selecciona primero un archivo de audio o video.", parent=self
            )
            return

        self._set_busy(True)
        self._set_status("Procesando")
        self._append_console("Iniciando transcripción...")
        media_file = self._selected_file
        model_size = self.var_model.get().strip() or DEFAULT_MODEL
        lang = self.var_lang.get().strip() or DEFAULT_LANG
        initial_prompt = self.var_prompt.get().strip() or None
        requested_device = self.var_device.get().strip().lower() or "auto"
        use_fp16 = bool(self.var_fp16.get())
        worker = threading.Thread(
            target=self._worker_process,
            args=(media_file, model_size, lang, requested_device, use_fp16, initial_prompt),
            daemon=True,
            name="whisper-transcription",
        )
        worker.start()

    def _resolve_device(self, requested_device: str, use_fp16: bool) -> tuple[str, bool]:
        torch_module = load_torch()

        cuda_available = bool(torch_module.cuda.is_available())

        if requested_device == "auto":
            device = "cuda" if cuda_available else "cpu"
        elif requested_device == "cuda":
            if not cuda_available:
                self._append_console("CUDA no está disponible. Se usará CPU.")
                device = "cpu"
                use_fp16 = False
            else:
                device = "cuda"
        else:
            device = "cpu"
            use_fp16 = False

        if device != "cuda":
            use_fp16 = False

        return device, use_fp16

    def _get_model(self, whisper_module, model_size: str, device: str):
        model_key = (model_size, device)
        if self._loaded_model is not None and self._loaded_model_key == model_key:
            self._append_console("Reutilizando el modelo cargado en memoria.")
            return self._loaded_model

        if self._loaded_model is not None:
            self._append_console("Liberando el modelo anterior...")
            self._loaded_model = None
            self._loaded_model_key = None
            gc.collect()
            torch_module = load_torch()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

        model = whisper_module.load_model(model_size, device=device)
        self._loaded_model = model
        self._loaded_model_key = model_key
        return model

    def _worker_process(
        self,
        media_file: Path,
        model_size: str,
        lang: str,
        requested_device: str,
        use_fp16: bool,
        initial_prompt: Optional[str],
    ) -> None:
        started_at = time.perf_counter()
        device = "cpu"
        try:
            self._set_status("Cargando dependencias")
            self._append_console("Cargando dependencias (torch/whisper)...")
            whisper_module = load_whisper()

            media_file = resolve_media_file(media_file)
            require_ffmpeg()
            language = normalize_language(whisper_module, lang)
            device, use_fp16 = self._resolve_device(requested_device, use_fp16)

            media_kind = (
                "Audio" if media_file.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS else "Video"
            )
            self._append_console(f"{media_kind}: {media_file.name}")
            self._append_console(f"Modelo: {model_size}")
            self._append_console(f"Tamaño: {media_file.stat().st_size / (1024 ** 2):.1f} MB")
            self._append_console(f"Idioma: {language or 'detección automática'}")
            self._append_console(f"Dispositivo: {device}")
            self._append_console(f"FP16: {'Sí' if use_fp16 else 'No'}")
            if initial_prompt:
                self._append_console("Se usará el contexto indicado para nombres y vocabulario.")

            torch_module = load_torch()
            if device == "cuda":
                self._append_console(f"GPU: {torch_module.cuda.get_device_name(0)}")

            self._set_status("Cargando modelo")
            self._append_console("Cargando modelo Whisper...")
            model = self._get_model(whisper_module, model_size, device)

            self._set_status("Transcribiendo audio")
            self._append_console(f"Transcribiendo {media_kind.lower()}...")
            with torch_module.inference_mode():
                result = model.transcribe(
                    str(media_file),
                    language=language,
                    fp16=use_fp16,
                    initial_prompt=initial_prompt,
                    word_timestamps=False,
                    verbose=None,
                )

            txt_file = media_file.with_suffix(".txt")
            srt_file = media_file.with_suffix(".srt")

            self._set_status("Guardando resultados")
            txt_content, srt_content = build_transcript_outputs(result)
            write_windows_text(txt_file, txt_content)
            write_windows_text(srt_file, srt_content)

            self._append_console(f"TXT generado: {txt_file}")
            self._append_console(f"SRT generado: {srt_file}")
            self._append_console(f"Tiempo total: {time.perf_counter() - started_at:.1f} s")
            self._set_status("Proceso terminado con éxito")

            self._ui(
                lambda: messagebox.showinfo(
                    APP_TITLE,
                    "Proceso finalizado exitosamente.\n\n"
                    f"TXT: {txt_file}\n"
                    f"SRT: {srt_file}",
                    parent=self,
                )
            )
        except Exception as exc:
            error_message = str(exc)
            if "out of memory" in error_message.lower():
                error_message = (
                    "La GPU se quedó sin memoria. Prueba el modelo 'turbo' o uno más pequeño."
                )
                if device == "cuda":
                    load_torch().cuda.empty_cache()
            self._append_console(f"Error: {error_message}")
            self._set_status("Error")
            self._ui(lambda msg=error_message: messagebox.showerror(APP_TITLE, msg, parent=self))
        finally:
            self._ui(lambda: self._set_busy(False))


def main() -> None:
    _windows_set_dpi_awareness()
    app = Video2TextApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        # Ctrl+C es una solicitud de cierre, no un error de la aplicación.
        app.close()


if __name__ == "__main__":
    main()
