from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Optional

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


APP_TITLE = "Video a Texto - Whisper"
WINDOW_SIZE = (760, 520)
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
DEFAULT_MODEL = "large"
DEFAULT_LANG = "es"


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


def resolve_video_file(video_file: Path) -> Path:
    if not video_file.exists() or not video_file.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {video_file}")

    suffix = video_file.suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(
            f"Extensión no compatible: {suffix or '(sin extensión)'}.\n"
            f"Extensiones permitidas: {supported}"
        )
    return video_file


def srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def write_windows_text(path: Path, content: str) -> None:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(normalized.replace("\n", "\r\n"))


class Video2TextApp(TkinterDnD.Tk if TkinterDnD else tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        _normalize_tk_scaling_to_96dpi(self)

        self.title(APP_TITLE)
        self._apply_theme()

        self._selected_file: Optional[Path] = None
        self._running = False

        self.var_model = tk.StringVar(value=DEFAULT_MODEL)
        self.var_lang = tk.StringVar(value=DEFAULT_LANG)
        self.var_device = tk.StringVar(value="auto")
        self.var_fp16 = tk.BooleanVar(value=True)

        self._build_ui()
        self._center_window()

    def _apply_theme(self) -> None:
        style = ttk.Style(self)
        for theme in ("vista", "winnative", "xpnative", style.theme_use()):
            try:
                style.theme_use(theme)
                break
            except Exception:
                continue

    def _center_window(self) -> None:
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        width = min(WINDOW_SIZE[0], max(680, sw - 80))
        height = min(WINDOW_SIZE[1], max(480, sh - 120))
        self.minsize(width, height)
        self.resizable(False, False)
        pos_x = (sw // 2) - (width // 2)
        pos_y = (sh // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{max(0, pos_x)}+{max(0, pos_y)}")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text=APP_TITLE, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            root,
            text="Arrastra un video o selecciónalo manualmente. "
            "El archivo .txt y .srt se generará en la misma carpeta del video.",
            foreground="#444444",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))

        file_box = ttk.LabelFrame(root, text="Archivo", padding=10)
        file_box.pack(fill="x")

        self.lbl_drop = ttk.Label(
            file_box,
            text="Suelta aquí el video (.mp4, .mkv, .mov, .avi, .webm)",
            anchor="center",
            relief="groove",
            padding=(10, 16),
        )
        self.lbl_drop.pack(fill="x")

        if TkinterDnD is not None:
            try:
                self.lbl_drop.drop_target_register(DND_FILES)
                self.lbl_drop.dnd_bind("<<Drop>>", self._on_drop_file)
            except Exception:
                pass

        self.lbl_file = ttk.Label(file_box, text="Archivo: (ninguno)")
        self.lbl_file.pack(anchor="w", pady=(8, 0))

        options = ttk.LabelFrame(root, text="Configuración", padding=10)
        options.pack(fill="x", pady=(10, 0))

        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Modelo").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.cmb_model = ttk.Combobox(
            options,
            textvariable=self.var_model,
            state="readonly",
            values=("tiny", "base", "small", "medium", "large"),
        )
        self.cmb_model.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(options, text="Idioma").grid(row=0, column=2, sticky="w", padx=(16, 8), pady=4)
        self.ent_lang = ttk.Entry(options, textvariable=self.var_lang)
        self.ent_lang.grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(options, text="Dispositivo").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.cmb_device = ttk.Combobox(
            options,
            textvariable=self.var_device,
            state="readonly",
            values=("auto", "cuda", "cpu"),
        )
        self.cmb_device.grid(row=1, column=1, sticky="ew", pady=4)

        self.chk_fp16 = ttk.Checkbutton(
            options,
            text="Usar FP16 en GPU",
            variable=self.var_fp16,
        )
        self.chk_fp16.grid(row=1, column=3, sticky="w", pady=4)

        progress_box = ttk.LabelFrame(root, text="Progreso", padding=10)
        progress_box.pack(fill="x", pady=(10, 0))

        self.lbl_status = ttk.Label(progress_box, text="Estado: Listo")
        self.lbl_status.pack(anchor="w")

        self.pbar = ttk.Progressbar(progress_box, orient="horizontal", mode="indeterminate")
        self.pbar.pack(fill="x", pady=(8, 0))

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(10, 0))

        self.btn_select = ttk.Button(actions, text="Buscar video", command=self._on_select_file)
        self.btn_select.pack(side="left")

        self.btn_open_folder = ttk.Button(
            actions,
            text="Abrir carpeta del archivo",
            command=self._on_open_folder,
            state="disabled",
        )
        self.btn_open_folder.pack(side="left", padx=(8, 0))

        self.btn_process = ttk.Button(
            actions,
            text="Procesar",
            command=self._on_process,
            state="disabled",
        )
        self.btn_process.pack(side="right")

        console_box = ttk.LabelFrame(root, text="Consola", padding=10)
        console_box.pack(fill="both", expand=True, pady=(10, 0))

        self.txt_console = ScrolledText(console_box, height=12, wrap="word", state="disabled")
        self.txt_console.pack(fill="both", expand=True)

        if TkinterDnD is None:
            self._append_console(
                "Arrastrar y soltar no está disponible porque tkinterdnd2 no está instalado. "
                "Puedes usar el botón 'Buscar video'."
            )

    def _ui(self, fn) -> None:
        self.after(0, fn)

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

        try:
            self.lbl_drop.config(state=state_normal)
        except Exception:
            pass

        if busy:
            self.pbar.start(12)
        else:
            self.pbar.stop()

    def _set_status(self, text: str) -> None:
        self._ui(lambda: self.lbl_status.config(text=f"Estado: {text}"))

    def _on_drop_file(self, event) -> None:
        path_str = _parse_drop_file(self, getattr(event, "data", "") or "")
        if not path_str:
            return
        self._set_selected_file(Path(path_str).expanduser())

    def _on_select_file(self) -> None:
        path_str = filedialog.askopenfilename(
            parent=self,
            title="Selecciona el video",
            filetypes=[
                ("Videos", "*.mp4 *.mkv *.mov *.avi *.webm"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not path_str:
            return
        self._set_selected_file(Path(path_str))

    def _set_selected_file(self, path: Path) -> None:
        try:
            path = resolve_video_file(path)
        except Exception as exc:
            messagebox.showwarning(APP_TITLE, str(exc), parent=self)
            return

        self._selected_file = path
        self.lbl_file.config(text=f"Archivo: {path}")
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
            messagebox.showwarning(APP_TITLE, "Selecciona primero un archivo de video.", parent=self)
            return

        self._set_busy(True)
        self._set_status("Procesando")
        self._append_console("Iniciando transcripción...")
        worker = threading.Thread(target=self._worker_process, daemon=True)
        worker.start()

    def _resolve_device(self) -> tuple[str, bool]:
        requested_device = self.var_device.get().strip().lower() or "auto"
        use_fp16 = bool(self.var_fp16.get())

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

    def _worker_process(self) -> None:
        try:
            self._set_status("Cargando dependencias")
            self._append_console("Cargando dependencias (torch/whisper)...")
            whisper_module = load_whisper()

            video_file = resolve_video_file(self._selected_file)
            model_size = self.var_model.get().strip() or DEFAULT_MODEL
            lang = self.var_lang.get().strip() or DEFAULT_LANG
            device, use_fp16 = self._resolve_device()

            self._append_console(f"Video: {video_file.name}")
            self._append_console(f"Modelo: {model_size}")
            self._append_console(f"Idioma: {lang}")
            self._append_console(f"Dispositivo: {device}")
            self._append_console(f"FP16: {'Sí' if use_fp16 else 'No'}")

            self._set_status("Cargando modelo")
            self._append_console("Cargando modelo Whisper...")
            model = whisper_module.load_model(model_size, device=device)

            self._set_status("Transcribiendo audio")
            self._append_console("Transcribiendo video...")
            result = model.transcribe(
                str(video_file),
                language=lang,
                fp16=use_fp16,
                word_timestamps=False,
            )

            txt_file = video_file.with_suffix(".txt")
            srt_file = video_file.with_suffix(".srt")

            self._set_status("Escribiendo TXT")
            transcript = (result.get("text") or "").strip()
            write_windows_text(txt_file, transcript + "\n")

            self._set_status("Escribiendo SRT")
            srt_lines: list[str] = []
            for idx, seg in enumerate(result.get("segments", []), start=1):
                srt_lines.append(str(idx))
                srt_lines.append(
                    f"{srt_timestamp(float(seg['start']))} --> {srt_timestamp(float(seg['end']))}"
                )
                srt_lines.append(str(seg.get("text", "")).strip())
                srt_lines.append("")

            write_windows_text(srt_file, "\n".join(srt_lines))

            self._append_console(f"TXT generado: {txt_file}")
            self._append_console(f"SRT generado: {srt_file}")
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
            self._append_console(f"Error: {exc}")
            self._set_status("Error")
            self._ui(lambda msg=str(exc): messagebox.showerror(APP_TITLE, msg, parent=self))
        finally:
            self._ui(lambda: self._set_busy(False))


def main() -> None:
    _windows_set_dpi_awareness()
    app = Video2TextApp()
    app.mainloop()


if __name__ == "__main__":
    main()
