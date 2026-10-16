"""Compila para el sistema actual conservando la variante de PyTorch instalada."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def check_application(command: list[str], report: Path) -> dict:
    report.unlink(missing_ok=True)
    result = subprocess.run(
        [*command, "--check-dependencies", str(report)], cwd=ROOT, timeout=180
    )
    if not report.is_file():
        raise RuntimeError("La comprobacion no genero un informe de dependencias.")
    diagnostic = json.loads(report.read_text(encoding="utf-8"))
    if result.returncode or diagnostic.get("ok") is not True:
        raise RuntimeError(f"Dependencias incompletas: {diagnostic}")
    print(json.dumps(diagnostic, indent=2), flush=True)
    return diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Video2Text", help="Nombre del ejecutable")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Exige bibliotecas CUDA incluidas, aunque no haya GPU local")
    args = parser.parse_args()
    if not args.name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in args.name):
        parser.error("El nombre solo admite letras ASCII, numeros, guiones y guiones bajos.")
    binaries = []
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if not path:
            raise RuntimeError(f"Instala {name} y agregalo al PATH antes de compilar.")
        binaries.extend(["--add-binary", f"{path}{os.pathsep}."])

    # Sin --upgrade: respeta una instalacion CUDA, ROCm o XPU elegida por el usuario.
    run(sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "pyinstaller")
    run(sys.executable, "-m", "pip", "check")
    build = ROOT / "build" / args.name
    build.mkdir(parents=True, exist_ok=True)
    diagnostic = check_application(
        [sys.executable, str(ROOT / "video2text.py")], build / "source-dependency-check.json"
    )
    if args.require_cuda and not diagnostic.get("torch_cuda"):
        raise RuntimeError("Este entorno no incluye PyTorch CUDA. Instala la variante CUDA antes de compilar.")
    # onedir en macOS: genera un .app con sus bibliotecas; onefile en Windows/Linux.
    options = ["--onedir" if sys.platform == "darwin" else "--onefile"]
    if sys.platform in ("win32", "darwin"):
        options.append("--windowed")
    if sys.platform == "win32":
        options.extend(["--icon", str(ROOT / "totext.ico")])
    run(
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--name", args.name, *options,
        "--workpath", str(build / "pyinstaller"), "--specpath", str(build),
        "--add-data", f"{ROOT / 'totext.ico'}{os.pathsep}.", *binaries,
        "--collect-all", "tkinterdnd2", "--collect-all", "whisper",
        "--collect-all", "tiktoken", "--collect-submodules", "tiktoken_ext",
        str(ROOT / "video2text.py"),
    )
    executable = ROOT / "dist" / (f"{args.name}.exe" if sys.platform == "win32" else args.name)
    if sys.platform == "darwin":
        executable = ROOT / "dist" / f"{args.name}.app" / "Contents/MacOS" / args.name
    packaged = check_application([str(executable)], build / "dependency-check.json")
    if args.require_cuda and not packaged.get("torch_cuda"):
        raise RuntimeError("El ejecutable generado no contiene las bibliotecas CUDA esperadas.")
    print(f"Ejecutable generado y comprobado: {executable}")


if __name__ == "__main__":
    main()
