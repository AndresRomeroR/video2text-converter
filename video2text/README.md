# Audio y Video a Texto

## Preparación del entorno

El código contempla Windows, Linux y macOS. Usa una versión de Python admitida
por la distribución de PyTorch que corresponda al equipo. No copies `.venv`
entre equipos o sistemas: crea un entorno nuevo desde la carpeta `video2text`.

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux y macOS (terminal):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Necesitas Tkinter/Tcl/Tk y FFmpeg con FFprobe en el `PATH`. En Linux puede ser
necesario instalar el paquete de Tkinter de la distribución (por ejemplo,
`python3-tk` en Debian/Ubuntu). La interfaz requiere una sesión gráfica.

Para usar GPU, instala **primero** la distribución apropiada de PyTorch, siguiendo
el [selector oficial](https://pytorch.org/get-started/locally/) o la
[guía oficial de Intel XPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).
Comprueba la compatibilidad del modelo de GPU, sistema operativo, Python y
controlador; tener una GPU de cierta marca no basta. Para CPU, puedes continuar
directamente con las dependencias. No se fija una variante CPU/CUDA en el proyecto.

Instala después Whisper `20250625`, TkDnD `0.6.3` y sus dependencias:

```powershell
python -m pip install -r requirements.txt
```

Verifica el entorno con:

```powershell
python -m pip check
python -c "import torch, whisper; print(torch.__version__, whisper.__version__)"
```

Entorno comprobado en Windows: Python `3.14.7`, PyTorch `2.14.1+cpu`,
Whisper `20250625` y `tkinterdnd2 0.6.3`, con inferencia real del modelo `tiny`.
Las rutas de selección GPU y respaldo se prueban mediante simulaciones;
la aceleración debe validarse en cada equipo con su hardware real.

No versiones `.venv`, `Scripts`, `Lib`, `Include`, `share` ni `pyvenv.cfg`.

## Ejecutar

Desde la carpeta `video2text`:

```powershell
.\.venv\Scripts\python.exe .\video2text.py
```

En Linux/macOS: `.venv/bin/python video2text.py`. Con el entorno activado,
`python video2text.py` funciona en los tres sistemas.

Al ejecutar este comando se abre una ventana. Mientras la ventana siga abierta,
PowerShell no mostrará un nuevo prompt: eso es normal y significa que la
aplicación está ejecutándose.

La aplicación permite seleccionar o arrastrar archivos de audio y video. Whisper
transcribe el audio y genera archivos `.txt` y `.srt` en la misma carpeta y con
el mismo nombre base que el archivo original.

Formatos admitidos:

- Audio: AAC, AIF, AIFF, FLAC, M4A, MP3, OGA, OGG, OPUS, WAV y WMA.
- Video: AVI, MKV, MOV, MP4 y WEBM.

Con el entorno activado, el comando equivalente es:

```powershell
python .\video2text.py
```

## Cerrar la aplicación

Usa el botón **X** de la ventana. También puedes presionar `Ctrl+C` en
PowerShell; la aplicación ahora terminará limpiamente, sin mostrar un
`KeyboardInterrupt`.

Para abrirla sin mantener una consola visible:

```powershell
.\.venv\Scripts\pythonw.exe .\video2text.py
```

## Dispositivos y portabilidad

Deja **Dispositivo: `auto`** para detectar los backends disponibles en PyTorch:

| Equipo / aceleración | Dispositivo | Requisito |
| --- | --- | --- |
| CPU Intel, AMD o ARM | `cpu` | PyTorch compatible con el sistema y la arquitectura |
| GPU NVIDIA | `cuda` | PyTorch CUDA y controlador compatible |
| GPU AMD | `rocm` (internamente `cuda`) | PyTorch ROCm y combinación de GPU/sistema admitida |
| GPU Intel | `xpu` | PyTorch XPU y GPU/controlador admitidos |
| GPU Apple | `mps` | PyTorch con MPS y macOS compatible |

En `auto` se consulta CUDA/ROCm, después XPU y MPS, y finalmente CPU. La elección
manual de `cpu` siempre se respeta. Un backend solicitado pero no disponible
se sustituye por CPU. ROCm utiliza la API `torch.cuda` según las
[reglas de PyTorch](https://docs.pytorch.org/docs/stable/notes/hip.html).

Si la carga del modelo o la inferencia en GPU falla con un error de ejecución u
operación no implementada, se libera el modelo y se reintenta **una sola vez en
CPU/FP32**, conservando modelo, idioma y contexto. El registro indica el motivo
y el dispositivo final. Esto contempla también operaciones de Whisper que no
estén implementadas en MPS/XPU; detectar la GPU no garantiza que todas las
operaciones sean compatibles. FP16 se habilita únicamente en CUDA/ROCm.

El código es compartido; los ejecutables y entornos son específicos de su
sistema, arquitectura y distribución de PyTorch. El EXE CPU generado en este
equipo seguirá usando CPU al copiarlo a otro Windows compatible, aunque ese
equipo tenga GPU. Para aprovecharla, instala el PyTorch apropiado en un entorno
nuevo y ejecuta el código o vuelve a compilar. Un `.exe` de Windows no se
ejecuta de forma nativa en Linux/macOS.

## Generar el ejecutable

Con el entorno activado, ejecuta en el sistema de destino:

```bash
python build_exe.py
```

En Windows también puedes seguir usando:
`powershell -ExecutionPolicy Bypass -File .\build_exe.ps1`.

Los resultados son `dist/Video2Text.exe` en Windows, `dist/Video2Text` en Linux
y `dist/Video2Text.app` en macOS. Incluyen Whisper, `tkinterdnd2`, FFmpeg,
FFprobe y la distribución de PyTorch instalada en ese entorno. El script
conserva esa distribución; no cambia automáticamente de CPU a CUDA, ROCm o XPU.
Los modelos se descargan a la caché del usuario en el primer uso.

Compila y prueba por separado en cada sistema y arquitectura de destino,
como indica [PyInstaller](https://pyinstaller.org/en/stable/operating-mode.html).
Las bibliotecas del sistema y los controladores de GPU también deben ser
compatibles en el equipo receptor.

El script instala las dependencias en el mismo Python que usa para compilar
y comprueba Torch, Whisper y TkDnD antes de generar el ejecutable. Si un EXE
anterior indica que falta `whisper` o `tkinterdnd2`, debes reconstruirlo:
instalar paquetes con pip no cambia el contenido de ese EXE.

Para comprobar el ejecutable sin iniciar una transcripción:

```powershell
Start-Process -FilePath .\dist\Video2Text.exe -ArgumentList '--check-dependencies', 'diagnostico.json' -WindowStyle Hidden -Wait
Get-Content .\diagnostico.json
```

El informe debe indicar `"ok": true`; verifica Torch, Whisper, el tokenizador,
los datos de audio, arrastrar y soltar, FFmpeg y FFprobe. Incluye `devices`,
`torch_cuda` y `torch_rocm` para identificar lo disponible en ese entorno.
Esta comprobación valida dependencias; no ejecuta inferencia en todas las GPU.

En código fuente (cualquier sistema):
`python video2text.py --check-dependencies diagnostico.json`.

Si cierras la ventana mientras se está transcribiendo un archivo, la aplicación
pedirá confirmación porque el procesamiento en curso se cancelará.

## Configuración recomendada

- **Modelo `base` o `small` en CPU**: útiles para empezar con menos tiempo y
  memoria. `turbo` ofrece mayor precisión y también funciona en CPU; una GPU
  compatible con memoria suficiente puede acelerar el procesamiento.
- **Idioma `es`**: evita el paso de detección y es ideal para contenido en español.
  También admite nombres como `español`, `inglés` o `portugués`. Usa `auto`
  cuando el idioma sea desconocido.
- **Contexto**: escribe nombres propios, siglas o vocabulario técnico que
  esperas encontrar; Whisper lo utiliza como guía para la transcripción.
- **Dispositivo `auto`**: detecta la aceleración compatible con el entorno;
  usa CPU como respaldo. FP16 solo se aplica en CUDA/ROCm.

El modelo queda cargado en memoria después de la primera transcripción. Las
siguientes ejecuciones con el mismo modelo y dispositivo comienzan más rápido.

## Atajos de la interfaz

- `Ctrl+O`: buscar o cambiar el archivo de audio o video.
- `Ctrl+Enter`: iniciar la transcripción.
- Haz clic en la zona **Arrastra el audio o video aquí** para abrir el selector.
- **Limpiar registro** vacía la consola visual sin borrar archivos generados.

## Validación y pruebas

La aplicación usa `ffprobe` antes de cargar Whisper para rechazar archivos
dañados o sin pista de audio. FFmpeg y FFprobe deben estar disponibles en
`PATH`.

Ejecuta las pruebas rápidas con:

```powershell
python -m unittest discover -s tests -v
```

La prueba de integración crea un OGG temporal, carga el modelo instalado y
ejecuta una inferencia real. Requiere que el modelo esté descargado o acceso
para descargarlo:

```powershell
$env:VIDEO2TEXT_RUN_WHISPER_INTEGRATION = "1"
$env:VIDEO2TEXT_TEST_MODEL = "tiny"
python -m unittest discover -s tests -p "test_whisper_integration.py" -v
```

En Linux/macOS:

```bash
VIDEO2TEXT_RUN_WHISPER_INTEGRATION=1 VIDEO2TEXT_TEST_MODEL=tiny python -m unittest discover -s tests -v
```

Define `VIDEO2TEXT_TEST_DEVICE` como `cpu`, `cuda`, `rocm`, `xpu` o `mps` para
validar un backend concreto. Si solicitas un backend no disponible, la prueba
de integración debe fallar para evitar confundir una prueba CPU con una GPU.
