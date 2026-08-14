# Audio y Video a Texto

## Preparación del entorno

Ejecuta desde la carpeta `video2text` con Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Instala PyTorch con la variante apropiada para la GPU y CUDA antes de instalar
las demás dependencias. El entorno validado usa PyTorch `2.5.1+cu121` con una
NVIDIA RTX 4070 SUPER. Después instala el resto:

```powershell
python -m pip install -r .\requirements.txt
```

Verifica el entorno con:

```powershell
python -m pip check
python -c "import torch, whisper; print(torch.__version__, whisper.__version__)"
```

No versiones `.venv`, `Scripts`, `Lib`, `Include`, `share` ni `pyvenv.cfg`.

## Ejecutar

Desde la carpeta `video2text`:

```powershell
.\.venv\Scripts\python.exe .\video2text.py
```

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

## Generar el ejecutable

Ejecuta esta única línea desde la carpeta `video2text` después de cada actualización: `powershell -ExecutionPolicy Bypass -File .\build_exe.ps1`

El resultado se genera en `dist\Video2Text.exe` con `totext.ico`, Whisper,
`tkinterdnd2`, FFmpeg y FFprobe incluidos. El modelo Whisper se descarga y se
almacena en la caché del usuario durante el primer uso si todavía no existe.

Si cierras la ventana mientras se está transcribiendo un archivo, la aplicación
pedirá confirmación porque el procesamiento en curso se cancelará.

## Configuración recomendada

- **Modelo `turbo`**: recomendado para la RTX 4070 SUPER; es mucho más rápido
  que `large-v3` y conserva una precisión cercana.
- **Idioma `es`**: evita el paso de detección y es ideal para contenido en español.
  También admite nombres como `español`, `inglés` o `portugués`. Usa `auto`
  cuando el idioma sea desconocido.
- **Contexto**: escribe nombres propios, siglas o vocabulario técnico que
  esperas encontrar; Whisper lo utiliza como guía para la transcripción.
- **Dispositivo `auto` y FP16 activo**: usa CUDA cuando está disponible y
  reduce memoria y tiempo de procesamiento.

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
python -m unittest discover -s tests -p "test_whisper_integration.py" -v
```
