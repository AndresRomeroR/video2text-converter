EJECUTAR DESDE LA RAIZ DEL REPO

```
.\video2text\Scripts\python.exe .\video2text\video2text.py
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

ALTERNATIVA (ENTRANDO A LA CARPETA `video2text`)

```
cd .\video2text
.\Scripts\activate
python .\video2text.py
```

Para instalar o actualizar Whisper a la versión validada por el proyecto:

```
python -m pip install -r .\requirements.txt
```

PyTorch con CUDA se administra por separado porque el paquete correcto depende
de la GPU y de la versión CUDA. No conviene reemplazarlo con una actualización
genérica de `pip` sin comprobar primero la variante CUDA.

## Cerrar la aplicación

Usa el botón **X** de la ventana. También puedes presionar `Ctrl+C` en
PowerShell; la aplicación ahora terminará limpiamente, sin mostrar un
`KeyboardInterrupt`.

Para abrirla sin mantener una consola visible, ejecuta desde la raíz:

```
.\video2text\Scripts\pythonw.exe .\video2text\video2text.py
```

Si cierras la ventana mientras se está transcribiendo un archivo, la aplicación
pedirá confirmación porque el procesamiento en curso se cancelará.

## Configuración recomendada

- **Modelo `turbo`**: recomendado para la RTX 4070 SUPER; es mucho más rápido
  que `large-v3` y conserva una precisión cercana.
- **Idioma `es`**: evita el paso de detección y es ideal para contenido en español.
  Usa `auto` cuando el idioma sea desconocido.
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
