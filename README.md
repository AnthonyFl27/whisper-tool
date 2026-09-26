# whisper-tool

Herramienta de terminal que **graba el audio que suena en tu equipo** (bocinas o audífonos), muestra un
**espectrograma en vivo** y lo **transcribe a texto** con [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
Sirve para transcribir audios que no se pueden descargar como archivo (por ejemplo, reproductores incrustados en
páginas web). Todo corre en local: el audio no sale de tu equipo.

## Requisitos

- Linux con **PipeWire** (usa `pw-record`).
- Python 3.9 o superior.
- Una terminal con colores de 24 bits (Konsole, GNOME Terminal, etc.).
- Conexión a internet la primera vez, para descargar el modelo elegido.

No hace falta instalar ffmpeg ni tener GPU.

## Uso

```bash
./whisper-tool
```

La primera ejecución crea el entorno virtual (`.venv-whisper/`) e instala las dependencias automáticamente.

### Modelos

| Modelo | Descarga aprox. | Notas |
|---|---|---|
| `tiny` | ~75 MB | el más rápido, menos preciso |
| `base` | ~145 MB | equilibrado |
| `small` | ~480 MB | buena precisión |
| `medium` | ~1,5 GB | muy preciso, lento en CPU |
| `large-v3` | ~3 GB | el más preciso, muy lento en CPU |

Los modelos se descargan de Hugging Face la primera vez que se usan y quedan en `~/.cache/huggingface`. El menú marca con ✔ los que ya tienes descargados; al elegir uno que no lo está se descarga mostrando una barra de progreso. Una vez descargado, las siguientes ejecuciones se saltan el menú y usan el último modelo (se recuerda en `.whisper-tool.json`). En la pantalla de inicio, `M` abre el menú para cambiar o descargar otro modelo.

### Controles

| Tecla | Acción |
|---|---|
| `↑` / `↓` o `1`-`5` | elegir modelo en el menú |
| `Enter` / `Espacio` | iniciar la grabación |
| `M` | cambiar o descargar otro modelo (pantalla de inicio) |
| `Espacio` / `P` | pausar o reanudar |
| `Ctrl+C` | terminar y transcribir |
| `Q` | salir sin grabar (en las pantallas iniciales) |

Las grabaciones y transcripciones se guardan en `records/` como `grabacion_AAAAMMDD_HHMMSS.wav` y `.txt`.

### Transcribir un archivo existente

```bash
./.venv-whisper/bin/python src/transcribe.py audio.mp3 [modelo]
```

## Notas

- La transcripción está fijada en **inglés** (`language="en"`). Para otro idioma, cambia ese valor en
  `src/grabar.py` y `src/transcribe.py`.
- Se captura **todo** el sonido del equipo: silencia notificaciones y música.
- Con audio vacío, Whisper puede inventar texto; el panel avisa cuando no hay señal.

