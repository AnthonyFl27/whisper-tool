# AGENTS.md

Contexto para asistentes de IA que trabajen en esta carpeta (`~/Documentos/whisper-tool`).

## Qué es este proyecto

Herramienta de terminal para **grabar el audio que suena en el equipo** (lo que sale por bocinas o audífonos), mostrar un **espectrograma en vivo** y **transcribirlo a texto** con `faster-whisper`. Se creó para poder transcribir audios de listening que no se pueden descargar como archivo (por ejemplo, reproductores incrustados en plataformas web).

El usuario habla español. Comentarios, mensajes de la interfaz y documentación van en español; los nombres de variables y funciones, en inglés/mixto como ya está en el código.

## Entorno

- Sistema: Fedora con KDE Plasma, audio con **PipeWire** (`pw-record`, `pactl`, `wpctl` disponibles).
- Python 3.14 en un entorno virtual local: `.venv-whisper/`. Siempre ejecutar con `./.venv-whisper/bin/python`, no con el `python3` del sistema (no tiene numpy ni faster-whisper).
- Paquetes instalados en el venv: `faster-whisper` (trae su propio decodificador, **no hace falta ffmpeg**), `numpy`, `rich`.
- Los comandos con privilegios se ejecutan con `sudo`. No instalar paquetes del sistema sin preguntar.
- En la caché de Hugging Face del usuario (`~/.cache/huggingface`) solo está `tiny` (el último usado, guardado en `.whisper-tool.json`); `base` **no** está descargado. Al probar, pasar `tiny` como argumento: con `base` la herramienta empieza a descargarlo.
- La carpeta es un repositorio git, sin commits salvo que el usuario los haga; no hacer commit ni push sin que lo pida.

## Archivos

| Archivo | Función |
|---|---|
| `whisper-tool` | Punto de entrada (nombre de la herramienta). Script bash que hace `cd` a su carpeta (resolviendo enlaces simbólicos), comprueba que exista `pw-record`, crea `.venv-whisper/` e instala `src/requirements.txt` si es la primera ejecución, y ejecuta `src/grabar.py`. Acepta el modelo como argumento; sin argumento, `grabar.py` muestra un menú. |
| `src/requirements.txt`, `src/AGENTS.md` | Dependencias de Python (las instala `whisper-tool`) y este archivo de contexto; viven en `src/`. |
| `README.md`, `LICENSE`, `.gitignore` | Archivos para publicar el proyecto en git, en la raíz (el `.gitignore` excluye el venv, `__pycache__/`, `records/` y `.whisper-tool.json`). |
| `src/grabar.py` | Interfaz de terminal con `rich`: grabación, espectro, espectrograma, pausa y transcripción. |
| `src/transcribe.py` | Transcribe un archivo de audio existente: `./.venv-whisper/bin/python src/transcribe.py audio.mp3 [modelo]`. Imprime el texto en la salida estándar. |
| `records/` | Se crea al grabar. Se guarda en la raíz del proyecto (no dentro de `src/`). Guarda `grabacion_AAAAMMDD_HHMMSS.wav` y `.txt` con la transcripción. |
| `.venv-whisper/` | Entorno virtual. No editar a mano; si se rompe, recrear con `python3 -m venv .venv-whisper` y `pip install -r src/requirements.txt`. |

## Cómo funciona `src/grabar.py`

-1. Selección de modelo (`main`): si se pasa como argumento se usa ese; si no, se usa el último guardado en `.whisper-tool.json` (`load_last_model`, solo si sigue descargado) y se va directo a la pantalla de inicio. Si no hay ninguno, `select_model` muestra un menú (tiny, base, small, medium, large-v3; flechas o 1-5, `Enter` confirma, `Q`/`Ctrl+C` salen) y marca con `is_downloaded` los que ya están en la caché de Hugging Face (`models--Systran--faster-whisper-<modelo>/snapshots/*/model.bin`). Si el elegido no está descargado, `download_model` lo baja con `snapshot_download` y una barra de `rich` (una subclase de `tqdm` que reenvía los bytes; depende de los internos de `huggingface_hub`, así que si una versión nueva rompe la barra, revisar `RichBar`). Si falla, vuelve al menú. En la pantalla de inicio, `M` reabre el menú para cambiar o descargar otro modelo. Sin TTY usa `base`.
0. Antes de grabar, `wait_for_start` muestra una pantalla "Listo para grabar" y espera: `Enter` o `Espacio` inician, `Q` o `Ctrl+C` salen sin grabar ni transcribir. Sin TTY se omite la espera y graba de inmediato. El nombre del archivo usa la hora de inicio real (después de la espera).
1. Lanza `pw-record --raw -P stream.capture.sink=true --rate 16000 --channels 1 --format s16 -` como subproceso. Esa propiedad hace que capture la **salida predeterminada** (el "monitor"), no el micrófono. El audio llega crudo (PCM 16 bit little-endian, 16 kHz, mono) por la salida estándar. **`--raw` es imprescindible**: sin él, `pw-record` (1.6.10) escribe una cabecera AU y las muestras en big-endian, el audio se lee como ruido y la transcripción sale vacía.
2. El bucle principal usa `select` sobre el pipe de `pw-record` y sobre `stdin` (modo `cbreak` de `tty`, así `Ctrl+C` sigue generando `KeyboardInterrupt`).
3. El audio se analiza en bloques fijos de `CHUNK` (100 ms, así cada fila del espectrograma dura lo mismo) con FFT (`analyze`): nivel en dB y magnitudes del espectro. Se guardan las últimas `HISTORY` lecturas sin agrupar en bandas: al dibujar, `to_columns` las convierte a tantas columnas como quepan (escala logarítmica de 80 Hz a 7,5 kHz, normalizadas a 0..1 entre `DB_FLOOR` -80 dB y `DB_CEIL` -25 dB; en graves interpola para no dibujar escalones).
4. **Interfaz adaptable:** las pantallas son `Adaptive`, un renderable que se arma en cada cuadro con el tamaño actual de la terminal, así nada queda con tamaño fijo al redimensionar. `recording_screen` reparte el alto: cabecera (estado, cronómetro, pico, modelo), medidor a todo el ancho, barras del espectro (2 a 8 filas), eje de frecuencias, espectrograma (el resto del alto, con medios bloques `▀`: dos lecturas por línea, la más reciente arriba) y pie con atajos. El contenido se limita a `MAX_WIDTH` columnas y se centra; con menos de 18 filas se quitan las separaciones y en ventanas muy chicas se omiten el espectro y el espectrograma. Los valores por debajo de `EMPTY` se dibujan como fondo. Las celdas se dibujan con `Segment` (clase `Cells`, que une celdas contiguas iguales) porque con `Text` era demasiado lento; `record` usa `auto_refresh=False` y redibuja hasta 20 veces por segundo, menos si dibujar tarda. Los menús (`screen`) se centran vertical y horizontalmente, y los atajos pasan a una línea por atajo si no caben.
5. **Controles:** `Enter`/`Espacio` inician y `M` cambia de modelo (pantalla de espera); ya grabando, `Espacio` o `P` pausa/reanuda y `Ctrl+C` termina. En pausa el audio se sigue leyendo del pipe pero se **descarta** (para que `pw-record` no se bloquee) y el cronómetro solo cuenta audio grabado.
6. Si pasan más de 2 s por debajo de `SILENCE_DB` (-55 dB), el panel muestra "SIN SEÑAL DE AUDIO" en amarillo.
7. Al terminar guarda el `.wav`, avisa si no hubo audio (`peak_db < SILENCE_DB`), transcribe con `WhisperModel(modelo, device="cpu", compute_type="int8")` en inglés (`language="en"`) y guarda el `.txt`.

## Cosas a tener en cuenta

- **Idioma fijo:** la transcripción fuerza `language="en"` porque el uso previsto son audios de inglés. Si se necesita otro idioma, hay que cambiarlo en `src/grabar.py` (`transcribe`) y en `src/transcribe.py`.
- **Silencio y alucinaciones:** Whisper inventa texto como "..." con audio vacío. Es esperado; el aviso de "sin señal" existe para detectarlo.
- **Captura de todo el sistema:** entra cualquier sonido del equipo (notificaciones, música). Conviene silenciar todo lo demás.
- **Necesita una terminal real** (TTY) con colores de 24 bits (Konsole funciona). Sin TTY, el script graba igual pero no hay control por teclado.
- **Sin pruebas automáticas.** Lo verificado hasta ahora: análisis con señales sintéticas (440 Hz + 2 kHz), grabación real en silencio, pantalla de inicio (`Q` cancela, `Enter` inicia) y pausa/terminación por teclas en un pseudo-terminal. Al probar con `pty.fork`, ejecutar con la ruta `./.venv-whisper/bin/python` como argv[0], si no el venv no se activa. La captura de audio real sonando la probó el usuario.
- No copiar el venv a otra ruta: guarda rutas absolutas. Recrearlo en su lugar.

## Convenciones al modificar

- Mantener el código simple y en un solo archivo por herramienta; sin dependencias nuevas salvo que hagan falta de verdad.
- Probar cambios de la interfaz en un pseudo-terminal (módulo `pty` de Python) o con una señal sintética, no solo leyendo el código.
- Borrar las grabaciones de prueba de `records/` al terminar.
- Antes de instalar software, descargar modelos grandes o enviar datos fuera del equipo, **preguntar al usuario**.

## Nota de uso responsable

Estas herramientas se crearon para ayudar con audios de tareas y ejercicios en los que se **permite el uso de IA**. Si el usuario las quiere usar en otro contexto, recordarle que revise las reglas de esa evaluación.
