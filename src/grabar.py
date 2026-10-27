#!/usr/bin/env python3
"""Graba el audio del sistema con un espectrograma en vivo y lo transcribe con faster-whisper.

Uso: whisper-tool [modelo]      (sin modelo, muestra un menú: tiny, base, small, medium, large-v3)
Preparar la grabacion con Enter o Espacio: empieza sola al detectar sonido, se pausa tras unos segundos de
silencio y sigue cuando vuelve el sonido. Detenerla con Ctrl+C.
"""
import json
import os
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
import tty
from collections import deque
from datetime import datetime
from functools import lru_cache
from itertools import groupby, islice
from pathlib import Path

import numpy as np
from rich.align import Align
from rich.color import Color
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (BarColumn, DownloadColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn, TransferSpeedColumn)
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

RATE = 16000
CHUNK = 1600  # 100 ms por lectura del espectrograma
FFT_SIZE = 2048
FREQ_LO, FREQ_HI = 80, 7500  # rango de frecuencias mostrado (escala logarítmica)
BLOCKS = " ▁▂▃▄▅▆▇█"
SILENCE_DB = -55.0  # por debajo de esto se considera sin senal
LOW_DB = -35.0  # un pico por debajo de esto (pero sobre SILENCE_DB) es audio muy bajo: Whisper rinde peor
AUTO_PAUSE = float(os.environ.get("WHISPER_TOOL_AUTO_PAUSE", 12))  # s de silencio tras los que la grabación se pausa sola
PREROLL = 5  # bloques (0,5 s) que se conservan antes de que suene algo, para no cortar el inicio al empezar o reanudar
PARAGRAPH_GAP = 1.5  # s de pausa entre frases que abren un párrafo nuevo en el texto
PARAGRAPH_MAX = 400  # caracteres a partir de los cuales se abre un párrafo nuevo al terminar una frase
DB_FLOOR, DB_CEIL = -80.0, -25.0
HISTORY = 600  # lecturas guardadas para el espectrograma (60 s, de sobra para una terminal alta)
MAX_WIDTH = 120  # ancho máximo del contenido; en terminales más anchas se centra
EMPTY = 0.06  # valores por debajo se dibujan como fondo, así el silencio no llena la pantalla
LEVELS = 48  # pasos de color (se cachean los estilos)

ROOT = Path(__file__).resolve().parent.parent  # raiz del proyecto (src/ esta un nivel abajo)
DATA_DIR = Path(os.environ.get("WHISPER_TOOL_DIR", ROOT))  # donde viven records/ y la config; cambiarlo sirve para pruebas
OUT_DIR = DATA_DIR / "records"
CONFIG_PATH = DATA_DIR / ".whisper-tool.json"  # recuerda el último modelo usado
FREQS = np.fft.rfftfreq(FFT_SIZE, 1 / RATE)
console = Console()


def analyze(samples):
    """Devuelve (nivel_dB, magnitudes del espectro). Las bandas se calculan al dibujar, según el ancho."""
    x = samples.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    level_db = 20 * np.log10(rms) if rms > 1e-9 else -120.0
    return level_db, (np.abs(np.fft.rfft(x * np.hanning(len(x)), n=FFT_SIZE)) / len(x)).astype(np.float32)


@lru_cache(maxsize=8)
def columns(n):
    """Para n columnas en escala logarítmica: bins de inicio y fin de cada una, y posición del centro en bins."""
    edges = np.geomspace(FREQ_LO, FREQ_HI, n + 1)
    lo = np.searchsorted(FREQS, edges[:-1])
    hi = np.maximum(np.searchsorted(FREQS, edges[1:]), lo + 1)
    center = np.sqrt(edges[:-1] * edges[1:]) / FREQS[1]
    return lo, hi, center


def to_columns(mags, n):
    """Magnitudes (lecturas x bins) -> valores 0..1 en n columnas de frecuencia."""
    lo, hi, center = columns(n)
    cs = np.concatenate([np.zeros((len(mags), 1), np.float32), np.cumsum(mags, axis=1)], axis=1)
    mean = (cs[:, hi] - cs[:, lo]) / (hi - lo)
    # en graves una columna es más angosta que un bin: se interpola para no dibujar escalones
    i0 = np.floor(center).astype(int)
    w = center - i0
    interp = mags[:, i0] * (1 - w) + mags[:, i0 + 1] * w
    bands = np.where(hi - lo > 1, mean, interp)
    db = 20 * np.log10(np.maximum(bands, 1e-9))
    return np.clip((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0, 1)


def heat(v):
    """Color de 0..1: azul oscuro -> cian -> amarillo -> rojo."""
    stops = [(0.0, (15, 20, 60)), (0.35, (30, 140, 200)), (0.65, (240, 210, 60)), (1.0, (240, 60, 50))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))
    return stops[-1][1]


PALETTE = [Color.from_rgb(*heat(i / (LEVELS - 1))) for i in range(LEVELS)]


def level(values):
    """Índices de PALETTE para valores 0..1; -1 donde se dibuja fondo."""
    return np.where(values < EMPTY, -1, (values * (LEVELS - 1)).astype(int)).tolist()


@lru_cache(maxsize=None)
def half_cell(top, bottom):
    """Celda con dos lecturas apiladas (medio bloque arriba y medio abajo)."""
    if top < 0 and bottom < 0:
        return Segment(" ")
    if top < 0:
        return Segment("▄", Style(color=PALETTE[bottom]))
    return Segment("▀", Style(color=PALETTE[top], bgcolor=PALETTE[bottom] if bottom >= 0 else None))


@lru_cache(maxsize=None)
def bar_cell(color, eighths, base):
    if eighths == 0:
        return Segment("▁", Style(color="grey23")) if base else Segment(" ")
    return Segment(BLOCKS[eighths], Style(color=PALETTE[max(color, 0)]))


class Cells:
    """Bloque de celdas ya coloreadas (una lista de Segment por línea); mucho más rápido que armar un Text.
    Las celdas contiguas iguales se unen en un solo segmento, que es lo que más pesa al dibujar."""

    def __init__(self, rows, indent=0):
        self.rows = [[Segment(" " * indent)] + [Segment(seg.text * len(list(run)), seg.style)
                                                 for seg, run in groupby(row)] for row in rows]

    def __rich_console__(self, console, options):
        for row in self.rows:
            yield from row
            yield Segment.line()


class Adaptive:
    """Renderable que se arma al dibujarse con el tamaño actual de la terminal, así se adapta al redimensionarla."""

    def __init__(self, build):
        self.build = build

    def __rich_console__(self, console, options):
        yield self.build(options.max_width, options.height or console.height)


def spectrum(values, height, indent):
    """Espectro actual en barras verticales, una columna por banda, con bloques de octavos."""
    colors = level(values)
    rows = []
    for row in range(height, 0, -1):
        eighths = np.clip(((values * height - (row - 1)) * 8).astype(int), 0, 8).tolist()
        rows.append([bar_cell(c, e, row == 1) for c, e in zip(colors, eighths)])
    return Cells(rows, indent)


def waterfall(history, width, lines, indent):
    """Espectrograma con la lectura más reciente arriba; cada línea de texto muestra dos lecturas."""
    grid = np.zeros((lines * 2, width))
    recent = list(islice(history, lines * 2))
    if recent:
        grid[:len(recent)] = to_columns(np.array(recent), width)
    q = level(grid)
    return Cells([[half_cell(t, b) for t, b in zip(top, bottom)] for top, bottom in zip(q[0::2], q[1::2])], indent)


def freq_axis(width):
    """Etiquetas de frecuencia en su posición (escala logarítmica); se omiten las que no caben."""
    chars = [" "] * width
    end = 0
    for hz, label in [(100, "100"), (200, "200"), (500, "500"), (1000, "1k"), (2000, "2k"), (5000, "5k Hz")]:
        col = round(width * np.log(hz / FREQ_LO) / np.log(FREQ_HI / FREQ_LO))
        if col >= end and col + len(label) <= width:
            chars[col:col + len(label)] = label
            end = col + len(label) + 2
    return Text("".join(chars), style="grey50", no_wrap=True, overflow="crop")


def meter(level_db, width):
    """Medidor de nivel a todo el ancho, con el valor en dB a la derecha."""
    label = f"{max(level_db, -99.9):6.1f} dB"
    n = max(0, width - len(label) - 2)
    filled = int(min(max((level_db - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0), 1) * n)
    text = Text(no_wrap=True, overflow="crop")
    for i in range(n):
        if i < filled:
            text.append("━", style=Style(color=PALETTE[int(i / n * (LEVELS - 1))]))
        else:
            text.append("─", style="grey23")
    text.append("  " + label, style="grey85")
    return text


def spread(left, right, width):
    """Una línea con un texto a la izquierda y otro a la derecha (el de la derecha se omite si no cabe)."""
    line = left.copy()
    gap = width - left.cell_len - right.cell_len
    if gap >= 2:
        line.append(" " * gap)
        line.append_text(right)
    line.no_wrap, line.overflow = True, "crop"
    return line


def pick(width, *options):
    """La primera opción de texto que cabe en el ancho (de la más larga a la más corta)."""
    return next((o for o in options if len(o) <= width), options[-1])


def recording_screen(st, width, height):
    """Pantalla de grabación para el tamaño actual: el espectro y el espectrograma ocupan el espacio libre."""
    cw = max(10, min(width - (4 if width >= 50 else 2), MAX_WIDTH))
    roomy = height >= 18  # con poca altura se quitan márgenes y separaciones
    free = height - 4 - (5 if roomy else 0)  # cabecera, medidor, eje y pie (+ márgenes y separaciones)
    bar_rows = max(2, min(8, free // 3)) if free >= 6 else 0
    wf_lines = max(0, free - bar_rows)

    blink = int(time.monotonic() * 2) % 2 == 0
    if st.mode == "waiting":
        status = Text("○ ESPERANDO SONIDO" if blink else "  ESPERANDO SONIDO", style="bold cyan")
        keys = ("Espacio / P: empezar ya    Ctrl+C: salir", "Espacio: empezar    Ctrl+C: salir", "Ctrl+C: salir")
    elif st.mode == "auto":
        status = Text("‖ PAUSA POR SILENCIO" if blink else "  PAUSA POR SILENCIO", style="bold cyan")
        keys = ("Sigue sola al volver el sonido    Espacio / P: seguir ya    Ctrl+C: terminar y transcribir",
                "Espacio: seguir    Ctrl+C: terminar", "Ctrl+C: terminar")
    else:
        if st.mode == "paused":
            status = Text("‖ EN PAUSA" if blink else "  EN PAUSA", style="bold cyan")
        elif st.silent_for > 2.0:
            status = Text("● SIN SEÑAL DE AUDIO", style="bold yellow")
        else:
            status = Text("● GRABANDO" if blink else "○ GRABANDO", style="bold red")
        keys = ("Espacio / P: pausar o continuar    Ctrl+C: terminar y transcribir",
                "Espacio: pausa    Ctrl+C: terminar", "Ctrl+C: terminar")
    mins, secs = divmod(int(st.elapsed), 60)
    status.append(f"   {mins:02d}:{secs:02d}", style="bold white")
    info = Text.assemble(("pico ", "grey50"), (f"{max(st.peak_db, -99.9):.1f} dB", "grey70"))
    if SILENCE_DB < st.peak_db < LOW_DB and st.elapsed >= 5:
        info.append("  ·  volumen bajo", style="bold yellow")
    info.append("  ·  modelo ", style="grey50")
    info.append(st.model, style="cyan")
    keys = pick(cw - 14, *keys)
    span = Text(f"últimos {wf_lines * 2 * CHUNK / RATE:.0f} s" if wf_lines else "", style="grey50")

    indent = (width - cw) // 2  # el contenido se centra en la terminal
    margin = Text(" " * indent, no_wrap=True, overflow="crop")
    gap = [Text("")] if roomy else []
    parts = gap + [margin + spread(status, info, cw)] + gap + [margin + meter(st.level_db, cw)] + gap
    if bar_rows:
        parts.append(spectrum(to_columns(st.current[None], cw)[0], bar_rows, indent))
    if bar_rows or wf_lines:
        parts.append(margin + freq_axis(cw))
    if wf_lines:
        parts.append(waterfall(st.history, cw, wf_lines, indent))
    parts += gap + [margin + spread(Text(keys, style="grey50"), span, cw)]
    return Group(*parts)


def hints(items, width):
    """Atajos de teclado en una sola línea si caben; si no, uno por línea."""
    line = "    ".join(items)
    return Text(line if len(line) <= width else "\n".join(items), style="grey62")


def screen(body):
    """Pantalla de menú: título y contenido centrados en la terminal, sin marco."""
    return Align.center(Group(Text("Grabador de audio del sistema", style="bold"), Text(""), body),
                        vertical="middle")


MODELS = [  # (nombre en faster-whisper, descarga aprox., descripcion)
    ("tiny", "~75 MB", "el más rápido, menos preciso"),
    ("base", "~145 MB", "equilibrado (recomendado)"),
    ("small", "~480 MB", "buena precisión"),
    ("medium", "~1,5 GB", "muy preciso, lento en CPU"),
    ("large-v3", "~3 GB", "el más preciso, muy lento en CPU"),
]


def is_downloaded(name):
    """True si el modelo ya está en la caché de Hugging Face (no habrá que descargarlo)."""
    from huggingface_hub import constants

    repo = Path(constants.HF_HUB_CACHE) / f"models--Systran--faster-whisper-{name}"
    return any(repo.glob("snapshots/*/model.bin"))


def load_last_model():
    """Último modelo elegido, o None si no hay o ya no está descargado."""
    try:
        name = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("model")
    except (OSError, ValueError, AttributeError):
        return None
    return name if name in {m[0] for m in MODELS} and is_downloaded(name) else None


def save_last_model(name):
    try:
        CONFIG_PATH.write_text(json.dumps({"model": name}), encoding="utf-8")
    except OSError:
        pass  # no es crítico


def download_model(name):
    """Descarga el modelo con una barra de progreso. Devuelve True si quedó descargado."""
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    progress = Progress(TextColumn("[bold cyan]{task.description}"), BarColumn(), DownloadColumn(),
                        TransferSpeedColumn(), TimeRemainingColumn(), console=console)
    task = progress.add_task(f"Descargando {name}", total=None)

    class RichBar(tqdm):
        """Barra de tqdm que no dibuja nada y reenvía los bytes descargados a la barra de rich."""

        def __init__(self, *args, **kwargs):
            # solo cuenta la barra de bytes totales (la de "Downloading bytes" duplicaría el avance)
            self.reports = kwargs.get("unit") == "B" and not str(kwargs.get("desc", "")).startswith("Downloading")
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            self.n += n or 0
            self.refresh()

        def refresh(self, *args, **kwargs):
            if self.reports and self.total:
                progress.update(task, total=self.total, completed=self.n)

    console.print(f"[cyan]Modelo «{name}» no descargado; descargando desde Hugging Face…[/cyan]")
    try:
        with progress:
            snapshot_download(f"Systran/faster-whisper-{name}", tqdm_class=RichBar,
                              allow_patterns=["config.json", "preprocessor_config.json", "model.bin",
                                              "tokenizer.json", "vocabulary.*"])
    except KeyboardInterrupt:
        console.print("[yellow]Descarga cancelada.[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]No se pudo descargar el modelo:[/red] {e}")
        return False
    console.print(f"[green]Modelo «{name}» descargado.[/green]")
    return True


def select_model(default="base"):
    """Menú para elegir el modelo de Whisper. Devuelve su nombre, o None si el usuario cancela."""
    names = [m[0] for m in MODELS]
    if not sys.stdin.isatty():
        return default
    index = names.index(default) if default in names else 1
    old_term = termios.tcgetattr(sys.stdin)
    downloaded = {name: is_downloaded(name) for name in names}

    def view(width, height):
        wide = width >= 76  # en ventanas angostas se omite la descripción
        body = Text(no_wrap=True, overflow="crop")
        body.append("Elige el modelo de Whisper\n\n", style="bold white")
        for i, (name, size, desc) in enumerate(MODELS):
            style = "bold cyan" if i == index else "grey70"
            body.append(f"{'▶' if i == index else ' '} {i + 1}. {name:<9}{size:<9}", style=style)
            if wide:
                body.append(f" {desc:<34}", style=style)
            body.append(("✔ descargado" if wide else "✔") if downloaded[name] else "", style="green")
            body.append("\n")
        body.append("\nLos modelos sin marcar se descargan al elegirlos.", style="grey50")
        return screen(Group(body, Text(""), hints(["↑/↓ o 1-5: elegir", "Enter: continuar", "Q / Ctrl+C: salir"],
                                                  width - 4)))

    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(Adaptive(view), console=console, screen=True, refresh_per_second=10) as live:
            while True:
                key = os.read(sys.stdin.fileno(), 32).decode(errors="ignore")
                if key in ("\n", "\r"):
                    return names[index]
                if key in ("q", "Q"):
                    return None
                if key == "\x1b[A" and index > 0:
                    index -= 1
                elif key == "\x1b[B" and index < len(MODELS) - 1:
                    index += 1
                elif key.isdigit() and 1 <= int(key) <= len(MODELS):
                    index = int(key) - 1
                live.refresh()
    except KeyboardInterrupt:
        return None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)


def wait_for_start(model_name):
    """Espera Enter o Espacio para iniciar. Devuelve "start", "change" (M: cambiar de modelo) o "quit"."""
    if not sys.stdin.isatty():
        return "start"  # sin teclado no hay forma de esperar: graba de inmediato
    old_term = termios.tcgetattr(sys.stdin)

    def view(width, height):
        return screen(Group(
            Text("Listo para grabar", style="bold white"), Text(""),
            Text.assemble(("Modelo: ", "grey62"), (model_name, "bold cyan")), Text(""),
            Text(f"Empieza a grabar sola cuando suene algo y se pausa tras {AUTO_PAUSE:.0f} s de silencio",
                 style="grey62"), Text(""),
            hints(["Enter / Espacio: preparar grabación", "M: cambiar modelo", "Q / Ctrl+C: salir"], width - 4),
        ))

    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(Adaptive(view), console=console, screen=True, refresh_per_second=10):
            while True:
                key = os.read(sys.stdin.fileno(), 32).decode(errors="ignore")
                if any(k in key for k in ("\n", "\r", " ")):
                    return "start"
                if any(k in key for k in ("m", "M")):
                    return "change"
                if any(k in key for k in ("q", "Q")):
                    return "quit"
    except KeyboardInterrupt:
        return "quit"
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)


class State:
    """Lo que muestra la pantalla de grabación; se dibuja con el tamaño de la terminal en cada cuadro."""

    def __init__(self, model):
        self.model = model
        self.elapsed = 0.0
        self.level_db = self.peak_db = -120.0
        self.current = np.zeros(len(FREQS), np.float32)  # espectro de las barras (con caída suave)
        self.history = deque(maxlen=HISTORY)
        self.silent_for = 0.0
        # waiting: espera sonido para empezar; recording; paused: pausa manual; auto: pausada por silencio
        self.mode = "recording"


class WavWriter:
    """Escribe el .wav mientras se graba y corrige la cabecera cada segundo: si el programa muere a mitad de una
    grabación, el archivo queda con todo el audio escrito hasta el último segundo y se puede abrir."""

    def __init__(self, path):
        self.file = open(path, "wb")
        self.size = 0  # bytes de audio escritos
        self.synced = 0
        self.sync()

    def sync(self):
        header = (b"RIFF" + struct.pack("<I", 36 + self.size) + b"WAVEfmt " +
                  struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16) + b"data" + struct.pack("<I", self.size))
        self.file.seek(0)
        self.file.write(header)
        self.file.seek(0, os.SEEK_END)
        self.file.flush()
        self.synced = self.size

    def write(self, data):
        self.file.write(data)
        self.size += len(data)
        if self.size - self.synced >= RATE * 2:
            self.sync()

    def close(self):
        self.sync()
        self.file.close()


def record(wav_path, model_name):
    """Graba hasta Ctrl+C. Con teclado, espera a que suene algo para empezar; la grabación se pausa sola tras
    AUTO_PAUSE s de silencio y sigue cuando vuelve el sonido. Devuelve (segundos grabados, pico en dB)."""
    st = State(model_name)
    wav = WavWriter(wav_path)
    proc = subprocess.Popen(
        ["pw-record", "--raw", "-P", "stream.capture.sink=true", "--rate", str(RATE), "--channels", "1",
         "--format", "s16", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    fd = proc.stdout.fileno()
    use_keys = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if use_keys else None
    st.mode = "waiting" if use_keys else "recording"  # sin teclado no hay cómo empezar a mano: graba de inmediato
    pending = bytearray()  # audio aún no analizado (se analiza en bloques de CHUNK: 100 ms por fila)
    preroll = deque(maxlen=PREROLL)  # últimos bloques sin grabar
    carry = b""
    last_sound = last_draw = time.monotonic()
    draw_every = 0.05  # hasta 20 cuadros por segundo; menos si la terminal es tan grande que dibujar tarda
    try:
        if use_keys:
            tty.setcbreak(sys.stdin.fileno())  # Ctrl+C sigue funcionando
        with Live(Adaptive(lambda w, h: recording_screen(st, w, h)), console=console, auto_refresh=False,
                  screen=True) as live:
            while True:
                watch = [fd] + ([sys.stdin] if use_keys else [])
                ready, _, _ = select.select(watch, [], [], 0.1)
                if use_keys and sys.stdin in ready:
                    key = os.read(sys.stdin.fileno(), 32).decode(errors="ignore")
                    if any(k in key for k in (" ", "p", "P")):
                        if st.mode == "recording":
                            st.mode = "paused"
                            st.current[:] = 0
                            st.level_db = -120.0
                            pending.clear()
                        else:  # pausa manual o por silencio, o esperando sonido: empieza o sigue ya
                            st.mode = "recording"
                            last_sound = time.monotonic()
                            preroll.clear()
                        last_draw = 0.0  # redibuja ya
                if fd in ready:
                    data = os.read(fd, CHUNK * 2)
                    if not data:
                        break
                    data = carry + data  # se mantiene la alineación de 16 bit aunque el audio se descarte
                    carry = data[len(data) // 2 * 2:]
                    data = data[: len(data) // 2 * 2]
                    if st.mode != "paused":  # en pausa manual el audio se lee y se descarta
                        pending += data
                        while len(pending) >= CHUNK * 2:
                            block = bytes(pending[:CHUNK * 2])
                            del pending[:CHUNK * 2]
                            st.level_db, mag = analyze(np.frombuffer(block, dtype=np.int16))
                            sound = st.level_db > SILENCE_DB
                            if sound:
                                last_sound = time.monotonic()
                            if st.mode == "recording":
                                wav.write(block)
                            elif sound:  # waiting/auto: suena algo, se empieza o se sigue (con el medio segundo previo)
                                st.mode = "recording"
                                for old in preroll:
                                    wav.write(old)
                                preroll.clear()
                                wav.write(block)
                            else:
                                preroll.append(block)
                            st.peak_db = max(st.peak_db, st.level_db)
                            st.history.appendleft(mag)
                            st.current = np.maximum(mag, st.current * 0.6)
                now = time.monotonic()
                if st.mode == "paused":
                    last_sound = now
                st.elapsed = wav.size / 2 / RATE
                st.silent_for = now - last_sound
                if st.mode == "recording" and st.silent_for >= AUTO_PAUSE:
                    st.mode = "auto"
                if now - last_draw >= draw_every:
                    live.refresh()
                    last_draw = time.monotonic()
                    draw_every = max(0.05, 2 * (last_draw - now))
    except KeyboardInterrupt:
        pass
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        proc.terminate()
        proc.wait()
        if st.mode == "recording" and pending:
            wav.write(bytes(pending))  # el último bloque, que aún no llegaba a los 100 ms
        wav.close()
    return wav.size / 2 / RATE, st.peak_db


def clock(t):
    """Segundos -> MM:SS."""
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def to_paragraphs(segments):
    """Segmentos (inicio, fin, texto) -> texto en párrafos separados por una línea en blanco. Se abre un párrafo
    nuevo tras una pausa de PARAGRAPH_GAP s, o al terminar una frase cuando el párrafo ya es largo."""
    paragraphs, current, last_end = [], [], 0.0
    for start, end, text in segments:
        if current and (start - last_end >= PARAGRAPH_GAP
                        or (len(" ".join(current)) >= PARAGRAPH_MAX and current[-1].endswith((".", "?", "!")))):
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        last_end = end
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def transcribe(wav_path, model_name):
    """Transcribe mostrando una barra de progreso y cada segmento con su tiempo. Devuelve [(inicio, fin, texto)]."""
    from faster_whisper import WhisperModel

    with console.status("[bold cyan]Cargando modelo…[/bold cyan]", spinner="dots"):
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        # vad_filter descarta los silencios antes de transcribir: evita el texto inventado en las pausas.
        # word_timestamps ajusta cada segmento a sus palabras; sin eso Whisper los estira sobre los silencios
        # y no se podrían medir las pausas entre frases para separar párrafos
        segments, info = model.transcribe(str(wav_path), language="en", vad_filter=True, word_timestamps=True)
    progress = Progress(TextColumn("[bold cyan]Transcribiendo"), BarColumn(), TaskProgressColumn(),
                        TimeElapsedColumn(), console=console)
    result = []
    with progress:
        task = progress.add_task("", total=info.duration or None)
        for s in segments:  # el trabajo real ocurre aquí: el generador transcribe a medida que se recorre
            text = s.text.strip()
            if text:
                result.append((s.start, s.end, text))
                progress.console.print(Text.assemble((f"{clock(s.start)}  ", "grey50"), text))
            progress.update(task, completed=min(s.end, info.duration))
        progress.update(task, completed=info.duration)
    return result


def ask_retry():
    """True si el usuario pide reintentar (R o Enter); False con Q o Ctrl+C. Sin teclado no pregunta."""
    if not sys.stdin.isatty():
        return False
    console.print("[bold]R[/bold] / Enter: reintentar    [bold]Q[/bold] / Ctrl+C: salir")
    old_term = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            key = os.read(sys.stdin.fileno(), 32).decode(errors="ignore").lower()
            if key in ("r", "\n", "\r"):
                return True
            if key == "q":
                return False
    except KeyboardInterrupt:
        return False
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)


def transcribe_with_retry(wav_path, model_name):
    """Devuelve (segmentos, segundos que tardó), o (None, 0) si falló o se canceló y no se reintentó.
    El .wav ya está guardado, así que un fallo no pierde la grabación."""
    while True:
        start = time.monotonic()
        try:
            return transcribe(wav_path, model_name), time.monotonic() - start
        except KeyboardInterrupt:
            console.print("\n[yellow]Transcripción cancelada.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]La transcripción falló:[/red] {e}")
        console.print(f"[grey62]El audio sigue guardado en {wav_path}[/grey62]")
        if not ask_retry():
            return None, 0


def copy_to_clipboard(text):
    """Copia el texto con wl-copy (Wayland) o xsel (X11). None si está desactivado; True/False según resultado."""
    if os.environ.get("WHISPER_TOOL_NO_CLIPBOARD"):
        return None
    for cmd in (["wl-copy"], ["xsel", "--clipboard", "--input"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=5, stderr=subprocess.DEVNULL)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    return False


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else load_last_model()
    change = model_name is None  # sin modelo conocido se muestra el menú; si no, se va directo a grabar
    while True:
        if change:
            previous = model_name
            model_name = select_model(previous or "base")
            if model_name is None:
                if previous is None:
                    return
                model_name = previous  # canceló el cambio: sigue con el modelo actual
            elif not is_downloaded(model_name) and not download_model(model_name):
                time.sleep(2)  # deja leer el error antes de volver al menú
                model_name = previous if previous and is_downloaded(previous) else None
                change = True
                continue
        elif not is_downloaded(model_name) and not download_model(model_name):
            model_name = None  # el modelo pasado por argumento no se pudo descargar
            time.sleep(2)
            change = True
            continue
        save_last_model(model_name)
        action = wait_for_start(model_name)
        if action == "start":
            break
        if action == "quit":
            return
        change = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = OUT_DIR / f"grabacion_{stamp}.wav"

    duration, peak_db = record(wav_path, model_name)
    if duration == 0:  # se salió antes de que sonara algo
        wav_path.unlink(missing_ok=True)
        console.print("\n[yellow]No se grabó nada.[/yellow]")
        return
    console.print(f"\n[green]Grabación guardada:[/green] {wav_path}  ({duration:.0f} s)")
    if peak_db < SILENCE_DB:
        console.print("[yellow]Aviso:[/yellow] no se detectó audio. Revisa el volumen y que el sonido salga por el "
                      "dispositivo de salida predeterminado.")
    elif peak_db < LOW_DB:
        console.print(f"[yellow]Aviso:[/yellow] el volumen fue muy bajo (pico {peak_db:.0f} dB). Sube el volumen del "
                      "sistema o del reproductor: la transcripción puede salir peor.")
    segments, took = transcribe_with_retry(wav_path, model_name)
    if segments is None:
        console.print(f"[grey62]Para transcribirlo más tarde: ./.venv-whisper/bin/python src/transcribe.py "
                      f"{wav_path} {model_name}[/grey62]")
        return
    text = to_paragraphs(segments)
    txt_path = wav_path.with_suffix(".txt")
    txt_path.write_text(text + "\n", encoding="utf-8")
    console.print(f"\n[green]Transcrito en {took:.0f} s[/green] (audio de {duration:.0f} s)")
    if segments:
        console.print(f"[grey62]Texto guardado en {txt_path}[/grey62]")
    else:
        console.print(f"[yellow](sin texto reconocido)[/yellow]  [grey62]{txt_path}[/grey62]")
    copied = copy_to_clipboard(text) if text else None
    if copied:
        console.print("[green]Texto copiado al portapapeles[/green]")
    elif copied is False:
        console.print("[yellow]No se pudo copiar al portapapeles[/yellow] (instala wl-clipboard o xsel)")


if __name__ == "__main__":
    main()
