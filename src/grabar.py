#!/usr/bin/env python3
"""Graba el audio del sistema con un espectrograma en vivo y lo transcribe con faster-whisper.

Uso: whisper-tool [modelo]      (sin modelo, muestra un menú: tiny, base, small, medium, large-v3)
Iniciar la grabacion con Enter o Espacio; detenerla con Ctrl+C.
"""
import json
import os
import select
import subprocess
import sys
import termios
import time
import tty
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.text import Text

RATE = 16000
CHUNK = 1600  # 100 ms por cuadro
FFT_SIZE = 2048
HEIGHT = 8  # filas de las barras
WATERFALL_ROWS = 10
BLOCKS = " ▁▂▃▄▅▆▇█"
SILENCE_DB = -55.0  # por debajo de esto se considera sin senal
DB_FLOOR, DB_CEIL = -80.0, -25.0

ROOT = Path(__file__).resolve().parent.parent  # raiz del proyecto (src/ esta un nivel abajo)
OUT_DIR = ROOT / "records"
CONFIG_PATH = ROOT / ".whisper-tool.json"  # recuerda el último modelo usado
console = Console()


def band_edges(n_bands):
    """Bordes de banda espaciados en escala logaritmica (80 Hz - 7.5 kHz)."""
    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / RATE)
    edges = np.geomspace(80, 7500, n_bands + 1)
    return [(np.searchsorted(freqs, lo), max(np.searchsorted(freqs, hi), np.searchsorted(freqs, lo) + 1))
            for lo, hi in zip(edges[:-1], edges[1:])]


def analyze(samples, edges):
    """Devuelve (nivel_dB, valores 0..1 por banda)."""
    x = samples.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    level_db = 20 * np.log10(rms) if rms > 1e-9 else -120.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=FFT_SIZE)) / len(x)
    bands = np.array([spec[a:b].mean() for a, b in edges])
    db = 20 * np.log10(np.maximum(bands, 1e-9))
    return level_db, np.clip((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0, 1)


def heat(v):
    """Color de 0..1: azul oscuro -> cian -> amarillo -> rojo."""
    stops = [(0.0, (15, 20, 60)), (0.35, (30, 140, 200)), (0.65, (240, 210, 60)), (1.0, (240, 60, 50))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))
    return stops[-1][1]


def bars(values):
    text = Text()
    for row in range(HEIGHT, 0, -1):
        for v in values:
            fill = v * HEIGHT - (row - 1)
            ch = BLOCKS[-1] if fill >= 1 else BLOCKS[int(max(fill, 0) * 8)]
            r, g, b = heat(v)
            text.append(ch * 2, style=f"rgb({r},{g},{b})")
        text.append("\n")
    text.rstrip()
    return text


def waterfall(history):
    text = Text()
    for i in range(WATERFALL_ROWS):
        if i < len(history):
            for v in history[i]:
                r, g, b = heat(v)
                text.append("██", style=f"rgb({r},{g},{b})")
        else:
            text.append("██" * (len(history[0]) if history else 0), style="rgb(15,20,40)")
        text.append("\n")
    text.rstrip()
    return text


def meter(level_db, width=40):
    frac = min(max((level_db - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0), 1)
    filled = int(frac * width)
    text = Text()
    for i in range(width):
        r, g, b = heat(i / width)
        text.append("█" if i < filled else "░", style=f"rgb({r},{g},{b})" if i < filled else "grey30")
    text.append(f"  {level_db:6.1f} dB")
    return text


def screen(body):
    """Pantalla de la interfaz: título centrado y contenido, sin marco."""
    return Padding(Align.center(Group(Text("Grabador de audio del sistema", style="bold"), Text(""), body)),
                   (1, 2))


def render(elapsed, level_db, values, history, silent_for, peak_db, n_bands, paused=False):
    blink = int(time.monotonic() * 2) % 2 == 0
    if paused:
        status = Text("⏸ EN PAUSA" if blink else "  EN PAUSA", style="bold cyan")
    elif silent_for > 2.0:
        status = Text("● SIN SEÑAL DE AUDIO", style="bold yellow")
    else:
        status = Text("● GRABANDO" if blink else "○ GRABANDO", style="bold red")
    mins, secs = divmod(int(elapsed), 60)
    head = Text.assemble(status, ("   " + f"{mins:02d}:{secs:02d}", "bold white"), ("   pico ", "grey62"),
                         (f"{peak_db:.1f} dB", "grey85"))
    labels = Text(f"80 Hz{' ' * (n_bands * 2 - 15)}7.5 kHz", style="grey50")
    body = Group(
        head, Text(""), meter(level_db), Text(""),
        bars(values), labels, Text(""),
        Text("Espectrograma (tiempo hacia abajo)", style="grey62"),
        waterfall(history),
        Text(""), Text("Espacio / P: pausar o continuar    Ctrl+C: terminar y transcribir", style="grey50"),
    )
    return screen(body)


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

    def view():
        body = Text()
        body.append("Elige el modelo de Whisper\n\n", style="bold white")
        for i, (name, size, desc) in enumerate(MODELS):
            style = "bold cyan" if i == index else "grey70"
            body.append(f"{'▶' if i == index else ' '} {i + 1}. {name:<9}{size:<9} {desc:<34}", style=style)
            body.append("✔ descargado\n" if downloaded[name] else "\n", style="green")
        body.append("\nLos modelos sin marcar se descargan al elegirlos.\n", style="grey50")
        body.append("↑/↓ o 1-5: elegir    Enter: continuar    Q / Ctrl+C: salir", style="grey62")
        return screen(body)

    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(view(), console=console, screen=True) as live:
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
                live.update(view())
    except KeyboardInterrupt:
        return None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)


def wait_for_start(model_name):
    """Espera Enter o Espacio para iniciar. Devuelve "start", "change" (M: cambiar de modelo) o "quit"."""
    if not sys.stdin.isatty():
        return "start"  # sin teclado no hay forma de esperar: graba de inmediato
    old_term = termios.tcgetattr(sys.stdin)
    body = Group(
        Text("Listo para grabar", style="bold white"), Text(""),
        Text.assemble(("Modelo: ", "grey62"), (model_name, "bold cyan")), Text(""),
        Text("Enter / Espacio: iniciar grabación    M: cambiar modelo    Q / Ctrl+C: salir", style="grey62"),
    )
    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(screen(body), console=console, screen=True):
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


def record(wav_path, n_bands):
    edges = band_edges(n_bands)
    history = deque(maxlen=WATERFALL_ROWS)
    proc = subprocess.Popen(
        ["pw-record", "-P", "stream.capture.sink=true", "--rate", str(RATE), "--channels", "1",
         "--format", "s16", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    fd = proc.stdout.fileno()
    use_keys = sys.stdin.isatty()
    old_term = termios.tcgetattr(sys.stdin) if use_keys else None
    peak_db = -120.0
    values = np.zeros(n_bands)
    level_db = -120.0
    frames = bytearray()
    carry = b""
    paused = False
    last_sound = time.monotonic()
    try:
        if use_keys:
            tty.setcbreak(sys.stdin.fileno())  # Ctrl+C sigue funcionando
        with Live(render(0, level_db, values, history, 0, peak_db, n_bands), console=console,
                  refresh_per_second=15, screen=True) as live:
            while True:
                watch = [fd] + ([sys.stdin] if use_keys else [])
                ready, _, _ = select.select(watch, [], [], 0.1)
                if use_keys and sys.stdin in ready:
                    key = os.read(sys.stdin.fileno(), 32).decode(errors="ignore")
                    if any(k in key for k in (" ", "p", "P")):
                        paused = not paused
                        if paused:
                            values = np.zeros(n_bands)
                            level_db = -120.0
                        last_sound = time.monotonic()
                if fd in ready:
                    data = os.read(fd, CHUNK * 2)
                    if not data:
                        break
                    if not paused:  # en pausa el audio se lee y se descarta
                        data = carry + data
                        carry = data[len(data) // 2 * 2:]
                        data = data[: len(data) // 2 * 2]
                        frames += data
                        samples = np.frombuffer(data, dtype=np.int16)
                        if len(samples):
                            level_db, values = analyze(samples, edges)
                            if level_db > SILENCE_DB:
                                last_sound = time.monotonic()
                            peak_db = max(peak_db, level_db)
                            history.appendleft(values)
                if paused:
                    last_sound = time.monotonic()
                live.update(render(len(frames) / 2 / RATE, level_db, values, history,
                                   time.monotonic() - last_sound, peak_db, n_bands, paused))
    except KeyboardInterrupt:
        pass
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        proc.terminate()
        proc.wait()
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))
    return len(frames) / 2 / RATE, peak_db


def transcribe(wav_path, model_name):
    from faster_whisper import WhisperModel

    with console.status("[bold cyan]Cargando modelo y transcribiendo…[/bold cyan]", spinner="dots"):
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(wav_path), language="en")
        return " ".join(s.text.strip() for s in segments)


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
    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = OUT_DIR / f"grabacion_{stamp}.wav"
    n_bands = max(16, min(40, (console.width - 12) // 2))

    duration, peak_db = record(wav_path, n_bands)
    console.print(f"\n[green]Grabación guardada:[/green] {wav_path}  ({duration:.0f} s)")
    if peak_db < SILENCE_DB:
        console.print("[yellow]Aviso:[/yellow] no se detectó audio. Revisa el volumen y que el sonido salga por el "
                      "dispositivo de salida predeterminado.")
    text = transcribe(wav_path, model_name)
    txt_path = wav_path.with_suffix(".txt")
    txt_path.write_text(text + "\n", encoding="utf-8")
    console.print(Panel(text or "(sin texto reconocido)", title="Transcripción", border_style="green",
                        padding=(1, 2)))
    console.print(f"[grey62]Texto guardado en {txt_path}[/grey62]")


if __name__ == "__main__":
    main()
