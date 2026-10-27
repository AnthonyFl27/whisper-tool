#!/usr/bin/env python3
"""Prueba de humo de whisper-tool: graba frases sintéticas y comprueba la grabación y la transcripción.

Uso: ./.venv-whisper/bin/python src/smoke_test.py

Necesita `espeak-ng`, `pw-play` y el modelo `tiny` ya descargado. Las frases SUENAN por las bocinas (hay que dejar el
equipo con volumen y sin otros sonidos) y tarda alrededor de un minuto. Todo se guarda en un directorio temporal
(WHISPER_TOOL_DIR): no toca `records/`, el modelo recordado ni el portapapeles. Sale con 0 si todo pasa, 1 si algo
falla y 2 si faltan requisitos.
"""
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import grabar  # noqa: E402

PHRASE = "Hello, this is a test of the recording tool. Please transcribe this sentence."
EXPECTED = {"hello", "test", "recording", "tool", "transcribe", "sentence"}
MODEL = "tiny"
AUTO_PAUSE = 3  # s de silencio para la pausa automática (en uso normal son 12)
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def seconds(path):
    with wave.open(str(path)) as r:
        return r.getnframes() / r.getframerate()


def test_wav_writer(tmp):
    """El .wav queda legible aunque el programa muera sin cerrarlo (la cabecera se corrige cada segundo)."""
    path = tmp / "incremental.wav"
    w = grabar.WavWriter(path)
    w.write(b"\x01\x00" * grabar.RATE * 3)  # 3 s, sin llamar a close()
    with wave.open(str(path)) as r:
        check("wav incremental legible sin close()", r.getnframes() >= grabar.RATE * 2,
              f"{r.getnframes() / grabar.RATE:.1f} s de 3 s escritos")
    w.close()
    with wave.open(str(path)) as r:
        check("wav incremental completo tras close()", r.getnframes() == grabar.RATE * 3 and r.getframerate() == grabar.RATE)


def test_text_helpers():
    check("reloj MM:SS", grabar.clock(75.4) == "01:15" and grabar.clock(3725) == "62:05")
    seg = [(0, 2, "Hello."), (2.2, 4, "How are you?"), (7, 9, "New topic."), (9.1, 10, "Same paragraph.")]
    check("párrafos por pausa", grabar.to_paragraphs(seg) == "Hello. How are you?\n\nNew topic. Same paragraph.")
    long = [(i, i + 0.9, "Word " * 20 + "end.") for i in range(6)]  # sin pausas, pero muy largo
    check("párrafos por longitud", len(grabar.to_paragraphs(long).split("\n\n")) > 1)


def drain(fd, seconds_, until=None):
    """Lee la salida del pseudo-terminal durante `seconds_` o hasta que aparezca `until`."""
    out, end = b"", time.time() + seconds_
    while time.time() < end:
        if select.select([fd], [], [], 0.2)[0]:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
            if until and until in out:
                break
    return out


def run_tool(tmp, steps):
    """Ejecuta grabar.py en un pseudo-terminal siguiendo `steps`, y termina con Ctrl+C. Los pasos son
    ("expect", bytes), ("key", bytes), ("wait", segundos) y ("play", wav). Devuelve (estado, pantalla en texto).
    Hay que seguir leyendo la pantalla mientras suena el audio: si el buffer se llena, la herramienta se bloquea."""
    env = dict(os.environ, WHISPER_TOOL_DIR=str(tmp), WHISPER_TOOL_NO_CLIPBOARD="1", TERM="xterm-256color",
               COLUMNS="100", LINES="30", WHISPER_TOOL_AUTO_PAUSE=str(AUTO_PAUSE))
    py = sys.executable
    pid, fd = pty.fork()
    if pid == 0:
        os.execve(py, [py, str(SRC / "grabar.py"), MODEL], env)
    out = b""
    for kind, arg in steps:
        if kind == "expect":
            out += drain(fd, 60, arg)
        elif kind == "key":
            os.write(fd, arg)
        elif kind == "wait":
            out += drain(fd, arg)
        elif kind == "play":
            player = subprocess.Popen(["pw-play", str(arg)])
            while player.poll() is None:
                out += drain(fd, 0.2)
    os.write(fd, b"\x03")
    deadline, status = time.time() + 180, None
    while time.time() < deadline:
        out += drain(fd, 1)
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
    else:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
    return status, re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", out.decode(errors="ignore"))


def make_voice(tmp):
    voice = tmp / "voice.wav"
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", PHRASE, "-w", str(voice)], check=True)
    return voice


def test_session(tmp, voice):
    """Espera en silencio, graba una frase, se pausa sola, y sigue al volver el sonido con una segunda frase."""
    folder = tmp / "session"
    status, screen = run_tool(folder, [
        ("expect", b"Listo para grabar"), ("key", b"\r"), ("expect", b"ESPERANDO SONIDO"),
        ("wait", 4),  # silencio inicial: no debe grabarse
        ("play", voice),
        ("expect", b"PAUSA POR SILENCIO"),
        ("wait", 5),  # silencio en pausa: no debe grabarse
        ("play", voice),
        ("wait", 1.5),
    ])
    check("la herramienta termina bien", status == 0, f"estado {status}")
    check("espera el sonido para empezar", "ESPERANDO SONIDO" in screen)
    check("se pausa sola por silencio", "PAUSA POR SILENCIO" in screen)

    wavs = sorted((folder / "records").glob("grabacion_*.wav"))
    check("se creó un .wav", len(wavs) == 1)
    if not wavs:
        print(screen[-1500:])
        return
    wav = wavs[0]
    with wave.open(str(wav)) as r:
        samples = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32)
        check("audio de 16 kHz mono", (r.getframerate(), r.getnchannels()) == (16000, 1))
    rms = float(samples.std())
    # con la cabecera AU / big-endian mal leída el audio es ruido a volumen máximo (RMS ~15000)
    check("el audio no es ruido saturado", 50 < rms < 9000, f"RMS {rms:.0f}")
    # dos frases + el silencio hasta la pausa (3 s) + medio segundo previo + el final; sin la espera ni la pausa
    # habría 9 s más
    p, d = seconds(voice), seconds(wav)
    check("no graba la espera inicial ni la pausa", 2 * p < d < 2 * p + 8, f"{d:.1f} s grabados, frases de {p:.1f} s")

    txt_path = wav.with_suffix(".txt")
    txt = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
    words = set(re.findall(r"[a-z]+", txt.lower()))
    found = EXPECTED & words
    check("transcripción no vacía", bool(txt.strip()), repr(txt.strip()[:80]))
    check("la transcripción contiene la frase", len(found) >= 3, f"{len(found)}/{len(EXPECTED)} palabras: {sorted(found)}")
    check("el texto se guarda en párrafos", "\n\n" in txt.strip())
    check("se muestra cuánto tardó", "Transcrito en" in screen)
    check("no avisa de volumen bajo con audio normal", "volumen fue muy bajo" not in screen)
    if failures:
        print("\n--- salida de la herramienta (final) ---\n" + screen[-1500:])


def test_low_volume(tmp, voice):
    """Con audio muy bajo (pero audible) la herramienta avisa al terminar."""
    with wave.open(str(voice)) as r:
        params, data = r.getparams(), np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16)
    quiet = tmp / "quiet.wav"
    with wave.open(str(quiet), "wb") as w:
        w.setparams(params)
        w.writeframes((data * 0.03).astype(np.int16).tobytes())  # unos 30 dB menos
    status, screen = run_tool(tmp / "quiet-session", [
        ("expect", b"Listo para grabar"), ("key", b"\r"), ("expect", b"ESPERANDO SONIDO"),
        ("play", quiet), ("wait", 1.5),
    ])
    check("avisa del volumen bajo", status == 0 and "volumen fue muy bajo" in screen)
    if "volumen fue muy bajo" not in screen:
        print("\n--- salida de la herramienta (final) ---\n" + screen[-1500:])


def main():
    missing = [c for c in ("espeak-ng", "pw-play", "pw-record") if not shutil.which(c)]
    if missing:
        sys.exit(f"Faltan comandos: {', '.join(missing)}")
    if not grabar.is_downloaded(MODEL):
        sys.exit(f"Falta el modelo '{MODEL}': ejecuta ./whisper-tool {MODEL} una vez para descargarlo.")
    with tempfile.TemporaryDirectory(prefix="whisper-tool-test-") as d:
        tmp = Path(d)
        test_text_helpers()
        test_wav_writer(tmp)
        voice = make_voice(tmp)
        print("\nGrabando frases sintéticas (van a sonar por las bocinas)…")
        test_session(tmp, voice)
        test_low_volume(tmp, voice)
    print(f"\n{'TODO BIEN' if not failures else 'FALLÓ: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
