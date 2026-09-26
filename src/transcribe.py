import sys
from faster_whisper import WhisperModel

if len(sys.argv) < 2:
    sys.exit("Uso: transcribe.py archivo.mp3 [modelo]")
model = WhisperModel(sys.argv[2] if len(sys.argv) > 2 else "base", device="cpu", compute_type="int8")
segments, info = model.transcribe(sys.argv[1], language="en")
print(" ".join(s.text.strip() for s in segments))
