"""Word timing for lyrics, for text "times". Whisper mishears sung words, so match its timings against the editor's lyrics rather than trusting its text, and have the editor check the words land on the voice. Needs setup.sh --whisper; its model is a large download on first use.
usage: lyrics_align.py SONG OUT.json [--from SONG_SECONDS --to SONG_SECONDS]"""
import argparse
import json
import subprocess
import tempfile

import mlx_whisper

ap = argparse.ArgumentParser(); ap.add_argument("song"); ap.add_argument("out"); ap.add_argument("--from", dest="a", type=float, default=0); ap.add_argument("--to", dest="b", type=float)
x = ap.parse_args()
with tempfile.NamedTemporaryFile(suffix=".wav") as t:
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(x.a), "-i", x.song] + (["-t", str(x.b - x.a)] if x.b else []) + ["-vn", "-ac", "1", "-ar", "16000", t.name]
    subprocess.run(cmd, check=True)
    r = mlx_whisper.transcribe(t.name, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="en", word_timestamps=True, condition_on_previous_text=False, no_speech_threshold=None)
words = [dict(word=w["word"].strip(), start=round(w["start"] + x.a, 3), end=round(w["end"] + x.a, 3)) for s in r["segments"] for w in s.get("words", [])]
json.dump(words, open(x.out, "w"), indent=0)
print(len(words), "words written to", x.out)
