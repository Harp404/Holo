"""Voice track for the demo: Piper TTS per cue (narrator/user = ryan, Muse = lessac), placed at cue times.

Usage: python build_audio.py --cues out/demo_cues.json --voices media/voice --out out/demo_audio.wav
"""
import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

VOICE = {"narr": "en_US-ryan-high", "user": "en_US-ryan-high", "muse": "en_US-lessac-high"}
SPEED = {"narr": 0.95, "user": 0.95, "muse": 0.92}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cues", required=True)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = json.loads(Path(a.cues).read_text())
    total = c["frames"] / c["fps"] + 1
    tmp = Path(a.out).with_suffix("")
    tmp.mkdir(parents=True, exist_ok=True)
    sr = None
    mix = None
    prev_end = 0.0
    for i, (t, v, text) in enumerate(c["cues"]):
        wav = tmp / f"{i:02d}_{v}.wav"
        if v == "wav":                      # recorded app speech (already a wav)
            with wave.open(text) as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
                if w.getnchannels() == 2: x = x.reshape(-1, 2).mean(1)
                r = w.getframerate()
            sr = sr or 22050
            if r != sr: x = np.interp(np.arange(0, len(x), r / sr), np.arange(len(x)), x).astype(np.float32)
            if mix is None: mix = np.zeros(int(total * sr) + sr * 5, np.float32)
            s0 = int(t * sr); mix[s0:s0 + len(x)] += x[: max(0, len(mix) - s0)]
            print(f"{t:7.2f}s +{len(x)/sr:5.2f}s app   {Path(text).name}")
            prev_end = t + len(x) / sr
            continue
        import re as _re
        text = _re.sub(r"\[\d+\]", "", text).replace("USB", "U.S.B.").replace("HP ", "H.P. ")
        subprocess.run(["piper", "-m", str(Path(a.voices) / f"{VOICE[v]}.onnx"), "-f", str(wav), "--length-scale", str(SPEED[v])],
                       input=text.encode(), check=True, capture_output=True)
        with wave.open(str(wav)) as w:
            sr = sr or w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
            if w.getframerate() != sr:
                raise SystemExit("sample rate mismatch")
        if mix is None:
            mix = np.zeros(int(total * sr) + sr * 5, np.float32)
        dur = len(x) / sr
        flag = "  OVERLAP %.2fs" % (prev_end - t) if t < prev_end else ""
        print(f"{t:7.2f}s +{dur:5.2f}s {v:5s} {text[:60]}{flag}")
        s0 = int(t * sr)
        mix[s0:s0 + len(x)] += x * (0.95 if v != "muse" else 1.0)
        prev_end = t + dur
    mix = np.clip(mix[: int(total * sr)], -1, 1)
    with wave.open(a.out, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    print("wrote", a.out, round(len(mix) / sr, 1), "s")


if __name__ == "__main__":
    main()
