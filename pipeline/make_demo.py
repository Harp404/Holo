"""Demo video with voiceover: one video -> SAM 3D Body -> Meta Motivo humanoid, Muse Spark explains the move.

python make_demo.py OUT.mp4
Needs: out/holo_kick.mp4, out/overlay.mp4 (kick), out/holo_sq.mp4, out/overlay_sq.mp4, out/intent_*.json, out/report_*.json
"""
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
OUT = sys.argv[1]
W, H, FPS = 1920, 1080, 30
FD = ROOT / "assets" / "fonts"
F = lambda s, w="Regular": ImageFont.truetype(str(FD / f"Inter-{w}.ttf"), s)  # noqa: E731
VOICES = ROOT.parent / "holo" / "media" / "voice"
BG = (12, 14, 19)
INK, SUB, ACC, GOOD = (240, 244, 250), (150, 162, 182), (255, 170, 70), (130, 220, 160)

tmp = Path(tempfile.mkdtemp())
k = [0]
cues = []


def put(im):
    im.save(tmp / f"{k[0]:06d}.jpg", quality=92)
    k[0] += 1


def say(text, voice="ryan"):
    cues.append((k[0] / FPS, voice, text))


def slide(title, lines=(), sec=5.0, sub=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((140, 300 if lines else 420), title, font=F(76, "SemiBold"), fill=INK)
    if sub:
        d.text((144, (300 if lines else 420) + 104), sub, font=F(34, "Light"), fill=SUB)
    y = 470
    for a, b in lines:
        d.text((144, y), a, font=F(38, "SemiBold"), fill=INK)
        d.text((144, y + 50), b, font=F(28), fill=SUB)
        y += 130
    n = int(sec * FPS)
    for i in range(n):
        t = i / FPS
        al = min(1, t / 0.4, (sec - t) / 0.4)
        put(Image.blend(Image.new("RGB", (W, H), BG), im, max(0, al)))


def clip(path, caption, sub="", hold_end=0.6):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    fr += [fr[-1]] * int(hold_end * FPS)
    for f in fr:
        h, w = f.shape[:2]
        s = min(W / w, (H - 150) / h)
        f = cv2.resize(f, (int(w * s), int(h * s)))
        im = Image.new("RGB", (W, H), BG)
        im.paste(Image.fromarray(f[:, :, ::-1]), ((W - f.shape[1]) // 2, 120))
        d = ImageDraw.Draw(im)
        d.text((60, 34), caption, font=F(40, "SemiBold"), fill=INK)
        if sub:
            d.text((62, 84), sub, font=F(24), fill=SUB)
        put(im)


def muse_card(intent, sec=7):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((140, 170), "Muse Spark reads the move", font=F(60, "SemiBold"), fill=INK)
    d.text((144, 250), "meta-models/muse-spark-open-1026 on Baseten", font=F(26), fill=SUB)
    d.text((144, 340), intent["action"], font=F(40, "SemiBold"), fill=ACC)
    y = 410
    for lab, key in (("Goal", "intent"), ("Balance", "balance"), ("Hardest for a robot", "hard_parts")):
        txt = intent.get(key, "")
        words, line, lines = txt.split(), "", []
        for wd in words:
            if F(28).getlength(line + " " + wd) > 1500:
                lines.append(line); line = wd
            else:
                line = (line + " " + wd).strip()
        lines.append(line)
        d.text((144, y), lab, font=F(26, "SemiBold"), fill=SUB)
        for ln in lines[:3]:
            y += 40
            d.text((144, y), ln, font=F(28), fill=INK)
        y += 70
    ed = intent.get("edits", [])[:3]
    if ed:
        d.text((144, y), "Edits it suggests", font=F(26, "SemiBold"), fill=SUB)
        d.text((144, y + 40), "  ·  ".join(ed)[:140], font=F(26), fill=INK)
    for _ in range(int(sec * FPS)):
        put(im)


kick_i = json.load(open(ROOT / "out/intent_kick.json"))
sq_i = json.load(open(ROOT / "out/intent_sq.json"))
kick_r = json.load(open(ROOT / "out/report_kick.json"))
sq_r = json.load(open(ROOT / "out/report_sq.json"))

say("This is Holo. Show a humanoid robot one ordinary video of a person, and it does the move, with physics, and no training.")
slide("Holo", sec=6.5, sub="teach a humanoid from one video  ·  SAM 3D Body + Meta Motivo + Muse Spark")
say("Robots normally learn a skill from hours of reinforcement learning or a motion capture suit. Holo uses one clip from a phone or the web.")
slide("One video in. A robot skill out.", [
    ("No mocap suit", "any phone or web video of a person"),
    ("No training per skill", "Meta Motivo, a behavioral foundation model, performs it zero-shot"),
    ("It understands the move", "Muse Spark explains what the person is doing and why"),
], sec=8)
say("A taekwondo side kick. SAM 3D Body rebuilds the full 3D body in every frame. Meta Motivo then performs it in a physics simulator, balancing on one leg.")
clip(ROOT / "out/holo_kick.mp4", "1  Video   →   2  SAM 3D Body   →   3  Meta Motivo (physics, zero-shot)")
say("Same camera, robot drawn over the person. The stance and the kick line up. On this fast kick the robot turns, which is a known limit we report honestly.")
clip(ROOT / "out/overlay.mp4", "Same camera: robot ghosted over the person", "kick height reached %d%%  ·  posture within %d cm per joint  ·  never fell"
     % (kick_r["kick_reached_pct"], round(kick_r["pose_error_cm"])))
say("Muse Spark reads the move before the robot does: the technique, the goal, where the weight is, and what will be hardest.", "lessac")
muse_card(kick_i, 8)
say("A second person, a different move: squats. Same pipeline, nothing changed.")
clip(ROOT / "out/holo_sq.mp4", "Squats: same pipeline, no changes")
clip(ROOT / "out/overlay_sq.mp4", "Same camera: robot over the person", "posture within %d cm per joint  ·  %s"
     % (round(sq_r["pose_error_cm"]), "never fell" if not sq_r["fell"] else "fell"))
muse_card(sq_i, 6)
say("Under the hood: SAM 3D Body for the body, inverse kinematics onto the humanoid, Meta Motivo for zero-shot physics control, and Muse Spark Open for understanding. All open Meta models.")
slide("How it works", [
    ("SAM 3D Body", "3D human body per frame from a single camera (MHR keypoints)"),
    ("Retarget", "person's bone directions on the humanoid's bone lengths, inverse kinematics in MuJoCo"),
    ("Meta Motivo", "tracking prompt → physics-based humanoid control, zero-shot"),
    ("Muse Spark Open", "names the move, its goal, balance and edits (Baseten)"),
], sec=11)
say("Holo. Show it once.")
slide("Holo", sec=4, sub="show it once  ·  github.com/Harp404/holo")

# ---- voice
wav = tmp / "voice.wav"
total = k[0] / FPS + 1
sr = 22050
mix = np.zeros(int(total * sr) + sr * 3, np.float32)
end = 0
for t, v, text in cues:
    p = tmp / f"c{len(str(t))}_{int(t * 100)}.wav"
    subprocess.run(["piper", "-m", str(VOICES / f"en_US-{v}-high.onnx"), "-f", str(p), "--length-scale", "0.95"],
                   input=text.encode(), check=True, capture_output=True)
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        sr = w.getframerate()
    s0 = int(max(t + 0.3, end) * sr)
    mix[s0:s0 + len(x)] += x[: max(0, len(mix) - s0)]
    end = s0 / sr + len(x) / sr + 0.2
    if end > total:
        print("WARN voice runs past video at", round(t, 1))
mix = np.clip(mix[: int(total * sr)], -1, 1)
with wave.open(str(wav), "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((mix * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "%06d.jpg"), "-i", str(wav),
                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", OUT], check=True)
print("wrote", OUT, round(k[0] / FPS, 1), "s")
