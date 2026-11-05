"""Demo video with voiceover. Every segment lasts as long as its narration (no overlaps, no cut-off lines), with fades.

python make_demo.py OUT.mp4
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
O = ROOT / "out"
OUT = sys.argv[1]
W, H, FPS = 1920, 1080, 30
FD = ROOT / "assets" / "fonts"
F = lambda s, w="Regular": ImageFont.truetype(str(FD / f"Inter-{w}.ttf"), s)  # noqa: E731
VOICES = ROOT.parent / "holo" / "media" / "voice"
BG = (12, 14, 19)
INK, SUB, ACC, BLUE = (240, 244, 250), (150, 162, 182), (255, 170, 70), (150, 200, 255)
tmp = Path(tempfile.mkdtemp())


def tts(text, voice="ryan"):
    p = tmp / f"v{abs(hash(text))}.wav"
    subprocess.run(["piper", "-m", str(VOICES / f"en_US-{voice}-high.onnx"), "-f", str(p), "--length-scale", "0.97"],
                   input=text.encode(), check=True, capture_output=True)
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return x, w.getframerate()


def wrap(text, font, width):
    out, line = [], ""
    for wd in text.split():
        if font.getlength((line + " " + wd).strip()) > width:
            out.append(line); line = wd
        else:
            line = (line + " " + wd).strip()
    return out + [line]


def slide_frames(title, lines=(), sub=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y0 = 300 if lines else 430
    d.text((140, y0), title, font=F(76, "SemiBold"), fill=INK)
    if sub:
        d.text((144, y0 + 104), sub, font=F(34, "Light"), fill=SUB)
    y = 480
    for a, b in lines:
        d.text((144, y), a, font=F(38, "SemiBold"), fill=INK)
        d.text((144, y + 50), b, font=F(28), fill=SUB)
        y += 128
    return [np.array(im)]


def video_frames(path, caption, sub=""):
    cap = cv2.VideoCapture(str(path)); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = min(W / w, (H - 140) / h)
        f = cv2.resize(f, (int(w * s), int(h * s)))
        im = Image.new("RGB", (W, H), BG)
        im.paste(Image.fromarray(f[:, :, ::-1]), ((W - f.shape[1]) // 2, 120))
        d = ImageDraw.Draw(im)
        d.text((60, 30), caption, font=F(40, "SemiBold"), fill=INK)
        if sub:
            d.text((62, 82), sub, font=F(24), fill=SUB)
        fr.append(np.array(im))
    return fr


def muse_card(intent):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((140, 150), "Muse Spark reads the move", font=F(60, "SemiBold"), fill=INK)
    d.text((144, 230), "meta-models/muse-spark-open-1026  ·  Baseten", font=F(26), fill=SUB)
    d.text((144, 320), intent["action"], font=F(40, "SemiBold"), fill=ACC)
    y = 400
    for lab, key in (("Goal", "intent"), ("Balance", "balance"), ("Hardest for a robot", "hard_parts")):
        d.text((144, y), lab, font=F(26, "SemiBold"), fill=SUB)
        for ln in wrap(intent.get(key, ""), F(28), 1560)[:3]:
            y += 40; d.text((144, y), ln, font=F(28), fill=INK)
        y += 68
    return [np.array(im)]


def plan_card(plan):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((140, 140), "You say:", font=F(34), fill=SUB)
    for i, ln in enumerate(wrap("“" + plan["request"] + "”", F(46, "SemiBold"), 1600)[:2]):
        d.text((140, 190 + 60 * i), ln, font=F(46, "SemiBold"), fill=INK)
    d.text((140, 350), "Muse Spark plans it (Motivo reward prompts):", font=F(30), fill=SUB)
    y = 410
    for k, s in enumerate(plan["steps"][:9]):
        d.text((160, y), f"{k + 1}.", font=F(30, "SemiBold"), fill=ACC)
        d.text((220, y), s["say"], font=F(30), fill=INK)
        d.text((900, y), s["task"] + f"   {s['seconds']} s", font=F(26), fill=BLUE)
        y += 52
    if plan.get("note"):
        for i, ln in enumerate(wrap("Muse: " + plan["note"], F(26), 1600)[:2]):
            d.text((140, y + 20 + 36 * i), ln, font=F(26), fill=SUB)
    return [np.array(im)]


kick_i, sq_i = json.load(open(O / "intent_kick.json")), json.load(open(O / "intent_sq.json"))
kick_r, sq_r = json.load(open(O / "report_kick.json")), json.load(open(O / "report_sq.json"))
plan = json.load(open(O / "plan3.json"))

SEG = [
    ("This is Holo. Show a humanoid robot one ordinary video of a person, and it does the move, with physics, and no training. Or just tell it what to do.",
     slide_frames("Holo", sub="teach a humanoid from one video  ·  or just tell it  ·  Meta SAM 3D Body + Motivo + Muse Spark")),
    ("Robots usually learn each skill from hours of reinforcement learning or a motion capture suit. Holo needs one clip from a phone or the web.",
     slide_frames("One video in. A robot skill out.", [
         ("No mocap suit", "any phone or web video of a person"),
         ("No training per skill", "Meta Motivo, a behavioral foundation model, performs it zero-shot"),
         ("It understands the move", "Muse Spark explains what the person is doing and why")])),
    ("A taekwondo side kick. SAM 3D Body rebuilds the full 3D body in every frame, and Meta Motivo performs it in physics, balancing on one leg.",
     video_frames(O / "holo_kick.mp4", "1  Video   →   2  SAM 3D Body   →   3  Meta Motivo (physics, zero-shot)")),
    ("Here is the honest check: the robot drawn over the person from the original camera. The kick reaches full height, but on this fast spin the robot turns away. We report that.",
     video_frames(O / "overlay.mp4", "Same camera: robot drawn over the person",
                  f"kick height {kick_r['kick_reached_pct']:.0f}% of the person  ·  posture within {kick_r['pose_error_cm']:.0f} cm per joint  ·  never fell  ·  turns on fast spins")),
    ("Muse Spark reads the move first: the technique, the goal, where the weight is, and what will be hardest for a robot.",
     muse_card(kick_i)),
    ("A different person, a different move: squats. Same pipeline, nothing changed.",
     video_frames(O / "holo_sq.mp4", "Squats: same pipeline, no changes")),
    (f"From the original camera, the robot stays on her, facing the same way. Posture within {sq_r['pose_error_cm']:.0f} centimeters per joint.",
     video_frames(O / "overlay_sq.mp4", "Same camera: robot drawn over the person",
                  f"posture within {sq_r['pose_error_cm']:.1f} cm per joint  ·  turn {sq_r['turn_error_deg_mean']:.0f}°  ·  never fell")),
    ("And you can just talk to it. Muse Spark turns your sentence into a plan of behaviours, and works out moves it doesn't have, like a wave, by itself.",
     plan_card(plan)),
    ("Meta Motivo performs the plan zero-shot, from reward prompts. Arms up, a real jump, a wave, and sitting down, all with physics.",
     video_frames(O / "talk3.mp4", "Muse Spark → Meta Motivo: talk to the humanoid", plan["request"])),
    ("Under the hood: SAM 3D Body for the body, retargeting with inverse kinematics, Meta Motivo for zero-shot physics control, and Muse Spark Open to understand and plan. All Meta models.",
     slide_frames("How it works", [
         ("SAM 3D Body", "3D human body per frame from one camera (MHR keypoints)"),
         ("Retarget", "the person's bone directions on the humanoid's bone lengths, IK in MuJoCo"),
         ("Meta Motivo", "tracking prompts (from video) and reward prompts (from Muse), zero-shot"),
         ("Muse Spark Open", "reads the move, plans behaviours from your words (Baseten)")])),
    ("Holo. Show it once, or just ask.",
     slide_frames("Holo", sub="show it once, or just ask  ·  github.com/Harp404/holo")),
]

frames_out = 0
audio = []
sr = 22050
writer = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(tmp / "v.mp4")], stdin=subprocess.PIPE)
for text, fr in SEG:
    x, sr = tts(text, "lessac" if text.startswith(("Muse Spark reads", "And you can")) else "ryan")
    need = int((len(x) / sr + 0.9) * FPS)
    n = max(len(fr), need)
    seq = fr + [fr[-1]] * (n - len(fr)) if len(fr) > 1 else fr * n
    fade = 9
    for i, f in enumerate(seq):
        a = min(1.0, (i + 1) / fade, (n - i) / fade)
        writer.stdin.write((f.astype(np.float32) * a + np.array(BG, np.float32) * (1 - a)).astype(np.uint8).tobytes())
    seg_audio = np.zeros(int(n / FPS * sr), np.float32)
    s0 = int(0.35 * sr)
    seg_audio[s0:s0 + len(x)] = x[: len(seg_audio) - s0]
    audio.append(seg_audio)
    frames_out += n
writer.stdin.close(); writer.wait()
mix = np.concatenate(audio)
with wave.open(str(tmp / "a.wav"), "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp / "v.mp4"), "-i", str(tmp / "a.wav"), "-c:v", "copy", "-c:a", "aac",
                "-b:a", "160k", "-shortest", OUT], check=True)
print("wrote", OUT, round(frames_out / FPS, 1), "s")
