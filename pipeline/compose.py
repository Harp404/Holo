"""Three-panel result video: any video -> SAM 3D Body -> Meta Motivo robot, with Muse's read of the move and a report card.

python compose.py SRC.mp4 MESH_DIR ROBOT.mp4 INTENT.json REPORT.json OUT.mp4 [--cropx 200]
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
for k in ("src", "mesh", "robot", "intent", "report", "out"):
    ap.add_argument(k)
ap.add_argument("--cropx", type=int, default=200)
a = ap.parse_args()

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
for f in ("assets/fonts/Inter-Regular.ttf", "/usr/share/fonts/truetype/inter/Inter-Regular.ttf"):
    if Path(f).exists():
        FONT = f
F = lambda s: ImageFont.truetype(FONT, s)  # noqa: E731
intent = json.load(open(a.intent))
rep = json.load(open(a.report))

def frames(path):
    cap = cv2.VideoCapture(path); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f)

src = frames(a.src)
mesh = [cv2.imread(str(p)) for p in sorted(Path(a.mesh).glob("*.jpg"))]
rob = frames(a.robot)
n = min(len(mesh), len(rob))
PW, PH = 600, 720
W, H = PW * 3 + 40, PH + 200

def crop(img, x0):
    h, w = img.shape[:2]
    x0 = max(0, min(w - PW, x0))
    c = img[:, x0:x0 + PW]
    return cv2.resize(c, (PW, PH)) if c.shape[:2] != (PH, PW) else c

with tempfile.TemporaryDirectory() as d:
    for i in range(n):
        s = crop(src[min(i, len(src) - 1)], a.cropx)
        m = mesh[i]
        if m.shape[1] > 3 * m.shape[0]:            # raw 4-panel SAM 3D Body render: take the mesh overlay panel
            mw = m.shape[1] // 4; m = m[:, 2 * mw:3 * mw]
        mo = crop(m, a.cropx)
        r = rob[i]; r = crop(cv2.resize(r, (int(r.shape[1] * PH / r.shape[0]), PH)), (int(r.shape[1] * PH / r.shape[0]) - PW) // 2)
        canvas = np.full((H, W, 3), 14, np.uint8)
        for k, p in enumerate((s, mo, r)):
            canvas[70:70 + PH, 10 + k * (PW + 10):10 + k * (PW + 10) + PW] = p
        im = Image.fromarray(canvas[:, :, ::-1]); dr = ImageDraw.Draw(im)
        for k, (t1, t2) in enumerate((("1  Any video", "phone / YouTube, no mocap suit"),
                                      ("2  SAM 3D Body", "full 3D body, every frame"),
                                      ("3  Meta Motivo", "humanoid does it with physics, zero-shot"))):
            x = 10 + k * (PW + 10)
            dr.text((x + 4, 14), t1, font=F(26), fill=(240, 244, 250))
            dr.text((x + 4, 46), t2, font=F(16), fill=(150, 165, 185))
        y = 70 + PH + 16
        dr.text((14, y), "Muse Spark:", font=F(20), fill=(255, 170, 70))
        dr.text((150, y), f"{intent['action']}  —  {intent['intent']}", font=F(19), fill=(235, 238, 245))
        dr.text((14, y + 36), "Balance:", font=F(17), fill=(150, 165, 185))
        bal = intent.get("balance", "")
        dr.text((110, y + 36), bal[:150] + ("…" if len(bal) > 150 else ""), font=F(17), fill=(200, 206, 215))
        extra = f"kick height {rep['kick_reached_pct']:.0f}% of the person   \u00b7   " if rep.get("kick_height_target_m", 0) > 0.6 else ""
        card = (f"pose copied within {rep['pose_error_cm']:.0f} cm/joint   \u00b7   {extra}"
                f"{'kept its balance' if not rep['fell'] else 'fell'}   \u00b7   no training, one video")
        dr.text((14, y + 76), card, font=F(19), fill=(140, 220, 160))
        im.save(f"{d}/{i:05d}.jpg", quality=92)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", f"{d}/%05d.jpg", "-c:v", "libx264", "-crf", "19",
                    "-pix_fmt", "yuv420p", a.out], check=True)
print("wrote", a.out, n, "frames")
