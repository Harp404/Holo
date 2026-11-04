"""Render a perception preview video: SAM 3 masks (soft fill + crisp outline + name) and hand joints.

Usage: python render_preview.py --frames FRAMES --scene OUT/scene.json --out preview.mp4 [--only "usb cable,smartphone"]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PALETTE = [(255, 170, 60), (90, 220, 255), (120, 255, 150), (255, 120, 200), (80, 160, 255), (255, 230, 90),
           (200, 140, 255), (120, 255, 230), (255, 140, 120), (170, 255, 90)]  # BGR
HAND = (255, 190, 90)  # light blue in BGR
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    scene = json.loads(Path(a.scene).read_text())
    mask_dir = Path(a.scene).parent / "masks"
    frames = sorted(Path(a.frames).glob("*.jpg"))
    names = {o["index"]: o["name"] for o in scene["objects"]}
    labels = {o["index"]: o["label"] for o in scene["objects"]}
    only = {s.strip() for s in a.only.split(",") if s.strip()}

    h, w = cv2.imread(str(frames[0])).shape[:2]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), scene["fps"], (w, h))
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        lab = cv2.imread(str(mask_dir / f"{i + 1:04d}.png"), cv2.IMREAD_UNCHANGED)
        if lab is not None:
            lab = cv2.resize(lab[..., 2] if lab.ndim == 3 else lab, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = img.copy()
            for idx in np.unique(lab):
                if idx == 0 or (only and labels.get(int(idx)) not in only):
                    continue
                m = (lab == idx).astype(np.uint8)
                is_hand = labels.get(int(idx)) == "hand"
                col = HAND if is_hand else PALETTE[(int(idx) - 1) % len(PALETTE)]
                overlay[m > 0] = col
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, col, 2, cv2.LINE_AA)
                if not is_hand:
                    ys, xs = np.where(m)
                    x, y = int(xs.mean()), int(ys.min()) - 10
                    txt = names[int(idx)]
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(img, (x - tw // 2 - 8, y - th - 8), (x + tw // 2 + 8, y + 6), (30, 30, 30), -1)
                    cv2.putText(img, txt, (x - tw // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
        fr = scene["frames"][i] if i < len(scene["frames"]) else {}
        for hand in fr.get("hands", []):
            pts = [(int(p[0] * w), int(p[1] * h)) for p in hand["kp"]]
            for a_, b_ in BONES:
                cv2.line(img, pts[a_], pts[b_], HAND, 3, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 4, (255, 245, 230), -1, cv2.LINE_AA)
        for c in fr.get("contacts", []):
            cv2.putText(img, f"{c['type']}: {names.get(c['object'], '?')}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 150), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
