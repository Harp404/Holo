"""Render the hologram preview from refined masks: translucent bluish tint + crisp outline on objects,
soft blue hands. Usage: .venv/bin/python render_holo.py --frames F --held OUT/pov_held --out preview.mp4"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TINT = np.array([255, 170, 70], np.float32)      # BGR bluish
RIM = np.array([255, 225, 150], np.float32)      # BGR light cyan-blue outline
HAND = np.array([255, 185, 95], np.float32)


def object_tint(img, m, t):
    m = m.astype(np.uint8)
    if not m.any():
        return img
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
    pad = cv2.copyMakeBorder(m, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    dist = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[8:-8, 8:-8]
    edge_glow = np.exp(-dist / 4.0) * m                      # slightly brighter just inside the edge
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # BORDER_REPLICATE: an object leaving the screen gets no fake outline along the frame edge
    rim = cv2.GaussianBlur((m - cv2.erode(m, k, borderType=cv2.BORDER_REPLICATE)).astype(np.float32), (0, 0), 0.7)
    rim[:3, :] = rim[-3:, :] = 0
    rim[:, :3] = rim[:, -3:] = 0
    halo = cv2.GaussianBlur(rim, (0, 0), 4) * (1 - m)
    pulse = 0.9 + 0.1 * np.sin(t * 3.0)
    a = (0.32 * soft + 0.18 * edge_glow) * pulse             # translucent: object stays visible
    out = img.astype(np.float32)
    out = out * (1 - a[..., None]) + TINT[None, None] * a[..., None]
    out = out * (1 - rim[..., None]) + RIM[None, None] * rim[..., None]
    out += halo[..., None] * RIM[None, None] * 0.35
    return np.clip(out, 0, 255).astype(np.uint8)


def hand_skin(img, m):
    m = m.astype(np.uint8)
    if not m.any():
        return img
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)
    rim = cv2.GaussianBlur((m - cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))).astype(np.float32), (0, 0), 1.2)
    out = img.astype(np.float32) * (1 - 0.40 * soft[..., None]) + HAND[None, None] * (0.40 * soft[..., None])
    out += rim[..., None] * HAND[None, None] * 0.7
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-hands", action="store_true", help="frames already contain 3D hands")
    ap.add_argument("--skip", default="", help="object ids drawn elsewhere (e.g. as 3D)")
    a = ap.parse_args()
    held = Path(a.held)
    meta = json.loads((held / "objects_refined.json").read_text())
    frames = sorted(Path(a.frames).glob("*.jpg"))
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"]))
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        skip = {int(x) for x in a.skip.split(",") if x}
        for o in meta["objects"]:
            if o["id"] in skip:
                continue
            p = held / "refined" / f"object{o['id']}" / f"{i + 1:04d}.png"
            if p.exists():
                img = object_tint(img, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127, i / meta["fps"])
        hp = held / "hands" / f"{i + 1:04d}.png"
        if hp.exists() and not a.no_hands:
            img = hand_skin(img, cv2.imread(str(hp), cv2.IMREAD_GRAYSCALE) > 127)
        vw.write(img)
    vw.release()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
