"""Egocentric hand-object roles per frame with RoleHOS (DINOv3 ConvNeXt backbone).

Labels: 0 BG, 1 left hand, 2 right hand, 3 left-hand object, 4 right-hand object, 5 two-hand object.
Writes roles/0001.png ... (label maps at frame size) and roles.json (per-frame hand / held-object boxes),
plus a preview video that shows ONLY the hands and what they hold.

Run with the RoleHOS venv from inside the RoleHOS repo:
  cd /root/holo/rolehos && .venv/bin/python /root/holo/perception/run_rolehos.py --frames F --out O
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from rolehos.runtime import load_model, predict

ROLES = {1: "left hand", 2: "right hand", 3: "left-hand object", 4: "right-hand object", 5: "two-hand object"}


def box(m, w, h):
    ys, xs = np.where(m)
    return [round(xs.min() / w * 1000), round(ys.min() / h * 1000), round(xs.max() / w * 1000), round(ys.max() / h * 1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area", type=float, default=0.0015)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "roles").mkdir(parents=True, exist_ok=True)
    model = load_model("full", a.seed, "cuda")[0].eval()
    frames = sorted(Path(a.frames).glob("*.jpg"))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    vw = cv2.VideoWriter(str(out / "roles_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    meta = {"fps": a.fps, "width": w, "height": h, "roles": ROLES, "frames": []}

    for i, fp in enumerate(frames):
        rgb = np.asarray(Image.open(fp).convert("RGB"))
        with torch.inference_mode():
            m448 = predict(model, rgb)
        lab = cv2.resize(m448.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(lab).convert("RGB").save(out / "roles" / f"{i + 1:04d}.png")

        rec = {}
        for k, name in ROLES.items():
            m = lab == k
            if m.mean() >= a.min_area:
                rec[name] = {"box": box(m, w, h), "area": round(float(m.mean()), 4)}
        meta["frames"].append(rec)

        # preview: hands soft blue, held objects warm orange with outline; nothing else
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        over = img.copy()
        for k in (1, 2):
            over[lab == k] = (255, 190, 90)
        for k in (3, 4, 5):
            mk = (lab == k).astype(np.uint8)
            if mk.mean() < a.min_area:
                continue
            over[mk > 0] = (60, 170, 255)
            cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (60, 200, 255), 2, cv2.LINE_AA)
        img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
        held = [n for n in ("left-hand object", "right-hand object", "two-hand object") if n in rec]
        if held:
            cv2.putText(img, "holding: " + ", ".join(held), (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 255), 2, cv2.LINE_AA)
        vw.write(img)
        if i % 100 == 0:
            print(f"  [rolehos] {i}/{len(frames)}", flush=True)

    vw.release()
    (out / "roles.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("wrote", out / "roles.json")


if __name__ == "__main__":
    main()
