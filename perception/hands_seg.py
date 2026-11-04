"""Exact hand masks: SAM 2.1-Large prompted with WiLoR's hand box + its 21 joints (positive points).

The WiLoR mesh silhouette (hull, plus a small reach) cuts the forearm off so the mask stops at the wrist.
Output: soft masks (0-255) per frame in OUT/NNNN.png (both hands merged) and OUT/NNNN_left|right.png.

Run (main venv): .venv/bin/python perception/hands_seg.py --frames frames --wilor out/wilor --out out/handseg
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--wilor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reach", type=int, default=28, help="px the mask may extend past the WiLoR hand hull")
    a = ap.parse_args()

    proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").cuda().eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W = Path(a.wilor)
    frames = sorted(Path(a.frames).glob("*.jpg"))
    for i, fp in enumerate(frames):
        n = i + 1
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        total = np.zeros((h, w), np.float32)
        jp = W / "json" / f"{n:04d}.json"
        recs = json.loads(jp.read_text()) if jp.exists() else []
        for r in recs:
            side = r["side"]
            mp = W / "meshes" / f"{n:04d}_{side}.npz"
            if not mp.exists():
                continue
            d = np.load(mp)
            uv, F = d["uv"], d["faces"]
            sil = np.zeros((h, w), np.uint8)
            for t in F:
                cv2.fillConvexPoly(sil, np.round(uv[t]).astype(np.int32), 1)
            if sil.sum() < 50:
                continue
            ys, xs = np.where(sil)
            pad = 10
            box = [float(max(0, xs.min() - pad)), float(max(0, ys.min() - pad)), float(min(w - 1, xs.max() + pad)), float(min(h - 1, ys.max() + pad))]
            pts = np.asarray(r["kp2d"], np.float32)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            pts = pts[ok]
            kw = dict(images=img, input_boxes=[[box]], return_tensors="pt")
            if len(pts) >= 3:
                kw["input_points"] = [[pts.tolist()]]
                kw["input_labels"] = [[[1] * len(pts)]]
            inp = proc(**kw).to("cuda")
            with torch.inference_mode():
                o = sam(**inp, multimask_output=False)
            lg = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"], binarize=False)[0][0, 0].numpy().astype(np.float32)
            m = cv2.GaussianBlur(lg, (0, 0), 2) > 0.0
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            # keep the component(s) touching the hand; cut the forearm with a soft hull fade
            hull = np.zeros((h, w), np.uint8)
            cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.fillConvexPoly(hull, cv2.convexHull(np.vstack(cnts)), 1)
            dist = cv2.distanceTransform(1 - hull, cv2.DIST_L2, 5)
            fade = np.clip(1 - dist / a.reach, 0, 1)
            if m.sum() < 0.35 * sil.sum():  # SAM missed: fall back to the WiLoR silhouette
                m = sil
            soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8) * fade
            cv2.imwrite(str(out / f"{n:04d}_{side}.png"), (soft * 255).astype(np.uint8))
            total = np.maximum(total, soft)
        cv2.imwrite(str(out / f"{n:04d}.png"), (total * 255).astype(np.uint8))
        if n % 100 == 0:
            print(f"  [hands_seg] {n}/{len(frames)}", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
