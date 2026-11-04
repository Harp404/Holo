"""Re-cut every held-object mask with SAM 2.1-Large so the whole object is covered and hands never are.

Per frame and object:
  box        union of the tracked mask's box and the HOI-DETR "1st object" box that matches it
             (HOI-DETR boxes the whole held object even when it is flipped or half hidden)
  positives  points spread over the tracked mask (minus the hand)
  negatives  points on the exact hand mask near the object
  pick       best of SAM's 3 proposals: high predicted IoU, agrees with the tracked mask, little hand
  clean      hand pixels removed again; tiny islands dropped
Output: OUT/objectK/NNNN.png (same layout as refined/)

Run (any venv with torch + transformers>=4.56): python objects_reseg.py --frames F --objects refined --handseg H --hoi pov.json --out refined2
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor


def spread_points(mask, k, rng):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    idx = rng.choice(len(xs), size=min(len(xs), 400), replace=False)
    pts = np.stack([xs[idx], ys[idx]], 1).astype(np.float32)
    # farthest-point sampling for good coverage
    out = [pts[0]]
    d = np.linalg.norm(pts - pts[0], axis=1)
    for _ in range(1, min(k, len(pts))):
        i = int(np.argmax(d))
        out.append(pts[i])
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return [p.tolist() for p in out]


def match_hoi_box(fr, mbox):
    """HOI-DETR 1st-object box that best overlaps the tracked mask's box (but isn't a huge background box)."""
    if fr is None:
        return None
    best, bs = None, 0.0
    marea = max(1.0, (mbox[2] - mbox[0]) * (mbox[3] - mbox[1]))
    for det in fr["detections"]:
        if det["class_id"] != 1 or det["score"] < 0.3:
            continue
        b = np.array(det["box"], float)
        ix = max(0, min(b[2], mbox[2]) - max(b[0], mbox[0]))
        iy = max(0, min(b[3], mbox[3]) - max(b[1], mbox[1]))
        barea = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        cover = ix * iy / marea
        if cover > 0.45 and barea < 3.5 * marea and cover > bs:
            best, bs = b, cover
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--objects", required=True, help="refined/ with objectK/NNNN.png")
    ap.add_argument("--handseg", required=True)
    ap.add_argument("--hoi", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").cuda().eval()
    hoi = json.load(open(a.hoi))["frames"]
    rng = np.random.default_rng(0)
    objs = sorted(Path(a.objects).glob("object*"))
    for od in objs:
        (Path(a.out) / od.name).mkdir(parents=True, exist_ok=True)
    stats = {"done": 0, "grew": 0}
    for fp in sorted(Path(a.frames).glob("*.jpg")):
        n = int(fp.stem)
        img = None
        hs = cv2.imread(str(Path(a.handseg) / f"{n:04d}.png"), cv2.IMREAD_GRAYSCALE)
        hand = (hs > 100) if hs is not None else None
        for od in objs:
            mp = od / f"{n:04d}.png"
            if not mp.exists():
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
            if hand is None:
                hand = np.zeros_like(m)
            core = cv2.erode((m & ~cv2.dilate(hand.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)).astype(np.uint8),
                             np.ones((5, 5), np.uint8)) > 0
            if core.sum() < 40:
                cv2.imwrite(str(Path(a.out) / od.name / mp.name), m.astype(np.uint8) * 255)
                continue
            if img is None:
                img = Image.open(fp).convert("RGB")
            H, W = m.shape
            ys, xs = np.where(m)
            mbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
            hb = match_hoi_box(hoi[n - 1] if n - 1 < len(hoi) else None, mbox)
            box = mbox if hb is None else np.array([min(mbox[0], hb[0]), min(mbox[1], hb[1]), max(mbox[2], hb[2]), max(mbox[3], hb[3])])
            pad = 0.04 * max(box[2] - box[0], box[3] - box[1])
            box = [float(max(0, box[0] - pad)), float(max(0, box[1] - pad)), float(min(W - 1, box[2] + pad)), float(min(H - 1, box[3] + pad))]
            pos = spread_points(core, 6, rng)
            near = hand.copy()
            near[: int(box[1]), :] = False
            near[int(box[3]) + 1:, :] = False
            near[:, : int(box[0])] = False
            near[:, int(box[2]) + 1:] = False
            near = cv2.erode(near.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            neg = spread_points(near, 5, rng)
            pts = pos + neg
            labels = [1] * len(pos) + [0] * len(neg)
            inp = proc(images=img, input_boxes=[[box]], input_points=[[pts]], input_labels=[[labels]], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                o = sam(**inp, multimask_output=True)
            masks = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy() > 0
            ious = o.iou_scores.cpu().numpy().reshape(-1)
            best, bscore = None, -9
            for i in range(masks.shape[0]):
                cand = masks[i]
                if cand.sum() < 30:
                    continue
                agree = (cand & m).sum() / max(1, m.sum())               # keeps the tracked object
                spill = (cand & hand).sum() / max(1, cand.sum())        # hand inside = bad
                growth = cand.sum() / max(1, m.sum())
                score = ious[i] + 0.6 * agree - 1.5 * spill - 0.4 * max(0, growth - 3.0)
                if score > bscore:
                    best, bscore = cand, score
            if best is None:
                best = m
            best = best & ~cv2.dilate(hand.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            nc, lab, st, _ = cv2.connectedComponentsWithStats(best.astype(np.uint8), 8)
            if nc > 1:
                keep = np.zeros(nc, bool)
                keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(25, 0.04 * st[1:, cv2.CC_STAT_AREA].max())
                best = keep[lab]
            stats["done"] += 1
            stats["grew"] += int(best.sum() > 1.15 * m.sum())
            cv2.imwrite(str(Path(a.out) / od.name / mp.name), best.astype(np.uint8) * 255)
        if n % 100 == 0:
            print(f"  [reseg] {n} {stats}", flush=True)
    print("done", stats)


if __name__ == "__main__":
    main()
