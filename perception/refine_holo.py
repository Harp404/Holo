"""Merge duplicate object tracks and sharpen their edges, then render the holographic preview.

- Merge: two tracks are the same physical object if their SAM 3 masks overlap (IoU) on most shared frames.
- Refine: BiRefNet (high-res dichotomous segmentation) on a tight crop around each SAM 3 mask, kept inside a
  dilated SAM 3 region -> crisp outlines, even on thin cables.
Run: .venv/bin/python refine_holo.py --frames F --held OUT/pov_held
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from track_held import hand_skin, holo_shell


def load_track(d: Path, n: int):
    out = {}
    for i in range(n):
        p = d / f"{i + 1:04d}.png"
        if p.exists():
            out[i] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
    return out


def merge_tracks(tracks, iou_same=0.5, share=0.6):
    ids = sorted(tracks)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            common = set(tracks[a]) & set(tracks[b])
            if len(common) < 5:
                continue
            hits = 0
            for f in common:
                ma, mb = tracks[a][f], tracks[b][f]
                inter = np.logical_and(ma, mb).sum()
                if inter / max(1, np.logical_or(ma, mb).sum()) >= iou_same:
                    hits += 1
            if hits / len(common) >= share:
                parent[find(b)] = find(a)
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    merged = {}
    for root, members in groups.items():
        fm = {}
        for m in members:
            for f, mask in tracks[m].items():
                fm[f] = mask if f not in fm else (fm[f] | mask)
        merged[root] = fm
    print("merged groups:", list(groups.values()))
    return merged


class BiRef:
    def __init__(self):
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        self.m = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).cuda().eval().half()
        self.tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def refine(self, rgb, sam_mask, pad=0.25):
        ys, xs = np.where(sam_mask)
        h, w = sam_mask.shape
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        x1, x2 = max(0, int(xs.min() - pad * bw)), min(w, int(xs.max() + pad * bw) + 1)
        y1, y2 = max(0, int(ys.min() - pad * bh)), min(h, int(ys.max() + pad * bh) + 1)
        crop = Image.fromarray(rgb[y1:y2, x1:x2])
        with torch.inference_mode():
            pred = self.m(self.tf(crop).unsqueeze(0).cuda().half())[-1].sigmoid()[0, 0].float().cpu().numpy()
        prob = cv2.resize(pred, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        region = cv2.dilate(sam_mask[y1:y2, x1:x2].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        out = np.zeros_like(sam_mask)
        out[y1:y2, x1:x2] = (prob > 0.5) & region
        # if BiRefNet locked onto something else in the crop, keep SAM 3's mask
        inter = np.logical_and(out, sam_mask).sum()
        if inter < 0.5 * sam_mask.sum():
            return sam_mask
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--held", required=True)
    a = ap.parse_args()

    held = Path(a.held)
    meta = json.loads((held / "objects.json").read_text())
    frames = sorted(Path(a.frames).glob("*.jpg"))
    n = len(frames)
    tracks = {o["id"]: load_track(held / "masks" / f"object{o['id']}", n) for o in meta["objects"]}
    merged = merge_tracks(tracks)
    biref = BiRef()

    out_dir = held / "refined"
    vw = cv2.VideoWriter(str(held / "holo_refined.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"]))
    frames_meta = []
    for oid in merged:
        (out_dir / f"object{oid}").mkdir(parents=True, exist_ok=True)
    for i, fp in enumerate(frames):
        bgr = cv2.imread(str(fp))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rec = {}
        for oid, fm in merged.items():
            m = fm.get(i)
            if m is None or not m.any():
                continue
            r = biref.refine(rgb, m)
            cv2.imwrite(str(out_dir / f"object{oid}" / f"{i + 1:04d}.png"), r.astype(np.uint8) * 255)
            ys, xs = np.where(r)
            rec[str(oid)] = [round(xs.min() / meta["width"] * 1000), round(ys.min() / meta["height"] * 1000),
                             round(xs.max() / meta["width"] * 1000), round(ys.max() / meta["height"] * 1000)]
            bgr = holo_shell(bgr, r, i / meta["fps"])
        hp = held / "hands" / f"{i + 1:04d}.png"
        if hp.exists():
            bgr = hand_skin(bgr, cv2.imread(str(hp), cv2.IMREAD_GRAYSCALE) > 127)
        frames_meta.append(rec)
        vw.write(bgr)
        if i % 100 == 0:
            print(f"  [refine] {i}/{n}", flush=True)
    vw.release()
    meta["objects"] = [{"id": oid, "name": f"object{oid}", "members": []} for oid in merged]
    meta["frames"] = frames_meta
    (held / "objects_refined.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("wrote holo_refined.mp4 and objects_refined.json")


if __name__ == "__main__":
    main()
