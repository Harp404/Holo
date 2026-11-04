"""Holo perception v2 - strongest model per job.

  which object is in the hand   RoleHOS (DINOv3 ConvNeXt backbone)        -> hold episodes + seed masks
  follow the whole object        DAM4SAM (distractor-aware SAM 2.1-Large)  -> forward AND backward from the seed
  crisp edges                    BiRefNet (high-res dichotomous seg)        -> refine each frame inside a crop
  same object, many grabs        mask-overlap merge (+ DINOv3 if needed)
  hands                          RoleHOS hand masks, temporally smoothed (3D hand meshes come next)

Run (single venv at /workspace/holo/.venv):
  cd /workspace/holo && .venv/bin/python perception/perceive_v2.py --frames frames/pov --out out/pov_v2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/workspace/holo")
sys.path.insert(0, str(ROOT / "perception"))
from refine_holo import BiRef, merge_tracks  # noqa: E402
from track_held import find_episodes, smooth_hands  # noqa: E402

HELD = (3, 4, 5)


def run_rolehos(frames, out: Path):
    sys.path.insert(0, str(ROOT / "rolehos"))
    cwd = os.getcwd()
    os.chdir(ROOT / "rolehos")
    from rolehos.runtime import load_model, predict
    model = load_model("full", 0, "cuda")[0].eval()
    os.chdir(cwd)
    (out / "roles").mkdir(parents=True, exist_ok=True)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    for i, fp in enumerate(frames):
        dst = out / "roles" / f"{i + 1:04d}.png"
        if dst.exists():
            continue
        rgb = np.asarray(Image.open(fp).convert("RGB"))
        with torch.inference_mode():
            m = predict(model, rgb)
        lab = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(lab).convert("RGB").save(dst)
    print(f"  [rolehos] {len(frames)} frames")
    del model
    torch.cuda.empty_cache()
    return sorted((out / "roles").glob("*.png"))


def dam4sam_track(frames, key, seed_mask):
    """Track one object forward and backward from `key` with a fresh DAM4SAM tracker each way."""
    cwd = os.getcwd()
    os.chdir(ROOT / "dam4sam")
    sys.path.insert(0, str(ROOT / "dam4sam"))
    from dam4sam_tracker import DAM4SAMTracker
    out = {}
    for order in (range(key, len(frames)), range(key, -1, -1)):
        tr = DAM4SAMTracker("sam21pp-L")
        for j, fi in enumerate(order):
            img = Image.open(frames[fi]).convert("RGB")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = tr.initialize(img, seed_mask) if j == 0 else tr.track(img)
            m = np.asarray(o["pred_mask"]).astype(bool)
            if m.any():
                out[fi] = m if fi not in out else (out[fi] | m)
        del tr
        torch.cuda.empty_cache()
    os.chdir(cwd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    a = ap.parse_args()

    frames = sorted(Path(a.frames).resolve().glob("*.jpg"))
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    h, w = cv2.imread(str(frames[0])).shape[:2]

    labs = run_rolehos(frames, out)
    eps = find_episodes(labs)
    print("hold episodes:", [(s, e) for s, e, _ in eps])

    # track each episode's object; skip an episode whose seed is already covered by a tracked object
    tracks = {}
    for s, e, key in eps:
        lab = cv2.imread(str(labs[key]), cv2.IMREAD_UNCHANGED)[..., 0]
        seed = np.isin(lab, HELD)
        covered = False
        for oid, fm in tracks.items():
            m = fm.get(key)
            if m is not None and np.logical_and(m, seed).sum() >= 0.5 * seed.sum():
                covered = True
                print(f"  episode {s}-{e}: already tracked as object {oid}")
                break
        if covered:
            continue
        oid = len(tracks) + 1
        print(f"  episode {s}-{e}: DAM4SAM object {oid} from keyframe {key}", flush=True)
        tracks[oid] = dam4sam_track(frames, key, seed)
        print(f"    tracked in {len(tracks[oid])}/{len(frames)} frames", flush=True)

    tracks = merge_tracks(tracks)
    biref = BiRef()
    (out / "hands").mkdir(exist_ok=True)
    hands = smooth_hands(labs, (h, w))
    meta = {"fps": a.fps, "width": w, "height": h, "objects": [{"id": k, "name": f"object{k}"} for k in tracks], "frames": []}
    for k in tracks:
        (out / "refined" / f"object{k}").mkdir(parents=True, exist_ok=True)
    for i, fp in enumerate(frames):
        rgb = np.asarray(Image.open(fp).convert("RGB"))
        rec = {}
        for k, fm in tracks.items():
            m = fm.get(i)
            if m is None or not m.any():
                continue
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            r = biref.refine(rgb, m)
            cv2.imwrite(str(out / "refined" / f"object{k}" / f"{i + 1:04d}.png"), r.astype(np.uint8) * 255)
            ys, xs = np.where(r)
            rec[str(k)] = [round(xs.min() / w * 1000), round(ys.min() / h * 1000), round(xs.max() / w * 1000), round(ys.max() / h * 1000)]
        cv2.imwrite(str(out / "hands" / f"{i + 1:04d}.png"), cv2.resize(hands[i], (w, h), interpolation=cv2.INTER_NEAREST) * 255)
        meta["frames"].append(rec)
    (out / "objects_refined.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("wrote", out / "objects_refined.json")


if __name__ == "__main__":
    main()
