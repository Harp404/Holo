"""Task objects = things the hands actually pick up, tracked whole and steady through the clip.

1. RoleHOS roles (run_rolehos.py) -> "hold episodes": frames where a hand holds a compact object
   (surfaces like a laptop deck or table are rejected by size).
2. DINOv3 groups episodes that hold the same physical object (grab, put down, grab again).
3. SAM 3 video tracker: seeded with the held-object mask at a clean keyframe, propagated forwards AND
   backwards, so the WHOLE object is followed - including before it is picked up and after release.
4. Hands: RoleHOS hand masks, temporally smoothed (no crawling edges).
5. Output: objects.json, masks/<obj>/0001.png, hands/0001.png, crops for naming, and a holographic preview.

Run from /root/holo/perception with its venv:
  .venv/bin/python track_held.py --frames F --roles OUT/pov_roles --out OUT/pov_held
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HELD = (3, 4, 5)


# ----------------------------------------------------------------------------- episodes

def held_mask(lab):
    return np.isin(lab, HELD)


def find_episodes(lab_paths, min_area=0.002, max_area=0.08, min_len=4):
    """Runs of frames with a compact held object. Returns [(start, end, keyframe)]."""
    areas, oks = [], []
    for p in lab_paths:
        lab = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., 0]
        m = held_mask(lab)
        a = m.mean()
        areas.append(a)
        oks.append(min_area <= a <= max_area)
    eps, i = [], 0
    while i < len(oks):
        if oks[i]:
            j = i
            while j + 1 < len(oks) and oks[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                seg = list(range(i, j + 1))
                med = np.median([areas[k] for k in seg])
                key = min(seg, key=lambda k: abs(areas[k] - med))
                eps.append((i, j, key))
            i = j + 1
        else:
            i += 1
    return eps


# ----------------------------------------------------------------------------- DINOv3 grouping

class Dino:
    def __init__(self, model_id="facebook/dinov3-vits16-pretrain-lvd1689m"):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()

    def embed(self, rgb, mask):
        ys, xs = np.where(mask)
        crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        crop[~mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]] = 0
        inp = self.proc(images=Image.fromarray(crop), return_tensors="pt").to("cuda")
        with torch.inference_mode():
            f = self.model(pixel_values=inp["pixel_values"].half()).pooler_output[0].float().cpu().numpy()
        return f / (np.linalg.norm(f) + 1e-6)


def crop_of(rgb, mask, pad=24):
    ys, xs = np.where(mask)
    y1, y2 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
    return rgb[y1:y2, x1:x2]


# ----------------------------------------------------------------------------- SAM 3 tracking

def track_objects(frames_dir, seeds, n_frames):
    """seeds: [(obj_id, keyframe, bool mask)] -> {obj_id: {frame: bool mask}}"""
    from sam3.model_builder import build_sam3_video_predictor

    pred = build_sam3_video_predictor()
    sid = pred.handle_request(dict(type="start_session", resource_path=str(frames_dir)))["session_id"]
    out = {}
    for oid, key, mask in seeds:
        pred.handle_request(dict(type="reset_session", session_id=sid))
        pred.handle_request(dict(type="add_mask", session_id=sid, frame_index=key, obj_id=oid,
                                 mask=torch.from_numpy(mask)))
        per = {}
        with torch.inference_mode():
            for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid, propagation_direction="both")):
                o = r["outputs"]
                ids = np.asarray(o["out_obj_ids"]).tolist()
                if oid in ids:
                    m = np.asarray(o["out_binary_masks"][ids.index(oid)]).astype(bool)
                    if m.any():
                        per[r["frame_index"]] = m
        print(f"  [sam3] object {oid}: tracked in {len(per)}/{n_frames} frames", flush=True)
        out[oid] = per
    pred.handle_request(dict(type="close_session", session_id=sid))
    return out


# ----------------------------------------------------------------------------- hands

def smooth_hands(lab_paths, shape, alpha=0.55):
    """Temporal EMA (forward + backward) on hand masks, then smooth contours."""
    h, w = shape
    raw = []
    for p in lab_paths:
        lab = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., 0]
        raw.append(np.isin(lab, (1, 2)).astype(np.float32))
    fwd, acc = [], raw[0]
    for r in raw:
        acc = alpha * r + (1 - alpha) * acc
        fwd.append(acc)
    out, acc = [None] * len(raw), fwd[-1]
    for i in range(len(raw) - 1, -1, -1):
        acc = alpha * fwd[i] + (1 - alpha) * acc
        soft = cv2.GaussianBlur(acc, (0, 0), 3)
        out[i] = (soft > 0.5).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------- holographic render

HOLO = np.array([255, 225, 110], np.float32)   # BGR: light cyan
HAND = np.array([255, 185, 95], np.float32)    # BGR: soft blue


def holo_shell(img, mask, t, color=HOLO, strength=1.0):
    """Thin light 'hologram cover': crisp rim, inner glow fading to the centre, scanlines, outer glow."""
    m = mask.astype(np.uint8)
    if not m.any():
        return img
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    inner = np.exp(-dist / 5.0) * m                           # bright near the rim, fades inward
    rim = (m - cv2.erode(m, k)).astype(np.float32)
    rim = cv2.GaussianBlur(rim, (0, 0), 0.6)
    outer = cv2.GaussianBlur(rim, (0, 0), 5) * (1 - m)        # soft glow just outside
    ys = np.arange(img.shape[0], dtype=np.float32)[:, None]
    scan = 0.82 + 0.18 * np.sin(ys * 0.55 - t * 5.0)          # moving scanlines
    a = (0.06 * m + 0.32 * inner * scan + 0.95 * rim + 0.45 * outer) * strength
    out = img.astype(np.float32) + a[..., None] * color[None, None, :] * 0.85
    return np.clip(out, 0, 255).astype(np.uint8)


def hand_skin(img, mask, color=HAND):
    m = mask.astype(np.uint8)
    if not m.any():
        return img
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rim = cv2.GaussianBlur((m - cv2.erode(m, k)).astype(np.float32), (0, 0), 1.2)
    base = img.astype(np.float32)
    tint = base * (1 - 0.42 * soft[..., None]) + color[None, None, :] * (0.42 * soft[..., None])
    tint += rim[..., None] * color[None, None, :] * 0.9
    return np.clip(tint, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--same", type=float, default=0.72, help="DINOv3 cosine to treat two grabs as one object")
    a = ap.parse_args()

    frames = sorted(Path(a.frames).glob("*.jpg"))
    labs = sorted((Path(a.roles) / "roles").glob("*.png"))
    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    h, w = cv2.imread(str(frames[0])).shape[:2]

    eps = find_episodes(labs)
    print("hold episodes:", [(s, e) for s, e, _ in eps])

    dino = Dino()
    seeds, protos = [], []
    for s, e, key in eps:
        rgb = np.asarray(Image.open(frames[key]).convert("RGB"))
        lab = cv2.imread(str(labs[key]), cv2.IMREAD_UNCHANGED)[..., 0]
        m = held_mask(lab)
        emb = dino.embed(rgb, m)
        match = next((oid for oid, p in protos if float(emb @ p) >= a.same), None)
        if match is None:
            oid = len(protos) + 1
            protos.append((oid, emb))
            seeds.append((oid, key, m))
            Image.fromarray(crop_of(rgb, m)).save(out / "crops" / f"object{oid}.jpg")
            print(f"  episode {s}-{e}: new object {oid} (keyframe {key})")
        else:
            print(f"  episode {s}-{e}: same object as {match} (DINOv3)")

    tracks = track_objects(Path(a.frames), seeds, len(frames))
    hands = smooth_hands(labs, (h, w))

    meta = {"fps": a.fps, "width": w, "height": h,
            "objects": [{"id": oid, "keyframe": key, "name": f"object{oid}"} for oid, key, _ in seeds],
            "episodes": [{"start": s, "end": e} for s, e, _ in eps], "frames": []}
    vw = cv2.VideoWriter(str(out / "holo_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    for oid in tracks:
        (out / "masks" / f"object{oid}").mkdir(parents=True, exist_ok=True)
    (out / "hands").mkdir(exist_ok=True)
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        rec = {}
        for oid, per in tracks.items():
            m = per.get(i)
            if m is None:
                continue
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            cv2.imwrite(str(out / "masks" / f"object{oid}" / f"{i + 1:04d}.png"), m.astype(np.uint8) * 255)
            ys, xs = np.where(m)
            rec[str(oid)] = [round(xs.min() / w * 1000), round(ys.min() / h * 1000), round(xs.max() / w * 1000), round(ys.max() / h * 1000)]
            img = holo_shell(img, m, i / a.fps)
        hm = cv2.resize(hands[i], (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / "hands" / f"{i + 1:04d}.png"), hm * 255)
        img = hand_skin(img, hm)
        meta["frames"].append(rec)
        vw.write(img)
    vw.release()
    (out / "objects.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("wrote", out / "objects.json", "and holo_preview.mp4")


if __name__ == "__main__":
    main()
