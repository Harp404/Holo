"""Holo perception pipeline (runs on a GPU server).

Turns a first-person video (JPEG frame folder) into a per-frame scene description Holo can render
and reason over:

  1. inventory  - Muse Spark Open names the distinct parts it sees (short noun phrases).
  2. SAM 3      - text-prompted video tracking: a pixel mask + stable object id per part, per frame.
  3. DINOv3     - appearance fingerprint per track; merges tracks split by occlusion and keeps
                  look-alike parts (four motors) from swapping identities.
  4. hands      - 21 joints per hand per frame.
  5. contacts   - which fingertips touch which part; grasp = thumb + index on the same part.

Output (out_dir):
  scene.json          {fps, width, height, objects[], frames[{objects[], hands[], contacts[]}]}
  masks/0001.png ...  uint8 label map at mask_width (0 = background, k = objects[k-1])

Usage:
  python perceive.py --frames ../HoloXR/Assets/StreamingAssets/Passthrough/drone --out out/drone \
      --goal "put the propellers on this drone" [--labels "green propeller,brushless motor,..."]
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


# ----------------------------------------------------------------------------- 1. inventory

def inventory(frame_path: Path, goal: str, server: str) -> list[str]:
    """Ask Muse Spark (through the Holo server) for short, segmentable noun phrases."""
    img = base64.b64encode(frame_path.read_bytes()).decode()
    body = json.dumps({"image": img, "goal": goal}).encode()
    req = urllib.request.Request(f"{server}/api/inventory", data=body, headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=180))
    labels = [l.strip().lower() for l in out.get("labels", []) if l.strip()]
    print("inventory:", labels)
    return labels


# ----------------------------------------------------------------------------- 2. SAM 3 tracking

def track_with_sam3(frames_dir: Path, labels: list[str], n_frames: int, multiplex: bool):
    """Returns {label: {obj_id: {frame_idx: bool mask}}}."""
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    builder = build_sam3_video_predictor
    if multiplex:
        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor as builder  # SAM 3.1
        except ImportError:
            pass
    predictor = builder()
    sid = predictor.handle_request(dict(type="start_session", resource_path=str(frames_dir)))["session_id"]

    results = {}
    probe = sorted({0, n_frames // 4, n_frames // 2, (3 * n_frames) // 4})
    for label in labels:
        predictor.handle_request(dict(type="reset_session", session_id=sid))
        # prompt on the first probe frame where the concept is found
        found = False
        for f in probe:
            out = predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=f, text=label))["outputs"]
            if len(out["out_obj_ids"]) > 0:
                found = True
                break
            predictor.handle_request(dict(type="reset_session", session_id=sid))
        if not found:
            print(f"  [sam3] '{label}': not found")
            continue
        per_obj: dict[int, dict[int, np.ndarray]] = {}
        with torch.inference_mode():
            for resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                fi = resp["frame_index"]
                o = resp["outputs"]
                for k, oid in enumerate(np.asarray(o["out_obj_ids"]).tolist()):
                    m = np.asarray(o["out_binary_masks"][k]).astype(bool)
                    if m.any():
                        per_obj.setdefault(int(oid), {})[fi] = m
        print(f"  [sam3] '{label}': {len(per_obj)} object(s)")
        results[label] = per_obj
    predictor.handle_request(dict(type="close_session", session_id=sid))
    return results


def track_with_sam3_images(frames: list[Path], labels: list[str], thresh: float = 0.45, iou_link: float = 0.3):
    """Fast path: SAM 3 image model per frame (backbone once, all text prompts reuse it), then link
    detections into tracks by mask overlap. DINOv3 later merges tracks split by occlusion.
    Returns {label: {track_id: {frame_idx: bool mask}}}."""
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    proc = Sam3Processor(model)
    proc.set_confidence_threshold(thresh)
    results = {l: {} for l in labels}
    active = {l: {} for l in labels}  # track_id -> (last_frame, small mask)
    next_id = 0

    def small(m):
        return m[::4, ::4]

    for fi, fp in enumerate(frames):
        img = Image.open(fp).convert("RGB")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(img)
            for label in labels:
                out = proc.set_text_prompt(label, state)
                masks = out["masks"]
                if masks is None or len(masks) == 0:
                    continue
                masks = masks.squeeze(1).bool().cpu().numpy()
                scores = out["scores"].float().cpu().numpy()
                order = np.argsort(-scores)
                used = set()
                for k in order:
                    m = masks[k]
                    if not m.any():
                        continue
                    sm = small(m)
                    best, best_iou = None, iou_link
                    for tid, (lf, lm) in active[label].items():
                        if tid in used or fi - lf > 6:
                            continue
                        inter = np.logical_and(sm, lm).sum()
                        iou = inter / max(1, np.logical_or(sm, lm).sum())
                        if iou > best_iou:
                            best, best_iou = tid, iou
                    if best is None:
                        best = next_id
                        next_id += 1
                        results[label][best] = {}
                    used.add(best)
                    results[label][best][fi] = m
                    active[label][best] = (fi, sm)
        if fi % 50 == 0:
            print(f"  [sam3-img] frame {fi}/{len(frames)}", flush=True)
    # drop flickers: tracks seen in fewer than 3 frames
    for label in labels:
        results[label] = {t: fm for t, fm in results[label].items() if len(fm) >= 3}
        print(f"  [sam3-img] '{label}': {len(results[label])} track(s)")
    return results


# ----------------------------------------------------------------------------- 3. DINOv3 identity

class Dino:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()

    def embed(self, img: Image.Image, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask)
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        crop = np.asarray(img)[y1:y2, x1:x2].copy()
        crop[~mask[y1:y2, x1:x2]] = 0
        inp = self.proc(images=Image.fromarray(crop), return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            feats = self.model(pixel_values=inp["pixel_values"].half()).pooler_output[0].float().cpu().numpy()
        return feats / (np.linalg.norm(feats) + 1e-6)


def build_tracks(raw, frames: list[Path], dino: "Dino | None", merge_sim: float = 0.72, min_len: int = 6):
    """Flatten {label:{oid:{f:mask}}} into tracks, fingerprint them, merge occlusion splits."""
    tracks = []
    for label, objs in raw.items():
        for oid, fm in objs.items():
            if len(fm) >= min_len:
                tracks.append({"label": label, "frames": dict(fm)})
    if dino is not None:
        for t in tracks:
            keys = sorted(t["frames"])
            sample = keys[:: max(1, len(keys) // 4)][:4]
            embs = [dino.embed(Image.open(frames[f]).convert("RGB"), t["frames"][f]) for f in sample]
            t["emb"] = np.mean(embs, axis=0)
            t["emb"] /= np.linalg.norm(t["emb"]) + 1e-6
        # merge same-label tracks that never co-exist and look alike (an object hidden then re-found)
        merged = True
        while merged:
            merged = False
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    a, b = tracks[i], tracks[j]
                    if a["label"] != b["label"] or set(a["frames"]) & set(b["frames"]):
                        continue
                    if float(a["emb"] @ b["emb"]) >= merge_sim:
                        a["frames"].update(b["frames"])
                        a["emb"] = (a["emb"] + b["emb"]) / 2
                        tracks.pop(j)
                        merged = True
                        break
                if merged:
                    break
    print("tracks after DINOv3 merge:", {l: sum(1 for t in tracks if t["label"] == l) for l in raw})
    # stable display names: "brushless motor 1", "brushless motor 2" ordered left-to-right on first sight
    by_label = {}
    for t in tracks:
        by_label.setdefault(t["label"], []).append(t)
    for label, ts in by_label.items():
        def first_x(t):
            f0 = min(t["frames"])
            return np.where(t["frames"][f0])[1].mean()
        ts.sort(key=first_x)
        for k, t in enumerate(ts):
            t["name"] = label if len(ts) == 1 else f"{label} {k + 1}"
    return tracks


# ----------------------------------------------------------------------------- 4. hands

def hands_mediapipe(frames: list[Path], model_path: str):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    out = []
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for i, f in enumerate(frames):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(Image.open(f).convert("RGB")))
            r = lm.detect_for_video(img, int(i * 1000 / 15))
            hands = []
            for k, pts in enumerate(r.hand_landmarks):
                side = r.handedness[k][0].category_name.lower() if r.handedness else "unknown"
                hands.append({"side": side, "kp": [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in pts]})
            out.append(hands)
    return out


# ----------------------------------------------------------------------------- 5. contacts + export

def export(tracks, hands, frames: list[Path], out_dir: Path, fps: float, mask_width: int):
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    W, H = Image.open(frames[0]).size
    mh = round(mask_width * H / W)
    objects = [{"index": k + 1, "name": t["name"], "label": t["label"]} for k, t in enumerate(tracks)]
    scene = {"fps": fps, "width": W, "height": H, "mask_width": mask_width, "mask_height": mh, "objects": objects, "frames": []}

    hand_idx = {k + 1 for k, t in enumerate(tracks) if t["label"] == "hand"}
    for fi, fpath in enumerate(frames):
        label_map = np.zeros((H, W), np.uint8)
        no_hand = np.zeros((H, W), np.uint8)
        fobjs = []
        for k, t in enumerate(tracks):
            m = t["frames"].get(fi)
            if m is None:
                continue
            label_map[m] = k + 1
            if t["label"] != "hand":
                no_hand[m] = k + 1
            ys, xs = np.where(m)
            fobjs.append({
                "index": k + 1,
                "box": [round(xs.min() / W * 1000), round(ys.min() / H * 1000), round(xs.max() / W * 1000), round(ys.max() / H * 1000)],
                "center": [round(xs.mean() / W * 1000), round(ys.mean() / H * 1000)],
                "area": round(m.mean(), 5),
            })
        Image.fromarray(label_map).resize((mask_width, mh), Image.NEAREST).convert("RGB").save(out_dir / "masks" / f"{fi + 1:04d}.png")

        # contacts: fingertip pixels inside a (slightly dilated) object mask
        contacts = []
        fh = hands[fi] if fi < len(hands) else []
        for hi, h in enumerate(fh):
            touched = {}
            for name, j in FINGERTIPS.items():
                x = int(min(W - 1, max(0, h["kp"][j][0] * W)))
                y = int(min(H - 1, max(0, h["kp"][j][1] * H)))
                obj = no_hand[max(0, y - 6):y + 7, max(0, x - 6):x + 7].max()  # hands touch parts, not themselves
                if obj:
                    touched.setdefault(int(obj), []).append(name)
            for obj, fingers in touched.items():
                kind = "grasp" if "thumb" in fingers and "index" in fingers else "touch"
                contacts.append({"hand": hi, "object": obj, "fingers": fingers, "type": kind})
        scene["frames"].append({"objects": fobjs, "hands": fh, "contacts": contacts})

    (out_dir / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))
    print(f"wrote {out_dir/'scene.json'} ({len(frames)} frames, {len(objects)} objects)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--goal", default="")
    ap.add_argument("--labels", default="", help="comma-separated; skips the Muse inventory call")
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--mask-width", type=int, default=640)
    ap.add_argument("--dino", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--hand-model", default="hand_landmarker.task")
    ap.add_argument("--multiplex", action="store_true", help="use SAM 3.1 Object Multiplex if available")
    ap.add_argument("--video-mode", action="store_true", help="SAM 3 video tracker (slower, one label per pass)")
    a = ap.parse_args()

    frames = sorted(Path(a.frames).glob("*.jpg"))
    if not frames:
        sys.exit(f"no frames in {a.frames}")
    out = Path(a.out)
    labels = [l.strip() for l in a.labels.split(",") if l.strip()] or inventory(frames[0], a.goal, a.server)

    cache = out / "sam3_raw.pkl"
    if cache.exists():
        import pickle
        packed = pickle.loads(cache.read_bytes())
        raw = {l: {o: {f: np.unpackbits(m[0])[: m[1][0] * m[1][1]].reshape(m[1]).astype(bool) for f, m in fm.items()} for o, fm in objs.items()} for l, objs in packed.items()}
        print("loaded cached SAM 3 tracks")
    else:
        if a.video_mode:
            raw = track_with_sam3(Path(a.frames), labels, len(frames), a.multiplex)
        else:
            raw = track_with_sam3_images(frames, labels)
        import pickle
        out.mkdir(parents=True, exist_ok=True)
        packed = {l: {o: {f: (np.packbits(m), m.shape) for f, m in fm.items()} for o, fm in objs.items()} for l, objs in raw.items()}
        cache.write_bytes(pickle.dumps(packed))
    dino = None if a.no_dino else Dino(a.dino)
    tracks = build_tracks(raw, frames, dino)
    hands = hands_mediapipe(frames, a.hand_model) if os.path.exists(a.hand_model) else [[] for _ in frames]
    export(tracks, hands, frames, out, a.fps, a.mask_width)


if __name__ == "__main__":
    main()
