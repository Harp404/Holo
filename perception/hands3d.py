"""3D hands for every frame with SAM 3D Body (hand decoder only), rendered as soft-blue VR hands.

Hand boxes come from RoleHOS hand masks (1 = left, 2 = right). Left hands are mirrored in, run as right
hands, and mirrored back (how SAM 3D Body's own full pipeline does it). The MHR mesh is cut down to the
hand + wrist (vertices near the predicted hand joints), projected with the predicted camera, and rasterised
with a z-buffer: Lambert + fresnel rim shading, so fingers read as separate 3D shapes.

Run (SAM 3D Body venv):
  cd /workspace/holo/sam3dbody && .venv/bin/python ../perception/hands3d.py --frames ../frames/pov --roles ../out/pov_v2/roles --out ../out/pov_v2
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/workspace/holo/sam3dbody")

LIGHT = np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8])
BLUE = np.array([255, 170, 90], np.float32)    # BGR soft blue
RIM = np.array([255, 235, 190], np.float32)    # BGR pale cyan rim


def hand_subset(verts, kp3d, radius=0.034, fade=0.014):
    """Vertices near the hand joints; alpha fades out over the last `fade` metres (soft wrist cut)."""
    kps = kp3d[np.abs(kp3d).sum(1) > 1e-6]
    if len(kps) == 0:
        return np.zeros(len(verts), bool), np.zeros(len(verts))
    d = np.min(np.linalg.norm(verts[:, None, :] - kps[None, :, :], axis=2), axis=1)
    return d < radius, np.clip((radius - d) / fade, 0, 1)


def vertex_normals(V, faces):
    fn = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)


def rasterize(buf, zbuf, verts_cam, faces, uv, valpha):
    """z-buffered raster of vertex normals + alpha into buffers (shaded later, smoothly, per pixel)."""
    tri = verts_cam[faces]
    vn = vertex_normals(verts_cam, faces)
    n = vn[faces].mean(1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    fa = valpha[faces].mean(1)
    h, w = zbuf.shape
    for t in np.argsort(-tri[:, :, 2].mean(1)):
        pts = uv[faces[t]]
        zt = tri[t, :, 2].mean()
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)
        sub_m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillConvexPoly(sub_m, np.round(pts).astype(np.int32) - [x0, y0], 1)
        sel = (sub_m > 0) & (zt < zbuf[y0:y1 + 1, x0:x1 + 1])
        if not sel.any():
            continue
        zbuf[y0:y1 + 1, x0:x1 + 1][sel] = zt
        buf[y0:y1 + 1, x0:x1 + 1][sel] = np.r_[n[t], fa[t]]


def shade(img, buf, alpha=0.85):
    """Smooth Lambert + fresnel from blurred normals, soft glowing silhouette."""
    cov = (buf[..., 3] > 0).astype(np.float32)
    if not cov.any():
        return img
    k = (0, 0)
    nb = cv2.GaussianBlur(buf[..., :3] * cov[..., None], k, 2.2)
    cb = cv2.GaussianBlur(cov, k, 2.2)
    n = nb / (np.linalg.norm(nb, axis=2, keepdims=True) + 1e-6)
    lam = np.clip(np.abs((n * -LIGHT).sum(2)), 0, 1)
    fres = (1 - np.abs(n[..., 2])) ** 2
    col = BLUE * (0.55 + 0.45 * lam[..., None]) + RIM * (0.8 * fres[..., None])
    a = cv2.GaussianBlur(buf[..., 3], k, 2.5) * np.clip(cb * 1.6, 0, 1) * alpha
    out = img * (1 - a[..., None]) + np.clip(col, 0, 255) * a[..., None]
    edge = np.clip(cov - cv2.erode(cov, np.ones((3, 3), np.uint8)), 0, 1)
    glow = cv2.GaussianBlur(edge, k, 3) * 0.6 + cv2.GaussianBlur(edge, k, 0.8) * 0.5
    glow *= cv2.GaussianBlur(buf[..., 3], k, 2)
    return out + glow[..., None] * RIM * 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--min-area", type=float, default=0.006)
    ap.add_argument("--focal", type=float, default=800.0, help="camera focal in px (MoGe-2 estimate for the POV clip)")
    ap.add_argument("--smooth", type=float, default=0.6, help="weight of the new frame in the temporal blend")
    ap.add_argument("--only", default="", help="comma-separated 1-based frame numbers to test")
    a = ap.parse_args()

    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf
    model, cfg = load_sam_3d_body_hf("facebook/sam-3d-body-dinov3")
    est = SAM3DBodyEstimator(model, cfg)
    faces_all = est.faces

    frames = sorted(Path(a.frames).glob("*.jpg"))
    out = Path(a.out)
    (out / "hands3d").mkdir(parents=True, exist_ok=True)
    h, w = cv2.imread(str(frames[0])).shape[:2]
    vw = cv2.VideoWriter(str(out / "hands3d_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    records = []
    prev = {}  # side -> (frame index, verts, uv) for temporal smoothing
    import torch
    cam = torch.tensor([[[a.focal, 0, w / 2], [0, a.focal, h / 2], [0, 0, 1]]], dtype=torch.float32)
    only = {int(x) - 1 for x in a.only.split(",") if x}
    for i, fp in enumerate(frames):
        if only and i not in only:
            continue
        done_jpg = out / "hands3d_frames" / f"{i + 1:04d}.jpg"
        if not only and done_jpg.exists():
            continue
        img = cv2.imread(str(fp))
        lab = cv2.imread(str(Path(a.roles) / f"{i + 1:04d}.png"), cv2.IMREAD_UNCHANGED)[..., 0]
        canvas = img.astype(np.float32)
        zbuf = np.full((h, w), np.inf, np.float32)
        buf = np.zeros((h, w, 4), np.float32)
        rec = []
        for side, k in (("left", 1), ("right", 2)):
            mk = lab == k
            if mk.mean() < a.min_area:
                continue
            ys, xs = np.where(mk)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
            src = img
            if side == "left":
                src = img[:, ::-1].copy()
                box = np.array([w - 1 - box[2], box[1], w - 1 - box[0], box[3]], np.float32)
            try:
                o = est.process_one_image(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), bboxes=box[None], cam_int=cam, inference_type="hand")[0]
            except Exception as e:  # keep going on odd frames
                print("frame", i, side, "failed:", e)
                continue
            V = np.asarray(o["pred_vertices"]) + np.asarray(o["pred_cam_t"])
            kp3 = np.asarray(o["pred_keypoints_3d"])
            keep, valpha = hand_subset(np.asarray(o["pred_vertices"]), kp3)
            fsel = keep[faces_all].all(1)
            faces = faces_all[fsel]
            f = float(np.asarray(o["focal_length"]).reshape(-1)[0])
            # align: solve the 3D translation t (linear least squares) so kp3d + t projects onto kp2d
            kp2 = np.asarray(o["pred_keypoints_2d"]).copy()
            sel_k = np.abs(kp3).sum(1) > 1e-6
            V0 = np.asarray(o["pred_vertices"])
            c = np.array([w / 2, h / 2])
            t = np.asarray(o["pred_cam_t"], np.float64)
            X, Q = kp3[sel_k].astype(np.float64), kp2[sel_k].astype(np.float64) - c
            if len(X) >= 4:
                wgt = np.ones(len(X))
                for _ in range(2):  # second pass down-weights outlier joints
                    A = np.zeros((2 * len(X), 3)); b = np.zeros(2 * len(X))
                    A[0::2, 0] = f; A[0::2, 2] = -Q[:, 0]; b[0::2] = Q[:, 0] * X[:, 2] - f * X[:, 0]
                    A[1::2, 1] = f; A[1::2, 2] = -Q[:, 1]; b[1::2] = Q[:, 1] * X[:, 2] - f * X[:, 1]
                    W = np.repeat(wgt, 2)
                    t_new = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)[0]
                    if t_new[2] + X[:, 2].min() <= 0.05:
                        break
                    t = t_new
                    P = (X[:, :2] + t[:2]) / (X[:, 2:3] + t[2]) * f
                    r = np.linalg.norm(P - Q, axis=1)
                    wgt = 1.0 / np.maximum(1.0, r / (np.median(r) * 2 + 1e-6))
            V = V0 + t
            uv = V[:, :2] / V[:, 2:3] * f + c
            if side == "left":
                V = V * np.array([-1, 1, 1])          # mirror back
                faces = faces[:, ::-1]                # keep winding consistent
                uv[:, 0] = w - 1 - uv[:, 0]
                kp2[:, 0] = w - 1 - kp2[:, 0]
            pv = prev.get(side)
            if pv is not None and pv[0] == i - 1 and pv[1].shape == V.shape:
                # blend toward the new pose; a big jump (re-detection) resets instead of smearing
                if np.median(np.abs(uv - pv[2])) < 0.08 * w:
                    k = a.smooth
                    V = k * V + (1 - k) * pv[1]
                    uv = k * uv + (1 - k) * pv[2]
            prev[side] = (i, V.copy(), uv.copy())
            rasterize(buf, zbuf, V, faces, uv, valpha)
            rec.append({"side": side, "focal": f, "cam_t": np.asarray(o["pred_cam_t"]).tolist(),
                        "kp2d": kp2.round(1).tolist()})
            np.savez_compressed(out / "hands3d" / f"{i + 1:04d}_{side}.npz", verts=V.astype(np.float32), faces=faces.astype(np.int32), focal=f)
        records.append(rec)
        canvas = shade(canvas, buf)
        frame_out = np.clip(canvas, 0, 255).astype(np.uint8)
        (out / "hands3d_frames").mkdir(exist_ok=True)
        cv2.imwrite(str(out / "hands3d_frames" / f"{i + 1:04d}.jpg"), frame_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "hands3d_json").mkdir(exist_ok=True)
        (out / "hands3d_json" / f"{i + 1:04d}.json").write_text(json.dumps(rec))
        vw.write(frame_out)
        if only:
            cv2.imwrite(str(out / f"hands3d_test_{i + 1:04d}.jpg"), np.clip(canvas, 0, 255).astype(np.uint8))
        if i % 50 == 0:
            print(f"  [hands3d] {i}/{len(frames)}", flush=True)
    vw.release()
    (out / "hands3d.json").write_text(json.dumps({"fps": a.fps, "width": w, "height": h, "frames": records}, separators=(",", ":")))
    print("wrote hands3d_preview.mp4")


if __name__ == "__main__":
    main()
