"""VR-style 3D hands with WiLoR (hand detector + ViT hand mesh, strong under hand-object occlusion).

Full MANO hand per detected hand (whole hand to the wrist, five separate fingers), projected with the
model's own full-image camera, rendered as a soft-blue shaded mesh with a glowing outline and bright
lines wherever one finger passes in front of another (so touching fingers never merge). Light
temporal smoothing per hand side.

Run (WiLoR venv):  wvenv/bin/python perception/hands_wilor.py --frames frames/pov --out out/pov_v4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from hands3d import BLUE, LIGHT, RIM, rasterize  # noqa: E402


def render(img, buf, zbuf, tips=(), objm=None, alpha=0.78):
    """Meta hand-representation style, in Holo blue: fresnel fill with slight transparency, light outline,
    fade at the wrist, fingertips glow when touching / about to touch an object."""
    out = img.astype(np.float32)
    cov = (buf[..., 3] > 0).astype(np.float32)
    if not cov.any():
        return img
    nb = cv2.GaussianBlur(buf[..., :3] * cov[..., None], (0, 0), 2.5)
    n = nb / (np.linalg.norm(nb, axis=2, keepdims=True) + 1e-6)
    lam = np.clip(np.abs((n * -LIGHT).sum(2)), 0, 1)
    fres = (1 - np.abs(n[..., 2])) ** 1.5
    deep = np.array([200, 120, 40], np.float32)        # BGR deeper blue (facing) -> light blue (grazing)
    col = deep * (0.55 + 0.45 * lam[..., None]) + (BLUE - deep) * fres[..., None] + RIM * (0.55 * fres[..., None] ** 2)
    fade = cv2.GaussianBlur(buf[..., 3], (0, 0), 1.0)    # per-vertex wrist fade
    a = cv2.GaussianBlur(cov, (0, 0), 0.8) * fade * (alpha + 0.2 * fres)
    out = out * (1 - a[..., None]) + np.clip(col, 0, 255) * a[..., None]
    z = np.where(np.isfinite(zbuf), zbuf, 0).astype(np.float32)
    for _ in range(2):
        z = np.where(z > 0, z, cv2.dilate(z, np.ones((5, 5), np.uint8)))
    z = cv2.GaussianBlur(z, (0, 0), 1.0)
    gz = np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3))
    inner = ((gz > 0.028) & (cv2.erode(cov, np.ones((3, 3), np.uint8)) > 0)).astype(np.float32)
    inner = cv2.GaussianBlur(inner, (0, 0), 0.6) * fade
    out = out * (1 - inner[..., None] * 0.7) + RIM * inner[..., None] * 0.7
    edge = np.clip(cov - cv2.erode(cov, np.ones((3, 3), np.uint8)), 0, 1) * fade
    glow = cv2.GaussianBlur(edge, (0, 0), 3.5) * 0.7 + cv2.GaussianBlur(edge, (0, 0), 0.8) * 0.8
    out = out + glow[..., None] * RIM * 0.6
    # contact: fingertips within reach of a held object light up
    if objm is not None and objm.any() and len(tips):
        dist = cv2.distanceTransform(1 - objm, cv2.DIST_L2, 5)
        h, w = cov.shape
        spot = np.zeros((h, w), np.float32)
        for x, y in tips:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h and dist[yi, xi] < 18:
                cv2.circle(spot, (xi, yi), 7, float(1 - dist[yi, xi] / 18), -1)
        spot = cv2.GaussianBlur(spot, (0, 0), 6) * 2.2
        out = out + np.clip(spot, 0, 1)[..., None] * np.array([255, 240, 200], np.float32) * 0.8
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", default="", help="dir with refined/objectK masks: fingertips near them glow")
    ap.add_argument("--smooth", type=float, default=0.65)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16, verbose=False)
    import trimesh
    faces_mano = np.asarray(pipe.wilor_model.mano.faces, np.int64)
    # smooth surface: subdivide MANO once; new vertices are edge midpoints -> rebuilt per frame
    _, faces_r = trimesh.remesh.subdivide(np.zeros((778, 3)), faces_mano)
    faces_r = np.asarray(faces_r, np.int64)
    faces_l = faces_r[:, ::-1].copy()

    out = Path(a.out)
    for d in ("frames", "meshes", "json"):
        (out / d).mkdir(parents=True, exist_ok=True)
    only = {int(x) for x in a.only.split(",") if x}
    prev = {}
    frames = sorted(Path(a.frames).glob("*.jpg"))
    for i, fp in enumerate(frames):
        n = i + 1
        if only and n not in only:
            continue
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]
        c = np.array([w / 2, h / 2])
        buf = np.zeros((h, w, 4), np.float32)
        zbuf = np.full((h, w), np.inf, np.float32)
        rec, tips = [], []
        outs = pipe.predict(img)
        # one hand per side: keep the most confident
        best = {}
        for o in outs:
            side = "right" if float(np.asarray(o["is_right"]).reshape(-1)[0]) > 0.5 else "left"
            sc = float(np.asarray(o.get("hand_bbox_conf", o.get("conf", 1.0))).reshape(-1)[0]) if isinstance(o, dict) else 1.0
            if side not in best or sc > best[side][0]:
                best[side] = (sc, o)
        for side, (_, o) in best.items():
            p = o["wilor_preds"]
            v = np.asarray(p["pred_vertices"][0], np.float64).copy()
            t = np.asarray(p["pred_cam_t_full"][0], np.float64)
            f = float(np.asarray(p["scaled_focal_length"]).reshape(-1)[0])
            v = trimesh.remesh.subdivide(v, faces_mano)[0]
            if side == "left":
                v[:, 0] = -v[:, 0]
            j3 = np.asarray(p["pred_keypoints_3d"][0], np.float64).copy()
            if side == "left":
                j3[:, 0] = -j3[:, 0]
            ax = j3[9] - j3[0]
            L = np.linalg.norm(ax) + 1e-9
            sproj = ((v - j3[0]) @ ax) / L / L          # 0 at wrist joint, 1 at knuckles
            valpha = np.clip((sproj + 0.15) / 0.45, 0, 1)
            V = v + t
            if (V[:, 2] <= 0.02).any():
                continue
            uv = V[:, :2] / V[:, 2:3] * f + c
            pv = prev.get(side)
            if pv is not None and pv[0] == n - 1 and np.median(np.abs(uv - pv[2])) < 0.08 * w:
                k = a.smooth
                V = k * V + (1 - k) * pv[1]
                uv = k * uv + (1 - k) * pv[2]
            prev[side] = (n, V.copy(), uv.copy())
            F = faces_r if side == "right" else faces_l
            rasterize(buf, zbuf, V, F, uv, valpha)
            tips.extend(np.asarray(p["pred_keypoints_2d"][0])[[4, 8, 12, 16, 20]].tolist())
            np.savez_compressed(out / "meshes" / f"{n:04d}_{side}.npz", verts=V.astype(np.float32), uv=uv.astype(np.float32), faces=F.astype(np.int32))
            kp = np.asarray(p["pred_keypoints_2d"][0]).round(1).tolist()
            # for driving Meta's hand model in Unity: camera-space joints (metres, OpenCV axes) + MANO params
            jc = (j3 + t)
            extra = {k: np.asarray(p[k][0]).round(5).tolist() for k in ("global_orient", "hand_pose", "betas") if k in p}
            rec.append({"side": side, "focal": f, "kp2d": kp, "joints_cam": jc.round(5).tolist(),
                        "cam_t": t.round(5).tolist(), "img_wh": [w, h], **extra})
        objm = np.zeros((h, w), np.uint8)
        for od in Path(a.objects).glob("object*") if a.objects else []:
            mp = od / f"{n:04d}.png"
            if mp.exists():
                objm |= (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        frame = render(img, buf, zbuf, tips=tips, objm=objm)
        cv2.imwrite(str(out / "frames" / fp.name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "json" / f"{n:04d}.json").write_text(json.dumps(rec))
        if only:
            cv2.imwrite(str(out / f"test_{n:04d}.jpg"), frame)
        if n % 50 == 0:
            print(f"  [wilor] {n}/{len(frames)}", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
