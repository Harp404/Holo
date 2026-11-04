"""Rigid objects as 3D holograms that rotate with the real object.

  depth + lens        MoGe-2 (metric depth + intrinsics from plain RGB)
  3D model            a simple mesh for the object (phone = thin box by default; SAM 3D Objects mesh if given)
  6D pose per frame   FoundationPose: register on the first frame of each visible run (seeded with our
                      refined mask), then track frame to frame; re-register after a gap
  render              translucent bluish faces + bright edges, drawn over the frames (z-sorted faces)

Run from /workspace/holo/foundationpose with its venv:
  .venv/bin/python ../perception/object3d.py --frames ../out/pov_v2/hands3d_frames --raw ../frames/pov \
      --mask-dir ../out/pov_v2/refined/object1 --out ../out/pov_v2/phone3d --extents 0.075,0.155,0.009
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

FP = Path("/workspace/holo/foundationpose")
sys.path.insert(0, str(FP))
os.chdir(FP)
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402

FACE = np.array([255, 170, 70], np.float32)   # BGR bluish
EDGE = np.array([255, 235, 170], np.float32)  # BGR bright cyan-white


def moge_model():
    from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()


def depth_and_K(model, rgb):
    t = torch.tensor(rgb / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
    with torch.inference_mode():
        o = model.infer(t)
    h, w = rgb.shape[:2]
    Kn = o["intrinsics"].cpu().numpy()
    K = np.array([[Kn[0, 0] * w, 0, Kn[0, 2] * w], [0, Kn[1, 1] * h, Kn[1, 2] * h], [0, 0, 1]], np.float64)
    d = o["depth"].cpu().numpy().astype(np.float32)
    d[~np.isfinite(d)] = 0
    return d, K


def draw_hologram(img, mesh, pose, K):
    V = (pose[:3, :3] @ mesh.vertices.T + pose[:3, 3:4]).T
    if (V[:, 2] <= 0.01).any():
        return img
    uv = (K @ V.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    out = img.astype(np.float32)
    face_layer = np.zeros_like(out)
    face_alpha = np.zeros(img.shape[:2], np.float32)
    order = np.argsort(-V[mesh.faces].mean(1)[:, 2])
    normals = mesh.face_normals @ pose[:3, :3].T
    for f in order:
        pts = np.round(uv[mesh.faces[f]]).astype(np.int32)
        view = -V[mesh.faces[f]].mean(0)
        facing = float(np.dot(normals[f], view / (np.linalg.norm(view) + 1e-9)))
        a = 0.22 + 0.18 * (1 - abs(facing))          # edges-on faces glow a bit more (fresnel-like)
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, pts, 1)
        sel = m > 0
        face_layer[sel] = FACE
        face_alpha[sel] = np.maximum(face_alpha[sel], a)
    out = out * (1 - face_alpha[..., None]) + face_layer * face_alpha[..., None]
    # outline the silhouette + feature edges
    edges = mesh.edges_unique[mesh.face_adjacency_edges] if len(mesh.face_adjacency_angles) else mesh.edges_unique
    sharp = mesh.face_adjacency_edges[mesh.face_adjacency_angles > np.radians(30)]
    line = np.zeros(img.shape[:2], np.float32)
    for a, b in sharp:
        cv2.line(line, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), 1.0, 2, cv2.LINE_AA)
    glow = cv2.GaussianBlur(line, (0, 0), 4)
    out = out * (1 - line[..., None] * 0.9) + EDGE * (line[..., None] * 0.9)
    out += glow[..., None] * EDGE * 0.35
    return np.clip(out, 0, 255).astype(np.uint8)


SYM = [np.diag([1, 1, 1]), np.diag([1, -1, -1]), np.diag([-1, 1, -1]), np.diag([-1, -1, 1])]


def silhouette(mesh, pose, K, shape):
    V = (pose[:3, :3] @ mesh.vertices.T + pose[:3, 3:4]).T
    m = np.zeros(shape, np.uint8)
    if (V[:, 2] <= 0.01).any():
        return m
    uv = (K @ V.T).T
    uv = np.round(uv[:, :2] / uv[:, 2:3]).astype(np.int32)
    cv2.fillPoly(m, [uv[f] for f in mesh.faces], 1)
    return m > 0


def iou(a, b):
    return float((a & b).sum()) / max(1, float((a | b).sum()))


def smooth(prev, pose, k=0.55):
    """Blend toward the new pose; pick the box-symmetric twin closest to the previous rotation first."""
    if prev is None:
        return pose
    from scipy.spatial.transform import Rotation as Rot
    R = min((pose[:3, :3] @ S for S in SYM), key=lambda r: np.linalg.norm(r - prev[:3, :3]))
    q0, q1 = Rot.from_matrix(prev[:3, :3]).as_quat(), Rot.from_matrix(R).as_quat()
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - k) * q0 + k * q1
    out = np.eye(4)
    out[:3, :3] = Rot.from_quat(q / np.linalg.norm(q)).as_matrix()
    out[:3, 3] = (1 - k) * prev[:3, 3] + k * pose[:3, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="frames to draw on (e.g. the 3D-hands frames)")
    ap.add_argument("--raw", required=True, help="raw camera frames (for depth + pose)")
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extents", default="0.075,0.155,0.009", help="box size in metres (w,h,d) if no mesh")
    ap.add_argument("--mesh", default="")
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--draw-only", action="store_true", help="re-draw saved poses onto new --frames")
    ap.add_argument("--min-iou", type=float, default=0.55)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "poses").mkdir(exist_ok=True)
    if a.mesh:
        mesh = trimesh.load(a.mesh, force="mesh")
    else:
        mesh = trimesh.creation.box(extents=[float(x) for x in a.extents.split(",")])
        mesh = mesh.subdivide().subdivide()
    if a.draw_only:
        moge = moge_model()
        poses = sorted((out / "poses").glob("*.txt"))
        first = Path(a.raw) / poses[0].name.replace(".txt", ".jpg")
        _, K0 = depth_and_K(moge, cv2.cvtColor(cv2.imread(str(first)), cv2.COLOR_BGR2RGB))
        have = {p.stem: p for p in poses}
        shown, last = None, -9
        for i, rp in enumerate(sorted(Path(a.raw).glob("*.jpg"))):
            canvas = cv2.imread(str(Path(a.frames) / rp.name))
            if rp.stem in have:
                shown = smooth(shown if last == i - 1 else None, np.loadtxt(have[rp.stem]))
                last = i
                canvas = draw_hologram(canvas, mesh, shown, K0)
            cv2.imwrite(str(out / "frames" / rp.name), canvas)
        print("redrawn:", out)
        return
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                         debug_dir=str(out / "debug"), debug=0, glctx=dr.RasterizeCudaContext())
    moge = moge_model()

    raw = sorted(Path(a.raw).glob("*.jpg"))
    draw = {p.name: p for p in Path(a.frames).glob("*.jpg")}
    tracking, last_seen, K0, shown = False, -999, None, None
    n_reg = 0
    for i, rp in enumerate(raw):
        name = rp.name
        canvas = cv2.imread(str(draw.get(name, rp)))
        mp = Path(a.mask_dir) / name.replace(".jpg", ".png")
        has_mask = mp.exists()
        if has_mask:
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
            if mask.sum() < 400:
                has_mask = False
        if not has_mask and not tracking:
            cv2.imwrite(str(out / "frames" / name), canvas)
            continue
        bgr = cv2.imread(str(rp))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth, K = depth_and_K(moge, rgb)
        if K0 is None:
            K0 = K
        if has_mask and (not tracking or i - last_seen > a.gap):
            m_er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            pose = est.register(K=K0, rgb=rgb, depth=depth, ob_mask=m_er, iteration=5)
            n_reg += 1
            tracking, shown = True, None
        elif tracking:
            pose = est.track_one(rgb=rgb, depth=depth, K=K0, iteration=2)
            if has_mask:
                # drift check: the box silhouette must still cover the observed phone; else re-register
                s_tr = iou(silhouette(mesh, pose, K0, mask.shape), mask)
                if s_tr < a.min_iou:
                    keep = est.pose_last.clone()
                    m_er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    p2 = est.register(K=K0, rgb=rgb, depth=depth, ob_mask=m_er, iteration=4)
                    n_reg += 1
                    if iou(silhouette(mesh, p2, K0, mask.shape), mask) > s_tr:
                        pose = p2
                    else:
                        est.pose_last = keep
        if has_mask:
            last_seen = i
        elif i - last_seen > a.gap:
            tracking, shown = False, None
            cv2.imwrite(str(out / "frames" / name), canvas)
            continue
        np.savetxt(out / "poses" / name.replace(".jpg", ".txt"), pose.reshape(4, 4))
        shown = smooth(shown, pose)
        cv2.imwrite(str(out / "frames" / name), draw_hologram(canvas, mesh, shown, K0))
        if i % 25 == 0:
            print(f"  [object3d] frame {i} re-registrations {n_reg}", flush=True)
    print("done:", out)


if __name__ == "__main__":
    main()
