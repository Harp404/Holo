"""Video -> per-frame 3D body of the main person (SAM 3D Body).
Saves joints in camera space (metres): kp3d (T, 70, 3) = pred_keypoints_3d + pred_cam_t, plus 2D keypoints and boxes.

python extract_body.py VIDEO OUT.npz [--fps 30] [--every 1]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/workspace/r/sam-3d-body")
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--render", default="", help="folder: write the SAM 3D Body mesh over each frame")
ap.add_argument("--ckpt", default="/workspace/r/ckpt/sam-3d-body-dinov3")
a = ap.parse_args()

dev = torch.device("cuda")
ck = a.ckpt
ckpt_file = os.path.join(ck, "model.ckpt") if os.path.isdir(ck) else ck
mhr = os.path.join(ck, "assets", "mhr_model.pt") if os.path.isdir(ck) else ""
model, cfg = load_sam_3d_body(ckpt_file, device=dev, mhr_path=mhr)
from tools.build_detector import HumanDetector  # noqa: E402
from tools.build_fov_estimator import FOVEstimator  # noqa: E402
det = HumanDetector(name="vitdet", device=dev, path="")
try:
    fov = FOVEstimator(name="moge2", device=dev, path="")
except Exception as e:  # noqa: BLE001
    print("fov estimator unavailable:", e)
    fov = None
est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=det, human_segmentor=None, fov_estimator=fov)

cap = cv2.VideoCapture(a.video)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = src_fps / a.fps
kp3d, kp2d, boxes, idx = [], [], [], []
i, nxt, prev_c = 0, 0.0, None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i + 1e-6 >= nxt:
        nxt += step
        outs = est.process_one_image(frame[:, :, ::-1].copy(), bbox_thr=0.5)
        if outs:
            # main person: biggest box, and near the previous one
            def score(o):
                b = o["bbox"]; area = (b[2] - b[0]) * (b[3] - b[1]); c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                return area - (0 if prev_c is None else 4 * np.linalg.norm(c - prev_c) ** 2)
            o = max(outs, key=score)
            b = o["bbox"]; prev_c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            kp3d.append(o["pred_keypoints_3d"][:70] + o["pred_cam_t"][None])
            if a.render:
                from tools.vis_utils import visualize_sample_together
                os.makedirs(a.render, exist_ok=True)
                cv2.imwrite(f"{a.render}/{len(kp3d):05d}.jpg", visualize_sample_together(frame, [o], est.faces).astype(np.uint8))
            kp2d.append(o["pred_keypoints_2d"][:70])
            boxes.append(b)
            idx.append(i)
    i += 1
np.savez_compressed(a.out, kp3d=np.array(kp3d, np.float32), kp2d=np.array(kp2d, np.float32), boxes=np.array(boxes, np.float32),
                    frames=np.array(idx), fps=a.fps, src_fps=src_fps)
print("EXTRACT_OK", len(kp3d), "frames")
