"""Hands as full, pixel-accurate blue covers.

  coverage   SAM 2.1-Large (image mode) prompted with the hand's box + its 3D-hand joints as positive points
             -> exact hand silhouette (every finger), cut to the 3D hand's neighbourhood (no forearm)
  shading    the SAM 3D Body mesh (saved by hands3d.py): smooth normals -> soft 3D light + rim
  fingers    depth discontinuities of the mesh drawn as thin bright lines, so touching fingers stay separate

Run (main venv):  .venv/bin/python perception/hands_cover.py --frames F --hands3d out/pov_v3 --roles R --out out/pov_v3/hands_cover
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

sys.path.insert(0, str(Path(__file__).parent))
from hands3d import BLUE, LIGHT, RIM, rasterize  # noqa: E402


def mesh_buffers(npz_paths, h, w):
    buf = np.zeros((h, w, 4), np.float32)
    zbuf = np.full((h, w), np.inf, np.float32)
    sils = []
    for p in npz_paths:
        d = np.load(p)
        V, F, f = d["verts"].astype(np.float64), d["faces"], float(d["focal"])
        uv = V[:, :2] / V[:, 2:3] * f + np.array([w / 2, h / 2])
        rasterize(buf, zbuf, V, F, uv, np.ones(len(V)))
        sil = np.zeros((h, w), np.uint8)
        cv2.fillPoly(sil, [np.round(uv[t]).astype(np.int32) for t in F], 1)
        sils.append(sil > 0)
    return buf, zbuf, sils


def joints(kp2d, h, w):
    """Valid hand joints: the invalid (body) entries are all duplicates of one point."""
    k = np.round(np.asarray(kp2d), 1)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    pts = k[cnt[inv.reshape(-1)] == 1]
    return pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]


def render(img, cover, buf, zbuf, alpha=0.86):
    out = img.astype(np.float32)
    cov = (cover > 0.05).astype(np.float32)
    soft = cv2.GaussianBlur(cover.astype(np.float32), (0, 0), 1.0)
    mcov = (buf[..., 3] > 0).astype(np.float32)
    nb = cv2.GaussianBlur(buf[..., :3] * mcov[..., None], (0, 0), 6)
    cb = cv2.GaussianBlur(mcov, (0, 0), 3)
    n = nb / (np.linalg.norm(nb, axis=2, keepdims=True) + 1e-6)
    lam = np.clip(np.abs((n * -LIGHT).sum(2)), 0, 1)
    fres = (1 - np.abs(n[..., 2])) ** 2
    inside = cv2.GaussianBlur(cv2.erode(mcov, np.ones((15, 15), np.uint8)), (0, 0), 9)
    wm = np.clip(inside * 1.2, 0, 1)[..., None]          # mesh shading fades smoothly to flat where mesh is absent
    lam = wm[..., 0] * lam + (1 - wm[..., 0]) * 0.7
    fres = wm[..., 0] * fres * 0.4
    col = BLUE * (0.45 + 0.55 * lam[..., None]) + RIM * (0.7 * fres[..., None])
    a = soft * alpha
    out = out * (1 - a[..., None]) + np.clip(col, 0, 255) * a[..., None]
    # finger separation: depth jumps inside the mesh
    z = np.where(np.isfinite(zbuf), zbuf, 0).astype(np.float32)
    for _ in range(3):  # extend depth past the mesh border so the border itself is not an "edge"
        z = np.where(z > 0, z, cv2.dilate(z, np.ones((9, 9), np.uint8)))
    z = cv2.GaussianBlur(z, (0, 0), 1.2)
    gz = np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3))
    inner = ((gz > 0.06) & (cv2.erode(mcov, np.ones((17, 17), np.uint8)) > 0) & (cover > 0.5)).astype(np.float32)
    inner = cv2.GaussianBlur(inner, (0, 0), 0.7)
    out = out * (1 - inner[..., None] * 0.55) + RIM * inner[..., None] * 0.55
    # glowing outline of the real hand silhouette
    edge = np.clip(cov - cv2.erode(cov, np.ones((3, 3), np.uint8)), 0, 1)
    glow = cv2.GaussianBlur(edge, (0, 0), 3) * 0.7 + cv2.GaussianBlur(edge, (0, 0), 0.8) * 0.6
    glow *= cv2.GaussianBlur(cover.astype(np.float32), (0, 0), 2) ** 2   # no outline where the wrist fades out
    return np.clip(out + glow[..., None] * RIM * 0.6, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="frames to draw on")
    ap.add_argument("--raw", required=True, help="raw camera frames (SAM input)")
    ap.add_argument("--hands3d", required=True, help="dir with hands3d/*.npz and hands3d_json/")
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reach", type=int, default=22, help="px the cover may extend beyond the 3D hand")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").cuda().eval()
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    H3 = Path(a.hands3d)
    raw = sorted(Path(a.raw).glob("*.jpg"))
    only = {int(x) for x in a.only.split(",") if x}
    for i, rp in enumerate(raw):
        n = i + 1
        if only and n not in only:
            continue
        canvas = cv2.imread(str(Path(a.frames) / rp.name))
        h, w = canvas.shape[:2]
        jp = H3 / "hands3d_json" / f"{n:04d}.json"
        recs = json.loads(jp.read_text()) if jp.exists() else []
        npzs = [H3 / "hands3d" / f"{n:04d}_{r['side']}.npz" for r in recs]
        npzs = [p for p in npzs if p.exists()]
        if not npzs:
            cv2.imwrite(str(out / "frames" / rp.name), canvas)
            continue
        buf, zbuf, sils = mesh_buffers(npzs, h, w)
        lab = cv2.imread(str(Path(a.roles) / f"{n:04d}.png"), cv2.IMREAD_UNCHANGED)[..., 0]
        rgb = Image.open(rp).convert("RGB")
        cover = np.zeros((h, w), np.float32)
        mesh_core = np.zeros((h, w), bool)
        for r, sil in zip(recs, sils):
            pts = joints(r["kp2d"], h, w)
            core = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            mesh_core |= core > 0
            cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:  # convex hull: back of the hand counts even where the mesh cut is ragged
                big = max(cnts, key=cv2.contourArea)
                cv2.fillConvexPoly(core, cv2.convexHull(big), 1)
            dist = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
            fade = np.clip(1 - dist / a.reach, 0, 1)
            near = fade > 0
            ys, xs = np.where(sil | (lab == (1 if r["side"] == "left" else 2)) & near)
            if len(xs) == 0:
                continue
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            kw = dict(images=rgb, input_boxes=[[box]], return_tensors="pt")
            if len(pts) >= 3:
                kw["input_points"] = [[pts.tolist()]]
                kw["input_labels"] = [[[1] * len(pts)]]
            inp = proc(**kw).to("cuda")
            with torch.inference_mode():
                o = sam(**inp, multimask_output=False)
            lg = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"], binarize=False)[0][0, 0].numpy()
            m = cv2.GaussianBlur(lg.astype(np.float32), (0, 0), 3) > 0.5
            if m.sum() < 0.3 * sil.sum():  # SAM missed: fall back to the mesh silhouette
                m = sil
            m = cv2.medianBlur(m.astype(np.uint8) * 255, 9) > 127   # kill speckle in SAM's unsure forearm zone
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
            cover = np.maximum(cover, m * cv2.GaussianBlur(fade, (0, 0), 3))
        buf[~mesh_core] = 0                      # drop stray wrist triangles from the shading
        zbuf[~mesh_core] = np.inf
        keep = cv2.morphologyEx((cover > 0.05).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cover = cover * cv2.dilate(keep, np.ones((3, 3), np.uint8))
        cv2.imwrite(str(out / "masks" / f"{n:04d}.png"), (cover * 255).astype(np.uint8))
        cv2.imwrite(str(out / "frames" / rp.name), render(canvas, cover, buf, zbuf), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if only:
            cv2.imwrite(str(out / f"test_{n:04d}.jpg"), render(canvas, cover, buf, zbuf))
        if n % 50 == 0:
            print(f"  [hands_cover] {n}/{len(raw)}", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
