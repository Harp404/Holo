"""Holo auto perception (GPU side). Nothing here is specific to one video.

Inputs: a frame folder, the labels Muse named in its plan, HOI-DETR per-frame predictions.
  1. SAM 3.1 video tracking for every label Muse named (text prompts, all instances)
  2. held objects: every hand -> object link from HOI-DETR becomes a SAM 3.1 box prompt -> exact mask
  3. DINOv3 dense patch-feature grid per frame, pooled inside each instance mask (identity among look-alikes)
Outputs (OUT/): masks/<label>/NNNN_<inst>.png, held/NNNN_<k>.png, dino/<label>.npz, meta.json

python auto_pod.py --frames F --labels "a|b|c" --hoi pov.json --out OUT
"""

import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np
import torch
from PIL import Image


def sam31_video(frames, labels, out, chunk=120):
    from sam3.model_builder import build_sam3_predictor
    pred = build_sam3_predictor(version="sam3.1")
    files = sorted(glob.glob(f"{frames}/*.jpg"))
    meta = {}
    for s in range(0, len(files), chunk):
        cdir = f"{out}/_chunk"
        shutil.rmtree(cdir, ignore_errors=True)
        os.makedirs(cdir)
        for i, f in enumerate(files[s:s + chunk]):
            os.symlink(os.path.abspath(f), f"{cdir}/{i:05d}.jpg")
        sess = pred.handle_request(dict(type="start_session", resource_path=cdir, offload_video_to_cpu=True))["session_id"]
        for lab in labels:
            pred.handle_request(dict(type="reset_session", session_id=sess))
            pred.handle_request(dict(type="add_prompt", session_id=sess, frame_index=0, text=lab))
            d = f"{out}/masks/{lab.replace(' ', '_')}"
            os.makedirs(d, exist_ok=True)
            for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sess)):
                o = r["outputs"]
                for i, oid in enumerate(o["out_obj_ids"].tolist()):
                    m = o["out_binary_masks"][i]
                    if m.any():
                        n = s + r["frame_index"] + 1
                        cv2.imwrite(f"{d}/{n:04d}_c{s}o{oid}.png", m.astype(np.uint8) * 255)
                        meta.setdefault(lab, 0)
                        meta[lab] += 1
        pred.handle_request(dict(type="close_session", session_id=sess))
        torch.cuda.empty_cache()
        print(f"  [sam3.1] frames {s + 1}-{min(len(files), s + chunk)}", flush=True)
    del pred
    torch.cuda.empty_cache()
    return meta


def held_masks(frames, hoi, out, max_area=0.35):
    """HOI-DETR hand->object links -> SAM 3.1 box prompts -> exact masks of what each hand holds."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    model = build_sam3_image_model(checkpoint_path=download_ckpt_from_hf(version="sam3.1"), load_from_HF=False)
    proc = Sam3Processor(model)
    os.makedirs(f"{out}/held", exist_ok=True)
    cnt = 0
    for fr in hoi:
        n = fr["frame_idx"] + 1
        links = [l for l in fr["hf"] if l["prob"] > 0.5]
        if not links:
            continue
        img = Image.open(f"{frames}/{n:04d}.jpg").convert("RGB")
        W, H = img.size
        st = None
        for k, l in enumerate(links):
            b = fr["detections"][l["b"]]["box"]
            if (b[2] - b[0]) * (b[3] - b[1]) > max_area * W * H:      # "touching the laptop" style boxes
                continue
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = proc.set_image(img) if st is None else st
                proc.reset_all_prompts(st)
                box = [(b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H]
                o = proc.add_geometric_prompt(box, True, st)
            if len(o["scores"]):
                i = int(torch.argmax(o["scores"]))
                m = o["masks"][i].float().cpu().numpy().reshape(H, W) > 0.5
                if m.any():
                    cv2.imwrite(f"{out}/held/{n:04d}_{k}.png", m.astype(np.uint8) * 255)
                    cnt += 1
    print("  [held] masks", cnt, flush=True)
    del model
    torch.cuda.empty_cache()


def dino_embed(frames, out, long_side=848):
    """DINOv3 dense patch-feature grid per frame (one pass), pooled inside every tracked instance mask.
    Gives each instance an appearance descriptor for identity matching among look-alikes."""
    from transformers import AutoModel
    name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dm = AutoModel.from_pretrained(name).cuda().eval()
    nreg = getattr(dm.config, "num_register_tokens", 4)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    by_frame = {}
    for d in sorted(glob.glob(f"{out}/masks/*")):
        for f in sorted(glob.glob(f"{d}/*.png")):
            by_frame.setdefault(int(os.path.basename(f)[:4]), []).append(f)
    os.makedirs(f"{out}/dino", exist_ok=True)
    rows = {}
    for n, fl in sorted(by_frame.items()):
        img = cv2.cvtColor(cv2.imread(f"{frames}/{n:04d}.jpg"), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        sc = long_side / max(H, W)
        gw, gh = int(W * sc) // 16, int(H * sc) // 16
        x = torch.tensor(cv2.resize(img, (gw * 16, gh * 16)), device="cuda").permute(2, 0, 1)[None].float() / 255
        with torch.inference_mode():
            tok = dm(pixel_values=(x - mean) / std).last_hidden_state[0, 1 + nreg:]     # (gh*gw, C) patch grid
        grid = torch.nn.functional.normalize(tok.float(), dim=-1).view(gh, gw, -1)
        for f in fl:
            m = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            mg = torch.tensor(cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA), device="cuda")
            if mg.sum() < 0.3:
                continue
            v = (grid * mg[..., None]).sum((0, 1)) / mg.sum()
            lab = os.path.basename(os.path.dirname(f))
            rows.setdefault(lab, ([], []))
            rows[lab][0].append(os.path.basename(f))
            rows[lab][1].append(torch.nn.functional.normalize(v, dim=0).cpu().numpy())
    for lab, (keys, feats) in rows.items():
        np.savez_compressed(f"{out}/dino/{lab}.npz", keys=np.array(keys), feats=np.stack(feats).astype(np.float16))
    print("  [dino] patch-grid descriptors for", {k: len(v[0]) for k, v in rows.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--hoi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-video", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labels = [l.strip() for l in a.labels.split("|") if l.strip()]
    if "hand" not in labels:
        labels.append("hand")          # hands are always tracked (occlusion: hands stay in front)
    meta = {"labels": labels}
    if not a.skip_video:
        meta["sam31_masks"] = sam31_video(a.frames, labels, a.out)
    held_masks(a.frames, json.load(open(a.hoi))["frames"], a.out)
    dino_embed(a.frames, a.out)
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    print("AUTO_DONE")


if __name__ == "__main__":
    main()
