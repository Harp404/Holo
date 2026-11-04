"""SAM 3.1 image segmentation service: text prompt -> instance masks (for the Holo app's live highlights).
Run on the GPU box:  python -m uvicorn sam31_server:app --host 127.0.0.1 --port 8790  (reach it via an SSH tunnel)."""
import base64
import io

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf

VERSION = "sam3.1"
_proc = None


def get_proc():
    """Loaded on demand; /unload frees it so image/3D generation can use the GPU + RAM (31 GB container)."""
    global _proc
    if _proc is None:
        ck = download_ckpt_from_hf(version="sam3.1")
        _proc = Sam3Processor(build_sam3_image_model(checkpoint_path=ck, load_from_HF=False))
    return _proc


app = FastAPI()


@app.post("/unload")
def unload():
    import gc
    global _proc
    _proc = None
    gc.collect()
    torch.cuda.empty_cache()
    return {"ok": True}


class Req(BaseModel):
    image: str
    text: str
    top_k: int = 3
    min_score: float = 0.35


@app.get("/health")
def health():
    return {"ok": True, "model": VERSION}


@app.post("/segment")
def segment(r: Req):
    img = Image.open(io.BytesIO(base64.b64decode(r.image))).convert("RGB")
    proc = get_proc()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = proc.set_image(img)
        o = proc.set_text_prompt(state=st, prompt=r.text)
    n = len(o["scores"])
    masks = o["masks"].float().cpu().numpy().reshape(n, img.height, img.width) if n else np.zeros((0, img.height, img.width))
    scores = o["scores"].float().cpu().numpy().reshape(-1)
    boxes = o["boxes"].float().cpu().numpy().reshape(-1, 4)
    out = []
    for i in np.argsort(-scores)[: r.top_k]:
        if scores[i] < r.min_score:
            continue
        ok, png = cv2.imencode(".png", (masks[i] > 0.5).astype(np.uint8) * 255)
        out.append({"score": float(scores[i]), "box": boxes[i].round(1).tolist(), "mask": base64.b64encode(png.tobytes()).decode()})
    return {"model": VERSION, "text": r.text, "instances": out, "width": img.width, "height": img.height}
