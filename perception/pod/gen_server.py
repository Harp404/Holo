"""Holo Build-mode generation service (GPU box): design image -> 3D mesh (GLB for viewing, STL for printing).

  POST /image  {prompt, seed?}          Z-Image-Turbo (open, Apache-2.0), 1024x1024 -> PNG (base64)
  POST /shape  {image (b64 PNG/JPG)}    background removal + Hunyuan3D-2 mini     -> GLB + STL (base64)
Run: gen/bin/python -m uvicorn gen_server:app --host 127.0.0.1 --port 8791  (reach it through an SSH tunnel)
"""
import base64
import io
import os
import sys
import tempfile
import time

import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, "/workspace/h/hy3d")
# the container sees 96 cores but has a ~10 CPU quota: 96 threads get throttled hard (rembg took 98 s)
NCPU = int(os.environ.get("HOLO_CPUS", "10"))
torch.set_num_threads(NCPU)
app = FastAPI()
_flux = None
_shape = None
_rembg = None


IMAGE_MODEL = os.environ.get("HOLO_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")


def flux():
    """Text-to-image (open weights): Z-Image-Turbo by default (Apache-2.0, 8 steps); FLUX.1-schnell if available."""
    global _flux
    if _flux is None:
        from diffusers import DiffusionPipeline
        # straight onto the GPU (~21 GB VRAM): CPU offload would hold ~20 GB of host RAM and the container is capped at 31 GB
        _flux = DiffusionPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    return _flux


class no_init:
    """Every weight comes from the checkpoint: skip torch / transformers random init (it was ~25 s of a ~37 s load)."""
    NAMES = ("uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_",
             "xavier_normal_", "constant_", "zeros_", "ones_", "orthogonal_")

    def __enter__(self):
        import transformers.modeling_utils as mu
        self.saved = {n: getattr(torch.nn.init, n) for n in self.NAMES}
        for n in self.NAMES:
            setattr(torch.nn.init, n, lambda t, *a, **k: t)
        self.mu, self.iw = mu, mu.PreTrainedModel.init_weights
        mu.PreTrainedModel.init_weights = lambda self_: None
        self.ru = torch.Tensor.uniform_
        torch.Tensor.uniform_ = lambda t, *a, **k: t

    def __exit__(self, *e):
        for n, f in self.saved.items():
            setattr(torch.nn.init, n, f)
        self.mu.PreTrainedModel.init_weights = self.iw
        torch.Tensor.uniform_ = self.ru


def shape():
    global _shape, _rembg
    if _shape is None:
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        with no_init():
            _shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", use_safetensors=True)
        import onnxruntime as ort
        from rembg import new_session
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = NCPU, 1
        _rembg = BackgroundRemover.__new__(BackgroundRemover)
        _rembg.session = new_session("bria-rmbg", sess_opts=so, providers=["CPUExecutionProvider"])
    return _shape, _rembg


class ImgReq(BaseModel):
    prompt: str
    seed: int = 7
    size: int = 1024


class ShapeReq(BaseModel):
    image: str
    steps: int = 30
    octree: int = 256


@app.get("/health")
def health():
    return {"ok": True}


def free(which):
    """Container RAM is capped (31 GB): keep only one big model loaded at a time."""
    import gc
    global _flux, _shape, _rembg
    if which == "image" and _flux is not None:
        _flux = None
    if which == "shape" and _shape is not None:
        _shape, _rembg = None, None
    gc.collect()
    torch.cuda.empty_cache()


@app.post("/image")
def image(r: ImgReq):
    free("shape")
    t = time.time()
    g = torch.Generator("cpu").manual_seed(r.seed)
    steps = 4 if "schnell" in IMAGE_MODEL else 9
    im = flux()(prompt=r.prompt, guidance_scale=0.0, num_inference_steps=steps, height=r.size, width=r.size, generator=g).images[0]
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return {"image": base64.b64encode(buf.getvalue()).decode(), "seconds": round(time.time() - t, 1)}


@app.post("/shape")
def mesh(r: ShapeReq):
    import trimesh
    free("image")
    t = time.time()
    pipe, rem = shape()
    print(f"shape model ready in {time.time() - t:.1f}s", flush=True)
    img = Image.open(io.BytesIO(base64.b64decode(r.image))).convert("RGB")
    t1 = time.time(); rgba = rem(img); print(f"rembg {time.time()-t1:.1f}s", flush=True)
    t1 = time.time()
    m = pipe(image=rgba, num_inference_steps=r.steps, octree_resolution=r.octree, num_chunks=20000,
             generator=torch.manual_seed(7), output_type="trimesh")[0]
    print(f"dit+decode {time.time()-t1:.1f}s", flush=True); t1 = time.time()
    # clean up for printing: largest component, fix normals, centre on the build plate
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: len(p.faces))
    m.fix_normals()
    m.apply_translation(-m.bounds.mean(0))
    print(f"cleanup {time.time()-t1:.1f}s", flush=True)
    with tempfile.TemporaryDirectory() as d:
        glb, stl = os.path.join(d, "m.glb"), os.path.join(d, "m.stl")
        m.export(glb)
        m.export(stl)
        out = {"glb": base64.b64encode(open(glb, "rb").read()).decode(), "stl": base64.b64encode(open(stl, "rb").read()).decode()}
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    out.update({"cutout": base64.b64encode(buf.getvalue()).decode(), "faces": int(len(m.faces)),
                "watertight": bool(m.is_watertight), "extents": [round(float(x), 4) for x in m.extents],
                "seconds": round(time.time() - t, 1)})
    return out
