"""Glassmorphism UI kit (Meta Horizon OS style) for the Holo video renders.

Frosted panels are real background blur (the scene behind shows through, softened), a faint white
veil, a thin light border, a soft top highlight and a gentle drop shadow. Text is Inter (variable),
light for titles and medium for labels, near-white.

Usage pattern per frame:
    img = glass(img, box, r)            # numpy BGR float/uint8, any number of panels
    pil = to_pil(img); d = ImageDraw.Draw(pil, "RGBA"); text(d, ...); img = from_pil(pil)
"""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "./HoloXR/Assets/Holo/UI/Fonts/Inter.ttf"
WHITE = (246, 248, 252)
SOFT = (205, 214, 228)
DIM = (160, 172, 190)
BLUE = (110, 190, 255)          # accent (RGB)


@lru_cache(maxsize=64)
def font(px: int, weight: str = "Regular"):
    f = ImageFont.truetype(FONT, int(px))
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


@lru_cache(maxsize=256)
def _rr_mask(w: int, h: int, r: int, ss: int = 4):
    """Anti-aliased rounded-rect alpha (supersampled)."""
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=r * ss, fill=255)
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0


@lru_cache(maxsize=256)
def _rr_ring(w: int, h: int, r: int, t: float, ss: int = 4):
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=r * ss, outline=255, width=max(1, int(t * ss)))
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def glass(img, box, r, alpha=1.0, veil=0.10, darken=0.22, blur=14, border=0.42, shadow=0.35, tint=(255, 255, 255)):
    """Frosted-glass panel at box=(x0,y0,x1,y1) on a BGR image. Returns float32 image."""
    out = img.astype(np.float32) if img.dtype != np.float32 else img
    H, W = out.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    w, h = x1 - x0, y1 - y0
    if w < 4 or h < 4 or alpha <= 0.01:
        return out
    r = int(min(r, h // 2, w // 2))
    # soft shadow
    if shadow > 0:
        pad = int(r * 0.9) + 6
        sx0, sy0, sx1, sy1 = max(0, x0 - pad), max(0, y0 - pad + pad // 3), min(W, x1 + pad), min(H, y1 + pad + pad // 3)
        sm = np.zeros((sy1 - sy0, sx1 - sx0), np.float32)
        m = _rr_mask(w, h, r)
        oy, ox = y0 - sy0 + pad // 3, x0 - sx0
        hh, ww = min(h, sm.shape[0] - oy), min(w, sm.shape[1] - ox)
        if hh > 0 and ww > 0:
            sm[oy:oy + hh, ox:ox + ww] = m[:hh, :ww]
            sm = cv2.GaussianBlur(sm, (0, 0), pad / 2.2) * shadow * alpha
            out[sy0:sy1, sx0:sx1] *= (1 - sm[..., None])
    # frosted body
    m = 3 * blur
    bx0, by0, bx1, by1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    region = cv2.GaussianBlur(out[by0:by1, bx0:bx1], (0, 0), blur)
    body = region[y0 - by0:y0 - by0 + h, x0 - bx0:x0 - bx0 + w]
    body = body * (1 - darken) + np.array(tint[::-1], np.float32) * veil
    grad = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None, None] ** 2.2   # soft top highlight
    body = body + 18 * grad
    a = _rr_mask(w, h, r)[..., None] * alpha
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + body * a
    if border > 0:
        ring = _rr_ring(w, h, r, 1.3)[..., None] * border * alpha
        # brighter on top edge, fainter at the bottom (light from above)
        ring = ring * (0.55 + 0.45 * np.linspace(1, 0, h, dtype=np.float32)[:, None, None])
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - ring) + 255 * ring
    return out


def to_pil(img):
    return Image.fromarray(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))


def from_pil(pil):
    return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR).astype(np.float32)


def text(d, xy, s, px, color=WHITE, weight="Regular", anchor="lm", alpha=1.0):
    d.text(xy, s, font=font(px, weight), fill=tuple(color) + (int(255 * alpha),), anchor=anchor)


def tw(s, px, weight="Regular"):
    return font(px, weight).getlength(s)


def dot(d, xy, r, color=BLUE, alpha=1.0, glow=True):
    x, y = xy
    if glow:
        for k, a in ((2.6, 0.10), (1.8, 0.18)):
            d.ellipse([x - r * k, y - r * k, x + r * k, y + r * k], fill=tuple(color) + (int(255 * a * alpha),))
    d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(color) + (int(255 * alpha),))


def wrap(s, px, width, weight="Regular"):
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if tw(t, px, weight) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines
