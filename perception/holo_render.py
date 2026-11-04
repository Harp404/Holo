"""Holo render v4: SAM 3.1 masks, role-coloured highlights (SAM 3 demo look), only what matters is lit.

Masks      SAM 3.1 (Object Multiplex) text-prompted tracking: "smartphone", "white USB cable", "USB plug",
           "USB port" (every port instance), "hand". Temporally smoothed, 2x anti-aliased.
Roles      colour = what the object means right now
             held         blue     (HOI-DETR hand->object link, or the hand touching it)
             grab target  amber    (step 1, breathing, until the hand takes it)
             destination  magenta  (the port the plug goes into; picked as the port the plug actually
                                    enters, then followed frame to frame)
             done         green    (HOI-DETR sees the plug acting on the laptop)
             muse         violet   (the object Muse is talking about)
           anything not relevant right now is not highlighted. No name pills.
Look       soft same-colour tint + crisp coloured edge + outer glow; your hands stay in front.
UI         frosted-glass step card (Muse's steps + learn tips), progress bar.

Usage: python holo_render.py --frames RAW --sam masks_dir --out DIR [--only 140,600]
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageDraw

import guide_holo as GH
import ui_glass as G

S = 2
ROLE = {  # BGR
    "held": np.array([255, 170, 70], np.float32),
    "grab": np.array([60, 200, 255], np.float32),
    "dest": np.array([200, 80, 255], np.float32),
    "done": np.array([120, 225, 90], np.float32),
    "muse": np.array([255, 110, 160], np.float32),
}
HOI_JSON = "./media/pov/hoidetr/pov.json"
MUSE_FRAMES = {342: ["white_USB_cable"], 600: ["USB_plug"], 560: ["USB_port"]}   # what Muse is talking about


def masks_for(sam, concept, n):
    return [cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127 for f in sorted(glob.glob(f"{sam}/{concept}/{n:04d}_*.png"))]


def union(ms, shape):
    u = np.zeros(shape, bool)
    for m in ms:
        u |= m
    return u


def up(m_soft, w, h, blur=1.2):
    u = cv2.resize(m_soft.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    u = cv2.GaussianBlur(u, (0, 0), blur * S)
    u = np.clip((u - 0.35) / 0.3, 0, 1)
    return u * u * (3 - 2 * u)


def highlight(out, m, hand, col, strength=1.0, pulse=1.0):
    """SAM 3 demo look: soft tint + crisp coloured edge + halo outside the edge; hands stay in front."""
    if m.max() < 0.05 or strength <= 0.01:
        return out
    keep = 1 - hand
    # no fill: the object itself stays fully visible, only its outline glows
    band = np.clip(1 - np.abs(m - 0.5) / 0.5, 0, 1) ** 0.7 * keep
    e = 3 * S
    band[:e] = band[-e:] = 0
    band[:, :e] = band[:, -e:] = 0
    la = np.clip(band * strength, 0, 1)
    edge_col = np.minimum(255, col * 0.8 + 75)
    out = out * (1 - la[..., None]) + edge_col * la[..., None]
    # halo: light that spills outside the object edge
    halo = np.clip(cv2.GaussianBlur(band, (0, 0), 4 * S) * 2.4, 0, 1) * keep
    halo2 = np.clip(cv2.GaussianBlur(band, (0, 0), 11 * S) * 2.2, 0, 1) * (1 - 0.5 * m) * keep
    g = (1.25 * halo + 0.8 * halo2) * strength * pulse
    return out * (1 - 0.25 * g[..., None]) + g[..., None] * col * 0.9


def hoi_holds(fr, mask):
    if fr is None or not mask.any():
        return False
    ys, xs = np.where(mask)
    mb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    marea = max(1.0, (mb[2] - mb[0]) * (mb[3] - mb[1]))
    for l in fr["hf"]:
        if l["prob"] < 0.5:
            continue
        b = np.array(fr["detections"][l["b"]]["box"], float)
        ix = max(0, min(b[2], mb[2]) - max(b[0], mb[0]))
        iy = max(0, min(b[3], mb[3]) - max(b[1], mb[1]))
        barea = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        if ix * iy / marea > 0.3 and ix * iy / barea > 0.1:
            return True
    return False


def touches(mask, hand1):
    k = np.ones((7, 7), np.uint8)
    return int((cv2.dilate(hand1.astype(np.uint8), k) > 0)[mask].sum()) > 25


def pick_ports(sam, _unused, n_ref, frames):
    """Destination port = the port instance the plug tip actually reaches (closest approach over the
    insertion window, before the plug covers it); then followed strictly frame to frame. While the plug
    covers it, the last seen mask is held."""
    def cents(n):
        out = []
        for f in sorted(glob.glob(f"{sam}/USB_port/{n:04d}_*.png")):
            m = cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127
            if m.sum() > 8:
                ys, xs = np.where(m)
                out.append((np.array([xs.mean(), ys.mean()]), m))
        return out

    def plug_tip(n):
        p = union(masks_for(sam, "USB_plug", n), (480, 854))
        if not p.any():
            return None
        ys, xs = np.where(p)
        k = np.argmin(xs)
        return np.array([xs[k], ys[k]], float)            # end of the plug towards the laptop

    best = None
    for n in range(max(min(frames), n_ref - 90), n_ref + 1):
        tip, cs = plug_tip(n), cents(n)
        if tip is None or not cs:
            continue
        for c, m in cs:
            dist = np.linalg.norm(c - tip)
            if best is None or dist < best[0]:
                best = (dist, n, c, m)
    if best is None:
        return {}
    _, nref, ref, mref = best
    chosen = {nref: mref}
    for direction in (-1, 1):
        cur, last = ref.copy(), mref
        n, miss = nref + direction, 0
        while n in frames:
            c = cents(n)
            near = [q for q in c if np.linalg.norm(q[0] - cur) < 12]
            if near:
                q = min(near, key=lambda q: np.linalg.norm(q[0] - cur))
                chosen[n] = q[1]
                cur, last, miss = 0.5 * cur + 0.5 * q[0], q[1], 0
            else:                              # covered or lost: show nothing rather than a stale outline
                miss += 1
            n += direction
    print("destination port chosen at frame", nref, "centroid", ref.round(1))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--sam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {int(x) for x in a.only.split(",") if x}
    hoi = json.load(open(HOI_JSON))["frames"]
    steps = [list(s) for s in GH.STEPS]
    done_at = GH.DONE_AT
    fs = [f["frame_idx"] + 1 for f in hoi if f["fs"] and f["frame_idx"] + 1 >= 600]
    if fs:
        done_at = fs[0]
    grab = [f["frame_idx"] + 1 for f in hoi if f["frame_idx"] + 1 >= steps[0][0] + 8 and any(l["prob"] > 0.9 for l in f["hf"])]
    grab_at = grab[0] if grab else 372
    steps[0][1], steps[1][0] = grab_at + 6, grab_at + 7
    frames = {int(Path(p).stem) for p in glob.glob(f"{a.frames}/*.jpg")}
    dest = pick_ports(a.sam, None, min(done_at - 4, 686), frames)
    print("grab at", grab_at, "done at", done_at, "destination port frames", len(dest))

    prev, strength = {}, {}
    card_a = 0.0
    for fp in sorted(Path(a.frames).glob("*.jpg")):
        n = int(fp.stem)
        if only and not any(0 <= o - n <= 6 for o in only):
            continue
        src = cv2.imread(str(fp))
        h0, w0 = src.shape[:2]
        w, h = w0 * S, h0 * S
        img = cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        t = n / a.fps
        hand1 = union(masks_for(a.sam, "hand", n), (h0, w0))
        hs = cv2.imread(f"./media/pov/handseg/{n:04d}.png", cv2.IMREAD_GRAYSCALE)
        if hs is not None:
            hand1 |= hs > 127
        hand = up(hand1, w, h, 1.0)
        step = next((i for i, s in enumerate(steps) if s[0] <= n <= s[1]), None)
        done = n >= done_at
        fr = hoi[n - 1] if n - 1 < len(hoi) else None

        objs = {c: union(masks_for(a.sam, c, n), (h0, w0)) for c in ("smartphone", "white_USB_cable", "USB_plug")}
        objs["USB_port"] = dest.get(n, np.zeros((h0, w0), bool)) if step == 2 or done else np.zeros((h0, w0), bool)
        roles = {}
        for c, m in objs.items():
            if not m.any():
                continue
            if c == "USB_port":
                roles[c] = "done" if done else "dest"
            elif n in MUSE_FRAMES and c in MUSE_FRAMES[n]:
                roles[c] = "muse"
            elif step == 0 and c == "white_USB_cable" and n < grab_at:
                roles[c] = "grab"
            elif done and c == "USB_plug":
                roles[c] = "done"
            elif (hoi_holds(fr, m) or touches(m, hand1)) and not (c == "white_USB_cable" and objs["USB_plug"].any()):
                roles[c] = "held"
        if n in MUSE_FRAMES:
            for c in MUSE_FRAMES[n]:
                if c in objs and objs[c].any():
                    roles[c] = "muse"
        for c in list(strength):
            if c not in roles:
                strength[c] = max(0.0, strength[c] - 0.25)
        for c, m in objs.items():
            if c in roles:
                strength[c] = 1.0 if only else min(1.0, strength.get(c, 0.0) + 0.3)
            st = strength.get(c, 0.0)
            if st <= 0.01 or not m.any():
                prev.pop(c, None)
                continue
            soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.7)
            if c in prev and prev[c].shape == soft.shape:
                soft = 0.6 * soft + 0.4 * prev[c]
            prev[c] = soft
            role = roles.get(c, "held")
            pulse = 0.75 + 0.35 * np.sin(t * 4.5) if role in ("grab", "dest") else 1.0
            img = highlight(img, up(soft, w, h), hand, ROLE[role], st, pulse)

        # ---- glass step card ----
        if step is not None or done:
            card_a = 1.0 if only else min(1.0, card_a + 0.2)
        else:
            card_a = max(0.0, card_a - 0.2)
        draws = []
        if card_a > 0:
            si = step if step is not None else 2
            s0, s1, title, tip = steps[si]
            prog = (si + min(1, (n - s0) / max(1, s1 - s0))) / len(steps)
            if done:
                title, tip, prog = "Cable connected", "Holo saw the plug go into the port", 1.0
            img, dr = GH.card(img, w, si + 1, len(steps), title, tip, prog, card_a, done)
            draws.append(dr)
        if draws:
            pil = G.to_pil(img)
            d = ImageDraw.Draw(pil, "RGBA")
            for dr in draws:
                dr(d)
            img = G.from_pil(pil)
        cv2.imwrite(str(out / fp.name), np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
    print("done", out)


if __name__ == "__main__":
    main()
