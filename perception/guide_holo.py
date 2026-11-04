"""Layer 3 - guidance over the perception video (2x frames from compose_holo.py).

  step card     top-centre frosted-glass card (Horizon style): step i/N, instruction, learn tip, progress bar
  grab cue      the object to pick up gets a breathing highlight + "Grab here" glass tag until the hand
                takes it (HOI-DETR hand->object link ends the step)
  target        place location (USB port) tracked through camera motion (LK flow + similarity on background),
                pulsing ring + progress arc; dashed path from the plug tip to the port that shortens as you go
  done          ring turns solid with a check, card shows the completed state

Usage: python guide_holo.py --base out/final_v12 --raw RAW --layer meta_layer --handseg handseg --objects objects --out out/final_v13
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageDraw

import ui_glass as G

S = 2
FONT = "./HoloXR/Assets/Holo/UI/Fonts/Inter.ttf"

# ---- the task, timed to this clip -------------------------------------------------------------
# steps + learn tips come from Muse Spark's plan for this scene (media/pov/muse/plan_0345.json)
STEPS = [
    # (first frame, last frame, title, learn tip)
    (343, 399, "Grip the USB plug end", "USB-A only fits one way: the flat side lines up with the port"),
    (400, 507, "Find the USB port on the laptop's side", "Laptop USB ports sit on the left or right edge, not the keyboard"),
    (508, 690, "Insert the USB plug into the port", "This makes the data and power connection to your laptop"),
]
DONE_AT = 683          # fallback; replaced by HOI-DETR's "plug acts on laptop" link below
PORT_REF = (686, (497.0, 212.0))   # (frame, port position in that frame, px at 1x)
GRAB_OBJECT = 2       # tracker id of the cable during step 1


def load_layer(path_k, path_w):
    k = cv2.imread(str(path_k)).astype(np.float32)
    w = cv2.imread(str(path_w)).astype(np.float32)
    a = np.clip(1 - (w - k).mean(2) / 255.0, 0, 1)
    c = k / np.maximum(a[..., None], 1e-3)
    return np.clip(c, 0, 255), a


def track_port(raw_dir, handseg_dir, n0, p0, n_from, n_to):
    """Propagate a point through camera motion with LK flow on background features (hands masked out)."""
    def gray(n):
        return cv2.cvtColor(cv2.imread(str(raw_dir / f"{n:04d}.jpg")), cv2.COLOR_BGR2GRAY)

    def bgmask(n):
        hs = cv2.imread(str(handseg_dir / f"{n:04d}.png"), cv2.IMREAD_GRAYSCALE)
        m = np.full(gray(n).shape, 255, np.uint8)
        if hs is not None:
            m[cv2.dilate((hs > 60).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0] = 0
        return m

    pos = {n0: np.array(p0, np.float64)}
    for direction in (-1, 1):
        n, p = n0, np.array(p0, np.float64)
        end = n_from if direction < 0 else n_to
        while n != end:
            nn = n + direction
            g0, g1 = gray(n), gray(nn)
            pts = cv2.goodFeaturesToTrack(g0, 400, 0.01, 7, mask=bgmask(n))
            if pts is not None and len(pts) >= 8:
                q, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=3)
                good = st.reshape(-1) == 1
                if good.sum() >= 8:
                    M, _ = cv2.estimateAffinePartial2D(pts[good], q[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is not None:
                        p = M[:, :2] @ p + M[:, 2]
            pos[nn] = p.copy()
            n = nn
    return pos



def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def short(t, n=7):
    """Short, easy step text: first clause, at most n words."""
    import re
    t = re.split(r"[;,(\u2014]| then | and then ", t or "")[0].strip().rstrip(".")
    w = t.split()
    return " ".join(w[:n]) + ("\u2026" if len(w) > n else "")


def card(img, W, i, n, title, tip, prog, A, done):
    """Frosted step card, top-left corner (keeps the middle of the view clear). Text is wrapped to fit inside."""
    title, tip = short(title, 7), short(tip, 9)
    inner = 250 * S
    tl = G.wrap(title, 17 * S, inner, "Medium")[:2]
    pl = G.wrap(tip, 11 * S, inner)[:2] if tip else []
    pw = 72 * S + inner + 18 * S
    ph = 22 * S + 22 * S * len(tl) + 15 * S * len(pl) + 14 * S
    x0, y0 = 18 * S, 18 * S
    img = G.glass(img, (x0, y0, x0 + pw, y0 + ph), 20 * S, alpha=A, blur=16 * S, darken=0.30, veil=0.08)

    def draw(d):
        a = lambda v: int(v * A)  # noqa: E731
        bx, by, r = x0 + 36 * S, y0 + 22 * S + 11 * S * len(tl) - 2 * S, 16 * S
        if done:
            d.ellipse([bx - r, by - r, bx + r, by + r], fill=(110, 200, 255, a(255)))
            d.line([(bx - 7 * S, by), (bx - 2 * S, by + 6 * S), (bx + 8 * S, by - 6 * S)], fill=(255, 255, 255, a(255)), width=3 * S)
        else:
            d.ellipse([bx - r, by - r, bx + r, by + r], outline=(235, 242, 255, a(200)), width=max(1, int(1.5 * S)))
            G.text(d, (bx, by), f"{i}/{n}", 11 * S, G.WHITE, "Medium", anchor="mm", alpha=A)
        y = y0 + 22 * S
        for ln in tl:
            G.text(d, (x0 + 68 * S, y), ln, 17 * S, G.WHITE, "Medium", alpha=A); y += 22 * S
        y += 2 * S
        for ln in pl:
            G.text(d, (x0 + 68 * S, y), ln, 11 * S, G.SOFT, alpha=A * 0.95); y += 15 * S
        bx0, bx1, byy = x0 + 18 * S, x0 + pw - 18 * S, y0 + ph - 7 * S
        d.rounded_rectangle([bx0, byy - S, bx1, byy + S], radius=S, fill=(255, 255, 255, a(45)))
        if prog > 0:
            d.rounded_rectangle([bx0, byy - S, bx0 + (bx1 - bx0) * min(1, prog), byy + S], radius=S, fill=(150, 210, 255, a(255)))
    return img, draw


def chip(img, xy, label, A, accent=True):
    x, y = xy
    pw, ph = G.tw(label, 14 * S, "Medium") + 40 * S, 30 * S
    x0, y0 = x - pw / 2, y - ph - 16 * S
    img = G.glass(img, (x0, y0, x0 + pw, y0 + ph), ph / 2, alpha=A, blur=10 * S, darken=0.28)

    def draw(d):
        G.dot(d, (x0 + 15 * S, y0 + ph / 2), 3.6 * S, alpha=A, glow=accent)
        G.text(d, (x0 + 26 * S, y0 + ph / 2), label, 14 * S, G.WHITE, "Medium", alpha=A)
        d.line([(x, y0 + ph), (x, y - 4 * S)], fill=(225, 238, 255, int(130 * A)), width=max(1, S))
    return img, draw


def target(d, c, prog, t, A, done):
    x, y = c
    r = 20 * S
    if done:
        d.ellipse([x - r, y - r, x + r, y + r], fill=(110, 200, 255, int(140 * A)), outline=(235, 245, 255, int(255 * A)), width=2 * S)
        d.line([(x - 7 * S, y), (x - 2 * S, y + 6 * S), (x + 9 * S, y - 6 * S)], fill=(255, 255, 255, int(255 * A)), width=3 * S)
        return
    pulse = 1 + 0.22 * (0.5 + 0.5 * np.sin(t * 5))
    R = r * pulse
    d.ellipse([x - R, y - R, x + R, y + R], outline=(150, 210, 255, int(90 * A)), width=max(1, S))
    d.ellipse([x - r * 0.62, y - r * 0.62, x + r * 0.62, y + r * 0.62], outline=(240, 246, 255, int(230 * A)), width=max(1, int(1.5 * S)))
    if prog > 0:
        d.arc([x - r, y - r, x + r, y + r], -90, -90 + 360 * min(1, prog), fill=(130, 205, 255, int(255 * A)), width=2 * S)
    d.ellipse([x - 2.5 * S, y - 2.5 * S, x + 2.5 * S, y + 2.5 * S], fill=(245, 250, 255, int(255 * A)))


def path(d, p, q, A):
    p, q = np.array(p, float), np.array(q, float)
    L = np.linalg.norm(q - p)
    if L < 34 * S:
        return
    u = (q - p) / L
    s_ = 0.0
    while s_ < L - 26 * S:
        a_, b_ = p + u * s_, p + u * min(s_ + 7 * S, L - 26 * S)
        d.line([tuple(a_), tuple(b_)], fill=(225, 240, 255, int(200 * A)), width=max(1, int(1.5 * S)))
        s_ += 13 * S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--layer", default="")
    ap.add_argument("--handseg", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw, hsd, O = Path(a.raw), Path(a.handseg), Path(a.objects)
    only = {int(x) for x in a.only.split(",") if x}

    global DONE_AT
    try:
        hoi = json.load(open("./media/pov/hoidetr/pov.json"))["frames"]
        fs = [f["frame_idx"] + 1 for f in hoi if f["fs"] and f["frame_idx"] + 1 >= 600]
        if fs:
            DONE_AT = fs[0]
        grab = [f["frame_idx"] + 1 for f in hoi if f["frame_idx"] + 1 >= STEPS[0][0] + 8 and any(l["prob"] > 0.9 for l in f["hf"])]
        if grab:
            g = grab[0] + 6
            STEPS[0] = (STEPS[0][0], g, STEPS[0][2], STEPS[0][3])
            STEPS[1] = (g + 1, STEPS[1][1], STEPS[1][2], STEPS[1][3])
            GRAB_END[0] = grab[0]
        print("HOI-DETR: grab at", grab[:1], "plug->laptop at", DONE_AT)
    except Exception as e:
        print("HOI-DETR timing unavailable:", e)
    port = track_port(raw, hsd, PORT_REF[0], PORT_REF[1], STEPS[2][0], 710)
    d0 = None
    card_a, grab_a = 0.0, 0.0
    for fp in sorted(Path(a.base).glob("*.jpg")):
        n = int(fp.stem)
        if only and n not in only:
            continue
        img = cv2.imread(str(fp)).astype(np.float32)
        H, W = img.shape[:2]
        t = n / a.fps
        draws = []
        step = next((i for i, s in enumerate(STEPS) if s[0] <= n <= s[1]), None)
        done = n >= DONE_AT

        # ---- grab cue on the object to pick up (until the hand takes it) ----
        want = 1.0 if (step == 0 and n < GRAB_END[0]) else 0.0
        grab_a = want if only else grab_a + (want - grab_a) * 0.3
        mp = O / "refined" / f"object{GRAB_OBJECT}" / f"{n:04d}.png"
        if grab_a > 0.02 and mp.exists():
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            mu = cv2.GaussianBlur(cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 1.5 * S)
            band = np.clip(1 - np.abs(mu - 0.5) / 0.35, 0, 1)
            breathe = 0.55 + 0.45 * np.sin(t * 4.5)
            g = cv2.GaussianBlur(band, (0, 0), 6 * S) * (0.7 + 0.6 * breathe) + band * 0.5
            img = img + np.clip(g * grab_a, 0, 1.2)[..., None] * np.array([255, 225, 170], np.float32) * 0.6
            ys, xs = np.where(m > 0.5)
            if len(xs):
                k = np.argmin(xs)                      # anchor on the cable's far end, away from its name tag
                img, dr = chip(img, (float(xs[k]) * S + 20 * S, float(ys[k]) * S), "Grab here", grab_a)
                draws.append(dr)

        # ---- place target: tracked USB port + path + progress ----
        if step == 2 or (done and n in port):
            pc = port[n] * S
            tip = None
            pm = O / "refined" / "object1" / f"{n:04d}.png"
            if pm.exists():
                m = cv2.imread(str(pm), cv2.IMREAD_GRAYSCALE) > 127
                ys, xs = np.where(m)
                if len(xs):
                    k = np.argmin((xs * S - pc[0]) ** 2 + (ys * S - pc[1]) ** 2)
                    tip = np.array([xs[k], ys[k]], float) * S
            prog = 0.0
            if tip is not None:
                dist = np.linalg.norm(tip - pc)
                d0 = dist if d0 is None else max(d0, dist)
                prog = 1 - dist / max(d0, 1)
            pr = 1.0 if done else prog
            tp = tip
            draws.append(lambda d, pc=pc, pr=pr, tp=tp: (path(d, tp, pc, 0.85) if (tp is not None and not done) else None, target(d, tuple(pc), pr, t, 1.0, done)))
            img, dr = chip(img, (pc[0] - 70 * S, pc[1] + 80 * S), "USB port", 1.0)
            draws.append(dr)

        # ---- step card ----
        if step is not None or done:
            card_a = 1.0 if only else min(1.0, card_a + 0.2)
        else:
            card_a = max(0.0, card_a - 0.2)
        if card_a > 0:
            si = step if step is not None else 2
            s0, s1, title, tip_ = STEPS[si]
            prog_bar = (si + min(1, (n - s0) / max(1, s1 - s0))) / len(STEPS)
            if done:
                title, tip_, prog_bar = "Cable connected", "Holo saw the plug go into the port", 1.0
            img, dr = card(img, W, si + 1, len(STEPS), title, tip_, prog_bar, card_a, done)
            draws.append(dr)

        pil = G.to_pil(img)
        d = ImageDraw.Draw(pil, "RGBA")
        for dr in draws:
            dr(d)
        cv2.imwrite(str(out / fp.name), np.clip(G.from_pil(pil), 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
    print("done", out)


GRAB_END = [372]

if __name__ == "__main__":
    main()
