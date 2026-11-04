"""Holo automatic render: no per-video knowledge. Everything comes from Muse's plan + the models.

  before you ask   whatever a hand holds (HOI-DETR link -> SAM 3.1 box mask) glows blue
  plan             Muse's steps (pick / place / target labels) from the frame where you asked
  step state       pick X   -> amber on X until HOI-DETR / SAM 3.1 see X in your hand
                   find Y   -> magenta on Y, done once SAM 3.1 sees Y steadily
                   X into Y -> magenta on Y, done when X touches Y or HOI-DETR links X acting on Y
                   (a later step that is already satisfied skips the earlier ones)
  look-alikes      when several instances match Y (e.g. ports), keep one identity with DINOv3 patch
                   descriptors + where the held object is heading
  look             outline glow only (SAM 3 demo style), role colours; frosted step card

Usage: python auto_render.py --frames RAW --auto AUTO_DIR --plan plan.json --hoi pov.json --out DIR [--only 1,2]
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
from holo_render import ROLE, highlight, up

S = 2


def key(label):
    return label.replace(" ", "_")


class Masks:
    def __init__(self, auto):
        self.auto = auto
        self.idx = {}
        for d in glob.glob(f"{auto}/masks/*"):
            lab = os.path.basename(d)
            for f in glob.glob(f"{d}/*.png"):
                n = int(os.path.basename(f)[:4])
                self.idx.setdefault((lab, n), []).append(f)
        self.held = {}
        for f in glob.glob(f"{auto}/held/*.png"):
            self.held.setdefault(int(os.path.basename(f)[:4]), []).append(f)
        self.dino = {}
        for f in glob.glob(f"{auto}/dino/*.npz"):
            z = np.load(f)
            self.dino.update({k: v.astype(np.float32) for k, v in zip(z["keys"], z["feats"])})

    def inst(self, label, n):
        return [(f, cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127) for f in sorted(self.idx.get((key(label), n), []))]

    def union(self, label, n, shape):
        u = np.zeros(shape, bool)
        for _, m in self.inst(label, n):
            u |= m
        return u

    def held_mask(self, n, shape):
        u = np.zeros(shape, bool)
        for f in self.held.get(n, []):
            u |= cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127
        return u


def overlap(a, b):
    return (a & b).sum() / max(1, min(a.sum(), b.sum()))


def touching(a, b, px=4):
    if not a.any() or not b.any():
        return False
    return bool((cv2.dilate(a.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0)[b].any())


def step_kind(s):
    place = s.get("place") if isinstance(s.get("place"), str) else None
    con = s.get("connect") if isinstance(s.get("connect"), list) and len(s.get("connect")) == 2 else None
    pick = s.get("pick") if isinstance(s.get("pick"), str) else None
    if place or con:
        return "place", (pick or (con[0] if con else None)), (place or con[1])
    if pick:
        return "pick", pick, None
    return "find", s.get("target_label"), None


class Guide:
    def __init__(self, plan, masks, hoi):
        self.steps = plan["steps"]
        self.kinds = [step_kind(s) for s in self.steps]
        self.M, self.hoi = masks, hoi
        self.i = 0
        self.streak = 0
        self.done_at = None
        self.ref = None          # (descriptor, centroid) of the destination instance

    def dest_instance(self, label, n, mover):
        cands = self.M.inst(label, n)
        if not cands:
            return None
        def score(c):
            f, m = c
            ys, xs = np.where(m)
            cen = np.array([xs.mean(), ys.mean()])
            s = 0.0
            d = self.M.dino.get(os.path.basename(f))
            if self.ref is not None and d is not None:
                s += 2.0 * float(d @ self.ref[0])                  # same-looking instance (DINOv3)
            if self.ref is not None:
                s -= np.linalg.norm(cen - self.ref[1]) / 120.0     # temporal continuity
            if mover is not None and mover.any():
                my, mx = np.where(mover)
                tipd = np.min(np.hypot(mx - cen[0], my - cen[1]))  # where the held object is heading
                s -= tipd / 80.0
            return s
        f, m = max(cands, key=score)
        ys, xs = np.where(m)
        cen = np.array([xs.mean(), ys.mean()])
        d = self.M.dino.get(os.path.basename(f))
        if d is not None:
            self.ref = (d if self.ref is None else (0.8 * self.ref[0] + 0.2 * d) / np.linalg.norm(0.8 * self.ref[0] + 0.2 * d), cen)
        return m

    def satisfied(self, k, n, held, shape):
        kind, a, b = self.kinds[k]
        if kind == "pick" and a:
            m = self.M.union(a, n, shape)
            return m.any() and held.any() and overlap(m, held) > 0.25
        if kind == "find" and a:
            return self.M.union(a, n, shape).sum() > 15
        if kind == "place" and b:
            # inserted = the destination gets covered by the held object's tip: the tracked destination
            # instance vanishes (or shrinks a lot) exactly where the mover's tip now is
            mover = self.M.union(a, n, shape) if a else held
            dest = self.dest_instance(b, n, mover)
            if dest is not None and dest.any():
                ys, xs = np.where(dest)
                area = dest.sum()
                self.dest_hist = (getattr(self, "dest_hist", []) + [area])[-12:]
                self.dest_last = np.array([xs.mean(), ys.mean()])
                big = np.median(self.dest_hist)
                covered = area < 0.5 * big
            else:
                covered = getattr(self, "dest_last", None) is not None
            if not mover.any() or getattr(self, "dest_last", None) is None:
                return False
            my, mx = np.where(mover)
            tip = np.min(np.hypot(mx - self.dest_last[0], my - self.dest_last[1]))
            fr = self.hoi[n - 1] if n - 1 < len(self.hoi) else None
            acting = bool(fr and fr["fs"])                      # HOI-DETR: held object acting on a second object
            return (covered and tip < 12) or (acting and tip < 20)
        return False

    def update(self, n, held, shape):
        if self.i >= len(self.steps):
            return
        for k in range(len(self.steps) - 1, self.i - 1, -1):     # a later satisfied step skips earlier ones
            if self.satisfied(k, n, held, shape):
                self.streak = self.streak + 1 if k == self.i or k > self.i else 1
                if self.streak >= 3:
                    self.i, self.streak = k + 1, 0
                    if self.i >= len(self.steps):
                        self.done_at = n
                return
        self.streak = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--auto", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--hoi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.load(open(a.plan))
    ask = int(plan.get("_asked_at_frame", 1))
    M = Masks(a.auto)
    hoi = json.load(open(a.hoi))["frames"]
    gd = Guide(plan, M, hoi)
    only = {int(x) for x in a.only.split(",") if x}
    picks = [k[1] for k in gd.kinds if k[0] == "place" and k[1]] + [k[1] for k in gd.kinds if k[0] == "pick" and k[1]]
    strength, prev, card_a = {}, {}, 0.0
    log = []
    for fp in sorted(Path(a.frames).glob("*.jpg")):
        n = int(fp.stem)
        src = cv2.imread(str(fp))
        h0, w0 = src.shape[:2]
        held = M.held_mask(n, (h0, w0))
        if not M.held:      # no HOI-DETR run: held = a planned pick object whose SAM 3.1 mask touches a hand
            hm = M.union("hand", n, (h0, w0))
            for lab in picks:
                for _, m in M.inst(lab, n):
                    if m.sum() < 0.35 * h0 * w0 and touching(m, hm, 6):
                        held |= m
        if n > ask:
            before = gd.i
            gd.update(n, held, (h0, w0))
            if gd.i != before:
                log.append((n, before, gd.i))
        if only and not any(0 <= o - n <= 4 for o in only):
            continue
        w, h = w0 * S, h0 * S
        img = cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        t = n / a.fps
        hand1 = M.union("hand", n, (h0, w0))
        hs = cv2.imread(f"{a.auto}/../handseg/{n:04d}.png", cv2.IMREAD_GRAYSCALE) if False else None
        hand = up(hand1, w, h, 1.0)

        layers = {}                                              # name -> (mask, role)
        if held.any():
            layers["held"] = (held, "held")
        if n > ask:
            if gd.done_at is not None and n - gd.done_at < 30:
                k, b = gd.kinds[-1][1], gd.kinds[-1][2]
                if b:
                    dm = gd.dest_instance(b, n, M.union(k, n, (h0, w0)) if k else held)
                    if dm is not None:
                        layers["dest"] = (dm, "done")
                if held.any():
                    layers["held"] = (held, "done")
            elif gd.i < len(gd.steps):
                kind, x, y = gd.kinds[gd.i]
                if kind == "pick" and x:
                    m = M.union(x, n, (h0, w0)) & ~held
                    if m.any():
                        layers["target"] = (m, "grab")
                elif kind == "find" and x:
                    m = M.union(x, n, (h0, w0))
                    if m.any():
                        layers["target"] = (m, "dest")
                elif kind == "place" and y:
                    mover = M.union(x, n, (h0, w0)) if x else held
                    dm = gd.dest_instance(y, n, mover)
                    if dm is not None:
                        layers["dest"] = (dm, "dest")
                    if mover.any():
                        layers["held"] = (mover | held, "held")
        for name in list(strength):
            if name not in layers:
                strength[name] = max(0.0, strength[name] - 0.25)
        for name, (m, role) in layers.items():
            strength[name] = 1.0 if only else min(1.0, strength.get(name, 0.0) + 0.3)
        for name in list(strength):
            st = strength[name]
            if name not in layers or st <= 0.01:
                prev.pop(name, None)
                continue
            m, role = layers[name]
            soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.7)
            if name in prev and prev[name].shape == soft.shape:
                soft = 0.6 * soft + 0.4 * prev[name]
            prev[name] = soft
            pulse = 0.75 + 0.35 * np.sin(t * 4.5) if role in ("grab", "dest") else 1.0
            img = highlight(img, up(soft, w, h), hand, ROLE[role], st, pulse)

        # frosted step card: Muse's own instruction + learn tip; progress = steps completed
        show = n > ask
        card_a = (1.0 if only else min(1.0, card_a + 0.2)) if show else max(0.0, card_a - 0.2)
        if card_a > 0:
            done = gd.done_at is not None
            si = min(gd.i, len(gd.steps) - 1)
            s = gd.steps[si]
            title, tip = s["instruction"], s.get("learn") or ""
            prog = gd.i / len(gd.steps)
            if done:
                title, tip, prog = "Done", "Holo saw every step happen", 1.0
            img, dr = GH.card(img, w, si + 1, len(gd.steps), title, tip, prog, card_a, done)
            pil = G.to_pil(img)
            dr(ImageDraw.Draw(pil, "RGBA"))
            img = G.from_pil(pil)
        cv2.imwrite(str(out / fp.name), np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
    json.dump({"asked_at": ask, "step_changes": log, "done_at": gd.done_at}, open(out / "guide_log.json", "w"), indent=1)
    print("step changes (frame, from, to):", log, "done at", gd.done_at)


if __name__ == "__main__":
    main()
