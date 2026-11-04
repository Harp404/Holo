"""Assemble the demo video: title -> before/after -> full run with chapter chips -> how it works -> models -> end.

Usage: python make_demo.py --seq out/final_seq --raw RAW_FRAMES --out out/demo_frames
then:  ffmpeg -framerate 12 -i out/demo_frames/%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p Holo_demo.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1708, 960, 12
FONT = "./HoloXR/Assets/Holo/UI/Fonts/Inter.ttf"
BG = (8, 13, 24)
INK = (240, 247, 255)
SUB = (150, 200, 245)
ACC = (120, 220, 255)
EDGE = (160, 210, 255)

CHAPTERS = []  # filled from the Muse timeline in main()


def build_chapters(tl):
    """Chapters from the actual interactions (no fixed frames)."""
    ev, kinds = tl["events"], tl.get("kinds", {})
    names = {"plan": ("Ask Muse", "voice -> Muse Spark sees the scene and plans the task"),
             "search": ("Muse + web search", "Muse writes the query, reads the results, answers with sources"),
             "answer": ("Ask Muse mid-task", "grounded answer + SAM 3.1 highlight on the exact part")}
    ch = [(0, "Perception", "SAM 3.1 tracks every part Muse names")]
    for e, (s0, s1) in sorted(ev.items(), key=lambda x: x[1][0]):
        t, d = names.get(kinds.get(e, "answer"))
        ch.append((s0, t, d))
        ch.append((s1, "Guidance", "amber = grab \u00b7 blue = in hand \u00b7 magenta = where it goes \u00b7 green = done"))
    return sorted(ch)


import ui_glass as G
import cv2
import numpy as np

BG_IMG = None   # blurred, darkened frame of the user's own video (set in main)


def F(px, weight="Regular"):
    return G.font(int(px), weight)


def blank():
    return BG_IMG.copy() if BG_IMG is not None else Image.new("RGB", (W, H), BG)


def glass_pil(im, box, r, **kw):
    a = G.glass(G.from_pil(im), box, r, **kw)
    return G.to_pil(a)


def centered(d, y, text, font, fill):
    d.text((W / 2, y), text, font=font, fill=fill, anchor="mm")


def glow_dot(d, x, y, r):
    for k in range(6, 0, -1):
        a = int(30 * k)
        d.ellipse([x - r * k / 2, y - r * k / 2, x + r * k / 2, y + r * k / 2], fill=(ACC[0] * a // 255, ACC[1] * a // 255, ACC[2] * a // 255))
    d.ellipse([x - r, y - r, x + r, y + r], fill=ACC)


def title_slide():
    im = glass_pil(blank(), (W / 2 - 520, 230, W / 2 + 520, 700), 44, blur=30, darken=0.25, veil=0.06, border=0.45)
    d = ImageDraw.Draw(im, "RGBA")
    G.dot(d, (W / 2, 315), 8)
    centered(d, 420, "Holo", F(132, "Light"), INK)
    centered(d, 525, "A live AI guide that sees what you see", F(36, "Light"), (225, 232, 245))
    centered(d, 578, "and shows you how to build, fix or use anything", F(36, "Light"), (225, 232, 245))
    d.rounded_rectangle([W / 2 - 92, 626, W / 2 + 92, 672], radius=23, outline=(235, 242, 255, 170), width=2)
    centered(d, 649, "See it  →", F(22, "Medium"), INK)
    centered(d, 820, "Meta Global AI Developer Hackathon · Muse Spark · SAM 3.1 · DINOv3", F(22, "Light"), (170, 185, 205))
    return im


def text_slide(title, lines, foot=""):
    nh = sum(110 if isinstance(l, tuple) else 54 for l in lines)
    im = glass_pil(blank(), (100, 90, W - 100, 250 + nh + 40), 40, blur=30, darken=0.3, veil=0.05, border=0.45)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((160, 170), title, font=F(58, "Light"), fill=INK, anchor="lm")
    y = 290
    for ln in lines:
        if isinstance(ln, tuple):
            head, body = ln
            G.dot(d, (168, y), 6)
            d.text((196, y), head, font=F(32, "Medium"), fill=INK, anchor="lm")
            d.text((196, y + 42), body, font=F(23, "Light"), fill=(205, 214, 228), anchor="lm")
            y += 110
        else:
            d.text((140, y), ln, font=F(30), fill=SUB, anchor="lm")
            y += 54
    if foot:
        d.text((140, H - 90), foot, font=F(22), fill=(110, 150, 195), anchor="lm")
    return im


def arch_slide():
    BW, BH = 330, 170
    im = blank()
    for (_, _, x, y) in [(0, 0, 100, 345), (0, 0, 500, 240), (0, 0, 500, 460), (0, 0, 900, 240), (0, 0, 900, 460), (0, 0, 1300, 345)]:
        im = glass_pil(im, (x, y, x + BW, y + BH), 26, blur=22, darken=0.32, veil=0.06, border=0.45)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((140, 120), "How Holo works", font=F(58, "Light"), fill=INK, anchor="lm")
    boxes = [
        ("Camera", "your first-person view:\nQuest, glasses or phone", 100, 345),
        ("Perception", "SAM 3.1: text → masks, tracked\nhand contact: what is held", 500, 240),
        ("Hands", "DINOv3 patch features:\nwhich instance is which", 500, 460),
        ("Muse Spark", "names parts · plans steps\nanswers · checks progress", 900, 240),
        ("Guidance", "grab cue · tracked target\nprogress · auto step-done", 900, 460),
        ("You", "see it on the real object,\nhands free", 1300, 345),
    ]
    for title, body, x, y in boxes:
        d.text((x + 24, y + 40), title, font=F(30, "Medium"), fill=INK, anchor="lm")
        yy = y + 90
        for ln in body.split("\n"):
            d.text((x + 24, yy), ln, font=F(19, "Light"), fill=(210, 220, 234), anchor="lm")
            yy += 30
    def arrow(p, q):
        d.line([p, q], fill=(200, 225, 255, 200), width=2)
        d.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=(220, 236, 255, 255))
    arrow((430, 430), (500, 325))
    arrow((430, 430), (500, 545))
    arrow((830, 325), (900, 325))
    arrow((830, 545), (900, 545))
    arrow((1065, 410), (1065, 460))
    arrow((1230, 325), (1300, 410))
    arrow((1230, 545), (1300, 450))
    d.text((140, 760), "On Quest, the headset's own hand tracking and passthrough camera feed the same pipeline;", font=F(23, "Light"), fill=(205, 214, 228), anchor="lm")
    d.text((140, 798), "this demo runs it on a recorded first-person video.", font=F(23, "Light"), fill=(205, 214, 228), anchor="lm")
    return im


def chip(img, idx):
    ch = [c for c in CHAPTERS if c[0] <= idx][-1]
    t, detail = ch[1], ch[2]
    f1, f2 = F(21, "Medium"), F(15, "Light")
    w = max(f1.getlength(t), f2.getlength(detail)) + 48
    x0, y0 = 24, H - 92
    img = glass_pil(img, (x0, y0, x0 + w, y0 + 66), 20, blur=14, darken=0.32, veil=0.06, border=0.4)
    d = ImageDraw.Draw(img, "RGBA")
    d.text((x0 + 24, y0 + 22), t, font=f1, fill=INK, anchor="lm")
    d.text((x0 + 24, y0 + 46), detail, font=f2, fill=(210, 220, 234), anchor="lm")
    return img


def fade(im, a):
    return Image.blend(blank(), im, max(0.0, min(1.0, a)))


def app_segment(d, put, hold, cue, k):
    """Live Unity app session: real Muse voice clips, GPU waits skipped and labelled."""
    cue("narr", "Now the Holo app itself. Muse plans from the camera view, SAM 3.1 highlights the part, and in build mode it designs the drone, renders it, and makes a 3D model you can hold and print.")
    hold(text_slide("Build mode", [
        ("Say what you want to make", "Muse Spark designs it: parts, 3D-printable parts, wiring, code and steps"),
        ("See it before you build it", "an open image model renders it; image-to-3D turns it into a watertight mesh"),
        ("Hold it, then print it", "pinch to move, two hands to scale, look from the top, export the STL"),
    ]), 11)
    rec = int((d / "DONE").read_text().strip())
    src_fps = 24
    skips = [tuple(l.split("\t")) for l in (d / "skips.tsv").read_text().splitlines() if l.strip()]
    skips = {int(f): int(float(s)) for f, s in skips}
    speech = [l.split("\t") for l in (d / "speech.tsv").read_text().splitlines() if l.strip()]
    base = k[0]
    for t, name in speech:
        cue("wav", str(d / name), base + float(t) * FPS)
    badge_until, badge = -1, ""
    for f in range(0, rec, src_fps // FPS):
        for sf, sec in skips.items():
            if f <= sf < f + src_fps // FPS:
                badge_until, badge = f + 30, f"+{sec} s"
        im = Image.open(d / f"{f:05d}.jpg").convert("RGB")
        sc = min(W / im.width, H / im.height)
        im = im.resize((int(im.width * sc), int(im.height * sc)))
        bg = blank(); bg.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        if f < badge_until:
            dr = ImageDraw.Draw(bg, "RGBA")
            dr.rounded_rectangle((W - 250, 30, W - 30, 92), 31, fill=(20, 26, 40, 200), outline=(150, 190, 255, 160))
            dr.text((W - 140, 61), "\u23e9 " + badge + " GPU", font=F(26), fill=(225, 235, 250), anchor="mm")
        put(bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--guide-log", default="")
    ap.add_argument("--cut", default="", help="optional editorial trim, source frames A:B")
    ap.add_argument("--app", default="", help="DemoRecorder folder of the live Unity app session (Build mode)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k = [0]
    cues = []   # (time s, voice, text)

    def cue(voice, text, at=None):
        cues.append((round((k[0] if at is None else at) / FPS, 2), voice, text))

    def put(im):
        im.save(out / f"{k[0]:05d}.jpg", quality=92)
        k[0] += 1

    def hold(im, sec, fin=0.5, fout=0.5):
        n = int(sec * FPS)
        for i in range(n):
            t = i / FPS
            put(fade(im, min(1, t / fin if fin else 1, (sec - t) / fout if fout else 1)))

    global BG_IMG
    bg = cv2.imread(str(sorted(Path(a.raw).glob("*.jpg"))[344]))
    bg = cv2.GaussianBlur(cv2.resize(bg, (W, H)), (0, 0), 28) * 0.42
    BG_IMG = G.to_pil(bg)
    tl = json.loads((Path(a.seq) / "timeline.json").read_text())
    CHAPTERS[:] = build_chapters(tl)
    cue("narr", "This is Holo. A live A.I. guide that sees what you see, and shows you how to build, fix, or use anything.")
    hold(title_slide(), 6.5)
    cue("narr", "Instructions usually live somewhere else. Chatbots can describe a task, but they can't point at the part in your hands. Holo shows it on the real thing.")
    hold(text_slide("The problem", [
        ("Instructions live somewhere else", "manuals, videos and forums: you keep looking away from what's in your hands"),
        ("Assistants can't point", "chatbots describe; they can't show you which part, where, or how to hold it"),
        ("Holo shows it on the real thing", "it sees what you see, names it, and guides your hands step by step"),
    ]), 10)
    # before / after
    cue("narr", "This is my own first person view. And the same view, with Holo.")
    seq = sorted(Path(a.seq).glob("*.jpg"))
    raw = sorted(Path(a.raw).glob("*.jpg"))
    for i in range(110, 170):
        L = Image.open(raw[i]).convert("RGB").resize((W // 2, H // 2))
        R = Image.open(seq[i]).convert("RGB").resize((W // 2, H // 2))
        im = blank()
        im.paste(L, (0, H // 4))
        im.paste(R, (W // 2, H // 4))
        d = ImageDraw.Draw(im)
        d.text((W // 4, H // 4 - 40), "Your view", font=F(30), fill=SUB, anchor="mm")
        d.text((3 * W // 4, H // 4 - 40), "With Holo", font=F(30), fill=INK, anchor="mm")
        put(im)
    # full run
    base = k[0]
    n2 = {int(kk): v for kk, v in tl["n2idx"].items()}
    ev = tl["events"]
    cut = set()
    if a.cut:
        c0, c1 = [int(x) for x in a.cut.split(":")]
        cut = set(range(n2[c0], n2[c1]))                      # editorial trim of idle footage
    keep = [i for i in range(len(seq)) if i not in cut]
    pos = {i: j for j, i in enumerate(keep)}
    at = lambda idx: base + pos[min(keep, key=lambda q: abs(q - idx))]  # noqa: E731
    glog = json.loads(Path(a.guide_log).read_text()) if a.guide_log else {}
    evs = sorted(ev.items(), key=lambda x: x[1][0])
    first = evs[0][1][0] if evs else len(seq) // 2
    cue("narr", "Holo sees what I'm holding and lights it up. Blue means in hand, and only what matters right now is highlighted.", at(8))
    cue("narr", "Real hands stay in front, and the glow follows each part as it moves.", at(first // 3))
    T0 = {"plan": 46, "answer": 34, "search": 62}
    for e, (s0, s1) in evs:
        kind = tl.get("kinds", {}).get(e, "answer")
        cue("user", tl["q"][e], at(s0 + 2))
        import re as _re
        spoken = _re.split(r"(?<=[.!?])\s+", _re.sub(r"\s*\[\d+\]", "", tl["a"][e]).strip())[0]   # first sentence aloud; full text on the panel
        cue("muse", spoken, at(s0 + T0[kind]))
        if kind == "plan":
            cue("narr", "Muse plans the task from what it sees. Amber shows what to grab, and each step moves on by itself when Holo sees it happen.", at(s1 + 2))
    if glog.get("done_at"):
        d = glog["done_at"]
        nn = min(n2, key=lambda k: abs(k - d))
        cue("narr", "Holo saw the last step happen. It turns green, and the task is done.", at(n2[nn]))
    for i, p in enumerate(seq):
        if i in cut:
            continue
        im = Image.open(p).convert("RGB")
        if im.size != (W, H):
            im = im.resize((W, H))
        put(chip(im, i))
    if a.app:
        app_segment(Path(a.app), put, hold, cue, k)
    cue("narr", "Under the hood: SAM 3.1 finds and tracks every object from a text prompt. DINOv3 keeps track of which part is which, and the highlight colour shows each object's role. Muse Spark sees, plans, searches, and answers.", k[0] + 16)
    hold(arch_slide(), 20)
    cue("narr", "Next: live on Quest three passthrough, and screen guidance for software.")
    hold(text_slide("Built on Meta", [
        ("Muse Spark", "scene understanding, step plans, grounded answers, object names (Baseten · muse-spark-open)"),
        ("SAM 3.1 + DINOv3", "text-prompted segmentation and multi-object tracking; DINOv3 patch features for identity"),
        ("Build mode", "Muse designs it · open image model renders it · image-to-3D mesh → STL to print"),
        ("Meta XR Interaction SDK", "Quest hand tracking + interactions for the headset app (Unity)"),
        ("Next", "live on Quest 3 passthrough · Muse Voice · software (screen) mode"),
    ]), 11)
    end = glass_pil(blank(), (W / 2 - 360, 300, W / 2 + 360, 620), 40, blur=30, darken=0.25, veil=0.06, border=0.45)
    d = ImageDraw.Draw(end, "RGBA")
    G.dot(d, (W / 2, 370), 8)
    centered(d, 455, "Holo", F(104, "Light"), INK)
    centered(d, 545, "see it · learn it · do it", F(32, "Light"), (225, 232, 245))
    cue("narr", "Holo. See it, learn it, do it.")
    hold(end, 5, fout=1.0)
    (out.parent / "demo_cues.json").write_text(json.dumps({"fps": FPS, "frames": k[0], "cues": cues}, indent=1))
    print("frames", k[0], "=", round(k[0] / FPS, 1), "s")


if __name__ == "__main__":
    main()
