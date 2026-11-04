"""Layer 4 - Muse (voice, reasoning, chat sidebar) over the guided video, plus the final frame sequence.

All Muse text is the real output of Muse Spark for these frames (media/pov/muse/*.json):
  plan_0345.json  "how do I connect this cable to my laptop?" -> identified scene + steps
  ask_0600.json   "which way up does the plug go in?"         -> answer + resource + target box
The video holds (freezes) on the asking frames while the voice pill, the ☰ sidebar and the answer play,
like the real app pausing the guidance while you talk to Muse.

Usage: python muse_holo.py --guided out/final_v15 --muse media/pov/muse --out out/final_seq
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ui_glass as G

S = 2
FONT = "./HoloXR/Assets/Holo/UI/Fonts/Inter.ttf"
INK = (240, 247, 255)
SUB = (150, 200, 245)
GLASS = (12, 20, 36)
GLASS_RECTS = []   # filled by voice_pill / sidebar each frame; frosted onto the base before the overlay
EDGE = (160, 210, 255)
ACCENT = (120, 220, 255)


def F(px):
    return ImageFont.truetype(FONT, int(px * S))


def wrap(d, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def menu_button(d, W, active, t):
    """Top-right ☰ (Muse sidebar)."""
    x, y, r = W - 34 * S, 34 * S, 18 * S
    glow = 0.5 + 0.5 * np.sin(t * 4) if active else 0
    GLASS_RECTS.append(((x - r, y - r, x + r, y + r), r, 1.0))
    if glow:
        d.ellipse([x - r, y - r, x + r, y + r], outline=(150, 215, 255, int(120 + 120 * glow)), width=max(1, int(1.5 * S)))
    for k in (-6, 0, 6):
        d.line([(x - 8 * S, y + k * S), (x + 8 * S, y + k * S)], fill=INK + (240,), width=2 * S)


def voice_pill(d, W, H, text, reveal, t, alpha=1.0):
    """Bottom-centre listening pill: mic + live waveform + transcript revealed word by word."""
    words = text.split()
    shown = " ".join(words[: max(1, int(round(len(words) * reveal)))])
    f = F(17)
    tw = d.textlength(text, font=f)
    pw, ph = int(tw + 120 * S), int(52 * S)
    x0, y0 = int(W / 2 - pw / 2), int(H - ph - 26 * S)
    A = lambda v: int(v * alpha)  # noqa: E731
    GLASS_RECTS.append(((x0, y0, x0 + pw, y0 + ph), ph / 2, alpha))
    cx, cy = x0 + 28 * S, y0 + ph // 2
    d.ellipse([cx - 13 * S, cy - 13 * S, cx + 13 * S, cy + 13 * S], fill=ACCENT + (A(255),))
    d.rounded_rectangle([cx - 4 * S, cy - 8 * S, cx + 4 * S, cy + 3 * S], radius=4 * S, fill=GLASS + (A(255),))
    d.arc([cx - 7 * S, cy - 6 * S, cx + 7 * S, cy + 7 * S], 20, 160, fill=GLASS + (A(255),), width=2 * S)
    # waveform
    for i in range(7):
        hgt = (4 + 9 * abs(np.sin(t * 9 + i * 1.3))) * S * (1 if reveal < 1 else 0.3)
        bx = x0 + 52 * S + i * 5 * S
        d.line([(bx, cy - hgt), (bx, cy + hgt)], fill=ACCENT + (A(230),), width=2 * S)
    d.text((x0 + 96 * S, cy), shown, font=f, fill=INK + (A(255),), anchor="lm")


def sidebar(d, W, H, slide, msgs, footer, t, left=False):
    """Right chat panel (opened from ☰). msgs: list of (role, text, extra-lines)."""
    pw = int(272 * S)
    x0 = int(W - 14 * S - pw * ease(slide)) if not left else int(14 * S - pw * (1 - ease(slide)))
    if slide <= 0:
        return
    y0, y1 = int(96 * S), int(H - 16 * S)
    x1 = x0 + pw
    GLASS_RECTS.append(((x0, y0, x1, y1), 22 * S, 1.0))
    # header
    d.ellipse([x0 + 18 * S, y0 + 18 * S, x0 + 30 * S, y0 + 30 * S], fill=ACCENT + (255,))
    d.text((x0 + 38 * S, y0 + 24 * S), "Muse", font=F(16), fill=INK + (255,), anchor="lm")
    d.text((x1 - 16 * S, y0 + 24 * S), "Muse Spark · vision", font=F(11), fill=SUB + (230,), anchor="rm")
    d.line([(x0 + 14 * S, y0 + 44 * S), (x1 - 14 * S, y0 + 44 * S)], fill=EDGE + (70,), width=S)
    y = y0 + 58 * S
    for role, text, extra in msgs:
        f = F(13)
        inner = pw - 48 * S
        lines = wrap(d, text, f, inner - 20 * S) if text else []
        if len(lines) > 7:                                   # never run past the panel
            lines = lines[:7]; lines[-1] = lines[-1].rstrip(".,") + "\u2026"
        ex2 = []
        for ex in extra:                                     # wrap step lines too
            ex2 += wrap(d, ex, F(12), inner - 12 * S) if ex else [""]
        extra = ex2
        if role == "thinking":
            dots = "".join("•" if (int(t * 6) + k) % 3 == 0 else "·" for k in range(3))
            d.text((x0 + 20 * S, y + 10 * S), f"Looking at your scene  {dots}", font=F(13), fill=SUB + (240,), anchor="lm")
            y += 30 * S
            continue
        bh = 16 * S + 18 * S * len(lines) + sum(17 * S for _ in extra)
        if role == "user":
            bx0, bx1, fill = x0 + 44 * S, x1 - 16 * S, (255, 255, 255, 34)
        elif role == "search":
            bx0, bx1, fill = x0 + 16 * S, x1 - 16 * S, (255, 255, 255, 14)
        else:
            bx0, bx1, fill = x0 + 16 * S, x1 - 16 * S, (255, 255, 255, 20)
        d.rounded_rectangle([bx0, y, bx1, y + bh], radius=12 * S, fill=fill)
        ty = y + 16 * S
        for li, ln in enumerate(lines):
            col = SUB if role == "search" else INK
            if role == "search" and li == 0:   # magnifier
                cx, cy = bx0 + 18 * S, ty
                d.ellipse([cx - 5 * S, cy - 5 * S, cx + 3 * S, cy + 3 * S], outline=ACCENT + (255,), width=2 * S)
                d.line([(cx + 2 * S, cy + 2 * S), (cx + 6 * S, cy + 6 * S)], fill=ACCENT + (255,), width=2 * S)
                d.text((bx0 + 30 * S, ty), ln, font=f, fill=col + (255,), anchor="lm")
            else:
                d.text((bx0 + 12 * S, ty), ln, font=f, fill=col + (255,), anchor="lm")
            ty += 18 * S
        for ex in extra:
            d.text((bx0 + 12 * S, ty + 2 * S), ex, font=F(12), fill=(205, 228, 250, 255), anchor="lm")
            ty += 17 * S
        y += bh + 10 * S
    if footer:
        d.text((x0 + 18 * S, y1 - 18 * S), footer, font=F(11), fill=SUB + (220,), anchor="lm")


def brackets(d, box, alpha, label):
    x0, y0, x1, y1 = box
    L = 14 * S
    c = ACCENT + (int(255 * alpha),)
    for (px, py, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        d.line([(px, py), (px + dx * L, py)], fill=c, width=3 * S)
        d.line([(px, py), (px, py + dy * L)], fill=c, width=3 * S)
    f = F(12)
    tw = d.textlength(label, font=f)
    d.rounded_rectangle([x0, y1 + 8 * S, x0 + tw + 20 * S, y1 + 32 * S], radius=12 * S, fill=GLASS + (int(200 * alpha),), outline=EDGE + (int(160 * alpha),), width=S)
    d.text((x0 + 10 * S, y1 + 20 * S), label, font=f, fill=INK + (int(255 * alpha),), anchor="lm")


def overlay_rgba(b64, W, H):
    import base64
    arr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)
    arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA))


def main():
    """Data-driven: every question, answer, search result and highlight comes from replies.json, which
    ask_muse.py fills with Muse's live replies to the user's spoken questions (frame + words)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--guided", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    from urllib.parse import urlparse
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {int(x) for x in a.only.split(",") if x}
    reps = json.load(open(a.replies))
    holds = {}
    for i, it in enumerate(reps):
        r = it["reply"]
        kind = "plan" if r.get("kind") == "plan" else ("search" if r.get("kind") == "search" else "answer")
        L = {"plan": 150, "search": 170, "answer": 120}[kind]
        holds[it["frame"]] = (L, f"E{i}", kind, it)
    frames = {int(p.stem): p for p in Path(a.guided).glob("*.jpg")}
    seq, timeline = [], {"events": {}, "n2idx": {}, "q": {}, "a": {}, "kinds": {}}
    for n in sorted(frames):
        if n in holds:
            L, ev, kind, it = holds[n]
            timeline["events"][ev] = [len(seq), len(seq) + L]
            r = it["reply"]
            timeline["q"][ev] = it["say"]
            timeline["a"][ev] = r.get("say") or r.get("summary") or r.get("answer") or ""
            timeline["kinds"][ev] = kind
            seq += [(n, k, ev) for k in range(L)]
        else:
            timeline["n2idx"][n] = len(seq)
            seq.append((n, None, None))
    timeline.update({"fps": a.fps, "total": len(seq)})
    (out / "timeline.json").write_text(json.dumps(timeline))
    ovl = {}
    for idx, (n, k, ev) in enumerate(seq):
        if only and idx not in only:
            continue
        img = Image.open(frames[n]).convert("RGBA")
        W, H = img.size
        GLASS_RECTS.clear()
        t = idx / a.fps
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        if ev is not None:
            L, _, kind, it = holds[n]
            r, q = it["reply"], it["say"]
            T0 = {"plan": 46, "answer": 34, "search": 60}[kind]          # frame within the hold when the answer lands
            k_open = {"plan": 30, "answer": 22, "search": 26}[kind]
            if k < k_open + 4:
                voice_pill(d, W, H, q, ease(k / max(1, k_open - 4)), t, alpha=1.0 if k < k_open else 1 - (k - k_open) / 4)
            hl = r.get("_highlight")
            left = False
            if hl and hl.get("box"):
                left = (hl["box"][0] + hl["box"][2]) / 2 / 854 > 0.5          # keep the panel off the highlight
                if k >= T0:
                    if ev not in ovl:
                        ovl[ev] = overlay_rgba(hl["overlay"], W, H)
                    A = min(1, (k - T0) / 6) * (1 if k < L - 8 else max(0, 1 - (k - (L - 8)) / 8))
                    o = ovl[ev].copy()
                    o.putalpha(o.getchannel("A").point(lambda v: int(v * A * (0.85 + 0.15 * np.sin(t * 4.5)))))
                    img = Image.alpha_composite(img, o)
            if k >= k_open:
                slide = ease((k - k_open) / 8) if k < L - 8 else 1 - ease((k - (L - 8)) / 8)
                msgs = [("user", q, [])]
                foot = ""
                if k < T0 and kind != "search":
                    msgs.append(("thinking", "", []))
                elif kind == "plan":
                    from guide_holo import short
                    steps = [f"{i + 1}.  {short(st['instruction'], 7)}" for i, st in enumerate(r.get("steps", []))][:6]
                    msgs.append(("muse", r.get("summary", "").split(". ")[0].rstrip(".") + ".", [""] + steps))
                    foot = f"Answered in {r.get('_latency_s', 0):.1f} s  ·  sees: {(r.get('identified') or '')[:38]}"
                elif kind == "answer":
                    ans = " ".join(r.get("answer", "").split(". ")[:2]).rstrip(".") + "."
                    msgs.append(("muse", ans, ["", f"Learn: {r['resource']}"] if r.get("resource") else []))
                    foot = f"Answered in {r.get('_latency_s', 0):.1f} s" + (f"  ·  highlighting: {hl['label']}" if hl else "")
                else:
                    res = [(x["title"][:40] + "…" if len(x["title"]) > 41 else x["title"]) for x in r.get("results", [])][:4]
                    if k < 36:
                        msgs.append(("thinking", "", []))
                    elif k < T0:
                        shown = res[: max(0, min(len(res), (k - 40) // 4 + 1))] if k >= 40 else []
                        msgs.append(("search", "Searching the web", [f"\u201c{r.get('query', '')[:44]}\u2026\u201d"] + ["\u00b7 " + x for x in shown]))
                    else:
                        msgs.append(("search", f"Searched the web \u00b7 {len(r.get('results', []))} results", [f"\u201c{r.get('query', '')[:44]}\u2026\u201d"]))
                        msgs.append(("muse", " ".join(r.get("answer", "").split(". ")[:2]).rstrip(".") + ".", []))
                        doms = ", ".join(dict.fromkeys(urlparse(x.get("url", "")).netloc.replace("www.", "") for x in r.get("sources", []) if isinstance(x, dict)))
                        foot = f"Sources: {doms}  \u00b7  {r.get('_latency_s', 0):.1f} s"
                sidebar(d, W, H, slide, msgs, foot, t, left=left)
        menu_button(d, W, ev is not None, t)
        if GLASS_RECTS:
            base = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR).astype(np.float32)
            for box, rr, al in GLASS_RECTS:
                base = G.glass(base, box, rr, alpha=al, blur=18 * S, darken=0.38, veil=0.07, border=0.4)
            img = G.to_pil(base).convert("RGBA")
        frame = Image.alpha_composite(img, ov).convert("RGB")
        frame.save(out / f"{idx:05d}.jpg", quality=93)
    print("frames", len(seq), "->", out)


if __name__ == "__main__":
    main()
