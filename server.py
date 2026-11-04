"""Holo backend: proxies vision/reasoning calls to Muse Spark Open on Baseten.

Endpoints take a camera frame (or screenshot) as a base64 JPEG and return
structured JSON that the browser turns into overlays. All boxes are
normalized to 0-1000 on both axes.
"""

import json
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
ENV = {**dotenv_values(ROOT.parent / ".env"), **dotenv_values(ROOT / ".env"), **os.environ}
BASETEN_KEY = ENV.get("meta_baseten") or ENV.get("BASETEN_API_KEY")
MODEL = ENV.get("HOLO_MODEL", "meta-models/muse-spark-open-1026")
BASE_URL = "https://inference.baseten.co/v1/chat/completions"
META_KEY = ENV.get("MODEL_API_KEY") or ENV.get("meta_api")
META_URL = "https://api.meta.ai/v1"

if not BASETEN_KEY:
    raise SystemExit("Missing Baseten key: set meta_baseten in ../.env or BASETEN_API_KEY")

app = FastAPI(title="Holo")
client = httpx.AsyncClient(timeout=120)

SYSTEM = """You are Holo, a live AI guide that sees through the user's camera (or their screen)
and shows them, step by step, how to build, fix or use something. You are precise, practical and brief.
Coordinates: every box is [x1, y1, x2, y2] normalized to 0-1000 over the image width and height.
Boxes must tightly cover the exact target (a pin, port, screw, button, wire end), not the whole object.
If something is too small or unclear to locate, say so and ask the user to move the camera closer.
Safety: refuse to guide work on mains electricity, gas, or lithium battery repair beyond simple
connections; tell the user to get a professional. Always answer with ONLY the JSON requested."""


class PlanReq(BaseModel):
    image: str
    goal: str
    mode: str = "physical"  # physical | screen
    objects: list[str] = []  # exact names of parts in a virtual/tracked scene, if known


class CheckReq(BaseModel):
    image: str
    goal: str
    step: dict
    mode: str = "physical"


class AudioReq(BaseModel):
    audio: str  # base64 WAV, mono 16-bit PCM, 16 or 24 kHz


class InventoryReq(BaseModel):
    image: str
    goal: str = ""


class AskReq(BaseModel):
    image: str | None = None
    question: str
    goal: str = ""
    step: dict | None = None


def _image_part(b64: str) -> dict:
    if not b64.startswith("data:"):
        b64 = "data:image/jpeg;base64," + b64
    return {"type": "image_url", "image_url": {"url": b64}}


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


async def spark(content: list, effort: str = "low", max_tokens: int = 4000) -> dict:
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "reasoning_effort": effort,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
    }
    t0 = time.time()
    r = await client.post(BASE_URL, json=body, headers={"Authorization": f"Api-Key {BASETEN_KEY}"})
    if r.status_code != 200:
        raise HTTPException(502, f"model error {r.status_code}: {r.text[:300]}")
    data = r.json()
    text = data["choices"][0]["message"].get("content")
    try:
        out = _extract_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(502, f"bad model output: {e}: {str(text)[:300]}")
    out["_latency_s"] = round(time.time() - t0, 2)
    out["_tokens"] = data.get("usage", {}).get("total_tokens")
    return out


PLAN_PROMPT = """User goal: "{goal}"
Mode: {mode} ({mode_hint})

Look at the image. Identify what the user is working with (exact device/board/app and model if visible).
Plan the task as short, concrete steps (max 8). For each step give where the user's hand or cursor
must go, which hand action to show, and a one-line beginner explanation of WHY.

Return ONLY this JSON:
{{
  "identified": "what you see, with model names if readable",
  "summary": "one sentence plan",
  "needs_closer_look": false,
  "steps": [
    {{
      "instruction": "imperative, under 15 words",
      "target_label": "2-5 words naming the exact thing, e.g. 'XT60 battery plug' or 'Save button' (a short visual noun phrase a segmentation model can find)",
      "target_box": [x1, y1, x2, y2] or null if not visible yet,
      "from_box": [x1, y1, x2, y2] of the part being moved INTO the target (e.g. the loose propeller), or null,
      "hand_action": "point" | "pinch" | "grip" | "press" | "turn" | "none",
      "done_when": "what will look different when this step is complete",
      "learn": "why this step matters / what this part is, one sentence",
      "target_object": "exact known part name the step is about, or null",
      "pick": "short visual noun phrase of the object to pick up (e.g. 'USB plug'), or null",
      "place": "short visual noun phrase of where it goes (e.g. 'USB port'), or null",
      "connect": ["exact name A", "exact name B"] or null
    }}
  ]
}}
{objects_hint}"""

CHECK_PROMPT = """User goal: "{goal}"
Mode: {mode}
Current step: {step}

Look at the image now. Has this step been completed (use the step's done_when)? Also re-locate the
target so the overlay stays on it. Give one short spoken coaching line: either confirm, or say exactly
what to fix (which direction to move, wrong part, wrong pin, angle) or ask to move closer.

Return ONLY this JSON:
{{
  "done": true | false,
  "target_box": [x1, y1, x2, y2] or null,
  "feedback": "short spoken line",
  "problem": null or "what is wrong"
}}"""

ASK_PROMPT = """User goal: "{goal}"
Current step: {step}
The user asks: "{question}"

Answer like an expert mentor in at most 3 short sentences, using what you see in the image if given.
If useful, name one learning resource (datasheet section, concept, or search phrase).
If the answer points at something visible, include its box.

Return ONLY this JSON:
{{"answer": "...", "resource": null or "...", "target_box": null or [x1, y1, x2, y2],
  "target_label": null or "2-4 word name of the one visible thing to highlight, e.g. 'USB port' or 'white USB cable'"}}"""

MODE_HINT = {
    "physical": "camera view of real objects; targets are physical parts",
    "screen": "screenshot of the user's computer screen; targets are UI elements to click or type into",
}


@app.post("/api/plan")
async def plan(req: PlanReq):
    objects_hint = ""
    if req.objects:
        objects_hint = (
            "Known parts in this scene (use these EXACT names in pick/place/connect): "
            + ", ".join(req.objects)
            + ". Use pick+place for moving/mounting a part, connect for wiring/soldering two things, "
              "and null for steps that are only pressing or checking. Map motors to mounts by number (Motor 1 -> Mount M1)."
        )
    prompt = PLAN_PROMPT.format(goal=req.goal, mode=req.mode, mode_hint=MODE_HINT.get(req.mode, ""), objects_hint=objects_hint)
    return await spark([_image_part(req.image), {"type": "text", "text": prompt}], effort="medium", max_tokens=8000)


INVENTORY_PROMPT = """User goal: "{goal}"
List every distinct physical thing in this image that matters for the goal, as SHORT noun phrases a
segmentation model can find (1-3 words, generic visual names, e.g. "green propeller", "brushless motor",
"xt60 connector", "wrench", "battery strap", "hand"). Include "hand" if hands are visible. No duplicates,
no abstract things, max 10.
Return ONLY JSON: {{"labels": ["..."]}}"""


@app.post("/api/inventory")
async def inventory(req: InventoryReq):
    """Muse Spark names what is in view; SAM 3 then segments and tracks each name."""
    prompt = INVENTORY_PROMPT.format(goal=req.goal)
    return await spark([_image_part(req.image), {"type": "text", "text": prompt}])


@app.post("/api/check")
async def check(req: CheckReq):
    step = json.dumps({k: req.step.get(k) for k in ("instruction", "target_label", "hand_action", "done_when")})
    prompt = CHECK_PROMPT.format(goal=req.goal, mode=req.mode, step=step)
    return await spark([_image_part(req.image), {"type": "text", "text": prompt}])


@app.post("/api/ask")
async def ask(req: AskReq):
    step = json.dumps(req.step or {})
    content = [{"type": "text", "text": ASK_PROMPT.format(goal=req.goal, step=step, question=req.question)}]
    if req.image:
        content.insert(0, _image_part(req.image))
    return await spark(content)


SEARCH_PROMPT = """User goal: "{goal}"
The user asks: "{question}"
Use what you see in the image to identify the exact device/part and model. Find the most likely cause and the
fix. Give a short answer (max 5 sentences) and, if a procedure is needed, up to 6 concrete steps.
Return ONLY JSON:
{{"answer": "...", "sources": ["url or title", ...], "resource": null or "...",
  "steps": [{{"instruction": "...", "target_label": "2-5 words", "target_box": null, "hand_action": "point|pinch|grip|press|turn|none", "done_when": "...", "learn": "..."}}]}}"""


async def meta_responses(content: list, tools: list | None = None, model: str = "muse-spark-1.3") -> dict:
    """Meta Model API (Responses) with built-in web_search. Needs MODEL_API_KEY."""
    body = {"model": model, "input": [{"role": "user", "content": content}]}
    if tools:
        body["tools"] = tools
    r = await client.post(f"{META_URL}/responses", json=body, headers={"Authorization": f"Bearer {META_KEY}"})
    if r.status_code != 200:
        raise HTTPException(502, f"meta error {r.status_code}: {r.text[:300]}")
    data = r.json()
    text = data.get("output_text") or "".join(
        c.get("text", "") for item in data.get("output", []) if item.get("type") == "message" for c in item.get("content", [])
    )
    return _extract_json(text)


class SearchReq(BaseModel):
    image: str | None = None
    question: str
    goal: str = ""


@app.post("/api/search")
async def search(req: SearchReq):
    """Find a fix: Muse with web search (Meta Model API) when available, else Muse Spark Open's own knowledge."""
    prompt = SEARCH_PROMPT.format(goal=req.goal, question=req.question)
    if META_KEY:
        content = [{"type": "input_text", "text": prompt}]
        if req.image:
            img = req.image if req.image.startswith("data:") else "data:image/jpeg;base64," + req.image
            content.insert(0, {"type": "input_image", "image_url": img})
        out = await meta_responses(content, tools=[{"type": "web_search"}])
        out["web"] = True
        return out
    return await agent_search(req)


# ---- agentic web search with Muse Spark (no Meta key needed) ---------------------------------------
QUERY_PROMPT = """User goal: "{goal}"
The user asks: "{question}"
Look at the image if given. Write the single best web search query to answer this precisely
(include exact part / model names you can see). Return ONLY JSON {{"query": "..."}}"""

ANSWER_PROMPT = """User goal: "{goal}"
The user asks: "{question}"
Web results for "{query}":
{results}

Answer in at most 3 short sentences, using the results and what you see in the image.
Cite sources inline as [1], [2]. Return ONLY JSON
{{"answer": "...", "sources": [{{"n": 1, "title": "...", "url": "..."}}], "resource": "one search phrase to learn more"}}"""


async def web_search(query: str, k: int = 5) -> list[dict]:
    """DuckDuckGo HTML results: title, url, snippet."""
    r = await client.post("https://html.duckduckgo.com/html/", data={"q": query},
                          headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) HoloAgent/1.0"})
    html = r.text
    out = []
    for m in re.finditer(r'class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>', html, re.S):
        url, title, snip = m.group(1), re.sub("<.*?>", "", m.group(2)), re.sub("<.*?>", "", m.group(3))
        if "duckduckgo.com" in url:
            continue
        out.append({"title": title.strip(), "url": url, "snippet": re.sub(r"\s+", " ", snip).strip()[:300]})
        if len(out) >= k:
            break
    return out


async def agent_search(req: "SearchReq") -> dict:
    """Muse decides the query -> we search the web -> Muse answers with citations."""
    t0 = time.time()
    img = [_image_part(req.image)] if req.image else []
    q = await spark(img + [{"type": "text", "text": QUERY_PROMPT.format(goal=req.goal, question=req.question)}], effort="low", max_tokens=1500)
    query = q.get("query") or req.question
    results = await web_search(query)
    listing = "\n".join(f"[{i + 1}] {r['title']} ({r['url']}): {r['snippet']}" for i, r in enumerate(results)) or "(no results)"
    out = await spark(img + [{"type": "text", "text": ANSWER_PROMPT.format(goal=req.goal, question=req.question, query=query, results=listing)}],
                      effort="medium", max_tokens=4000)
    out.update({"web": True, "query": query, "results": results, "_latency_s": round(time.time() - t0, 2)})
    return out


@app.post("/api/transcribe")
async def transcribe(req: AudioReq):
    """Muse Voice Transcribe (Meta Model API) when MODEL_API_KEY is set, else local faster-whisper."""
    if not META_KEY:
        return await stt(req)
    import base64, uuid

    wav = base64.b64decode(req.audio)
    settings = {"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    files = {
        "request": (None, json.dumps(settings), "application/json"),
        "audio": ("speech.wav", wav, "audio/wav"),
    }
    r = await client.post(
        f"{META_URL}/asr/transcribe", params={"sessionId": uuid.uuid4().hex},
        headers={"Authorization": f"Bearer {META_KEY}"}, files=files,
    )
    if r.status_code != 200:
        raise HTTPException(502, f"transcribe error {r.status_code}: {r.text[:300]}")
    return {"text": r.json().get("transcript", "")}


# ---- local voice: faster-whisper (wake word "Muse" + dictation) and Piper (Muse's voice) -------------
import base64 as _b64
import io as _io
import subprocess as _sp
import wave as _wave

import numpy as _np
from starlette.concurrency import run_in_threadpool

_whisper: dict = {}
VOICE_DIR = ROOT / "media" / "voice"
WAKE = re.compile(r"\b(muse|mews|muze|moose|meuse|mius|myuse|mus|muses|amuse)\b", re.I)


def _wm(name: str):
    if name not in _whisper:
        from faster_whisper import WhisperModel
        _whisper[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _whisper[name]


def _pcm16k(b64: str) -> "_np.ndarray":
    with _wave.open(_io.BytesIO(_b64.b64decode(b64))) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = _np.frombuffer(w.readframes(w.getnframes()), _np.int16).astype(_np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != 16000:
        x = _np.interp(_np.arange(0, len(x), sr / 16000), _np.arange(len(x)), x).astype(_np.float32)
    return x


def _transcribe(x, model: str) -> str:
    segs, _ = _wm(model).transcribe(x, beam_size=1, language="en", vad_filter=False,
                                     initial_prompt="Muse, Holo, USB, laptop, cable, port.")
    return " ".join(s.text for s in segs).strip()


def _strip_wake(text: str) -> str:
    m = WAKE.search(text)
    return text[m.end():].lstrip(" ,.!?") if m else text


@app.post("/api/stt")
async def stt(req: AudioReq):
    """Local dictation (faster-whisper base.en). Strips a leading 'Muse' wake word."""
    text = await run_in_threadpool(_transcribe, _pcm16k(req.audio), "base.en")
    return {"text": _strip_wake(text), "raw": text}


@app.post("/api/wake")
async def wake(req: AudioReq):
    """Wake word check on a short window (faster-whisper tiny.en): is the user saying 'Muse'?"""
    x = _pcm16k(req.audio)
    if float(_np.sqrt((x ** 2).mean() + 1e-12)) < 0.008:
        return {"wake": False, "text": ""}
    text = await run_in_threadpool(_transcribe, x, "tiny.en")
    return {"wake": bool(WAKE.search(text)), "text": text}


class TTSReq(BaseModel):
    text: str
    voice: str = "en_US-lessac-high"


def _tts(text: str, voice: str) -> bytes:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        _sp.run(["piper", "-m", str(VOICE_DIR / f"{voice}.onnx"), "-f", f.name, "--length-scale", "0.95"],
                input=text.encode(), capture_output=True, check=True)
        return Path(f.name).read_bytes()


@app.post("/api/tts")
async def tts(req: TTSReq):
    """Muse's voice (Piper, local). Returns a base64 WAV."""
    wav = await run_in_threadpool(_tts, req.text, req.voice)
    return {"audio": _b64.b64encode(wav).decode()}


class ChatReq(BaseModel):
    text: str
    image: str | None = None
    goal: str = ""


SEARCH_WORDS = re.compile(r"\b(search|look up|google|web|online|price|buy|latest|spec|specs|version|compatible|support|"
                          r"usb ?3|usb ?2|charge|wattage|datasheet|manual|review|which model)\b", re.I)
PLAN_WORDS = re.compile(r"\b(how (do|can|to)|help me|guide|steps|walk me|show me how|set ?up|install|connect|fix|build|assemble|replace)\b", re.I)


def _say(text: str, n: int = 2) -> str:
    text = re.sub(r"\s*\[\d+\](\[\d+\])*", "", text or "")
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


@app.post("/api/chat")
async def chat(req: ChatReq):
    """One entry point for the panel / voice: routes to plan, grounded answer, or agentic web search."""
    q = req.text.strip()
    if PLAN_WORDS.search(q) and not SEARCH_WORDS.search(q):
        out = await plan(PlanReq(image=req.image or "", goal=q, mode="physical"))
        n = len(out.get("steps", []))
        return {"kind": "plan", **out, "say": f"{out.get('summary', '')} I've split it into {n} steps."}
    if SEARCH_WORDS.search(q):
        out = await agent_search(SearchReq(image=req.image, question=q, goal=req.goal))
        return {"kind": "search", **out, "say": _say(out.get("answer", ""))}
    out = await ask(AskReq(image=req.image, question=q, goal=req.goal))
    return {"kind": "answer", **out, "say": _say(out.get("answer", ""))}


# ---- SAM 3.1 live highlight (only what is asked about) ----------------------------------------------
SAM31_URL = ENV.get("SAM31_URL", "http://127.0.0.1:8790")   # SAM 3.1 service on the GPU box (SSH tunnel)
ROLE_RGB = {"held": (70, 170, 255), "grab": (255, 200, 60), "dest": (255, 80, 200), "done": (90, 225, 120), "muse": (160, 110, 255)}


class SegReq(BaseModel):
    image: str
    text: str
    role: str = "muse"
    top_k: int = 1


def _glow_overlay(masks: list, w: int, h: int, rgb, scale: int = 2) -> bytes:
    """RGBA PNG (2x) in the SAM 3 demo look: soft tint, crisp edge, halo. Straight alpha."""
    import cv2
    W, H = w * scale, h * scale
    m = _np.zeros((h, w), _np.float32)
    for mk in masks:
        m = _np.maximum(m, mk.astype(_np.float32))
    u = cv2.GaussianBlur(cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 1.2 * scale)
    u = _np.clip((u - 0.35) / 0.3, 0, 1)
    u = u * u * (3 - 2 * u)
    edge = _np.clip(1 - _np.abs(u - 0.5) / 0.42, 0, 1) ** 0.8
    halo = _np.clip(cv2.GaussianBlur(edge, (0, 0), 4 * scale) * 2.4, 0, 1)
    halo2 = _np.clip(cv2.GaussianBlur(edge, (0, 0), 11 * scale) * 2.2, 0, 1) * (1 - 0.5 * u)
    a = _np.clip(1.0 * edge + 0.9 * halo + 0.6 * halo2, 0, 1)        # outline glow only, no fill
    col = _np.array(rgb, _np.float32)
    edge_col = _np.minimum(255, col * 0.8 + 75)
    rgbimg = col[None, None] * (1 - edge[..., None]) + edge_col[None, None] * edge[..., None]
    rgba = _np.dstack([rgbimg[..., ::-1], a * 255]).astype(_np.uint8)          # BGRA for cv2
    ok, png = cv2.imencode(".png", rgba)
    return png.tobytes()


@app.post("/api/segment")
async def segment(req: SegReq):
    """Highlight exactly the thing asked about: SAM 3.1 text prompt -> glow overlay PNG (+ box)."""
    import cv2
    r = await client.post(f"{SAM31_URL}/segment", json={"image": req.image, "text": req.text, "top_k": req.top_k}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, f"sam3.1 error {r.status_code}")
    d = r.json()
    inst = d.get("instances", [])
    if not inst:
        return {"found": False, "text": req.text, "model": d.get("model")}
    masks = [cv2.imdecode(_np.frombuffer(_b64.b64decode(i["mask"]), _np.uint8), cv2.IMREAD_GRAYSCALE) > 127 for i in inst]
    png = await run_in_threadpool(_glow_overlay, masks, d["width"], d["height"], ROLE_RGB.get(req.role, ROLE_RGB["muse"]))
    return {"found": True, "text": req.text, "model": d.get("model"), "box": inst[0]["box"], "score": inst[0]["score"],
            "width": d["width"], "height": d["height"], "overlay": _b64.b64encode(png).decode()}


# ---- Build mode: Muse designs it -> image -> 3D model (view in Holo, export STL to print) -------------
GEN_URL = ENV.get("GEN_URL", "http://127.0.0.1:8791")
BUILD_DIR = ROOT / "media" / "builds"

DESIGN_PROMPT = """User request: "{goal}"
{seen}
You are Holo's maker assistant. Design what the user wants to build, practical for a beginner with common
parts. Return ONLY JSON:
{{
  "name": "short product name",
  "summary": "one sentence",
  "image_prompt": "a product render prompt for an image model, describing the finished build's actual SHAPE concretely (e.g. 'four circular propeller guard rings joined in a square frame', counts, proportions, colours) so it can't be mistaken for a generic product: the build alone, one solid object, centred, plain light-grey studio background, 3/4 view from above, photorealistic",
  "parts": [{{"name": "...", "qty": 1, "note": "spec or why"}}],
  "printable": [{{"name": "part to 3D print, e.g. frame or enclosure", "note": "material / infill"}}],
  "steps": [{{"instruction": "imperative, under 14 words", "target_label": "2-4 word visual noun phrase of the part", "learn": "one sentence why"}}],
  "wiring": ["A -> B: what connects and why", "..."],
  "code": {{"language": "e.g. Arduino C++ or Betaflight CLI", "filename": "...", "content": "short, complete starter code or config"}},
  "safety": "one sentence"
}}"""


class BuildReq(BaseModel):
    goal: str
    image: str | None = None      # optional: photo of the parts the user has


async def _design(req):
    import uuid
    t0 = time.time()
    seen = "The image shows the parts the user already has; use them where possible." if req.image else ""
    content = ([_image_part(req.image)] if req.image else []) + [{"type": "text", "text": DESIGN_PROMPT.format(goal=req.goal, seen=seen)}]
    design = await spark(content, effort="medium", max_tokens=8000)
    bid = uuid.uuid4().hex[:8]
    d = BUILD_DIR / bid
    d.mkdir(parents=True, exist_ok=True)
    design["_goal"] = req.goal
    (d / "design.json").write_text(json.dumps(design, indent=1))
    if isinstance(design.get("code"), dict) and design["code"].get("content"):
        (d / Path(design["code"].get("filename") or "code.txt").name).write_text(design["code"]["content"])
    return {"id": bid, "design": design, "timings": {"design_s": round(time.time() - t0, 1)}}


def _bdir(bid):
    d = BUILD_DIR / Path(bid).name
    if not (d / "design.json").exists():
        raise HTTPException(404, "no such build")
    return d


async def _render(bid):
    d = _bdir(bid)
    design = json.loads((d / "design.json").read_text())
    try:   # free the GPU box for generation (SAM 3.1 reloads on the next highlight)
        await client.post(f"{SAM31_URL}/unload", timeout=30)
    except Exception:
        pass
    t = time.time()
    r = await client.post(f"{GEN_URL}/image", json={"prompt": design.get("image_prompt") or design.get("_goal")}, timeout=400)
    img = r.json()["image"]
    (d / "render.png").write_bytes(_b64.b64decode(img))
    return {"id": bid, "image": img, "image_s": round(time.time() - t, 1)}


async def _mesh(bid):
    d = _bdir(bid)
    img = _b64.b64encode((d / "render.png").read_bytes()).decode()
    t = time.time()
    r = await client.post(f"{GEN_URL}/shape", json={"image": img}, timeout=600)
    sh = r.json()
    (d / "model.glb").write_bytes(_b64.b64decode(sh["glb"]))
    (d / "model.stl").write_bytes(_b64.b64decode(sh["stl"]))
    mesh = {"faces": sh["faces"], "watertight": sh["watertight"], "extents": sh["extents"]}
    (d / "mesh.json").write_text(json.dumps(mesh))
    return {"id": bid, "glb": sh["glb"], "stl_path": str(d / "model.stl"), "glb_path": str(d / "model.glb"),
            "mesh": mesh, "shape_s": round(time.time() - t, 1)}


@app.post("/api/build/design")
async def build_design(req: BuildReq):
    """Stage 1 (~10 s): Muse designs it (parts, printables, steps, wiring, code)."""
    return await _design(req)


@app.post("/api/build/{bid}/render")
async def build_render(bid: str):
    """Stage 2 (~60 s): open image model renders the finished build."""
    return await _render(bid)


@app.post("/api/build/{bid}/mesh")
async def build_mesh(bid: str):
    """Stage 3: image -> 3D mesh (GLB to view, watertight STL to print)."""
    return await _mesh(bid)


@app.get("/api/builds")
async def builds():
    out = []
    for d in sorted(BUILD_DIR.glob("*/design.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        ds = json.loads(d.read_text())
        out.append({"id": d.parent.name, "name": ds.get("name"), "goal": ds.get("_goal"),
                    "has_model": (d.parent / "model.glb").exists()})
    return {"builds": out}


@app.get("/api/build/{bid}")
async def build_get(bid: str):
    """Reopen a saved build (design + render + model)."""
    d = _bdir(bid)
    out = {"id": bid, "design": json.loads((d / "design.json").read_text()), "timings": {}}
    if (d / "render.png").exists():
        out["image"] = _b64.b64encode((d / "render.png").read_bytes()).decode()
    if (d / "model.glb").exists():
        out.update({"glb": _b64.b64encode((d / "model.glb").read_bytes()).decode(),
                    "stl_path": str(d / "model.stl"), "glb_path": str(d / "model.glb")})
        out["mesh"] = json.loads((d / "mesh.json").read_text()) if (d / "mesh.json").exists() else None
    return out


@app.post("/api/build")
async def build(req: BuildReq):
    """All stages in one call: design -> render -> mesh."""
    t0 = time.time()
    out = await _design(req)
    try:
        r = await _render(out["id"])
        out["image"] = r["image"]; out["timings"]["image_s"] = r["image_s"]
        m = await _mesh(out["id"])
        out.update({k: m[k] for k in ("glb", "stl_path", "glb_path", "mesh")}); out["timings"]["shape_s"] = m["shape_s"]
    except Exception as e:
        out["gen_error"] = str(e)[:300]
    out["timings"]["total_s"] = round(time.time() - t0, 1)
    return out


@app.get("/api/health")
async def health():
    return {"ok": True, "model": MODEL, "voice": bool(META_KEY)}


if (ROOT / "static").is_dir():
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index():
    if (ROOT / "static" / "index.html").exists():
        return FileResponse(ROOT / "static" / "index.html")
    return {"ok": True, "service": "Holo agent server", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(ENV.get("HOLO_PORT", 8765)))
