"""Muse Spark watches the clip and explains the movement: what it is, the intent, the phases, and how to edit it.

python muse_intent.py VIDEO OUT.json
"""
import base64
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

ENV = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
KEY, MODEL = ENV["meta_baseten"], "meta-models/muse-spark-open-1026"
video, out = sys.argv[1], sys.argv[2]

with tempfile.TemporaryDirectory() as d:          # 8 frames across the clip, tiled 4x2 with timestamps
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video]))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf",
                    f"fps={8 / dur},scale=360:-1,drawtext=text='%{{pts\\:hms}}':x=6:y=6:fontsize=18:fontcolor=yellow,tile=4x2",
                    "-frames:v", "1", f"{d}/sheet.jpg"], check=True)
    img = base64.b64encode(open(f"{d}/sheet.jpg", "rb").read()).decode()

PROMPT = """These 8 frames (left to right, top to bottom, timestamps shown) are one movement a humanoid robot must learn to copy.
Describe it like a coach briefing a robot. Return ONLY JSON:
{"action": "short name, e.g. 'taekwondo side kick (yop chagi)'",
 "intent": "one sentence: what the movement is for / its goal",
 "phases": [{"t": "0:00-0:02", "what": "short", "key_body_parts": ["left leg", ...]}],
 "balance": "where the weight is and what keeps it stable",
 "hard_parts": "what will be hardest for a robot to copy and why",
 "edits": ["3 useful variations a user might ask for, e.g. 'kick lower (waist height)'"]}"""
body = {"model": MODEL, "max_tokens": 1500, "reasoning_effort": "low",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}},
                                                  {"type": "text", "text": PROMPT}]}]}
req = urllib.request.Request("https://inference.baseten.co/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json", "Authorization": f"Api-Key {KEY}"})
txt = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
m = re.search(r"\{.*\}", txt, re.S)
res = json.loads(m.group(0))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
