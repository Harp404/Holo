"""Muse Spark turns what you say into a sequence of humanoid behaviours (HumEnv tasks) for Meta Motivo.

python muse_plan.py "raise your arms and then jump" PLAN.json
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

KEY = dotenv_values(Path(__file__).resolve().parent.parent / ".env")["meta_baseten"]
TASKS = """move-ego-<angle>-<speed>: walk/run; angle 0 forward, 90 left, -90 right, 180 back; speed 0 (stand still), 2 (walk), 4 (run)
move-ego-low-<angle>-<speed>: move crouched low (angle 0/90/-90/180, speed 0 or 2)
jump-2: jump up repeatedly
rotate-z-5-0.8 / rotate-z--5-0.8: spin in place left / right
raisearms-<l>-<r>: arm heights, each l (low, down), m (middle, sideways), h (high, up), e.g. raisearms-h-h both up, raisearms-h-l left up only
move-ego-0-2-raisearms-<l>-<r>: walk forward while holding arms
crouch-0: crouch down
sitonground: sit on the floor
lieonground-up / lieonground-down: lie on the back / on the front
split-0.5 / split-1: leg split (half / full)
crawl-0.4-2-u: crawl forward (u face up, d face down)
headstand: headstand"""
PROMPT = f"""You control a physics-simulated humanoid robot. Turn the user's request into a short sequence of behaviours.
Available behaviours (use these exact names):
{TASKS}
User: "{sys.argv[1]}"
Rules: 2-10 steps, 0.6-5 seconds each. Start from standing. Movements that are not in the list
(waving, nodding, marching, dancing, clapping...) you compose yourself from the available behaviours, e.g. by alternating
poses quickly (0.6-0.8 s each) or chaining them. If the user asks for something not in the list, choose the closest
behaviours and say so. Return ONLY JSON:
{{"title": "short title", "steps": [{{"task": "exact name", "seconds": 3, "say": "short caption of what the robot does"}}],
  "note": "one sentence: anything you could not do exactly"}}"""
body = {"model": "meta-models/muse-spark-open-1026", "max_tokens": 1200, "reasoning_effort": "low",
        "messages": [{"role": "user", "content": PROMPT}]}
req = urllib.request.Request("https://inference.baseten.co/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json", "Authorization": f"Api-Key {KEY}"})
txt = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
plan = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
plan["request"] = sys.argv[1]
json.dump(plan, open(sys.argv[2], "w"), indent=1)
print(json.dumps(plan, indent=1))
