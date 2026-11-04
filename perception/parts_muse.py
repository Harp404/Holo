"""'Muse, I've got these parts - what can I build?' Muse designs from the parts it sees (Build mode designer),
then SAM 3.1 outlines every part Muse named in that frame. Adds an 'answer' reply with a combined highlight.

python parts_muse.py --frames F --frame N --say "..." --replies replies.json [--server URL]
"""
import argparse, base64, io, json, time, urllib.request
import numpy as np
from PIL import Image


def post(url, body, timeout=300):
    r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


ap = argparse.ArgumentParser()
ap.add_argument("--frames"); ap.add_argument("--frame", type=int); ap.add_argument("--say"); ap.add_argument("--replies")
ap.add_argument("--server", default="http://127.0.0.1:8765")
a = ap.parse_args()
img = base64.b64encode(open(f"{a.frames}/{a.frame:04d}.jpg", "rb").read()).decode()
q = a.say.split(",", 1)[-1].strip()
t = time.time()
import os
cache = a.replies + ".design.json"
if os.path.exists(cache):
    d = json.load(open(cache))
else:
    d = post(f"{a.server}/api/build/design", {"image": img, "goal": q})["design"]
    json.dump(d, open(cache, "w"))
lat = round(time.time() - t, 2)
# the parts Muse says are in view: ask it for short visual names of each, then SAM 3.1 each
names = []
for p in d.get("parts", []):
    n = p["name"]
    names.append(n)
comb, found = None, []
import re
for n in names:
    core = re.sub(r"\s*\(.*?\)", "", n).strip()
    s = {}
    for qn in (core, core + " box", " ".join(core.split()[-2:]) + " box"):   # SAM likes short visual phrases
        try:
            s = post(f"{a.server}/api/segment", {"image": img, "text": qn, "role": "grab"}, timeout=90)
        except Exception as e:
            print("seg fail", qn, e); continue
        if s.get("found"):
            break
    if not s.get("found"):
        continue
    ov = np.array(Image.open(io.BytesIO(base64.b64decode(s["overlay"]))).convert("RGBA")).astype(np.float32)
    comb = ov if comb is None else np.maximum(comb, ov)
    found.append(n)
print("design:", d.get("name"), "| parts:", names, "| outlined:", found)
ans = f"You can build a {d.get('name')}. " + (d.get("summary") or "") + " Parts I can see: " + ", ".join(found) + "."
r = {"kind": "answer", "answer": ans, "say": f"Yes. With these parts you can build a {d.get('name')}. I've outlined each one.",
     "target_label": None, "_latency_s": lat, "_design": d}
if comb is not None:
    buf = io.BytesIO(); Image.fromarray(comb.astype(np.uint8), "RGBA").save(buf, "PNG")
    r["_highlight"] = {"overlay": base64.b64encode(buf.getvalue()).decode(), "box": None, "label": ", ".join(found)}
rep = json.load(open(a.replies))
rep = [x for x in rep if x["frame"] != a.frame]
rep.insert(0, {"frame": a.frame, "say": a.say, "kind": "chat", "question": q, "reply": r})
json.dump(rep, open(a.replies, "w"), indent=1)
