"""Run the user's spoken questions (frame + words) live through the Holo server, save Muse's real replies.
Each reply that names a visible target also gets a SAM 3.1 highlight overlay for that frame.

python ask_muse.py --frames RAW --interactions interactions.json --out replies.json [--server http://127.0.0.1:8765]
"""
import argparse, base64, json, time, urllib.request


def post(url, body, timeout=300):
    r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--plan", default="", help="reuse an existing plan reply for kind=plan (same frame + words)")
    a = ap.parse_args()
    goal = ""
    out = []
    for it in json.load(open(a.interactions)):
        img = base64.b64encode(open(f"{a.frames}/{it['frame']:04d}.jpg", "rb").read()).decode()
        q = it["say"].split(",", 1)[-1].strip() if it["say"].lower().startswith(("hey muse", "muse")) else it["say"]
        t = time.time()
        if it["kind"] == "plan":
            r = json.load(open(a.plan)) if a.plan else post(f"{a.server}/api/plan", {"image": img, "goal": q, "mode": "physical"})
            r["kind"] = "plan"
            goal = q
        else:
            r = post(f"{a.server}/api/chat", {"image": img, "text": q, "goal": goal})
        r["_latency_s"] = r.get("_latency_s") or round(time.time() - t, 2)
        lab = r.get("target_label")
        if isinstance(lab, str) and lab:
            try:
                s = post(f"{a.server}/api/segment", {"image": img, "text": lab, "role": "muse"}, timeout=90)
                if s.get("found"):
                    r["_highlight"] = {"overlay": s["overlay"], "box": s["box"], "label": lab}
            except Exception as e:
                print("segment failed:", e)
        out.append({**it, "question": q, "reply": r})
        print(it["frame"], it["kind"], "->", (r.get("summary") or r.get("answer") or "")[:90], "| highlight:", lab)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
