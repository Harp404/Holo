"""SAM 3.1 (Object Multiplex) text-prompted video segmentation + tracking over the POV clip.
For each concept prompt, every instance is tracked through the video; masks saved per frame per object id."""
import json, os, sys, cv2, numpy as np, torch
from sam3.model_builder import build_sam3_predictor
frames, out = sys.argv[1], sys.argv[2]
prompts = sys.argv[3].split("|")
pred = build_sam3_predictor(version="sam3.1")
sess = pred.handle_request(dict(type="start_session", resource_path=frames, offload_video_to_cpu=True))["session_id"]
meta = {}
for p in prompts:
    pred.handle_request(dict(type="reset_session", session_id=sess))
    pred.handle_request(dict(type="add_prompt", session_id=sess, frame_index=0, text=p))
    d = os.path.join(out, p.replace(" ", "_")); os.makedirs(d, exist_ok=True)
    stats = {}
    for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sess)):
        fi, o = r["frame_index"], r["outputs"]
        ids = o["out_obj_ids"].tolist(); probs = o.get("out_probs", np.ones(len(ids)))
        for i, oid in enumerate(ids):
            m = o["out_binary_masks"][i]
            if m.any():
                cv2.imwrite(os.path.join(d, f"{fi+1:04d}_{oid}.png"), m.astype(np.uint8) * 255)
                stats.setdefault(str(oid), []).append([fi + 1, float(probs[i])])
    meta[p] = {k: {"frames": len(v), "first": v[0][0], "last": v[-1][0], "mean_prob": round(float(np.mean([x[1] for x in v])), 3)} for k, v in stats.items()}
    print(p, json.dumps(meta[p]), flush=True)
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
print("DONE")
