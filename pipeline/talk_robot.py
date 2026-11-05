"""Talk to the humanoid: a Muse plan (sequence of behaviours) -> Meta Motivo reward prompts -> one continuous physics performance.

python talk_robot.py PLAN.json OUT.mp4
PLAN.json: {"title": "...", "steps": [{"task": "raisearms-h-h", "seconds": 3, "say": "raising both arms"}, ...]}
Each task is one of HumEnv's reward functions; Motivo infers a behaviour (z) from rewards relabelled on its own
inference buffer (zero-shot, no training) and the steps run back to back in one simulation.
"""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import h5py
import imageio
import mujoco
import numpy as np
import torch
from humenv import make_humenv
from humenv.env import make_from_name
from metamotivo.fb_cpr.huggingface import FBcprModel
from PIL import Image, ImageDraw, ImageFont

plan = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
N = 60000                                   # buffer samples used for each reward prompt
dev = "cuda"
model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(dev)
env, _ = make_humenv(num_envs=1, render_mode="rgb_array", state_init="Default")
u = env.unwrapped
m, d = u.model, u.data
env_dt = m.opt.timestep * u.action_repeat

# robot look (same as the video pipeline)
for g in range(m.ngeom):
    b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    if b == "world":
        continue
    m.geom_matid[g] = -1
    if b == "Head":
        m.geom_type[g] = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        m.geom_size[g] = [0.085, 0.095, 0.11]
    if any(k in b for k in ("Head", "Hand", "Toe", "Ankle", "Wrist", "Neck", "Elbow", "Knee")):
        m.geom_rgba[g] = (0.13, 0.14, 0.16, 1)
    elif b == "Chest":
        m.geom_rgba[g] = (0.25, 0.55, 1.0, 1)
    else:
        m.geom_rgba[g] = (0.93, 0.94, 0.96, 1)

f = h5py.File("/workspace/r/mm1data/data/buffer_inference_500000.hdf5", "r")
idx = np.sort(np.random.default_rng(0).choice(f["qpos"].shape[0], N, replace=False))
B = {k: f[k][idx] for k in ("next_observation", "next_qpos", "next_qvel", "action")}
sim = mujoco.MjData(m)


def z_for(task):
    """Reward prompt: score Motivo's own buffer states with the task's reward, infer the behaviour vector."""
    rf = make_from_name(task)
    r = np.zeros((N, 1), np.float32)
    for i in range(N):
        r[i] = rf(m, B["next_qpos"][i], B["next_qvel"][i], B["action"][i])
    nobs = torch.tensor(B["next_observation"], dtype=torch.float32, device=dev)
    return model.reward_wr_inference(next_obs=nobs, reward=torch.tensor(r, device=dev))


cache = {}
for s in plan["steps"]:
    if s["task"] not in cache:
        cache[s["task"]] = z_for(s["task"])
        print("prompt ready:", s["task"], flush=True)

R = mujoco.Renderer(m, 720, 1280)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 3.4, 150, -6
FONT = "/workspace/r/Inter-SemiBold.ttf"
fnt = ImageFont.truetype(FONT, 34) if os.path.exists(FONT) else ImageFont.load_default()
fsm = ImageFont.truetype(FONT.replace("SemiBold", "Regular"), 24) if os.path.exists(FONT) else ImageFont.load_default()

obs, _ = env.reset()
ob = torch.tensor(obs["proprio"][None], dtype=torch.float32, device=dev)
frames, log = [], []
look = d.xpos[1].copy()
for k, s in enumerate(plan["steps"]):
    z = cache[s["task"]]
    for t in range(int(s["seconds"] / env_dt)):
        a = model.act(ob, z, mean=True)
        o, *_ = env.step(a.cpu().numpy().ravel())
        ob = torch.tensor(o["proprio"][None], dtype=torch.float32, device=dev)
        look = 0.9 * look + 0.1 * d.xpos[1]
        cam.lookat[:] = [look[0], look[1], 1.0]
        R.update_scene(d, camera=cam)
        im = Image.fromarray(R.render())
        dr = ImageDraw.Draw(im, "RGBA")
        dr.rounded_rectangle((30, 580, 900, 690), 18, fill=(10, 12, 18, 200))
        dr.text((52, 592), f"{k + 1}/{len(plan['steps'])}  {s['say']}", font=fnt, fill=(240, 244, 250))
        dr.text((52, 646), f"Motivo reward prompt: {s['task']}", font=fsm, fill=(150, 200, 255))
        frames.append(np.array(im))
    log.append({"task": s["task"], "pelvis_z": round(float(d.xpos[1][2]), 2)})
imageio.mimsave(OUT, frames, fps=int(round(1 / env_dt)))
json.dump(log, open(OUT.replace(".mp4", "_log.json"), "w"), indent=1)
print("TALK_OK", len(frames), "frames", log)
