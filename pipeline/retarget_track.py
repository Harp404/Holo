"""SAM 3D Body joints -> HumEnv humanoid motion (IK) -> Meta Motivo performs it with physics (zero-shot tracking).

python retarget_track.py BODY.npz OUTDIR [--model facebook/metamotivo-M-1] [--speed 1.0]
Writes: OUTDIR/target.mp4 (the reference motion), OUTDIR/robot.mp4 (Motivo in physics), OUTDIR/motion.npz
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio
import mujoco
import numpy as np
import torch
from humenv import make_humenv
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser()
ap.add_argument("body")
ap.add_argument("out")
ap.add_argument("--model", default="facebook/metamotivo-M-1")
ap.add_argument("--speed", type=float, default=1.0, help="<1 = slow the target down (tracking is better on slower motion)")
ap.add_argument("--video", default="", help="source video: render the robot from the same camera, ghosted over it")
ap.add_argument("--zsmooth", type=int, default=5, help="average Motivo's tracking context over this many steps (less jitter)")
ap.add_argument("--assist", type=float, default=0.0, help="heading assist gain (Nm/rad); 0 = pure Motivo")
ap.add_argument("--hold", type=float, default=0.0, help="position assist gain (N/m) toward the person's path; 0 = none")
ap.add_argument("--steer", type=float, default=0.0, help="closed-loop heading steering gain (training-free, no external force); 0 = off")
ap.add_argument("--steer_pos", type=float, default=0.0, help="closed-loop position steering gain")
ap.add_argument("--ilc", type=int, default=0, help="practice rounds: correct heading/drift from the previous attempt")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

D = np.load(a.body)
kp = D["kp3d"].astype(np.float64)            # (T, 70, 3) camera space: x right, y down, z forward
fps = float(D["fps"])
k2 = D["kp2d"].astype(np.float64)
cam_xyz = kp.copy()

# MHR70 keypoints -> humanoid bodies (person's left = L_)
MAP = {"L_Hip": [9], "R_Hip": [10], "L_Knee": [11], "R_Knee": [12], "L_Ankle": [13], "R_Ankle": [14],
       "L_Toe": [15, 16], "R_Toe": [18, 19], "L_Shoulder": [5], "R_Shoulder": [6], "L_Elbow": [7], "R_Elbow": [8],
       "L_Wrist": [62], "R_Wrist": [41], "Neck": [69], "Head": [3, 4]}
W = {"Head": 0.6, "Neck": 0.8, "L_Toe": 0.7, "R_Toe": 0.7}

# camera -> world (z up). Centre on the first pelvis, put the lowest foot point on the ground.
w = np.stack([kp[..., 0], kp[..., 2], -kp[..., 1]], -1)
pel = (w[:, 9] + w[:, 10]) / 2
w[..., 0] -= pel[0, 0]
w[..., 1] -= pel[0, 1]
CAM = np.array([-pel[0, 0], -pel[0, 1], 0.0])            # original camera position in world (x, depth); z fixed below
feet = w[:, [13, 14, 15, 16, 18, 19]]
w[..., 2] -= np.percentile(feet[..., 2].min(1), 5) - 0.03
CAM[2] = -np.percentile(feet[..., 2].min(1), 5) + 0.03 + 0.0     # camera height = ground offset (camera at y=0)
targets = {b: w[:, ix].mean(1) for b, ix in MAP.items()}
targets["Pelvis"] = (w[:, 9] + w[:, 10]) / 2
T = len(w)

env, _ = make_humenv(num_envs=1, render_mode="rgb_array", state_init="Default")
u = env.unwrapped
m, d = u.model, u.data
env_dt = m.opt.timestep * u.action_repeat


def robot_style():
    """Humanoid-robot look: white shell limbs, graphite head/hands/feet/joints, a little gloss."""
    shell, dark, accent = (0.93, 0.94, 0.96, 1), (0.13, 0.14, 0.16, 1), (0.25, 0.55, 1.0, 1)
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if b in ("world",):
            continue
        m.geom_matid[g] = -1
        if b == "Head":
            m.geom_type[g] = mujoco.mjtGeom.mjGEOM_ELLIPSOID
            m.geom_size[g] = [0.085, 0.095, 0.11]
        if any(k in b for k in ("Head", "Hand", "Toe", "Ankle", "Wrist", "Neck", "Elbow", "Knee")):
            m.geom_rgba[g] = dark
        elif b in ("Chest",):
            m.geom_rgba[g] = accent
        else:
            m.geom_rgba[g] = shell


robot_style()


def set_view(renderer_cam_target):
    """Render from where the real camera was, looking at the robot (same framing as the source video)."""
    cam = u.renderer if hasattr(u, "renderer") else None
    return cam


VIEW = mujoco.MjvCamera()
VIEW.type = mujoco.mjtCamera.mjCAMERA_FREE


def render_from_source():
    lk = d.xpos[bid["Pelvis"]].copy(); lk[2] = 0.9
    v = CAM - lk
    VIEW.lookat[:] = lk
    VIEW.distance = float(np.linalg.norm(v)) * 0.9
    VIEW.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
    VIEW.elevation = float(-np.degrees(np.arcsin(np.clip((lk[2] - CAM[2] - 0.9) / max(1e-3, np.linalg.norm(v)), -1, 1)))) - 6
    if not hasattr(render_from_source, "r"):
        render_from_source.r = mujoco.Renderer(m, 720, 960)
    render_from_source.r.update_scene(d, camera=VIEW)
    return render_from_source.r.render()
bid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
names = [b for b in targets]
hinge_lo = m.jnt_range[1:, 0].copy()
hinge_hi = m.jnt_range[1:, 1].copy()
lim = m.jnt_limited[1:].astype(bool)
hinge_lo[~lim], hinge_hi[~lim] = -np.pi, np.pi


def fk(p):
    d.qpos[:3] = p[:3]
    q = R.from_rotvec(p[3:6]).as_quat()           # x y z w
    d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
    d.qpos[7:] = p[6:]
    mujoco.mj_kinematics(m, d)


def solve(t, p0, prev):
    tg = np.stack([targets[n][t] for n in names])
    wt = np.array([W.get(n, 1.0) for n in names])[:, None]

    def res(p):
        fk(p)
        r = ((d.xpos[[bid[n] for n in names]] - tg) * wt).ravel()
        return np.concatenate([r, 0.05 * (p[6:] - prev[6:]), 0.02 * p[6:]])
    lo = np.concatenate([[-np.inf] * 6, hinge_lo])
    hi = np.concatenate([[np.inf] * 6, hinge_hi])
    return least_squares(res, np.clip(p0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method="trf", max_nfev=200, x_scale="jac").x


# ---- proper retargeting: person's bone directions, humanoid's bone lengths
mujoco.mj_resetData(m, d); d.qpos[2] = 1.0; d.qpos[3] = 1.0; mujoco.mj_kinematics(m, d)
H = {n: d.xpos[bid[n]].copy() for n in names}
L = lambda a_, b_, src: np.linalg.norm(src[b_] - src[a_])      # noqa: E731
CHAINS = [("Pelvis", "L_Hip"), ("L_Hip", "L_Knee"), ("L_Knee", "L_Ankle"), ("L_Ankle", "L_Toe"),
          ("Pelvis", "R_Hip"), ("R_Hip", "R_Knee"), ("R_Knee", "R_Ankle"), ("R_Ankle", "R_Toe"),
          ("Pelvis", "Neck"), ("Neck", "Head"), ("Neck", "L_Shoulder"), ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist"),
          ("Neck", "R_Shoulder"), ("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_Wrist")]
leg_h = L("L_Hip", "L_Knee", H) + L("L_Knee", "L_Ankle", H)
leg_p = np.median(np.linalg.norm(targets["L_Knee"] - targets["L_Hip"], axis=1) + np.linalg.norm(targets["L_Ankle"] - targets["L_Knee"], axis=1))
k_root = leg_h / max(leg_p, 1e-3)
newt = {n: np.zeros_like(v) for n, v in targets.items()}
p0 = targets["Pelvis"][0].copy()
newt["Pelvis"] = p0 * [1, 1, 0] + (targets["Pelvis"] - p0 * [1, 1, 0]) * k_root
newt["Pelvis"][:, :2] = p0[:2] + (targets["Pelvis"][:, :2] - p0[:2]) * k_root
for a_, b_ in CHAINS:
    v = targets[b_] - targets[a_]
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
    newt[b_] = newt[a_] + v * L(a_, b_, H)
fl = np.minimum(newt["L_Toe"][:, 2], newt["R_Toe"][:, 2])
for n in newt:
    newt[n][:, 2] -= np.percentile(fl, 5) - 0.02                  # feet back on the ground
targets = newt
print("retarget: leg ratio", round(float(k_root), 3))

# initial guess: pelvis position, facing from hips/shoulders
hipv = targets["R_Hip"][0] - targets["L_Hip"][0]
yaw = np.arctan2(hipv[1], hipv[0])
p = np.zeros(6 + m.nq - 7)
p[:3] = targets["Pelvis"][0]
p[3:6] = R.from_euler("z", yaw - np.pi / 2).as_rotvec()   # humanoid's L->R hip axis is -x... refined by IK
best = None
for yaw_off in (0, np.pi / 2, np.pi, -np.pi / 2):             # pick the facing that fits best
    q0 = p.copy(); q0[3:6] = R.from_euler("z", yaw + yaw_off).as_rotvec()
    s = solve(0, q0, q0)
    fk(s); err = np.linalg.norm(d.xpos[[bid[n] for n in names]] - np.stack([targets[n][0] for n in names]))
    if best is None or err < best[0]:
        best = (err, s)
P = [best[1]]
for t in range(1, T):
    P.append(solve(t, P[-1], P[-1]))
P = np.array(P)
ik_err = []
for t in range(T):
    fk(P[t]); ik_err.append(np.linalg.norm(d.xpos[[bid[n] for n in names]] - np.stack([targets[n][t] for n in names]), axis=1).mean())
ik_err = np.array(ik_err)
# smooth (keep rotvec continuous)
for j in range(P.shape[1]):
    P[:, j] = savgol_filter(P[:, j], min(9, T - (1 - T % 2)), 3) if T > 9 else P[:, j]

# resample to the env rate (and optional slow-down)
src_t = np.arange(T) / fps
dst_t = np.arange(0, src_t[-1], env_dt * a.speed)
Pr = np.stack([np.interp(dst_t, src_t, P[:, j]) for j in range(P.shape[1])], 1)
qpos = np.zeros((len(Pr), m.nq))
for i, p_ in enumerate(Pr):
    fk(p_); qpos[i] = d.qpos
qvel = np.zeros((len(qpos), m.nv))
for i in range(len(qpos) - 1):
    mujoco.mj_differentiatePos(m, qvel[i], env_dt, qpos[i], qpos[i + 1])
qvel[-1] = qvel[-2]
np.savez_compressed(f"{a.out}/motion.npz", qpos=qpos, qvel=qvel, env_dt=env_dt)
print("IK done:", T, "frames ->", len(qpos), "env steps; fit err m:", round(float(best[0]), 3))

# reference motion video + observations for the prompt
obs, tgt_frames = [], []
tgt_xpos = []
for i in range(len(qpos)):
    u.set_physics(qpos[i], qvel[i])
    obs.append(u.get_obs()["proprio"])
    tgt_xpos.append(d.xpos[1:].copy())
    tgt_frames.append(render_from_source())
imageio.mimsave(f"{a.out}/target.mp4", tgt_frames, fps=int(round(1 / env_dt / a.speed)))

# Meta Motivo: zero-shot tracking prompt -> act with physics
from metamotivo.fb_cpr.huggingface import FBcprModel  # noqa: E402
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = FBcprModel.from_pretrained(a.model).to(dev)


def obs_of(qp):
    qv = np.zeros((len(qp), m.nv))
    for i_ in range(len(qp) - 1):
        mujoco.mj_differentiatePos(m, qv[i_], env_dt, qp[i_], qp[i_ + 1])
    qv[-1] = qv[-2]
    ob_ = []
    for i_ in range(len(qp)):
        u.set_physics(qp[i_], qv[i_]); ob_.append(u.get_obs()["proprio"])
    return torch.tensor(np.array(ob_), dtype=torch.float32, device=dev), qv


ASSIST = []
HOLD = []


def rollout(qp, render=False):
    O_, qv = obs_of(qp)
    zz = model.tracking_inference(next_obs=O_[1:])
    if a.zsmooth > 1:
        zs = torch.nn.functional.avg_pool1d(zz.T[None], a.zsmooth, stride=1, padding=a.zsmooth // 2, count_include_pad=False)[0].T[:len(zz)]
        zz = zs * (zz.norm(dim=1, keepdim=True) / zs.norm(dim=1, keepdim=True).clamp_min(1e-6))
    u.set_physics(qp[0], qv[0])
    ob_ = torch.tensor(u.get_obs()["proprio"][None], dtype=torch.float32, device=dev)
    if a.steer > 0 or a.steer_pos > 0:
        return rollout_steer(qp, qv, render)
    fr_, qs, xs = [], [], []
    tyaw = hip_yaw(np.array([np.array(tgt_xpos)[min(t + 1, len(tgt_xpos) - 1)] for t in range(len(zz))]))
    ASSIST.clear(); HOLD.clear()
    for t in range(len(zz)):
        act = model.act(ob_, zz[t:t + 1], mean=True)
        if a.assist > 0:                       # spotter: gentle yaw torque at the pelvis toward the person's facing
            ry = hip_yaw(d.xpos[1:][None])[0]
            e = (tyaw[t] - ry + np.pi) % (2 * np.pi) - np.pi
            tau = a.assist * e - 0.15 * a.assist * d.qvel[5]
            tau = float(np.clip(tau, -60, 60))
            d.xfrc_applied[bid["Pelvis"], 5] = tau
            ASSIST.append(abs(tau))
        if a.hold > 0:                         # spotter: gentle horizontal pull toward where the person is
            tgt_p = np.array(tgt_xpos)[min(t + 1, len(tgt_xpos) - 1)][0, :2]
            f = a.hold * (tgt_p - d.xpos[bid["Pelvis"], :2]) - 0.3 * a.hold * d.qvel[:2]
            f = np.clip(f, -120, 120)
            d.xfrc_applied[bid["Pelvis"], :2] = f
            HOLD.append(float(np.linalg.norm(f)))
        o, *_ = env.step(act.cpu().numpy().ravel())
        ob_ = torch.tensor(o["proprio"][None], dtype=torch.float32, device=dev)
        if render:
            fr_.append(render_from_source())
        qs.append(d.qpos.copy()); xs.append(d.xpos[1:].copy())
    return fr_, qs, np.array(xs)


def root_yaw(q):
    w_, x_, y_, z_ = q[3:7]
    return np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))


def rollout_steer(qp, qv, render):
    """Closed-loop steering through the command only: each step, the next few target frames get an extra turn rate
    (toward the person's heading) and an extra drift velocity (toward the person's position), expressed the way
    Motivo sees them (heading-relative), then the tracking context is recomputed. No training, no external forces."""
    W_ = 8
    sim = mujoco.MjData(m)
    u.set_physics(qp[0], qv[0])
    ob_ = torch.tensor(u.get_obs()["proprio"][None], dtype=torch.float32, device=dev)
    fr_, qs, xs = [], [], []
    Tn = len(qp) - 1
    for t in range(Tn):
        ry, ty = root_yaw(d.qpos), root_yaw(qp[min(t + 1, Tn)])
        e_yaw = (ty - ry + np.pi) % (2 * np.pi) - np.pi
        w_c = float(np.clip(a.steer * e_yaw, -2.5, 2.5))
        e_xy = qp[min(t + 1, Tn), :2] - d.qpos[:2]
        v_world = np.clip(a.steer_pos * e_xy, -0.8, 0.8)
        c, s_ = np.cos(ty - ry), np.sin(ty - ry)          # robot-heading frame -> target-heading frame
        v_cmd = np.array([c * v_world[0] - s_ * v_world[1], s_ * v_world[0] + c * v_world[1]])
        win = []
        for f in range(t + 1, min(t + 1 + W_, Tn + 1)):
            q_, v_ = qp[f].copy(), qv[f].copy()
            v_[:2] += v_cmd
            v_[5] += w_c                                   # free-joint angular velocity (z) -> extra turn
            u_data = u.data
            sim.qpos[:], sim.qvel[:] = q_, v_
            u.data = sim; mujoco.mj_forward(m, sim)
            win.append(u.get_obs()["proprio"]); u.data = u_data
        zz = model.tracking_inference(next_obs=torch.tensor(np.array(win), dtype=torch.float32, device=dev))[:1]
        act = model.act(ob_, zz, mean=True)
        o, *_ = env.step(act.cpu().numpy().ravel())
        ob_ = torch.tensor(o["proprio"][None], dtype=torch.float32, device=dev)
        if render:
            fr_.append(render_from_source())
        qs.append(d.qpos.copy()); xs.append(d.xpos[1:].copy())
    return fr_, qs, np.array(xs)


def hip_yaw(x):
    hv = x[:, bid["R_Hip"] - 1] - x[:, bid["L_Hip"] - 1]
    return np.arctan2(hv[:, 1], hv[:, 0])


def corrected(qp, cyaw, cxy):
    """turn the target's facing by cyaw(t) and shift it by cxy(t): counter-steer what the robot over-did last time"""
    q2 = qp.copy()
    for t in range(len(q2)):
        r0 = R.from_quat([q2[t, 4], q2[t, 5], q2[t, 6], q2[t, 3]])
        r1 = R.from_euler("z", cyaw[t]) * r0
        x_, y_, z_, w_ = r1.as_quat()
        q2[t, 3:7] = [w_, x_, y_, z_]
        q2[t, :2] += cxy[t]
    return q2


tg_all = np.array(tgt_xpos)
cyaw = np.zeros(len(qpos)); cxy = np.zeros((len(qpos), 2))
best_try = None
for it in range(max(1, a.ilc + 1)):
    _, qs, xs = rollout(corrected(qpos, cyaw, cxy))
    tg_ = tg_all[1:len(xs) + 1]
    dy = (hip_yaw(xs) - hip_yaw(tg_) + np.pi) % (2 * np.pi) - np.pi
    dxy = xs[:, 0, :2] - tg_[:, 0, :2]
    turn = float(np.degrees(np.abs(dy)).mean()); drift_e = float(np.linalg.norm(dxy[-1]))
    print(f"practice {it}: turn {turn:.1f} deg, drift {drift_e * 100:.0f} cm", flush=True)
    if best_try is None or turn + 30 * drift_e < best_try[0]:
        best_try = (turn + 30 * drift_e, cyaw.copy(), cxy.copy(), it)
    k = 0.6
    pad = np.concatenate([[0.0], savgol_filter(dy, min(15, len(dy) - (1 - len(dy) % 2)), 2)])
    cyaw = cyaw - k * pad
    padxy = np.vstack([[0, 0], np.stack([savgol_filter(dxy[:, c], min(15, len(dxy) - (1 - len(dxy) % 2)), 2) for c in range(2)], 1)])
    cxy = cxy - 0.5 * padxy
print("best practice round:", best_try[3])
frames, err, rob_xpos_arr = rollout(corrected(qpos, best_try[1], best_try[2]), render=True)
rob_xpos = list(rob_xpos_arr)
imageio.mimsave(f"{a.out}/robot.mp4", frames, fps=int(round(1 / env_dt / a.speed)))
np.savez_compressed(f"{a.out}/rollout.npz", qpos=np.array(err), env_dt=env_dt)
# ---- same camera as the source video: robot ghosted over the person
if a.video:
    cap = cv2.VideoCapture(a.video); srcf = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        srcf.append(fr)
    Hh, Ww = srcf[0].shape[:2]
    cx, cy = Ww / 2, Hh / 2
    Z = cam_xyz[..., 2]; ok_ = Z > 0.3
    fx = np.median(((k2[..., 0] - cx) * Z / np.where(np.abs(cam_xyz[..., 0]) > 0.05, cam_xyz[..., 0], np.nan))[ok_ & (np.abs(cam_xyz[..., 0]) > 0.05)])
    m.vis.global_.fovy = float(np.degrees(2 * np.arctan(cy / fx)))
    sc = mujoco.MjvCamera(); sc.type = mujoco.mjtCamera.mjCAMERA_FREE
    sc.lookat[:] = CAM + np.array([0, 3.0, 0]); sc.distance = 3.0; sc.azimuth = 90.0; sc.elevation = 0.0
    rr = mujoco.Renderer(m, Hh, Ww)
    floor_ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
    over, alone = [], []
    for t, q in enumerate(err):
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        rr.update_scene(d, camera=sc); img = rr.render()
        rr.enable_segmentation_rendering(); rr.update_scene(d, camera=sc); seg = rr.render(); rr.disable_segmentation_rendering()
        mask = (seg[..., 0] >= 0) & ~np.isin(seg[..., 0], floor_ids) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        src_i = min(len(srcf) - 1, int(round((t + 1) * env_dt * a.speed * fps * (len(srcf) / max(1, T)))))
        base = srcf[src_i].copy()
        bgr = img[..., ::-1]
        o_ = base.copy(); o_[mask] = (0.45 * base[mask] + 0.55 * bgr[mask]).astype(np.uint8)
        over.append(o_[..., ::-1])
        al = np.full_like(base, 18); al[mask] = bgr[mask]; alone.append(al[..., ::-1])
    imageio.mimsave(f"{a.out}/overlay.mp4", over, fps=int(round(1 / env_dt / a.speed)))
    tov = []                                                   # the fitted target motion (before Motivo), same camera
    for t in range(len(qpos)):
        d.qpos[:] = qpos[t]; mujoco.mj_forward(m, d)
        rr.update_scene(d, camera=sc); img = rr.render()
        rr.enable_segmentation_rendering(); rr.update_scene(d, camera=sc); seg = rr.render(); rr.disable_segmentation_rendering()
        mask = (seg[..., 0] >= 0) & ~np.isin(seg[..., 0], floor_ids) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        src_i = min(len(srcf) - 1, int(round(t * env_dt * a.speed * fps * (len(srcf) / max(1, T)))))
        base = srcf[src_i].copy(); bgr = img[..., ::-1]
        o_ = base.copy(); o_[mask] = (0.45 * base[mask] + 0.55 * bgr[mask]).astype(np.uint8)
        tov.append(o_[..., ::-1])
    imageio.mimsave(f"{a.out}/overlay_target.mp4", tov, fps=int(round(1 / env_dt / a.speed)))
    imageio.mimsave(f"{a.out}/robot_samecam.mp4", alone, fps=int(round(1 / env_dt / a.speed)))
    print("overlay: fovy", round(m.vis.global_.fovy, 1), "fx", round(float(fx), 1))

# ---- report card
tg = np.array(tgt_xpos[1:len(rob_xpos) + 1]); rb = np.array(rob_xpos)
n = min(len(tg), len(rb)); tg, rb = tg[:n], rb[:n]
def rel(x):
    """posture: relative to the pelvis and turned to face the same way (yaw from the hip line)"""
    x = x - x[:, :1]
    hv = x[:, bid["R_Hip"] - 1] - x[:, bid["L_Hip"] - 1]
    yaw = np.arctan2(hv[:, 1], hv[:, 0])
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    Rz = np.stack([np.stack([c, -s_, 0 * c], -1), np.stack([s_, c, 0 * c], -1), np.stack([0 * c, 0 * c, 1 + 0 * c], -1)], -2)
    return np.einsum("tij,tkj->tki", Rz, x)
mpjpe = np.linalg.norm(rel(rb) - rel(tg), axis=-1).mean(1)      # per step, metres
best_lag, best_e = 0, mpjpe.mean()
for lag in range(1, 9):                                         # robot running a few frames behind?
    e = np.linalg.norm(rel(rb[lag:]) - rel(tg[:-lag]), axis=-1).mean()
    if e < best_e:
        best_lag, best_e = lag, e
glob = np.linalg.norm(rb - tg, axis=-1).mean(1)
pel = rb[:, 0, 2]
fell = bool((pel < 0.5).any())
foot = [bid[k] - 1 for k in ("L_Ankle", "R_Ankle", "L_Toe", "R_Toe")]
kick_t, kick_r = tg[:, foot, 2].max(), rb[:, foot, 2].max()
def yaw_of(x):
    hv = x[:, bid["R_Hip"] - 1] - x[:, bid["L_Hip"] - 1]
    return np.degrees(np.arctan2(hv[:, 1], hv[:, 0]))
dyaw = (yaw_of(rb) - yaw_of(tg) + 180) % 360 - 180
drift = np.linalg.norm(rb[:, 0, :2] - tg[:, 0, :2], axis=1)
rep = {"assist_gain": a.assist, "assist_torque_mean_Nm": round(float(np.mean(ASSIST)) if ASSIST else 0.0, 1),
       "hold_gain": a.hold, "hold_force_mean_N": round(float(np.mean(HOLD)) if HOLD else 0.0, 1),
       "turn_error_deg_mean": round(float(np.abs(dyaw).mean()), 1), "turn_error_deg_end": round(float(abs(dyaw[-1])), 1),
       "drift_cm_end": round(float(drift[-1] * 100), 1), "model": a.model, "zsmooth": a.zsmooth,
       "pose_error_cm": round(float(mpjpe.mean() * 100), 1), "pose_error_lag_aligned_cm": round(float(best_e * 100), 1), "lag_frames": best_lag, "pose_error_worst_cm": round(float(mpjpe.max() * 100), 1),
       "position_error_cm": round(float(glob.mean() * 100), 1), "fell": fell,
       "kick_height_target_m": round(float(kick_t), 2), "kick_height_robot_m": round(float(kick_r), 2),
       "kick_reached_pct": round(float(100 * kick_r / max(kick_t, 1e-3)), 0), "ik_fit_cm": round(float(ik_err.mean()) * 100, 1), "speed": a.speed}
import json  # noqa: E402
json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
print("REPORT", json.dumps(rep))
print("TRACK_OK", len(frames), "steps")
