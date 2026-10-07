"""Closed-loop rollouts: fly the same scenario twice with the same controller,
once with MuJoCo as the plant (ground truth) and once with the learned model as
the plant, and compare.

WHY CLOSED LOOP
A quadrotor is open-loop unstable: replaying the logged rotor speeds from a
recorded start diverges from tiny errors whatever the model quality, so the
cube-style open-loop rollout says nothing here. But the drone is never flown
open loop. With the controller in the loop, both plants are pulled toward the
same reference, so the two trajectories stay comparable and their difference
measures model error as a controller experiences it. This is also exactly how
the model will be used inside the MPC.

WHAT IS SHARED BETWEEN THE TWO RUNS
  scenario   wall pose, heading, reference waypoints and push profile, wind
  controller the generator's geometric controller and gains, run ONCE PER FRAME
             (100 Hz) from a finite-difference state estimate (velocity and
             angular velocity from the last two poses, like a mocap system).
             The training data used 500 Hz control with the true velocity;
             here the controller must only see what both plants provide.
  excitation the same random rotor-command noise sequence
  motors     the same first-order lag (the learned plant reproduces MuJoCo's
             filter update exactly), and the same Verlet-aligned rotor speeds
  start      hovering at the reference start
The only difference is the plant: MuJoCo (with per-rotor drag) or the model.

Outputs per run: COM and attitude per frame, contact and aero forces (MuJoCo
labels, or the model's predictions in N), per-pad-node contact forces.
"""

import dataclasses
import math

import mujoco
import numpy as np
import torch

import mujoco_drone_generator as MG
from drone_config import DroneConfig
from force_gns import so3_log as so3_log_t


# ======================================================================
# Small numpy helpers
# ======================================================================

def _so3_log(R):
    return so3_log_t(torch.as_tensor(R, dtype=torch.float64).unsqueeze(0))[0].numpy()


def _orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


# ======================================================================
# Scenario (everything both runs share)
# ======================================================================

def sample_eval_scenario(cfg_true, gs, rng, kind=None):
    """Same distributions as the generator (wall yaw/tilt, heading offset,
    scenario script, wind, gains, excitation), without gusts. The rotor-command
    excitation is drawn per frame, since the controller runs per frame here."""
    c_off = np.asarray(cfg_true.com_offset)
    dt = gs.timestep * gs.substeps
    M = len(cfg_true.rotor_pos)

    psi = rng.uniform(0, 2 * math.pi)
    beta = math.radians(rng.uniform(-1, 1) * gs.wall_tilt_max_deg) if rng.random() < gs.tilted_wall_prob else 0.0
    n = np.array([-math.cos(beta) * math.cos(psi), -math.cos(beta) * math.sin(psi), math.sin(beta)])
    wall_point = np.array([0.0, 0.0, 1.5])

    nh = -n.copy()
    nh[2] = 0.0
    nh /= np.linalg.norm(nh)
    yo = math.radians(rng.uniform(-1, 1) * gs.yaw_offset_max_deg)
    heading = np.array([[math.cos(yo), -math.sin(yo), 0], [math.sin(yo), math.cos(yo), 0], [0, 0, 1]]) @ nh

    if kind is None:
        kinds, probs = zip(*gs.scenario_probs)
        kind = str(rng.choice(kinds, p=np.asarray(probs) / sum(probs)))
    lever = float(np.linalg.norm((np.asarray(cfg_true.rod_tip) - c_off)[:2]))
    yaw_cap = lambda c_h: gs.yaw_torque_budget / (lever * (gs.mu * c_h + abs(math.sin(yo))) + 1e-9)
    s0, seg = MG.make_scenario(kind, rng, cfg_true.pad_contact_radius, rng.uniform(*gs.duration), yaw_cap)
    T_frames = int(MG.Reference(s0, seg).T / dt) + 1

    w_mean = np.zeros(3)
    if rng.random() > 0.3:
        w_ang = rng.uniform(0, 2 * math.pi)
        w_mean = rng.uniform(0, gs.wind_max) * np.array([math.cos(w_ang), math.sin(w_ang), 0.0])

    gains = dict(wp=4.0 * rng.uniform(0.8, 1.2), wr=15.0 * rng.uniform(0.8, 1.2), wy=4.0 * rng.uniform(0.8, 1.2))

    ex_sig, ex_tau = rng.uniform(0, gs.excite_max), 0.05
    ex = np.zeros((T_frames, M))
    e = np.zeros(M)
    for k in range(T_frames):
        e = e - e * dt / ex_tau + ex_sig * math.sqrt(2 * dt / ex_tau) * rng.normal(size=M)
        ex[k] = e

    return dict(kind=kind, s0=s0, seg=seg, n=n, wall_point=wall_point, heading=heading, w_mean=w_mean,
                gains=gains, ex=ex, T_frames=T_frames, yaw_offset_deg=math.degrees(yo),
                wall_tilt_deg=math.degrees(beta), excitation=ex_sig)


def _controller(cfg_true, gs, gains):
    t_hover = cfg_true.mass * cfg_true.gravity / len(cfg_true.rotor_pos)
    ctrl = MG.GeometricController(cfg_true, np.random.default_rng(0), gs.t_max_over_hover * t_hover)
    wp, wr, wy = gains["wp"], gains["wr"], gains["wy"]
    Jd = np.diag(ctrl.J)
    bw = np.array([wr, wr, wy])
    ctrl.kp, ctrl.kv = wp ** 2, 2.0 * 0.9 * wp
    ctrl.kR, ctrl.kW = Jd * bw ** 2, Jd * 2.0 * 0.9 * bw
    ctrl.gains = dict(gains)
    return ctrl


def _reference(sc, cfg_true):
    ref = MG.Reference(sc["s0"], sc["seg"])
    n, wp = sc["n"], sc["wall_point"]
    t1, t2 = MG.wall_tangents(n)
    c_off = np.asarray(cfg_true.com_offset)
    R_h = np.column_stack([sc["heading"], np.cross([0, 0, 1.0], sc["heading"]), [0, 0, 1.0]])
    tip_off = R_h @ (np.asarray(cfg_true.rod_tip) - c_off)

    def p_com_ref(t):
        r, ph = ref(t)
        return wp + r["d"] * n + r["a"] * t1 + r["b"] * t2 - tip_off, r["F"], ph

    return p_com_ref, R_h


# ======================================================================
# The loop (shared by both plants)
# ======================================================================

def _run(sc, cfg_true, gs, start_plant, step_plant):
    """start_plant(com0, R0) -> None; step_plant(k, u) -> (com_next, R_next).
    Returns per-frame COM, attitude, phase, commands, and frames completed."""
    dt = gs.timestep * gs.substeps
    ctrl = _controller(cfg_true, gs, sc["gains"])
    p_com_ref, R_h = _reference(sc, cfg_true)
    T = sc["T_frames"]
    com0 = p_com_ref(0.0)[0]
    start_plant(com0, R_h)

    coms, Rs, phase, cmds = [com0.copy()], [R_h.copy()], [], []
    crash = None
    for k in range(T):
        t = k * dt
        com, R = coms[-1], Rs[-1]
        if not (np.all(np.isfinite(com)) and np.all(np.isfinite(R))) or com[2] < 0.2 or R[2, 2] < 0.5:
            crash = f"state left the envelope at t={t:.2f}s"
            break
        # Finite-difference state estimate (what a mocap system gives).
        com_p, R_p = (coms[-2], Rs[-2]) if len(coms) > 1 else (com, R)
        v = (com - com_p) / dt
        w_b = _so3_log(R_p.T @ R) / dt
        eps = 1e-3
        p_d, F_push, ph = p_com_ref(t)
        v_d = (p_com_ref(t + eps)[0] - p_com_ref(t - eps)[0]) / (2 * eps)
        a_d = (p_com_ref(t + eps)[0] - 2 * p_d + p_com_ref(t - eps)[0]) / eps ** 2
        T_cmd = ctrl(com, v, R, w_b, p_d, v_d, a_d, -sc["n"] * F_push, sc["heading"])
        u = np.clip(T_cmd * (1.0 + sc["ex"][k]), 0.0, ctrl.t_max)
        phase.append(ph)
        cmds.append(u)
        try:
            com_n, R_n = step_plant(k, u)
        except MG.Crash as e:
            crash = str(e)
            break
        coms.append(com_n)
        Rs.append(R_n)
    n_done = len(phase)
    return dict(com=np.asarray(coms[:n_done]), R=np.asarray(Rs[:n_done]), phase=np.asarray(phase),
                cmd=np.asarray(cmds), n_frames=n_done, completed=crash is None, crash=crash)


# MuJoCo as the plant. Returns the run plus aligned forces in the generator's
# trajectory format (so it can also be loaded and used like any other data).
def run_mujoco(sc, cfg_true, gs, model_mj=None):
    model_mj = model_mj or mujoco.MjModel.from_xml_string(MG.build_xml(cfg_true, gs))
    plant = MG.MujocoDronePlant(model_mj, cfg_true, gs)
    d = plant.data
    S, h = gs.substeps, gs.timestep
    M = len(cfg_true.rotor_pos)
    t_hover = cfg_true.mass * cfg_true.gravity / M
    n_sub = sc["T_frames"] * S
    logs = dict(thrust=np.zeros((n_sub, M)), Fa=np.zeros((n_sub, 3)), Fc=np.zeros((n_sub, 3)),
                tc=np.zeros((n_sub, 3)), pad=np.zeros((n_sub, len(plant.pad_gids), 3)))

    def start(com0, R0):
        MG.set_wall_pose(model_mj, d, sc["n"], sc["wall_point"])
        d.qpos[0:3] = com0 - R0 @ plant.c_off
        d.qpos[3:7] = MG._mat2quat(R0)
        d.qvel[:] = 0.0
        d.act[:] = t_hover
        d.ctrl[:] = t_hover
        mujoco.mj_forward(model_mj, d)

    def step(k, u):
        d.ctrl[:] = u
        for j in range(S):
            i = k * S + j
            logs["thrust"][i], logs["Fa"][i], logs["Fc"][i], logs["tc"][i], logs["pad"][i] = \
                plant.substep(sc["w_mean"], i * h)
        com, R, _, _ = plant.state()
        return com, R

    run = _run(sc, cfg_true, gs, start, step)
    T = run["n_frames"]
    # Frame 0: thrust at hover (the start state), forces copied from frame 1.
    al = {k: MG.align_to_frames(v[:T * S], S, T, first=np.full(M, t_hover) if k == "thrust" else None)
          for k, v in logs.items()}
    run.update(F_contact=al["Fc"], tau_contact=al["tc"], F_pad=al["pad"], F_aero=al["Fa"],
               rotor_speed=np.sqrt(np.maximum(al["thrust"], 0.0) / cfg_true.k_f), thrust_log=logs["thrust"][:T * S])
    return run


def run_to_traj(run, sc, cfg_true, gs):
    """MuJoCo closed-loop run -> the generator's trajectory dict (loadable data)."""
    c_off = np.asarray(cfg_true.com_offset)
    pos = run["com"] - np.einsum('tij,j->ti', run["R"], c_off)
    f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    traj = dict(pos=torch.tensor(pos, dtype=torch.float64),
                quat=f32([MG._mat2quat(R) for R in run["R"]]),
                rotor_speed=f32(run["rotor_speed"]), wind=f32(np.tile(sc["w_mean"], (run["n_frames"], 1))),
                phase=torch.tensor(run["phase"], dtype=torch.long),
                F_contact=f32(run["F_contact"]), tau_contact=f32(run["tau_contact"]),
                F_pad=f32(run["F_pad"]), F_aero=f32(run["F_aero"]),
                wall_normal=f32(sc["n"]), wall_point=f32(sc["wall_point"]))
    traj["meta"] = dict(dt=gs.timestep * gs.substeps, gravity=cfg_true.gravity, rotor_speed_hold="aligned",
                        mass=cfg_true.mass, inertia=cfg_true.inertia, com_offset=cfg_true.com_offset,
                        k_f=cfg_true.k_f, k_m=cfg_true.k_m, mu=gs.mu, k_rotor=gs.k_rotor, scenario=sc["kind"],
                        wind_mean=sc["w_mean"].tolist(), excitation=sc["excitation"],
                        wall_tilt_deg=sc["wall_tilt_deg"], yaw_offset_deg=sc["yaw_offset_deg"],
                        control="frame_rate_fd_estimate", **sc["gains"])
    return traj


# The learned model as the plant. Motor lag is simulated per MuJoCo timestep
# with MuJoCo's own filter update (thrust during a substep = activation before
# that substep's update), then aligned to frames exactly like the data.
@torch.no_grad()
def run_model(sc, cfg_true, gs, model, device="cpu", use_contact=True):
    model.eval()
    S, h, tau = gs.substeps, gs.timestep, gs.motor_tau
    M = len(cfg_true.rotor_pos)
    t_hover = cfg_true.mass * cfg_true.gravity / M
    w_tri = np.r_[np.arange(S), S - np.arange(S)] / S ** 2
    st = dict(hist=[], act=np.full(M, t_hover), prev=np.full((S, M), t_hover))
    out = dict(F_contact=[], F_pad=[], F_aero=[], rotor_speed=[], c_w=[])
    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device).unsqueeze(0)
    wall_n, wind = f32(sc["n"]), f32(sc["w_mean"])

    def start(com0, R0):
        st["hist"] = [(com0.copy(), R0.copy())] * (model.h + 1)

    def step(k, u):
        this = np.empty((S, M))
        for j in range(S):
            this[j] = st["act"]
            st["act"] = st["act"] + h * (u - st["act"]) / tau
        thrust = w_tri @ np.vstack([st["prev"], this])
        st["prev"] = this
        omega = np.sqrt(np.maximum(thrust, 0.0) / cfg_true.k_f)

        anchor = st["hist"][-1][0]                                   # re-center, as in training
        com_h = [f32(c - anchor) for c, _ in st["hist"]]
        R_h = [f32(R) for _, R in st["hist"]]
        com_n, R_n, aux = model.step(com_h, R_h, f32(omega), wind, wall_n, f32(sc["wall_point"] - anchor),
                                     use_contact=use_contact)
        out["F_contact"].append(model.to_newtons(aux["f_c"])[0].cpu().numpy())
        out["F_pad"].append(model.to_newtons(aux["phi_c"])[0].cpu().numpy() if use_contact
                            else np.zeros((model.contact_rest.shape[0], 3)))
        out["F_aero"].append(model.to_newtons(aux["aero"]["F"])[0].cpu().numpy())
        out["c_w"].append(aux["c_w"][0, :, 0].cpu().numpy() if use_contact
                          else np.zeros(model.contact_rest.shape[0]))
        out["rotor_speed"].append(omega)
        com_next = com_n[0].double().cpu().numpy() + anchor
        R_next = _orthonormalize(R_n[0].double().cpu().numpy())
        st["hist"] = st["hist"][1:] + [(com_next, R_next)]
        return com_next, R_next

    run = _run(sc, cfg_true, gs, start, step)
    T = run["n_frames"]
    for k, v in out.items():
        run[k] = np.asarray(v[:T])
    return run


# One-step (teacher-forced) predictions along a recorded run: at every frame the
# model sees the TRUE history and rotor speeds, so these forces have no
# compounding error. They measure the force decomposition directly.
@torch.no_grad()
def one_step_forces(model, run, sc, device="cpu"):
    model.eval()
    T = run["n_frames"]
    hN = model.h + 1
    idx = np.clip(np.arange(T)[:, None] + np.arange(-model.h, 1)[None, :], 0, None)       # (T, h+1)
    com = run["com"][idx] - run["com"][:, None, :]                                     # re-centered
    t64 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    com_h = [t64(com[:, j]) for j in range(hN)]
    R_h = [t64(run["R"][idx[:, j]]) for j in range(hN)]
    wall_n = t64(np.tile(sc["n"], (T, 1)))
    wall_c = t64(sc["wall_point"][None] - run["com"])
    _, _, aux = model.step(com_h, R_h, t64(run["rotor_speed"]), t64(np.tile(sc["w_mean"], (T, 1))), wall_n, wall_c)
    return dict(F_contact=model.to_newtons(aux["f_c"]).cpu().numpy(),
                F_pad=model.to_newtons(aux["phi_c"]).cpu().numpy(),
                F_aero=model.to_newtons(aux["aero"]["F"]).cpu().numpy())


# ======================================================================
# Metrics
# ======================================================================

def pad_center_positions(run, cfg_true):
    tip = np.asarray(cfg_true.rod_tip) - np.asarray(cfg_true.com_offset)
    return run["com"] + np.einsum('tij,j->ti', run["R"], tip)


def closed_loop_metrics(true, pred, onestep, sc, cfg_true, touch_N=0.2, impact_N=10.0):
    T = min(true["n_frames"], pred["n_frames"])
    n = sc["n"]
    pt, pp = pad_center_positions(true, cfg_true)[:T], pad_center_positions(pred, cfg_true)[:T]
    err = 1e3 * np.linalg.norm(pt - pp, axis=1)
    Fn_t = true["F_contact"][:T] @ n
    touching = Fn_t > touch_N
    out = dict(completed=float(pred["completed"]), frames_survived_frac=pred["n_frames"] / true["n_frames"],
               pad_err_mean_mm=float(err.mean()), pad_err_max_mm=float(err.max()),
               pad_err_contact_mm=float(err[touching].mean()) if touching.any() else float("nan"),
               pad_err_free_mm=float(err[~touching].mean()) if (~touching).any() else float("nan"))

    # Heading (yaw) and full attitude difference between the two runs.
    def heading(R):
        x = R[:, :, 0]
        return np.arctan2(x[:, 1], x[:, 0])
    dyaw = np.degrees(np.angle(np.exp(1j * (heading(pred["R"][:T]) - heading(true["R"][:T])))))
    R_rel = np.einsum('tji,tjk->tik', true["R"][:T], pred["R"][:T])
    att = np.degrees(np.arccos(np.clip((np.trace(R_rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    out.update(yaw_err_mean_deg=float(np.abs(dyaw).mean()), yaw_err_max_deg=float(np.abs(dyaw).max()),
               yaw_err_contact_deg=float(np.abs(dyaw[touching]).mean()) if touching.any() else float("nan"),
               attitude_err_mean_deg=float(att.mean()))

    def onset(Fn):
        idx = np.flatnonzero(Fn > touch_N)
        return idx[0] if idx.size else None

    dt = true["com"].shape[0] and (1.0 / 100.0)
    a, b = onset(Fn_t), onset(pred["F_contact"][:T] @ n)
    out["contact_onset_err_ms"] = float(1e3 * (b - a) * dt) if (a is not None and b is not None) else float("nan")

    sustained = touching & (np.linalg.norm(true["F_contact"][:T], axis=1) < impact_N)
    for name, F in (("closed_loop", pred["F_contact"][:T]), ("one_step", onestep["F_contact"][:T])):
        if sustained.any():
            d = F[sustained] - true["F_contact"][:T][sustained]
            out[f"{name}_sustained_contact_rmse_N"] = float(np.sqrt((d ** 2).sum(1).mean()))
            out[f"{name}_sustained_normal_mean_err_N"] = float((d @ n).mean())
    if sustained.any():
        out["sustained_contact_label_rms_N"] = float(np.sqrt((true["F_contact"][:T][sustained] ** 2).sum(1).mean()))
    # High-frequency content of the normal force during sustained contact:
    # RMS of the frame-to-frame change (N per 10 ms step), MuJoCo vs. model.
    both = sustained[1:] & sustained[:-1]
    if both.any():
        for name, F in (("label", true["F_contact"][:T]), ("one_step", onestep["F_contact"][:T]),
                        ("closed_loop", pred["F_contact"][:T])):
            dF = np.diff(F @ n)[both]
            out[f"fn_jitter_{name}_N"] = float(np.sqrt((dF ** 2).mean()))
    da = onestep["F_aero"][:T] - true["F_aero"][:T]
    out["one_step_aero_rmse_N"] = float(np.sqrt((da ** 2).sum(1).mean()))
    out["aero_label_rms_N"] = float(np.sqrt((true["F_aero"][:T] ** 2).sum(1).mean()))
    return out


def evaluate_closed_loop(model, cfg_true, gs, n_runs=6, seed=1234, kinds=("tap", "push", "slide"),
                         device="cpu", verbose=True):
    """Runs n_runs scenarios (cycling through `kinds`) with both plants.
    Returns (list of per-run dicts with runs + metrics, averaged metrics)."""
    model_mj = mujoco.MjModel.from_xml_string(MG.build_xml(cfg_true, gs))
    chk = MG.true_config_from_model(model_mj, cfg_true, gs)
    if abs(chk.mass - cfg_true.mass) > 1e-6:
        raise ValueError("MuJoCo model built from this config does not match it (mass differs)")
    results = []
    for i in range(n_runs):
        kind = kinds[i % len(kinds)] if kinds else None
        attempt = 0
        while True:
            sc = sample_eval_scenario(cfg_true, gs, np.random.default_rng([seed, i, attempt]), kind=kind)
            true = run_mujoco(sc, cfg_true, gs, model_mj)
            if true["completed"]:
                break
            attempt += 1
            if attempt > 10:
                raise RuntimeError("MuJoCo run keeps crashing; check the eval settings")
        pred = run_model(sc, cfg_true, gs, model, device)
        onestep = one_step_forces(model, true, sc, device)
        m = closed_loop_metrics(true, pred, onestep, sc, cfg_true)
        results.append(dict(scenario=sc, true=true, pred=pred, one_step=onestep, metrics=m))
        if verbose:
            print(f"  closed loop {i}: {sc['kind']:6s} {true['n_frames']} frames | "
                  f"{'completed' if pred['completed'] else 'MODEL RUN STOPPED: ' + pred['crash']} | "
                  f"pad err mean {m['pad_err_mean_mm']:.1f} mm (contact {m['pad_err_contact_mm']:.1f}) | "
                  f"yaw err {m['yaw_err_mean_deg']:.1f} deg (max {m['yaw_err_max_deg']:.1f}) | "
                  f"onset err {m['contact_onset_err_ms']:.0f} ms | sustained contact RMSE: one-step "
                  f"{m.get('one_step_sustained_contact_rmse_N', float('nan')):.2f} N, closed-loop "
                  f"{m.get('closed_loop_sustained_contact_rmse_N', float('nan')):.2f} N "
                  f"(label RMS {m.get('sustained_contact_label_rms_N', float('nan')):.2f})", flush=True)
    keys = sorted({k for r in results for k in r["metrics"]})
    avg = {k: float(np.nanmean([r["metrics"].get(k, np.nan) for r in results])) for k in keys}
    return results, avg


def gen_settings_for(meta, **overrides):
    """GenSettings matching the training data (mu, k_rotor from its metadata)."""
    gs = MG.GenSettings()
    upd = {k: meta[k] for k in ("mu", "k_rotor") if meta and k in meta}
    upd.update(overrides)
    return dataclasses.replace(gs, **upd)


def config_from_checkpoint(ck):
    return DroneConfig(**ck["drone_cfg"])
