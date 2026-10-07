"""Generate MuJoCo trajectories of the aerial manipulator touching, pushing on,
and sliding along a wall, in the format drone_data.py loads.

HOW THE DRONE IS FLOWN
The model never learns the controller. A controller is only needed so the sim
drone goes where we want it; what gets recorded is the motion plus the rotor
speeds that actually produced it, and the model takes those rotor speeds as
inputs. So any controller that reliably makes the contacts we want is fine.

  reference      a randomized scenario (free flight, near-wall hover, tap,
                 push, push-and-slide) written as pad-center waypoints in the
                 wall frame, smoothed with min-jerk segments, plus a push-force
                 profile
  controller     standard geometric (SE(3)) position + attitude controller
                 (Lee et al. 2010) at 500 Hz, with a slow yaw loop. Pushing is a
                 feedforward force into the wall on top of position tracking,
                 so the controller needs no force measurement
  mixer          desired collective thrust + body torques -> 4 rotor thrusts,
                 roll/pitch first, then as much yaw as still fits
  excitation     random band-limited noise on the rotor commands, so rotor
                 speeds are not a fixed function of the state (closed-loop
                 data is otherwise biased toward the controller's own choices)
  motors         MuJoCo first-order filter on thrust (motor lag)

PHYSICS IN THE SIM
  rotor thrust + yaw reaction: site actuators, gear = [0 0 1 0 0 -s*k_m/k_f]
  body aero:  MuJoCo ellipsoid fluid model with opt.wind (gusts via OU process)
  rotor drag: per rotor, k_rotor * m * omega_i * u_perp_i at that rotor's hub,
              where u_perp_i is the hub's local airspeed (wind minus hub velocity,
              including rotation) with the rotor-axis component removed. Applied
              through xfrc_applied as the summed force plus its torque about the
              COM. Same law and coefficient as the aero GNN's rotor anchor.
              (MuJoCo has no rotor drag; without it sim aero is unlike hardware.)
  pad:        one small collision sphere per pad node from build_drone_graph(),
              so the sim contact points ARE the graph's pad nodes
  NOT modeled: rotor downwash / near-wall rotor aero, rotor gyroscopics, flex

WHAT IS RECORDED (per frame, dt = timestep * substeps)
  pos, quat           body-frame origin (= mocap origin), wxyz
  rotor_speed         Verlet-aligned: sqrt of the triangle-weighted mean of
                      thrust/k_f over the two intervals around the frame, which
                      is exactly what the second difference sees (meta says
                      "aligned"). With high-rate ESC telemetry, hardware can
                      compute the same thing.
  rotor_speed_cmd     commanded speed at the frame (for a motor model later)
  wind, wall_normal, wall_point, phase (commanded scenario phase)
  labels (aligned the same way, validation only):
      F_contact, tau_contact (about the COM), F_pad (per pad sphere), F_aero
  meta                dt, gravity, mass/inertia/COM read from the compiled
                      model, k_f, k_m, mu, scenario, and generator settings

STABILITY NOTES (learned building this; they apply to hardware too)
  - Pushing needs nose-down pitch (about atan(F/mg)). With a level rod at or
    below the COM, that pitch drops the pad below the COM, and the wall's
    normal force then pitches the drone further into the wall: a runaway once
    the moment exceeds pitch authority. With the old level rod 2 cm below the
    rotor plane, ~13% of slides ended this way; angling the rod up ~8 deg
    (DroneConfig.rod_tip z = +5 cm) removed it in 24/24 trials.
  - Yaw authority is ~1.6 cm of lever (k_m/k_f) against the rod's ~0.4 m.
    Horizontal friction, and the normal force on a yawed rod (which grows with
    the yaw error), can out-torque it, so push forces are capped from a yaw budget.

Run:  python mujoco_drone_generator.py --out data/mj_drone --n 300
      python mujoco_drone_generator.py --out data/mj_drone --n 5 --verify
      python mujoco_drone_generator.py --replay data/mj_drone/3.pt   (viewer)
On ROAR, split with --start (array task index * n) so seeds never overlap.
"""

import argparse
import dataclasses
import json
import math
import os

import mujoco
import numpy as np
import torch

from drone_config import DroneConfig, build_drone_graph

PHASES = {"free": 0, "approach": 1, "hold": 2, "slide": 3, "release": 4, "hover_near": 5}


@dataclasses.dataclass
class GenSettings:
    timestep: float = 0.0005
    substeps: int = 20              # dt = 0.01 s, 100 Hz recorded
    ctrl_every: int = 4             # controller at 500 Hz
    duration: tuple = (4.0, 7.0)    # seconds, random per trajectory
    motor_tau: float = 0.025        # s, thrust first-order lag
    t_max_over_hover: float = 2.5
    mu: float = 0.4                 # wall friction (fixed per dataset: the model learns one mu)
    k_rotor: float = 5e-5           # rotor drag / m, per rotor, per (rad/s)
    wall_solref: tuple = (0.004, 1.0)
    # Pad contact softness (MuJoCo averages the pad's and wall's solref). Equal
    # to wall_solref = rigid pad: it touches on ONE edge point unless aligned to
    # well under a degree. A larger time constant mimics a rubber pad, which
    # spreads load over several nodes. Match the real pad.
    pad_solref: tuple = (0.004, 1.0)
    wind_max: float = 3.0           # m/s mean wind
    gust_std: float = 0.4           # m/s
    excite_max: float = 0.04        # max relative rotor-command noise
    tilted_wall_prob: float = 0.3
    wall_tilt_max_deg: float = 10.0
    yaw_offset_max_deg: float = 10.0
    yaw_torque_budget: float = 0.15 # N m of yaw torque the pad's moments may use (see make_scenario)
    scenario_probs: tuple = (("free", 0.10), ("hover_near", 0.15), ("tap", 0.20),
                             ("push", 0.25), ("slide", 0.30))


# ======================================================================
# Model
# ======================================================================

def _v(x):
    return " ".join(f"{float(a):.6g}" for a in x)


# Builds the MJCF from the DroneConfig geometry. Mass properties are NOT taken
# from the config: MuJoCo computes them from the geoms, and they are read back
# with true_config_from_model() so training uses exactly the simulated values.
def build_xml(cfg, gs):
    g_pts = build_drone_graph(cfg, com_relative=False)
    pts = g_pts["rest_nodes"].numpy()
    pad_idx = g_pts["pad_indices"]
    tip = np.asarray(cfg.rod_tip)
    axis = tip - np.asarray(cfg.rod_base)
    axis = axis / np.linalg.norm(axis)
    pad_r = max([r for r, _ in cfg.pad_rings], default=0.0) + cfg.pad_contact_radius

    rotors = ""
    for i, (p, a) in enumerate(zip(cfg.rotor_pos, cfg.rotor_axis)):
        rotors += f"""
      <geom name="arm{i}" type="capsule" fromto="0 0 0 {_v(p)}" size="0.008" mass="0.04" rgba=".3 .3 .3 1"/>
      <geom name="prop{i}" type="cylinder" pos="{_v(p)}" zaxis="{_v(a)}" size="0.06 0.005" mass="0.05" rgba=".2 .5 .9 .5"/>
      <site name="rotor{i}" pos="{_v(p)}" zaxis="{_v(a)}"/>"""

    pads = ""
    for k, j in enumerate(pad_idx):
        pads += f"""
      <geom name="pad{k}" type="sphere" pos="{_v(pts[j])}" size="{cfg.pad_contact_radius}" mass="0.001"
            friction="{gs.mu} 0.005 0.0001" condim="3" solref="{_v(gs.pad_solref)}" rgba=".9 .2 .2 1"/>"""

    acts = ""
    for i, s in enumerate(cfg.spin_dir):
        acts += f"""
    <general name="m{i}" site="rotor{i}" gear="0 0 1 0 0 {-s * cfg.k_m / cfg.k_f:.6g}"
             dyntype="filter" dynprm="{gs.motor_tau}" gainprm="1" biastype="none"
             ctrllimited="true" ctrlrange="0 100"/>"""

    return f"""
<mujoco model="aerial_manipulator">
  <option timestep="{gs.timestep}" gravity="0 0 {-cfg.gravity}" density="1.225"
          viscosity="1.8e-5" integrator="Euler" cone="elliptic"/>
  <default><geom contype="1" conaffinity="0"/></default>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="floor" type="plane" size="5 5 .1" contype="0" conaffinity="0" rgba=".8 .8 .8 1"/>
    <!-- mocap body: MuJoCo does not re-pose static worldbody geoms when model.geom_pos changes -->
    <body name="wall_body" mocap="true" pos="0 0 -10">
      <geom name="wall" type="box" size="0.02 2 2" contype="0" conaffinity="1"
            friction="{gs.mu} 0.005 0.0001" solref="{_v(gs.wall_solref)}" rgba=".7 .6 .5 .6"/>
    </body>
    <body name="drone" pos="0 0 1">
      <freejoint/>
      <geom name="core" type="box" size="0.06 0.06 0.025" mass="0.9" fluidshape="ellipsoid" rgba=".1 .1 .1 1"/>{rotors}
      <geom name="rod" type="capsule" fromto="{_v(cfg.rod_base)} {_v(tip - 0.012 * axis)}" size="0.006" mass="0.06"
            fluidshape="ellipsoid" rgba=".6 .6 .6 1"/>
      <geom name="sensor" type="cylinder" pos="{_v(np.asarray(cfg.rod_base) + 0.02 * axis)}" zaxis="{_v(axis)}"
            size="0.015 0.012" mass="0.05" rgba=".9 .7 .1 1"/>
      <geom name="pad_disk" type="cylinder" pos="{_v(tip - 0.002 * axis)}" zaxis="{_v(axis)}"
            size="{pad_r:.4f} 0.002" mass="0.01" contype="0" conaffinity="0" rgba=".9 .4 .4 .6"/>{pads}
    </body>
  </worldbody>
  <actuator>{acts}
  </actuator>
</mujoco>"""


# Reads the simulated mass properties back into a DroneConfig. The body frame
# of the MuJoCo drone IS the mocap frame, so body_ipos is the COM offset.
def true_config_from_model(model, cfg, gs):
    bid = model.body("drone").id
    Ri = np.zeros(9)
    mujoco.mju_quat2Mat(Ri, model.body_iquat[bid])
    Ri = Ri.reshape(3, 3)
    I = Ri @ np.diag(model.body_inertia[bid]) @ Ri.T
    return dataclasses.replace(
        cfg, mass=float(model.body_mass[bid]),
        inertia=tuple(tuple(float(x) for x in row) for row in I),
        com_offset=tuple(float(x) for x in model.body_ipos[bid]),
        dt=gs.timestep * gs.substeps)


# ======================================================================
# Controller
# ======================================================================

def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


# Geometric tracking controller on SE(3) with a thrust/torque mixer.
class GeometricController:

    def __init__(self, cfg, rng, t_max):
        self.m, self.g = cfg.mass, cfg.gravity
        self.J = np.asarray(cfg.inertia)
        c = np.asarray(cfg.com_offset)
        rp = np.asarray(cfg.rotor_pos) - c
        s = np.asarray(cfg.spin_dir, dtype=float)
        # Mixer for (near-)vertical rotor axes: [f, Mx, My, Mz] = A @ T
        self.A = np.vstack([np.ones(len(rp)), rp[:, 1], -rp[:, 0], -s * cfg.k_m / cfg.k_f])
        self.A_inv = np.linalg.pinv(self.A)
        self.t_min, self.t_max = 0.05 * self.m * self.g / len(rp), t_max

        # Gains, randomized +-20% per trajectory so the data is not tied to one tuning.
        wp = 4.0 * rng.uniform(0.8, 1.2)                     # position bandwidth, rad/s
        wr = 15.0 * rng.uniform(0.8, 1.2)                    # roll/pitch bandwidth, rad/s
        wy = 4.0 * rng.uniform(0.8, 1.2)                     # yaw bandwidth: slow, it has little authority
        self.kp, self.kv = wp ** 2, 2.0 * 0.9 * wp
        Jd = np.diag(self.J)
        bw = np.array([wr, wr, wy])
        self.kR, self.kW = Jd * bw ** 2, Jd * 2.0 * 0.9 * bw
        self.max_tilt = math.radians(35.0)
        self.gains = dict(wp=wp, wr=wr, wy=wy)

    # Returns rotor thrust commands (N).
    def __call__(self, com, v, R, w_body, p_d, v_d, a_d, F_ff, heading):
        e3 = np.array([0.0, 0.0, 1.0])
        F = self.m * (-self.kp * (com - p_d) - self.kv * (v - v_d) + a_d + self.g * e3) + F_ff

        # Limit tilt.
        b3 = F / np.linalg.norm(F)
        tilt = math.acos(np.clip(b3 @ e3, -1.0, 1.0))
        if tilt > self.max_tilt:
            h = b3 - (b3 @ e3) * e3
            h /= np.linalg.norm(h) + 1e-12
            b3 = math.cos(self.max_tilt) * e3 + math.sin(self.max_tilt) * h

        b2 = np.cross(b3, heading)
        b2 /= np.linalg.norm(b2)
        R_d = np.column_stack([np.cross(b2, b3), b2, b3])
        f = F @ (R @ e3)

        E = R_d.T @ R - R.T @ R_d
        e_R = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        M = -self.kR * e_R - self.kW * w_body + np.cross(w_body, self.J @ w_body)
        return self.mix(f, M)

    # Mixer with roll/pitch priority. Yaw authority on a quad is small
    # (k_m/k_f ~ 1.6 cm of lever), so roll/pitch and collective are allocated
    # first, then the largest fraction of the yaw torque that still fits is
    # added. (Dropping yaw entirely when it does not fit leaves a large yaw
    # error with no correction at all, and the drone spins up against the wall.)
    def mix(self, f, M):
        T0 = self.A_inv @ np.r_[f, M[0], M[1], 0.0]
        if T0.max() > self.t_max:
            T0 = T0 + (self.t_max - T0.max())
        elif T0.min() < self.t_min:
            T0 = T0 + (self.t_min - T0.min())
        Ty = self.A_inv @ np.r_[0.0, 0.0, 0.0, M[2]]
        s = 1.0
        for t0, ty in zip(T0, Ty):
            if ty > 1e-12:
                s = min(s, (self.t_max - t0) / ty)
            elif ty < -1e-12:
                s = min(s, (t0 - self.t_min) / -ty)
        return np.clip(T0 + max(s, 0.0) * Ty, self.t_min, self.t_max)


# ======================================================================
# Scenarios: pad-center waypoints in the wall frame + push-force profile
# ======================================================================

class Reference:
    """Segments of (duration, target, phase); targets are d (pad-center distance
    from the wall along n), a/b (along-wall coordinates), F (push force, N).
    Each segment is a min-jerk move from the previous target."""

    def __init__(self, start, segments):
        self.start = dict(start)
        self.t, self.vals, self.phase = [0.0], [dict(start)], []
        for dur, tgt, ph in segments:
            v = dict(self.vals[-1])
            v.update(tgt)
            self.t.append(self.t[-1] + dur)
            self.vals.append(v)
            self.phase.append(PHASES[ph])
        self.T = self.t[-1]

    def __call__(self, t):
        t = min(max(t, 0.0), self.T - 1e-9)
        k = int(np.searchsorted(self.t, t, side="right") - 1)
        tau = (t - self.t[k]) / (self.t[k + 1] - self.t[k])
        s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
        a, b = self.vals[k], self.vals[k + 1]
        return {key: a[key] + (b[key] - a[key]) * s for key in a}, self.phase[k]


# Builds a randomized scenario. r_c = pad contact-sphere radius: a pad-center
# distance of r_c means the pad is just touching (if it is flat to the wall).
# Returns the start values and the list of segments.
def make_scenario(kind, rng, r_c, duration, yaw_force_cap=None):
    U = rng.uniform
    start = dict(d=U(0.25, 0.6), a=U(-0.3, 0.3), b=U(-0.3, 0.3), F=0.0)
    cur, seg = dict(start), []

    def add(dur, phase, **tgt):
        seg.append((dur, tgt, phase))
        cur.update(tgt)

    def approach(d_target, speed):
        add(max(abs(cur["d"] - d_target) / speed, 0.3), "approach", d=d_target)

    add(U(0.3, 0.6), "free")                                          # settle
    if kind == "free":
        while sum(sg[0] for sg in seg) < duration:
            add(U(0.8, 2.0), "free", d=U(0.4, 1.2), a=U(-0.6, 0.6), b=U(-0.4, 0.4))
    elif kind == "hover_near":
        # Graded standoff distances with the gate provably off (data-plan stage 2).
        for d in sorted(U(r_c + 0.005, 0.15, size=int(rng.integers(2, 4))), reverse=True):
            add(U(0.6, 1.2), "approach", d=d)
            add(U(0.8, 1.5), "hover_near")
        add(U(0.6, 1.0), "release", d=U(0.3, 0.6))
    elif kind == "tap":
        # Impacts: fly in at speed with no push, touch briefly, back off.
        for _ in range(int(rng.integers(1, 3))):
            approach(r_c - U(0.0, 0.01), U(0.15, 0.6))
            add(U(0.1, 0.4), "hold")
            add(U(0.5, 1.0), "release", d=U(0.15, 0.4))
    elif kind in ("push", "slide"):
        # Pushing is limited by yaw authority. Horizontal friction at the pad,
        # and the normal force when the rod is yawed off the wall normal, act
        # on the rod's ~0.4 m lever arm, and a quad's yaw torque is only
        # k_m/k_f ~ 1.6 cm times the thrust differential. The yawed-rod moment
        # also grows with the yaw error (pushing a stick against a wall), so
        # once yaw saturates it runs away. yaw_force_cap(|cos ang|) is the
        # largest push that keeps both inside the yaw budget.
        slides = [(U(0, 2 * math.pi), U(0.1, 0.4), U(0.05, 0.25))
                  for _ in range(int(rng.integers(1, 3)))] if kind == "slide" else []
        f_max = 6.0
        if yaw_force_cap is not None:
            f_max = min([f_max, yaw_force_cap(0.0)] + [yaw_force_cap(abs(math.cos(a))) for a, _, _ in slides])
        f_min = min(1.0, 0.5 * f_max)
        approach(r_c - U(0.0, 0.005), U(0.1, 0.4))
        add(U(0.3, 0.6), "hold", F=U(f_min, f_max))                   # ramp the push in
        add(U(0.5, 1.5), "hold", F=U(f_min, f_max))                   # vary it while holding
        for ang, dist, speed in slides:
            add(dist / speed, "slide", a=cur["a"] + dist * math.cos(ang),
                b=cur["b"] + dist * math.sin(ang))
        add(U(0.3, 0.6), "release", F=0.0)
        add(U(0.6, 1.0), "release", d=U(0.2, 0.5))
    else:
        raise ValueError(kind)
    add(U(0.3, 0.6), "free")
    return start, seg


# ======================================================================
# Simulation of one trajectory
# ======================================================================

class Crash(Exception):
    pass


# ======================================================================
# MuJoCo plant: one physics substep at a time
# ======================================================================

class MujocoDronePlant:
    """The MuJoCo drone with the forces MuJoCo does not model added by hand
    (per-rotor drag at each hub), plus force logging. Used by the data
    generator and by the closed-loop evaluation, so both run identical physics.

    substep(wind, t) applies rotor drag for the given wind, advances one MuJoCo
    timestep, and returns what acted during that step: rotor thrusts (M), aero
    force (3, MuJoCo fluid + rotor drag), contact force on the pad (3), its
    torque about the COM (3), and the per-pad-sphere contact forces (P, 3).
    Raises Crash if any drone geom other than a pad sphere touches the wall."""

    def __init__(self, model, cfg_true, gs, data=None):
        self.model, self.cfg, self.gs = model, cfg_true, gs
        self.data = data if data is not None else mujoco.MjData(model)
        self.bid = model.body("drone").id
        self.wall_gid = model.geom("wall").id
        n_pad = len(build_drone_graph(cfg_true)["pad_indices"])
        self.pad_gids = {model.geom(f"pad{k}").id: k for k in range(n_pad)}
        self.drone_gids = set(np.flatnonzero(model.geom_bodyid == self.bid).tolist())
        self.c_off = np.asarray(cfg_true.com_offset)
        self.rotor_r = np.asarray(cfg_true.rotor_pos) - self.c_off          # hubs relative to the COM, body frame
        a = np.asarray(cfg_true.rotor_axis, dtype=float)
        self.rotor_a = a / np.linalg.norm(a, axis=1, keepdims=True)
        self.qfrc_fluid = "qfrc_fluid" if hasattr(self.data, "qfrc_fluid") else "qfrc_passive"
        self._f6 = np.zeros(6)

    # COM position, attitude, body angular velocity, COM velocity (world).
    def state(self):
        d = self.data
        R = _quat2mat(d.qpos[3:7])
        w_b = d.qvel[3:6].copy()
        return d.qpos[0:3] + R @ self.c_off, R, w_b, d.qvel[0:3] + np.cross(R @ w_b, R @ self.c_off)

    def substep(self, wind, t):
        m, d, cfg, gs = self.model, self.data, self.cfg, self.gs
        com, R, w_b, v_com = self.state()

        # Per-rotor drag at each hub, from that hub's local airspeed.
        m.opt.wind[:] = wind
        omega_true = np.sqrt(np.maximum(d.act, 0.0) / cfg.k_f)
        r_hub = self.rotor_r @ R.T                                           # (M,3) lever arms, world
        a_hub = self.rotor_a @ R.T                                           # (M,3) rotor axes, world
        u_hub = wind - (v_com + np.cross(R @ w_b, r_hub))                    # local airspeed at each hub
        u_perp = u_hub - np.sum(u_hub * a_hub, axis=1, keepdims=True) * a_hub
        F_hub = cfg.mass * gs.k_rotor * omega_true[:, None] * u_perp         # (M,3)
        F_rd = F_hub.sum(0)
        d.xfrc_applied[self.bid, :3] = F_rd                                  # xfrc_applied acts at the COM,
        d.xfrc_applied[self.bid, 3:] = np.cross(r_hub, F_hub).sum(0)         # so add the hubs' moment

        mujoco.mj_step(m, d)

        # Forces used during the step just taken (MuJoCo leaves them from the
        # pre-integration forward pass).
        Fc, tc = np.zeros(3), np.zeros(3)
        pad = np.zeros((len(self.pad_gids), 3))
        for i in range(d.ncon):
            con = d.contact[i]
            if self.wall_gid not in (con.geom1, con.geom2):
                continue
            other = con.geom2 if con.geom1 == self.wall_gid else con.geom1
            if other not in self.pad_gids:
                if other in self.drone_gids:
                    raise Crash(f"{m.geom(other).name} hit the wall at t={t:.2f}s")
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            F = con.frame.reshape(3, 3).T @ self._f6[:3]                     # force on geom2
            if other == con.geom1:
                F = -F
            Fc += F
            tc += np.cross(con.pos - com, F)
            pad[self.pad_gids[other]] += F
        return d.actuator_force.copy(), getattr(d, self.qfrc_fluid)[0:3] + F_rd, Fc, tc, pad


# Verlet alignment: x is a per-substep log (n_sub, ...); returns per-frame values
# with triangle weights over the two intervals around each frame, which is
# exactly what the second difference of the recorded positions sees.
def align_to_frames(x, S, T_frames, first=None):
    w = np.r_[np.arange(S), S - np.arange(S)] / S ** 2                     # sums to 1
    out = np.empty((T_frames,) + x.shape[1:])
    for k in range(1, T_frames):
        out[k] = np.tensordot(w, x[(k - 1) * S:(k + 1) * S], axes=1)
    out[0] = out[1] if first is None else first
    return out


# In-plane wall axes: t1 horizontal, t2 "up the wall".
def wall_tangents(n):
    t1 = np.cross(n, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(t1, n)


# Places the wall (a mocap box) so its surface passes through wall_point with
# outward normal n. MuJoCo does not re-pose static worldbody geoms when
# model.geom_pos changes, which is why the wall lives on a mocap body.
def set_wall_pose(model, data, n, wall_point):
    t1, _ = wall_tangents(n)
    half = model.geom_size[model.geom("wall").id][0]
    data.mocap_pos[0] = wall_point - half * n
    data.mocap_quat[0] = _mat2quat(np.column_stack([n, t1, np.cross(n, t1)]))   # box local x = n


def simulate_trajectory(model, cfg_true, gs, rng):
    plant = MujocoDronePlant(model, cfg_true, gs)
    data = plant.data
    S, h = gs.substeps, gs.timestep
    dt = h * S
    c_off = plant.c_off
    m, M = cfg_true.mass, len(cfg_true.rotor_pos)
    t_hover = m * cfg_true.gravity / M
    ctrl = GeometricController(cfg_true, rng, gs.t_max_over_hover * t_hover)

    # ---- wall pose: random yaw, sometimes tilted ----
    psi = rng.uniform(0, 2 * math.pi)
    beta = math.radians(rng.uniform(-1, 1) * gs.wall_tilt_max_deg) if rng.random() < gs.tilted_wall_prob else 0.0
    n = np.array([-math.cos(beta) * math.cos(psi), -math.cos(beta) * math.sin(psi), math.sin(beta)])
    wall_point = np.array([0.0, 0.0, 1.5])
    t1, t2 = wall_tangents(n)
    set_wall_pose(model, data, n, wall_point)

    # ---- heading: rod (body x) points at the wall, plus a random yaw offset ----
    nh = -n.copy()
    nh[2] = 0.0
    nh /= np.linalg.norm(nh)
    yo = math.radians(rng.uniform(-1, 1) * gs.yaw_offset_max_deg)
    heading = np.array([[math.cos(yo), -math.sin(yo), 0], [math.sin(yo), math.cos(yo), 0], [0, 0, 1]]) @ nh
    R_h = np.column_stack([heading, np.cross([0, 0, 1.0], heading), [0, 0, 1.0]])
    tip_off = R_h @ (np.asarray(cfg_true.rod_tip) - c_off)      # pad center relative to COM, level

    # ---- scenario ----
    kinds, probs = zip(*gs.scenario_probs)
    kind = str(rng.choice(kinds, p=np.asarray(probs) / sum(probs)))
    lever = float(np.linalg.norm((np.asarray(cfg_true.rod_tip) - c_off)[:2]))
    yaw_cap = lambda c_h: gs.yaw_torque_budget / (lever * (gs.mu * c_h + abs(math.sin(yo))) + 1e-9)
    s0, seg = make_scenario(kind, rng, cfg_true.pad_contact_radius, rng.uniform(*gs.duration), yaw_cap)
    ref = Reference(s0, seg)
    T_frames = int(ref.T / dt) + 1

    def p_com_ref(t):
        r, ph = ref(t)
        p_tip = wall_point + r["d"] * n + r["a"] * t1 + r["b"] * t2
        return p_tip - tip_off, r["F"], ph

    # ---- wind: mean + OU gusts ----
    w_mean = np.zeros(3)
    if rng.random() > 0.3:
        w_ang = rng.uniform(0, 2 * math.pi)
        w_mean = rng.uniform(0, gs.wind_max) * np.array([math.cos(w_ang), math.sin(w_ang), 0.0])
    gust_sig = rng.uniform(0, gs.gust_std)
    gust = np.zeros(3)
    ex_sig, ex_tau = rng.uniform(0, gs.excite_max), 0.05
    ex = np.zeros(M)

    # ---- initial state: hovering at the start reference ----
    p0, _, _ = p_com_ref(0.0)
    data.qpos[0:3] = p0 - R_h @ c_off
    data.qpos[3:7] = _mat2quat(R_h)
    data.act[:] = t_hover
    data.ctrl[:] = t_hover
    mujoco.mj_forward(model, data)

    n_sub = T_frames * S
    log_thrust = np.zeros((n_sub, M))
    log_Fc, log_tc, log_Fa = np.zeros((n_sub, 3)), np.zeros((n_sub, 3)), np.zeros((n_sub, 3))
    log_pad = np.zeros((n_sub, len(plant.pad_gids), 3))
    frames = dict(pos=[], quat=[], rotor_speed_cmd=[], wind=[], phase=[])

    for j in range(n_sub):
        t = j * h
        com, R, w_b, v_com = plant.state()

        # Crash checks.
        if not np.all(np.isfinite(data.qpos)) or com[2] < 0.2 or (R @ [0, 0, 1])[2] < 0.5:
            raise Crash(f"state left the envelope at t={t:.2f}s")

        # Record the frame state (before the substeps that follow it).
        if j % S == 0:
            frames["pos"].append(data.qpos[0:3].copy())
            frames["quat"].append(data.qpos[3:7].copy())
            frames["rotor_speed_cmd"].append(np.sqrt(np.maximum(data.ctrl, 0) / cfg_true.k_f))
            frames["wind"].append(w_mean + gust)
            frames["phase"].append(p_com_ref(t)[2])

        # Controller at its own rate.
        if j % gs.ctrl_every == 0:
            eps = 1e-3
            p_d, F_push, _ = p_com_ref(t)
            v_d = (p_com_ref(t + eps)[0] - p_com_ref(t - eps)[0]) / (2 * eps)
            a_d = (p_com_ref(t + eps)[0] - 2 * p_d + p_com_ref(t - eps)[0]) / eps ** 2
            T_cmd = ctrl(com, v_com, R, w_b, p_d, v_d, a_d, -n * F_push, heading)
            hc = gs.ctrl_every * h
            ex += -ex * hc / ex_tau + ex_sig * math.sqrt(2 * hc / ex_tau) * rng.normal(size=M)
            data.ctrl[:] = np.clip(T_cmd * (1.0 + ex), 0.0, ctrl.t_max)

        # Wind gusts, then one physics substep (rotor drag + MuJoCo + force logging).
        gust += -gust * h / 1.0 + gust_sig * math.sqrt(2 * h) * np.r_[rng.normal(size=2), 0.2 * rng.normal()]
        log_thrust[j], log_Fa[j], log_Fc[j], log_tc[j], log_pad[j] = plant.substep(w_mean + gust, t)

    align = lambda x: align_to_frames(x, S, T_frames)
    thrust_al = align(log_thrust)
    traj = dict(
        pos=frames["pos"], quat=frames["quat"],
        rotor_speed=np.sqrt(np.maximum(thrust_al, 0.0) / cfg_true.k_f),
        rotor_speed_cmd=frames["rotor_speed_cmd"], wind=frames["wind"], phase=frames["phase"],
        F_contact=align(log_Fc), tau_contact=align(log_tc), F_pad=align(log_pad), F_aero=align(log_Fa))
    # Positions in float64: float32 resolves only ~1e-7 m at 1.5 m from the
    # origin, about 1% of the per-step aero signal the model learns from.
    dtypes = {"phase": torch.long, "pos": torch.float64}
    traj = {k: torch.tensor(np.asarray(v), dtype=dtypes.get(k, torch.float32)) for k, v in traj.items()}
    traj["wall_normal"] = torch.tensor(n, dtype=torch.float32)
    traj["wall_point"] = torch.tensor(wall_point, dtype=torch.float32)
    traj["meta"] = dict(dt=dt, gravity=cfg_true.gravity, rotor_speed_hold="aligned",
                        mass=cfg_true.mass, inertia=cfg_true.inertia, com_offset=cfg_true.com_offset,
                        k_f=cfg_true.k_f, k_m=cfg_true.k_m, mu=gs.mu, k_rotor=gs.k_rotor,
                        scenario=kind, wind_mean=w_mean.tolist(), excitation=ex_sig,
                        wall_tilt_deg=math.degrees(beta), yaw_offset_deg=math.degrees(yo),
                        aero_label_source=plant.qfrc_fluid, **ctrl.gains)
    return traj


# ======================================================================
# Check: the logged forces must explain the recorded motion
# ======================================================================

# Verlet residual of the COM after removing thrust, gravity, and the logged
# contact + aero labels, in units of g. If the conventions (frame alignment,
# contact sign, COM offset, thrust axis) are right, this is ~0.
def verify_trajectory(traj, cfg):
    from force_data import quat_wxyz_to_R
    R = quat_wxyz_to_R(traj["quat"].double())
    com = traj["pos"].double() + R @ torch.tensor(cfg.com_offset, dtype=torch.float64)
    dt, m = cfg.dt, cfg.mass
    a_meas = (com[2:] - 2 * com[1:-1] + com[:-2]) / dt ** 2
    om = traj["rotor_speed"].double()[1:-1]
    axes = torch.tensor(cfg.rotor_axis, dtype=torch.float64)
    F_thr = R[1:-1] @ ((cfg.k_f * om ** 2) @ axes).unsqueeze(-1)
    F_ext = (traj["F_contact"] + traj["F_aero"]).double()[1:-1]
    g = torch.tensor([0, 0, -cfg.gravity], dtype=torch.float64)
    res = a_meas - F_thr.squeeze(-1) / m - F_ext / m - g
    in_contact = traj["F_contact"][1:-1].norm(dim=-1) > 1e-6
    out = dict(resid_all_g=float(res.norm(dim=-1).mean() / cfg.gravity),
               contact_frac=float(in_contact.float().mean()),
               normal_force_ok=bool(((traj["F_contact"] @ traj["wall_normal"]) > -1e-4).all()))
    if in_contact.any():
        out["resid_contact_g"] = float(res[in_contact].norm(dim=-1).mean() / cfg.gravity)
        out["mean_contact_N"] = float(traj["F_contact"][1:-1][in_contact].norm(dim=-1).mean())
    return out


# ======================================================================
# Driver
# ======================================================================

def generate(out, n, start=0, seed=0, cfg=None, gs=None, verify=False, verbose=True):
    cfg, gs = cfg or DroneConfig(), gs or GenSettings()
    model = mujoco.MjModel.from_xml_string(build_xml(cfg, gs))
    cfg_true = true_config_from_model(model, cfg, gs)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "drone_config.json"), "w") as f:
        json.dump(cfg_true.to_dict(), f, indent=2)

    crashes = 0
    for i in range(start, start + n):
        attempt = 0
        while True:
            rng = np.random.default_rng([seed, i, attempt])
            try:
                traj = simulate_trajectory(model, cfg_true, gs, rng)
                break
            except Crash as e:
                crashes += 1
                attempt += 1
                if verbose:
                    print(f"  traj {i} attempt {attempt}: discarded ({e})")
        torch.save(traj, os.path.join(out, f"{i}.pt"))
        if verbose:
            msg = f"traj {i}: {traj['meta']['scenario']:10s} T={traj['pos'].shape[0]}"
            if verify:
                v = verify_trajectory(traj, cfg_true)
                msg += "  " + "  ".join(f"{k}={v[k]:.4g}" if isinstance(v[k], float) else f"{k}={v[k]}"
                                        for k in v)
            print(msg, flush=True)
    if verbose:
        print(f"done: {n} trajectories, {crashes} discarded; config -> {out}/drone_config.json")
    return cfg_true


# Plays a saved trajectory back in the MuJoCo viewer (needs a display; on
# macOS run it with mjpython). Kinematic replay: it sets the recorded poses.
def replay(path, speed=1.0):
    import time
    import mujoco.viewer
    traj = torch.load(path, weights_only=False)
    with open(os.path.join(os.path.dirname(path), "drone_config.json")) as f:
        cfg = DroneConfig(**json.load(f))
    model = mujoco.MjModel.from_xml_string(build_xml(cfg, GenSettings()))
    data = mujoco.MjData(model)
    set_wall_pose(model, data, traj["wall_normal"].double().numpy(), traj["wall_point"].double().numpy())
    print(f"{path}: {traj['meta']['scenario']}, {traj['pos'].shape[0]} frames")
    with mujoco.viewer.launch_passive(model, data) as v:
        for k in range(traj["pos"].shape[0]):
            if not v.is_running():
                break
            data.qpos[0:3] = traj["pos"][k].numpy()
            data.qpos[3:7] = traj["quat"][k].numpy()
            mujoco.mj_forward(model, data)
            v.sync()
            time.sleep(cfg.dt / speed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mj_drone")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--replay", default=None, help="path to a saved .pt to play back in the viewer")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    if a.replay:
        replay(a.replay, a.speed)
    else:
        generate(a.out, a.n, a.start, a.seed, verify=a.verify)
