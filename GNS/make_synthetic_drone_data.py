"""Generate synthetic drone + rod + wall trajectories in the format drone_data.py
expects. This is NOT the MuJoCo sim: it is a minimal stand-in (substepped rigid
body, rotor thrust/yaw from the same model, quadratic body drag + rotor drag,
spring-damper tip contact with regularized Coulomb friction) so the pipeline
can be tested end to end, and so the MuJoCo generator has a reference for the
file format, the frame alignment, and the wrench labels.

Rotor speeds are held constant over each frame (zero-order hold), the same as
MuJoCo ctrl held across substeps. See drone_data.py for why that matters.
"""

import math
import os

import numpy as np
import torch

from drone_config import DroneConfig, rotor_and_inertia_tensors, build_drone_graph


def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _exp(w):
    th = np.linalg.norm(w)
    K = _hat(w)
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1 - math.cos(th)) / th ** 2 * K @ K


def _rot(axis, ang):
    return _exp(np.asarray(axis, float) * ang)


# Rotation matrix -> quaternion (wxyz), Shepperd's method.
def R_to_quat_wxyz(R):
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q)


# Simulates one trajectory in a canonical frame (wall normal -x, rod along +x),
# then rotates the whole scene by a random yaw so wall poses vary.
def simulate_one(cfg, rng, T=120, substeps=20, k_body=0.02, k_rotor=5e-5,
                 k_wall=2000.0, c_wall=30.0, mu=0.3, rpm_walk_std=0.002, rpm_clip=0.03):
    geo = {k: v.double().numpy() for k, v in rotor_and_inertia_tensors(cfg).items()}
    graph = build_drone_graph(cfg)
    r_tip = graph["rest_nodes"][graph["tip_index"]].double().numpy()
    m, g, dt = cfg.mass, cfg.gravity, cfg.dt
    I = np.asarray(cfg.inertia, float)
    I_inv = np.linalg.inv(I)
    hstep = dt / substeps
    M = len(cfg.rotor_pos)
    omega_h = math.sqrt(m * g / (M * cfg.k_f))

    # Canonical wall: plane x = 0, normal pointing back toward the drone (-x).
    n = np.array([-1.0, 0.0, 0.0])
    c = np.zeros(3)

    # Initial state: tip 3-12 cm from the wall, flying toward it, slight nose-down pitch.
    pitch = rng.uniform(0.0, math.radians(6.0))
    R = _rot([0, 1, 0], pitch)
    gap = rng.uniform(0.03, 0.12)
    p = -(R @ r_tip) + np.array([-gap, 0.0, 1.0])
    v = np.array([rng.uniform(0.1, 0.5), rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05)])
    w_b = rng.normal(0.0, 0.05, 3)
    wind = np.r_[rng.uniform(-2.0, 2.0, 2), 0.0]
    delta = np.zeros(M)

    out = dict(pos=[], quat=[], rotor_speed=[], F_contact=[], com_true=[])
    for t in range(T):
        # Rotor speeds held over this frame: hover + slow differential random walk.
        delta = np.clip(delta + rng.normal(0.0, rpm_walk_std, M), -rpm_clip, rpm_clip)
        omega = omega_h * (1.0 + delta) / math.sqrt(max(math.cos(pitch), 0.5))

        out["pos"].append(p - R @ np.asarray(cfg.com_offset))
        out["quat"].append(R_to_quat_wxyz(R))
        out["rotor_speed"].append(omega.copy())
        out["com_true"].append(p.copy())

        Fc_acc = np.zeros(3)
        for _ in range(substeps):
            w2 = omega ** 2
            F_i = (cfg.k_f * w2)[:, None] * geo["rotor_axis"]
            T_b = (np.cross(geo["rotor_pos"], F_i)
                   - (geo["spin_dir"] * cfg.k_m * w2)[:, None] * geo["rotor_axis"]).sum(0)
            F_b = F_i.sum(0)

            # Aero: same law as drone_dynamics.aero_law_body, applied at the COM.
            u_b = R.T @ (wind - v)
            ta = geo["thrust_axis"]
            u_perp = u_b - (u_b @ ta) * ta
            F_b = F_b + m * (k_body * np.linalg.norm(u_b) * u_b + k_rotor * omega.sum() * u_perp)

            # Tip contact: spring-damper normal, regularized Coulomb friction.
            p_tip = p + R @ r_tip
            v_tip = v + R @ np.cross(w_b, r_tip)
            pen = -(p_tip - c) @ n
            F_c = np.zeros(3)
            if pen > 0:
                Fn = max(k_wall * pen - c_wall * (v_tip @ n), 0.0)
                v_t = v_tip - (v_tip @ n) * n
                F_c = Fn * n - mu * Fn * v_t / math.sqrt(v_t @ v_t + 1e-6)
            T_b = T_b + np.cross(r_tip, R.T @ F_c)

            # Semi-implicit Euler substep.
            v = v + hstep * ((R @ F_b + F_c) / m + np.array([0.0, 0.0, -g]))
            p = p + hstep * v
            w_b = w_b + hstep * (I_inv @ (T_b - np.cross(w_b, I @ w_b)))
            R = R @ _exp(w_b * hstep)
            Fc_acc += F_c
        out["F_contact"].append(Fc_acc / substeps)     # frame-average contact force, N

    # Random yaw of the whole scene.
    Rz = _rot([0, 0, 1], rng.uniform(0, 2 * math.pi))
    pos = np.asarray(out["pos"]) @ Rz.T
    quat = np.stack([R_to_quat_wxyz(Rz @ _quat_to_R(q)) for q in out["quat"]])
    return {
        "pos": torch.tensor(pos, dtype=torch.float32),
        "quat": torch.tensor(quat, dtype=torch.float32),
        "rotor_speed": torch.tensor(np.asarray(out["rotor_speed"]), dtype=torch.float32),
        "wind": torch.tensor(Rz @ wind, dtype=torch.float32),
        "wall_normal": torch.tensor(Rz @ n, dtype=torch.float32),
        "wall_point": torch.tensor(Rz @ c, dtype=torch.float32),
        "F_contact": torch.tensor(np.asarray(out["F_contact"]) @ Rz.T, dtype=torch.float32),
        "com_true": torch.tensor(np.asarray(out["com_true"]) @ Rz.T, dtype=torch.float32),
        "meta": {"dt": dt, "gravity": g, "rotor_speed_hold": "zoh",
                 "k_body": k_body, "k_rotor": k_rotor, "mu": mu},
    }


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_dataset(folder, n_traj, cfg=None, seed=0, **kw):
    cfg = cfg or DroneConfig()
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_traj):
        torch.save(simulate_one(cfg, rng, **kw), os.path.join(folder, f"{i}.pt"))
    return cfg


if __name__ == "__main__":
    make_dataset("data/synthetic_drone", 64)
