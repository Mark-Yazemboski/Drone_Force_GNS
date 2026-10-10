"""Plot one generated trajectory: pad distance to the wall, contact forces,
rotor speeds, and tilt, with the commanded scenario phases shaded.
Run: python plot_drone_trajectory.py data/mj_drone/3.pt  [out.png]
"""

#py mujoco_drone_generator.py --out data/mj_test --n 100
#py mujoco_drone_generator.py --replay data/mj_test/6.pt
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from drone_config import DroneConfig, build_drone_graph
from force_data import quat_wxyz_to_R

PHASE_NAMES = {0: "free", 1: "approach", 2: "hold", 3: "slide", 4: "release", 5: "hover_near"}
PHASE_COLORS = {0: "white", 1: "#dbe9f6", 2: "#f6dbdb", 3: "#f6eedb", 4: "#e2f6db", 5: "#ece0f6"}


def plot_trajectory(path, out=None):
    d = torch.load(path, weights_only=False)
    with open(os.path.join(os.path.dirname(path), "drone_config.json")) as f:
        cfg = DroneConfig(**json.load(f))
    g = build_drone_graph(cfg)
    rest, pad = g["rest_nodes"], g["pad_indices"]

    R = quat_wxyz_to_R(d["quat"])
    com = d["pos"] + R @ torch.tensor(cfg.com_offset)
    nodes = com.unsqueeze(1) + torch.einsum('tij,nj->tni', R, rest)
    n, c = d["wall_normal"], d["wall_point"]
    dist = ((nodes[:, pad] - c) * n).sum(-1)                       # (T, n_pad)
    t = torch.arange(d["pos"].shape[0]) * cfg.dt

    Fc = d["F_contact"]
    Fn = Fc @ n
    Ft = (Fc - Fn.unsqueeze(-1) * n).norm(dim=-1)
    tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))

    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for a in ax:                                                    # phase shading
        ph = d["phase"]
        start = 0
        for k in range(1, len(ph) + 1):
            if k == len(ph) or ph[k] != ph[start]:
                a.axvspan(float(t[start]), float(t[k - 1]) + cfg.dt, color=PHASE_COLORS[int(ph[start])], lw=0)
                start = k

    ax[0].plot(t, 1e3 * dist.min(dim=1).values, label="closest pad node")
    ax[0].plot(t, 1e3 * dist[:, 0], "--", lw=0.8, label="pad center")
    ax[0].axhline(1e3 * cfg.pad_contact_radius, color="k", lw=0.6, ls=":")
    ax[0].set_ylabel("dist to wall (mm)")
    ax[0].set_ylim(-5, min(150.0, float(1e3 * dist.max()) + 5))
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(t, Fn, label="normal")
    ax[1].plot(t, Ft, label="tangential")
    ax[1].plot(t, d["meta"]["mu"] * Fn, ":", lw=0.8, label="mu * normal")
    ax[1].set_ylabel("contact force (N)")
    ax[1].legend(loc="upper right", fontsize=8)

    ax[2].plot(t, d["rotor_speed"])
    ax[2].set_ylabel("rotor speed (rad/s)")

    ax[3].plot(t, tilt)
    ax[3].set_ylabel("tilt (deg)")
    ax[3].set_xlabel("time (s)")

    m = d["meta"]
    phases = ", ".join(f"{v}" for k, v in PHASE_NAMES.items() if (d["phase"] == k).any())
    fig.suptitle(f"{os.path.basename(path)}: {m['scenario']} | wall tilt {m['wall_tilt_deg']:.1f} deg, "
                 f"yaw offset {m['yaw_offset_deg']:.1f} deg | phases: {phases}", fontsize=10)
    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=110)
    return out


if __name__ == "__main__":
    print(plot_trajectory(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
