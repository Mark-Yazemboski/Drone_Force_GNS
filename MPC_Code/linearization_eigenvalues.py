"""
How unstable is the GNN's linearization, in free flight vs in contact?

Takes a saved run, linearizes the model at every recorded MuJoCo state (with the rotor speeds the model actually saw,
i.e. after the motor lag), and reports for each step:
    - max |eigenvalue| of A_k        (above 1: a mode that grows every 10 ms step on its own)
    - horizon gain                   (largest singular value of A_{k+N-1} ... A_k: how much the next N steps can
                                      amplify a deviation, the number that made the QP ill-conditioned)
Steps are split by whether MuJoCo had the pad in contact.

Run:  python linearization_eigenvalues.py final_v1234
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Drone_Dynamics
from MPC_controller import motor_step
from visualize import set_plot_style, REPORT_WIDTH

SCRIPT_FOLDER = Path(__file__).resolve().parent
Trained_model_path = SCRIPT_FOLDER / "MPC_Model" / "Trained_Model.pt"

CONTACT_CUTOFF_GAP = 0.001      #same as the MPC run file
HORIZON_LENGTH = 50             #same as the MPC
CHUNK = 25                      #states linearized per batch (keeps memory small)
CONTACT_FORCE = 0.05            #N, MuJoCo normal force above this counts as contact

run_folder = SCRIPT_FOLDER / "MPC_runs" / sys.argv[1]
data = np.load(run_folder / "run_data.npz")
states, commands, times = data["states"], data["inputs"], data["times"]
dt = float(data["simulation_dt"])

device = torch.device("cpu")
model = Drone_Dynamics(device, Trained_model_path, wall_normal_vector=data["wall_normal_vector"],
                       wall_center_point=data["wall_center_point"], wind_vector=data["wind_vector"],
                       contact_cutoff_gap=CONTACT_CUTOFF_GAP)

#Rotor speeds the model saw at each step: the commands passed through the motor lag, starting at hover
effective = np.zeros_like(commands)
speed_squared = data["hover_input"] ** 2
for k, command in enumerate(commands):
    effective[k], speed_squared = motor_step(speed_squared, command, dt)

#Linearize at every recorded state, in chunks
A = []
for start in range(0, len(commands), CHUNK):
    stop = min(start + CHUNK, len(commands))
    A_chunk, *_ = model.linearize_horizon(torch.as_tensor(states[start:stop], dtype=torch.float32),
                                          torch.as_tensor(effective[start:stop], dtype=torch.float32), 1)
    A.extend(A_chunk.numpy().astype(float))
    print(f"linearized {stop} / {len(commands)}")
A = np.array(A)

#Max |eigenvalue| per step, and the horizon gain of the N steps starting there
eig_max = np.array([np.abs(np.linalg.eigvals(matrix)).max() for matrix in A])
horizon_gain = np.full(len(A), np.nan)
for k in range(len(A) - HORIZON_LENGTH + 1):
    product = np.eye(A.shape[1])
    for matrix in A[k:k + HORIZON_LENGTH]:
        product = matrix @ product
    horizon_gain[k] = np.linalg.norm(product, 2)

#Split by contact (normal_force[k + 1] is the force over step k)
in_contact = data["normal_force"][1:len(A) + 1] > CONTACT_FORCE
for name, mask in [("free flight", ~in_contact), ("contact", in_contact)]:
    if mask.any():
        gains = horizon_gain[mask][np.isfinite(horizon_gain[mask])]
        print(f"{name:12s} ({mask.sum():4d} steps): max|eig| median {np.median(eig_max[mask]):.3f}, "
              f"95th pct {np.percentile(eig_max[mask], 95):.3f}, max {eig_max[mask].max():.3f} | "
              f"horizon gain median {np.median(gains):.1e}, max {gains.max():.1e}")

#Plot: max |eigenvalue| and horizon gain over the run, contact shaded
set_plot_style()
fig, axes = plt.subplots(2, 1, figsize=(REPORT_WIDTH, 3.0), sharex=True)
step_times = times[:len(A)]
axes[0].plot(step_times, eig_max, color="#2a78d6")
axes[0].axhline(1.0, color="black", linewidth=0.6, linestyle="--")
axes[0].set_ylabel(r"max $|\lambda(A_k)|$")
axes[1].semilogy(step_times, horizon_gain, color="#2a78d6")
axes[1].set_ylabel(f"{HORIZON_LENGTH}-step gain")
for ax in axes:
    ax.fill_between(step_times, 0, 1, where=in_contact, transform=ax.get_xaxis_transform(),
                    color="gray", alpha=0.15, linewidth=0)
axes[-1].set_xlabel("time (s)")
axes[-1].set_xlim(step_times[0], step_times[-1])
fig.tight_layout()
for suffix in (".png", ".svg"):
    fig.savefig(run_folder / f"linearization_eigenvalues{suffix}")
print("Saved", run_folder / "linearization_eigenvalues.png")
