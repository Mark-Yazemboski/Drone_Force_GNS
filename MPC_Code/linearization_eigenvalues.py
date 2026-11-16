"""
How unstable is the GNN's linearization, in free flight vs in contact?

Takes a saved run, linearizes the model at every recorded MuJoCo state (with the rotor speeds the model actually saw,
i.e. after the motor lag), and reports for each step:
    - max |eigenvalue| of A_k        (above 1: a mode that grows every 10 ms step on its own)
    - horizon gain                   (largest singular value of A_{k+N-1} ... A_k: how much the next N steps can
                                      amplify a deviation, the number that made the QP ill-conditioned)
both for the raw model (open loop, A_k) and for the pre-stabilized prediction the MPC actually works with
(A_k + B_k K_k, with the LQR gains from the controller's prestabilizing_gains over the N steps starting at k).
Steps are split by whether MuJoCo had the pad in contact.

Run:  python linearization_eigenvalues.py final_v1234
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Drone_Dynamics
from MPC_controller import motor_step, prestabilizing_gains
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
A, B = [], []
for start in range(0, len(commands), CHUNK):
    stop = min(start + CHUNK, len(commands))
    A_chunk, B_chunk, *_ = model.linearize_horizon(torch.as_tensor(states[start:stop], dtype=torch.float32),
                                                   torch.as_tensor(effective[start:stop], dtype=torch.float32), 1)
    A.extend(A_chunk.numpy().astype(float))
    B.extend(B_chunk.numpy().astype(float))
    print(f"linearized {stop} / {len(commands)}")
A, B = np.array(A), np.array(B)

def growth(matrices):
    #Max |eigenvalue| of the first matrix, and the largest singular value of the product of all of them
    product = np.eye(matrices[0].shape[0])
    for matrix in matrices:
        product = matrix @ product
    return np.abs(np.linalg.eigvals(matrices[0])).max(), np.linalg.norm(product, 2)

#Per step: open loop (A_k) and pre-stabilized (A_k + B_k K_k, gains from the N steps starting at k, like the MPC)
eig_max, horizon_gain = np.full(len(A), np.nan), np.full(len(A), np.nan)
eig_max_closed, horizon_gain_closed = np.full(len(A), np.nan), np.full(len(A), np.nan)
for k in range(len(A) - HORIZON_LENGTH + 1):
    window_A, window_B = A[k:k + HORIZON_LENGTH], B[k:k + HORIZON_LENGTH]
    gains = prestabilizing_gains(window_A, window_B)
    eig_max[k], horizon_gain[k] = growth(window_A)
    eig_max_closed[k], horizon_gain_closed[k] = growth([a + b @ K for a, b, K in zip(window_A, window_B, gains)])

#Split by contact (normal_force[k + 1] is the force over step k)
in_contact = data["normal_force"][1:len(A) + 1] > CONTACT_FORCE
has_window = np.isfinite(eig_max)
for label, eigs, gains in [("open loop  A", eig_max, horizon_gain), ("closed loop A+BK", eig_max_closed, horizon_gain_closed)]:
    print(label)
    for name, mask in [("free flight", ~in_contact & has_window), ("contact", in_contact & has_window)]:
        if mask.any():
            print(f"  {name:12s} ({mask.sum():4d} steps): max|eig| median {np.median(eigs[mask]):.3f}, "
                  f"95th pct {np.percentile(eigs[mask], 95):.3f}, max {eigs[mask].max():.3f} | "
                  f"horizon gain median {np.median(gains[mask]):.1e}, max {gains[mask].max():.1e}")

#Plot: max |eigenvalue| and horizon gain over the run, contact shaded
set_plot_style()
fig, axes = plt.subplots(2, 1, figsize=(REPORT_WIDTH, 3.0), sharex=True)
step_times = times[:len(A)]
axes[0].plot(step_times, eig_max, color="#2a78d6", label="open loop $A_k$")
axes[0].plot(step_times, eig_max_closed, color="#2ca02c", label="pre-stabilized $A_k + B_k K_k$")
axes[0].axhline(1.0, color="black", linewidth=0.6, linestyle="--")
axes[0].set_ylabel(r"max $|\lambda|$")
axes[0].legend(loc="upper left", frameon=False)
axes[1].semilogy(step_times, horizon_gain, color="#2a78d6")
axes[1].semilogy(step_times, horizon_gain_closed, color="#2ca02c")
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
