"""
Replays a saved MPC run through MuJoCo and logs the wall's normal force at EVERY physics substep.

get_normal_force() returns the force from the last 0.5 ms substep only. The GNN's training labels (and its
predictions) are averages over the whole 10 ms step. This script compares the two on a run you already have:
no MPC, no GPU, a few seconds. It changes nothing in the run folder except adding one plot.

Run:  python replay_contact_force.py base
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from model import load_checkpoint
from mujoco_simulator import MujocoDroneSim

SCRIPT_FOLDER = Path(__file__).resolve().parent
Trained_model_path = SCRIPT_FOLDER / "MPC_Model" / "Trained_Model.pt"
MOTOR_TAU = 0.025           #same as the MPC run

run_folder = SCRIPT_FOLDER / "MPC_runs" / sys.argv[1]
data = np.load(run_folder / "run_data.npz")
_, drone_cfg, _, _ = load_checkpoint(Trained_model_path, "cpu")

wall_normal = data["wall_normal_vector"]
sim = MujocoDroneSim(drone_cfg, wall_normal, data["wall_center_point"], data["wind_vector"], motor_tau=MOTOR_TAU)

#Logs the normal force after every physics substep (read_contacts is called once per substep)
substep_force = []
read_contacts = sim.read_contacts

def read_and_log():
    force = read_contacts()
    substep_force.append(0.0 if force is None else force @ wall_normal)
    return force

sim.read_contacts = read_and_log

#Same start and same inputs as the saved run
state = sim.reset(data["initial_state"], data["hover_input"])
simulation_dt = float(data["simulation_dt"])
for rotor_speeds in data["inputs"]:
    state = sim.step(rotor_speeds, simulation_dt)

steps = len(data["inputs"])
mismatch = np.abs(state - data["states"][steps]).max()
print(f"Replay vs saved final state, largest difference: {mismatch:.1e}" + ("  (WARNING: the replay didn't reproduce the run)" if mismatch > 1e-4 else ""))

#One row per control step: the force at each of its substeps. Row i goes with times[i + 1].
substep_force = np.array(substep_force).reshape(steps, -1)
averaged = substep_force.mean(axis=1)
sampled = data["normal_force"][1:steps + 1]
predicted = data["predicted_force"][1:steps + 1]
times = data["times"][1:steps + 1]

#Reference, and the hold / slide parts of the contact phase
reference_time = data["reference_time"]
reference_force = np.interp(times, reference_time, data["reference_normal_force"])
in_contact = np.interp(times, reference_time, data["reference_contact_flag"]) > 0.5
velocity = np.stack([np.interp(times, reference_time, data["reference_pad_velocity"][:, i]) for i in range(3)], axis=1)
in_plane_speed = np.linalg.norm(velocity - np.outer(velocity @ wall_normal, wall_normal), axis=1)
full_force = reference_force > 0.9 * reference_force.max()
phases = {"hold": in_contact & full_force & (in_plane_speed < 1e-3),
          "slide": in_contact & (in_plane_speed >= 1e-3)}

rms = lambda error: np.sqrt(np.nanmean(error ** 2))
print(f"\n{'':6s} {'sampled (last substep)':>26s} | {'averaged over the step':>26s} | {'substeps':>9s} | {'GNN prediction error vs':>24s}")
print(f"{'':6s} {'touching':>9s} {'mean N':>7s} {'error N':>8s} | {'mean N':>16s} {'error N':>9s} | {'touching':>9s} | {'sampled':>11s} {'averaged':>12s}")
for name, phase in phases.items():
    if not phase.any():
        continue
    print(f"{name:6s} {100 * (sampled[phase] > 0.02).mean():8.0f}% {sampled[phase].mean():7.2f} {rms(sampled[phase] - reference_force[phase]):8.2f} | "
          f"{averaged[phase].mean():16.2f} {rms(averaged[phase] - reference_force[phase]):9.2f} | "
          f"{100 * (substep_force[phase] > 0).mean():8.0f}% | "
          f"{rms(predicted[phase] - sampled[phase]):11.2f} {rms(predicted[phase] - averaged[phase]):12.2f}")

plt.figure(figsize=(9, 4))
plt.plot(times, sampled, color="0.75", linewidth=0.5, label="MuJoCo, last substep (what the plots show now)")
plt.plot(times, averaged, linewidth=1.0, label="MuJoCo, averaged over the step")
plt.plot(times, predicted, linewidth=1.0, label="GNN prediction")
plt.plot(times, reference_force, "k--", linewidth=1.0, label="reference")
plt.ylim(-0.2, 3 * reference_force.max())
plt.xlabel("time (s)")
plt.ylabel("normal force (N)")
plt.legend()
plt.tight_layout()
plt.savefig(run_folder / "force_sampled_vs_averaged.png", dpi=200)
print(f"\nSaved {run_folder / 'force_sampled_vs_averaged.png'}")
