"""
Checks the linearized model for a friction deadband: when the pad is stuck on the wall, does a change in
rotor speed move it in the wall plane?

Takes a few stuck frames and a few free-flight frames from a saved MPC run and prints, in mm/s of next-step
velocity per rad/s of rotor speed:
    B plane / B normal:  largest singular value of the B-matrix velocity rows, in-plane and normal to the wall
    B coll:              in-plane response to a collective change (all four rotors together), from B
    +d/-d:               the same collective response from a real step of +d and -d rad/s (secant, not tangent)

Deadband: B is ~0 at the stuck frames (vs free flight), and the secant grows once d is big enough to break
the pad loose. The d where that happens is the perturbation size to use for the smoothed Jacobian.

Run:  python deadband_check.py <run name>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import Drone_Dynamics


SCRIPT_FOLDER = Path(__file__).resolve().parent
STUCK_FORCE = 0.3           #N, MuJoCo normal force above this at both ends of the step counts as pressed on the wall
STUCK_SPEED = 0.002         #m/s, pad in-plane speed below this counts as stuck
FREE_GAP = 0.02             #m, pad surface at least this far off the wall counts as free flight
DELTAS = [1, 10, 30, 100]   #rad/s, collective steps for the secant check


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run", help="run name (folder in MPC_runs)")
ap.add_argument("--frames", type=int, default=5, help="frames of each kind to check")
args = ap.parse_args()

run_folder = SCRIPT_FOLDER / "MPC_runs" / args.run
data = np.load(run_folder / "run_data.npz")
settings = json.loads((run_folder / "settings.json").read_text())

#Same model and cutoff the run used (falls back to the local copy if the saved path is from another computer)
model_path = Path(settings["model_path"])
if not model_path.exists():
    model_path = SCRIPT_FOLDER / "MPC_Model" / "Trained_Model.pt"
model = Drone_Dynamics(torch.device("cpu"), model_path, wall_normal_vector=data["wall_normal_vector"],
                       wall_center_point=data["wall_center_point"], wind_vector=data["wind_vector"],
                       contact_cutoff_gap=settings["contact_cutoff_gap"])

#Wall frame: rows are [out of wall, along wall, up wall]
n = data["wall_normal_vector"]
along = np.cross(n, [0.0, 0.0, 1.0])
along = along / np.linalg.norm(along)
W = np.vstack([n, along, np.cross(along, n)])

times, states, inputs, normal_force = data["times"], data["states"], data["inputs"], data["normal_force"]

#Pad gap and in-plane speed at every state
with torch.no_grad():
    states_torch = torch.as_tensor(states, dtype=torch.float32)
    _, pad_velocity, _ = model.get_pad_values(states_torch)
    pad_distance, _ = model.get_wall_distances(states_torch)
pad_velocity = pad_velocity.numpy()
gap = pad_distance.numpy() - model.cfg.pad_contact_radius
in_plane_speed = np.linalg.norm(pad_velocity - (pad_velocity @ n)[:, None] * n, axis=1)

#Step i goes from states[i] to states[i+1] with inputs[i]
steps = np.arange(len(inputs))
stuck = steps[(normal_force[steps] > STUCK_FORCE) & (normal_force[steps + 1] > STUCK_FORCE)
              & (in_plane_speed[steps] < STUCK_SPEED)]
free = steps[(np.abs(normal_force[steps]) < 0.02) & (gap[steps] > FREE_GAP)]


def spread(indices):
    #Up to --frames indices, spread evenly over the run
    if len(indices) == 0:
        return indices
    return indices[np.linspace(0, len(indices) - 1, min(args.frames, len(indices))).astype(int)]


def check(i):
    x = torch.as_tensor(states[i], dtype=torch.float32)
    u = torch.as_tensor(inputs[i], dtype=torch.float32)

    #Tangent: B-matrix rows for the next-step velocity (error-state rows 3:6), in the wall frame, mm/s per rad/s
    _, B, _, _, _, _ = model.linearize(x, u, 1)
    B_velocity = 1000 * W @ B[3:6].cpu().numpy()                     # (3, 4): normal, along, up
    B_plane = np.linalg.norm(B_velocity[1:], 2)
    B_normal = np.linalg.norm(B_velocity[0])
    B_collective = np.linalg.norm(B_velocity[1:] @ np.ones(4))

    #Secant: next-step in-plane velocity change from a real collective step of +d and -d
    def in_plane_velocity(rotor_speeds):
        with torch.no_grad():
            next_state, _, _ = model.propagate_state(x[None], rotor_speeds[None])
        return (W @ next_state[0, 3:6].numpy())[1:]

    v0 = in_plane_velocity(u)
    secants = []
    for d in DELTAS:
        plus = np.linalg.norm(in_plane_velocity(u + d) - v0) / d
        minus = np.linalg.norm(v0 - in_plane_velocity(u - d)) / d
        secants.append(f"{1000 * plus:6.2f}/{1000 * minus:<6.2f}")

    print(f"{times[i]:6.2f} {normal_force[i]:6.2f} {1000 * gap[i]:7.1f} {1000 * in_plane_speed[i]:8.1f} | "
          f"{B_plane:8.3f} {B_normal:8.3f} {B_collective:8.3f} | " + " ".join(secants))
    return B_plane


header = (f"{'t s':>6s} {'Fn N':>6s} {'gap mm':>7s} {'pad mm/s':>8s} | "
          f"{'B plane':>8s} {'B normal':>8s} {'B coll':>8s} | "
          + " ".join(f"{f'd={d} +/-':^13s}" for d in DELTAS))

results = {}
for name, indices in [("STUCK", stuck), ("FREE FLIGHT", free)]:
    print(f"\n{name}: {len(indices)} frames found, checking {min(args.frames, len(indices))}")
    if len(indices) == 0:
        continue
    print(header)
    results[name] = [check(i) for i in spread(indices)]

if len(results) == 2:
    stuck_median, free_median = np.median(results["STUCK"]), np.median(results["FREE FLIGHT"])
    print(f"\nmedian in-plane B: stuck {stuck_median:.3f}, free {free_median:.3f} mm/s per rad/s "
          f"(stuck/free = {stuck_median / free_median:.2f})")
