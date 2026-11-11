"""
Checks the linearized model for a friction deadband: when the pad is stuck on the wall, does a change in
rotor speed move it in the wall plane?

Takes a few stuck frames and a few free-flight frames from a saved MPC run and prints:
    B plane / B normal:  largest singular value of the B-matrix velocity rows, in-plane and normal to the wall
                         (mm/s of next-step velocity per rad/s of rotor speed)
    B coll:              in-plane response to a collective change (all four rotors together), from B
    dD KK:               in-plane pad displacement (mm) from holding +D rad/s collective for K steps of 0.01 s,
                         vs holding the recorded input, as model/MuJoCo. Open loop (no attitude control) in both;
                         MuJoCo starts from the saved pose at rest, with its motor lag.

Deadband over one step: B at the stuck frames is ~0 compared with free flight.
Deadband over the horizon: B looks normal, but the K-step displacement at the stuck frames is ~0 compared
with free flight (friction catches up after the first step).
Model too sticky: at the stuck frames MuJoCo moves the pad much farther than the model does.

Run:  python deadband_check.py <run name>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import Drone_Dynamics
from mujoco_simulator import MujocoDroneSim


SCRIPT_FOLDER = Path(__file__).resolve().parent
STUCK_FORCE = 0.3           #N, MuJoCo normal force above this at both ends of the step counts as pressed on the wall
STUCK_SPEED = 0.002         #m/s, pad in-plane speed below this counts as stuck
FREE_GAP = 0.02             #m, pad surface at least this far off the wall counts as free flight
MULTI_STEP = [(10, 10), (10, 50), (100, 10), (100, 50)]    #(rad/s, 0.01 s steps) for the multi-step check


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

#MuJoCo with the same wall and wind, for the open-loop push comparison
sim = MujocoDroneSim(model.cfg, data["wall_normal_vector"], data["wall_center_point"], data["wind_vector"])

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

    #Multi-step: in-plane pad displacement (mm) from holding +d collective for K steps, vs holding u,
    #in the model and in MuJoCo (reset to this state at rest, motors already at u)
    def in_plane(state):
        with torch.no_grad():
            pad_position, _, _ = model.get_pad_values(torch.as_tensor(state, dtype=torch.float32)[None])
        return (W @ pad_position[0].numpy())[1:]

    def model_pad(rotor_speeds, steps):
        with torch.no_grad():
            trajectory = model.rollout(x, rotor_speeds.repeat(steps, 1), 1)
        return in_plane(trajectory[-1].numpy())

    def mujoco_pad(rotor_speeds, steps):
        state = sim.reset(states[i], inputs[i])
        for _ in range(steps):
            state = sim.step(rotor_speeds, model.dt)
        return in_plane(state)

    displacements = []
    for d, steps in MULTI_STEP:
        model_moved = np.linalg.norm(model_pad(u + d, steps) - model_pad(u, steps))
        mujoco_moved = np.linalg.norm(mujoco_pad(inputs[i] + d, steps) - mujoco_pad(inputs[i], steps))
        displacements.append(f"{1000 * model_moved:6.1f}/{1000 * mujoco_moved:<6.1f}")

    print(f"{times[i]:6.2f} {normal_force[i]:6.2f} {1000 * gap[i]:7.1f} {1000 * in_plane_speed[i]:8.1f} | "
          f"{B_plane:8.3f} {B_normal:8.3f} {B_collective:8.3f} | " + " ".join(displacements))
    return B_plane


header = (f"{'t s':>6s} {'Fn N':>6s} {'gap mm':>7s} {'pad mm/s':>8s} | "
          f"{'B plane':>8s} {'B normal':>8s} {'B coll':>8s} | "
          + " ".join(f"{f'd{d} K{steps}':^13s}" for d, steps in MULTI_STEP))

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
