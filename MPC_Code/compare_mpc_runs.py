"""
Compares saved MPC runs (MPC_runs/<name>/run_data.npz) in one table.

Phantom force:     the model's predicted normal force on frames where MuJoCo has no contact force
                   and the pad is 1-10 mm off the wall (what the MPC sees while hovering just off it)
Force prediction:  model vs MuJoCo normal force in sustained contact (one step from the real state)
Contact phase:     share of the reference's contact phase the pad actually touches, how many times it
                   loses contact (bounces), normal force error vs the reference, in-plane pad error

Run:  python compare_mpc_runs.py rev_cutoff rev_no_cutoff apart_cutoff apart_no_cutoff
"""

import argparse
from pathlib import Path

import numpy as np


SCRIPT_FOLDER = Path(__file__).resolve().parent
PAD_RADIUS = 0.004          #m, pad contact sphere radius (model_pad_distance is to the sphere's center)
CONTACT_FORCE = 0.05        #N, MuJoCo normal force above this counts as touching
NO_FORCE = 0.02             #N, MuJoCo normal force below this counts as no contact


def quat_to_rotation(q):
    #[w, x, y, z] -> rotation matrix
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def count_contact_losses(touching, min_gap=3):
    #Times the pad goes from touching to not touching for at least min_gap frames
    losses, gap = 0, 0
    was_touching = False
    for t in touching:
        if t:
            if was_touching and gap >= min_gap:
                losses += 1
            was_touching, gap = True, 0
        else:
            gap += 1
    return losses


def run_metrics(run_folder):
    data = np.load(run_folder / "run_data.npz")
    times, states = data["times"], data["states"]
    mujoco_force, predicted_force = data["normal_force"], data["predicted_force"]
    n = data["wall_normal_vector"]
    wall_point = data["wall_center_point"]

    #Reference at every saved time (the saved reference is the coarse plan)
    ref_time = data["reference_time"]
    ref_force = np.interp(times, ref_time, data["reference_normal_force"])
    ref_contact = np.round(np.interp(times, ref_time, data["reference_contact_flag"])) > 0.5
    ref_pad = np.stack([np.interp(times, ref_time, data["reference_pad_position"][:, i]) for i in range(3)], 1)

    #Pad offset (COM -> pad, body frame) from the start: the reference starts at the starting pad position
    R0 = quat_to_rotation(data["initial_state"][6:10])
    pad_offset = R0.T @ (data["reference_pad_position"][0] - data["initial_state"][0:3])
    pad = np.array([s[0:3] + quat_to_rotation(s[6:10]) @ pad_offset for s in states])

    #In-plane pad error (normal direction removed: during contact the MPC doesn't track it)
    in_plane = lambda v: v - (v @ n)[:, None] * n
    pad_error = 1000 * np.linalg.norm(in_plane(pad - ref_pad), axis=1)

    #predicted_force[j] is the step from state j-1 to j, so its gap is from state j-1
    gap = np.full(len(times), np.nan)
    gap[1:] = data["model_pad_distance"][:-1] - PAD_RADIUS

    out = dict(seed=int(data["seed"]) if "seed" in data else -1, push_force=float(ref_force.max()))

    phantom = (mujoco_force < NO_FORCE) & (gap > 0.001) & (gap < 0.010)
    out["phantom_n"] = int(phantom.sum())
    out["phantom_mean"] = float(np.abs(predicted_force[phantom]).mean()) if phantom.any() else np.nan
    out["phantom_max"] = float(np.abs(predicted_force[phantom]).max()) if phantom.any() else np.nan

    sustained = (mujoco_force > CONTACT_FORCE) & (mujoco_force < 10.0)
    out["force_rmse"] = float(np.sqrt(np.mean((predicted_force[sustained] - mujoco_force[sustained]) ** 2))) \
        if sustained.any() else np.nan

    touching = mujoco_force > CONTACT_FORCE
    out["contact_frac"] = float(touching[ref_contact].mean()) if ref_contact.any() else np.nan
    out["contact_losses"] = count_contact_losses(touching[ref_contact])
    out["force_err"] = float(np.sqrt(np.mean((mujoco_force[ref_contact] - ref_force[ref_contact]) ** 2)))
    out["pad_err_rms"] = float(np.sqrt(np.mean(pad_error[ref_contact] ** 2)))
    out["pad_err_max"] = float(pad_error[ref_contact].max())
    return out


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("runs", nargs="+", help="run names (folders in MPC_runs)")
args = ap.parse_args()

rows = {name: run_metrics(SCRIPT_FOLDER / "MPC_runs" / name) for name in args.runs}

print(f"\n{'':>22s} | {'phantom F_n (1-10 mm off)':^28s} | {'force pred':^10s} | {'contact phase vs reference':^52s}")
print(f"{'run':>22s} | {'frames':>7s} {'mean N':>9s} {'max N':>9s} | {'RMSE N':>10s} | "
      f"{'touching':>9s} {'losses':>7s} {'F_n RMSE N':>11s} {'pad RMS mm':>11s} {'pad max mm':>11s}")
for name, m in rows.items():
    print(f"{name:>22s} | {m['phantom_n']:7d} {m['phantom_mean']:9.3f} {m['phantom_max']:9.3f} | {m['force_rmse']:10.3f} | "
          f"{100 * m['contact_frac']:8.0f}% {m['contact_losses']:7d} {m['force_err']:11.3f} "
          f"{m['pad_err_rms']:11.1f} {m['pad_err_max']:11.1f}")

seeds = {m["seed"] for m in rows.values()}
print("\nseeds:", ", ".join(f"{name} {m['seed']} (push {m['push_force']:.2f} N)" for name, m in rows.items()))
if len(seeds) > 1:
    print("WARNING: the runs used different seeds, so they flew different scenarios: compare with care")
