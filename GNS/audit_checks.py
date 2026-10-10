"""Leakage audit. Two checks, each prints PASS/FAIL with the numbers behind it.

    python audit_checks.py --run-file Test_1.py            # both checks
    python audit_checks.py --run-file Test_1.py --only causality
    python audit_checks.py --run-file Test_1.py --only labels

1. GENERATOR CAUSALITY (needs MuJoCo, not your data)
   Simulates the same trajectory twice from the same seed. The second time, an
   external force pulse hits the drone partway through frame interval [m, m+1].
   Everything recorded up to and including frame m must be IDENTICAL, in
   particular rotor_speed[m], which is aligned over [m-1, m+1] and so includes
   the motor response during the interval where the pulse happens. It may only
   change if the controller saw the pulse before frame m+1, i.e. if commands
   inside an interval react to what happens inside it.
     ctrl_every = substeps (current, 100 Hz)  -> must be identical    (the check)
     ctrl_every = 4        (old 500 Hz)       -> differs              (shows the old leak)

2. FORCE LABELS NEVER REACH TRAINING (uses --n-traj trajectories of your data)
   Trains a short two-stage run twice from the same seed: once with the real
   F_contact / F_aero / tau_contact / F_pad labels, once with the labels
   replaced by random noise. Training losses, validation losses, the chosen
   checkpoints, and every weight must be bit-identical. The labels only feed
   the force diagnostics, so those are the one thing allowed to differ (and
   must, or the poisoning did nothing). Runs on CPU with deterministic
   algorithms, because GPU scatter-adds are not bitwise reproducible even
   without poisoning.
"""

import argparse
import copy
import dataclasses
import importlib.util
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_run_file(path):
    spec = importlib.util.spec_from_file_location("run_settings", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)                 # top level only; its __main__ block does not run
    return mod


# ----------------------------------------------------------------------
# 1. Generator causality
# ----------------------------------------------------------------------

def check_causality(n_seeds=4, pulse_N=1.5):
    import mujoco
    import mujoco_drone_generator as MG
    from drone_config import DroneConfig

    print("\n" + "=" * 72 + "\n1. GENERATOR CAUSALITY\n" + "=" * 72)
    ok_all = True
    for ctrl_every, must_match in ((MG.GenSettings().substeps, True), (4, False)):
        gs = dataclasses.replace(MG.GenSettings(), ctrl_every=ctrl_every)
        model = mujoco.MjModel.from_xml_string(MG.build_xml(DroneConfig(), gs))
        cfg_true = MG.true_config_from_model(model, DroneConfig(), gs)
        dt = gs.timestep * gs.substeps
        label = f"ctrl_every={ctrl_every} ({1 / (gs.timestep * ctrl_every):.0f} Hz)"
        diffs = []
        for seed in range(n_seeds):
            try:
                base = MG.simulate_trajectory(model, cfg_true, gs, np.random.default_rng([99, seed]))
            except MG.Crash:
                continue
            T = base["pos"].shape[0]
            # pulse inside interval [m, m+1], away from both ends; m where something is happening
            Fc = base["F_contact"].norm(dim=-1)
            touch = torch.nonzero(Fc > 0.2).flatten()
            m = int(touch[0]) - 1 if len(touch) else T // 2       # just before contact onset if any
            m = min(max(m, 5), T - 5)
            t0 = (m + 0.3) * dt
            pulse = (t0, t0 + 0.4 * dt, np.array([pulse_N, -pulse_N, 0.5 * pulse_N]))
            try:
                hit = MG.simulate_trajectory(model, cfg_true, gs, np.random.default_rng([99, seed]), pulse=pulse)
            except MG.Crash:
                continue
            past = slice(0, m + 1)
            d_rot = float((base["rotor_speed"][past] - hit["rotor_speed"][past]).abs().max())
            d_pos = float((base["pos"][past] - hit["pos"][past]).abs().max())
            d_after = float((base["pos"][m + 2:m + 6] - hit["pos"][m + 2:m + 6]).abs().max())
            diffs.append((seed, m, d_rot, d_pos, d_after))
        if not diffs:
            print(f"  {label}: every trajectory crashed, cannot check")
            ok_all = False
            continue
        worst_rot = max(d[2] for d in diffs)
        worst_pos = max(d[3] for d in diffs)
        reacted = min(d[4] for d in diffs)
        print(f"  {label}: {len(diffs)} trajectories, pulse inside frame interval [m, m+1]")
        print(f"      max |rotor_speed[0..m] change| = {worst_rot:.3e} rad/s   "
              f"max |pos[0..m] change| = {worst_pos:.3e} m")
        print(f"      (pulse did act: pos change at frames m+2..m+5 >= {reacted:.2e} m)")
        if must_match:
            ok = worst_rot == 0.0 and worst_pos == 0.0 and reacted > 0
            ok_all &= ok
            print(f"      {'PASS' if ok else 'FAIL'}: recorded inputs up to frame m do not depend on the future")
        else:
            print(f"      {'as expected' if worst_rot > 0 else 'unexpected'}: the old 500 Hz controller "
                  f"{'leaks' if worst_rot > 0 else 'did not leak here'} future information into rotor_speed[m]")
    return ok_all


# ----------------------------------------------------------------------
# 2. Labels never reach training
# ----------------------------------------------------------------------

def _poison(dataset, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for d in dataset:
        d = dict(d)
        for k in ("F_contact", "F_aero", "tau_contact", "F_pad"):
            if k in d:
                d[k] = 5.0 * torch.randn(d[k].shape, generator=g)
        out.append(d)
    return out


def _train_once(cfg, s, train, val, folder, seed):
    from train_drone_gns import train_drone_force_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    path = os.path.join(folder, "m.pt")
    model, phys, history = train_drone_force_model(cfg, s, train, val, path, device="cpu", verbose=False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state.update({f"phys.{k}": v.clone() for k, v in phys.state_dict().items()})
    return state, history


def check_labels(run, n_traj, epochs):
    from drone_data import build_drone_dataset
    print("\n" + "=" * 72 + "\n2. FORCE LABELS NEVER REACH TRAINING\n" + "=" * 72)
    torch.use_deterministic_algorithms(True)
    cfg = run.drone
    s = dataclasses.replace(run.settings, aero_epochs=epochs, contact_epochs=epochs, val_interval=1,
                            coeff_warmup_epochs=0)
    ids = list(run.train_range)[:n_traj]
    vids = list(run.val_range)[:max(2, n_traj // 3)]
    train, _ = build_drone_dataset(ids, run.trajectory_folder, cfg, run.rotor_speed_in_rpm, run.rotor_speed_hold)
    val, _ = build_drone_dataset(vids, run.trajectory_folder, cfg, run.rotor_speed_in_rpm, run.rotor_speed_hold)
    print(f"  {len(train)} train / {len(val)} val trajectories, {epochs} epochs per stage, CPU, batch {s.batch_size}")

    with tempfile.TemporaryDirectory() as tmp:
        a_dir, b_dir = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        os.makedirs(a_dir), os.makedirs(b_dir)
        print("  training with the real labels ...", flush=True)
        st_a, h_a = _train_once(cfg, s, copy.deepcopy(train), copy.deepcopy(val), a_dir, seed=0)
        print("  training with the labels replaced by noise ...", flush=True)
        st_b, h_b = _train_once(cfg, s, _poison(train, 1), _poison(val, 2), b_dir, seed=0)

    ok = True
    for stage in ("aero", "contact"):
        tr_a = [t[1:] for t in h_a[stage]["train"]]
        tr_b = [t[1:] for t in h_b[stage]["train"]]
        vl_a = [v["loss"] for v in h_a[stage]["val"]]
        vl_b = [v["loss"] for v in h_b[stage]["val"]]
        same_tr, same_vl = tr_a == tr_b, vl_a == vl_b
        best_a, best_b = h_a[stage].get("best", {}).get("epoch"), h_b[stage].get("best", {}).get("epoch")
        print(f"  [{stage}] train losses identical: {same_tr}   val losses identical: {same_vl}   "
              f"best epoch {best_a} vs {best_b}")
        ok &= same_tr and same_vl and best_a == best_b
    worst = max(float((st_a[k].double() - st_b[k].double()).abs().max()) for k in st_a)
    print(f"  max |weight difference| over all {len(st_a)} tensors: {worst:.3e}")
    ok &= worst == 0.0

    # The diagnostics DO read the labels, so they must differ (else nothing was poisoned).
    fa = h_a["contact"]["val"][-1]["forces"]
    fb = h_b["contact"]["val"][-1]["forces"]
    poisoned = fa.get("aero_ref") != fb.get("aero_ref")
    print(f"  force diagnostics differ, as they should (aero label RMS {fa.get('aero_ref', float('nan')):.4f} vs "
          f"{fb.get('aero_ref', float('nan')):.4f} N): {poisoned}")
    ok &= poisoned
    print(f"  {'PASS' if ok else 'FAIL'}: training never sees the force labels")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-file", default=os.path.join(HERE, "run_drone.py"))
    ap.add_argument("--only", choices=("causality", "labels"), default=None)
    ap.add_argument("--n-traj", type=int, default=12, help="training trajectories for the label check")
    ap.add_argument("--epochs", type=int, default=3, help="epochs per stage for the label check")
    a = ap.parse_args()

    results = {}
    if a.only in (None, "causality"):
        results["generator causality"] = check_causality()
    if a.only in (None, "labels"):
        results["labels never reach training"] = check_labels(load_run_file(a.run_file), a.n_traj, a.epochs)
    print("\n" + "=" * 72)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 72)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
