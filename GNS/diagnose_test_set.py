"""Where does the test contact loss come from? Per-trajectory and per-window breakdown.

    python diagnose_test_set.py --run-file Test_1.py
    python diagnose_test_set.py --run-file baseline_100hz_1.py --model models/baseline_100hz_1/baseline_100hz_1_final.pt

Uses the run file for the data folder, splits, and (by default) the model path.
Prints:
  1. the overall contact loss on the test set and on the validation set, computed
     exactly as in the CSV (same windows, same normalization), as a check
  2. per test trajectory, sorted by its SHARE of the total loss: scenario, number
     of windows, mean loss, peak MuJoCo contact force, peak pad approach speed,
     and the single worst window (frame, label force there)
  3. how concentrated the loss is (share from the worst 1% / 5% of windows)
  4. the same top contributors on the validation set, for contrast
  5. training coverage: for the worst test trajectories, what fraction of
     training trajectories reach that peak force / approach speed
If one or two test trajectories carry most of the loss and sit outside what the
training set covers, it is a data-coverage problem, not a model problem.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from drone_data import build_drone_dataset, annotate_pad_distance, iterate_drone_chains
from train_drone_gns import (load_checkpoint, _make_phys, stage_windows, validation_loss,
                             per_window_contact_loss)


def load_run_file(path):
    spec = importlib.util.spec_from_file_location("run_settings", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def parse_range(txt, default):
    if txt is None:
        return list(default)
    a, b = txt.split("-")
    return list(range(int(a), int(b) + 1))


per_window_loss = per_window_contact_loss          # shared with evaluate_on_dataset


def traj_stats(d, dt):
    F = d["F_contact"].norm(dim=-1).numpy() if "F_contact" in d else np.zeros(d["T"])
    dist = d["pad_dmin"].numpy()
    v_in = -np.diff(dist) / dt                                   # m/s toward the wall
    near = dist[1:] < 0.02
    return dict(peak_F=float(F.max()), peak_approach=float(v_in[near].max()) if near.any() else 0.0, F=F)


def breakdown(model, phys, data, ids, scen, s, device, label, top):
    w = stage_windows(data, data, s)
    idx = w["vl_contact"]
    loss_csv = validation_loss(model, phys, data, idx, s, device, "contact")
    pw = per_window_loss(model, data, idx, s, device)
    print(f"\n[{label}] contact loss: {loss_csv:.4g} (as in the CSV) | per-window mean {pw.mean():.4g} "
          f"| {len(pw)} windows from {len(data)} trajectories")
    tot = pw.sum()
    srt = np.sort(pw)[::-1]
    for frac in (0.01, 0.05):
        n = max(1, int(frac * len(pw)))
        print(f"    worst {100 * frac:.0f}% of windows ({n}) carry {100 * srt[:n].sum() / tot:.1f}% of the loss")
    rows = []
    ti_arr = np.array([t for t, _ in idx])
    st_arr = np.array([st for _, st in idx])
    for ti, d in enumerate(data):
        m = ti_arr == ti
        if not m.any():
            continue
        st = traj_stats(d, float(model.dt))
        j = int(np.argmax(pw[m]))
        frame = int(st_arr[m][j]) + s.h
        rows.append(dict(id=ids[ti], scen=scen[ti], n=int(m.sum()), mean=float(pw[m].mean()),
                         share=float(pw[m].sum() / tot), worst=float(pw[m][j]), frame=frame,
                         F_at=float(st["F"][frame:frame + s.multistep].max()),
                         peak_F=st["peak_F"], peak_v=st["peak_approach"]))
    rows.sort(key=lambda r: -r["share"])
    print(f"    {'traj':>5} {'scenario':>10} {'windows':>7} {'mean loss':>10} {'share':>6} | "
          f"{'peak |F| N':>10} {'approach m/s':>12} | worst window: {'loss':>8} {'frame':>6} {'|F| there':>9}")
    for r in rows[:top]:
        print(f"    {r['id']:>5} {r['scen']:>10} {r['n']:>7} {r['mean']:>10.4g} {100 * r['share']:>5.1f}% | "
              f"{r['peak_F']:>10.2f} {r['peak_v']:>12.3f} | {'':>14}{r['worst']:>8.3g} {r['frame']:>6} {r['F_at']:>9.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-file", required=True)
    ap.add_argument("--model", default=None, help="checkpoint (default: the run file's _final.pt)")
    ap.add_argument("--test", default=None, help="override test range, e.g. 400-499")
    ap.add_argument("--val", default=None, help="override val range, e.g. 300-399")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    run = load_run_file(a.run_file)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = a.model or run.save_model_path.replace(".pt", "_final.pt")
    model, cfg, s, ck = load_checkpoint(path, device)
    phys = _make_phys(cfg, s).to(device)
    phys.load_state_dict(ck["phys"])
    print(f"model: {path}\ndata:  {run.trajectory_folder}")

    def load(ids):
        data, _ = build_drone_dataset(ids, run.trajectory_folder, cfg, run.rotor_speed_in_rpm, run.rotor_speed_hold)
        annotate_pad_distance(data, model.pad_rest.cpu())
        scen = [torch.load(os.path.join(run.trajectory_folder, f"{i}.pt"), weights_only=False)
                .get("meta", {}).get("scenario", "?") for i in ids]
        return data, scen

    test_ids = parse_range(a.test, run.test_range)
    val_ids = parse_range(a.val, run.val_range)
    test, test_scen = load(test_ids)
    val, val_scen = load(val_ids)

    print("\n" + "=" * 100 + "\nTEST SET" + "\n" + "=" * 100)
    rows = breakdown(model, phys, test, test_ids, test_scen, s, device, "test", a.top)
    print("\n" + "=" * 100 + "\nVALIDATION SET (for contrast)" + "\n" + "=" * 100)
    breakdown(model, phys, val, val_ids, val_scen, s, device, "val", min(a.top, 5))

    print("\n" + "=" * 100 + "\nTRAINING COVERAGE" + "\n" + "=" * 100)
    train_ids = list(run.train_range)
    train, train_scen = load(train_ids)
    tr = [traj_stats(d, float(model.dt)) for d in train]
    pF = np.array([t["peak_F"] for t in tr])
    pV = np.array([t["peak_approach"] for t in tr])
    for name, x, u in (("peak |F_contact|", pF, "N"), ("peak approach speed", pV, "m/s")):
        q = np.percentile(x, [50, 90, 99, 100])
        print(f"  training {name}: median {q[0]:.2f}  p90 {q[1]:.2f}  p99 {q[2]:.2f}  max {q[3]:.2f} {u}")
    print("  worst test trajectories vs training:")
    for r in rows[:5]:
        print(f"    traj {r['id']:>4} ({r['scen']}): peak |F| {r['peak_F']:.2f} N -> {100 * (pF >= r['peak_F']).mean():.1f}% "
              f"of training trajectories reach it | approach {r['peak_v']:.3f} m/s -> "
              f"{100 * (pV >= r['peak_v']).mean():.1f}% reach it")
    print("\nReplay one in MuJoCo:  python mujoco_drone_generator.py --replay "
          f"{os.path.join(run.trajectory_folder, str(rows[0]['id']) + '.pt')}" if rows else "")


if __name__ == "__main__":
    main()
