"""Collect saved training diagnostics and record experiment settings and results
in CSV reports. This module reads physics and loss-history checkpoints to
extract recovered parameters, recent training-loss statistics, validation
progress, and parameter drift. Its report writer combines supplied settings
and metrics into a report beside the model and updates a master CSV with one
row per run, adding columns as needed and replacing an earlier row for the
same run name. The experiment runner uses these functions to make completed
runs easier to inspect and compare.
"""

import csv
import os
from datetime import datetime

import numpy as np
import torch

# Name of the CSV file used for run reports.
# These are individual run reports saved alongside the model.
# This is not the master CSV; it only contains the report for this specific run.
REPORT_NAME = "run_report.csv"


#This function saves the run report for a specific experiment run. 
# It writes a detailed CSV in the model folder and updates the master CSV.
#inputs:
# model_folder : directory where the model and its run report will be saved
# settings     : dictionary containing the run configuration
# metrics      : dictionary containing the results of the run
# run_name     : name of the run, used as the row identifier in the master CSV
# master_csv   : path to the master CSV file that aggregates all runs
def save_run_report(model_folder, settings, metrics, run_name, master_csv):

    # Ensure the model folder exists before saving the report.
    os.makedirs(model_folder, exist_ok=True)

    # Prepare the rows for the CSV report. Each row is a tuple of (section, key, value).
    rows = [("meta", "run_name", run_name),
            ("meta", "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]

    # Add settings to the report rows.
    for k in sorted(settings.keys()):
        rows.append(("settings", str(k), str(settings[k])))

    # Add metrics to the report rows. Lists/tuples are expanded into separate rows.
    for k, v in metrics.items():

        # Check if the metric value is a list or tuple, indicating multiple sub-metrics.
        if isinstance(v, (list, tuple)):         

            # Expand each element of the list/tuple into separate rows with descriptive names.
            for name, vi in zip(("airborne", "contact", "settled"), v):
                rows.append(("metrics", f"{k}_{name}", repr(float(vi))))
        else:
            rows.append(("metrics", str(k), repr(float(v))))

    # Save the individual run report CSV in the model folder.
    report_path = os.path.join(model_folder, REPORT_NAME)

    # Write the CSV file with the prepared rows.
    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        w.writerows(rows)
    print(f"[run_report] saved {report_path}")

    # ---- master file: one wide row per run, columns grow as needed ----

    # Flatten the rows into a single dictionary for the master CSV.
    flat = {"run_name": run_name, "timestamp": rows[1][2]}

    # Convert the remaining rows into key-value pairs for the master CSV.
    for section, key, value in rows[2:]:
        flat[f"{section}.{key}"] = value

    # Read the existing master CSV if it exists, and update the fieldnames to include any new keys.
    existing, fieldnames = [], []
    if os.path.exists(master_csv):
        with open(master_csv, newline="") as f:
            rdr = csv.DictReader(f)
            fieldnames = list(rdr.fieldnames or [])
            existing = [row for row in rdr]
    for k in flat:
        if k not in fieldnames:
            fieldnames.append(k)
    existing = [r for r in existing if r.get("run_name") != run_name]  # replace reruns
    existing.append(flat)

    # Write the updated master CSV with all runs, including the current one.
    with open(master_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(existing)
    print(f"[run_report] master updated: {master_csv} ({len(existing)} runs)")


# Load a checkpoint from disk and return None when it is unavailable.
def _load(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [run_report] could not read {os.path.basename(path)}: {e}")
        return None


# Convert a saved [(epoch, value), ...] trace into separate arrays.
def _trace_arrays(trace):
    if not trace:
        return None
    a = np.asarray(trace, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
        return None
    return a[:, 0], a[:, 1]


# Read all available diagnostics associated with one saved model.
def collect_run_diagnostics(save_model_path, last_n=20):

    # Checkpoint files use the model path as their filename stem.
    stem = os.path.splitext(save_model_path)[0]
    phys_path, hist_path = stem + "_physics.pt", stem + "_loss_history.pt"
    out = {}

    # Collect recovered physical parameters from the physics checkpoint.
    ph = _load(phys_path)
    if ph is None:
        print(f"  [run_report] MISSING {os.path.basename(phys_path)}"
              " -> no recovered_mu / recovered_k_over_m")
    else:
        # Extract the recovered physical parameters if they exist.
        for key in ("recovered_mu", "recovered_k_over_m"):
            if key in ph:
                out[key] = float(ph[key])

    # Collect losses, optimizer progress, and learned-parameter traces.
    hi = _load(hist_path)
    if hi is None:
        print(f"  [run_report] MISSING {os.path.basename(hist_path)}"
              " -> no loss curves, no mu_trace, no k_trace")
    else:
        # Average the final training-loss values to reduce epoch-to-epoch noise.
        tv = hi.get("train_loss_values") or []
        if tv:
            tail = np.asarray(tv[-last_n:], dtype=float)
            tail = tail[np.isfinite(tail)]
            if tail.size:
                # The denominator of gamma = 0.03 * L_pred / raw. Averaged,
                # because the per-epoch value bounces by ~10%.
                out["final_train_loss"] = float(tail.mean())
                out["final_train_loss_std"] = (float(tail.std(ddof=1))
                                               if tail.size > 1 else 0.0)
                out["final_train_loss_n"] = float(tail.size)
        # Copy scalar training-summary values into the report metrics.
        for src, dst in (("best_val_loss", "best_val_loss"),
                         ("best_val_epoch", "best_val_epoch"),
                         ("global_step", "total_optimizer_steps"),
                         ("epochs_completed", "epochs_completed"),
                         ("stopped_early", "stopped_early")):
            if hi.get(src) is not None:
                out[dst] = float(hi[src])

        # Measure how much each learned physical parameter changed during training.
        # Its endpoint is recovered_mu / recovered_k_over_m above. The drift is
        # what distinguishes "identified" from "never moved" - a monotone descent
        # that has not arrived is a different result from a value that parked.
        for name, key in (("mu", "mu_trace"), ("k", "k_trace")):
            tr = _trace_arrays(hi.get(key))
            if tr is None:
                # Say WHY rather than leaving a silent NaN in the CSV: an
                # absent key and an empty list mean different things.
                print(f"  [run_report] no usable '{key}' "
                      + ("(key absent - trainer did not save it)"
                         if key not in hi else "(present but empty - "
                         "was the parameter learnable?)")
                      + f" -> no {name} drift")
                continue
            ep, val = tr
            out[f"{name}_init"] = float(val[0])
            out[f"{name}_drift"] = float(val[-1] - val[0])
            # Estimate recent movement; a slope near zero suggests convergence.
            n_tail = max(2, len(val) // 10)
            out[f"{name}_tail_slope_per_1k_ep"] = float(
                np.polyfit(ep[-n_tail:], val[-n_tail:], 1)[0] * 1000.0)
    return out
