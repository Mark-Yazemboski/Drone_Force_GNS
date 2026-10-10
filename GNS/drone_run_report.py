"""Collect a drone training run into one flat set of metrics and write it with
the same report writer as the cube runs (run_report.save_run_report): a
run_report.csv beside the model and one row per run in the master CSV, with
columns added as needed and a rerun of the same run name replacing its row.

Metrics collected:
  per stage   epochs, final training loss (mean/std of the last epochs),
              best validation loss and its epoch, pad error by regime at that epoch
  coefficients  init, final, drift, recent slope (per 100 epochs) for k_f, k_m,
              k_rot, k_body, k_rod, k_pad (stage 1) and mu (stage 2); true values
              and relative error where the generator recorded them (k_f, k_m,
              k_rot, mu). A slope near zero means the value settled; a large
              slope means it was still moving when training stopped.
  test        everything evaluate_on_dataset() returns, prefixed "test_"
"""

from dataclasses import asdict

import numpy as np

from run_report import save_run_report


def _tail_slope(epochs, values, per=100.0):
    n = max(2, len(values) // 10)
    if len(values) < 2:
        return float("nan")
    return float(np.polyfit(np.asarray(epochs[-n:], float), np.asarray(values[-n:], float), 1)[0] * per)


def collect_drone_metrics(history, test_metrics=None, true_values=None, last_n=10):
    out = {}
    for stage in ("aero", "contact"):
        h = history.get(stage)
        if not h or not h["train"]:
            continue
        pred = np.asarray([p for _, _, p in h["train"]], float)
        tail = pred[-last_n:]
        out[f"{stage}_epochs"] = len(pred)
        out[f"{stage}_final_train_loss"] = float(tail.mean())
        out[f"{stage}_final_train_loss_std"] = float(tail.std(ddof=1)) if tail.size > 1 else 0.0
        if h["val"]:
            best = min(h["val"], key=lambda v: v["loss"])
            out[f"{stage}_best_val_loss"] = best["loss"]
            out[f"{stage}_best_val_epoch"] = best["epoch"]
            for regime, m in best["kstep"].items():
                out[f"{stage}_best_pad_err_mm_{regime}"] = m["pad_end_mm"]

    # Learned physical coefficients: k's move in stage 1, mu in stage 2.
    true_values = true_values or {}
    traces = (("aero", ("k_f", "k_m", "k_rot", "k_body", "k_rod", "k_pad")), ("contact", ("mu",)))
    for stage, names in traces:
        params = history.get(stage, {}).get("params", [])
        if not params:
            continue
        ep = [p["epoch"] for p in params]
        # "final" = the coefficients of the checkpoint actually kept (best
        # validation epoch), which is the model that gets evaluated. Older
        # histories without a "best" entry fall back to the last epoch.
        kept = history[stage].get("best", params[-1])
        for name in names:
            v = [p[name] for p in params]
            out[f"{name}_init"] = v[0]
            out[f"{name}_final"] = kept[name]
            out[f"{name}_last_epoch"] = v[-1]
            out[f"{name}_drift"] = kept[name] - v[0]
            out[f"{name}_slope_per_100ep"] = _tail_slope(ep, v)
            if name in true_values:
                out[f"{name}_true"] = true_values[name]
                out[f"{name}_rel_err"] = (kept[name] - true_values[name]) / true_values[name]

    for k, v in (test_metrics or {}).items():
        out[f"test_{k}"] = v
    return out


# True coefficient values from the MuJoCo generator's metadata (None if absent).
def true_values_from_meta(meta, cfg):
    out = {"k_f": cfg.k_f, "k_m": cfg.k_m}
    if meta and "k_rotor" in meta:
        out["k_rot"] = float(meta["k_rotor"])
    if meta and "mu" in meta:
        out["mu"] = float(meta["mu"])
    return out


def save_drone_run_report(model_folder, run_name, master_csv, settings, cfg, metrics, extra_settings=None):
    flat_settings = dict(asdict(settings))
    flat_settings.update({f"drone.{k}": v for k, v in cfg.to_dict().items()})
    flat_settings.update(extra_settings or {})
    clean = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool)) else float("nan"))
             for k, v in metrics.items()}
    save_run_report(model_folder, flat_settings, clean, run_name, master_csv)
