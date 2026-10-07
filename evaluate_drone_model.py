"""Evaluate a trained drone force model and save its figures.

  1. Test set (held-out generator trajectories): validation losses for both
     stages, k-step error out to val_horizon by regime, and learned vs. MuJoCo
     forces (sustained contact, impacts, aero, rotor axial correction).
  2. Closed loop (see drone_closed_loop.py): fresh scenarios flown by the same
     controller with MuJoCo and with the model as the plant. Reports pad error,
     contact-onset timing, and contact force error (one step and closed loop).
  3. Figures for the first few closed-loop runs: a time-series PNG and a GIF.

Returns one flat dict of metrics for the run report.
"""

import os

import numpy as np
import torch

from train_drone_gns import load_checkpoint, evaluate_on_dataset, _make_phys
from drone_closed_loop import evaluate_closed_loop, gen_settings_for
from visualize_drone_model import plot_closed_loop, animate_closed_loop


def evaluate_drone_run(model_path, test_data, meta, out_folder, n_closed_loop=6,
                       closed_loop_kinds=("tap", "push", "slide"), closed_loop_seed=1234,
                       n_visualize=3, make_gifs=True, gif_stride=3, device="cpu"):
    model, cfg, s, ck = load_checkpoint(model_path, device)
    phys = _make_phys(cfg, s).to(device)
    phys.load_state_dict(ck["phys"])
    os.makedirs(out_folder, exist_ok=True)
    metrics = {}

    if test_data:
        print(f"\ntest set: {len(test_data)} trajectories")
        test = evaluate_on_dataset(model, phys, test_data, s, device)
        metrics.update({f"test_{k}": v for k, v in test.items()})
        H = s.val_horizon
        print(f"  loss: aero {test['aero_loss']:.3e}  contact {test['contact_loss']:.3e}")
        print(f"  pad error after {H} steps: " + "   ".join(
            f"{r} {test[f'kstep{H}_{r}_pad_end_mm']:.3f} mm" for r in ("free", "near", "contact", "all")
            if f"kstep{H}_{r}_pad_end_mm" in test))
        for name in ("sustained", "impact"):
            if f"force_{name}_err" in test:
                print(f"  contact force, {name}: RMSE {test[f'force_{name}_err']:.3f} N "
                      f"(label RMS {test[f'force_{name}_ref']:.3f} N)")
        if "force_aero_err" in test:
            print(f"  aero force: RMSE {test['force_aero_err']:.4f} N (label RMS {test['force_aero_ref']:.4f} N)")

    if n_closed_loop:
        gs = gen_settings_for(meta)
        print(f"\nclosed loop: {n_closed_loop} scenarios, same controller with MuJoCo and with the model as plant")
        results, avg = evaluate_closed_loop(model, cfg, gs, n_runs=n_closed_loop, seed=closed_loop_seed,
                                            kinds=closed_loop_kinds, device=device)
        metrics.update({f"closed_loop_{k}": v for k, v in avg.items()})
        for kind in sorted({r["scenario"]["kind"] for r in results}):
            sel = [r["metrics"] for r in results if r["scenario"]["kind"] == kind]
            for k in ("pad_err_mean_mm", "pad_err_contact_mm", "contact_onset_err_ms",
                      "closed_loop_sustained_contact_rmse_N", "one_step_sustained_contact_rmse_N"):
                metrics[f"closed_loop_{kind}_{k}"] = float(np.nanmean([m.get(k, np.nan) for m in sel]))
        mu_true = meta.get("mu") if meta else None
        for i, r in enumerate(results[:n_visualize]):
            r["mu_true"] = mu_true
            stem = os.path.join(out_folder, f"closed_loop_{i}_{r['scenario']['kind']}")
            plot_closed_loop(r, cfg, stem + ".png")
            if make_gifs:
                animate_closed_loop(r, cfg, stem + ".gif", stride=gif_stride)
            print(f"  saved {os.path.basename(stem)}.png" + (" and .gif" if make_gifs else ""))
        torch.save([dict(scenario=r["scenario"], metrics=r["metrics"]) for r in results],
                   os.path.join(out_folder, "closed_loop_summary.pt"))
    return metrics
