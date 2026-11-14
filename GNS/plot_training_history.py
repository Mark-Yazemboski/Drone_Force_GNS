"""Training history for both stages, in the same report style as the MPC figures
(see visualize_drone_model.set_plot_style). Four figures, each .png and .svg:

    <path stem>_aero, <path stem>_contact
        validation loss (normalized Huber prediction loss on validation windows, log
        scale). The star and the dotted line mark the checkpoint kept ("best model").
    <path stem>_coefficients_aero
        k_f, k_m, k_rot divided by their true values (1.0 = recovered exactly)
    <path stem>_coefficients_contact
        the learned mu parameter (it only feeds the friction regularizer), the
        friction-to-normal ratio of the PREDICTED sliding forces on validation (what the
        model actually does), the same ratio for MuJoCo's labels, and the true mu
    Legend values are at the best model.

x axis: optimizer steps (bottom) and epochs (top). Histories from before step-based
training have no steps-per-epoch record and are plotted in epochs. Histories without a
Huber validation loss fall back to the MSE one.

plot_loss_history also returns how much the val MSE fell over the last 25% of each
stage. Still falling a few percent or more -> more training would likely help. With
lr_schedule="cosine" the learning rate goes to 0 at the end of each stage, so the tail
flattens partly by construction; look at the middle of the curve too.

Standalone:  python plot_training_history.py models/<run>/<run>_history.pt [out.png]
"""

import os
import sys

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from visualize_drone_model import (set_plot_style, figure_legend, save_figure, REPORT_WIDTH,
                                   MUJOCO_COLOR, ONE_STEP_COLOR, RUN_COLOR)

C_VAL, C_BEST = ONE_STEP_COLOR, MUJOCO_COLOR
COEFF_COLORS = {"k_f": ONE_STEP_COLOR, "k_m": "#2ca02c", "k_rot": RUN_COLOR}
COEFF_LABELS = {"k_f": "$k_f$", "k_m": "$k_m$", "k_rot": r"$k_\mathrm{rot}$"}


def val_tail_change(hist, frac=0.25):
    """Relative change of the val loss over the last `frac` of the stage
    (negative = still improving). NaN if there are too few validation points."""
    val = [(v["epoch"], v["loss"]) for v in hist.get("val", [])]
    if len(val) < 2 or not hist.get("train"):
        return float("nan")
    last_epoch = hist["train"][-1][0]
    cut = last_epoch - frac * (last_epoch + 1)
    ep = np.array([e for e, _ in val])
    loss = np.array([l for _, l in val])
    early = loss[ep <= cut]
    ref = early[-1] if len(early) else loss[0]
    return float(loss[-1] / ref - 1.0)


# Epoch index -> optimizer steps completed at the end of that epoch (or the epoch
# itself for histories without a steps-per-epoch record).
def _xaxis(hist):
    spe = hist.get("steps_per_epoch")
    if spe:
        return (lambda ep: (np.asarray(ep, float) + 1) * spe), "optimizer step", spe
    return (lambda ep: np.asarray(ep, float)), "epoch", None


def _epoch_axis(ax, spe):
    if spe:
        top = ax.secondary_xaxis("top", functions=(lambda x: x / spe, lambda e: e * spe))
        top.set_xlabel("epoch")


def _kept_epoch(hist):
    return hist.get("best", {}).get("epoch")


def _kept_index(epochs, hist, fallback):
    kept = _kept_epoch(hist)
    return int(np.flatnonzero(epochs == kept)[0]) if kept is not None and (epochs == kept).any() else fallback


def _kept_line(ax, X, hist):
    kept = _kept_epoch(hist)
    if kept is not None:
        ax.axvline(X(kept), color=C_BEST, ls=":", lw=0.8)


def _loss_figure(hist):
    X, xlabel, spe = _xaxis(hist)
    v = hist["val"]
    ve = np.array([x["epoch"] for x in v])
    loss = np.array([x.get("loss_huber", np.nan) for x in v], float)
    label = "val Huber"
    if not np.isfinite(loss).any():
        loss, label = np.array([x["loss"] for x in v], float), "val MSE"

    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 2.4))
    ax.plot(X(ve), loss, "o-", color=C_VAL, ms=2, label=label)
    i = _kept_index(ve, hist, int(np.nanargmin(loss)))
    ax.plot(X(ve[i]), loss[i], "*", color=C_BEST, ms=8, zorder=6, label="best model")
    _kept_line(ax, X, hist)
    ax.set_yscale("log")
    # plain numbers (0.002, not 2x10^-3); label the minor ticks too when the range is under a decade
    plain = FuncFormatter(lambda y, _: f"{y:g}")
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(plain if np.nanmax(loss) / np.nanmin(loss) < 10 else NullFormatter())
    ax.grid(True, which="both", alpha=0.2, linewidth=0.4)
    _epoch_axis(ax, spe)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("normalized loss")
    figure_legend(fig, [ax], ncol=2)
    return fig


def _coeff_figure(hist, stage, true):
    p = hist.get("params", [])
    if not p:
        return None
    X, xlabel, spe = _xaxis(hist)
    ep = X([q["epoch"] for q in p])
    best = hist.get("best") or p[-1]            # coefficients of the kept checkpoint
    fig, ax = plt.subplots(figsize=(REPORT_WIDTH, 2.4))
    if stage == "aero":
        for k in ("k_f", "k_m", "k_rot"):
            if true.get(k):
                ax.plot(ep, np.array([q[k] for q in p]) / true[k], color=COEFF_COLORS[k],
                        label=f"{COEFF_LABELS[k]}: {best[k] / true[k]:.3f}")
        ax.axhline(1.0, color=C_BEST, ls=":", lw=0.6)
        ax.set_ylabel("learned / true")
        ncol = 3
    else:
        mu = np.array([q["mu"] for q in p])
        ax.plot(ep, mu, color=RUN_COLOR, label=rf"learned $\mu$: {best['mu']:.3f}")
        # Friction-to-normal ratio of the sliding forces, at each validation pass.
        vals = [v for v in hist.get("val", []) if "slide_mu_implied" in v.get("forces", {})]
        if vals:
            ve = np.array([v["epoch"] for v in vals])
            imp = np.array([v["forces"]["slide_mu_implied"] for v in vals])
            lab = np.array([v["forces"].get("slide_mu_implied_label", np.nan) for v in vals])
            j = _kept_index(ve, hist, len(imp) - 1)
            ax.plot(X(ve), imp, "o-", color=ONE_STEP_COLOR, ms=1.5, lw=0.8,
                    label=rf"$\|F_t\|/F_n$, model: {imp[j]:.3f}")
            if np.isfinite(lab).any():
                ax.axhline(float(np.nanmean(lab)), color=MUJOCO_COLOR, ls="--", lw=0.8,
                           label=rf"$\|F_t\|/F_n$, MuJoCo: {np.nanmean(lab):.3f}")
        if true.get("mu"):
            ax.axhline(true["mu"], color=MUJOCO_COLOR, ls=":", lw=0.8, label=rf"true $\mu$: {true['mu']:.2f}")
        # Early in training the implied ratio can be far off (tiny normal forces);
        # cap the axis so it does not squash the mu lines.
        ax.set_ylim(0.0, min(1.0, max(0.6, ax.get_ylim()[1])))
        ax.set_ylabel(r"$\mu$")
        ncol = 2
    _kept_line(ax, X, hist)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    _epoch_axis(ax, spe)
    ax.set_xlabel(xlabel)
    figure_legend(fig, [ax], ncol=ncol)
    return fig


def plot_loss_history(history, path, true=None, title=None):
    """history: the dict saved as <run>_history.pt. true: {"k_f", "k_m", "k_rot", "mu"}
    (see drone_run_report.true_values_from_meta). Saves the four figures listed at the
    top of this file. title is not drawn (report figures have no titles).
    Returns {"<stage>_val_loss_tail_change": x}."""
    true = true or {}
    stem = os.path.splitext(path)[0]
    set_plot_style()
    out = {}
    for st in ("aero", "contact"):
        hist = history.get(st, {})
        if not hist.get("train"):
            continue
        if hist.get("val"):
            save_figure(_loss_figure(hist), f"{stem}_{st}.png")
        fig = _coeff_figure(hist, st, true)
        if fig is not None:
            save_figure(fig, f"{stem}_coefficients_{st}.png")
        out[f"{st}_val_loss_tail_change"] = val_tail_change(hist)
    return out


if __name__ == "__main__":
    import torch
    hist_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else hist_path.replace("_history.pt", "_loss_history.png")
    plot_loss_history(torch.load(hist_path, weights_only=False), out_path)
    print(f"saved {os.path.splitext(out_path)[0]}_aero / _contact / _coefficients_* (.png, .svg)")
