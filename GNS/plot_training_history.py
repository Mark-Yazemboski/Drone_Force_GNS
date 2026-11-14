"""Loss history for both training stages, to judge whether more training would help.

One figure per stage (<path stem>_aero and <path stem>_contact, each .png and .svg), in
the same report style as the MPC figures (see visualize_drone_model.set_plot_style).

x axis: optimizer steps (bottom) and epochs (top). Training length is set in steps,
so curves from runs with different amounts of data line up. Histories from
before step-based training have no steps-per-epoch record and are plotted in epochs.

Top panel, log scale:
    train                 the normalized prediction loss on training windows, in the training
                          loss (MSE or Huber; for MSE, 1 = no better than zero learned force).
                          Inputs are noised, so it sits above val.
    train + regularizers  the stage's full training loss (anchor, smoothness, friction terms...)
    val MSE / val Huber   the normalized prediction loss on validation windows
    val pad error         pad position error after val_horizon steps (mm, right axis)
    The star is on the metric that chose the checkpoint (best_metric), at the
    checkpoint kept ("best model"); the dotted vertical line marks it in both panels.
Bottom panel (its own legend): learned coefficients, with their values at the best model.
    aero     k_f, k_m, k_rot divided by their true values (1.0 = recovered exactly)
    contact  the learned mu parameter (it only feeds the friction regularizer), the
             friction-to-normal ratio of the PREDICTED sliding forces on validation
             (what the model actually does), the same ratio for MuJoCo's labels, and
             the true mu

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
                                   ACTUAL_COLOR, REFERENCE_COLOR, MODEL_COLOR)

C_TRAIN, C_TOTAL, C_VAL, C_PAD, C_BEST = ACTUAL_COLOR, "#9DBFE6", REFERENCE_COLOR, "#8a4fbf", "#2C2C2A"
COEFF_COLORS = {"k_f": ACTUAL_COLOR, "k_m": MODEL_COLOR, "k_rot": REFERENCE_COLOR}
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


def _val_series(hist):
    """epoch -> (MSE, Huber, pad error mm) for every validation pass."""
    v = hist.get("val", [])
    ep = np.array([x["epoch"] for x in v])
    mse = np.array([x["loss"] for x in v], float)
    hub = np.array([x.get("loss_huber", np.nan) for x in v], float)
    pad = np.array([x.get("kstep", {}).get("all", {}).get("pad_end_mm", np.nan) for x in v], float)
    return ep, mse, hub, pad


def _kept_line(ax, X, hist):
    kept = _kept_epoch(hist)
    if kept is not None:
        ax.axvline(X(kept), color=C_BEST, ls=":", lw=0.8)


# Returns the axes drawn on (the pad-error axis is a second y axis), for the legend.
def _loss_panel(ax, hist, best_metric="val_loss"):
    X, _, spe = _xaxis(hist)
    tr = np.array(hist["train"], dtype=float)           # (epoch, total, pred)
    ax.plot(X(tr[:, 0]), tr[:, 2], color=C_TRAIN, label="train")
    ax.plot(X(tr[:, 0]), tr[:, 1], color=C_TOTAL, ls="--", label="train + regularizers")
    axes = [ax]
    if hist.get("val"):
        ve, mse, hub, pad = _val_series(hist)
        ax.plot(X(ve), mse, "o-", color=C_VAL, ms=2, label="val MSE")
        if np.isfinite(hub).any():
            ax.plot(X(ve), hub, "o--", color=C_VAL, ms=1.5, alpha=0.6, label="val Huber")
        ax2 = None
        if np.isfinite(pad).any():
            ax2 = ax.twinx()
            ax2.plot(X(ve), pad, "s-", color=C_PAD, ms=1.5, lw=0.8, alpha=0.8, label="val pad error")
            ax2.set_yscale("log")
            # plain numbers (not 3x10^0); label the minor ticks too when the range is under a decade
            plain = FuncFormatter(lambda v, _: f"{v:g}")
            ax2.yaxis.set_major_formatter(plain)
            ax2.yaxis.set_minor_formatter(plain if np.nanmax(pad) / np.nanmin(pad) < 10 else NullFormatter())
            ax2.set_ylabel("pad error (mm)", color=C_PAD)
            ax2.tick_params(axis="y", colors=C_PAD)
            axes.append(ax2)
        # the star sits on the metric that chose the checkpoint, at the kept checkpoint
        series = {"val_loss": (ax, mse), "val_huber": (ax, hub), "pad_err": (ax2, pad)}
        target, y = series.get(best_metric, (ax, mse))
        kept = _kept_epoch(hist)
        if target is not None and np.isfinite(y).any():
            i = int(np.flatnonzero(ve == kept)[0]) if kept is not None and (ve == kept).any() else int(np.nanargmin(y))
            target.plot(X(ve[i]), y[i], "*", color=C_BEST, ms=8, zorder=6, label="best model")
        _kept_line(ax, X, hist)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2, linewidth=0.4)
    _epoch_axis(ax, spe)
    ax.set_ylabel("normalized loss")
    return axes


def _coeff_panel(ax, hist, stage, true):
    p = hist.get("params", [])
    if not p:
        ax.set_visible(False)
        return
    X, xlabel, _ = _xaxis(hist)
    ep = X([q["epoch"] for q in p])
    best = hist.get("best") or p[-1]            # coefficients of the kept checkpoint
    if stage == "aero":
        for k in ("k_f", "k_m", "k_rot"):
            if true.get(k):
                ax.plot(ep, np.array([q[k] for q in p]) / true[k], color=COEFF_COLORS[k],
                        label=f"{COEFF_LABELS[k]}: {best[k] / true[k]:.3f}")
        ax.axhline(1.0, color=C_BEST, ls=":", lw=0.6)
        ax.set_ylabel("learned / true")
    else:
        mu = np.array([q["mu"] for q in p])
        ax.plot(ep, mu, color=REFERENCE_COLOR, label=rf"learned $\mu$: {best['mu']:.3f}")
        # Friction-to-normal ratio of the sliding forces, at each validation pass.
        vals = [v for v in hist.get("val", []) if "slide_mu_implied" in v.get("forces", {})]
        if vals:
            ve = np.array([v["epoch"] for v in vals])
            imp = np.array([v["forces"]["slide_mu_implied"] for v in vals])
            lab = np.array([v["forces"].get("slide_mu_implied_label", np.nan) for v in vals])
            kept = _kept_epoch(hist)
            j = int(np.flatnonzero(ve == kept)[0]) if kept is not None and (ve == kept).any() else len(imp) - 1
            ax.plot(X(ve), imp, "o-", color=ACTUAL_COLOR, ms=1.5, lw=0.8,
                    label=rf"$\|F_t\|/F_n$, model: {imp[j]:.3f}")
            if np.isfinite(lab).any():
                ax.axhline(float(np.nanmean(lab)), color=ACTUAL_COLOR, ls="--", lw=0.8,
                           label=rf"$\|F_t\|/F_n$, MuJoCo: {np.nanmean(lab):.3f}")
        if true.get("mu"):
            ax.axhline(true["mu"], color=C_BEST, ls="--", lw=0.8, label=rf"true $\mu$: {true['mu']:.2f}")
        # Early in training the implied ratio can be far off (tiny normal forces);
        # cap the axis so it does not squash the mu lines.
        ax.set_ylim(0.0, min(1.0, max(0.6, ax.get_ylim()[1])))
        ax.set_ylabel(r"$\mu$")
    _kept_line(ax, X, hist)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.legend(loc="best")


def plot_loss_history(history, path, true=None, title=None):
    """history: the dict saved as <run>_history.pt. true: {"k_f", "k_m", "k_rot", "mu"}
    (see drone_run_report.true_values_from_meta). Saves <path stem>_aero and
    <path stem>_contact (.png and .svg). title is not drawn (report figures have no titles).
    Returns {"<stage>_val_loss_tail_change": x}."""
    true = true or {}
    stages = [st for st in ("aero", "contact") if history.get(st, {}).get("train")]
    best_metric = history.get("settings", {}).get("best_metric", "val_loss")
    stem = os.path.splitext(path)[0]
    set_plot_style()
    out = {}
    for st in stages:
        fig, axes = plt.subplots(2, 1, figsize=(REPORT_WIDTH, 4.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.6, 1]))
        drawn = _loss_panel(axes[0], history[st], best_metric)
        _coeff_panel(axes[1], history[st], st, true)
        figure_legend(fig, drawn, ncol=2)
        save_figure(fig, f"{stem}_{st}.png")
        out[f"{st}_val_loss_tail_change"] = val_tail_change(history[st])
    return out


if __name__ == "__main__":
    import torch
    hist_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else hist_path.replace("_history.pt", "_loss_history.png")
    plot_loss_history(torch.load(hist_path, weights_only=False), out_path)
    print(f"saved {os.path.splitext(out_path)[0]}_aero / _contact (.png, .svg)")
