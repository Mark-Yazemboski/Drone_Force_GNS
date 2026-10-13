"""Loss history for both training stages, to judge whether more training would help.

x axis: optimizer steps (bottom) and epochs (top). Training length is set in steps,
so curves from runs with different amounts of data line up. Histories from
before step-based training have no steps-per-epoch record and are plotted in epochs.

Top row (one column per stage, aero then contact), log scale:
    train pred   the normalized prediction loss on training windows, in the training
                 loss (MSE or Huber; for MSE, 1 = no better than zero learned force).
                 Inputs are noised, so it sits above val.
    train total  pred + the stage's regularizers (anchor, smoothness, friction terms...)
    val          the normalized prediction loss on validation windows, MSE (solid) and
                 Huber (dashed), whichever trains. The star marks the lowest MSE; the dotted vertical line marks the checkpoint
                 actually kept (best_metric), when that differs.
Bottom row: learned coefficients.
    aero     k_f, k_m, k_rot divided by their true values (1.0 = recovered exactly)
    contact  the learned mu parameter (it only feeds the friction regularizer), the
             friction-to-normal ratio of the PREDICTED sliding forces on validation
             (what the model actually does), the same ratio for MuJoCo's labels, and
             the true mu

Each top panel is titled with how much the val loss fell over the last 25% of the
stage's epochs. Still falling a few percent or more -> more epochs would likely
help. With lr_schedule="cosine" the learning rate goes to 0 at the end of each
stage, so the tail flattens partly by construction; look at the middle of the
curve too, and compare against a run with more epochs to be sure.

Standalone:  python plot_training_history.py models/<run>/<run>_history.pt [out.png]
"""

import os
import sys

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_PRED, C_TOTAL, C_VAL, C_BEST = "#378ADD", "#9DBFE6", "#D85A30", "#2C2C2A"
COEFF_COLORS = {"k_f": "#185FA5", "k_m": "#2ca02c", "k_rot": "#D85A30", "mu": "#D85A30"}


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
        top.set_xlabel("epoch", fontsize=8)
        top.tick_params(labelsize=7)


def _kept_epoch(hist):
    return hist.get("best", {}).get("epoch")


def _loss_panel(ax, hist, stage, pred_loss="mse"):
    X, xlabel, spe = _xaxis(hist)
    tr = np.array(hist["train"], dtype=float)           # (epoch, total, pred)
    ax.plot(X(tr[:, 0]), tr[:, 2], color=C_PRED, lw=1.4, label=f"train pred ({pred_loss})")
    ax.plot(X(tr[:, 0]), tr[:, 1], color=C_TOTAL, lw=1.0, ls="--", label="train total")
    if hist.get("val"):
        ve = np.array([v["epoch"] for v in hist["val"]])
        vl = np.array([v["loss"] for v in hist["val"]])
        ax.plot(X(ve), vl, "o-", color=C_VAL, ms=3, lw=1.4, label="val (MSE)")
        if all("loss_huber" in v for v in hist["val"]):
            ax.plot(X(ve), [v["loss_huber"] for v in hist["val"]], "o--", color=C_VAL, ms=2, lw=1.0,
                    alpha=0.6, label="val (Huber)")
        b = int(np.argmin(vl))
        where = f"step {int(X(ve[b]))}" if spe else f"ep {ve[b]}"
        ax.plot(X(ve[b]), vl[b], "*", color=C_BEST, ms=12, zorder=5, label=f"lowest val MSE ({where})")
        kept = _kept_epoch(hist)
        if kept is not None and kept != ve[b]:
            ax.axvline(X(kept), color=C_BEST, ls=":", lw=1.2, label="kept checkpoint")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel(xlabel)
    _epoch_axis(ax, spe)
    ax.set_ylabel("normalized loss (1 = zero learned force)")
    ch = val_tail_change(hist)
    tail = "" if np.isnan(ch) else f"\nval loss over the last 25% of training: {100 * ch:+.1f}%"
    ax.set_title(f"{stage} stage{tail}", fontsize=10)
    ax.legend(fontsize=8, loc="best")


def _coeff_panel(ax, hist, stage, true):
    p = hist.get("params", [])
    if not p:
        ax.set_visible(False)
        return
    X, xlabel, spe = _xaxis(hist)
    ep = X([q["epoch"] for q in p])
    if stage == "aero":
        for k in ("k_f", "k_m", "k_rot"):
            if true.get(k):
                ax.plot(ep, np.array([q[k] for q in p]) / true[k], color=COEFF_COLORS[k], lw=1.4,
                        label=f"{k}: final {p[-1][k] / true[k]:.3f}x true")
        ax.axhline(1.0, color=C_BEST, ls=":", lw=1)
        ax.set_ylabel("learned / true")
    else:
        mu = np.array([q["mu"] for q in p])
        ax.plot(ep, mu, color=COEFF_COLORS["mu"], lw=1.4, label=f"learned mu parameter: final {mu[-1]:.3f}")
        # Friction-to-normal ratio of the sliding forces, at each validation pass.
        vals = [v for v in hist.get("val", []) if "slide_mu_implied" in v.get("forces", {})]
        if vals:
            ve = X([v["epoch"] for v in vals])
            imp = np.array([v["forces"]["slide_mu_implied"] for v in vals])
            lab = np.array([v["forces"].get("slide_mu_implied_label", np.nan) for v in vals])
            ax.plot(ve, imp, "o-", color="#185FA5", ms=3, lw=1.2,
                    label=f"implied by predicted forces (|F_t|/F_n): last {imp[-1]:.3f}")
            if np.isfinite(lab).any():
                ax.axhline(float(np.nanmean(lab)), color="#185FA5", ls="--", lw=1,
                           label=f"implied by MuJoCo labels {np.nanmean(lab):.3f}")
        if true.get("mu"):
            ax.axhline(true["mu"], color=C_BEST, ls="--", lw=1, label=f"true mu {true['mu']:.2f}")
        kept = _kept_epoch(hist)
        if kept is not None:
            ax.axvline(X(kept), color=C_BEST, ls=":", lw=1.2)
        # Early in training the implied ratio can be far off (tiny normal forces);
        # cap the axis so it does not squash the mu lines.
        ax.set_ylim(0.0, min(1.0, max(0.6, ax.get_ylim()[1])))
        ax.set_ylabel("friction coefficient")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(xlabel)
    _epoch_axis(ax, spe)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "no true values passed", transform=ax.transAxes, ha="center", color="gray")


def plot_loss_history(history, path, true=None, title=None):
    """history: the dict saved as <run>_history.pt. true: {"k_f", "k_m", "k_rot", "mu"}
    (see drone_run_report.true_values_from_meta). Returns {"<stage>_val_loss_tail_change": x}."""
    true = true or {}
    stages = [st for st in ("aero", "contact") if history.get(st, {}).get("train")]
    if not stages:
        return {}
    fig, axes = plt.subplots(2, len(stages), figsize=(6.5 * len(stages), 8), squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.6, 1]))
    out = {}
    for j, st in enumerate(stages):
        _loss_panel(axes[0, j], history[st], st, history.get("settings", {}).get("pred_loss", "mse"))
        _coeff_panel(axes[1, j], history[st], st, true)
        out[f"{st}_val_loss_tail_change"] = val_tail_change(history[st])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import torch
    hist_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else hist_path.replace("_history.pt", "_loss_history.png")
    plot_loss_history(torch.load(hist_path, weights_only=False), out_path)
    print(f"saved {out_path}")
