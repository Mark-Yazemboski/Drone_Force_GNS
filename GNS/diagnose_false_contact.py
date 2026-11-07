"""Does the contact model predict force when the pad is NOT touching the wall?

One model step from every TRUE test-set state with the pad within 2 cm of the wall
(no rollout, so no drift: this is exactly the force the MPC sees for a given state).
Frames are binned by the pad's gap to the wall (closest pad sphere surface, mm,
negative = pressed in) and split by the MuJoCo label:

  no-contact frames (MuJoCo |F| < 0.02 N): predicted |F| mean / 95th percentile, share of
      frames above 0.1 N, and the mean gate of the closest pad node. Force here is phantom.
  contact frames: MuJoCo mean |F| vs predicted mean |F|.

    python diagnose_false_contact.py --data data/MPC_mj_drone_wind --range 1050 1200 \
        --models models/rev_e-2/rev_e-2_final.pt
"""

import argparse

import numpy as np
import torch

from drone_data import build_drone_dataset, build_chain_index, annotate_pad_distance, iterate_drone_chains
from train_drone_gns import load_checkpoint, _window

GAP_EDGES_MM = (-np.inf, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
NO_CONTACT_N = 0.02


@torch.no_grad()
def false_contact_table(model, cfg, data, device, max_dist=0.025):
    h = model.h
    annotate_pad_distance(data, model.pad_rest.cpu())
    idx = [(ti, s) for ti, s in build_chain_index(data, h, 1)
           if data[ti]["pad_dmin"][s + h] < max_dist]
    pad = model.contact_mask.view(-1).bool()
    gap, f_true, f_pred, gate = [], [], [], []
    for b in iterate_drone_chains(data, idx, 4096, h, 1, device, shuffle=False):
        com_h, R_h = _window(b, h)
        _, _, aux = model.step(com_h, R_h, b["omega"][:, 0], b["wind"][:, 0], b["wall_n"], b["wall_c"])
        nodes = aux["hist"][-1][:, pad]                                    # (B, P, 3) pad nodes now
        d = ((nodes - b["wall_c"].unsqueeze(1)) * b["wall_n"].unsqueeze(1)).sum(-1)
        i_min = d.argmin(1)
        gap.append(1e3 * (d.min(1).values - cfg.pad_contact_radius))
        gate.append(aux["c_w"][:, pad, 0].gather(1, i_min[:, None])[:, 0])
        f_true.append(b["F_contact"][:, 0].norm(dim=-1))
        f_pred.append(model.to_newtons(aux["f_c"]).norm(dim=-1))
    cat = lambda x: torch.cat(x).cpu().numpy()
    gap, f_true, f_pred, gate = cat(gap), cat(f_true), cat(f_pred), cat(gate)

    rows = []
    for lo, hi in zip(GAP_EDGES_MM[:-1], GAP_EDGES_MM[1:]):
        in_bin = (gap >= lo) & (gap < hi)
        off = in_bin & (f_true < NO_CONTACT_N)
        on = in_bin & (f_true >= NO_CONTACT_N)
        row = dict(lo=lo, hi=hi, n_off=int(off.sum()), n_on=int(on.sum()))
        if off.any():
            row.update(phantom_mean=f_pred[off].mean(), phantom_p95=np.percentile(f_pred[off], 95),
                       phantom_over_0p1=(f_pred[off] > 0.1).mean(), gate=gate[off].mean())
        if on.any():
            row.update(true_mean=f_true[on].mean(), pred_mean=f_pred[on].mean())
        rows.append(row)
    return rows


def print_table(rows):
    print(f"  {'gap (mm)':>12s} | {'no-contact frames':>17s} {'pred |F| mean':>13s} {'p95':>6s} {'>0.1 N':>7s} "
          f"{'gate':>5s} | {'contact frames':>14s} {'MuJoCo |F|':>10s} {'pred |F|':>8s}")
    for r in rows:
        lab = f"{r['lo']:.1f} to {r['hi']:.1f}" if np.isfinite(r["lo"]) else f"< {r['hi']:.1f}"
        off = (f"{r['n_off']:17d} {r['phantom_mean']:13.3f} {r['phantom_p95']:6.3f} "
               f"{100 * r['phantom_over_0p1']:6.1f}% {r['gate']:5.2f}") if r["n_off"] else f"{0:17d}" + " " * 35
        on = f"{r['n_on']:14d} {r['true_mean']:10.3f} {r['pred_mean']:8.3f}" if r["n_on"] else f"{0:14d}"
        print(f"  {lab:>12s} | {off} | {on}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--range", type=int, nargs=2, required=True, help="trajectory ids [start, stop)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    for path in a.models:
        model, cfg, s, _ = load_checkpoint(path, a.device)
        data, _ = build_drone_dataset(range(*a.range), a.data, cfg)
        print("\n" + "=" * 100 + f"\n{path}   (gate d0 {1e3 * s.contact_d0:.1f} mm, tau {1e3 * s.contact_tau:.1f} mm, "
              f"pad sphere radius {1e3 * cfg.pad_contact_radius:.1f} mm)")
        print_table(false_contact_table(model, cfg, data, a.device))


if __name__ == "__main__":
    main()
