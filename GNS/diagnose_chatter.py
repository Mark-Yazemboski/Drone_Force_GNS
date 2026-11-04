"""Where does the contact chatter come from, and how large is it? Run on one or more models:

    python diagnose_chatter.py --data data/MPC_mj_drone_wind --range 1050 1200 \
        --models models/New_Data_For_MPC/New_Data_For_MPC_final.pt models/fric_cap/fric_cap_final.pt

1. STEP STABILITY at sustained contact (MuJoCo force 0.2-10 N), from true states: one
   model step linearized in all 6 DOF of each history pose (see step_stability). A free
   body has eigenvalues of modulus 1 (printed for reference: the same frames with the
   contact net off); an eigenvalue near -1 means the step overcorrects and
   flips sign every step (period-2 chatter), beyond -1 it grows. Also reports whether that
   most-negative mode is rotation or translation. Split into stuck and sliding frames.
2. OPEN-LOOP CHATTER from mid-contact starts (recorded rotor speeds, no controller):
   RMS step-to-step change and RMS second difference of the normal and friction force,
   model vs MuJoCo, during sustained contact over the first 0.5 s.
3. REVERSAL PENALTY SIZE: the w_contact_reversal term and the prediction loss on
   stage-2 windows of this data, and the weight at which the penalty is 10% / 30% of it.
Nothing is trained. With --master_csv the numbers are also written into each model's
row of the master tracker (run name = the model's folder name), as metrics.chatter_*
columns. evaluate_drone_model.py calls chatter_metrics() itself, so new runs get these
columns automatically; this flag is for runs that were evaluated before that.

    python diagnose_chatter.py --data data/MPC_mj_drone_wind --range 1050 1200 \
        --models models/*/*_final.pt --master_csv models/drone_master_tracker.csv
"""

import argparse
import csv
import os

import numpy as np
import torch

from drone_data import build_drone_dataset, build_chain_index, filter_chain_index, annotate_pad_distance, \
    iterate_drone_chains
from drone_open_loop import start_frames, batched_rollout
from train_drone_gns import load_checkpoint, _make_phys, unroll_loss, contact_reversal, _window


def load_data(folder, rng, cfg):
    data, _ = build_drone_dataset(range(*rng), folder, cfg)
    return data


# ---------------------------------------------------------------- 1. step stability
# Linearizes one model step about a true state in all 6 DOF of each of the h+1 history
# poses (state ordered newest first, pose = [position (3), rotation (3)], rotations
# perturbed as R <- exp(d) R in the world frame, as in the MPC's linearize):
#     dz[k+1] = A dz[k],  A = [[A_new], [I 0 0], [0 I 0]]   (18 x 18 for h = 2)
# A free body has eigenvalues of modulus 1 (and 0). For the eigenvalue with the most
# negative real part, rot_share says whether that mode lives in rotation or translation:
# the rotation part of its eigenvector (newest pose) times the pad lever arm, as a share
# of rotation + translation, i.e. which one moves the pad.
# use_contact=False gives the reference: the same frames with the contact net off, i.e.
# thrust + aero only (about 1.00). Runs in float64: eigenvalue 1 is a repeated root, so
# float32 rounding in A would show up magnified in the eigenvalues.
@torch.no_grad()
def step_stability(model, data, device, eps=1e-6, stride=2, max_frames=6000, use_contact=True):
    import copy
    from force_gns import so3_exp, so3_log
    model = copy.deepcopy(model).double()
    h = model.h
    n_hist = h + 1
    lever = float(model.contact_rest.norm(dim=-1).max())
    eps_r = eps / lever                                                # same displacement at the pad
    idx = build_chain_index(data, h, 1, stride=stride)
    rows = []
    for b in iterate_drone_chains(data, idx, 1024, h, 1, device, shuffle=False):
        F = b["F_contact"][:, 0]
        n = b["wall_n"]
        Fn = (F * n).sum(-1)
        keep = (F.norm(dim=-1) > 0.2) & (F.norm(dim=-1) < 10.0) & (Fn > 0.2)
        if not keep.any():
            continue
        sel = lambda x: x[keep].double()
        com_h = [sel(c) for c in _window(b, h)[0]]
        R_h = [sel(R) for R in _window(b, h)[1]]
        om, wind, wn, wc = sel(b["omega"][:, 0]), sel(b["wind"][:, 0]), sel(n), sel(b["wall_c"])
        B = len(om)

        # pad slip speed over the last step (stuck vs sliding)
        pad = model.pad_rest
        p1 = com_h[-1].unsqueeze(1) + torch.einsum('bij,nj->bni', R_h[-1], pad)
        p0 = com_h[-2].unsqueeze(1) + torch.einsum('bij,nj->bni', R_h[-2], pad)
        v = (p1 - p0).mean(1)
        stuck = (v - (v * wn).sum(-1, keepdim=True) * wn).norm(dim=-1) < 3e-4      # slip gate, 0.03 m/s

        com_nom, R_nom, _ = model.step(com_h, R_h, om, wind, wn, wc, use_contact=use_contact)
        A_new = torch.zeros(B, 6, 6 * n_hist, device=om.device, dtype=om.dtype)
        for col in range(6 * n_hist):
            age, dof = divmod(col, 6)                                  # age 0 = newest pose
            j = n_hist - 1 - age
            out = []
            for sgn in (1.0, -1.0):
                ch, Rh = list(com_h), list(R_h)
                e = torch.zeros(B, 3, device=om.device, dtype=om.dtype)
                if dof < 3:
                    e[:, dof] = sgn * eps
                    ch[j] = ch[j] + e
                else:
                    e[:, dof - 3] = sgn * eps_r
                    Rh[j] = so3_exp(e) @ Rh[j]
                com_n, R_n, _ = model.step(ch, Rh, om, wind, wn, wc, use_contact=use_contact)
                out.append(torch.cat([com_n - com_nom, so3_log(R_n @ R_nom.transpose(-1, -2))], -1))
            A_new[:, :, col] = (out[0] - out[1]) / (2 * (eps if dof < 3 else eps_r))
        A = torch.zeros(B, 6 * n_hist, 6 * n_hist, device=om.device, dtype=om.dtype)
        A[:, :6] = A_new
        A[:, 6:, :-6] = torch.eye(6 * (n_hist - 1), device=om.device, dtype=om.dtype)
        ev, vec = torch.linalg.eig(A.cpu())
        i_neg = ev.real.argmin(-1)
        v_neg = vec[torch.arange(B), :, i_neg]                         # eigenvector of that eigenvalue
        t_part = v_neg[:, 0:3].abs().norm(dim=-1)
        r_part = v_neg[:, 3:6].abs().norm(dim=-1) * lever
        rows.append(dict(stuck=stuck.cpu().numpy(), radius=ev.abs().max(-1).values.numpy(),
                         most_negative=ev.real.min(-1).values.numpy(),
                         rot_share=(r_part / (t_part + r_part + 1e-30)).numpy()))
        if sum(len(r["radius"]) for r in rows) >= max_frames:
            break

    out = {}
    cat = lambda key, pick: np.concatenate([r[key][r["stuck"] == pick] for r in rows])
    for regime, pick in (("stuck", True), ("sliding", False)):
        rad, neg, rot = cat("radius", pick), cat("most_negative", pick), cat("rot_share", pick)
        if rad.size:
            p2 = neg < -0.9
            out[regime] = dict(n=rad.size, radius_median=np.median(rad), frac_unstable=np.mean(rad > 1.02),
                               frac_period2=np.mean(p2), most_negative_median=np.median(neg),
                               rot_share_period2=np.median(rot[p2]) if p2.any() else float("nan"),
                               rot_share_all=np.median(rot))
    return out


# ---------------------------------------------------------------- 2. open-loop chatter
# Frames inside runs of at least min_len consecutive True values (excludes the one or
# two frames a tap or a brief touch spends between 0.05 and 10 N).
def long_runs(mask, min_len=5):
    out = np.zeros_like(mask)
    start = None
    for i, v in enumerate(np.r_[mask, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out[start:i] = True
            start = None
    return out


def open_loop_chatter(model, cfg, data, device, horizon=50):
    runs = []
    for d in data:
        s = start_frames(d, model.h, cfg.dt).get("mid_contact")
        if s is not None:
            runs.append(dict(d=d, start=s))
    res = batched_rollout(model, cfg, runs, horizon, device=device)
    acc = {k: [] for k in ("model_n", "mujoco_n", "model_t", "mujoco_t")}
    for r, p in zip(runs, res):
        n_fr = p["n_frames"] - 1
        true = r["d"]["F_contact"][r["start"]:r["start"] + n_fr].numpy()
        pred = p["F_contact"][:n_fr]
        nrm = r["d"]["wall_n"].numpy()
        mag = np.linalg.norm(true, axis=1)
        sustained = long_runs((mag > 0.05) & (mag < 10.0))
        for tag, F in (("model", pred), ("mujoco", true)):
            Fn = F @ nrm
            Ft = F - Fn[:, None] * nrm
            for comp, x in (("n", Fn[:, None]), ("t", Ft)):
                d1 = np.diff(x, axis=0)
                d2 = np.diff(x, 2, axis=0)
                m1 = sustained[1:] & sustained[:-1]
                m2 = sustained[2:] & sustained[1:-1] & sustained[:-2]
                acc[f"{tag}_{comp}"].append((np.linalg.norm(d1, axis=1)[m1], np.linalg.norm(d2, axis=1)[m2]))
    rms = lambda parts, i: float(np.sqrt(np.mean(np.concatenate([p[i] for p in parts]) ** 2)))
    return {k: (rms(v, 0), rms(v, 1)) for k, v in acc.items()}, len(runs)


# ---------------------------------------------------------------- 3. penalty size
@torch.no_grad()
def penalty_size(model, s, cfg, data, device, max_batches=20):
    phys = _make_phys(cfg, s).to(device)
    annotate_pad_distance(data, model.pad_rest.cpu())
    K = s.multistep
    idx = filter_chain_index(data, build_chain_index(data, s.h, K, stride=3), s.h, K, max_dist=s.contact_max_pad_dist)
    pred = rev = 0.0
    nb = 0
    torch.manual_seed(0)                                   # same batches for every model
    for b in iterate_drone_chains(data, idx, 1024, s.h, K, device, shuffle=True):
        _, p, _ = unroll_loss(model, phys, b, s, "contact")
        com_h, R_h = _window(b, s.h)
        F_series, T_series = [], []
        for k in range(K):
            com_n, R_n, aux = model.step(com_h, R_h, b["omega"][:, k], b["wind"][:, k], b["wall_n"], b["wall_c"])
            F_series.append(aux["f_c"])
            T_series.append(aux["tau_c"])
            com_h, R_h = com_h[1:] + [com_n], R_h[1:] + [R_n]
        pred += float(p)
        rev += float(contact_reversal(F_series, T_series, model))
        nb += 1
        if nb >= max_batches:
            break
    return pred / nb, rev / nb


# ---------------------------------------------------------------- all three, flat
# One flat dict of every number above (keys chatter_*), for the run report / master CSV.
def chatter_metrics(model_path, data, device):
    model, cfg, s, _ = load_checkpoint(model_path, device)
    out = {}
    ref = step_stability(model, data, device, use_contact=False)
    for regime, v in step_stability(model, data, device).items():
        t = f"chatter_{regime}_"
        out[t + "n"] = v["n"]
        out[t + "radius"] = v["radius_median"]
        out[t + "radius_contact_off"] = ref[regime]["radius_median"]
        out[t + "unstable_frac"] = v["frac_unstable"]
        out[t + "period2_frac"] = v["frac_period2"]
        out[t + "most_negative_eig"] = v["most_negative_median"]
        out[t + "rot_share"] = v["rot_share_all"]

    ch, n_runs = open_loop_chatter(model, cfg, data, device)
    out["chatter_open_loop_n"] = n_runs
    for comp, label in (("n", "normal"), ("t", "friction")):
        for src in ("model", "mujoco"):
            d1, d2 = ch[f"{src}_{comp}"]
            out[f"chatter_open_loop_{label}_{src}_dF_N"] = d1
            out[f"chatter_open_loop_{label}_{src}_d2F_N"] = d2

    pred, rev = penalty_size(model, s, cfg, data, device)
    out["chatter_pred_loss"] = pred
    out["chatter_reversal_raw"] = rev
    return out


def print_chatter(m):
    print("1. step stability at sustained contact, full 6-DOF linearization of one step from true states")
    for regime in ("stuck", "sliding"):
        t = f"chatter_{regime}_"
        if t + "n" not in m:
            continue
        print(f"   {regime:7s} ({m[t + 'n']:5d} frames)  spectral radius median {m[t + 'radius']:.3f} "
              f"(contact net off: {m[t + 'radius_contact_off']:.3f}) | "
              f"unstable (>1.02) {100 * m[t + 'unstable_frac']:4.1f}% | "
              f"period-2 (eig < -0.9) {100 * m[t + 'period2_frac']:4.1f}% | "
              f"most negative eig median {m[t + 'most_negative_eig']:+.2f} | "
              f"in rotation {100 * m[t + 'rot_share']:.0f}%")

    print("2. open-loop chatter, mid-contact starts, first 0.5 s, sustained contact "
          "(RMS change per step / RMS second difference, N)")
    for label in ("normal", "friction"):
        t = f"chatter_open_loop_{label}_"
        print(f"   {label:8s} model {m[t + 'model_dF_N']:.3f} / {m[t + 'model_d2F_N']:.3f}   "
              f"MuJoCo {m[t + 'mujoco_dF_N']:.3f} / {m[t + 'mujoco_d2F_N']:.3f}   "
              f"({m['chatter_open_loop_n']} starts)")

    pred, rev = m["chatter_pred_loss"], m["chatter_reversal_raw"]
    print(f"3. stage-2 windows: prediction loss {pred:.3e}, reversal penalty (raw) {rev:.3e}")
    if rev > 0:
        print(f"   w_contact_reversal for a 10% / 30% share of the prediction loss: "
              f"{0.1 * pred / rev:.3g} / {0.3 * pred / rev:.3g}")


# Writes the metrics into run_name's existing row of the master CSV (adds columns as needed).
def add_to_master(master_csv, run_name, metrics):
    with open(master_csv, newline="") as f:
        rdr = csv.DictReader(f)
        fieldnames = list(rdr.fieldnames)
        rows = list(rdr)
    row = next((r for r in rows if r["run_name"] == run_name), None)
    if row is None:
        print(f"   [master] no row named {run_name}, skipped")
        return
    for k, v in metrics.items():
        col = f"metrics.{k}"
        if col not in fieldnames:
            fieldnames.append(col)
        row[col] = repr(float(v))
    with open(master_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"   [master] wrote {len(metrics)} chatter columns to row {run_name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--range", type=int, nargs=2, required=True, help="trajectory ids [start, stop)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--master_csv", default=None, help="also write the numbers into this tracker")
    a = ap.parse_args()

    for path in a.models:
        _, cfg, s, _ = load_checkpoint(path, "cpu")
        data = load_data(a.data, a.range, cfg)
        print("\n" + "=" * 78 + f"\n{path}   (w_contact_reversal={getattr(s, 'w_contact_reversal', 0.0)}, "
              f"friction_cap={s.friction_cap}, contact_noise_frac={s.contact_noise_frac})")
        m = chatter_metrics(path, data, a.device)
        print_chatter(m)
        if a.master_csv:
            add_to_master(a.master_csv, os.path.basename(os.path.dirname(os.path.abspath(path))), m)


if __name__ == "__main__":
    main()
