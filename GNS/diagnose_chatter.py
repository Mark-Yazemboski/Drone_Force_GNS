"""Where does the contact chatter come from, and how large is it? Run on one or more models:

    python diagnose_chatter.py --data data/MPC_mj_drone_wind --range 1050 1200 \
        --models models/New_Data_For_MPC/New_Data_For_MPC_final.pt models/fric_cap/fric_cap_final.pt

1. STEP STABILITY at sustained contact (MuJoCo force 0.2-10 N), from true states.
   The model's next COM position along a direction u depends on the last three:
       x[k+1] = a0 x[k] + a1 x[k-1] + a2 x[k-2]   (linearized, a = finite differences of model.step)
   A free body has a = (2, -1, 0), eigenvalues (1, 1, 0). An eigenvalue near -1 means the
   step overcorrects and flips sign every step (period-2 chatter); beyond -1 it grows.
   Reported along the wall normal and along the friction direction, split into stuck
   (pad slip below slip_v0) and sliding frames.
2. OPEN-LOOP CHATTER from mid-contact starts (recorded rotor speeds, no controller):
   RMS step-to-step change and RMS second difference of the normal and friction force,
   model vs MuJoCo, during sustained contact over the first 0.5 s.
3. REVERSAL PENALTY SIZE: the w_contact_reversal term and the prediction loss on
   stage-2 windows of this data, and the weight at which the penalty is 10% / 30% of it.
Nothing is trained or saved.
"""

import argparse

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
@torch.no_grad()
def step_stability(model, data, device, eps=2e-5, stride=2, max_frames=20000):
    h = model.h
    idx = build_chain_index(data, h, 1, stride=stride)
    rows = []
    for b in iterate_drone_chains(data, idx, 2048, h, 1, device, shuffle=False):
        F = b["F_contact"][:, 0]
        n = b["wall_n"]
        Fn = (F * n).sum(-1)
        Ft = F - Fn.unsqueeze(-1) * n
        keep = (F.norm(dim=-1) > 0.2) & (F.norm(dim=-1) < 10.0) & (Fn > 0.2) & (Ft.norm(dim=-1) > 0.02)
        if not keep.any():
            continue
        sel = lambda x: x[keep]
        com_h = [sel(c) for c in _window(b, h)[0]]
        R_h = [sel(R) for R in _window(b, h)[1]]
        om, wind, wn, wc = sel(b["omega"][:, 0]), sel(b["wind"][:, 0]), sel(n), sel(b["wall_c"])
        t_hat = Ft[keep] / Ft[keep].norm(dim=-1, keepdim=True)

        # pad slip speed over the last step (stuck vs sliding)
        pad = model.pad_rest
        p1 = com_h[-1].unsqueeze(1) + torch.einsum('bij,nj->bni', R_h[-1], pad)
        p0 = com_h[-2].unsqueeze(1) + torch.einsum('bij,nj->bni', R_h[-2], pad)
        v = (p1 - p0).mean(1)
        v_t = (v - (v * wn).sum(-1, keepdim=True) * wn).norm(dim=-1)
        stuck = v_t < 3e-4                                            # slip gate, 0.03 m/s

        for name, u in (("normal", wn), ("friction", t_hat)):
            a = []
            for j in (2, 1, 0):                                       # x[k], x[k-1], x[k-2]
                out = []
                for sgn in (1.0, -1.0):
                    ch = list(com_h)
                    ch[j] = ch[j] + sgn * eps * u
                    com_n, _, _ = model.step(ch, R_h, om, wind, wn, wc)
                    out.append((com_n * u).sum(-1))
                a.append((out[0] - out[1]) / (2 * eps))
            A = torch.zeros(len(u), 3, 3, device=u.device)
            A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = a
            A[:, 1, 0] = 1.0
            A[:, 2, 1] = 1.0
            ev = torch.linalg.eigvals(A.double().cpu())
            rows.append(dict(dir=name, stuck=stuck.cpu().numpy(),
                             radius=ev.abs().max(-1).values.numpy(),
                             most_negative=ev.real.min(-1).values.numpy()))
        if sum(len(r["radius"]) for r in rows) >= 2 * max_frames:
            break

    out = {}
    for name in ("normal", "friction"):
        for regime, pick in (("stuck", True), ("sliding", False)):
            rad = np.concatenate([r["radius"][r["stuck"] == pick] for r in rows if r["dir"] == name])
            neg = np.concatenate([r["most_negative"][r["stuck"] == pick] for r in rows if r["dir"] == name])
            if rad.size:
                out[(name, regime)] = dict(n=rad.size, radius_median=np.median(rad),
                                           frac_unstable=np.mean(rad > 1.02),
                                           frac_period2=np.mean(neg < -0.9),
                                           most_negative_median=np.median(neg))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--range", type=int, nargs=2, required=True, help="trajectory ids [start, stop)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    for path in a.models:
        model, cfg, s, _ = load_checkpoint(path, a.device)
        data = load_data(a.data, a.range, cfg)
        print("\n" + "=" * 78 + f"\n{path}   (friction_cap={s.friction_cap}, contact_noise_frac={s.contact_noise_frac})")

        print("1. step stability at sustained contact (eigenvalues of the linearized step)")
        for (name, regime), v in step_stability(model, data, a.device).items():
            print(f"   {name:8s} {regime:7s} ({v['n']:5d} frames)  spectral radius median {v['radius_median']:.3f} | "
                  f"unstable (>1.02) {100 * v['frac_unstable']:4.1f}% | period-2 (eig < -0.9) {100 * v['frac_period2']:4.1f}% | "
                  f"most negative eig median {v['most_negative_median']:+.2f}")

        print("2. open-loop chatter, mid-contact starts, first 0.5 s, sustained contact "
              "(RMS change per step / RMS second difference, N)")
        ch, n_runs = open_loop_chatter(model, cfg, data, a.device)
        for comp, label in (("n", "normal"), ("t", "friction")):
            m, j = ch[f"model_{comp}"], ch[f"mujoco_{comp}"]
            print(f"   {label:8s} model {m[0]:.3f} / {m[1]:.3f}   MuJoCo {j[0]:.3f} / {j[1]:.3f}   ({n_runs} starts)")

        pred, rev = penalty_size(model, s, cfg, data, a.device)
        print(f"3. stage-2 windows: prediction loss {pred:.3e}, reversal penalty (raw) {rev:.3e}")
        if rev > 0:
            print(f"   w_contact_reversal for a 10% / 30% share of the prediction loss: "
                  f"{0.1 * pred / rev:.3g} / {0.3 * pred / rev:.3g}")


if __name__ == "__main__":
    main()
