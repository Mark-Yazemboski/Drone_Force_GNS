"""Open-loop evaluation: start the model from a recorded test trajectory's true
state and feed it that trajectory's RECORDED rotor speeds every step. No
controller. MuJoCo's result is the recording itself.

A quadrotor is open-loop unstable, so this diverges for ANY model, even a
perfect one (MuJoCo integrates in 0.5 ms Euler substeps, the model in 10 ms
Verlet steps): what is measured is how long the prediction stays close.

Start points per test trajectory (trajectories with no contact get only the first):
    in_air        1 s before the first contact (free flight)
    pre_contact   0.2 s before contact onset
    mid_contact   the middle of the longest contact
    pre_release   0.2 s before that contact ends
Each start is rolled out for `horizon_s` seconds, twice: with the contact network
on, and off (known physics + aero only), and stopped early once the pad is more
than `max_pad_err_m` from MuJoCo's or the drone tilts past `max_tilt_deg`.

Reported per start kind, contact on / off:
    time until the pad error passes 1 cm and 5 cm (median over runs; runs that never
    pass it count as the full horizon, so the median is a lower bound), pad error at
    0.5 / 1 / 2 s (median), and the fraction of runs cut off by the crash limits.
Figures: pad error vs time on a log axis (one panel per start kind; exponential
growth is a straight line), and per start kind a time-series PNG and a GIF in the
closed-loop style (the PNG also shows the contact-network-off run, dashed).

All starts are rolled out together as one batch.
"""

import os

import numpy as np
import torch

START_KINDS = ("in_air", "pre_contact", "mid_contact", "pre_release")


def _orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(R.shape[:-1])
    D[..., -1] = d
    return (U * D[..., None, :]) @ Vt


def start_frames(d, h, dt, touch_N=0.2):
    """{kind: frame} for one trajectory (see module docstring)."""
    T = d["T"]
    lo, hi = h, T - 2
    clip = lambda f: int(np.clip(f, lo, hi))
    if "F_contact" not in d:
        return {"in_air": clip(T // 3)}
    Fn = (d["F_contact"] @ d["wall_n"]).numpy()
    idx = np.flatnonzero(Fn > touch_N)
    if not idx.size:
        return {"in_air": clip(T // 3)}
    # longest contiguous contact (gaps of a few frames from chatter are bridged)
    breaks = np.flatnonzero(np.diff(idx) > 5)
    segs = np.split(idx, breaks + 1)
    seg = max(segs, key=len)
    a, b = int(seg[0]), int(seg[-1])
    n1, n02 = round(1.0 / dt), round(0.2 / dt)
    out = {"in_air": clip(a - n1), "pre_contact": clip(a - n02),
           "mid_contact": clip((a + b) // 2), "pre_release": clip(b - n02)}
    return out


def pad_center(com, R, cfg):
    tip = np.asarray(cfg.rod_tip) - np.asarray(cfg.com_offset)
    return com + np.einsum('...ij,j->...i', R, tip)


@torch.no_grad()
def batched_rollout(model, cfg, runs, horizon, use_contact=True, device="cpu",
                    max_pad_err_m=0.30, max_tilt_deg=60.0):
    """runs: list of dict(d=trajectory, start=frame). Returns one dict per run:
    com (n,3), R (n,3,3), F_contact/F_aero (n,3), F_pad (n,N,3), n_frames, crashed.
    Index i is frame start + i; index 0 is the true start state."""
    model.eval()
    h, B = model.h, len(runs)
    H = horizon
    n_steps = np.array([min(H, r["d"]["T"] - 1 - r["start"]) for r in runs])
    com_hist = np.stack([r["d"]["com"][r["start"] - h:r["start"] + 1].numpy() for r in runs])          # (B,h+1,3)
    R_hist = np.stack([r["d"]["R"][r["start"] - h:r["start"] + 1].double().numpy() for r in runs])     # (B,h+1,3,3)
    wall_n = np.stack([r["d"]["wall_n"].numpy() for r in runs])
    wall_c = np.stack([r["d"]["wall_c"].numpy() for r in runs])
    Nc = model.contact_rest.shape[0]
    com = np.full((B, H + 1, 3), np.nan); com[:, 0] = com_hist[:, -1]
    R = np.full((B, H + 1, 3, 3), np.nan); R[:, 0] = R_hist[:, -1]
    F_c = np.zeros((B, H + 1, 3)); F_a = np.zeros((B, H + 1, 3)); F_pad = np.zeros((B, H + 1, Nc, 3))
    alive = n_steps > 0
    n_frames = np.ones(B, int)
    crashed = np.zeros(B, bool)
    f32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    cos_max = np.cos(np.radians(max_tilt_deg))

    for k in range(H):
        if not alive.any():
            break
        fr = np.array([min(r["start"] + k, r["d"]["T"] - 2) for r in runs])      # finished runs: clamped, ignored
        omega = np.stack([r["d"]["omega"][f].numpy() for r, f in zip(runs, fr)])
        wind = np.stack([r["d"]["wind"][f].numpy() for r, f in zip(runs, fr)])
        anchor = com_hist[:, -1]
        com_l = [f32(com_hist[:, j] - anchor) for j in range(h + 1)]
        R_l = [f32(R_hist[:, j]) for j in range(h + 1)]
        com_n, R_n, aux = model.step(com_l, R_l, f32(omega), f32(wind), f32(wall_n), f32(wall_c - anchor),
                                     use_contact=use_contact)
        c_next = com_n.double().cpu().numpy() + anchor
        R_next = _orthonormalize(R_n.double().cpu().numpy())
        # forces acting during step k (from frame start + k)
        a = alive
        F_c[a, k] = model.to_newtons(aux["f_c"]).cpu().numpy()[a]
        F_a[a, k] = model.to_newtons(aux["aero"]["F"]).cpu().numpy()[a]
        if use_contact:
            F_pad[a, k] = model.to_newtons(aux["phi_c"]).cpu().numpy()[a]
        com[a, k + 1], R[a, k + 1] = c_next[a], R_next[a]
        n_frames[a] = k + 2
        # crash limits, against MuJoCo at the same frame
        tr = [(r["d"]["com"][f + 1].numpy(), r["d"]["R"][f + 1].double().numpy()) for r, f in zip(runs, fr)]
        err = np.linalg.norm(pad_center(c_next, R_next, cfg)
                             - np.stack([pad_center(c, Rt, cfg) for c, Rt in tr]), axis=1)
        tilt_bad = R_next[:, 2, 2] < cos_max
        crash_now = a & ((err > max_pad_err_m) | tilt_bad | ~np.isfinite(err))
        crashed |= crash_now
        # advance the history of the runs still going; freeze the others
        go = a & ~crash_now & (k + 1 < n_steps)
        com_hist[go] = np.concatenate([com_hist[go, 1:], c_next[go, None]], 1)
        R_hist[go] = np.concatenate([R_hist[go, 1:], R_next[go, None]], 1)
        alive = go

    out = []
    for i in range(B):
        n = n_frames[i]
        F_c[i, n - 1], F_a[i, n - 1], F_pad[i, n - 1] = F_c[i, max(n - 2, 0)], F_a[i, max(n - 2, 0)], F_pad[i, max(n - 2, 0)]
        out.append(dict(com=com[i, :n], R=R[i, :n], F_contact=F_c[i, :n], F_aero=F_a[i, :n], F_pad=F_pad[i, :n],
                        n_frames=int(n), crashed=bool(crashed[i])))
    return out


def _truth_slice(d, start, n):
    sl = slice(start, start + n)
    out = dict(com=d["com"][sl].numpy(), R=d["R"][sl].double().numpy(), n_frames=n,
               F_contact=d["F_contact"][sl].numpy() if "F_contact" in d else np.zeros((n, 3)),
               F_aero=d["F_aero"][sl].numpy() if "F_aero" in d else np.zeros((n, 3)))
    out["phase"] = d["phase"][sl].numpy() if "phase" in d else np.zeros(n, int)
    return out


def _pad_error(pred, truth, cfg):
    n = pred["n_frames"]
    return np.linalg.norm(pad_center(pred["com"], pred["R"], cfg) - pad_center(truth["com"][:n], truth["R"][:n], cfg), axis=1)


def _time_to(err, thr, dt, horizon_s):
    i = np.flatnonzero(err > thr)
    return (float(i[0] * dt), False) if i.size else (float(horizon_s), True)     # (time, never reached)


def evaluate_open_loop(model, cfg, data, out_folder, horizon_s=2.0, n_gifs=4, gif_stride=3, device="cpu",
                       max_pad_err_m=0.30, max_tilt_deg=60.0, verbose=True, mu_true=None):
    import matplotlib
    if not os.environ.get("DISPLAY") and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize_drone_model import animate_closed_loop

    dt = float(cfg.dt)
    H = int(round(horizon_s / dt))
    runs = []
    for d in data:
        for kind, s in start_frames(d, model.h, dt).items():
            runs.append(dict(d=d, start=s, kind=kind))
    if not runs:
        return {}
    if verbose:
        counts = {k: sum(r["kind"] == k for r in runs) for k in START_KINDS}
        print(f"\nopen loop: {len(runs)} starts from {len(data)} test trajectories "
              f"({', '.join(f'{k} {v}' for k, v in counts.items())}), {horizon_s:g} s, recorded rotor speeds, no controller")
    res = {flag: batched_rollout(model, cfg, runs, H, use_contact=flag, device=device,
                                 max_pad_err_m=max_pad_err_m, max_tilt_deg=max_tilt_deg)
           for flag in (True, False)}

    metrics, curves = {}, {}
    for kind in START_KINDS:
        sel = [i for i, r in enumerate(runs) if r["kind"] == kind]
        if not sel:
            continue
        for flag, tag in ((True, "contact_on"), (False, "contact_off")):
            t1, t5, e = [], [], {0.5: [], 1.0: [], 2.0: []}
            crash = []
            cur = []
            for i in sel:
                tr = _truth_slice(runs[i]["d"], runs[i]["start"], res[flag][i]["n_frames"])
                err = _pad_error(res[flag][i], tr, cfg)
                t1.append(_time_to(err, 0.01, dt, horizon_s)[0])
                t5.append(_time_to(err, 0.05, dt, horizon_s)[0])
                for ts in e:
                    j = int(round(ts / dt))
                    e[ts].append(err[j] if j < len(err) else (max_pad_err_m if res[flag][i]["crashed"] else np.nan))
                crash.append(res[flag][i]["crashed"])
                cur.append(err)
            p = f"open_loop_{kind}_{tag}"
            metrics[f"{p}_t1cm_s"] = float(np.median(t1))
            metrics[f"{p}_t5cm_s"] = float(np.median(t5))
            for ts, v in e.items():
                metrics[f"{p}_pad_err_{ts:g}s_mm"] = float(1e3 * np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
            metrics[f"{p}_crash_frac"] = float(np.mean(crash))
            curves[(kind, flag)] = cur
        metrics[f"open_loop_{kind}_n"] = len(sel)
        if verbose:
            on, off = f"open_loop_{kind}_contact_on", f"open_loop_{kind}_contact_off"
            print(f"  {kind:12s} ({len(sel):3d} starts)  time to 1 cm / 5 cm: contact on "
                  f"{metrics[on + '_t1cm_s']:.2f} / {metrics[on + '_t5cm_s']:.2f} s, off "
                  f"{metrics[off + '_t1cm_s']:.2f} / {metrics[off + '_t5cm_s']:.2f} s | pad error at 1 s: on "
                  f"{metrics[on + '_pad_err_1s_mm']:.1f} mm, off {metrics[off + '_pad_err_1s_mm']:.1f} mm | "
                  f"crashed: on {100 * metrics[on + '_crash_frac']:.0f}%, off {100 * metrics[off + '_crash_frac']:.0f}%")

    # ---- error-growth figure ----
    os.makedirs(out_folder, exist_ok=True)
    kinds = [k for k in START_KINDS if (k, True) in curves]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.8), squeeze=False, sharey=True)
    for ax, kind in zip(axes[0], kinds):
        for flag, col, lab in ((False, "#888780", "contact network off"), (True, "#185FA5", "model")):
            cur = curves[(kind, flag)]
            for err in cur:
                ax.plot(np.arange(len(err)) * dt, 1e3 * np.maximum(err, 1e-5), color=col, lw=0.5, alpha=0.25)
            M = np.full((len(cur), H + 1), np.nan)
            for j, err in enumerate(cur):
                M[j, :len(err)] = err
                if len(err) < H + 1 and len(err) > 0:          # crashed / cut: hold at the cutoff for the median
                    M[j, len(err):] = max(err[-1], max_pad_err_m)
            med = np.nanmedian(M, axis=0)
            ax.plot(np.arange(H + 1) * dt, 1e3 * np.maximum(med, 1e-5), color=col, lw=2.0, label=f"{lab} (median)")
        for thr in (10, 50):
            ax.axhline(thr, color="k", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e3 * max_pad_err_m * 1.2)
        ax.set_title(f"start: {kind.replace('_', ' ')} ({len(curves[(kind, True)])})", fontsize=10)
        ax.set_xlabel("time since start (s)")
        ax.grid(True, which="both", alpha=0.2)
    axes[0, 0].set_ylabel("pad position error vs MuJoCo (mm)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Open loop: recorded rotor speeds, no controller (dotted: 1 cm, 5 cm)", fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(out_folder, "open_loop_error.png")
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    if verbose:
        print(f"  saved {os.path.basename(fig_path)}")

    # ---- per start kind: a time-series PNG (like the closed-loop ones) and a GIF, ----
    # ---- for the first test trajectory that has that start (same ones every run) ----
    from visualize_drone_model import plot_closed_loop
    made = 0
    for kind in kinds:
        i = next(i for i, r in enumerate(runs) if r["kind"] == kind)
        r, pred, off = runs[i], res[True][i], res[False][i]
        d = r["d"]
        sc = dict(n=d["wall_n"].double().numpy(), wall_point=d["wall_c"].double().numpy(),
                  kind=d.get("scenario", "?"))
        tid = d.get("traj_id", i)
        stem = os.path.join(out_folder, f"open_loop_{kind}_traj{tid}")
        tr_full = _truth_slice(d, r["start"], min(H + 1, d["T"] - r["start"]))   # MuJoCo over the whole horizon
        err = _pad_error(pred, tr_full, cfg)
        err_off = _pad_error(off, tr_full, cfg)
        t1, t5 = _time_to(err, 0.01, dt, horizon_s)[0], _time_to(err, 0.05, dt, horizon_s)[0]
        o1, o5 = _time_to(err_off, 0.01, dt, horizon_s)[0], _time_to(err_off, 0.05, dt, horizon_s)[0]
        at = lambda e, ts: f"{1e3 * e[int(round(ts / dt))]:.1f}" if int(round(ts / dt)) < len(e) else "cut"
        title = (f"open loop, start {kind.replace('_', ' ')} | trajectory {tid} ({sc['kind']}), frame {r['start']}\n"
                 f"time to 1 cm / 5 cm: model {t1:.2f} / {t5:.2f} s, contact network off {o1:.2f} / {o5:.2f} s | "
                 f"pad error at 0.5 / 1 s: {at(err, 0.5)} / {at(err, 1.0)} mm"
                 + ("  | MODEL RUN STOPPED: crash limit" if pred["crashed"] else ""))
        plot_closed_loop(dict(scenario=sc, true=tr_full, pred=pred, one_step=None, contact_off=off,
                              mu_true=mu_true), cfg, stem + ".png", title=title, mode="open loop")
        gif = made < n_gifs
        if gif:
            tr = _truth_slice(d, r["start"], pred["n_frames"])
            animate_closed_loop(dict(scenario=sc, true=tr, pred=pred, one_step=None), cfg, stem + ".gif",
                                stride=gif_stride, mode="open loop")
            made += 1
        if verbose:
            print(f"  saved {os.path.basename(stem)}.png" + (" and .gif" if gif else "")
                  + ("  (crash limit reached)" if pred["crashed"] else ""))
    return metrics
