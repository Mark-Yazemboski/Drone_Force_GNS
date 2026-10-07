"""Two-stage training for the drone force model.

  Stage 1 (aero):    contact-free windows only, contact GNN off. Trains the aero
                     GNN and k_f, k_m, and the anchor-law coefficients.
  Stage 2 (contact): all windows, aero GNN and physical coefficients frozen.
                     Trains the contact GNN and mu.

Both stages use the same multistep unroll loss through the integrator (rotor
speeds and wind teacher-forced from the logs), the same input noise, and the
same per-sample rotation augmentation. Normalization statistics are fit right
before each stage (see drone_normalization.py for what and why).

Validation:
  - k-step error out to the MPC horizon, split by regime from ground truth
    (free / near-wall / contact), measured at the pad center and the COM
  - MuJoCo only: learned contact and aero forces vs. the force labels, in N.
    Labels are never used for training.
"""

import copy
import os
import time
from dataclasses import dataclass, asdict

import torch

from force_gns import nodes_from_state
from physics_losses import PhysicsLosses
from drone_config import DroneConfig
from drone_gns import DroneForceModel
from drone_data import (build_chain_index, split_chain_index, annotate_pad_distance,
                        iterate_drone_chains, rotate_drone_chain)
from drone_normalization import xy_pooled_scale, rms_scale


@dataclass
class DroneTrainSettings:
    # ---- model ----
    h: int = 2                      # velocity-history length (h+1 input frames)
    # Each GNN:
    #   latent_dim           width of every MLP and of the node/edge latents
    #   mlp_layers           Linear layers in every MLP (2 = Linear-ReLU-Linear, as in the cube)
    #   msg_passing_steps    rounds of message passing, each with its own weights
    #   msg_passing_repeats  times the whole sequence of steps is rerun with the same weights
    # Total rounds = steps x repeats; information moves one edge per round.
    aero_latent_dim: int = 128
    aero_mlp_layers: int = 2
    aero_msg_passing_steps: int = 4     # front rotor -> rod -> rod -> rod -> pad is 4 hops
    aero_msg_passing_repeats: int = 1
    contact_latent_dim: int = 128
    contact_mlp_layers: int = 2
    contact_msg_passing_steps: int = 3  # COM -> pad center -> ring -> ring covers the contact graph
    contact_msg_passing_repeats: int = 1
    contact_d0: float = 0.006       # gate center: pad sphere radius (4 mm) + margin
    contact_tau: float = 0.0015
    aero_dist_max: float = 1.0      # rotor-to-wall distance is clamped/scaled by this (m)
    # Physical coefficients (learned in stage 1, frozen in stage 2).
    learn_thrust_coeffs: bool = True     # k_f, k_m (kept near thrust-stand values by w_prior)
    # k_f, k_m start at (and the prior is centered on) the DroneConfig values
    # times these scales. Leave at 1. In sim, set e.g. 1.1 to test whether
    # training recovers a thrust coefficient that was measured 10% wrong.
    k_f_scale: float = 1.0
    k_m_scale: float = 1.0
    learn_drag_coeffs: bool = True       # k_rot, k_body, k_rod, k_pad (the anchor-law coefficients)
    k_rot_init: float = 5e-5        # rotor drag:  k_rot * w_j * u_perp      (per rotor, 1/rad)
    k_body_init: float = 0.02       # body drag:   k_body * |u| u            (1/m)
    k_rod_init: float = 0.05        # rod drag:    k_rod * |u_perp| u_perp   (1/m)
    k_pad_init: float = 0.05        # pad drag:    k_pad * |u| u             (1/m)
    # ---- windows / batches ----
    multistep: int = 4
    batch_size: int = 256
    aero_min_pad_dist: float = 0.026  # stage-1 (aero) windows: every pad node stays farther than
                                      # this from the wall for the whole window (m)
    # Input noise (random walk on the input window). None = set per stage as
    # noise_frac x that stage's median residual (the typical size of the signal
    # it learns); rotation noise gives the same displacement at the pad.
    noise_frac: float = 0.2
    com_noise: float = None         # m/step, explicit override
    rot_noise: float = None         # rad/step, explicit override
    rotate_aug: bool = True
    stats_batches: int = 40
    # ---- stage 1: aero ----
    aero_epochs: int = 200
    aero_lr: float = 3e-4
    w_aero_anchor: float = 0.1      # each aero node's force toward its drag law
    w_aero_smooth: float = 0.01     # aero force changes smoothly between steps
    w_axial: float = 0.1            # rotor axial thrust correction toward zero (no law to anchor to)
    w_prior: float = 1e-3           # k_f, k_m toward their thrust-stand values
    # ---- stage 2: contact ----
    contact_epochs: int = 300
    contact_lr: float = 1e-4
    w_fric_dir: float = 1.0
    w_fric_mag: float = 1.0
    w_fric_cone: float = 1.0
    mu_init: float = 0.3
    learn_mu: bool = True
    slip_v0: float = 3e-4           # m/step (0.03 m/s at dt = 0.01 s); slides are 0.05-0.25 m/s
    slip_tau: float = 1e-4
    # ---- validation ----
    val_horizon: int = 20           # steps rolled forward per validation window (match the MPC horizon)
    val_stride: int = 5             # a validation window starts every val_stride frames
    val_interval: int = 10          # validate every val_interval epochs
    near_thresh: float = 0.05


# ======================================================================
# Helpers
# ======================================================================

def _window(batch, h):
    return ([batch["com_win"][:, j] for j in range(h + 1)],
            [batch["R_win"][:, j] for j in range(h + 1)])


def _loss_nodes(model, com, R):
    # Loss is measured on every aero node (COM, rotors, rod, pad): together they
    # pin down both translation and attitude.
    return nodes_from_state(com, R, model.aero_rest)


def _set_trainable(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def _batches(data, index, s, device, K, train, noise=(0.0, 0.0)):
    for b in iterate_drone_chains(data, index, s.batch_size, s.h, K, device,
                                  noise[0] if train else 0.0, noise[1] if train else 0.0,
                                  shuffle=train):
        yield rotate_drone_chain(b) if (train and s.rotate_aug) else b


# Input-noise levels for a stage: explicit settings win, otherwise a fraction of
# the stage's typical (median) residual. Noise much larger than the signal buries
# it: the cube's 3e-5 m/step is ~3x the drone's whole aero signal at dt = 0.01 s.
# The median, not the RMS, because impacts make the contact residual heavy-tailed:
# an RMS-based level is set by taps and drowns the 1-5 N pushes.
def _stage_noise(model, s, stage):
    scale = float(model.noise_ref_contact if stage == "contact" else model.noise_ref_aero)
    com = s.com_noise if s.com_noise is not None else s.noise_frac * scale
    lever = float(model.aero_rest.norm(dim=1).max())
    rot = s.rot_noise if s.rot_noise is not None else s.noise_frac * scale / lever
    return com, rot


# ======================================================================
# Normalization statistics
# ======================================================================

# Node-acceleration errors of the current model over a K-step unroll, with the
# stage's learned force switched off (aero is zero at init; contact is off).
# Their RMS is the stage's loss scale, so a loss of ~1 means "no better than
# predicting zero learned force" for the same unroll length used in training.
@torch.no_grad()
def _zero_force_residuals(model, data, index, s, device):
    res = []
    for i, b in enumerate(_batches(data, index, s, device, s.multistep, train=False)):
        if i >= s.stats_batches:
            break
        com_h, R_h = _window(b, s.h)
        for k in range(s.multistep):
            com_n, R_n, _ = model.step(com_h, R_h, b["omega"][:, k], b["wind"][:, k],
                                       b["wall_n"], b["wall_c"], use_contact=False)
            x_prev, x_curr = _loss_nodes(model, com_h[-2], R_h[-2]), _loss_nodes(model, com_h[-1], R_h[-1])
            a_true = _loss_nodes(model, b["tgt_com"][:, k], b["tgt_R"][:, k]) - 2 * x_curr + x_prev
            res.append(a_true - (_loss_nodes(model, com_n, R_n) - 2 * x_curr + x_prev))
            com_h, R_h = com_h[1:] + [com_n], R_h[1:] + [R_n]
    return torch.cat([r.reshape(-1, 3) for r in res])


# Before stage 1, on clean contact-free windows:
#   aero input stats (body-frame airspeeds), the aero output scale (RMS of the
#   residual measured - thrust - gravity, i.e. the aero signal), and the stage-1
#   loss scale (RMS node-acceleration error of a model with zero learned force,
#   one number for all axes so a millimeter counts the same in every direction).
@torch.no_grad()
def fit_aero_stats(model, data, free_idx, s, device):
    feats, res_com = [], []
    for i, b in enumerate(_batches(data, free_idx, s, device, 1, train=False)):
        if i >= s.stats_batches:
            break
        com_h, R_h = _window(b, s.h)
        u_b, _ = model.aero_airspeed(com_h[-2], com_h[-1], R_h[-2], R_h[-1], b["wind"][:, 0])
        feats.append(torch.cat([u_b, u_b.norm(dim=-1, keepdim=True)], -1).reshape(-1, 4))
        f_thr, _ = model.thrust_wrench(b["omega"][:, 0], R_h[-1])
        res_com.append(b["tgt_com"][:, 0] - 2 * com_h[-1] + com_h[-2] - f_thr - model.g_step)
    model.aero_in.fit(torch.cat(feats))
    model.aero_scale.copy_(rms_scale(torch.cat(res_com)))
    res = _zero_force_residuals(model, data, free_idx, s, device)
    model.loss_scale_aero.copy_(rms_scale(res))
    model.noise_ref_aero.copy_(res.norm(dim=-1).median() / 3 ** 0.5)


# Before stage 2, with the trained aero model:
#   contact input stats over all windows (the non-contact acceleration now
#   includes aero); the contact output scale from contact windows (residual
#   after thrust, aero, gravity); the stage-2 loss scale over all windows.
@torch.no_grad()
def fit_contact_stats(model, data, all_idx, contact_idx, s, device):
    dyns, edges = [], []
    for i, b in enumerate(_batches(data, all_idx, s, device, 1, train=False)):
        if i >= s.stats_batches:
            break
        com_h, R_h = _window(b, s.h)
        nc = model.noncontact(com_h, R_h, b["omega"][:, 0], b["wind"][:, 0], b["wall_n"], b["wall_c"])
        dyn, _, e = model.contact_features(nc["hist"], nc["a_nc"], b["wall_n"], b["wall_c"])
        dyns.append(dyn)
        edges.append(e)
    model.contact_in.fit(torch.cat(dyns))
    model.contact_edge.fit(torch.cat(edges))

    res_com = []
    for i, b in enumerate(_batches(data, contact_idx, s, device, 1, train=False)):
        if i >= s.stats_batches:
            break
        com_h, R_h = _window(b, s.h)
        nc = model.noncontact(com_h, R_h, b["omega"][:, 0], b["wind"][:, 0], b["wall_n"], b["wall_c"],
                              need_contact_inputs=False)
        res_com.append(b["tgt_com"][:, 0] - 2 * com_h[-1] + com_h[-2] - nc["f_nc"] - model.g_step)
    model.contact_scale.copy_(xy_pooled_scale(torch.cat(res_com)))
    # Loss scale over the same windows stage 2 trains on (all of them), so ~1
    # still means "no better than zero contact force"; noise from the typical
    # residual on contact windows, the signal the contact model has to learn.
    model.loss_scale_contact.copy_(rms_scale(_zero_force_residuals(model, data, all_idx, s, device)))
    res = _zero_force_residuals(model, data, contact_idx, s, device)
    model.noise_ref_contact.copy_(res.norm(dim=-1).median() / 3 ** 0.5)


# ======================================================================
# Loss
# ======================================================================

def unroll_loss(model, phys, batch, s, stage):
    h, K = s.h, batch["tgt_com"].shape[1]
    use_contact = stage == "contact"
    loss_scale = model.loss_scale_contact if use_contact else model.loss_scale_aero
    com_h, R_h = _window(batch, h)
    n_b1 = batch["wall_n"].unsqueeze(1)
    fric_w = dict(w_fric_dir=s.w_fric_dir, w_fric_mag=s.w_fric_mag, w_fric_cone=s.w_fric_cone)

    pred_terms, raws, F_series = [], {}, []
    for k in range(K):
        com_n, R_n, aux = model.step(com_h, R_h, batch["omega"][:, k], batch["wind"][:, k],
                                     batch["wall_n"], batch["wall_c"], use_contact=use_contact)
        x_prev, x_curr = _loss_nodes(model, com_h[-2], R_h[-2]), _loss_nodes(model, com_h[-1], R_h[-1])
        a_pred = _loss_nodes(model, com_n, R_n) - 2 * x_curr + x_prev
        a_true = _loss_nodes(model, batch["tgt_com"][:, k], batch["tgt_R"][:, k]) - 2 * x_curr + x_prev
        pred_terms.append(((a_pred - a_true) / loss_scale).pow(2).mean())

        if stage == "aero":
            a = aux["aero"]
            sc = model.aero_scale
            raws["aero_anchor"] = raws.get("aero_anchor", 0.0) + \
                ((a["f_body"] - a["law_body"]) / sc).pow(2).sum(-1).mean() / K
            raws["axial"] = raws.get("axial", 0.0) + (a["axial"] / sc).pow(2).mean() / K
            F_series.append(a["F"])
        else:
            zeros = torch.zeros_like(com_n)
            for name, v in phys.compute_step_terms(aux["phi_c"], aux["c_w"], aux["v_node"], n_b1,
                                                   zeros, zeros, fric_w).items():
                raws[name] = raws.get(name, 0.0) + v / K

        com_h = com_h[1:] + [com_n]
        R_h = R_h[1:] + [R_n]

    pred = torch.stack(pred_terms).mean()
    total = pred
    if stage == "aero":
        if K > 1:
            raws["aero_smooth"] = torch.stack([((F_series[k] - F_series[k - 1]) / model.aero_scale).pow(2).sum(-1).mean()
                                               for k in range(1, K)]).mean()
        weights = dict(aero_anchor=s.w_aero_anchor, axial=s.w_axial, aero_smooth=s.w_aero_smooth)
        total = total + sum(weights[n] * v for n, v in raws.items()) + s.w_prior * model.params.prior_loss()
    else:
        total = total + PhysicsLosses.weighted_total(raws, fric_w)
    return total, pred.detach(), {n: float(torch.as_tensor(v).detach()) for n, v in raws.items()}


# ======================================================================
# Validation
# ======================================================================

# k-step error out to the horizon, split by ground-truth regime over the window
# (closest pad node to the wall): contact < d0 <= near < near_thresh <= free.
@torch.no_grad()
def kstep_validation(model, data, index, s, device, use_contact):
    model.eval()
    H = s.val_horizon
    pad_c = model.aero_pad_idx
    sums = {}
    for b in _batches(data, index, s, device, H, train=False):
        com_h, R_h = _window(b, s.h)
        com_err = tip_err = None
        dmin = torch.full((b["B"],), 1e9, device=device)
        tip_sum = 0.0
        for k in range(H):
            com_n, R_n, _ = model.step(com_h, R_h, b["omega"][:, k], b["wind"][:, k],
                                       b["wall_n"], b["wall_c"], use_contact=use_contact)
            p_pred = _loss_nodes(model, com_n, R_n)[:, pad_c]
            p_true = _loss_nodes(model, b["tgt_com"][:, k], b["tgt_R"][:, k])[:, pad_c]
            tip_err = (p_pred - p_true).norm(dim=-1)
            tip_sum = tip_sum + tip_err
            com_err = (com_n - b["tgt_com"][:, k]).norm(dim=-1)
            pads = nodes_from_state(b["tgt_com"][:, k], b["tgt_R"][:, k], model.pad_rest)
            dmin = torch.minimum(dmin, ((pads - b["wall_c"].unsqueeze(1)) * b["wall_n"].unsqueeze(1)).sum(-1).min(1).values)
            com_h, R_h = com_h[1:] + [com_n], R_h[1:] + [R_n]
        regime = torch.where(dmin < model.contact_d0, 2, torch.where(dmin < s.near_thresh, 1, 0))
        for r_id, name in ((0, "free"), (1, "near"), (2, "contact"), (None, "all")):
            m = torch.ones_like(regime, dtype=torch.bool) if r_id is None else regime == r_id
            if m.any():
                acc = sums.setdefault(name, [0.0, 0.0, 0.0, 0])
                acc[0] += float(com_err[m].sum())
                acc[1] += float(tip_err[m].sum())
                acc[2] += float(tip_sum[m].sum()) / H
                acc[3] += int(m.sum())
    model.train()
    return {n: dict(com_end_mm=1e3 * a[0] / a[3], pad_end_mm=1e3 * a[1] / a[3],
                    pad_mean_mm=1e3 * a[2] / a[3], n=a[3]) for n, a in sums.items()}


# MuJoCo only: one-step learned forces vs. the generator's force labels (N).
@torch.no_grad()
def force_validation(model, data, index, s, device, max_batches=20):
    if not all("F_contact" in d and "F_aero" in d for d in data):
        return {}
    model.eval()
    acc = dict(c_sq=0.0, c_ref=0.0, cn=0.0, ct=0.0, n_c=0, false_f=0.0, n_f=0, a_sq=0.0, a_ref=0.0, n_a=0)
    for i, b in enumerate(_batches(data, index, s, device, 1, train=False)):
        if i >= max_batches:
            break
        com_h, R_h = _window(b, s.h)
        _, _, aux = model.step(com_h, R_h, b["omega"][:, 0], b["wind"][:, 0], b["wall_n"], b["wall_c"])
        n = b["wall_n"]
        Fc_p, Fc_t = model.to_newtons(aux["f_c"]), b["F_contact"][:, 0]
        Fa_p, Fa_t = model.to_newtons(aux["aero"]["F"]), b["F_aero"][:, 0]
        touching = Fc_t.norm(dim=-1) > 0.05
        if touching.any():
            d = Fc_p[touching] - Fc_t[touching]
            nn_ = n[touching]
            dn = (d * nn_).sum(-1)
            acc["c_sq"] += float(d.pow(2).sum(-1).sum())
            acc["c_ref"] += float(Fc_t[touching].pow(2).sum(-1).sum())
            acc["cn"] += float(dn.pow(2).sum())
            acc["ct"] += float((d - dn.unsqueeze(-1) * nn_).pow(2).sum(-1).sum())
            acc["n_c"] += int(touching.sum())
        if (~touching).any():
            acc["false_f"] += float(Fc_p[~touching].norm(dim=-1).sum())
            acc["n_f"] += int((~touching).sum())
        acc["a_sq"] += float((Fa_p - Fa_t).pow(2).sum(-1).sum())
        acc["a_ref"] += float(Fa_t.pow(2).sum(-1).sum())
        acc["n_a"] += Fa_t.shape[0]
    model.train()
    out = {}
    if acc["n_c"]:
        out.update(contact_rmse_N=(acc["c_sq"] / acc["n_c"]) ** 0.5,
                   contact_label_rms_N=(acc["c_ref"] / acc["n_c"]) ** 0.5,
                   contact_normal_rmse_N=(acc["cn"] / acc["n_c"]) ** 0.5,
                   contact_tangential_rmse_N=(acc["ct"] / acc["n_c"]) ** 0.5)
    if acc["n_f"]:
        out["false_contact_mean_N"] = acc["false_f"] / acc["n_f"]
    if acc["n_a"]:
        out.update(aero_rmse_N=(acc["a_sq"] / acc["n_a"]) ** 0.5,
                   aero_label_rms_N=(acc["a_ref"] / acc["n_a"]) ** 0.5)
    return out


# ======================================================================
# Checkpoints
# ======================================================================

def save_checkpoint(path, model, phys, cfg, s, extra=None):
    torch.save({"model": model.state_dict(), "phys": phys.state_dict(), "drone_cfg": cfg.to_dict(),
                "settings": asdict(s), **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = DroneConfig(**ck["drone_cfg"])
    s = DroneTrainSettings(**ck["settings"])
    model = DroneForceModel(cfg, s).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, s, ck


# ======================================================================
# Training
# ======================================================================

def _make_phys(cfg, s):
    return PhysicsLosses(phi_g=cfg.gravity * cfg.dt ** 2, ang_scale_vec=torch.ones(3),
                         mu_init=s.mu_init, learn_mu=s.learn_mu, k_init=1.0, learn_k=False,
                         slip_v0=s.slip_v0, slip_tau=s.slip_tau)


def train_stage(stage, model, phys, cfg, s, train_data, train_idx, val_data, val_idx, stem,
                device, history, verbose):
    if stage == "aero":
        _set_trainable(model, False)
        _set_trainable(model.aero, True)
        _set_trainable(model.params, True)
        params, lr, epochs = list(model.aero.parameters()) + list(model.params.parameters()), s.aero_lr, s.aero_epochs
    else:
        _set_trainable(model, False)
        _set_trainable(model.contact, True)
        params, lr, epochs = list(model.contact.parameters()) + list(phys.parameters()), s.contact_lr, s.contact_epochs
    opt = torch.optim.Adam(params, lr=lr)
    use_contact = stage == "contact"
    hist = history.setdefault(stage, dict(train=[], val=[], params=[]))
    best, best_state = float("inf"), None
    noise = _stage_noise(model, s, stage)
    if verbose:
        print(f"[{stage}] input noise: COM {noise[0]:.2e} m/step, rotation {noise[1]:.2e} rad/step")

    for epoch in range(epochs):
        t0 = time.time()
        tot = pred = 0.0
        nb, raw_acc = 0, {}
        for b in _batches(train_data, train_idx, s, device, s.multistep, train=True, noise=noise):
            loss, p, raws = unroll_loss(model, phys, b, s, stage)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            pred += float(p)
            nb += 1
            for k, v in raws.items():
                raw_acc[k] = raw_acc.get(k, 0.0) + v
        nb = max(nb, 1)
        hist["train"].append((epoch, tot / nb, pred / nb))
        pv = dict(epoch=epoch, **model.params.as_dict(), mu=float(phys.mu.detach()))
        hist["params"].append(pv)
        msg = (f"[{stage}] ep {epoch:4d}  loss {tot / nb:.3e}  pred {pred / nb:.3e}  "
               + " ".join(f"{k}={v / nb:.2e}" for k, v in raw_acc.items())
               + (f"  k_f {pv['k_f']:.3e} k_rot {pv['k_rot']:.2e}" if stage == "aero" else f"  mu {pv['mu']:.3f}")
               + f"  {time.time() - t0:.1f}s")

        if (epoch + 1) % s.val_interval == 0 or epoch == epochs - 1:
            kval = kstep_validation(model, val_data, val_idx, s, device, use_contact)
            fval = force_validation(model, val_data, val_idx, s, device)
            hist["val"].append(dict(epoch=epoch, kstep=kval, forces=fval))
            score = kval.get("all", {}).get("pad_end_mm", float("inf"))
            msg += f"  | val pad err @{s.val_horizon}: " + " ".join(
                f"{r} {kval[r]['pad_end_mm']:.2f}mm" for r in ("free", "near", "contact", "all") if r in kval)
            if fval:
                msg += "  | " + " ".join(f"{k} {v:.3f}" for k, v in fval.items())
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                save_checkpoint(f"{stem}_{stage}_best.pt", model, phys, cfg, s,
                                {"stage": stage, "epoch": epoch, "history": history})
        if verbose:
            print(msg, flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)          # continue from the best validation point
    return model


def train_drone_force_model(cfg, s, train_data, val_data, save_path, device=None,
                            stages=("aero", "contact"), verbose=True, init_checkpoint=None):
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stem = os.path.splitext(save_path)[0]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    model = DroneForceModel(cfg, s).to(device)
    phys = _make_phys(cfg, s).to(device)
    if init_checkpoint:                              # e.g. start stage 2 from a saved stage-1 model
        ck = torch.load(init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])

    pad_rest = model.pad_rest.cpu()
    annotate_pad_distance(train_data, pad_rest)
    annotate_pad_distance(val_data, pad_rest)
    free_d, contact_d = s.aero_min_pad_dist, s.contact_d0

    tr_all = build_chain_index(train_data, s.h, s.multistep)
    tr_free, tr_contact = split_chain_index(train_data, tr_all, s.h, s.multistep, free_d, contact_d)
    va_all = build_chain_index(val_data, s.h, s.val_horizon, stride=s.val_stride)
    va_free, _ = split_chain_index(val_data, va_all, s.h, s.val_horizon, free_d, contact_d)
    if verbose:
        print(f"windows: train {len(tr_all)} (contact-free {len(tr_free)}, contact {len(tr_contact)}) | "
              f"val {len(va_all)} (contact-free {len(va_free)})")
    history = {}

    if "aero" in stages:
        if not tr_free:
            raise ValueError("no contact-free training windows for stage 1")
        fit_aero_stats(model, train_data, tr_free, s, device)
        if verbose:
            print(f"stage 1 scales: aero output {float(model.aero_scale):.3e}, "
                  f"loss {float(model.loss_scale_aero):.3e} (m/step^2)")
        train_stage("aero", model, phys, cfg, s, train_data, tr_free, val_data, va_free,
                    stem, device, history, verbose)

    if "contact" in stages:
        if not tr_contact:
            raise ValueError("no contact training windows for stage 2")
        fit_contact_stats(model, train_data, tr_all, tr_contact, s, device)
        if verbose:
            print(f"stage 2 scales: contact output {[f'{v:.3e}' for v in model.contact_scale.tolist()]}, "
                  f"loss {float(model.loss_scale_contact):.3e} (m/step^2)")
        train_stage("contact", model, phys, cfg, s, train_data, tr_all, val_data, va_all,
                    stem, device, history, verbose)

    _set_trainable(model, True)
    save_checkpoint(f"{stem}_final.pt", model, phys, cfg, s, {"stage": "final", "history": history})
    torch.save(history, f"{stem}_history.pt")
    return model, phys, history
