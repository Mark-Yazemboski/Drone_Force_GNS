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
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from force_gns import nodes_from_state
from physics_losses import PhysicsLosses
from drone_config import DroneConfig
from drone_gns import DroneForceModel
from drone_data import (build_chain_index, filter_chain_index, annotate_pad_distance,
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
    contact_dist_clamp: tuple = (-0.05, 0.5)   # contact-GNN wall-distance feature clamped to this range (m),
                                               # as in the cube; a tighter range (e.g. (-0.01, 0.02)) spends
                                               # more of the normalized feature on the last few mm
    # Physical coefficients (learned in stage 1, frozen in stage 2).
    learn_thrust_coeffs: bool = True     # k_f, k_m (kept near thrust-stand values by w_prior)
    # k_f, k_m start at (and the prior is centered on) the DroneConfig values
    # times these scales. Leave at 1. In sim, set e.g. 1.1 to test whether
    # training recovers a thrust coefficient that was measured 10% wrong.
    k_f_scale: float = 1.0
    k_m_scale: float = 1.0
    learn_drag_coeffs: bool = True       # k_rot, k_body, k_rod, k_pad (the anchor-law coefficients)
    # How the drag coefficients are learned:
    #   "lstsq"    exact least squares. The laws are linear in the k's, so they are
    #              fit in closed form: from the measured motion residual (measured
    #              acceleration - thrust - gravity, contact-free windows) before stage 1,
    #              then refit to the aero network's total force and torque every
    #              drag_refit_interval epochs. No learning rate, no drift.
    #   "gradient" gradient descent through w_aero_coeff_fit. Converges very slowly:
    #              the laws are strongly correlated (rotor vs body drag ~0.94), so
    #              from a poor start the k's crawl along a long valley.
    drag_coeff_fit: str = "lstsq"
    drag_refit_interval: int = 5
    # Initial drag coefficients ("gradient" fit, or learn_drag_coeffs=False). With
    # "lstsq" they are replaced by the fit from the motion residual before stage 1.
    k_rot_init: float = 5e-5        # rotor drag:  k_rot * w_j * u_perp      (per rotor, 1/rad)
    k_body_init: float = 0.02       # body drag:   k_body * |u| u            (1/m)
    k_rod_init: float = 0.05        # rod drag:    k_rod * |u_perp| u_perp   (1/m)
    k_pad_init: float = 0.05        # pad drag:    k_pad * |u| u             (1/m)
    # ---- windows / batches ----
    multistep: int = 4
    batch_size: int = 256
    eval_batch_size: int = 2048     # no-grad passes (stats, drag refit, validation): no effect on the
                                    # optimization, only speed. stats_batches counts these batches.
    aero_min_pad_dist: float = 0.026  # stage-1 (aero) windows: every pad node stays farther than
                                      # this from the wall for the whole window (m)
    contact_max_pad_dist: float = 0.05  # stage-2 (contact) windows: the pad comes closer than this
                                        # at some frame of the window (m). None = every window.
                                        # Farther windows carry no gradient for the contact GNN
                                        # (gate ~ e^-29 at 5 cm), so skipping them only saves time.
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
    w_aero_anchor: float = 0.1      # each aero node's force toward its drag law (moves the network only)
    w_aero_coeff_fit: float = 0.01  # drag coefficients fit to the network's total aero force/torque
                                    # (moves only the k's; Adam normalizes their step, so the weight
                                    # mainly sets how much this term adds to the logged total loss)
    w_aero_smooth: float = 0.01     # aero force changes smoothly between steps
    w_axial: float = 0.1            # rotor axial thrust correction toward zero (no law to anchor to)
    w_prior: float = 1e-3           # k_f, k_m toward their thrust-stand values
    coeff_lr: float = 1e-3          # learning rate for the physical coefficients (k's in stage 1, mu in stage 2)
    coeff_warmup_epochs: int = 10   # coefficients held fixed for the first epochs of each stage, so they
                                    # are not pulled toward an untrained network's forces (capped at
                                    # half the stage's epochs)
    lr_schedule: str = None         # None (constant) or "cosine" (network lr decays to 0 over each stage)
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
    best_metric: str = "val_loss"   # "val_loss" (same normalized loss as training, on validation
                                    # windows, no noise/augmentation) or "pad_err" (pad error at val_horizon)
    impact_force_N: float = 10.0    # force-label validation splits contact frames at this magnitude
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
    for b in iterate_drone_chains(data, index, s.batch_size if train else s.eval_batch_size, s.h, K, device,
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
#   contact input stats over the stage-2 windows (the non-contact acceleration
#   now includes aero); the contact output scale from touching windows (residual
#   after thrust, aero, gravity); the stage-2 loss scale over the stage-2 windows.
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
# Drag coefficients by least squares
# ======================================================================

# Fits [k_rot, k_body, k_rod, k_pad] in closed form (non-negative least squares),
# on clean contact-free windows (first step of each), in the body frame:
#   target="residual": the measured COM residual (measured acceleration - thrust
#       - gravity). Motion data only, so it works the same on hardware. Force only.
#   target="network":  the aero network's total force and torque about the COM.
# Returns the coefficient vector and R^2 of the fit.
@torch.no_grad()
def fit_drag_coefficients(model, data, index, s, device, target):
    from scipy.optimize import nnls
    A_rows, y_rows = [], []
    L = float(model.aero_rest.norm(dim=-1).max())
    r_b = model.aero_rest.unsqueeze(0)
    for i, b in enumerate(_batches(data, index, s, device, 1, train=False)):
        if i >= s.stats_batches:
            break
        com_h, R_h = _window(b, s.h)
        omega, wind = b["omega"][:, 0], b["wind"][:, 0]
        u_b, _ = model.aero_airspeed(com_h[-2], com_h[-1], R_h[-2], R_h[-1], wind)
        basis = model.drag_basis(u_b, omega)                                   # (B, Na, 3, 4)
        A_F = basis.sum(1)                                                     # (B, 3, 4)
        if target == "residual":
            f_thr, _ = model.thrust_wrench(omega, R_h[-1])
            res_w = b["tgt_com"][:, 0] - 2 * com_h[-1] + com_h[-2] - f_thr - model.g_step
            A_rows.append(A_F)
            y_rows.append(torch.einsum('bji,bj->bi', R_h[-1], res_w))         # body frame
        else:
            a = model.aero_forces(com_h[-2], com_h[-1], R_h[-2], R_h[-1], omega, wind, b["wall_n"], b["wall_c"])
            A_T = torch.linalg.cross(r_b.unsqueeze(-1).expand_as(basis), basis, dim=2).sum(1) / L
            A_rows.append(torch.cat([A_F, A_T], 1))
            y_rows.append(torch.cat([a["f_drag"].sum(1), torch.linalg.cross(r_b.expand_as(a["f_drag"]),
                                                                            a["f_drag"], dim=-1).sum(1) / L], 1))
    A = torch.cat(A_rows).reshape(-1, 4).double().cpu().numpy()
    y = torch.cat(y_rows).reshape(-1).double().cpu().numpy()
    col = np.linalg.norm(A, axis=0) + 1e-300
    k, _ = nnls(A / col, y)
    k = k / col
    r2 = 1.0 - float(((A @ k - y) ** 2).sum() / max((y ** 2).sum(), 1e-300))
    return k, r2


def _fmt_k(k):
    return "  ".join(f"{n} {v:.3g}" for n, v in zip(("k_rot", "k_body", "k_rod", "k_pad"), k))


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
            # Per-node anchor: pulls each node's drag toward its law. The law is
            # DETACHED, so this term shapes the network and never moves the k's.
            # (The data pins only the TOTAL aero wrench, not how it is split among
            # nodes; letting k follow each node made the k's drift with whatever
            # split the network happened to use, and dragged the total down.)
            raws["aero_anchor"] = raws.get("aero_anchor", 0.0) + \
                ((a["f_drag"] - a["law_body"].detach()) / sc).pow(2).sum(-1).mean() / K
            # Coefficient fit: the summed laws (force and torque about the COM,
            # body frame) against the network's total, which is DETACHED. Only the
            # k's move: a regression of the data-pinned aero wrench onto the laws.
            if s.drag_coeff_fit == "gradient":
                r_b = model.aero_rest.unsqueeze(0)
                f_net, law = a["f_drag"].detach(), a["law_body"]
                L = r_b.norm(dim=-1).max()
                raws["aero_coeff_fit"] = raws.get("aero_coeff_fit", 0.0) + (
                    ((f_net.sum(1) - law.sum(1)) / sc).pow(2).sum(-1).mean()
                    + ((torch.linalg.cross(r_b.expand_as(law), f_net - law, dim=-1).sum(1)) / (sc * L)).pow(2).sum(-1).mean()
                ) / K
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
        weights = dict(aero_anchor=s.w_aero_anchor, aero_coeff_fit=s.w_aero_coeff_fit, axial=s.w_axial,
                       aero_smooth=s.w_aero_smooth)
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
# Contact frames are split into sustained contact (pushing/sliding, label below
# impact_force_N) and impacts (taps), because a few large tap impulses would
# otherwise dominate a single RMSE and hide how well pushes and slides are learned.
@torch.no_grad()
def force_validation(model, data, index, s, device, max_batches=40):
    if not all("F_contact" in d and "F_aero" in d for d in data):
        return {}
    model.eval()
    acc, means = {}, set()

    def add(key, sq, n, mean=False):
        a_ = acc.setdefault(key, [0.0, 0])
        a_[0] += float(sq)
        a_[1] += float(n)
        if mean:
            means.add(key)

    def angle_deg(a, b):
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
        return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))

    pad_rest = model.aero_rest[model.aero_pad_idx].unsqueeze(0)               # pad center, COM-relative

    for i, b in enumerate(_batches(data, index, s, device, 1, train=False)):
        if i >= max_batches:
            break
        com_h, R_h = _window(b, s.h)
        _, _, aux = model.step(com_h, R_h, b["omega"][:, 0], b["wind"][:, 0], b["wall_n"], b["wall_c"])
        n = b["wall_n"]
        Fc_p, Fc_t = model.to_newtons(aux["f_c"]), b["F_contact"][:, 0]
        mag = Fc_t.norm(dim=-1)
        for name, m in (("sustained", (mag > 0.05) & (mag < s.impact_force_N)),
                        ("impact", mag >= s.impact_force_N)):
            if m.any():
                d = Fc_p[m] - Fc_t[m]
                dn = (d * n[m]).sum(-1)
                add(f"{name}_err", d.pow(2).sum(-1).sum(), m.sum())
                add(f"{name}_ref", Fc_t[m].pow(2).sum(-1).sum(), m.sum())
                add(f"{name}_normal", dn.pow(2).sum(), m.sum())
                add(f"{name}_tangential", (d - dn.unsqueeze(-1) * n[m]).pow(2).sum(-1).sum(), m.sum())
        # Contact torque about the COM, and the centre of pressure: where the
        # contact force's line of action meets the wall plane. Motion pins down
        # the net contact wrench (force + torque), so these are identifiable;
        # the split among pad nodes is not.
        if "tau_contact" in b:
            tau_p, tau_t = model.to_newtons(aux["tau_c"]), b["tau_contact"][:, 0]
            m = (mag > 0.05) & (mag < s.impact_force_N)
            if m.any():
                add("sustained_torque_err", (tau_p[m] - tau_t[m]).pow(2).sum(-1).sum(), m.sum())
                add("sustained_torque_ref", tau_t[m].pow(2).sum(-1).sum(), m.sum())

            def cop(F, tau, m):
                F, tau, nm = F[m], tau[m], n[m]
                x0 = com_h[-1][m] + torch.linalg.cross(F, tau, dim=-1) / F.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
                t_ = ((b["wall_c"][m] - x0) * nm).sum(-1, keepdim=True) / (F * nm).sum(-1, keepdim=True)
                return x0 + t_ * F
            Fn_p_ = (Fc_p * n).sum(-1)
            m = ((Fc_t * n).sum(-1) > 0.5) & (mag < s.impact_force_N) & (Fn_p_ > 0.05)
            if m.any():
                e = 1e3 * (cop(Fc_p, tau_p, m) - cop(Fc_t, tau_t, m)).norm(dim=-1)
                add("cop_err_mm", e.sum(), m.sum(), mean=True)
        free = mag <= 0.05
        if free.any():
            add("false_contact", Fc_p[free].norm(dim=-1).sum(), free.sum())

        # Sliding friction: frames where MuJoCo's pad is pressed on the wall
        # (normal > 0.2 N) and its center slides faster than the slip gate.
        tang = lambda F: F - (F * n).sum(-1, keepdim=True) * n
        pad_now = nodes_from_state(com_h[-1], R_h[-1], pad_rest)[:, 0]
        pad_prev = nodes_from_state(com_h[-2], R_h[-2], pad_rest)[:, 0]
        v_t = tang(pad_now - pad_prev)                                          # m/step
        Ft_p, Ft_t = tang(Fc_p), tang(Fc_t)
        Fn_t, Fn_p = (Fc_t * n).sum(-1), (Fc_p * n).sum(-1)
        # The population is defined by MuJoCo and the kinematics only, so a weak
        # or missing predicted force counts against the model instead of being
        # filtered out.
        sl = (Fn_t > 0.2) & (mag < s.impact_force_N) & (v_t.norm(dim=-1) > s.slip_v0) & \
             (Ft_t.norm(dim=-1) > 0.02)
        if sl.any():
            k = int(sl.sum())
            add("slide_fric_dir_err_deg", angle_deg(Ft_p[sl], Ft_t[sl]).sum(), k, mean=True)
            add("slide_fric_vs_slip_deg", angle_deg(Ft_p[sl], -v_t[sl]).sum(), k, mean=True)
            add("slide_fric_vs_slip_deg_label", angle_deg(Ft_t[sl], -v_t[sl]).sum(), k, mean=True)
            # sum|Ft| / sum Fn over the sliding frames (a ratio of sums stays finite
            # when the model predicts ~0 normal force on some frame)
            add("slide_mu_implied", Ft_p[sl].norm(dim=-1).sum(), Fn_p[sl].sum().clamp_min(1e-9), mean=True)
            add("slide_mu_implied_label", Ft_t[sl].norm(dim=-1).sum(), Fn_t[sl].sum(), mean=True)
            # Cancellation among pad nodes: 1 - |sum phi_t| / sum |phi_t| (0 = all aligned).
            phi_t = model.to_newtons(aux["phi_c"])[sl]
            phi_t = phi_t - (phi_t * n[sl].unsqueeze(1)).sum(-1, keepdim=True) * n[sl].unsqueeze(1)
            cancel = 1.0 - phi_t.sum(1).norm(dim=-1) / phi_t.norm(dim=-1).sum(1).clamp_min(1e-12)
            add("slide_fric_cancellation", cancel.sum(), k, mean=True)
            node_mag = model.to_newtons(aux["phi_c"])[sl].norm(dim=-1)
            add("slide_active_nodes", (node_mag > 0.1 * node_mag.sum(1, keepdim=True)).float().sum(), k, mean=True)
        Fa_p, Fa_t = model.to_newtons(aux["aero"]["F"]), b["F_aero"][:, 0]
        add("aero_err", (Fa_p - Fa_t).pow(2).sum(-1).sum(), Fa_t.shape[0])
        add("aero_ref", Fa_t.pow(2).sum(-1).sum(), Fa_t.shape[0])
        # Rotor axial thrust correction (N per rotor); should stay ~0 in MuJoCo.
        ax = model.to_newtons(aux["aero"]["axial"]) @ model.rotor_sel.T
        add("axial", ax.pow(2).sum(), ax.numel())
    model.train()
    out = {}
    for key, (v, n) in acc.items():
        # RMS, except false_contact (mean |F|) and the sliding diagnostics (means)
        out[key] = v / n if (key == "false_contact" or key in means) else (v / n) ** 0.5
        if key not in ("axial", "aero_ref", "aero_err") and key not in means:
            out[key + "_n"] = int(n)
        if key == "slide_fric_dir_err_deg":
            out["slide_frames_n"] = int(n)
    return out


# Full evaluation on a held-out set (the test trajectories): validation losses
# for both stages, k-step error by regime, and (MuJoCo) learned vs. true forces.
# Returns a flat dict of floats for the run report.
@torch.no_grad()
def evaluate_on_dataset(model, phys, data, s, device="cpu"):
    annotate_pad_distance(data, model.pad_rest.cpu())
    w = stage_windows(data, data, s)
    out = {"aero_loss": validation_loss(model, phys, data, w["vl_aero"], s, device, "aero"),
           "contact_loss": validation_loss(model, phys, data, w["vl_contact"], s, device, "contact")}
    for regime, m in kstep_validation(model, data, w["va_all"], s, device, use_contact=True).items():
        for k, v in m.items():
            out[f"kstep{s.val_horizon}_{regime}_{k}"] = v
    # Forces are scored on EVERY frame here (training-time validation samples
    # every val_stride frames), so short impacts are not skipped.
    every_frame = build_chain_index(data, s.h, 1, stride=1)
    for k, v in force_validation(model, data, every_frame, s, device, max_batches=10 ** 6).items():
        out[f"force_{k}"] = v
    return out


def _format_forces(f, impact_N):
    lines = ["    forces vs MuJoCo labels (RMSE / label RMS, N):"]
    for name, label in (("sustained", f"contact, sustained (<{impact_N:g} N)"),
                        ("impact", f"contact, impacts (>={impact_N:g} N)")):
        if f"{name}_err" in f:
            lines.append(f"      {label:30s} {f[name + '_err']:7.3f} / {f[name + '_ref']:7.3f}   "
                         f"[normal {f[name + '_normal']:.3f}, tangential {f[name + '_tangential']:.3f}]  "
                         f"({f[name + '_err_n']} frames)")
    if "sustained_torque_err" in f:
        lines.append(f"      {'contact torque, sustained':30s} {f['sustained_torque_err']:7.3f} / "
                     f"{f['sustained_torque_ref']:7.3f} N m")
    if "cop_err_mm" in f:
        lines.append(f"      {'centre of pressure':30s} {f['cop_err_mm']:7.2f} mm mean error")
    if "false_contact" in f:
        lines.append(f"      {'false contact (not touching)':30s} {f['false_contact']:7.3f} mean |F|")
    if "aero_err" in f:
        expl = 100 * (1 - (f["aero_err"] / max(f["aero_ref"], 1e-12)) ** 2)
        lines.append(f"      {'aero':30s} {f['aero_err']:7.4f} / {f['aero_ref']:7.4f}   ({expl:.0f}% of label variance explained)")
    if "axial" in f:
        lines.append(f"      {'rotor axial correction':30s} {f['axial']:7.4f} RMS per rotor (should be ~0 in sim)")
    if "slide_fric_dir_err_deg" in f:
        lines.append(f"    sliding friction ({f['slide_frames_n']} frames): direction error vs MuJoCo "
                     f"{f['slide_fric_dir_err_deg']:.1f} deg | angle from -slip: model {f['slide_fric_vs_slip_deg']:.1f}, "
                     f"MuJoCo {f['slide_fric_vs_slip_deg_label']:.1f} deg")
        lines.append(f"      implied mu |Ft|/Fn: model {f['slide_mu_implied']:.3f}, MuJoCo {f['slide_mu_implied_label']:.3f} | "
                     f"node cancellation {f['slide_fric_cancellation']:.2f} (0 = aligned) | "
                     f"active pad nodes {f['slide_active_nodes']:.1f}")
    return lines


# Same normalized loss as training (no input noise, no augmentation), on
# validation windows of the training unroll length.
@torch.no_grad()
def validation_loss(model, phys, data, index, s, device, stage):
    model.eval()
    tot = n = 0.0
    for b in _batches(data, index, s, device, s.multistep, train=False):
        tot += float(unroll_loss(model, phys, b, s, stage)[1]) * b["B"]
        n += b["B"]
    model.train()
    return tot / max(n, 1)


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


def train_stage(stage, model, phys, cfg, s, train_data, train_idx, val_data, val_idx, val_loss_idx, stem,
                device, history, verbose):
    # Network weights and physical coefficients get separate learning rates: the
    # coefficients (k's, mu) see weak, noisy gradients through the anchor and
    # friction losses and move very slowly at a network learning rate.
    _set_trainable(model, False)
    lstsq = stage == "aero" and s.drag_coeff_fit == "lstsq" and s.learn_drag_coeffs
    if stage == "aero":
        _set_trainable(model.aero, True)
        _set_trainable(model.params, True)
        coeffs = model.params.thrust_parameters() + ([] if lstsq else model.params.drag_parameters())
        groups = [dict(params=list(model.aero.parameters()), lr=s.aero_lr),
                  dict(params=coeffs, lr=s.coeff_lr)]
        epochs = s.aero_epochs
    else:
        _set_trainable(model.contact, True)
        groups = [dict(params=list(model.contact.parameters()), lr=s.contact_lr),
                  dict(params=list(phys.parameters()), lr=s.coeff_lr)]
        epochs = s.contact_epochs
    coeff_params = groups[1]["params"]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.Adam(groups)
    # Learning-rate schedule for the network weights (coefficients keep theirs).
    sched = None
    if s.lr_schedule == "cosine":
        f_net = lambda e: 0.5 * (1.0 + math.cos(math.pi * min(e, epochs) / max(epochs, 1)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, [f_net] + [lambda e: 1.0] * (len(groups) - 1))
    elif s.lr_schedule is not None:
        raise ValueError(f"lr_schedule must be None or 'cosine', got {s.lr_schedule!r}")
    # Coefficient warmup, capped at half the stage so short runs still learn mu / k_f.
    warmup = min(s.coeff_warmup_epochs, epochs // 2)
    if verbose and warmup < s.coeff_warmup_epochs and coeff_params:
        print(f"[{stage}] note: coeff_warmup_epochs={s.coeff_warmup_epochs} >= half of {epochs} epochs; "
              f"using {warmup} so the coefficients ({'k_f, k_m' if stage == 'aero' else 'mu'}) can still move")
    use_contact = stage == "contact"
    hist = history.setdefault(stage, dict(train=[], val=[], params=[]))
    best, best_state, best_phys = float("inf"), None, None
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
            if epoch < warmup:
                for cp in coeff_params:
                    cp.grad = None                 # Adam skips parameters without a gradient
            opt.step()
            tot += float(loss.detach())
            pred += float(p)
            nb += 1
            for k, v in raws.items():
                raw_acc[k] = raw_acc.get(k, 0.0) + v
        nb = max(nb, 1)
        if sched is not None:
            sched.step()
        if lstsq and (epoch + 1) % s.drag_refit_interval == 0:
            k, r2 = fit_drag_coefficients(model, train_data, train_idx, s, device, "network")
            model.params.set_drag(k)
            if verbose:
                print(f"[{stage}] drag coefficients refit to the network's aero wrench (R^2 {r2:.4f}): {_fmt_k(k)}")
        hist["train"].append((epoch, tot / nb, pred / nb))
        pv = dict(epoch=epoch, **model.params.as_dict(), mu=float(phys.mu.detach()))
        hist["params"].append(pv)
        msg = (f"[{stage}] ep {epoch:4d}  loss {tot / nb:.3e}  pred {pred / nb:.3e}  "
               + " ".join(f"{k}={v / nb:.2e}" for k, v in raw_acc.items())
               + (f"  k_f {pv['k_f']:.3e} k_rot {pv['k_rot']:.2e}" if stage == "aero" else f"  mu {pv['mu']:.3f}")
               + f"  {time.time() - t0:.1f}s")

        if verbose:
            print(msg, flush=True)

        if (epoch + 1) % s.val_interval == 0 or epoch == epochs - 1:
            vloss = validation_loss(model, phys, val_data, val_loss_idx, s, device, stage)
            kval = kstep_validation(model, val_data, val_idx, s, device, use_contact)
            fval = force_validation(model, val_data, val_idx, s, device)
            hist["val"].append(dict(epoch=epoch, loss=vloss, kstep=kval, forces=fval))
            score = vloss if s.best_metric == "val_loss" else kval.get("all", {}).get("pad_end_mm", float("inf"))
            if verbose:
                which = "contact-free windows" if stage == "aero" else "all windows"
                lines = [f"[{stage}] validation after epoch {epoch} ({which})",
                         f"    val loss {vloss:.3e}   (train pred {pred / nb:.3e})",
                         f"    pad error after {s.val_horizon} steps: " + "   ".join(
                             f"{r} {kval[r]['pad_end_mm']:.3f} mm" for r in ("free", "near", "contact", "all") if r in kval)
                         + "   (windows: " + ", ".join(f"{r} {kval[r]['n']}" for r in ("free", "near", "contact") if r in kval) + ")"]
                if fval:
                    lines += _format_forces(fval, s.impact_force_N)
                if score < best:
                    prev = "first" if best == float("inf") else f"was {best:.4g}"
                    lines.append(f"    * new best {s.best_metric} {score:.4g} ({prev}) -> saved {os.path.basename(stem)}_{stage}_best.pt")
                print("\n".join(lines), flush=True)
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                best_phys = copy.deepcopy(phys.state_dict())      # mu lives here, not in the model
                hist["best"] = dict(hist["params"][-1])           # coefficients at the kept epoch
                save_checkpoint(f"{stem}_{stage}_best.pt", model, phys, cfg, s,
                                {"stage": stage, "epoch": epoch, "history": history})

    if best_state is not None:
        model.load_state_dict(best_state)          # continue from the best validation point,
        phys.load_state_dict(best_phys)            # with the coefficients that went with it
    return model


# Window sets for both stages. Distances are the closest pad node to the wall
# over the whole window (ground truth, so the split is the same for every model).
#   tr_aero     stage 1: pad farther than aero_min_pad_dist throughout
#   tr_contact  stage 2: pad closer than contact_max_pad_dist at some frame
#   tr_touch    pad inside the gate distance at some frame (contact output scale, noise)
#   va_*        validation windows of length val_horizon (k-step error, force check)
#   vl_*        validation windows of the training unroll length (validation loss)
# The stage-2 k-step validation uses every window, so free-flight error stays visible.
def stage_windows(train_data, val_data, s):
    K, H = s.multistep, s.val_horizon
    tr_all = build_chain_index(train_data, s.h, K)
    va_all = build_chain_index(val_data, s.h, H, stride=s.val_stride)
    vl_all = build_chain_index(val_data, s.h, K, stride=s.val_stride)
    cmax = s.contact_max_pad_dist
    return dict(
        n_train_all=len(tr_all),
        tr_aero=filter_chain_index(train_data, tr_all, s.h, K, min_dist=s.aero_min_pad_dist),
        tr_contact=filter_chain_index(train_data, tr_all, s.h, K, max_dist=cmax),
        tr_touch=filter_chain_index(train_data, tr_all, s.h, K, max_dist=s.contact_d0),
        va_aero=filter_chain_index(val_data, va_all, s.h, H, min_dist=s.aero_min_pad_dist),
        va_all=va_all,
        vl_aero=filter_chain_index(val_data, vl_all, s.h, K, min_dist=s.aero_min_pad_dist),
        vl_contact=filter_chain_index(val_data, vl_all, s.h, K, max_dist=cmax))


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
    w = stage_windows(train_data, val_data, s)
    if verbose:
        print(f"windows: train {w['n_train_all']} total | stage 1 (pad > {100 * s.aero_min_pad_dist:.1f} cm "
              f"throughout) {len(w['tr_aero'])} | stage 2 (pad < "
              + (f"{100 * s.contact_max_pad_dist:.1f} cm" if s.contact_max_pad_dist else "any distance")
              + f" at some frame) {len(w['tr_contact'])}, of which touching {len(w['tr_touch'])}")
    history = {}

    if "aero" in stages:
        if not w["tr_aero"]:
            raise ValueError("no contact-free training windows for stage 1")
        fit_aero_stats(model, train_data, w["tr_aero"], s, device)
        if verbose:
            print(f"stage 1 scales: aero output {float(model.aero_scale):.3e}, "
                  f"loss {float(model.loss_scale_aero):.3e} (m/step^2)")
        if s.learn_drag_coeffs and s.drag_coeff_fit == "lstsq":
            k, r2 = fit_drag_coefficients(model, train_data, w["tr_aero"], s, device, "residual")
            model.params.set_drag(k)
            if verbose:
                print(f"initial drag coefficients from the measured motion residual (R^2 {r2:.4f}): {_fmt_k(k)}")
        train_stage("aero", model, phys, cfg, s, train_data, w["tr_aero"], val_data, w["va_aero"], w["vl_aero"],
                    stem, device, history, verbose)

    if "contact" in stages:
        if not w["tr_touch"]:
            raise ValueError("no contact training windows for stage 2")
        fit_contact_stats(model, train_data, w["tr_contact"], w["tr_touch"], s, device)
        if verbose:
            print(f"stage 2 scales: contact output {[f'{v:.3e}' for v in model.contact_scale.tolist()]}, "
                  f"loss {float(model.loss_scale_contact):.3e} (m/step^2)")
        train_stage("contact", model, phys, cfg, s, train_data, w["tr_contact"], val_data, w["va_all"],
                    w["vl_contact"], stem, device, history, verbose)

    _set_trainable(model, True)
    save_checkpoint(f"{stem}_final.pt", model, phys, cfg, s, {"stage": "final", "history": history})
    torch.save(history, f"{stem}_history.pt")
    return model, phys, history
