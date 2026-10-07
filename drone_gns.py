"""The drone force model: known propulsion and gravity, an aero GNN, a contact
GNN, and a rigid-body integrator, combined into one differentiable step.

ONE STEP, IN ORDER
  1. Propulsion (known physics): T_j = k_f w_j^2 along each rotor axis,
     yaw reaction k_m w_j^2. k_f, k_m are learnable with a thrust-stand prior.
  2. Aero GNN: 9 nodes (COM, 4 rotors, rod nodes, 1 pad node). Inputs are
     body-frame local airspeeds (+ rotor speed and wall distance at rotors).
     Each node outputs a local force; torque comes from lever arms. It never
     receives contact information.
  3. Non-contact acceleration: thrust + aero + gravity turned into the
     acceleration of every contact-graph node, as if the wall were absent.
  4. Contact GNN: COM hub + 9 pad nodes. Inputs are velocity history, wall
     distance, the non-contact acceleration, the wall normal, and node type.
     Outputs per node: tangential (projected into the wall plane) + softplus
     normal, multiplied by a distance gate that is computed, not predicted.
  5. Rigid-body integrator: Verlet on the COM, angular-momentum update on SO(3).

Units: positions in m; everything the integrator consumes is a specific force
or torque (per unit mass) in per-recorded-step^2 units, as in the cube code.
Aero inputs use m/s because the drag laws are written in m/s.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from force_gns import contact_weight, so3_log, nodes_from_state
from drone_config import rotor_and_inertia_tensors, build_aero_graph, build_contact_graph, N_AERO_TYPES
from drone_dynamics import rotor_wrench_body, node_accel_from_wrench, rigid_step_drone
from drone_normalization import Normalizer


# ======================================================================
# Learnable physical coefficients (log space, like mu)
# ======================================================================
class DronePhysicsParams(nn.Module):

    def __init__(self, k_f, k_m, k_rot, k_body, k_rod, k_pad,
                 learn_thrust=True, learn_drag=True,
                 kf_prior_rel_std=0.05, km_prior_rel_std=0.15):
        super().__init__()
        for name, v, learn in (("kf", k_f, learn_thrust), ("km", k_m, learn_thrust),
                               ("krot", k_rot, learn_drag), ("kbody", k_body, learn_drag),
                               ("krod", k_rod, learn_drag), ("kpad", k_pad, learn_drag)):
            t = torch.log(torch.tensor(float(v)))
            if learn:
                setattr(self, "log_" + name, nn.Parameter(t))
            else:
                self.register_buffer("log_" + name, t)
        self.register_buffer("log_kf0", torch.log(torch.tensor(float(k_f))))
        self.register_buffer("log_km0", torch.log(torch.tensor(float(k_m))))
        self.kf_sig = math.log1p(kf_prior_rel_std)
        self.km_sig = math.log1p(km_prior_rel_std)

    @property
    def k_f(self):
        return torch.exp(self.log_kf)

    @property
    def k_m(self):
        return torch.exp(self.log_km)

    @property
    def k_rot(self):
        return torch.exp(self.log_krot)

    @property
    def k_body(self):
        return torch.exp(self.log_kbody)

    @property
    def k_rod(self):
        return torch.exp(self.log_krod)

    @property
    def k_pad(self):
        return torch.exp(self.log_kpad)

    DRAG_NAMES = ("k_rot", "k_body", "k_rod", "k_pad")

    # The four drag coefficients as a vector, in drag_basis() column order.
    def drag_vector(self):
        return torch.stack([self.k_rot, self.k_body, self.k_rod, self.k_pad])

    @torch.no_grad()
    def set_drag(self, k, floor=1e-9):
        for name, v in zip(("log_krot", "log_kbody", "log_krod", "log_kpad"), k):
            getattr(self, name).copy_(torch.log(torch.as_tensor(max(float(v), floor))))

    def drag_parameters(self):
        return [getattr(self, n) for n in ("log_krot", "log_kbody", "log_krod", "log_kpad")
                if isinstance(getattr(self, n), nn.Parameter)]

    def thrust_parameters(self):
        return [getattr(self, n) for n in ("log_kf", "log_km") if isinstance(getattr(self, n), nn.Parameter)]

    # Values for logging.
    def as_dict(self):
        with torch.no_grad():
            return {k: float(getattr(self, k)) for k in ("k_f", "k_m", "k_rot", "k_body", "k_rod", "k_pad")}

    # Gaussian prior on k_f and k_m in log space (thrust-stand values).
    def prior_loss(self):
        return (((self.log_kf - self.log_kf0) / self.kf_sig).pow(2)
                + ((self.log_km - self.log_km0) / self.km_sig).pow(2))


# ======================================================================
# The two message-passing networks
# ======================================================================
# MLP with n_layers Linear layers (ReLU between them), e.g. n_layers=2 is
# Linear -> ReLU -> Linear, the same as the cube model.
def mlp(in_dim, width, out_dim, n_layers, layernorm=False):
    layers, d = [], in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, out_dim))
    if layernorm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class MessagePassingStep(nn.Module):
    """One round of message passing (same update as the cube's GNSLayer, with
    configurable MLP depth): each edge is updated from its two end nodes, each
    node from the sum of its incoming edges, both with residual + LayerNorm."""

    def __init__(self, latent, mlp_layers):
        super().__init__()
        self.edge_mlp = mlp(3 * latent, latent, latent, mlp_layers)
        self.node_mlp = mlp(2 * latent, latent, latent, mlp_layers)
        self.edge_norm = nn.LayerNorm(latent)
        self.node_norm = nn.LayerNorm(latent)

    def forward(self, x, edge_index, edge_attr):
        senders, receivers = edge_index[0], edge_index[1]
        edge_attr = edge_attr + self.edge_norm(self.edge_mlp(torch.cat([x[senders], x[receivers], edge_attr], -1)))
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, receivers, edge_attr)
        return x + self.node_norm(self.node_mlp(torch.cat([x, agg], -1))), edge_attr


class _GNN(nn.Module):
    """Encoder -> message passing -> per-node decoder.

    msg_passing_steps   rounds of message passing, each with its OWN weights
    msg_passing_repeats how many times that whole sequence is run again with the
                        SAME weights
    Total rounds = steps x repeats; information travels one edge per round.
    4 steps x 1 repeat and 2 steps x 2 repeats both give 4 rounds (same reach),
    but the first has 4 sets of weights and the second 2, reused.
    mlp_layers          Linear layers in every MLP (encoders, edge/node updates, decoder)"""

    def __init__(self, node_in, edge_in, latent, steps, repeats, mlp_layers, n_out, zero_init_out=False):
        super().__init__()
        self.repeats = repeats
        self.node_encoder = mlp(node_in, latent, latent, mlp_layers, layernorm=True)
        self.edge_encoder = mlp(edge_in, latent, latent, mlp_layers, layernorm=True)
        self.processor_layers = nn.ModuleList([MessagePassingStep(latent, mlp_layers) for _ in range(steps)])
        self.decoder = mlp(latent, latent, n_out, mlp_layers)
        if zero_init_out:
            with torch.no_grad():
                self.decoder[-1].weight.zero_()
                self.decoder[-1].bias.zero_()

    def forward(self, x, edge_index, edge_attr, num_graphs):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for _ in range(self.repeats):
            for layer in self.processor_layers:
                x, edge_attr = layer(x, edge_index, edge_attr)
        return self.decoder(x.reshape(num_graphs, -1, x.shape[-1]))


class AeroGNN(_GNN):
    """Outputs 4 raw numbers per node. Rotors: [in-plane drag (3, axis component
    removed), axial correction]. Other nodes: [drag (3), unused]. The last layer
    starts at zero, so training starts from thrust + gravity only."""

    N_IN = 4 + 2 + N_AERO_TYPES

    def __init__(self, latent, steps, repeats, mlp_layers):
        super().__init__(self.N_IN, 4, latent, steps, repeats, mlp_layers, 4, zero_init_out=True)


class ContactGNN(_GNN):
    """Outputs 4 raw numbers per node: tangential (3) and normal magnitude (1).
    The last layer starts at zero weights with a negative normal bias, so stage 2
    starts from (almost) no contact force: softplus(-4) ~ 0.02 of the output scale
    per touching node, and no friction. With 9 pad nodes summing, a random start
    would apply several times the contact scale in random directions."""

    def __init__(self, node_in, latent, steps, repeats, mlp_layers, normal_bias_init=-4.0):
        super().__init__(node_in, 4, latent, steps, repeats, mlp_layers, 4, zero_init_out=True)
        with torch.no_grad():
            self.decoder[-1].bias[3] = normal_bias_init


# Contact force per node: tangential part projected into the wall plane,
# softplus normal part along n, times the gate. n_hat is per sample (B,1,3).
def assemble_contact_forces(contact_raw, c_w, n_hat, scale_vec):
    t_scaled = contact_raw[..., 0:3] * scale_vec
    t_vec = t_scaled - (t_scaled * n_hat).sum(-1, keepdim=True) * n_hat
    s_n = (scale_vec * n_hat).norm(dim=-1, keepdim=True)
    return c_w * (t_vec + Fnn.softplus(contact_raw[..., 3:4]) * s_n * n_hat)


def _cross(a, b):
    return torch.linalg.cross(a, b, dim=-1)


# ======================================================================
# Full model
# ======================================================================
class DroneForceModel(nn.Module):
    """Holds geometry, normalization, and every learned part as one module, so
    a single state_dict saves everything needed for rollout and the MPC."""

    def __init__(self, cfg, s):
        super().__init__()
        self.cfg, self.h = cfg, s.h
        self.mass, self.dt = cfg.mass, cfg.dt
        self.contact_d0, self.contact_tau = s.contact_d0, s.contact_tau
        self.dist_clamp = (-0.05, 0.5)
        self.aero_dist_max = s.aero_dist_max

        # Rotor layout and inertia.
        for k, v in rotor_and_inertia_tensors(cfg).items():
            self.register_buffer(k, v)
        M = self.rotor_pos.shape[0]
        self.register_buffer("g_step", torch.tensor([0.0, 0.0, -cfg.gravity]) * cfg.dt ** 2)
        self.omega_hover = math.sqrt(cfg.mass * cfg.gravity / (M * cfg.k_f))

        # Aero graph.
        ga = build_aero_graph(cfg)
        Na = ga["rest_nodes"].shape[0]
        self.aero_pad_idx = ga["pad_idx"]
        self.register_buffer("aero_rest", ga["rest_nodes"])
        self.register_buffer("aero_onehot", ga["onehot"])
        self.register_buffer("aero_edges", ga["edge_index"])
        sel = torch.zeros(M, Na)                                  # rotor m -> aero node
        sel[torch.arange(M), torch.tensor(ga["rotor_idx"])] = 1.0
        self.register_buffer("rotor_sel", sel)
        node_axis = sel.T @ ga["rotor_axis"]                      # rotor axis at rotor nodes, 0 elsewhere
        self.register_buffer("aero_node_axis", node_axis)
        t = ga["node_type"]
        for name, k in (("com", 0), ("rotor", 1), ("rod", 2), ("pad", 3)):
            self.register_buffer(f"mask_{name}", (t == k).float().view(1, -1, 1))
        self.register_buffer("rod_axis", ga["rod_axis"])
        src, dst = ga["edge_index"]
        d = ga["rest_nodes"][src] - ga["rest_nodes"][dst]
        L_ref = ga["rest_nodes"].norm(dim=1).max()
        self.register_buffer("aero_edge_attr", torch.cat([d, d.norm(dim=1, keepdim=True)], 1) / L_ref)

        # Contact graph.
        gc = build_contact_graph(cfg)
        self.register_buffer("contact_rest", gc["rest_nodes"])
        self.register_buffer("contact_onehot", gc["onehot"])
        self.register_buffer("contact_edges", gc["edge_index"])
        self.register_buffer("contact_mask", gc["contact_mask"].float().view(1, -1, 1))
        self.register_buffer("pad_rest", gc["rest_nodes"][gc["contact_mask"]])

        # Normalization (fit later; see drone_normalization.py).
        h = s.h
        self.aero_in = Normalizer(4)                                        # body frame: u (3), |u|
        n_dyn = 3 * h + 3 + 3
        vec_cols = [3 * k for k in range(h)] + [3 * h]                      # velocity history, a_nc
        self.contact_in = Normalizer(n_dyn, world_vec_cols=vec_cols)
        self.contact_edge = Normalizer(4, world_vec_cols=[0])
        self.register_buffer("aero_scale", torch.tensor(1.0))
        self.register_buffer("contact_scale", torch.ones(3))
        self.register_buffer("loss_scale_aero", torch.tensor(1.0))
        self.register_buffer("loss_scale_contact", torch.tensor(1.0))
        self.register_buffer("noise_ref_aero", torch.tensor(0.0))      # median residual, per axis
        self.register_buffer("noise_ref_contact", torch.tensor(0.0))

        # Learned parts.
        self.params = DronePhysicsParams(cfg.k_f * s.k_f_scale, cfg.k_m * s.k_m_scale,
                                         s.k_rot_init, s.k_body_init,
                                         s.k_rod_init, s.k_pad_init,
                                         learn_thrust=s.learn_thrust_coeffs, learn_drag=s.learn_drag_coeffs)
        self.aero = AeroGNN(s.aero_latent_dim, s.aero_msg_passing_steps, s.aero_msg_passing_repeats,
                            s.aero_mlp_layers)
        n_static = 3 + gc["onehot"].shape[1]
        self.contact = ContactGNN(n_dyn + n_static, s.contact_latent_dim, s.contact_msg_passing_steps,
                                  s.contact_msg_passing_repeats, s.contact_mlp_layers)
        self._ei = {}

    def batched_edges(self, name, B):
        key = (name, B, self.aero_rest.device)
        if key not in self._ei:
            ei = getattr(self, name)
            n = (self.aero_rest if name == "aero_edges" else self.contact_rest).shape[0]
            self._ei[key] = torch.cat([ei + b * n for b in range(B)], dim=1)
        return self._ei[key]

    # ---------------- known physics ----------------
    def thrust_wrench(self, omega, R):
        F_b, T_b = rotor_wrench_body(omega, self.rotor_pos, self.rotor_axis, self.spin_dir,
                                     self.params.k_f, self.params.k_m)
        sc = self.dt ** 2 / self.mass
        return (R @ F_b.unsqueeze(-1)).squeeze(-1) * sc, (R @ T_b.unsqueeze(-1)).squeeze(-1) * sc

    # ---------------- aero ----------------
    # Local relative airspeed at every aero node, body frame, m/s, plus lever arms.
    def aero_airspeed(self, com_prev, com_curr, R_prev, R_curr, wind):
        v = (com_curr - com_prev) / self.dt
        w = so3_log(R_curr @ R_prev.transpose(-1, -2)) / self.dt
        r = torch.einsum('bij,nj->bni', R_curr, self.aero_rest)
        vel = v.unsqueeze(1) + _cross(w.unsqueeze(1).expand_as(r), r)
        u_b = torch.einsum('bji,bnj->bni', R_curr, wind.unsqueeze(1) - vel)
        return u_b, r

    def aero_forces(self, com_prev, com_curr, R_prev, R_curr, omega, wind, wall_n, wall_c):
        B = com_curr.shape[0]
        u_b, r = self.aero_airspeed(com_prev, com_curr, R_prev, R_curr, wind)
        un = u_b.norm(dim=-1, keepdim=True)

        # Rotor-only features, physically scaled; zero at non-rotor nodes.
        p_rot = com_curr.unsqueeze(1) + torch.einsum('mn,bnk->bmk', self.rotor_sel, r)
        d_rot = ((p_rot - wall_c.unsqueeze(1)) * wall_n.unsqueeze(1)).sum(-1)
        rot_feat = torch.stack([omega / self.omega_hover - 1.0,
                                d_rot.clamp(0.0, self.aero_dist_max) / self.aero_dist_max], -1)
        rot_feat = torch.einsum('bmk,mn->bnk', rot_feat, self.rotor_sel)

        x = torch.cat([self.aero_in(torch.cat([u_b, un], -1)), rot_feat,
                       self.aero_onehot.expand(B, -1, -1)], -1)
        raw = self.aero(x.reshape(-1, x.shape[-1]), self.batched_edges("aero_edges", B),
                        self.aero_edge_attr.repeat(B, 1), B)

        # Rotor structure: drag in the disk plane plus a correction along the axis.
        # The axial correction is always on and penalized toward zero (w_axial).
        # In MuJoCo thrust is exactly k_f w^2, so a correction that grows there is
        # leaked contact or error: a built-in diagnostic.
        a = self.aero_node_axis
        f_b = raw[..., 0:3] * self.aero_scale
        f_b = f_b - self.mask_rotor * (f_b * a).sum(-1, keepdim=True) * a
        axial = raw[..., 3] * self.aero_scale * self.mask_rotor.squeeze(-1)
        f_drag = f_b
        f_b = f_b + axial.unsqueeze(-1) * a

        f_w = torch.einsum('bij,bnj->bni', R_curr, f_b)
        F = f_w.sum(1)
        tau = _cross(r, f_w).sum(1)

        # Anchor laws per node type (body frame): law = basis @ [k_rot, k_body, k_rod, k_pad].
        basis = self.drag_basis(u_b, omega)
        law = basis @ self.params.drag_vector()
        return dict(F=F, tau=tau, f_body=f_b, f_drag=f_drag, law_body=law, axial=axial, basis=basis)

    # Per-node drag laws per unit coefficient, body frame, per step^2: (B, Na, 3, 4),
    # columns [rotor drag w_j u_perp, body |u| u, rod |u_perp| u_perp, pad |u| u].
    # The laws are linear in the coefficients, which is what lets them be fit
    # exactly by least squares (see train_drone_gns.fit_drag_coefficients).
    def drag_basis(self, u_b, omega):
        a = self.aero_node_axis
        un = u_b.norm(dim=-1, keepdim=True)
        om_node = (omega @ self.rotor_sel).unsqueeze(-1)                     # (B, Na, 1)
        u_rot = u_b - (u_b * a).sum(-1, keepdim=True) * a
        u_rod = u_b - (u_b * self.rod_axis).sum(-1, keepdim=True) * self.rod_axis
        cols = [self.mask_rotor * om_node * u_rot,
                self.mask_com * un * u_b,
                self.mask_rod * u_rod.norm(dim=-1, keepdim=True) * u_rod,
                self.mask_pad * un * u_b]
        return torch.stack(cols, -1) * self.dt ** 2

    # ---------------- contact ----------------
    # Raw contact-GNN inputs. Dynamic features are normalized; static are not.
    def contact_features(self, hist, a_nc, wall_n, wall_c):
        B, N, _ = hist[-1].shape
        v_list = [hist[-(k + 1)] - hist[-(k + 2)] for k in range(len(hist) - 1)]
        n_ = wall_n.unsqueeze(1)
        dist = ((hist[-1] - wall_c.unsqueeze(1)) * n_).sum(-1, keepdim=True).clamp(*self.dist_clamp)
        dyn = torch.cat(v_list + [a_nc, (a_nc * n_).sum(-1, keepdim=True),
                                  (v_list[0] * n_).sum(-1, keepdim=True), dist], -1)
        static = torch.cat([n_.expand(B, N, 3), self.contact_onehot.expand(B, -1, -1)], -1)
        x_flat = hist[-1].reshape(B * N, 3)
        ei = self.batched_edges("contact_edges", B)
        d = x_flat[ei[0]] - x_flat[ei[1]]
        return dyn.reshape(B * N, -1), static.reshape(B * N, -1), torch.cat([d, d.norm(dim=-1, keepdim=True)], -1)

    def contact_forces(self, hist, a_nc, wall_n, wall_c):
        B = hist[-1].shape[0]
        dyn, static, e = self.contact_features(hist, a_nc, wall_n, wall_c)
        x = torch.cat([self.contact_in(dyn), static], -1)
        raw = self.contact(x, self.batched_edges("contact_edges", B), self.contact_edge(e), B)
        n_ = wall_n.unsqueeze(1)
        dist = ((hist[-1] - wall_c.unsqueeze(1)) * n_).sum(-1, keepdim=True)
        c_w = contact_weight(dist, self.contact_d0, self.contact_tau) * self.contact_mask
        return assemble_contact_forces(raw, c_w, n_, self.contact_scale), c_w

    # ---------------- non-contact part of a step ----------------
    # Thrust + aero, and (if contact will run) the contact-graph node history and
    # each node's non-contact acceleration. Shared by step() and the stats pass.
    def noncontact(self, com_hist, R_hist, omega, wind, wall_n, wall_c, need_contact_inputs=True):
        com_prev, com_curr = com_hist[-2], com_hist[-1]
        R_prev, R_curr = R_hist[-2], R_hist[-1]
        f_thr, tau_thr = self.thrust_wrench(omega, R_curr)
        aero = self.aero_forces(com_prev, com_curr, R_prev, R_curr, omega, wind, wall_n, wall_c)
        out = dict(f_thr=f_thr, tau_thr=tau_thr, aero=aero,
                   f_nc=f_thr + aero["F"], tau_nc=tau_thr + aero["tau"])
        if need_contact_inputs:
            w_step = so3_log(R_curr @ R_prev.transpose(-1, -2))
            r_c = torch.einsum('bij,nj->bni', R_curr, self.contact_rest)
            out["r_c"] = r_c
            out["a_nc"] = node_accel_from_wrench(out["f_nc"] + self.g_step, out["tau_nc"], w_step,
                                                 R_curr, self.J_body, r_c)
            out["hist"] = [nodes_from_state(c, R, self.contact_rest) for c, R in zip(com_hist, R_hist)]
        return out

    # ---------------- one step ----------------
    def step(self, com_hist, R_hist, omega, wind, wall_n, wall_c,
             use_contact=True, detach_nc=True):
        """
        com_hist, R_hist: lists of h+1 COM positions (B,3) / attitudes (B,3,3), oldest first
        omega (B,M) rad/s, wind (B,3) m/s, wall_n / wall_c (B,3)
        detach_nc: stop contact-loss gradients flowing into thrust/aero through the
                   non-contact acceleration (training). Set False to differentiate
                   the contact force w.r.t. rotor speed (MPC).
        Returns (com_next, R_next, aux).
        """
        aux = self.noncontact(com_hist, R_hist, omega, wind, wall_n, wall_c, need_contact_inputs=use_contact)
        f_nc, tau_nc = aux["f_nc"], aux["tau_nc"]
        f_c = torch.zeros_like(f_nc)
        tau_c = torch.zeros_like(tau_nc)
        if use_contact:
            hist, a_nc = aux["hist"], aux["a_nc"]
            phi_c, c_w = self.contact_forces(hist, a_nc.detach() if detach_nc else a_nc, wall_n, wall_c)
            f_c = phi_c.sum(1)
            tau_c = _cross(aux["r_c"], phi_c).sum(1)
            aux.update(phi_c=phi_c, c_w=c_w, v_node=hist[-1] - hist[-2])
        aux.update(f_c=f_c, tau_c=tau_c)

        com_next, R_next = rigid_step_drone(f_nc + f_c, tau_nc + tau_c, com_hist[-2], com_hist[-1],
                                            R_hist[-2], R_hist[-1], self.J_body, self.g_step)
        return com_next, R_next, aux

    # Per-step specific force -> newtons.
    def to_newtons(self, x):
        return self.mass * x / self.dt ** 2
