"""Define the graph neural network and rigid-body dynamics used to predict cube
motion. The network encodes node and edge features, passes messages between
surface nodes, and produces per-node contact outputs plus a body-level fluid
output. Supporting functions convert these outputs into contact forces and
fluid accelerations, apply a smooth contact gate and optional analytic drag,
and advance the cube's center of mass and orientation while preserving its
rigid shape. Training and rollout share these model and dynamics components;
positions use meters and accelerations use meters per recorded step squared.
"""

import torch
import torch.nn as nn

from force_data import BLOCK_WIDTH

# Solid cube inertia over mass: I/m = s^2 / 6, identical about every axis.
# Isotropic inertia => w x (I w) = I_scalar * (w x w) = 0 exactly.
I_OVER_M = (BLOCK_WIDTH ** 2) / 6.0


# hat operator: converts a rotation vector to a skew-symmetric matrix
def _hat(w):
    zeros = torch.zeros_like(w[..., 0])
    wx, wy, wz = w.unbind(-1)
    return torch.stack([
        torch.stack([zeros, -wz,    wy], dim=-1),
        torch.stack([wz,    zeros, -wx], dim=-1),
        torch.stack([-wy,   wx,    zeros], dim=-1),
    ], dim=-2)


# Exponential map for SO(3): rotation vector -> rotation matrix
def so3_exp(w):
    theta = w.norm(dim=-1, keepdim=True).unsqueeze(-1)          # (..., 1, 1)
    K = _hat(w)
    small = theta < 1e-4
    theta_safe = theta.clamp_min(1e-12)
    A = torch.where(small, 1.0 - theta ** 2 / 6.0, torch.sin(theta_safe) / theta_safe)
    B = torch.where(small, 0.5 - theta ** 2 / 24.0, (1.0 - torch.cos(theta_safe)) / theta_safe ** 2)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(K.shape)
    return eye + A * K + B * (K @ K)

# Logarithm map for SO(3): rotation matrix -> rotation vector
def so3_log(R):
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta).unsqueeze(-1)                  # (..., 1)
    vee = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1) * 0.5
    small = theta < 1e-4
    sin_safe = torch.sin(theta).clamp_min(1e-12)
    factor = torch.where(small, 1.0 + theta ** 2 / 6.0, theta / sin_safe)
    return factor * vee


#This is the main GNS layer used in the ForceGNSModel.
#Outputs updated node features and edge features after one round of message passing.
class GNSLayer(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()

        #Defines the edge MLP as a linear layer followed by a ReLU and another linear layer.
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        #Defines the node MLP as a linear layer followed by a ReLU and another linear layer.
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        #Layer normalization for edge and node features to stabilize training.
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(node_dim)

    #Forward pass for the GNS layer: updates node and edge features based on the current graph structure.
    def forward(self, x, edge_index, edge_attr):
        senders, receivers = edge_index[0], edge_index[1]
        edge_input = torch.cat([x[senders], x[receivers], edge_attr], dim=-1)
        edge_attr = edge_attr + self.edge_norm(self.edge_mlp(edge_input))
        node_agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device, dtype=x.dtype)
        node_agg.index_add_(0, receivers, edge_attr)
        x = x + self.node_norm(self.node_mlp(torch.cat([x, node_agg], dim=-1)))
        return x, edge_attr

#This is the main ForceGNS model that uses the GNSLayer for 
# message passing and has separate heads for contact and fluid predictions.
# outputs contact_raw (B, N, 4) and fluid_raw (B, 6)
class ForceGNSModel(nn.Module):
    """
    Encoder and processor are identical to the existing GNSModel. Two heads:

    CONTACT head (per node, 4 raw numbers):
        [0:3] tangential force raw (the wall-normal component is projected out
              downstream, so only the in-plane part survives)
        [3]   normal force magnitude raw -> softplus (can push, never pull)

    FLUID head (per graph, 6 raw numbers, from MEAN-POOLED node latents):
        [0:3] COM force  -> a_fluid (m/step^2)
        [3:6] COM torque -> alpha_fluid (rad/step^2)
        Final layer ZERO-INITIALIZED: with the analytic drag baseline on, the
        model starts exactly as "contact + gravity + calibrated drag" and the
        head learns only the residual (PIROM practice).

    forward(x, edge_index, edge_attr, num_graphs) with plain tensors (no PyG):
        returns (contact_raw (B, N, 4), fluid_raw (B, 6)).
    """

    N_CONTACT_OUT = 4
    N_FLUID_OUT = 6

    def __init__(self, node_in_dim, edge_in_dim, latent_dim, L, K,
                 normal_bias_init=-2.0):
        super().__init__()
        self.K = K

        #Defines the encoder as a series of linear layers with ReLU activations and layer normalization for both nodes and edges.
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        #Defines the processor as a list of GNS layers, each performing message passing and feature updates.
        self.processor_layers = nn.ModuleList([
            GNSLayer(latent_dim, latent_dim, latent_dim) for _ in range(L)
        ])
        #Defines the contact decoder as a series of linear layers with ReLU activations, outputting per-node contact predictions.
        self.decoder_contact = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.N_CONTACT_OUT)
        )
        #Defines the fluid head as a series of linear layers with ReLU activations, outputting per-graph fluid predictions.
        self.fluid_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.N_FLUID_OUT)
        )

        #Initializes the biases for the contact decoder and the weights and biases for the fluid head 
        #to ensure reasonable starting predictions.
        with torch.no_grad():
            # Normal-force channel starts small: softplus(-2) ~ 0.127, so four
            # resting contact nodes supply roughly the cube's weight at init
            # instead of several times it.
            self.decoder_contact[-1].bias[3] = normal_bias_init
            # Fluid head starts at exactly zero (baseline carries at init).
            self.fluid_head[-1].weight.zero_()
            self.fluid_head[-1].bias.zero_()

    #Forward pass for the ForceGNS model: encodes nodes and edges, applies message passing, 
    #and decodes contact and fluid predictions.
    def forward(self, x, edge_index, edge_attr, num_graphs):

        #Encode the node and edge features using the respective encoders.
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        #Apply message passing through the processor layers K times.
        for _ in range(self.K):
            for layer in self.processor_layers:
                x, edge_attr = layer(x, edge_index, edge_attr)

        #Reshape the node features to per-graph format and decode contact and fluid predictions.
        N = x.shape[0] // num_graphs
        x_g = x.reshape(num_graphs, N, -1)
        contact_raw = self.decoder_contact(x_g)                  # (B, N, 4)
        fluid_raw = self.fluid_head(x_g.mean(dim=1))             # (B, 6)

        #Return the raw contact and fluid predictions.
        return contact_raw, fluid_raw


# ======================================================================
# Output assembly
# ======================================================================

# calculate contact weight based on distance, threshold, and softness parameter
def contact_weight(dist, d0, tau):
    return torch.sigmoid((d0 - dist) / tau)

# Assemble the contact forces from the raw contact predictions, contact weights, wall normal, and scaling vector.
def assemble_contact_forces(contact_raw, c_w, wall_normal, scale_vec):
    """
    contact_raw: (B, N, 4) contact-head output
    c_w:         (B, N, 1) contact weight in [0, 1]
    wall_normal: (3,) wall normal (normalized here)
    scale_vec:   (3,) output scale in m/step^2, [s_xy, s_xy, s_z] from the
                 acceleration-target stats (equal x/y keeps the scaling
                 z-rotation equivariant, so the rotation augmentation is valid)

    Returns phi_c (B, N, 3), the per-node contact specific force.
    """
    n_hat = wall_normal / wall_normal.norm().clamp_min(1e-12)
    t_raw = contact_raw[..., 0:3]
    n_raw = contact_raw[..., 3:4]

    # Scale the tangential component before projecting it to ensure correct tangential force.
    t_scaled = t_raw * scale_vec
    t_vec = t_scaled - (t_scaled * n_hat).sum(-1, keepdim=True) * n_hat

    # Output scale along the wall normal.
    s_n = (scale_vec * n_hat).norm()
    n_mag = torch.nn.functional.softplus(n_raw) * s_n

    # Combine the tangential and normal components to get the total contact force.
    return c_w * (t_vec + n_mag * n_hat)


# Convert raw fluid predictions into linear and angular accelerations.
def fluid_wrench_from_raw(fluid_raw, scale_vec, ang_scale_vec):
    return fluid_raw[:, 0:3] * scale_vec, fluid_raw[:, 3:6] * ang_scale_vec

# Analytic quadratic drag acceleration as a per-step^2 COM acceleration.
# Used when baseline drag is turned on
def drag_accel_step(wind, v_com_step, dt, k_over_m):
    """Analytic quadratic body-drag baseline as a per-step^2 COM acceleration.
    Uses the k/m coefficient CALIBRATED FROM DATA (wind_error_analysis.py).
    Applied at the COM => zero torque, exactly like MuJoCo.
    wind: (B, 3) m/s;  v_com_step: (B, 3) m/step (= com_curr - com_prev)."""
    u = wind - v_com_step / dt
    return k_over_m * u.norm(dim=-1, keepdim=True) * u * (dt * dt)


# ======================================================================
# Rigid-body dynamics layer (Verlet on COM, Lie-group Verlet on rotation)
# ======================================================================

# Convert rigid-body state (COM position and rotation) into world-frame node positions.
def nodes_from_state(com, R, rest_nodes):
    return com.unsqueeze(1) + torch.einsum('bij,nj->bni', R, rest_nodes)


# Advance the cube's center-of-mass position and orientation by one time step.
# Uses contact, gravity, and any optional fluid accelerations to update the state.
def rigid_step(phi_contact, com_prev, com_curr, R_prev, R_curr, rest_nodes,
               g_step, extra_accel=None, extra_alpha=None):
    """
    One Newton-Euler step in per-step units. Differentiable end to end; no SVD
    anywhere (SVD gradients are ill-conditioned for a cube's isotropic corner
    set, which is why (COM, R) is carried as explicit state instead of fitting
    the rotation from node positions each step).

    phi_contact: (B, N, 3) per-node contact specific forces (m/step^2)
    extra_accel: optional (B, 3) COM acceleration (m/step^2) - analytic drag
                 plus the learned fluid force live here (both act at the COM,
                 so neither contributes torque)
    extra_alpha: optional (B, 3) angular acceleration (rad/step^2) - the
                 learned fluid torque lives here
    g_step:      (3,) gravity as m/step^2, i.e. [0, 0, -g] * dt^2

    Returns (com_next, R_next).
    """
        # Sum the translational accelerations and apply Verlet integration to the COM.
    a_com = phi_contact.sum(dim=1) + g_step
    if extra_accel is not None:
        a_com = a_com + extra_accel
    com_next = 2.0 * com_curr - com_prev + a_com

        # Compute contact torque from each node's lever arm and contact force.
    r = torch.einsum('bij,nj->bni', R_curr, rest_nodes)          # (B, N, 3)
    alpha = torch.cross(r, phi_contact, dim=-1).sum(dim=1) / I_OVER_M
    if extra_alpha is not None:
        alpha = alpha + extra_alpha

        # Update angular velocity from the previous rotation, then integrate rotation.
    w_prev = so3_log(R_curr @ R_prev.transpose(-1, -2))          # (B, 3) rad/step
    R_next = so3_exp(w_prev + alpha) @ R_curr
    return com_next, R_next
