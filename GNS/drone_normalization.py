"""Normalization for the drone force model.

WHAT GETS NORMALIZED, AND HOW
  1. Network inputs are z-scored, (x - mean) / std, with statistics from clean
     training windows (no input noise).
     - Body-frame features (aero GNN airspeeds) use per-column statistics.
       They do not change under the rotation augmentation (a rotation about
       gravity changes the world frame, not the body frame), so nothing special
       is needed.
     - World-frame vectors (contact GNN velocities, non-contact accelerations,
       edge displacements) use xy-POOLED statistics: x and y share one std and
       have mean 0, z keeps its own. Per-axis stats would scale x and y
       differently, so a rotated copy of the same sample would look like a
       different input, which undoes what the rotation augmentation teaches.
     - Scalars that do not depend on direction (|u|, distances, components
       along the wall normal) use plain per-column statistics.
     - Rotor speed and rotor-to-wall distance use physical scales instead of
       statistics: omega / omega_hover and clamp(d, 0, d_max) / d_max.
     - One-hot types and the wall normal are passed through unchanged.
  2. Network outputs are multiplied by a scale, so the last layer works with
     O(1) numbers.
     - Aero: the RMS of the residual acceleration (measured - thrust - gravity)
       on contact-free windows, i.e. the size of the aero signal itself.
     - Contact: the xy-pooled std of the residual after thrust, the trained
       aero model, and gravity, on windows WITH contact, i.e. the size of the
       contact signal. This is why contact stats are computed after stage 1.
  3. The loss is divided by one number per stage: the RMS node-acceleration
     error of the model with zero learned force over the same K-step unroll and
     the same windows that stage trains on (thrust + gravity in stage 1; thrust
     + trained aero + gravity in stage 2). So a loss of 1 means "no better than
     predicting zero learned force" (the test checks this: 1.000 and ~0.98).
     One number for all axes keeps a millimeter of error equally important in
     every direction. The std of total acceleration would be dominated by
     maneuvering instead.
  4. Input noise is a fraction of each stage's MEDIAN residual per step. Noise
     much larger than the signal a stage learns would bury it, and the median
     keeps rare large impacts from setting the level.

Stats live in buffers inside the model, so they are saved with the checkpoint
and the exact same normalization is used at rollout and in the MPC.
"""

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """z-score with stored statistics. `world_vec_cols` lists the starting
    column of each world-frame 3-vector, which gets xy-pooled statistics."""

    def __init__(self, dim, world_vec_cols=()):
        super().__init__()
        self.world_vec_cols = tuple(world_vec_cols)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def forward(self, x):
        return (x - self.mean) / self.std

    @torch.no_grad()
    def fit(self, x, min_std=1e-8):
        x = x.reshape(-1, x.shape[-1]).double()
        mean, std = x.mean(0), x.std(0)
        for s in self.world_vec_cols:
            pooled = torch.sqrt(0.5 * (x[:, s] ** 2 + x[:, s + 1] ** 2).mean())
            mean[s:s + 2] = 0.0
            std[s:s + 2] = pooled
        self.mean.copy_(mean.float())
        self.std.copy_(std.clamp_min(min_std).float())
        return self


# xy-pooled RMS of a set of world-frame vectors: (3,) = [s_xy, s_xy, s_z].
@torch.no_grad()
def xy_pooled_scale(v, min_scale=1e-10):
    v = v.reshape(-1, 3).double()
    s_xy = torch.sqrt(0.5 * (v[:, 0] ** 2 + v[:, 1] ** 2).mean())
    s_z = torch.sqrt((v[:, 2] ** 2).mean())
    return torch.stack([s_xy, s_xy, s_z]).float().clamp_min(min_scale)


# Scalar RMS over all components (for body-frame outputs).
@torch.no_grad()
def rms_scale(v, min_scale=1e-10):
    return torch.sqrt((v.double() ** 2).mean()).float().clamp_min(min_scale)
