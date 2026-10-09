"""Load drone trajectories and turn them into training batches.

TRAJECTORY FILE FORMAT (one .pt per trajectory, a DICT, not a list - the
list-index format is what broke view_mojoco_traj.py before):
    "pos"          (T,3)  mocap-origin position, m
    "quat"         (T,4)  mocap orientation, wxyz
    "rotor_speed"  (T,M)  measured rotor speeds, rad/s (pass rotor_speed_in_rpm=True for RPM)
    "meta"["rotor_speed_hold"]  "zoh"     rotor_speed[t] is held over frame t -> t+1
                                          (MuJoCo ctrl held across substeps)
                                "instant" rotor_speed[t] is sampled AT frame t
                                          (hardware telemetry of a lagged motor)
                                "aligned" rotor_speed[t] is already the Verlet-consistent
                                          (triangle-weighted) value; used as is
                                          (mujoco_drone_generator.py writes this)
    "wall_normal"  (3,)   unit normal pointing OUT of the wall, into free space
    "wall_point"   (3,)   any point on the wall plane
    "wind"         (3,) or (T,3)  m/s, optional (zeros if absent)
    "meta"         dict, optional: {"dt", "gravity", "rotor_speed_hold", "mass",
                   "inertia", "com_offset", "k_f", "k_m"}; any of those present
                   are checked against DroneConfig
    labels, optional, validation only (never inputs):
    "F_contact" (T,3) N, "tau_contact" (T,3) N m, "F_aero" (T,3) N

The loader converts the mocap pose to the COM pose using cfg.com_offset, so an
offset error shows up here, once, instead of as a fake centripetal force.

FRAME ALIGNMENT. Verlet uses the second difference com[t+1] - 2 com[t] + com[t-1],
which spans TWO frame intervals. With rotor speeds held per frame, that equals
dt^2 * (a[t-1] + a[t]) / 2, not dt^2 * a[t]. Thrust and yaw moment are linear in
omega^2, so the loader replaces omega[t] with sqrt((omega[t-1]^2 + omega[t]^2) / 2)
for "zoh" data, which makes the thrust term exact. Skipping this puts a
one-half-step lag on every thrust change, and the network learns it as contact.
"""

import math
import os

import torch

from force_data import quat_wxyz_to_R
from force_gns import so3_exp, nodes_from_state


# Loads one trajectory dict and converts it to the COM state the model integrates.
def load_drone_trajectory(path, cfg, rotor_speed_in_rpm=False, rotor_speed_hold=None):
    raw = torch.load(path, weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a dict trajectory (see drone_data.py header)")

    # Positions stay float64 until each window is re-centered (see iterate_drone_chains).
    pos = torch.as_tensor(raw["pos"], dtype=torch.float64)
    R = quat_wxyz_to_R(torch.as_tensor(raw["quat"], dtype=torch.float64))
    com = pos + R @ torch.tensor(cfg.com_offset, dtype=torch.float64)
    R = R.float()

    omega = torch.as_tensor(raw["rotor_speed"], dtype=torch.float32)
    if rotor_speed_in_rpm:
        omega = omega * (2.0 * math.pi / 60.0)

    # Frame alignment (see header). Explicit argument wins over the file's meta.
    hold = rotor_speed_hold or raw.get("meta", {}).get("rotor_speed_hold", "instant")
    if hold == "zoh":
        omega = torch.cat([omega[:1], torch.sqrt(0.5 * (omega[:-1] ** 2 + omega[1:] ** 2))], dim=0)
    elif hold not in ("instant", "aligned"):
        raise ValueError(f"rotor_speed_hold must be 'zoh', 'instant' or 'aligned', got {hold!r}")

    T = com.shape[0]
    wind = torch.as_tensor(raw.get("wind", torch.zeros(3)), dtype=torch.float32)
    if wind.dim() == 1:
        wind = wind.expand(T, 3).clone()

    n = torch.as_tensor(raw["wall_normal"], dtype=torch.float32)
    d = {"com": com, "R": R, "omega": omega, "wind": wind,
         "wall_n": n / n.norm(), "wall_c": torch.as_tensor(raw["wall_point"], dtype=torch.float64),
         "T": T}
    for key in ("F_contact", "tau_contact", "F_aero", "F_pad", "phase"):
        if key in raw:
            d[key] = torch.as_tensor(raw[key], dtype=torch.float32)
    return d, raw.get("meta", {})


# Loads a range of trajectories and checks dt / gravity against the config, so a
# mismatch fails loudly instead of being absorbed into the learned forces.
def build_drone_dataset(traj_range, folder, cfg, rotor_speed_in_rpm=False, rotor_speed_hold=None):
    dataset, meta0 = [], None
    for i in traj_range:
        d, meta = load_drone_trajectory(os.path.join(folder, f"{i}.pt"), cfg,
                                         rotor_speed_in_rpm, rotor_speed_hold)
        if meta0 is None:
            meta0 = meta
        if d["omega"].shape[1] != len(cfg.rotor_pos):
            raise ValueError(f"traj {i}: {d['omega'].shape[1]} rotor speeds, config has {len(cfg.rotor_pos)} rotors")
        dataset.append(d)

    # Physical constants: a mismatch here would be silently absorbed into the
    # learned forces, so it is an error, not a warning.
    for key in ("dt", "gravity", "mass", "inertia", "com_offset", "k_f", "k_m"):
        if meta0 and key in meta0:
            a = torch.as_tensor(meta0[key], dtype=torch.float64)
            b = torch.as_tensor(getattr(cfg, key), dtype=torch.float64)
            if a.shape != b.shape or not torch.allclose(a, b, rtol=1e-4, atol=1e-7):
                raise ValueError(f"data {key}={meta0[key]} but DroneConfig.{key}={getattr(cfg, key)}")
    return dataset, (meta0 or {})


# Every valid (trajectory, start frame) for a window of h+1 inputs and K targets.
def build_chain_index(dataset, h, multistep, stride=1):
    span = h + 1 + multistep
    return [(ti, s) for ti, d in enumerate(dataset) for s in range(0, d["T"] - span + 1, stride)]


# Stores, per frame, the smallest signed distance from any pad node to the wall
# (ground truth). Used only to sort windows into contact-free / contact, which
# works the same on hardware (it needs poses and the wall, not force labels).
def annotate_pad_distance(dataset, pad_rest_nodes):
    for d in dataset:
        nodes = nodes_from_state(d["com"] - d["wall_c"], d["R"].double(), pad_rest_nodes.double())
        d["pad_dmin"] = (nodes * d["wall_n"].double()).sum(-1).min(dim=1).values.float()
    return dataset


# Keeps the windows whose closest pad approach over the whole window (inputs and
# targets) is below max_dist and/or above min_dist. None = no bound.
def filter_chain_index(dataset, chain_index, h, multistep, max_dist=None, min_dist=None):
    span = h + 1 + multistep
    out = []
    for ti, s in chain_index:
        dmin = float(dataset[ti]["pad_dmin"][s:s + span].min())
        if (max_dist is None or dmin < max_dist) and (min_dist is None or dmin > min_dist):
            out.append((ti, s))
    return out


# Splits a chain index by the closest pad approach over the whole window
# (inputs and targets): "free" if every frame stays farther than free_dist,
# "contact" if any frame comes inside contact_dist.
def split_chain_index(dataset, chain_index, h, multistep, free_dist, contact_dist):
    span = h + 1 + multistep
    free, contact = [], []
    for ti, s in chain_index:
        dmin = float(dataset[ti]["pad_dmin"][s:s + span].min())
        if dmin > free_dist:
            free.append((ti, s))
        elif dmin < contact_dist:
            contact.append((ti, s))
    return free, contact


# Yields batches of chains: an (h+1)-frame input window, K target frames, and the
# rotor speeds / wind for each of the K steps (teacher-forced from the logs).
# Every window is RE-CENTERED on its current COM (in float64) before casting to
# float32. Nothing in the model depends on absolute position, so this changes no
# physics, but it matters numerically: at ~1.5 m from the origin float32 resolves
# ~1e-7 m, and the second differences the model learns from are ~1e-5 m/step^2.
# Force labels (MuJoCo only) come along for validation when every trajectory has them.
# Noise goes on the INPUT window only. COM and rotation noise are set separately:
# at the pad, rotation noise is multiplied by the rod length.
# Each dataset is packed once per device into concatenated tensors (positions kept
# float64), and every batch is one vectorized gather from them. The old per-sample
# Python loop cost ~0.05 ms per window, which made the batch builder the bottleneck
# at large batch sizes. Batches are identical to the loop version, sample for sample.
_PACK_CACHE = {}


def _packed(dataset, device, label_keys):
    key = (id(dataset), str(device), tuple(label_keys))
    hit = _PACK_CACHE.get(key)
    if hit is not None and hit[0] is dataset and hit[1] == len(dataset):
        return hit[2]
    T = torch.tensor([d["T"] for d in dataset])
    p = dict(offset=(torch.cumsum(T, 0) - T).to(device),
             com=torch.cat([d["com"] for d in dataset]).to(device, torch.float64),
             R=torch.cat([d["R"] for d in dataset]).to(device),
             omega=torch.cat([d["omega"] for d in dataset]).to(device),
             wind=torch.cat([d["wind"] for d in dataset]).to(device),
             wall_n=torch.stack([d["wall_n"] for d in dataset]).to(device),
             wall_c=torch.stack([d["wall_c"] for d in dataset]).to(device, torch.float64),
             **{k: torch.cat([d[k] for d in dataset]).to(device) for k in label_keys})
    _PACK_CACHE[key] = (dataset, len(dataset), p)       # holding the list keeps its id from being reused
    return p


def iterate_drone_chains(dataset, chain_index, batch_size, h, multistep, device,
                         com_noise=0.0, rot_noise=0.0, shuffle=True):
    order = torch.randperm(len(chain_index)) if shuffle else torch.arange(len(chain_index))
    label_keys = [k for k in ("F_contact", "F_aero", "tau_contact") if all(k in d for d in dataset)]
    p = _packed(dataset, device, label_keys)
    idx = torch.as_tensor(chain_index, dtype=torch.long).reshape(-1, 2)
    ar_in = torch.arange(h + 1, device=device)
    ar_k = torch.arange(multistep, device=device)

    for start in range(0, len(chain_index), batch_size):
        sel = idx[order[start:start + batch_size]].to(device)
        B = sel.shape[0]
        ti = sel[:, 0]
        base = p["offset"][ti] + sel[:, 1]                       # global frame of window start
        f_in = base[:, None] + ar_in                             # (B, h+1) input frames
        f_tg = base[:, None] + h + 1 + ar_k                      # (B, K)   target frames
        f_u = base[:, None] + h + ar_k                           # (B, K)   inputs for step t -> t+1
        anchor = p["com"][base + h]                              # (B, 3) float64
        batch = dict(
            com_win=(p["com"][f_in] - anchor[:, None]).float(), R_win=p["R"][f_in],
            tgt_com=(p["com"][f_tg] - anchor[:, None]).float(), tgt_R=p["R"][f_tg],
            omega=p["omega"][f_u], wind=p["wind"][f_u],
            wall_n=p["wall_n"][ti], wall_c=(p["wall_c"][ti] - anchor).float(),
            **{k: p[k][f_u] for k in label_keys})
        batch["B"] = B

        if com_noise > 0:
            vel_noise = torch.randn(B, h, 3, device=device) * com_noise
            batch["com_win"][:, 1:] += torch.cumsum(vel_noise, dim=1)
        if rot_noise > 0:
            w_cum = torch.cumsum(torch.randn(B, h, 3, device=device) * rot_noise, dim=1)
            batch["R_win"][:, 1:] = so3_exp(w_cum.reshape(B * h, 3)).reshape(B, h, 3, 3) @ batch["R_win"][:, 1:]
        yield batch


# Random rotation about world z (the gravity axis), a different angle for every
# sample, applied to the whole scene: drone poses, targets, wind, the wall, and
# any force labels. Rotor speeds are body-frame quantities and do not change,
# and gravity points along the rotation axis, so the rotated sample is exactly
# as physical as the original. Rotating the drone but not a vertical wall would
# create impossible samples, which is why the wall rotates too.
def rotate_drone_chain(batch):
    B = batch["com_win"].shape[0]
    dev = batch["com_win"].device
    th = torch.rand(B, device=dev) * 2.0 * math.pi
    c, s = torch.cos(th), torch.sin(th)
    z, o = torch.zeros_like(th), torch.ones_like(th)
    Rz = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1),
                      torch.stack([z, z, o], -1)], -2)                 # (B, 3, 3)
    for k in ("com_win", "tgt_com", "wind", "F_contact", "F_aero", "tau_contact"):
        if k in batch:
            batch[k] = torch.einsum('bij,btj->bti', Rz, batch[k])
    for k in ("wall_n", "wall_c"):
        batch[k] = torch.einsum('bij,bj->bi', Rz, batch[k])
    batch["R_win"] = Rz.unsqueeze(1) @ batch["R_win"]
    batch["tgt_R"] = Rz.unsqueeze(1) @ batch["tgt_R"]
    return batch
