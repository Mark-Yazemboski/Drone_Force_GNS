"""Define the aerial manipulator's physical description and the graph the
contact GNN runs on. DroneConfig holds the measured mass properties, the rotor
layout and thrust/yaw coefficients, the rod geometry, and the graph options.
Everything is entered in the MOCAP body frame (the frame the motion-capture
rigid body reports), and the helpers convert it to COM-relative quantities,
because the dynamics are written about the center of mass. build_drone_graph()
creates the node set (one body node, nodes along the rod, a tip node or a small
pad), node types, contact mask, and hand-wired edges.

Every default number below is a PLACEHOLDER. Replace mass, inertia, COM offset,
k_f, and k_m with measured values (or read them from the MuJoCo model for sim).
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch

import dataclasses

# Node kinds used while building the geometry.
NODE_BODY, NODE_ROD, NODE_TIP = 0, 1, 2
N_NODE_TYPES = 3

# One-hot node types for each GNN.
# Aero graph:    COM [1,0,0,0], rotor [0,1,0,0], rod [0,0,1,0], pad [0,0,0,1]
# Contact graph: COM [1,0],     pad [0,1]       (+ rod [0,0,1] only if rod_can_contact)
AERO_COM, AERO_ROTOR, AERO_ROD, AERO_PAD = 0, 1, 2, 3
N_AERO_TYPES = 4


@dataclass
class DroneConfig:
    # ---- mass properties (measure WITH the rod and sensor mounted) ----
    mass: float = 1.5                                   # kg
    # Inertia about the COM, body axes, kg m^2. Full 3x3: the rod makes it non-isotropic.
    inertia: tuple = ((0.015, 0.0, 0.0),
                      (0.0, 0.030, 0.0),
                      (0.0, 0.0, 0.035))
    # COM position in the mocap body frame (m). Mocap origin != COM unless you calibrate it.
    com_offset: tuple = (0.04, 0.0, -0.01)

    # ---- rotors (mocap body frame) ----
    rotor_pos: tuple = ((0.12, 0.12, 0.0), (-0.12, 0.12, 0.0),
                        (-0.12, -0.12, 0.0), (0.12, -0.12, 0.0))
    rotor_axis: tuple = ((0.0, 0.0, 1.0),) * 4
    spin_dir: tuple = (1, -1, 1, -1)       # +1 = rotor spins CCW about its own axis
    k_f: float = 1.0e-5                     # thrust coeff,   T   = k_f * omega^2  (N, omega in rad/s)
    k_m: float = 1.6e-7                     # yaw-moment coeff, Q = k_m * omega^2  (N m)

    # ---- rod / end effector (mocap body frame) ----
    # Angled ~8 deg up: pushing needs nose-down pitch, and a level rod at or
    # below the COM then puts the pad below the COM, where the wall's normal
    # force pitches the drone further into the wall (unstable in the MuJoCo
    # generator). Set these to the real geometry.
    rod_base: tuple = (0.10, 0.0, 0.0)
    rod_tip: tuple = (0.45, 0.0, 0.05)
    n_rod_nodes: int = 3            # rod nodes in the AERO graph (base included, tip excluded);
                                    # the contact graph has none unless rod_can_contact
    tip_type: str = "pad"           # "pad" -> center + rings; "point" -> 1 tip node
    # Pad rings as (radius m, node count), in the plane perpendicular to the rod
    # at rod_tip. More/denser rings = finer pad mesh. In MuJoCo each pad node is
    # a small collision sphere of radius pad_contact_radius, so the gate distance
    # contact_d0 should be about pad_contact_radius plus a margin.
    pad_rings: tuple = ((0.025, 8),)
    pad_contact_radius: float = 0.004
    pad_knn: int = 4                # pad-internal edges: each pad node to its k nearest pad nodes
    rod_can_contact: bool = False   # True lets rod nodes take contact (rod lying on the wall)
    edge_mode: str = "chain"        # "chain" (body-rod-tip + body->tip shortcut + pad kNN) or "full"

    # ---- physics / timing ----
    gravity: float = 9.81
    dt: float = 0.01                # recorded step (s); checked against the data's meta

    def to_dict(self):
        return asdict(self)


# Returns the COM-relative rotor geometry and the inertia-over-mass tensor as torch tensors.
# J = I/m is the drone's version of the cube's scalar I_OVER_M.
def rotor_and_inertia_tensors(cfg):
    c = np.asarray(cfg.com_offset, dtype=np.float64)
    rotor_pos = np.asarray(cfg.rotor_pos, dtype=np.float64) - c
    axes = np.asarray(cfg.rotor_axis, dtype=np.float64)
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    J = np.asarray(cfg.inertia, dtype=np.float64) / cfg.mass

    # Mean thrust axis: the direction rotor drag acts perpendicular to.
    thrust_axis = axes.sum(0)
    thrust_axis = thrust_axis / np.linalg.norm(thrust_axis)

    f32 = lambda a: torch.tensor(a, dtype=torch.float32)
    return dict(rotor_pos=f32(rotor_pos), rotor_axis=f32(axes),
                spin_dir=f32(np.asarray(cfg.spin_dir, dtype=np.float64)),
                J_body=f32(J), thrust_axis=f32(thrust_axis))


# Builds the node set, node types, contact mask, and edges for the drone graph.
# Node 0 is the body node at the COM; rod nodes follow; then the tip center;
# then the pad-ring nodes. Positions are COM-relative unless com_relative=False,
# in which case they are in the mocap/body frame (what the MuJoCo XML needs).
def build_drone_graph(cfg, com_relative=True):
    c = np.asarray(cfg.com_offset, dtype=np.float64)
    base = np.asarray(cfg.rod_base, dtype=np.float64)
    tip = np.asarray(cfg.rod_tip, dtype=np.float64)

    nodes, types = [c.copy()], [NODE_BODY]

    # Rod nodes, base included, tip excluded.
    for s in np.linspace(0.0, 1.0, cfg.n_rod_nodes, endpoint=False):
        nodes.append(base + s * (tip - base))
        types.append(NODE_ROD)

    # Tip / pad center.
    tip_index = len(nodes)
    nodes.append(tip)
    types.append(NODE_TIP)

    # Pad rings in the plane perpendicular to the rod.
    pad = [tip_index]
    if cfg.tip_type == "pad":
        axis = (tip - base) / np.linalg.norm(tip - base)
        tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        for radius, count in cfg.pad_rings:
            for k in range(count):
                ang = 2.0 * np.pi * k / count
                pad.append(len(nodes))
                nodes.append(tip + radius * (np.cos(ang) * u + np.sin(ang) * v))
                types.append(NODE_TIP)
    elif cfg.tip_type != "point":
        raise ValueError(f"tip_type must be 'point' or 'pad', got {cfg.tip_type!r}")

    nodes = np.asarray(nodes)
    types = np.asarray(types)
    N = len(nodes)

    # Edges. At this node count kNN over the whole body buys nothing, so the
    # body/rod part is hand-wired and only the pad uses kNN (among pad nodes).
    pairs = set()
    if cfg.edge_mode == "full":
        pairs = {(i, j) for i in range(N) for j in range(N) if i != j}
    elif cfg.edge_mode == "chain":
        chain = list(range(tip_index + 1))                  # body, rod..., tip center
        for a, b in zip(chain[:-1], chain[1:]):
            pairs |= {(a, b), (b, a)}
        pairs |= {(0, tip_index), (tip_index, 0)}           # shortcut: body state reaches the pad in one hop
        if len(pad) > 1:
            P = nodes[pad]
            dist = np.linalg.norm(P[:, None] - P[None], axis=-1)
            for i in range(len(pad)):
                for j in np.argsort(dist[i])[1:cfg.pad_knn + 1]:
                    pairs |= {(pad[i], pad[j]), (pad[j], pad[i])}
    else:
        raise ValueError(f"edge_mode must be 'chain' or 'full', got {cfg.edge_mode!r}")
    edge_index = np.array(sorted(pairs)).T

    # Which nodes may carry a contact force.
    contact_mask = types == NODE_TIP
    if cfg.rod_can_contact:
        contact_mask |= types == NODE_ROD

    if com_relative:
        nodes = nodes - c

    return dict(rest_nodes=torch.tensor(nodes, dtype=torch.float32),
                node_type=torch.tensor(types, dtype=torch.long),
                contact_mask=torch.tensor(contact_mask),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                tip_index=tip_index, pad_indices=pad)


# ======================================================================
# The two graphs the model actually runs on
# ======================================================================

# Contact graph: COM hub + pad nodes (rod nodes only if the rod may touch).
# Edges: COM <-> pad center, each pad node <-> its pad_knn nearest pad nodes.
def build_contact_graph(cfg):
    n_rod = cfg.n_rod_nodes if cfg.rod_can_contact else 0
    g = build_drone_graph(dataclasses.replace(cfg, n_rod_nodes=n_rod))
    kind = g["node_type"]
    onehot_index = torch.where(kind == NODE_BODY, 0, torch.where(kind == NODE_TIP, 1, 2))
    n_types = 3 if n_rod > 0 else 2
    g["onehot"] = torch.nn.functional.one_hot(onehot_index, n_types).float()
    return g


# Aero graph: COM, one node per rotor, rod nodes, one pad node at the rod tip.
# Edges (bidirectional): every rotor <-> COM; COM <-> rod_1 <-> ... <-> rod_n <-> pad;
# each front rotor (ahead of the COM along the rod direction) <-> its nearest
# rod node, so information about its wash can reach the rod.
# All positions COM-relative, body frame.
def build_aero_graph(cfg):
    c = np.asarray(cfg.com_offset, dtype=np.float64)
    rotors = np.asarray(cfg.rotor_pos, dtype=np.float64) - c
    axes = np.asarray(cfg.rotor_axis, dtype=np.float64)
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    base = np.asarray(cfg.rod_base, dtype=np.float64) - c
    tip = np.asarray(cfg.rod_tip, dtype=np.float64) - c
    rod_axis = (tip - base) / np.linalg.norm(tip - base)

    nodes, types = [np.zeros(3)], [AERO_COM]
    rotor_idx = []
    for p in rotors:
        rotor_idx.append(len(nodes))
        nodes.append(p)
        types.append(AERO_ROTOR)
    rod_idx = []
    for s in np.linspace(0.0, 1.0, cfg.n_rod_nodes, endpoint=False):
        rod_idx.append(len(nodes))
        nodes.append(base + s * (tip - base))
        types.append(AERO_ROD)
    pad_idx = len(nodes)
    nodes.append(tip)
    types.append(AERO_PAD)
    nodes = np.asarray(nodes)

    pairs = set()
    for i in rotor_idx:
        pairs |= {(0, i), (i, 0)}
    chain = [0] + rod_idx + [pad_idx]
    for a, b in zip(chain[:-1], chain[1:]):
        pairs |= {(a, b), (b, a)}
    if rod_idx:
        for i in rotor_idx:
            if nodes[i] @ rod_axis > 0:
                j = rod_idx[int(np.argmin(np.linalg.norm(nodes[rod_idx] - nodes[i], axis=1)))]
                pairs |= {(i, j), (j, i)}
    edge_index = np.array(sorted(pairs)).T

    t = torch.tensor(types, dtype=torch.long)
    return dict(rest_nodes=torch.tensor(nodes, dtype=torch.float32),
                node_type=t, onehot=torch.nn.functional.one_hot(t, N_AERO_TYPES).float(),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                rotor_idx=rotor_idx, rod_idx=rod_idx, pad_idx=pad_idx,
                rotor_axis=torch.tensor(axes, dtype=torch.float32),
                rod_axis=torch.tensor(rod_axis, dtype=torch.float32))
