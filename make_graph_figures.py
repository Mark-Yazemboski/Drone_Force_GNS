"""Draw the drone and its two graphs (aero, contact) from the real geometry."""
import dataclasses, numpy as np, mujoco, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import mujoco_drone_generator as G
from drone_config import DroneConfig, build_drone_graph

cfg0, gs = DroneConfig(), G.GenSettings()
cfg = G.true_config_from_model(mujoco.MjModel.from_xml_string(G.build_xml(cfg0, gs)), cfg0, gs)
com = np.array(cfg.com_offset)
rotors = np.array(cfg.rotor_pos)
base, tip = np.array(cfg.rod_base), np.array(cfg.rod_tip)
axis = (tip - base) / np.linalg.norm(tip - base)
pad_r = max(r for r, _ in cfg.pad_rings) + cfg.pad_contact_radius

C = dict(com="#444441", rotor="#7F77DD", rod="#1D9E75", pad="#D85A30", ghost="#B4B2A9", wall="#D3D1C7")
OUT = "/mnt/user-data/outputs/drone/figures/"

def circle(center, normal, r, n=40):
    tmp = np.array([0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(normal, tmp); u /= np.linalg.norm(u); v = np.cross(normal, u)
    a = np.linspace(0, 2 * np.pi, n)
    return center + r * (np.outer(np.cos(a), u) + np.outer(np.sin(a), v))

def draw_drone(ax, ghost=False, wall=False):
    a = 0.18 if ghost else 0.9
    col = C["ghost"] if ghost else None
    # core box
    hx, hy, hz = 0.06, 0.06, 0.025
    V = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([V[f] for f in faces], facecolor=col or "#5F5E5A",
                                         edgecolor=col or "#2C2C2A", alpha=a * 0.8, lw=0.4))
    for p in rotors:   # arms + rotor disks
        ax.plot(*zip([0, 0, 0], p), color=col or "#444441", lw=2.5 if not ghost else 1.5, alpha=a)
        disk = circle(p, np.array([0, 0, 1.0]), 0.06)
        ax.add_collection3d(Poly3DCollection([disk], facecolor=col or "#85B7EB", edgecolor=col or "#185FA5",
                                             alpha=a * (0.45 if not ghost else 0.6), lw=0.6))
    ax.plot(*zip(base, tip), color=col or "#888780", lw=4 if not ghost else 2, alpha=a, solid_capstyle="round")
    ax.add_collection3d(Poly3DCollection([circle(tip, axis, pad_r)], facecolor=col or "#F0997B",
                                         edgecolor=col or "#993C1D", alpha=a * 0.8, lw=0.6))
    if wall:
        x = tip[0] + cfg.pad_contact_radius
        W = [[x, -0.16, -0.08], [x, 0.16, -0.08], [x, 0.16, 0.14], [x, -0.16, 0.14]]
        ax.add_collection3d(Poly3DCollection([W], facecolor=C["wall"], edgecolor="#888780", alpha=0.35, lw=0.6))

def setup(ax, title, lim=None, elev=28, azim=-62, zoom=1.2):
    lim = lim or ((-0.20, 0.57), (-0.22, 0.22), (-0.10, 0.15))
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_box_aspect([lim[0][1] - lim[0][0], lim[1][1] - lim[1][0], lim[2][1] - lim[2][0]], zoom=zoom)
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=12, pad=0)

def draw_graph(ax, pos, edges, types, size=60):
    segs = [(pos[i], pos[j]) for i, j in edges if i < j]
    ax.add_collection3d(Line3DCollection(segs, colors="#5F5E5A", linewidths=1.1, alpha=0.9))
    for t, c in (("com", C["com"]), ("rotor", C["rotor"]), ("rod", C["rod"]), ("pad", C["pad"])):
        idx = [k for k, tt in enumerate(types) if tt == t]
        if idx:
            P = pos[idx]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=size, c=c, edgecolors="white", linewidths=0.8, depthshade=False, zorder=5)

# ---- aero graph: COM, 4 rotors, 3 rod nodes, 1 pad node ----
rod_nodes = [base + s * (tip - base) for s in (0, 1 / 3, 2 / 3)]
aero_pos = np.vstack([com, rotors, rod_nodes, tip])
aero_types = ["com"] + ["rotor"] * 4 + ["rod"] * 3 + ["pad"]
aero_edges = [(0, k) for k in range(1, 5)] + [(0, 5), (5, 6), (6, 7), (7, 8)]
front = [k for k in range(1, 5) if aero_pos[k, 0] > 0]
aero_edges += [(k, 5) for k in front]
aero_edges = [(min(a, b), max(a, b)) for a, b in aero_edges]

# ---- contact graph: COM + 9 pad nodes (no rod nodes) ----
g = build_drone_graph(dataclasses.replace(cfg, n_rod_nodes=0), com_relative=False)
c_pos = g["rest_nodes"].numpy().astype(float)
c_types = ["com"] + ["pad"] * (len(c_pos) - 1)
c_edges = {(min(a, b), max(a, b)) for a, b in g["edge_index"].T.tolist()}

def legend(fig, items, y=0.04):
    hs = [plt.Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=c, markeredgecolor="white", label=l)
          for l, c in items]
    fig.legend(handles=hs, loc="lower center", ncol=len(items), frameon=False, fontsize=10, bbox_to_anchor=(0.5, y))

# individual figures
fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(projection="3d")
draw_drone(ax, wall=True); ax.scatter(*com, s=40, c=C["com"], depthshade=False)
setup(ax, "Aerial manipulator: quadrotor, rod, contact pad, wall")
fig.savefig(OUT + "drone.png", dpi=200, bbox_inches="tight", transparent=False, facecolor="white"); plt.close(fig)

fig = plt.figure(figsize=(7, 5.4)); ax = fig.add_subplot(projection="3d")
draw_drone(ax, ghost=True); draw_graph(ax, aero_pos, aero_edges, aero_types)
setup(ax, "Aero GNN: 9 nodes")
legend(fig, [("COM", C["com"]), ("Rotor", C["rotor"]), ("Rod", C["rod"]), ("Pad", C["pad"])])
fig.savefig(OUT + "aero_graph.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

fig = plt.figure(figsize=(11, 5.4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
draw_drone(ax, ghost=True); draw_graph(ax, c_pos, c_edges, c_types, size=45)
setup(ax, "Contact GNN: COM + 9 pad nodes")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
draw_graph(ax2, c_pos[1:], [(a - 1, b - 1) for a, b in c_edges if a > 0 and b > 0], ["pad"] * (len(c_pos) - 1), size=110)
ax2.add_collection3d(Poly3DCollection([circle(tip, axis, pad_r)], facecolor="#F0997B", edgecolor="#993C1D", alpha=0.18, lw=0.6))
r = pad_r * 1.25
setup(ax2, "Pad close-up (seen from the wall, kNN edges)", lim=((tip[0] - r, tip[0] + r), (-r, r), (tip[2] - r, tip[2] + r)), elev=8, azim=-172, zoom=1.0)
legend(fig, [("COM", C["com"]), ("Pad", C["pad"])])
fig.savefig(OUT + "contact_graph.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# combined comparison
fig = plt.figure(figsize=(16, 4.6))
for k, (title, fn) in enumerate([
        ("Physical drone", lambda a: (draw_drone(a, wall=True), a.scatter(*com, s=40, c=C["com"], depthshade=False))),
        ("Aero GNN", lambda a: (draw_drone(a, ghost=True), draw_graph(a, aero_pos, aero_edges, aero_types))),
        ("Contact GNN", lambda a: (draw_drone(a, ghost=True), draw_graph(a, c_pos, c_edges, c_types, size=45)))]):
    a = fig.add_subplot(1, 3, k + 1, projection="3d"); fn(a); setup(a, title)
legend(fig, [("COM", C["com"]), ("Rotor", C["rotor"]), ("Rod", C["rod"]), ("Pad", C["pad"])], y=0.02)
fig.savefig(OUT + "drone_vs_graphs.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("aero edges", sorted(aero_edges)); print("contact nodes", len(c_pos), "edges", len(c_edges))
