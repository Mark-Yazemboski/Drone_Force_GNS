"""Figures and animations for closed-loop evaluation (see drone_closed_loop.py).

plot_closed_loop()     time series for one scenario (also used for the
                       open-loop runs, mode="open loop"):
                         1. closest pad node to the wall (MuJoCo vs model)
                         2. contact normal force: MuJoCo label, one-step model, closed-loop model
                         3. friction (tangential magnitude), with mu x normal for reference
                         4. aero force magnitude
                         5. pad position error of the closed-loop model run
animate_closed_loop()  GIF: MuJoCo drone (gray) and model drone (blue) flying the
                         same scenario with the same controller, with contact
                         arrows, beside a live normal-force trace.

Static figures use the same report style as the MPC figures (MPC_Code/visualize.py):
set_plot_style, figure_legend and save_figure below, also used by plot_training_history.py
and drone_open_loop.py. Each is saved as .png (for viewing) and .svg (for the thesis).

Force arrows follow the cube visualizer: NORMAL green, FRICTION orange, aero
magenta at the COM. Lengths are log-scaled (see arrow_len) so taps of ~30 N and
pushes of ~1 N are both visible; friction gets a larger gain because it is
smaller. Model arrows are drawn per pad node; MuJoCo's are the net contact force
at the pad center, drawn thinner and lighter.
"""

import os

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from drone_config import build_drone_graph, build_contact_graph
from drone_closed_loop import pad_center_positions

PHASE_NAMES = {0: "free", 1: "approach", 2: "hold", 3: "slide", 4: "release", 5: "hover near"}
PHASE_COLORS = {0: "white", 1: "#dbe9f6", 2: "#f6dbdb", 3: "#f6eedb", 4: "#e2f6db", 5: "#ece0f6"}   # GIF only
C_TRUE, C_ONE, C_CL = "#2C2C2A", "#378ADD", "#D85A30"         # GIF only: MuJoCo, one-step model, model run
C_NORMAL, C_FRIC, C_AERO = "#2ca02c", "#ff7f0e", "#d62798"
C_DRONE_TRUE, C_DRONE_PRED = "#888780", "#185FA5"


# ---------------------------------------------------------------- report style
# Same text and layout as the MPC figures. MuJoCo = black, model one step = blue
# (dotted), the model flying on its own (closed or open loop) = orange, and the
# model with its contact network off = gray (dashed).
MUJOCO_COLOR = "black"
ONE_STEP_COLOR = "#2a78d6"
RUN_COLOR = "#eb6834"
OFF_COLOR = "#9a9890"

#Figure width for one column of a two-column page (inches). Figures are made at the size
#they're printed, so the text comes out at the same size as the body text.
REPORT_WIDTH = 3.5


def set_plot_style(font_size=9):
    #Report-style figures (same idea as the MATLAB defaults): Times font, text about the size of
    #10 pt body text, thin lines, white background. STIX is a Times look-alike that
    #ships with matplotlib, so this works on computers without Times New Roman (like Roar).
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def figure_legend(fig, axes, ncol=3):
    #One legend above all the panels instead of one per panel (saves a lot of room at report size)
    handles = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.0)
    rows = -(-len(handles) // ncol)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.045 * rows * 4.0 / fig.get_figheight()))


def save_figure(fig, path):
    #Saves <path>.png (for viewing) and <path>.svg (sharp text in the thesis), then closes the figure
    stem = os.path.splitext(path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".svg")
    plt.close(fig)


def _pad_nodes(cfg):
    g = build_drone_graph(cfg)
    return g["rest_nodes"].numpy()[g["pad_indices"]]                  # COM-relative, body frame


def _wall_dist(run, cfg, n, wall_point):
    pads = run["com"][:, None, :] + np.einsum('tij,nj->tni', run["R"], _pad_nodes(cfg))
    return ((pads - wall_point) @ n).min(axis=1)


def _shade(ax, phase, dt):
    start = 0
    for k in range(1, len(phase) + 1):
        if k == len(phase) or phase[k] != phase[start]:
            ax.axvspan(start * dt, k * dt, color=PHASE_COLORS[int(phase[start])], lw=0, zorder=0)
            start = k


def _tangential(F, n):
    return np.linalg.norm(F - (F @ n)[:, None] * n, axis=1)


# Contact-force axes use a symmetric log scale: linear below FORCE_LINTHRESH_N (so the
# exact zeros off the wall still plot), logarithmic above it, so a 1-5 N push and a
# 100 N tap are both readable on one axis.
FORCE_LINTHRESH_N = 0.1


def _force_axis(ax, top):
    ax.set_yscale("symlog", linthresh=FORCE_LINTHRESH_N, linscale=0.5)
    ax.set_ylim(-0.02, top)


# Upper axis limit that clips nothing: the max over every trace drawn (MuJoCo,
# one step, closed loop), so neither true impact peaks nor model overshoots
# are hidden by the axis.
def _full_range(*traces, floor=1.0):
    vals = np.concatenate([np.ravel(t) for t in traces])
    vals = vals[np.isfinite(vals)]
    return max(floor, 1.1 * float(vals.max())) if vals.size else floor


def plot_closed_loop(res, cfg, path, title=None, mode="closed loop"):
    """mode: "closed loop" or "open loop" (drone_open_loop.py). Optional entries of res:
    one_step (one-step model forces along the MuJoCo run; None for open loop) and
    contact_off (the same run with the contact network switched off). Open loop puts
    the pad error on a log axis (the growth is exponential) with 1 cm / 5 cm lines.
    Saves <path>.png and .svg. title is not drawn (report figures have no titles): the
    callers print it instead."""
    sc, true, pred = res["scenario"], res["true"], res["pred"]
    one, off = res.get("one_step"), res.get("contact_off")
    n, wp = sc["n"], sc["wall_point"]
    dt = cfg.dt
    Tt, Tp = true["n_frames"], pred["n_frames"]
    tt, tp = np.arange(Tt) * dt, np.arange(Tp) * dt
    T = min(Tt, Tp)
    run_label = f"model, {mode}"
    radius = cfg.pad_contact_radius

    set_plot_style()
    fig, ax = plt.subplots(5, 1, figsize=(REPORT_WIDTH, 5.8), sharex=True)

    #Gap between the closest pad sphere and the wall (mm), zoomed in near the wall
    ax[0].plot(tt, 1e3 * (_wall_dist(true, cfg, n, wp) - radius), color=MUJOCO_COLOR, label="MuJoCo")
    ax[0].plot(tp, 1e3 * (_wall_dist(pred, cfg, n, wp) - radius), color=RUN_COLOR, label=run_label)
    if off is not None:
        ax[0].plot(np.arange(off["n_frames"]) * dt, 1e3 * (_wall_dist(off, cfg, n, wp) - radius),
                   color=OFF_COLOR, ls="--", label="model, contact off")
    ax[0].axhline(0.0, color="0.7", linewidth=0.6)                  # wall surface (gray: MuJoCo is black)
    ax[0].set_ylim(-2.0, 20.0)
    ax[0].set_ylabel("gap (mm)")

    #Normal force, friction magnitude and aero force magnitude: MuJoCo, one step, model run
    traces = [("$F_n$ (N)", lambda r: r["F_contact"] @ n, 1.0),
              (r"$\|F_t\|$ (N)", lambda r: _tangential(r["F_contact"], n), 0.3),
              (r"$\|F_\mathrm{aero}\|$ (N)", lambda r: np.linalg.norm(r["F_aero"], axis=1), None)]
    for a, (label, value, floor) in zip(ax[1:4], traces):
        drawn = [value(true), value(pred)]
        a.plot(tt, drawn[0], color=MUJOCO_COLOR)
        if one is not None:
            drawn.append(value(one))
            a.plot(tt, drawn[-1], color=ONE_STEP_COLOR, linestyle=":", linewidth=1.2, label="model, one step")
        a.plot(tp, drawn[1], color=RUN_COLOR)
        if floor is not None:
            _force_axis(a, _full_range(*drawn, floor=floor))
        a.set_ylabel(label)

    #Pad position error of the model run vs MuJoCo
    pc_true = pad_center_positions(true, cfg)
    err = 1e3 * np.linalg.norm(pc_true[:T] - pad_center_positions(pred, cfg)[:T], axis=1)
    ax[4].plot(tp[:T], np.maximum(err, 1e-3) if mode == "open loop" else err, color=RUN_COLOR)
    if off is not None:
        To = min(Tt, off["n_frames"])
        e_off = 1e3 * np.linalg.norm(pc_true[:To] - pad_center_positions(off, cfg)[:To], axis=1)
        ax[4].plot(np.arange(To) * dt, np.maximum(e_off, 1e-3), color=OFF_COLOR, ls="--")
    if mode == "open loop":
        ax[4].set_yscale("log")
        for thr in (10, 50):
            ax[4].axhline(thr, color="black", ls=":", lw=0.6)
        ax[4].set_ylim(1e-2, 400)
    ax[4].set_ylabel("pad error (mm)")
    ax[4].set_xlabel("time since start (s)" if mode == "open loop" else "time (s)")
    ax[4].set_xlim(0, max(tt[-1], tp[-1]))

    figure_legend(fig, ax, ncol=2)
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------- 3D drone

def _drone_polylines(cfg):
    """Body-frame polylines (relative to the COM) for a schematic drone."""
    c = np.asarray(cfg.com_offset)
    lines = []
    hx, hy, hz = 0.06, 0.06, 0.025
    V = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for a, b in [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]:
        lines.append(np.array([V[a], V[b]]))
    ang = np.linspace(0, 2 * np.pi, 25)
    for p, ax_ in zip(np.asarray(cfg.rotor_pos), np.asarray(cfg.rotor_axis, float)):
        lines.append(np.array([np.zeros(3), p]))
        u = np.cross(ax_, [1.0, 0, 0] if abs(ax_[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(ax_, u)
        lines.append(p + 0.06 * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)))
    base, tip = np.asarray(cfg.rod_base), np.asarray(cfg.rod_tip)
    lines.append(np.array([base, tip]))
    axis = (tip - base) / np.linalg.norm(tip - base)
    u = np.cross(axis, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    r = max([rr for rr, _ in cfg.pad_rings], default=0.0) + cfg.pad_contact_radius
    lines.append(tip + r * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v)))
    return [ln - c for ln in lines]


ARROW_REF_N, ARROW_REF_LEN, ARROW_MAX_LEN = 1.0, 0.15, 0.30


def arrow_len(F, ref_N=ARROW_REF_N, L_ref=ARROW_REF_LEN, L_max=ARROW_MAX_LEN):
    """Newtons -> metres, log-scaled so ref_N draws L_ref long (1 N -> 15 cm,
    0.3 N -> 6 cm), capped at L_max so 30 N taps stay on screen."""
    return np.minimum(L_ref * np.log1p(F / ref_N) / np.log(2.0), L_max)


def _scaled(vecs, gain):
    mag = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return np.where(mag > 1e-9, vecs / np.maximum(mag, 1e-12) * gain * arrow_len(mag), 0.0)


def _moving_average(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], axis=1)


def animate_closed_loop(res, cfg, path, stride=3, fps=None, friction_gain=2.5, min_N=0.03,
                        view_half_width=0.33, true_force_avg_frames=5, dpi=80, camera="fixed",
                        mode="closed loop"):
    """mode: "closed loop" (model as the plant under the same controller) or
    "open loop" (recorded rotor speeds, no controller; drone_open_loop.py). It only
    changes the labels. res["one_step"] may be None (open loop has no one-step trace).
    camera: "fixed" (default) keeps one view of the contact region and a fixed,
    gridded wall, so sliding reads correctly; "follow" centers on the drone every
    frame (bigger drone, but the background moves with it, which makes sliding look
    reversed). view_half_width: half-width of the "follow" view (m).
    true_force_avg_frames: MuJoCo's contact force chatters during slides; its arrows
    and HUD value use a centered moving average over this many frames (5 = 50 ms).
    The force trace on the right shows the raw label."""
    sc, true, pred = res["scenario"], res["true"], res["pred"]
    n, wp = sc["n"], sc["wall_point"]
    T = min(true["n_frames"], pred["n_frames"])
    frames = list(range(0, T, stride))
    fps = fps or max(1, round(1.0 / (cfg.dt * stride)))
    polys = _drone_polylines(cfg)
    contact_rest = build_contact_graph(cfg)["rest_nodes"].numpy()      # model's per-node forces live here
    tip_rest = np.asarray(cfg.rod_tip) - np.asarray(cfg.com_offset)

    fig = plt.figure(figsize=(12, 5.8))
    ax = fig.add_axes([0.0, 0.11, 0.58, 0.81], projection="3d")
    axf = fig.add_axes([0.64, 0.56, 0.33, 0.36])
    axd = fig.add_axes([0.64, 0.10, 0.33, 0.36])

    # Camera: fixed direction (looking along the wall, from the free side),
    # centered on the drone each frame so it stays large.
    nh = n.copy()
    nh[2] = 0
    nh /= np.linalg.norm(nh)
    t1 = np.cross(nh, [0, 0, 1.0])
    cam = 0.55 * nh + 0.85 * t1
    ax.view_init(elev=18, azim=np.degrees(np.arctan2(cam[1], cam[0])))
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
    pad_t = pad_center_positions(true, cfg)
    centers = 0.5 * (true["com"] + pad_t)
    F_true_avg = _moving_average(true["F_contact"], true_force_avg_frames)

    # Wall patch: redrawn each frame at the pad's projection onto the wall and
    # sized to the view (matplotlib does not clip 3D polygons to the axes).
    w1 = np.cross(n, [0, 0, 1.0])
    w1 /= np.linalg.norm(w1)
    w2 = np.cross(w1, n)
    wall_art = []

    def draw_wall(k):
        for a_ in wall_art:
            a_.remove()
        wall_art.clear()
        p = pad_t[k] - ((pad_t[k] - wp) @ n) * n
        hw = 0.75 * view_half_width
        W = [p + a * hw * w1 + b * hw * w2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        wall_art.append(ax.add_collection3d(Poly3DCollection([W], facecolor="#D3D1C7", edgecolor="#888780",
                                                             alpha=0.35)))

    if camera == "fixed":
        # One view for the whole GIF, centered on where the pad works near the wall,
        # and a fixed wall with a 5 cm grid as a stationary reference.
        d_pad = (pad_t[:T] - wp) @ n
        near = d_pad < 0.10
        sel = near if near.any() else np.ones(T, bool)
        c_fix = centers[:T][sel].mean(0)
        ext = (centers[:T].max(0) - centers[:T].min(0)).max()
        hw_fix = float(np.clip(0.5 * ext + 0.25, 0.40, 0.60))
        ax.set_xlim(c_fix[0] - hw_fix, c_fix[0] + hw_fix)
        ax.set_ylim(c_fix[1] - hw_fix, c_fix[1] + hw_fix)
        ax.set_zlim(c_fix[2] - hw_fix, c_fix[2] + hw_fix)
        p0 = pad_t[:T][sel].mean(0)
        p0 = p0 - ((p0 - wp) @ n) * n
        hw_w = 0.8 * hw_fix
        W = [p0 + a * hw_w * w1 + b * hw_w * w2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        ax.add_collection3d(Poly3DCollection([W], facecolor="#D3D1C7", edgecolor="#888780", alpha=0.30))
        for s_ in np.arange(-np.floor(hw_w / 0.05) * 0.05, hw_w + 1e-9, 0.05):
            for u, v_ in ((w1, w2), (w2, w1)):
                a_, b_ = p0 + s_ * u - hw_w * v_, p0 + s_ * u + hw_w * v_
                ax.plot([a_[0], b_[0]], [a_[1], b_[1]], [a_[2], b_[2]], color="#B4B2A9", lw=0.5, alpha=0.8)
    elif camera != "follow":
        raise ValueError(f"camera must be 'fixed' or 'follow', got {camera!r}")

    def make_lines(color, lw, alpha):
        return [ax.plot([], [], [], color=color, lw=lw, alpha=alpha)[0] for _ in polys]

    lines_t = make_lines(C_DRONE_TRUE, 2.6, 0.45)
    lines_p = make_lines(C_DRONE_PRED, 1.2, 0.95)
    trail_t, = ax.plot([], [], [], color=C_DRONE_TRUE, lw=0.8, alpha=0.6)
    trail_p, = ax.plot([], [], [], color=C_DRONE_PRED, lw=0.8, alpha=0.6)
    hud = fig.text(0.02, 0.97, "", fontsize=9, va="top", family="monospace")
    blue = ("blue: model as the plant, same controller and scenario" if mode == "closed loop" else
            "blue: model, open loop (recorded rotor speeds, no controller)")
    fig.text(0.02, 0.02, f"gray (thick): MuJoCo    {blue}\n"
                         f"arrows: green normal, orange friction (x{friction_gain:g}), magenta aero at COM; "
                         f"log-scaled, {100 * ARROW_REF_LEN:.0f} cm = {ARROW_REF_N:g} N (capped at {100 * ARROW_MAX_LEN:.0f} cm)\n"
                         f"model arrows per pad node; MuJoCo arrows = net force, {10 * true_force_avg_frames} ms average",
             fontsize=8, va="bottom")

    # Right panels: normal force and pad distance, with a moving cursor.
    dt = cfg.dt
    tt = np.arange(T) * dt
    _shade(axf, true["phase"][:T], dt)
    _shade(axd, true["phase"][:T], dt)
    axf.plot(tt, true["F_contact"][:T] @ n, color=C_TRUE, lw=1.0, label="MuJoCo")
    axf.plot(tt, pred["F_contact"][:T] @ n, color=C_CL, lw=1.0, label=f"model, {mode}")
    traces = [true["F_contact"][:T] @ n, pred["F_contact"][:T] @ n]
    if res.get("one_step") is not None:
        axf.plot(tt, res["one_step"]["F_contact"][:T] @ n, color=C_ONE, lw=0.8, label="model, one step")
        traces.append(res["one_step"]["F_contact"][:T] @ n)
    _force_axis(axf, _full_range(*traces))
    axf.set_ylabel("normal force (N, symlog)", fontsize=9)
    axf.legend(fontsize=7, loc="upper right")
    axd.plot(tt, 1e3 * _wall_dist(true, cfg, n, wp)[:T], color=C_TRUE, lw=1.0)
    axd.plot(tt, 1e3 * _wall_dist(pred, cfg, n, wp)[:T], color=C_CL, lw=1.0)
    axd.set_ylim(-5, 120)
    axd.set_ylabel("pad to wall (mm)", fontsize=9)
    axd.set_xlabel("time (s)", fontsize=9)
    cur_f = axf.axvline(0, color="k", lw=0.8)
    cur_d = axd.axvline(0, color="k", lw=0.8)
    quivers = []

    def world(run, k, P):
        return run["com"][k] + P @ run["R"][k].T

    def arrows(origins, F, alpha, lw):
        Fn = (F @ n)[:, None] * n
        Ft = F - Fn
        for vec, col, gain in ((Fn, C_NORMAL, 1.0), (Ft, C_FRIC, friction_gain)):
            keep = np.linalg.norm(vec, axis=1) > min_N
            if keep.any():
                s = _scaled(vec[keep], gain)
                o = origins[keep]
                quivers.append(ax.quiver(o[:, 0], o[:, 1], o[:, 2], s[:, 0], s[:, 1], s[:, 2], color=col,
                                         alpha=alpha, linewidth=lw, arrow_length_ratio=0.3))

    def update(i):
        k = frames[i]
        for q in quivers:
            q.remove()
        quivers.clear()
        if camera == "follow":
            c = centers[k]
            draw_wall(k)
            ax.set_xlim(c[0] - view_half_width, c[0] + view_half_width)
            ax.set_ylim(c[1] - view_half_width, c[1] + view_half_width)
            ax.set_zlim(c[2] - view_half_width, c[2] + view_half_width)
        for run, lines in ((true, lines_t), (pred, lines_p)):
            for ln, P in zip(lines, polys):
                Wp = world(run, k, P)
                ln.set_data_3d(Wp[:, 0], Wp[:, 1], Wp[:, 2])
        for run, trail in ((true, trail_t), (pred, trail_p)):
            pc = pad_center_positions(run, cfg)[max(0, k - 150):k + 1]
            trail.set_data_3d(pc[:, 0], pc[:, 1], pc[:, 2])
        # Model: per pad node. MuJoCo: net force at the pad center.
        arrows(world(pred, k, contact_rest), pred["F_pad"][k], 0.95, 2.2)
        arrows(world(true, k, tip_rest[None]), F_true_avg[k][None], 0.45, 1.4)
        Fa = pred["F_aero"][k]
        if np.linalg.norm(Fa) > min_N:
            s = _scaled(Fa[None], 1.0)[0]
            cp = pred["com"][k]
            quivers.append(ax.quiver(cp[0], cp[1], cp[2], s[0], s[1], s[2], color=C_AERO, linewidth=1.8,
                                     arrow_length_ratio=0.3))
        err = 1e3 * np.linalg.norm(pad_center_positions(true, cfg)[k] - pad_center_positions(pred, cfg)[k])
        hud.set_text(f"t = {k * dt:5.2f} s   {PHASE_NAMES[int(true['phase'][k])]:<10s}\n"
                     f"normal force  MuJoCo {F_true_avg[k] @ n:5.2f} N   model {pred['F_contact'][k] @ n:5.2f} N\n"
                     f"pad position error {err:5.1f} mm")
        cur_f.set_xdata([k * dt])
        cur_d.set_xdata([k * dt])
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return path
