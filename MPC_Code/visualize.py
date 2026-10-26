from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utilities import quat_to_rot_matrix_np


#Colors: actual = blue, reference = orange (dashed)
ACTUAL_COLOR = "#2a78d6"
REFERENCE_COLOR = "#eb6834"
MODEL_COLOR = "#2ca02c"

def to_wall_frame(points, wall_normal, wall_point):
    #World points (T, 3) -> wall frame [out of wall, along wall, up wall]
    wall_normal = np.asarray(wall_normal, dtype=float)
    along_wall = np.cross(wall_normal, [0.0, 0.0, 1.0])
    along_wall = along_wall / np.linalg.norm(along_wall)
    up_wall = np.cross(along_wall, wall_normal)
    wall_rotation = np.vstack([wall_normal, along_wall, up_wall])
    return (np.asarray(points) - wall_point) @ wall_rotation.T


def pad_from_states(state_history, pad_offset):
    #Pad position and velocity from the drone states
    #state_history: (T, 19) [pos, vel, quat, ang_vel, vel_prev, ang_vel_prev]
    #pad_offset: COM -> pad center, body frame (3,)
    pad_positions = []
    pad_velocities = []
    for state in state_history:
        R = quat_to_rot_matrix_np(state[6:10])
        lever_arm = R @ pad_offset
        pad_positions.append(state[0:3] + lever_arm)
        pad_velocities.append(state[3:6] + np.cross(state[10:13], lever_arm))
    return np.asarray(pad_positions), np.asarray(pad_velocities)


def drone_shape(drone_cfg):
    #Drone geometry in the body frame, relative to the COM (the state is about the COM)
    com = np.asarray(drone_cfg.com_offset, dtype=float)
    rotors = np.asarray(drone_cfg.rotor_pos, dtype=float) - com
    rod_base = np.asarray(drone_cfg.rod_base, dtype=float) - com
    rod_tip = np.asarray(drone_cfg.rod_tip, dtype=float) - com
    pad_radius = max(radius for radius, _ in drone_cfg.pad_rings)

    #Circle outlines for the propellers (horizontal) and the pad (perpendicular to the rod)
    angles = np.linspace(0, 2 * np.pi, 24)
    prop_circle = 0.06 * np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)

    rod_axis = (rod_tip - rod_base) / np.linalg.norm(rod_tip - rod_base)
    u = np.cross(rod_axis, [0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u)
    v = np.cross(rod_axis, u)
    pad_circle = rod_tip + pad_radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))

    return dict(rotors=rotors, rod_base=rod_base, rod_tip=rod_tip,
                prop_circle=prop_circle, pad_circle=pad_circle)


def animate_flight(drone_cfg, wall_normal, wall_point, state_history, dt,
                   reference_positions=None, pad_offset=None, speed=1.0, frame_stride=3, save_path=None):
    #3D replay of the flight with matplotlib: drone, wall, pad path, and (optionally) the reference path
    #state_history: (T, 19)   dt: time between states (s)
    #reference_positions: (T, 3) reference pad positions (optional)
    #frame_stride: draw every n-th state (3 at 100 Hz -> ~33 fps)
    shape = drone_shape(drone_cfg)
    wall_normal = np.asarray(wall_normal, dtype=float)
    wall_point = np.asarray(wall_point, dtype=float)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")

    #Wall: a square in the wall plane around the wall point
    along_wall = np.cross(wall_normal, [0.0, 0.0, 1.0])
    along_wall = along_wall / np.linalg.norm(along_wall)
    up_wall = np.cross(along_wall, wall_normal)
    half_size = 0.6
    wall_corners = [wall_point + half_size * (a * along_wall + b * up_wall)
                    for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    ax.add_collection3d(Poly3DCollection([wall_corners], color="#b8a98f", alpha=0.35))

    #Pad path (actual) and reference path
    if pad_offset is not None:
        pad_positions, _ = pad_from_states(state_history, pad_offset)
        ax.plot(*pad_positions.T, color=ACTUAL_COLOR, linewidth=1, alpha=0.5, label="pad path")
    if reference_positions is not None:
        ax.plot(*np.asarray(reference_positions).T, color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="reference")

    #Drone parts, updated every frame
    arm_lines = [ax.plot([], [], [], color="black", linewidth=2)[0] for _ in shape["rotors"]]
    prop_lines = [ax.plot([], [], [], color=ACTUAL_COLOR, linewidth=1)[0] for _ in shape["rotors"]]
    rod_line, = ax.plot([], [], [], color="gray", linewidth=2)
    pad_line, = ax.plot([], [], [], color="#e34948", linewidth=2)
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    #Fixed, equal-scale axes around the whole flight and the wall
    all_points = np.vstack([state_history[:, 0:3], np.asarray(wall_corners)])
    center = all_points.mean(axis=0)
    half_range = 0.5 * np.max(all_points.max(axis=0) - all_points.min(axis=0)) + 0.3
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="upper right", fontsize=8)

    frame_indices = list(range(0, len(state_history), frame_stride))

    def to_world(points, position, R):
        return position + points @ R.T

    def update(frame_number):
        k = frame_indices[frame_number]
        state = state_history[k]
        position = state[0:3]
        R = quat_to_rot_matrix_np(state[6:10])

        #Arms and propeller disks
        for i, rotor in enumerate(shape["rotors"]):
            hub = to_world(rotor[None, :], position, R)[0]
            arm_lines[i].set_data_3d([position[0], hub[0]], [position[1], hub[1]], [position[2], hub[2]])
            prop = to_world(shape["prop_circle"] + rotor, position, R)
            prop_lines[i].set_data_3d(prop[:, 0], prop[:, 1], prop[:, 2])

        #Rod and pad
        rod = to_world(np.vstack([shape["rod_base"], shape["rod_tip"]]), position, R)
        rod_line.set_data_3d(rod[:, 0], rod[:, 1], rod[:, 2])
        pad = to_world(shape["pad_circle"], position, R)
        pad_line.set_data_3d(pad[:, 0], pad[:, 1], pad[:, 2])

        time_text.set_text(f"t = {k * dt:.2f} s")
        return arm_lines + prop_lines + [rod_line, pad_line, time_text]

    interval_ms = 1000 * dt * frame_stride / speed
    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval_ms, blit=False)

    if save_path is not None:
        animation.save(save_path, writer=PillowWriter(fps=int(1000 / interval_ms)))
    plt.show()
    return animation



def plot_tracking(times, state_history, reference, pad_offset, wall_normal, wall_point,
                  normal_force_history=None, predicted_force_history=None,
                  mujoco_friction_history=None, predicted_friction_history=None,
                  model_pad_distance_history=None, contact_d0=None,
                  force_ylim=None, save_path=None):
    #Reference vs what the drone actually did (and what the model thought), as two figures:
    #   position figure: pad position in the wall frame, and the model's pad distance to the wall
    #   force figure:    normal force and friction, MuJoCo vs model
    #times: (T,)   state_history: (T, 19)   reference: trajectory_generator.sample(times)
    #normal_force_history / predicted_force_history: (T,) MuJoCo / model normal force (N)
    #mujoco_friction_history / predicted_friction_history: (T, 3) MuJoCo / model friction, world frame (N)
    #model_pad_distance_history: (T,) model's min pad-to-wall distance (m)
    #contact_d0: contact gate distance (m), drawn on the distance plot
    #force_ylim: top of the force plots (N); default is 2.5x the largest reference force
    #save_path: e.g. run_folder / "tracking.png" saves tracking_position.png and tracking_force.png
    wall_point = np.asarray(wall_point, dtype=float)

    #Actual and reference pad position in the wall frame (cm)
    pad_positions, _ = pad_from_states(state_history, pad_offset)
    actual_wall = 100 * to_wall_frame(pad_positions, wall_normal, wall_point)
    reference_wall = 100 * to_wall_frame(reference["pad_position"], wall_normal, wall_point)

    #Contact phase, shaded gray on every plot
    contact = reference["contact_flag"] > 0.5


    # ---------------- Position figure ----------------
    position_fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    position_fig.suptitle("Pad position")

    #Pad position: out of wall, along wall, up wall
    labels = ["out of wall (cm)", "along wall (cm)", "up wall (cm)"]
    for i in range(3):
        axes[i].plot(times, actual_wall[:, i], color=ACTUAL_COLOR, label="actual")
        axes[i].plot(times, reference_wall[:, i], color=REFERENCE_COLOR, linestyle="--", label="reference")
        axes[i].set_ylabel(labels[i])
    axes[0].axhline(0.0, color="black", linewidth=0.8)      # wall surface
    axes[0].legend(loc="upper right", fontsize=8)

    #Model's pad distance to the wall (mm), zoomed in near the wall, with the contact gate distance
    ax = axes[3]
    if model_pad_distance_history is not None:
        ax.plot(times, 1000 * model_pad_distance_history, color=ACTUAL_COLOR, label="model pad distance")
    ax.axhline(0.0, color="black", linewidth=0.8)
    distance_top = 20.0
    if contact_d0 is not None:
        ax.axhline(1000 * contact_d0, color="#e34948", linestyle="--", linewidth=1, label="gate d0")
        distance_top = max(distance_top, 4 * 1000 * contact_d0)
    ax.set_ylim(-5.0, distance_top)
    ax.set_ylabel("pad dist (mm)")
    ax.legend(loc="upper right", fontsize=8)

    shade_contact(axes, times, contact)
    position_fig.tight_layout()


    # ---------------- Force figure ----------------
    #Force plots are capped so impact spikes don't hide the tracking
    force_top = force_ylim if force_ylim is not None else max(2.5 * np.nanmax(reference["normal_force"]), 1.0)

    force_fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    force_fig.suptitle("Contact force")

    #Normal force: reference, MuJoCo, and what the model predicted (spikes above the cap marked with a triangle)
    ax = axes[0]
    ax.plot(times, reference["normal_force"], color=REFERENCE_COLOR, linestyle="--", label="reference")
    if normal_force_history is not None:
        ax.plot(times, normal_force_history, color=ACTUAL_COLOR, label="MuJoCo")
        mark_off_scale(ax, times, normal_force_history, force_top)
    if predicted_force_history is not None:
        ax.plot(times, predicted_force_history, color=MODEL_COLOR, linestyle=":", linewidth=1.5, label="model")
    ax.set_ylim(-0.1 * force_top, force_top)
    ax.set_ylabel("normal force (N)")
    ax.legend(loc="upper right", fontsize=8)

    #Friction along and up the wall: MuJoCo vs model (no reference, friction isn't commanded)
    friction_labels = ["friction along (N)", "friction up (N)"]
    for i in range(2):
        ax = axes[1 + i]
        if mujoco_friction_history is not None:
            mujoco_wall = to_wall_frame(mujoco_friction_history, wall_normal, np.zeros(3))
            ax.plot(times, mujoco_wall[:, i + 1], color=ACTUAL_COLOR, label="MuJoCo")
            mark_off_scale(ax, times, mujoco_wall[:, i + 1], force_top)
            mark_off_scale(ax, times, -mujoco_wall[:, i + 1], force_top, bottom=True)
        if predicted_friction_history is not None:
            predicted_wall = to_wall_frame(predicted_friction_history, wall_normal, np.zeros(3))
            ax.plot(times, predicted_wall[:, i + 1], color=MODEL_COLOR, linestyle=":", linewidth=1.5, label="model")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(-force_top, force_top)
        ax.set_ylabel(friction_labels[i])
    axes[1].legend(loc="upper right", fontsize=8)

    shade_contact(axes, times, contact)
    force_fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        position_fig.savefig(save_path.with_name(save_path.stem + "_position.png"), dpi=150)
        force_fig.savefig(save_path.with_name(save_path.stem + "_force.png"), dpi=150)
    plt.show()


def shade_contact(axes, times, contact):
    #Shades the contact phase gray and labels the time axis
    for ax in axes:
        ax.fill_between(times, 0, 1, where=contact, transform=ax.get_xaxis_transform(),
                        color="gray", alpha=0.15, linewidth=0)
    axes[-1].set_xlabel("time (s)")


def mark_off_scale(ax, times, values, limit, bottom=False):
    #Marks values beyond the plot's limit with a triangle at the edge (top, or bottom for negatives)
    off_scale = values > limit
    if np.any(off_scale):
        edge = -limit if bottom else limit
        ax.plot(times[off_scale], np.full(off_scale.sum(), edge), linestyle="none",
                marker="v" if bottom else "^", markersize=5, color=ACTUAL_COLOR, clip_on=False)
