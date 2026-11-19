"""
Phase 1 model checks for the MPC's prediction model (the GNN), on a saved run:

1. Linearization: nonlinear GNN vs its linearizations, open loop from the same state with the same input step.
   - LTV: re-linearized at every step along the nominal trajectory (what the MPC uses)
   - LTI: linearized once at the start point
   The inputs are the run's rotor speeds plus a constant step of up to +-INPUT_STEP rad/s per rotor. The error is shown
   relative to the change the step causes in the nonlinear response (that change is what a linearization has to
   predict), with ERROR_THRESHOLD marked.

2. Discretization / update rate: the continuous plant (MuJoCo, 0.5 ms physics) vs the discrete GNN at sample times
   Ts = 0.01, 0.02, 0.05 s. The run's commands are held for Ts (zero-order hold) in both; the GNN steps Ts at a time
   with the motor-lagged rotor speed averaged over Ts (as the MPC would with MPC_dt = Ts).

Each check runs from two start points: free flight (halfway to first contact) and mid contact.
Plots go to MPC_runs/<run>/phase1/.

Run:  python phase1_model_checks.py final_v1234
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from model import Drone_Dynamics
from mujoco_simulator import MujocoDroneSim
import MPC_controller
from MPC_controller import motor_step
from utilities import quat_to_rot_matrix, quat_to_rot_matrix_np
from visualize import (set_plot_style, figure_legend, save_figures, to_wall_frame, REPORT_WIDTH,
                       ACTUAL_COLOR, MODEL_COLOR)

SCRIPT_FOLDER = Path(__file__).resolve().parent
Trained_model_path = SCRIPT_FOLDER / "MPC_Model" / "Trained_Model.pt"

#---------------------------------- Settings ----------------------------------------

CONTACT_CUTOFF_GAP = 0.001          #same as the MPC run file
HORIZON_LENGTH = 50                 #steps of 0.01 s, the MPC horizon
INPUT_STEP = 20.0                   #rad/s, largest input step per rotor for the linearization check
ERROR_THRESHOLD = 0.02              #marked on the linearization error plots (fraction of the step response)
SAMPLE_TIMES = [0.01, 0.02, 0.05]   #s, discretizations compared (multiples of the GNN's 0.01 s step)
CONTACT_FORCE = 0.05                #N, MuJoCo normal force above this counts as contact
RANDOM_SEED = 0

#Error floors, so a response the step barely moves doesn't turn into a huge percentage
POSITION_FLOOR = 0.001              #m
FORCE_FLOOR = 0.05                  #N

LTI_COLOR = "0.45"                  #gray
SAMPLE_COLORS = [MODEL_COLOR, "#e34948", "#8c564b"]

#Rows of the 17-number output vector (same packing as model.outputs)
NORMAL_FORCE, PAD_POSITION = 0, slice(3, 6)


#---------------------------------- Setup ----------------------------------------

run_folder = SCRIPT_FOLDER / "MPC_runs" / sys.argv[1]
output_folder = run_folder / "phase1"
output_folder.mkdir(exist_ok=True)

data = np.load(run_folder / "run_data.npz")
states, commands = data["states"], data["inputs"]
dt = float(data["simulation_dt"])
wall_normal, wall_point = data["wall_normal_vector"], data["wall_center_point"]

model = Drone_Dynamics(torch.device("cpu"), Trained_model_path, wall_normal_vector=wall_normal,
                       wall_center_point=wall_point, wind_vector=data["wind_vector"],
                       contact_cutoff_gap=CONTACT_CUTOFF_GAP)
model_steps_per_dt = round(dt / model.dt)

def to_torch(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)

def motor_state_at(step):
    #Squared rotor speeds at the start of a step, from the commands sent before it (the sim starts at hover)
    speed_squared = data["hover_input"] ** 2
    for command in commands[:step]:
        _, speed_squared = motor_step(speed_squared, command, dt)
    return speed_squared

def lagged(command_sequence, speed_squared, step_time):
    #Effective rotor speeds the model sees for each held command, and the motor state after them
    effective = []
    for command in command_sequence:
        speed, speed_squared = motor_step(speed_squared, command, step_time)
        effective.append(speed)
    return np.array(effective), speed_squared

def gnn_outputs(state, inputs, model_steps):
    #Nonlinear GNN from one state: outputs at each step (F_n is the average over the step), each input held model_steps
    state, outputs = to_torch(state).unsqueeze(0), []
    with torch.no_grad():
        for rotor_speeds in inputs:
            next_state, _, info = model.propagate_multi(state, to_torch(rotor_speeds).unsqueeze(0), model_steps)
            outputs.append(model.outputs(state, info["F_n"])[0].numpy())
            state = next_state
    return np.array(outputs)

def pad_and_force(outputs, origin):
    #Pad position in the wall frame [normal, along, up] relative to origin (mm), and the normal force (N)
    return 1000 * to_wall_frame(outputs[:, PAD_POSITION], wall_normal, origin), outputs[:, NORMAL_FORCE]

#Start points: halfway to the first contact, and the middle of the contact phase
in_contact = np.flatnonzero(data["normal_force"][1:len(commands) + 1] > CONTACT_FORCE)
if len(in_contact) == 0:
    sys.exit("No contact in this run")
last_start = len(commands) - HORIZON_LENGTH - round(max(SAMPLE_TIMES) / dt)
start_points = {"free flight": in_contact[0] // 2, "contact": in_contact[len(in_contact) // 2]}
start_points = {name: int(np.clip(step, 0, last_start)) for name, step in start_points.items()}

set_plot_style()
prediction_time = np.arange(HORIZON_LENGTH) * dt
rng = np.random.default_rng(RANDOM_SEED)
position_labels = ["normal (mm)", "along (mm)", "up (mm)"]


#---------------------------------- 1. Linearization ----------------------------------------

def linearized_outputs(x0, nominal_inputs, inputs):
    #LTV (re-linearized at every step of the nominal trajectory) and LTI (linearized once at the start) predictions
    nominal_states = model.rollout(to_torch(x0), to_torch(nominal_inputs), 1)
    A, B, C, D, x_next, y_nominal = [m.numpy().astype(float) for m in
                                     model.linearize_horizon(nominal_states[:-1], to_torch(nominal_inputs), 1)]
    du = inputs - nominal_inputs

    #LTV: deviations from the nominal trajectory, dx_{k+1} = A_k dx_k + B_k du_k
    ltv, dx = [], np.zeros(A.shape[1])
    for k in range(len(du)):
        ltv.append(y_nominal[k] + C[k] @ dx + D[k] @ du[k])
        dx = A[k] @ dx + B[k] @ du[k]

    #LTI: deviations from the start point (x0, u0), with the one-step drift f(x0, u0) - x0 as the affine term
    x0_t, x1_t = to_torch(x0).unsqueeze(0), to_torch(x_next[0]).unsqueeze(0)
    drift = model.state_error(x1_t, x0_t, quat_to_rot_matrix(x1_t[:, 6:10]),
                              quat_to_rot_matrix(x0_t[:, 6:10]))[0].numpy().astype(float)
    lti, dx = [], np.zeros(A.shape[1])
    for k in range(len(du)):
        du0 = inputs[k] - nominal_inputs[0]
        lti.append(y_nominal[0] + C[0] @ dx + D[0] @ du0)
        dx = drift + A[0] @ dx + B[0] @ du0
    return np.array(ltv), np.array(lti), y_nominal

summary = []
for name, s in start_points.items():
    nominal_inputs, _ = lagged(commands[s:s + HORIZON_LENGTH], motor_state_at(s), dt)
    inputs = nominal_inputs + rng.uniform(-INPUT_STEP, INPUT_STEP, size=4)       #constant step, held over the horizon
    nonlinear = gnn_outputs(states[s], inputs, model_steps_per_dt)
    ltv, lti, nominal = linearized_outputs(states[s], nominal_inputs, inputs)

    origin = states[s][0:3] + quat_to_rot_matrix_np(states[s][6:10]) @ model.pad_offset.numpy()
    curves = {label: pad_and_force(outputs, origin) for label, outputs in
              [("nonlinear GNN", nonlinear), ("LTV", ltv), ("LTI", lti), ("nominal", nominal)]}

    #Response figure: pad position (wall frame) and normal force, nonlinear vs linearized
    fig, axes = plt.subplots(4, 1, figsize=(REPORT_WIDTH, 4.6), sharex=True)
    for label, color, style in [("nonlinear GNN", ACTUAL_COLOR, "-"), ("LTV", MODEL_COLOR, ":"), ("LTI", LTI_COLOR, "--")]:
        position, force = curves[label]
        for i in range(3):
            axes[i].plot(prediction_time, position[:, i], color=color, linestyle=style, linewidth=1.2, label=label)
        axes[3].plot(prediction_time, force, color=color, linestyle=style, linewidth=1.2, label=label)
    for ax, label in zip(axes, position_labels + ["$F_n$ (N)"]):
        ax.set_ylabel(label)
    axes[-1].set_xlabel("prediction time (s)")
    axes[-1].set_xlim(prediction_time[0], prediction_time[-1])
    figure_legend(fig, axes)

    #Error figure: error relative to the change the input step causes in the nonlinear response
    nonlinear_position, nonlinear_force = curves["nonlinear GNN"]
    nominal_position, nominal_force = curves["nominal"]
    position_scale = max(np.linalg.norm(nonlinear_position - nominal_position, axis=1).max(), 1000 * POSITION_FLOOR)
    force_scale = max(np.abs(nonlinear_force - nominal_force).max(), FORCE_FLOOR)

    error_fig, error_axes = plt.subplots(2, 1, figsize=(REPORT_WIDTH, 2.6), sharex=True)
    for label, color, style in [("LTV", MODEL_COLOR, "-"), ("LTI", LTI_COLOR, "--")]:
        position, force = curves[label]
        position_error = 100 * np.linalg.norm(position - nonlinear_position, axis=1) / position_scale
        force_error = 100 * np.abs(force - nonlinear_force) / force_scale
        error_axes[0].plot(prediction_time, position_error, color=color, linestyle=style, label=label)
        error_axes[1].plot(prediction_time, force_error, color=color, linestyle=style, label=label)
        summary.append(f"  {name:12s} {label}: pad error max {position_error.max():6.1f}% "
                       f"(of {position_scale:.1f} mm), force error max {force_error.max():6.1f}% (of {force_scale:.2f} N)")
    for ax, label in zip(error_axes, ["pad error (%)", "$F_n$ error (%)"]):
        ax.axhline(100 * ERROR_THRESHOLD, color="black", linewidth=0.6, linestyle="--",
                   label=f"{100 * ERROR_THRESHOLD:g}% threshold")
        ax.set_ylabel(label)
        ax.set_yscale("log")                #LTI errors can be orders of magnitude above LTV's
        ax.set_ylim(bottom=0.1)
    error_axes[-1].set_xlabel("prediction time (s)")
    error_axes[-1].set_xlim(prediction_time[0], prediction_time[-1])
    figure_legend(error_fig, error_axes)

    key = name.replace(" ", "_")
    save_figures({f"{key}_response": fig, f"{key}_error": error_fig}, output_folder / "linearization.png")
    plt.close(fig)
    plt.close(error_fig)

print("Linearization (error relative to the input step's effect on the nonlinear response):")
print("\n".join(summary))


#---------------------------------- 2. Discretization / update rate ----------------------------------------

sim = MujocoDroneSim(model.cfg, wall_normal, wall_point, data["wind_vector"], motor_tau=MPC_controller.MOTOR_TAU)

def mujoco_branch(start, held_commands):
    #Replays the run's commands up to start (reproduces the run exactly), then applies held_commands (one per 0.01 s)
    #Returns the state at start, and the outputs at every 0.01 s after it: pad position from the state, and the
    #normal force averaged over each 0.01 s
    state = sim.reset(data["initial_state"], data["hover_input"])
    for command in commands[:start]:
        state = sim.step(command, dt)
    if np.abs(state - states[start]).max() > 1e-6:
        print("WARNING: the MuJoCo replay didn't reproduce the run, check the simulator settings")
    start_state, pads, forces = state.copy(), [], []
    for command in held_commands:
        pads.append(state[0:3] + quat_to_rot_matrix_np(state[6:10]) @ model.pad_offset.numpy())
        state = sim.step(command, dt)
        forces.append(sim.get_normal_force())
    return start_state, np.array(pads), np.array(forces)

summary = []
for name, s in start_points.items():
    origin = None
    fig, axes = plt.subplots(4, 1, figsize=(REPORT_WIDTH, 4.6), sharex=True)
    error_fig, error_axes = plt.subplots(2, 1, figsize=(REPORT_WIDTH, 2.6), sharex=True)

    for sample_time, color in zip(SAMPLE_TIMES, SAMPLE_COLORS):
        hold = round(sample_time / dt)
        samples = HORIZON_LENGTH // hold + 1
        held = commands[s:s + samples * hold:hold]                         #command at the start of each sample

        #Continuous plant: MuJoCo with each command held for sample_time
        start_state, mujoco_pad, mujoco_force = mujoco_branch(s, np.repeat(held, hold, axis=0)[:HORIZON_LENGTH])
        if origin is None:
            origin = mujoco_pad[0]
        mujoco_position = 1000 * to_wall_frame(mujoco_pad, wall_normal, origin)

        #Discrete GNN at sample_time: one model call per sample, lagged speed averaged over the sample
        effective, _ = lagged(held, motor_state_at(s), sample_time)
        gnn_position, gnn_force = pad_and_force(gnn_outputs(start_state, effective, hold * model_steps_per_dt), origin)
        sample_times = np.arange(len(held)) * sample_time
        keep = sample_times < HORIZON_LENGTH * dt
        sample_times, gnn_position, gnn_force = sample_times[keep], gnn_position[keep], gnn_force[keep]

        #MuJoCo at the sample times (force averaged over each sample, like the GNN's)
        index = (sample_times / dt).round().astype(int)
        mujoco_sampled_force = np.array([mujoco_force[i:i + hold].mean() for i in index])
        position_error = np.linalg.norm(gnn_position - mujoco_position[index], axis=1)
        force_error = np.abs(gnn_force - mujoco_sampled_force)

        label = f"$T_s$ = {1000 * sample_time:g} ms"
        for i in range(3):
            axes[i].plot(prediction_time, mujoco_position[:, i], color=color, linewidth=0.8)
            axes[i].plot(sample_times, gnn_position[:, i], color=color, linestyle="none", marker="o", markersize=2)
        axes[0].plot([], [], color=color, linewidth=1.5, label=label)
        axes[3].plot(prediction_time, mujoco_force, color=color, linewidth=0.8)
        axes[3].plot(sample_times, gnn_force, color=color, linestyle="none", marker="o", markersize=2)
        error_axes[0].plot(sample_times, position_error, color=color, marker="o", markersize=2, label=label)
        error_axes[1].plot(sample_times, force_error, color=color, marker="o", markersize=2, label=label)
        summary.append(f"  {name:12s} Ts {1000 * sample_time:3.0f} ms: pad error RMS {np.sqrt(np.mean(position_error ** 2)):5.2f} mm, "
                       f"max {position_error.max():5.2f} mm | force error RMS {np.sqrt(np.mean(force_error ** 2)):.3f} N")

    for ax, label in zip(axes, position_labels + ["$F_n$ (N)"]):
        ax.set_ylabel(label)
    for ax, label in zip(error_axes, ["pad error (mm)", "$F_n$ error (N)"]):
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    for figure_axes in (axes, error_axes):
        figure_axes[-1].set_xlabel("prediction time (s)")
        figure_axes[-1].set_xlim(prediction_time[0], prediction_time[-1])
    #Line style says which model, color says which sample time
    axes[0].add_line(Line2D([], [], color="black", linewidth=0.8, label="MuJoCo (continuous)"))
    axes[0].add_line(Line2D([], [], color="black", linestyle="none", marker="o", markersize=2, label="GNN (discrete)"))
    figure_legend(fig, axes, ncol=3)
    figure_legend(error_fig, error_axes)

    key = name.replace(" ", "_")
    save_figures({f"{key}_response": fig, f"{key}_error": error_fig}, output_folder / "discretization.png")
    plt.close(fig)
    plt.close(error_fig)

print("Discretization (discrete GNN vs continuous MuJoCo, commands held for Ts):")
print("\n".join(summary))
print("Saved plots to", output_folder)
