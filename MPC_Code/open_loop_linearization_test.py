"""
Open-loop prediction test: is the linearization a good enough approximation?

No controller in the loop. Takes a saved MuJoCo run (MPC_runs/<name>/run_data.npz) and, at a few start
points, predicts N steps ahead from the measured state using:
    - the nonlinear GNN rollout (the model's "truth")
    - linear models re-linearized every X steps about the GNN rollout
      (X = 1 is LTV, which is what the MPC does; X = N is frozen at the first step)

Two comparisons:
    - linear vs GNN:   linearization error
    - GNN vs MuJoCo:   model error (context: if the linearization error is much smaller, it's not the bottleneck)

Each start point is run twice:
    - Recorded inputs: the inputs the controller actually applied. LTV is exact here by construction
      (it's linearized about this exact rollout), so it's only a sanity check that should read ~0.
    - Perturbed inputs: the linear models are still linearized about the recorded inputs, but are fed
      inputs perturbed by about the trust region size. This is what the MPC actually does each solve.

Run:  python -u open_loop_linearization_test.py --run geometric_baseline
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from model import Drone_Dynamics


SCRIPT_FOLDER = Path(__file__).resolve().parent
Trained_model_path = SCRIPT_FOLDER / "MPC_Model" / "Trained_Model.pt"

#---------------------------------- Settings ----------------------------------------

HORIZON_LENGTH = 50             #prediction steps (one model step each)
MPC_DT_MULTIPLE = 1             #keep at 1: the recorded inputs change every sim step, so they can't be held over longer MPC steps
INPUT_PERTURBATION = 20.0       #rad/s, max perturbation per rotor per step. Set this to INPUT_TRUST_REGION from the MPC script
CONTACT_FORCE_THRESHOLD = 0.1   #N, MuJoCo normal force above this counts as contact
RANDOM_SEED = 0                 #for the input perturbations

#Linearization schedules: re-linearize every X steps (X = 1 is LTV, X = horizon length is frozen)
SCHEDULES = {"LTV": 1, "every 5": 5, "every 10": 10, "frozen": HORIZON_LENGTH}

#Rows of the output vector used here (same packing as model.outputs)
NORMAL_FORCE = 0
TILT = slice(1, 3)              #roll, pitch
PAD_POSITION = slice(3, 6)

ERROR_LABELS = {"pad position": "Pad position error (mm)",
                "normal force": "Normal force error (N)",
                "tilt": "Tilt error (deg)"}

device = torch.device("cpu")


#---------------------------------- Helpers ----------------------------------------

def to_torch(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)


def to_np(value):
    return value.detach().cpu().numpy()


def quat_to_rotation(quaternion):
    #The state stores quaternions as wxyz, scipy wants xyzw
    w, x, y, z = quaternion
    return Rotation.from_quat([x, y, z, w])


def state_error(state, reference_state):
    #18-number error state between two 19-number states (same convention as model.state_error: R = exp(dtheta) R_ref)
    rotation_error = quat_to_rotation(state[6:10]) * quat_to_rotation(reference_state[6:10]).inv()
    dtheta = rotation_error.as_rotvec()
    return np.concatenate([state[0:6] - reference_state[0:6], dtheta, state[10:19] - reference_state[10:19]])


def perturb(state, dx):
    #Applies an 18-number error state to a 19-number state (same convention as model.perturb)
    rotation = Rotation.from_rotvec(dx[6:9]) * quat_to_rotation(state[6:10])
    x, y, z, w = rotation.as_quat()
    return np.concatenate([state[0:6] + dx[0:6], [w, x, y, z], state[10:19] + dx[9:18]])


#---------------------------------- Predictions ----------------------------------------

def linearize_along(model, initial_state, nominal_inputs):
    #Linearizes the GNN at every step of its own rollout under the nominal inputs
    #Each linearization is [A, B, C, D, next nominal state, nominal outputs]
    nominal_states = [initial_state]
    linearizations = []
    for rotor_speeds in nominal_inputs:
        A, B, C, D, next_state, nominal_output = model.linearize(to_torch(nominal_states[-1]), to_torch(rotor_speeds), MPC_DT_MULTIPLE)
        linearizations.append([to_np(A), to_np(B), to_np(C), to_np(D), to_np(next_state), to_np(nominal_output)])
        nominal_states.append(to_np(next_state))
    return nominal_states, linearizations


def linear_prediction(nominal_states, nominal_inputs, linearizations, inputs, refresh_every):
    #Rolls out the linear model, re-linearizing (about the GNN nominal) every refresh_every steps
    #Between refreshes the same linearization point and A, B, C, D are used, so the model is affine about that point
    state = nominal_states[0]
    outputs = []
    for k in range(len(inputs)):
        #Step whose linearization is in use at step k
        j = refresh_every * (k // refresh_every)
        A, B, C, D, next_nominal_state, nominal_output = linearizations[j]

        dx = state_error(state, nominal_states[j])
        du = inputs[k] - nominal_inputs[j]

        outputs.append(nominal_output + C @ dx + D @ du)
        state = perturb(next_nominal_state, A @ dx + B @ du)
    return np.array(outputs)


def gnn_rollout(model, initial_state, inputs):
    #Nonlinear GNN rollout. Outputs at steps 0..N-1 (F_n at step k comes from x_k and u_k)
    state = to_torch(initial_state)
    outputs = []
    with torch.no_grad():
        for rotor_speeds in inputs:
            next_state, _, info = model.propagate_multi(state, to_torch(rotor_speeds), MPC_DT_MULTIPLE)
            outputs.append(to_np(model.outputs(state, info["F_n"])))
            state = next_state
    return np.array(outputs)


def mujoco_outputs(model, states, normal_force):
    #The same outputs from the recorded MuJoCo states
    #normal_force[k+1] was recorded after stepping from state k to k+1, so it's the force over step k
    outputs = []
    with torch.no_grad():
        for k in range(len(states) - 1):
            outputs.append(to_np(model.outputs(to_torch(states[k]), to_torch(normal_force[k + 1]))))
    return np.array(outputs)


#---------------------------------- Metrics ----------------------------------------

def prediction_errors(outputs, reference):
    #Error of outputs against reference at each horizon step
    pad_error = np.linalg.norm(outputs[:, PAD_POSITION] - reference[:, PAD_POSITION], axis=1)
    force_error = np.abs(outputs[:, NORMAL_FORCE] - reference[:, NORMAL_FORCE])
    tilt_error = np.linalg.norm(outputs[:, TILT] - reference[:, TILT], axis=1)
    return {"pad position": 1000 * pad_error,
            "normal force": force_error,
            "tilt": np.degrees(tilt_error)}


def plot_errors(axes_column, errors, label, **style):
    for ax, values in zip(axes_column, errors.values()):
        ax.plot(values, label=label, **style)


def print_summary(label, errors):
    #Mean error over the horizon
    print(f"    {label:>14}:  pad {errors['pad position'].mean():8.2f} mm   "
          f"force {errors['normal force'].mean():7.3f} N   tilt {errors['tilt'].mean():7.3f} deg")


def pick_start_steps(normal_force, run_length):
    #Start points in free flight, just before contact onset (so the horizon crosses it), and mid contact
    last_start = run_length - HORIZON_LENGTH
    contact_steps = np.where(normal_force > CONTACT_FORCE_THRESHOLD)[0]

    if len(contact_steps) == 0:
        print("No contact in this run, only testing free flight")
        return {"free flight": last_start // 2}

    onset = contact_steps[0]
    start_steps = {"free flight": onset // 2,
                   "contact onset": onset - HORIZON_LENGTH // 2,
                   "mid contact": contact_steps[len(contact_steps) // 2]}
    return {name: int(np.clip(step, 0, last_start)) for name, step in start_steps.items()}


#---------------------------------- Main ----------------------------------------

name = "Geometric_Baseline"

run_folder = SCRIPT_FOLDER / "MPC_runs" / name
output_folder = run_folder / "open_loop"
output_folder.mkdir(exist_ok=True)

#states[i] -> inputs[i] -> states[i+1]
data = np.load(run_folder / "run_data.npz")
states = data["states"]
inputs = data["inputs"]
normal_force = data["normal_force"]
dt = float(data["simulation_dt"])

model = Drone_Dynamics(device, Trained_model_path,
                       wall_normal_vector=data["wall_normal_vector"],
                       wall_center_point=data["wall_center_point"],
                       wind_vector=data["wind_vector"])

rng = np.random.default_rng(RANDOM_SEED)
start_steps = pick_start_steps(normal_force, len(inputs))

for start_name, s in start_steps.items():
    print(f"\n{start_name} (start step {s})")

    recorded_inputs = inputs[s:s + HORIZON_LENGTH]
    perturbed_inputs = recorded_inputs + rng.uniform(-INPUT_PERTURBATION, INPUT_PERTURBATION, size=recorded_inputs.shape)

    #Linearized once about the GNN rollout under the recorded inputs, then shared by every schedule and both cases
    nominal_states, linearizations = linearize_along(model, states[s], recorded_inputs)

    #The nominal outputs from linearize are the GNN rollout under the recorded inputs
    gnn_recorded = np.array([linearization[5] for linearization in linearizations])
    gnn_perturbed = gnn_rollout(model, states[s], perturbed_inputs)
    mujoco = mujoco_outputs(model, states[s:s + HORIZON_LENGTH + 1], normal_force[s:s + HORIZON_LENGTH + 1])

    cases = {"Recorded inputs": (recorded_inputs, gnn_recorded),
             f"Perturbed inputs (±{INPUT_PERTURBATION:g} rad/s)": (perturbed_inputs, gnn_perturbed)}

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)

    for column, (case_name, (case_inputs, gnn_outputs)) in enumerate(cases.items()):
        print(f"  {case_name}")

        #Linearization error
        for schedule_name, refresh_every in SCHEDULES.items():
            linear_outputs = linear_prediction(nominal_states, recorded_inputs, linearizations, case_inputs, refresh_every)
            errors = prediction_errors(linear_outputs, gnn_outputs)
            plot_errors(axes[:, column], errors, f"{schedule_name} vs GNN")
            print_summary(schedule_name, errors)

        #Model error (MuJoCo was only run with the recorded inputs)
        if column == 0:
            errors = prediction_errors(gnn_outputs, mujoco)
            plot_errors(axes[:, column], errors, "GNN vs MuJoCo", color="black", linestyle="--")
            print_summary("GNN vs MuJoCo", errors)

        axes[0, column].set_title(case_name)

    for row, label in enumerate(ERROR_LABELS.values()):
        axes[row, 0].set_ylabel(label)
        for column in range(2):
            axes[row, column].set_ylim(bottom=0)
            axes[row, column].grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel(f"Horizon step k ({dt:g} s each)")

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Open-loop prediction error: {start_name} (start step {s})")
    fig.tight_layout()
    fig.savefig(output_folder / f"{start_name.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

print(f"\nSaved plots to {output_folder}")
