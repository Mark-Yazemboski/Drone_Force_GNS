#Re-makes the plots for a finished run from its run_data.npz, without rerunning the MPC.
#Works for MPC runs and the geometric baseline; anything an older run didn't save is just left off the plots.
#   python replot_run.py push_test_N50                    (a run name in MPC_runs)
#   python replot_run.py push_test_N50 --force_max 2      (top of the force plots, N)
#   python replot_run.py push_test_N50 --lookahead 20     (MPC plan line this many steps ahead)
#   python replot_run.py push_test_N50 --fan_every 0.25   (draw a full-horizon plan every 0.25 s)
#   python replot_run.py push_test_N50 --gif              (also remake the flight GIF)
#   python replot_run.py push_test_N50 --no_show          (just save the figures)

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import Drone_Dynamics
from scenario_generator import Trajectory_Generator
from visualize import plot_tracking, animate_flight

SCRIPT_FOLDER = Path(__file__).resolve().parent
Trained_model_path = SCRIPT_FOLDER / "MPC_Model" / "Trained_Model.pt"

parser = argparse.ArgumentParser()
parser.add_argument("name", help="name of the run's folder in MPC_runs")
parser.add_argument("--force_max", type=float, default=None, help="top of the force plots (N)")
parser.add_argument("--lookahead", type=int, default=10, help="steps ahead for the MPC plan line")
parser.add_argument("--fan_every", type=float, default=0.5, help="seconds between full-horizon plans")
parser.add_argument("--gif", action="store_true", help="also remake the flight GIF")
parser.add_argument("--no_show", action="store_true", help="save the figures without opening them")
args = parser.parse_args()

#Loads the saved run
run_folder = SCRIPT_FOLDER / "MPC_runs" / args.name
data = np.load(run_folder / "run_data.npz")

def saved(key):
    #Older runs (and the geometric baseline) don't have everything, so missing keys are None
    return data[key] if key in data.files else None

wall_normal_vector = data["wall_normal_vector"]
wall_center_point = data["wall_center_point"]
simulation_dt = float(data["simulation_dt"])
times = data["times"]
states = data["states"]

#The model gives the drone geometry (pad offset)
model = Drone_Dynamics(torch.device("cpu"), Trained_model_path, wall_normal_vector=wall_normal_vector,
                       wall_center_point=wall_center_point, wind_vector=data["wind_vector"])
pad_offset = model.pad_offset.cpu().numpy()

#MPC timestep: saved in newer runs, otherwise from settings.json (older runs, simulation_dt * MPC_dt_multiple)
mpc_dt = saved("mpc_dt")
settings_file = run_folder / "settings.json"
if mpc_dt is None and settings_file.exists():
    settings = json.loads(settings_file.read_text())
    mpc_dt = settings["simulation_dt"] * settings["MPC_dt_multiple"]

#MPC contact gate center: saved in newer runs, otherwise the value the run file uses
mpc_gate_d0 = saved("mpc_gate_d0")
if mpc_gate_d0 is None:
    mpc_gate_d0 = model.cfg.pad_contact_radius + 0.002

#Inputs line up with the states (inputs[i] applied from times[i] to times[i+1]), so pad the last one with NaN
inputs = saved("inputs")
if inputs is not None and len(inputs) < len(times):
    inputs = np.vstack([inputs, np.full((len(times) - len(inputs), inputs.shape[1]), np.nan)])

#Rebuilds the reference from the saved copy, so it can be sampled at the run's times
trajectory_generator = Trajectory_Generator((0, 0), (0, 0), 0.1)
trajectory_generator.reference_time = data["reference_time"]
trajectory_generator.reference_trajectory = {key[len("reference_"):]: data[key] for key in data.files
                                             if key.startswith("reference_") and key != "reference_time"}

plot_tracking(times, states, trajectory_generator.sample(times), pad_offset, wall_normal_vector, wall_center_point,
              normal_force_history=saved("normal_force"),
              predicted_force_history=saved("predicted_force"),
              mujoco_friction_history=saved("mujoco_friction"),
              predicted_friction_history=saved("predicted_friction"),
              model_pad_distance_history=saved("model_pad_distance"),
              contact_d0=float(mpc_gate_d0),
              input_history=inputs, hover_input=saved("hover_input"),
              planned_outputs=saved("planned_outputs"), mpc_dt=None if mpc_dt is None else float(mpc_dt),
              plan_lookahead=args.lookahead, plan_fan_every=args.fan_every,
              force_ylim=args.force_max, save_path=run_folder / "tracking.png", show=not args.no_show)
print("Saved plots to", run_folder)

if args.gif:
    animate_flight(model.cfg, wall_normal_vector, wall_center_point, states, dt=simulation_dt,
                   reference_positions=trajectory_generator.sample(times)["pad_position"],
                   pad_offset=pad_offset, save_path=run_folder / "flight.gif")
