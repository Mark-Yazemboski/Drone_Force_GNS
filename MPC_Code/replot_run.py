#Re-makes the plots for a saved MPC run from its run_data.npz, without rerunning the MPC.
#   python replot_run.py push_test_N50                  (a run name in MPC_runs)
#   python replot_run.py push_test_N50 --force_max 2    (set the top of the force plot, N)
#   python replot_run.py push_test_N50 --gif            (also remake the flight GIF)

import argparse
from pathlib import Path

import numpy as np
import torch

from model import Drone_Dynamics
from scenario_generator import Trajectory_Generator
from visualize import plot_tracking, animate_flight

SCRIPT_FOLDER = Path(__file__).resolve().parent
Trained_model_path = SCRIPT_FOLDER / "MPC_Model" / "Trained_Model.pt"

run_name = "testing_freeflight"

#Loads the saved run
run_folder = SCRIPT_FOLDER / "MPC_runs" / run_name
data = np.load(run_folder / "run_data.npz")

wall_normal_vector = data["wall_normal_vector"]
wall_center_point = data["wall_center_point"]
simulation_dt = float(data["simulation_dt"])
times = data["times"]
states = data["states"]

#inputs[i] is applied between states[i] and states[i+1], so there's one fewer input than states
inputs = np.vstack([data["inputs"], np.full((1, data["inputs"].shape[1]), np.nan)])

#The model gives the drone geometry (pad offset) and the contact gate distance
model = Drone_Dynamics(torch.device("cpu"), Trained_model_path, wall_normal_vector=wall_normal_vector,
                       wall_center_point=wall_center_point, wind_vector=data["wind_vector"])
pad_offset = model.pad_offset.cpu().numpy()

#Rebuilds the reference from the saved copy, so it can be sampled at the run's times
trajectory_generator = Trajectory_Generator((0, 0), (0, 0), 0.1)
trajectory_generator.reference_time = data["reference_time"]
trajectory_generator.reference_trajectory = {key[len("reference_"):]: data[key] for key in data.files
                                             if key.startswith("reference_") and key != "reference_time"}

plot_tracking(times, states, trajectory_generator.sample(times), pad_offset, wall_normal_vector, wall_center_point,
              normal_force_history=data["normal_force"], input_history=inputs,
              predicted_force_history=data["predicted_force"],
              model_pad_distance_history=data["model_pad_distance"],
              contact_d0=float(model.model.contact_d0), force_ylim=None,
              save_path=run_folder / "tracking.png")

animate_flight(model.cfg, wall_normal_vector, wall_center_point, states, dt=simulation_dt,
                reference_positions=trajectory_generator.reference_trajectory["pad_position"],
                pad_offset=pad_offset, save_path=run_folder / "flight.gif")