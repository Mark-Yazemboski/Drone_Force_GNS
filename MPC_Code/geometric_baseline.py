"""
Baseline run: the geometric (Lee et al. 2010) controller flying the MPC's reference trajectory in MuJoCo.

Same controller as the training-data generator, without the excitation noise and gain randomization.
Saves MPC_runs/<name>/run_data.npz in the same format as the MPC runs (so open_loop_linearization_test.py
can use it), plus a tracking plot.

Use the same --seed as an MPC run to fly the same wall, wind, and reference:
    python -u geometric_baseline.py --name geometric_seed3 --seed 3
"""

import argparse
from pathlib import Path

import numpy as np

from model import load_checkpoint
from mujoco_simulator import MujocoDroneSim
from scenario_generator import Trajectory_Generator
from utilities import quat_to_rot_matrix_np
from visualize import plot_tracking, animate_flight


SCRIPT_FOLDER = Path(__file__).resolve().parent
Trained_model_path = SCRIPT_FOLDER / "MPC_Model" / "Trained_Model.pt"

#---------------------------------- Settings ----------------------------------------

simulation_dt = 0.01                #controller rate (100 Hz), same as the MPC script's simulation_dt

POSITION_BANDWIDTH = 4.0            #rad/s
ATTITUDE_BANDWIDTH = 15.0           #rad/s, roll and pitch
YAW_BANDWIDTH = 3.0                 #rad/s, slower since yaw authority is small (wasn't in the generator notes, so chosen here)
DAMPING = 0.9
MAX_TILT = np.radians(35.0)

#Rotor thrust limits as fractions of the per-rotor hover thrust (m g / 4)
MIN_THRUST_FRACTION = 0.05
MAX_THRUST_FRACTION = 2.5

E3 = np.array([0.0, 0.0, 1.0])


#---------------------------------- Controller ----------------------------------------

def vee(skew_matrix):
    #Skew-symmetric matrix -> vector
    return np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])


class GeometricController:

    def __init__(self, sim, heading):
        cfg = sim.cfg
        self.mass = sim.mass
        self.gravity = cfg.gravity
        self.k_f = cfg.k_f

        #Inertia about the COM in body axes (MuJoCo stores the principal moments and the rotation to them)
        principal_rotation = quat_to_rot_matrix_np(sim.mj_model.body_iquat[sim.drone_id])
        principal_moments = sim.mj_model.body_inertia[sim.drone_id]
        self.inertia = principal_rotation @ np.diag(principal_moments) @ principal_rotation.T

        #Gains
        self.k_x = POSITION_BANDWIDTH ** 2
        self.k_v = 2 * DAMPING * POSITION_BANDWIDTH
        bandwidths = np.array([ATTITUDE_BANDWIDTH, ATTITUDE_BANDWIDTH, YAW_BANDWIDTH])
        self.K_R = bandwidths ** 2 * np.diag(self.inertia)
        self.K_Omega = 2 * DAMPING * bandwidths * np.diag(self.inertia)

        #Horizontal direction the body x axis should point along
        self.heading = heading

        #Mixer: [total thrust, roll torque, pitch torque, yaw torque] = mixer @ rotor thrusts
        rotor_positions = sim.rotor_positions           #relative to the COM, body frame
        spin = np.asarray(cfg.spin_dir, dtype=float)
        self.mixer = np.vstack([np.ones(4),
                                rotor_positions[:, 1],
                                -rotor_positions[:, 0],
                                -spin * cfg.k_m / cfg.k_f])

        hover_thrust = self.mass * self.gravity / 4
        self.thrust_min = MIN_THRUST_FRACTION * hover_thrust
        self.thrust_max = MAX_THRUST_FRACTION * hover_thrust


    def compute(self, state, com_reference, velocity_reference, push_force, wall_normal):
        #Rotor speeds for one control step
        position = state[0:3]
        velocity = state[3:6]
        R = quat_to_rot_matrix_np(state[6:10])

        #The state stores angular velocity in the world frame; the attitude controller needs body rates
        body_rates = R.T @ state[10:13]

        #Position control, with the push into the wall as a feedforward force
        position_error = position - com_reference
        velocity_error = velocity - velocity_reference
        desired_force = self.mass * (-self.k_x * position_error - self.k_v * velocity_error + self.gravity * E3)
        desired_force = desired_force - push_force * wall_normal

        #Desired thrust axis, with the tilt limited
        b3 = desired_force / np.linalg.norm(desired_force)
        if np.arccos(np.clip(b3[2], -1.0, 1.0)) > MAX_TILT:
            horizontal = np.array([b3[0], b3[1], 0.0])
            horizontal = horizontal / np.linalg.norm(horizontal)
            b3 = np.cos(MAX_TILT) * E3 + np.sin(MAX_TILT) * horizontal

        #Desired attitude: thrust along b3, body x as close to the heading as possible
        b2 = np.cross(b3, self.heading)
        b2 = b2 / np.linalg.norm(b2)
        b1 = np.cross(b2, b3)
        R_desired = np.column_stack([b1, b2, b3])

        #Collective thrust along the current thrust axis
        thrust = desired_force @ R[:, 2]

        #Attitude control (desired rates are zero for these slow references)
        attitude_error = 0.5 * vee(R_desired.T @ R - R.T @ R_desired)
        torque = -self.K_R * attitude_error - self.K_Omega * body_rates
        torque = torque + np.cross(body_rates, self.inertia @ body_rates)

        return self.mix(thrust, torque)


    def mix(self, thrust, torque):
        #Rotor speeds for the wanted thrust and torques. Roll and pitch get priority: yaw is cut
        #in half, then dropped, until the rotor thrusts fit in their limits
        for yaw_fraction in [1.0, 0.5, 0.0]:
            wrench = np.array([thrust, torque[0], torque[1], yaw_fraction * torque[2]])
            rotor_thrusts = np.linalg.solve(self.mixer, wrench)
            if np.all(rotor_thrusts >= self.thrust_min) and np.all(rotor_thrusts <= self.thrust_max):
                break

        #If even the no-yaw version doesn't fit, clip it to the limits
        rotor_thrusts = np.clip(rotor_thrusts, self.thrust_min, self.thrust_max)
        return np.sqrt(rotor_thrusts / self.k_f)


#---------------------------------- Setup ----------------------------------------

name = "Geometric_Baseline"
seed = 0
np.random.seed(seed)

run_folder = SCRIPT_FOLDER / "MPC_runs" / name
run_folder.mkdir(parents=True, exist_ok=True)

#Drone config from the trained model's checkpoint (the same config the MPC script gives the sim)
_, drone_cfg, _, _ = load_checkpoint(Trained_model_path, "cpu")
pad_offset = np.asarray(drone_cfg.rod_tip) - np.asarray(drone_cfg.com_offset)

#Same generator settings as the MPC script, so the same seed gives the same scenario
trajectory_generator = Trajectory_Generator(Wind_range=(0, 5),
                                            wall_yaw_range=(0, 2 * np.pi),
                                            Ref_traj_dt=0.1,)

wind_vector, wall_normal_vector, wall_center_point = trajectory_generator.generate_wall_and_wind()
trajectory_generator.set_pad_offset(pad_offset)

trajectory_time_length, reference_trajectory = trajectory_generator.generate_trajectory()
initial_state = trajectory_generator.initial_state

sim = MujocoDroneSim(drone_cfg, wall_normal_vector, wall_center_point, wind_vector, motor_tau=0.0005)

#Holds the starting heading for the whole run (the generator starts the drone pointed at the wall)
R_initial = quat_to_rot_matrix_np(initial_state[6:10])
heading = np.array([R_initial[0, 0], R_initial[1, 0], 0.0])
heading = heading / np.linalg.norm(heading)

controller = GeometricController(sim, heading)

#Level attitude at that heading, used to turn the pad reference into a COM reference
R_heading = np.column_stack([heading, np.cross(E3, heading), E3])

hover_input = controller.mix(sim.mass * sim.cfg.gravity, np.zeros(3))
state = sim.reset(initial_state, hover_input)

times = np.arange(0.0, trajectory_time_length, simulation_dt)
reference = trajectory_generator.sample(times)


#---------------------------------- Simulation ----------------------------------------

#states[i] -> inputs[i] -> states[i+1], and normal_force[i+1] is measured after that step
states = [state]
inputs = []
normal_force = [0.0]

for i in range(len(times)):
    com_reference = reference["pad_position"][i] - R_heading @ pad_offset

    rotor_speeds = controller.compute(state, com_reference, reference["pad_velocity"][i],
                                      reference["normal_force"][i], wall_normal_vector)
    state = sim.step(rotor_speeds, simulation_dt)

    states.append(state)
    inputs.append(rotor_speeds)
    normal_force.append(sim.get_normal_force())


states = np.array(states)
inputs = np.array(inputs)
normal_force = np.array(normal_force)
state_times = np.arange(len(states)) * simulation_dt


#---------------------------------- Save ----------------------------------------

np.savez(run_folder / "run_data.npz",
         times=state_times,
         states=states,
         inputs=inputs,
         normal_force=normal_force,
         initial_state=initial_state,
         hover_input=hover_input,
         wind_vector=wind_vector,
         wall_normal_vector=wall_normal_vector,
         wall_center_point=wall_center_point,
         simulation_dt=simulation_dt,
         seed=seed,
         reference_time=trajectory_generator.reference_time,
         **{"reference_" + key: value for key, value in reference_trajectory.items()})
print("Saved run data to", run_folder / "run_data.npz")

#Inputs padded with a NaN row so they line up with the states, like the MPC runs
input_history = np.vstack([inputs, np.full((1, inputs.shape[1]), np.nan)])

plot_tracking(state_times, states, trajectory_generator.sample(state_times),
              pad_offset, wall_normal_vector, wall_center_point,
              normal_force_history=normal_force, input_history=input_history,
              save_path=run_folder / "tracking.png")

#Reference pad positions at the same times as the states, so the reference path lines up with the flight
sampled_reference = trajectory_generator.sample(state_times)

animate_flight(drone_cfg, wall_normal_vector, wall_center_point, states, dt=simulation_dt,
               reference_positions=sampled_reference["pad_position"],
               pad_offset=pad_offset,
               save_path=run_folder / "flight.gif")
