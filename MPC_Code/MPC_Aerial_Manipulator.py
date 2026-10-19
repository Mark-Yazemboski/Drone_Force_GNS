import gurobipy as gp
from gurobipy import GRB
import torch
import numpy as np
from MPC_Code import model
from model import Drone_Dynamics
from scenario_generator import Trajectory_Generator


MOTOR_MIN = 0.0 #Rad/s
MOTOR_MAX = 1000.0 #Rad/s
MOTOR_CHANGE_LIM = 100.0 #Rad/s per control step

CONTACT_MAX_DIST = 0.005 # meters
ROTOR_MIN_DIST = 0.05 # meters

ROLL_LIM = np.pi / 6 # radians, 30 degrees
PITCH_LIM = np.pi / 6 # radians, 30 degrees

POSITION_WEIGHT = 1.0
VELOCITY_WEIGHT = 1.0
ROD_ORIENTATION_WEIGHT = 1.0
ANG_VEL_WEIGHT = 1.0
NORMAL_FORCE_WEIGHT = 1.0
INPUT_WEIGHT = 1.0
INPUT_CHANGE_WEIGHT = 1.0


trajectory_generator = Trajectory_Generator(Wind_max_range=1.0, Trajectory_length=200, max_wall_tilt=np.pi/6)  # Example parameters
wind_vector, wall_normal_vector, wall_center_point = trajectory_generator.generate_trajectory()

state_size = 18
input_size = 4
N = 10  # Prediction horizon

Trained_model_path = r"C:\Users\myaze\Desktop\College\Grad\Grad Research\Drone_Force_GNS\MPC_Code\MPC_Model\Trained_Model.pt"
model = Drone_Dynamics(Trained_model_path, wall_normal_vector=wall_normal_vector, wall_center_point=wall_center_point)
mpc = gp.Model("Aerial_Manipulator")
mpc.Params.OutputFlag = 0

d0, tau = model.model.contact_d0, model.model.contact_tau

def calculate_contact_gate(pad_dist):
    return torch.sigmoid(torch.tensor((d0 - pad_dist) / tau)).item()

def build_mpc(A,B,C,D,nominal_state,nominal_input,nominal_output,reference_trajectory,hover_thrust_values,horizon_length):

    dx = mpc.addMVar((horizon_length + 1, 18), lb=-GRB.INFINITY, name="dx")
    du = mpc.addMVar((horizon_length, 4), lb=MOTOR_MIN - nominal_input, ub=MOTOR_MAX - nominal_input, name="du")

    mpc.addConstr(dx[0] == 0)
    cost = 0

    pad_ref_position = reference_trajectory['pad_position']
    pad_ref_velocity = reference_trajectory['pad_velocity']
    normal_force_ref = reference_trajectory['normal_force']
    contact_flag = reference_trajectory['contact_flag']

    for k in range(horizon_length):
        mpc.addConstr(dx[k + 1] == A[k] @ dx[k] + B[k] @ du[k])
        linearized_state = nominal_state[k] + dx[k]
        linearized_output = nominal_output[k] + C[k] @ dx[k] + D[k] @ du[k]
        linearized_input = nominal_input[k] + du[k]
        previous_lin_input = nominal_input[k-1] + du[k-1] if k > 0 else hover_thrust_values

        angular_velocity = linearized_state[10:13]
     
        lin_normal_force = linearized_output[0]
        lin_roll = linearized_output[1]
        lin_pitch = linearized_output[2]
        lin_pad_pos = linearized_output[3:6]
        lin_pad_vel = linearized_output[6:9]
        lin_rod_unit = linearized_output[9:12]
        lin_pad_dist   = linearized_output[12]     # 9 values, not just [12]
        lin_rotor_dist = linearized_output[13:17]     # 4 values

        e_rod = -(lin_rod_unit[0] * wall_normal_vector[1] - lin_rod_unit[1] * wall_normal_vector[0])

        contact_gate = calculate_contact_gate(nominal_output[k][12])
        
        contact_sigma = contact_gate * contact_flag[k]

        tangent_projection = np.eye(3) - contact_sigma * np.outer(wall_normal_vector, wall_normal_vector)

        

        cost += (tangent_projection @ (lin_pad_pos - pad_ref_position[k])) @ (tangent_projection @ (lin_pad_pos - pad_ref_position[k])) * POSITION_WEIGHT
        cost += (tangent_projection @ (lin_pad_vel - pad_ref_velocity[k])) @ (tangent_projection @ (lin_pad_vel - pad_ref_velocity[k])) * VELOCITY_WEIGHT
        cost += e_rod * e_rod * ROD_ORIENTATION_WEIGHT
        cost += angular_velocity @ angular_velocity * ANG_VEL_WEIGHT
        cost += (linearized_input - hover_thrust_values) @ (linearized_input - hover_thrust_values) * INPUT_WEIGHT
        cost += (linearized_input - previous_lin_input) @ (linearized_input - previous_lin_input) * INPUT_CHANGE_WEIGHT if k > 0 else 0
        cost += contact_sigma * (lin_normal_force - normal_force_ref[k]) * (lin_normal_force - normal_force_ref[k]) * NORMAL_FORCE_WEIGHT

        if contact_sigma >= 0.5:
            mpc.addConstr(lin_pad_dist <= CONTACT_MAX_DIST)

        for i in range(input_size):
            mpc.addConstr(linearized_input[i] - previous_lin_input[i] <= MOTOR_CHANGE_LIM)
            mpc.addConstr(linearized_input[i] - previous_lin_input[i] >= -MOTOR_CHANGE_LIM)
            mpc.addConstr(lin_rotor_dist[i] >= ROTOR_MIN_DIST)


        mpc.addConstr(lin_roll <= ROLL_LIM)
        mpc.addConstr(lin_roll >= -ROLL_LIM)
        mpc.addConstr(lin_pitch <= PITCH_LIM)
        mpc.addConstr(lin_pitch >= -PITCH_LIM)

    
    mpc.setObjective(cost, GRB.MINIMIZE)

    return mpc


