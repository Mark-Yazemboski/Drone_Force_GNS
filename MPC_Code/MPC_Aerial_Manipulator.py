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


#Calculates the contact gate using a sigmoid function based on the pad distance, reference distance, and a smoothing factor tau
def calculate_contact_gate(pad_dist, d0, tau):
    return torch.sigmoid(torch.tensor((d0 - pad_dist) / tau)).item()

#Performs MPC control using the Sequential Quadratic Programming (SQP) approach
def mpc_control(A,B,C,D,nominal_state,nominal_input,nominal_output,reference_trajectory,hover_thrust_values,horizon_length,d0,tau):

    #Creates the decision variables for the state and input deviations
    #Sets limits for the input deviations based on the motor constraints
    dx = mpc.addMVar((horizon_length + 1, 18), lb=-GRB.INFINITY, name="dx")
    du = mpc.addMVar((horizon_length, 4), lb=MOTOR_MIN - nominal_input, ub=MOTOR_MAX - nominal_input, name="du")

    #Adds the constraint that the initial state deviation is zero
    mpc.addConstr(dx[0] == 0)

    
    #Extracts the reference trajectory components for the pad position, pad velocity, normal force, and contact flag
    pad_ref_position = reference_trajectory['pad_position']
    pad_ref_velocity = reference_trajectory['pad_velocity']
    normal_force_ref = reference_trajectory['normal_force']
    contact_flag = reference_trajectory['contact_flag']


    #Initializes the cost function to zero
    cost = 0
    for k in range(horizon_length):

        #Adds the linearized system dynamics constraint for the current time step
        mpc.addConstr(dx[k + 1] == A[k] @ dx[k] + B[k] @ du[k])

        #Computes the linearized state, output, and input for the current time step
        linearized_state = nominal_state[k] + dx[k]
        linearized_output = nominal_output[k] + C[k] @ dx[k] + D[k] @ du[k]
        linearized_input = nominal_input[k] + du[k]

        #Computes the previous linearized input for the current time step
        previous_lin_input = nominal_input[k-1] + du[k-1] if k > 0 else hover_thrust_values

        #Extracts the relevant components from the linearized state and output for the current time step
        angular_velocity = linearized_state[10:13]
        lin_normal_force = linearized_output[0]
        lin_roll = linearized_output[1]
        lin_pitch = linearized_output[2]
        lin_pad_pos = linearized_output[3:6]
        lin_pad_vel = linearized_output[6:9]
        lin_rod_unit = linearized_output[9:12]
        lin_pad_dist   = linearized_output[12]
        lin_rotor_dist = linearized_output[13:17]

        #Computes the rod orientation error with respect to the wall normal vector
        e_rod = -(lin_rod_unit[0] * wall_normal_vector[1] - lin_rod_unit[1] * wall_normal_vector[0])

        #Calculates the contact gate for the current time step
        contact_gate = calculate_contact_gate(nominal_output[k][12], d0, tau)

        #Computes the contact sigma by multiplying the contact gate with the contact flag
        contact_sigma = contact_gate * contact_flag[k]

        #Computes the tangent projection matrix based on the contact sigma and wall normal vector
        tangent_projection = np.eye(3) - contact_sigma * np.outer(wall_normal_vector, wall_normal_vector)

        #Computes the cost associated with the current time step

        #Position error cost
        cost += (tangent_projection @ (lin_pad_pos - pad_ref_position[k])) @ (tangent_projection @ (lin_pad_pos - pad_ref_position[k])) * POSITION_WEIGHT
        #Velocity error cost
        cost += (tangent_projection @ (lin_pad_vel - pad_ref_velocity[k])) @ (tangent_projection @ (lin_pad_vel - pad_ref_velocity[k])) * VELOCITY_WEIGHT
        #Rod orientation error cost
        cost += e_rod * e_rod * ROD_ORIENTATION_WEIGHT
        #Angular velocity error cost
        cost += angular_velocity @ angular_velocity * ANG_VEL_WEIGHT
        #Input deviation cost
        cost += (linearized_input - hover_thrust_values) @ (linearized_input - hover_thrust_values) * INPUT_WEIGHT
        #Input change cost
        cost += (linearized_input - previous_lin_input) @ (linearized_input - previous_lin_input) * INPUT_CHANGE_WEIGHT if k > 0 else 0
        #Normal force error cost
        cost += contact_sigma * (lin_normal_force - normal_force_ref[k]) * (lin_normal_force - normal_force_ref[k]) * NORMAL_FORCE_WEIGHT

        #Addes constraints related to contact with the wall
        #and how the end effector should be within a certain distance from the wall during contact
        if contact_sigma >= 0.5:
            mpc.addConstr(lin_pad_dist <= CONTACT_MAX_DIST)

        #Adds the constraints related to motor change limits and rotor minimum distance for each rotor
        for i in range(input_size):
            mpc.addConstr(linearized_input[i] - previous_lin_input[i] <= MOTOR_CHANGE_LIM)
            mpc.addConstr(linearized_input[i] - previous_lin_input[i] >= -MOTOR_CHANGE_LIM)
            mpc.addConstr(lin_rotor_dist[i] >= ROTOR_MIN_DIST)

        #Adds the constraints related to roll and pitch limits
        mpc.addConstr(lin_roll <= ROLL_LIM)
        mpc.addConstr(lin_roll >= -ROLL_LIM)
        mpc.addConstr(lin_pitch <= PITCH_LIM)
        mpc.addConstr(lin_pitch >= -PITCH_LIM)

    #Sets the objective as minimizing the total cost
    mpc.setObjective(cost, GRB.MINIMIZE)

    #Optimizes the MPC problem to find the optimal du input sequence
    mpc.optimize()

    #Takes the optimized du input sequence and adds it to the nominal input to get the actual input sequence
    input_sequence = du + nominal_input

    return input_sequence

#Sets the dt of the simulation our controller will be running inside
simulation_dt = 0.01


#-------------------- GENERATE REFERENCE TRAJECTORY ---------------------------------------------

#Sets the dt at which the reference trajectory plan is generated
Trajectory_plan_dt = 0.1  

#Generates the reference trajectory given a trajectory length (seconds), maximum wall tilt, wind range, and the trajectory plan dt
trajectory_generator = Trajectory_Generator(Wind_range=(0,10), Trajectory_length=200, max_wall_tilt=np.pi/6, Trajectory_plan_dt=Trajectory_plan_dt)  # Example parameters

#Generates a reference trajectory based on the previously defined trajectory generator, returns the wind vector, wall normal vector, and wall center point
#aswell as the reference trajectory [end_effector_position, end_effector_velocity, normal_force, contact_flag]
wind_vector, wall_normal_vector, wall_center_point, trajectory_time_length, reference_trajectory = trajectory_generator.generate_trajectory()



#-------------------- INITIALIZE MODEL ---------------------------------------------

#path of the trained contact and fluid model GNN
Trained_model_path = r"C:\Users\myaze\Desktop\College\Grad\Grad Research\Drone_Force_GNS\MPC_Code\MPC_Model\Trained_Model.pt"

#Initializes the drone dynamics model with the trained GNN and environment parameters
model = Drone_Dynamics(Trained_model_path, wall_normal_vector=wall_normal_vector, wall_center_point=wall_center_point, wind_vector=wind_vector)

#Pulls out the contact model parameters from the trained GNN to be used in the MPC controller for the contact determination
d0, tau = model.model.contact_d0, model.model.contact_tau


# ---------------------- INITIALIZE MPC CONTROLLER ----------------------

#Initialize the MPC controller and names it "Aerial_Manipulator"
mpc = gp.Model("Aerial_Manipulator")
mpc.Params.OutputFlag = 0

#Sets the MPC timestep and horizon length
MPC_dt = simulation_dt*5
N = 10  

#Initializes the time since the last MPC update
time_since_last_MPC_update = 0.0

#Interpolates the reference trajectory to match the MPC timestep
ref_traj_interp = trajectory_generator.interpolate(MPC_dt)

#Sets the state and input sizes for the MPC controller
state_size = 18 #State: [position(3), velocity(3), quaternion(4), angular_velocity(3), velocity_km1(3), angular_velocity_km1(3)]
input_size = 4 #Input: [Rotor1rpm, Rotor2rpm, Rotor3rpm, Rotor4rpm]


#---------------- SIMULATION SETUP ----------------


#Time vector for the simulation
t = np.linspace(0, trajectory_time_length, int(trajectory_time_length / simulation_dt))

#Initializes the state vector for the simulation with the initial state from the trajectory generator
state_vector = np.zeros((trajectory_time_length//simulation_dt,state_size))
state_vector[0] = trajectory_generator.initial_state

#Initializes the input vector for the simulation with the hover input as the current input
input_vector = np.zeros((trajectory_time_length//simulation_dt,input_size))
hover_input = model.calculate_hover_inputs()
input_vector[0] = hover_input
input_vector_current = input_vector[0]

#Initializes the time step index for the simulation loop
time_step_index = 0

#Simulation loop for the aerial manipulator
for time_step in t:

    #propigates the system dynamics through the mojoco simulation using the current input
    state_vector[time_step_index+1] = mojoco_simulation_step(state_vector[time_step_index],input_vector_current, simulation_dt)
    state_vector_current = state_vector[time_step_index+1]

    #runs the mpc controller when the time since the last update exceeds the MPC timestep
    if time_since_last_MPC_update >= MPC_dt:
        time_since_last_MPC_update = 0.0

        #Rolls out the previous input sequence moved by one timestep, with the last input repeated at the end
        rollout_input_sequence = np.array(list(input_sequence[1:]) + [input_sequence[-1]])

        #initilizes the lists for storing linearized system matrices and nominal states
        A_list, B_list, C_list, D_list = [], [], [], []
        nom_states = []

        #Runs through the prediction horizon to linearize the system around the nominal trajectory
        nom_states.append(state_vector_current)
        for i in range(N):
            A, B, C, D, x_next_nom, nominal_output = model.linearize(nom_states[-1], rollout_input_sequence[i])
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)
            D_list.append(D)
            nom_states.append(x_next_nom)

        #Computes the MPC control input sequence based on the linearized system and nominal trajectory (SQP method)
        #Outputs the computed input sequence for the current MPC step
        input_sequence = mpc_control(A_list,B_list,C_list,D_list,nom_states[:-1],rollout_input_sequence,nominal_output,reference_trajectory,hover_input,N,d0,tau)

        #Sets the current input to the first element of the computed input sequence
        input_vector_current = input_sequence[0]

    #Stores the current input in the input vector for the current time step
    input_vector[time_step_index] = input_vector_current

    #Increments the time since the last MPC update and the time step index for the next iteration
    time_since_last_MPC_update += simulation_dt
    time_step_index += 1