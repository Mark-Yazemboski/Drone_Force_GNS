import gurobipy as gp
from gurobipy import GRB
import torch
import numpy as np
from model import Drone_Dynamics
from scenario_generator import Trajectory_Generator
from utilities import quat_error_map


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

INPUT_TRUST_REGION = 100.0 # Rad/s per control step



#Sets the state and input sizes for the MPC controller
STATE_SIZE = 19 #State: [position(3), velocity(3), quaternion(4), angular_velocity(3), velocity_km1(3), angular_velocity_km1(3)]
INPUT_SIZE = 4 #Input: [Rotor1rpm, Rotor2rpm, Rotor3rpm, Rotor4rpm]


#Calculates the contact gate using a sigmoid function based on the pad distance, reference distance, and a smoothing factor tau
def calculate_contact_gate(pad_dist, d0, tau):
    return torch.sigmoid(torch.tensor((d0 - pad_dist) / tau)).item()

#Maps the 3-number rotation error onto the 4-number quaternion (first order, world-frame perturbation)


#Builds the full 19-number linearized state from the nominal state and the 18-number correction
def linearize_state(x_nom, dx_k):
    G = quat_error_map(x_nom[6:10])
    return [x_nom[0:3]   + dx_k[0:3],          # position
            x_nom[3:6]   + dx_k[3:6],          # velocity
            x_nom[6:10]  + G @ dx_k[6:9],      # quaternion 
            x_nom[10:13] + dx_k[9:12],         # angular velocity
            x_nom[13:16] + dx_k[12:15],        # previous velocity
            x_nom[16:19] + dx_k[15:18]]        # previous angular velocity

def to_np(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)


#Performs MPC control using the Sequential Quadratic Programming (SQP) approach
def mpc_control(A,B,C,D,nominal_state,nominal_input,nominal_output,input_vector_current,reference_trajectory,hover_thrust_values,horizon_length,d0,tau):

    #Converts any torch tensors to numpy, since Gurobi only works with numpy
    nominal_state = [to_np(x) for x in nominal_state]
    nominal_output = [to_np(y) for y in nominal_output]
    nominal_input = to_np(nominal_input)
    hover_thrust_values = to_np(hover_thrust_values)
    input_vector_current = to_np(input_vector_current)

    
    #Initializes the MPC optimization model using Gurobi
    mpc = gp.Model("Aerial_Manipulator")
    mpc.Params.OutputFlag = 0

    #Creates the decision variables for the state and input deviations
    #Sets limits for the input deviations based on the motor constraints
    dx = mpc.addMVar((horizon_length + 1, STATE_SIZE - 1), lb=-GRB.INFINITY, name="dx")
    du = mpc.addMVar((horizon_length, INPUT_SIZE), lb=MOTOR_MIN - nominal_input, ub=MOTOR_MAX - nominal_input, name="du")

    #adds trust region constraints for the input deviations
    #this makes sure the solution stays within a reasonable bound so
    #the linearization remains valid
    mpc.addConstr(du <= INPUT_TRUST_REGION)
    mpc.addConstr(du >= -INPUT_TRUST_REGION)

    #Adds the constraint that the initial state deviation is zero
    mpc.addConstr(dx[0] == 0)

    
    #Extracts the reference trajectory components for the pad position, pad velocity, normal force, and contact flag
    pad_ref_position = reference_trajectory['pad_position']
    pad_ref_velocity = reference_trajectory['pad_velocity']
    normal_force_ref = reference_trajectory['normal_force']
    contact_flag = reference_trajectory['contact_flag']


    #Initializes the cost function to zero
    cost = 0
    for k in range(horizon_length + 1):

        #Calculates the contact gate for the current time step
        contact_gate = calculate_contact_gate(nominal_output[k][12], d0, tau)

        #Computes the contact sigma by multiplying the contact gate with the contact flag
        contact_sigma = contact_gate * contact_flag[k]


        #Computes the linearized output for the current time step
        #outputs dont depend on the input deviations in this linearization
        if k < horizon_length:
            dy = C[k] @ dx[k] + D[k] @ du[k]
        else:
            dy = C[k] @ dx[k]
        linearized_output = nominal_output[k] + dy

        #extracts the individual components from the linearized output
        lin_roll = linearized_output[1]
        lin_pitch = linearized_output[2]
        lin_pad_pos = linearized_output[3:6]
        lin_pad_vel = linearized_output[6:9]
        lin_rod_unit = linearized_output[9:12]
        lin_pad_dist   = linearized_output[12]
        lin_rotor_dist = linearized_output[13:17]
        
        #All terms related to the input, and use k = 0 .. N-1
        if k < horizon_length:

            lin_normal_force = linearized_output[0]

            #Computes the linearized input for the current time step
            linearized_input = nominal_input[k] + du[k]

            #Computes the previous linearized input for the current time step
            previous_lin_input = nominal_input[k-1] + du[k-1] if k > 0 else input_vector_current

            #Adds the linearized system dynamics constraint for the current time step
            mpc.addConstr(dx[k + 1] == A[k] @ dx[k] + B[k] @ du[k])

            #Input deviation cost
            cost += (linearized_input - hover_thrust_values) @ (linearized_input - hover_thrust_values) * INPUT_WEIGHT
            #Input change cost
            cost += (linearized_input - previous_lin_input) @ (linearized_input - previous_lin_input) * INPUT_CHANGE_WEIGHT
            #Normal force error cost
            cost += contact_sigma * (lin_normal_force - normal_force_ref[k]) * (lin_normal_force - normal_force_ref[k]) * NORMAL_FORCE_WEIGHT

            #Adds the constraints related to motor change limits for each rotor
            mpc.addConstr(linearized_input - previous_lin_input <= MOTOR_CHANGE_LIM)
            mpc.addConstr(linearized_input - previous_lin_input >= -MOTOR_CHANGE_LIM)


        #All terms related to the state, output, and input linearization, and use k = 1 .. N
        #dx_0 = 0, so k = 0 is a constant
        if k >= 1:

            #Computes the linearized state, output, and input for the current time step
            position, velocity, quaternion, angular_velocity, velocity_prev, angular_velocity_prev = linearize_state(nominal_state[k], dx[k])


            

            #Computes the rod orientation error with respect to the wall normal vector
            e_rod = -(lin_rod_unit[0] * wall_normal_vector[1] - lin_rod_unit[1] * wall_normal_vector[0])

            #Computes the tangent projection matrix based on the contact sigma and wall normal vector
            tangent_projection = np.eye(3) - contact_sigma * np.outer(wall_normal_vector, wall_normal_vector)
            
            #Position error cost
            cost += (tangent_projection @ (lin_pad_pos - pad_ref_position[k])) @ (tangent_projection @ (lin_pad_pos - pad_ref_position[k])) * POSITION_WEIGHT
            #Velocity error cost
            cost += (tangent_projection @ (lin_pad_vel - pad_ref_velocity[k])) @ (tangent_projection @ (lin_pad_vel - pad_ref_velocity[k])) * VELOCITY_WEIGHT
            #Rod orientation error cost
            cost += e_rod * e_rod * ROD_ORIENTATION_WEIGHT
            #Angular velocity error cost
            cost += angular_velocity @ angular_velocity * ANG_VEL_WEIGHT

            #Addes constraints related to contact with the wall
            #and how the end effector should be within a certain distance from the wall during contact
            if contact_sigma >= 0.5:
                mpc.addConstr(lin_pad_dist <= CONTACT_MAX_DIST)

            #Adds the constraint related to the minimum distance of the rotors from the wall
            mpc.addConstr(lin_rotor_dist >= ROTOR_MIN_DIST)

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
    input_sequence = du.X + nominal_input

    return input_sequence

#Sets the dt of the simulation our controller will be running inside
simulation_dt = 0.01


#-------------------- GENERATE REFERENCE TRAJECTORY ---------------------------------------------

#Sets the dt at which the reference trajectory plan is generated
Trajectory_plan_dt = 0.1  

#Defines the ranges for the wall yaw angle and wind speed magnitude
wall_yaw_range = (-np.pi/6, np.pi/6)
wind_range = (0, 10)

#Generates the reference trajectory given a trajectory length (seconds), maximum wall tilt, wind range, and the trajectory plan dt
trajectory_generator = Trajectory_Generator(wind_range, wall_yaw_range, Trajectory_plan_dt)

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

#Extracts the trained GNN's internal timestep this will be used so we can make the MPC dt a multiple of the GNN's internal timestep
GNN_trained_dt = model.dt

#Sets the pad offset in the trajectory generator based on the model's pad offset
trajectory_generator.set_pad_offset(model.pad_offset)


# ---------------------- INITIALIZE MPC VARIABLES ----------------------

#Sets the MPC timestep as a multiple of the GNN's internal timestep
#it needs to be a multiple because the GNN model can only operate at its internal timestep
#and we need to know how many times the GNN model will be called within one MPC timestep
MPC_dt_multiple = 5
MPC_dt = GNN_trained_dt*MPC_dt_multiple

#Sets the MPC horizon length
Horizon_length = 10  

#Initializes the time since the last MPC update
time_since_last_MPC_update = MPC_dt

#Interpolates the reference trajectory to match the MPC timestep
ref_traj_interp = trajectory_generator.interpolate(MPC_dt)


#---------------- SIMULATION SETUP ----------------


#Time vector for the simulation
t = np.linspace(0, trajectory_time_length, int(trajectory_time_length / simulation_dt))

#Initializes the state vector for the simulation with the initial state from the trajectory generator
state_vector = np.zeros((len(t)+1,STATE_SIZE))
state_vector[0] = trajectory_generator.initial_state

#Initializes the input vector for the simulation with the hover input as the current input
input_vector = np.zeros((len(t)+1,INPUT_SIZE))
hover_input = model.calculate_hover_inputs()
input_vector[0] = hover_input
input_vector_current = input_vector[0]

input_sequence = np.array([input_vector_current for _ in range(Horizon_length)])

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
        
        #Obtains the reference trajectory window for the current MPC update
        reference_window = trajectory_generator.get_reference_window(time_step_index, MPC_dt, Horizon_length)

        #Rolls out the previous input sequence moved by one timestep, with the last input repeated at the end
        rollout_input_sequence = np.array(list(input_sequence[1:]) + [input_sequence[-1]])

        #initilizes the lists for storing linearized system matrices and nominal states
        A_list, B_list, C_list, D_list = [], [], [], []
        nom_states = []
        nom_outputs = []
        #Runs through the prediction horizon to linearize the system around the nominal trajectory
        nom_states.append(state_vector_current)
        for i in range(Horizon_length):
            #Converts the current nominal state and input to torch tensors for linearization
            torch_nom_state = torch.as_tensor(nom_states[-1], dtype=torch.float32)
            torch_input = torch.as_tensor(rollout_input_sequence[i], dtype=torch.float32)

            #Linearizes the system around the current nominal state and input
            A, B, C, D, x_next_nom, nominal_output = model.linearize(torch_nom_state, torch_input, MPC_dt_multiple)

            #Saves the linearized system matrices and nominal states for the current step
            A_list.append(A.detach().cpu().numpy())
            B_list.append(B.detach().cpu().numpy())
            C_list.append(C.detach().cpu().numpy())
            D_list.append(D.detach().cpu().numpy())
            nom_states.append(x_next_nom)
            nom_outputs.append(nominal_output)

        
        #Converts the current nominal state and input to torch tensors for linearization
        torch_nom_state = torch.as_tensor(nom_states[-1], dtype=torch.float32)
        torch_input = torch.as_tensor(rollout_input_sequence[-1], dtype=torch.float32)

        #Runs linearizartion on the last state in the prediction horizon to obtain the output matrices and nominal output at the final state
        _, _, C, D, _, nominal_output = model.linearize(torch_nom_state, torch_input, MPC_dt_multiple)
        C_list.append(C.detach().cpu().numpy())
        D_list.append(D.detach().cpu().numpy())
        nom_outputs.append(nominal_output)

        #Computes the MPC control input sequence based on the linearized system and nominal trajectory (SQP method)
        #Outputs the computed input sequence for the current MPC step
        input_sequence = mpc_control(A_list,B_list,C_list,D_list,nom_states,rollout_input_sequence,nom_outputs,input_vector_current, reference_window,hover_input,Horizon_length,d0,tau)

        #Sets the current input to the first element of the computed input sequence
        input_vector_current = input_sequence[0]


    #Stores the current input in the input vector for the current time step
    input_vector[time_step_index] = input_vector_current

    #Increments the time since the last MPC update and the time step index for the next iteration
    time_since_last_MPC_update += simulation_dt
    time_step_index += 1