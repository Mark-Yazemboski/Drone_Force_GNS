#The MPC controller: its weights and limits, and the QP that picks the rotor speeds.
#mpc_control builds the QP one horizon step at a time (easier to read);
#mpc_control_batched builds the exact same QP from stacked matrices (much faster).

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from utilities import quat_error_map, to_np, calculate_contact_gate, timer


MOTOR_MIN = 0.0 #Rad/s
MOTOR_MAX = 1000.0 #Rad/s
MOTOR_CHANGE_LIM = 100.0 #Rad/s per control step

CONTACT_MAX_DIST = 0.005 # meters
ROTOR_MIN_DIST = 0.05 # meters

ROLL_LIM = np.pi / 4 # radians, 45 degrees
PITCH_LIM = np.pi / 4 # radians, 45 degrees

POSITION_WEIGHT        = 1 / 0.02**2   # 2 cm
VELOCITY_WEIGHT        = 1 / 0.1**2    # 0.1 m/s
ROD_ORIENTATION_WEIGHT = 1 / 0.3**2    # 0.3 rad (~17 deg)
ANG_VEL_WEIGHT         = 1 / 0.5**2    # 0.5 rad/s
NORMAL_FORCE_WEIGHT    = 1 / 0.3**2    # 0.3 N
INPUT_WEIGHT           = 1 / 100**2    # 100 rad/s from hover
INPUT_CHANGE_WEIGHT    = 1 / 20**2     # 20 rad/s per step

SLACK_WEIGHT_ROLL = 100
SLACK_WEIGHT_PITCH = 100
SLACK_WEIGHT_CONTACT_DISTANCE = 100
SLACK_WEIGHT_ROTOR_DISTANCE = 1000

INPUT_TRUST_REGION = 100.0 # Rad/s per control step



#Sets the state and input sizes for the MPC controller
STATE_SIZE = 19 #State: [position(3), velocity(3), quaternion(4), angular_velocity(3), velocity_km1(3), angular_velocity_km1(3)]
INPUT_SIZE = 4 #Input: [Rotor1rpm, Rotor2rpm, Rotor3rpm, Rotor4rpm]


#The QP is small, so extra Gurobi threads don't help it
gurobi_env = gp.Env(empty=True)
gurobi_env.setParam("OutputFlag", 0)
gurobi_env.setParam("Threads", 2)
gurobi_env.start()


#Makes an error variable equal to the expression and returns its square.
#This keeps the objective's Q matrix diagonal, so Gurobi always sees a convex QP
def squared_error(mpc, expression, size):
    error = mpc.addMVar(size, lb=-GRB.INFINITY)
    mpc.addConstr(error == expression)
    return error @ error

#Builds the full 19-number linearized state from the nominal state and the 18-number correction
def linearize_state(x_nom, dx_k):

    # Computes the quaternion error mapping matrix for the linearization
    # Converts small changes in euler angles to a quaternion perturbation
    G = quat_error_map(x_nom[6:10])
    return [x_nom[0:3]   + dx_k[0:3],          # position
            x_nom[3:6]   + dx_k[3:6],          # velocity
            x_nom[6:10]  + G @ dx_k[6:9],      # quaternion 
            x_nom[10:13] + dx_k[9:12],         # angular velocity
            x_nom[13:16] + dx_k[12:15],        # previous velocity
            x_nom[16:19] + dx_k[15:18]]        # previous angular velocity

#Performs MPC control using the Sequential Quadratic Programming (SQP) approach
def mpc_control(A,B,C,D,nominal_state,nominal_input,nominal_output,input_vector_current,reference_trajectory,hover_thrust_values,horizon_length,d0,tau, wall_normal_vector):

    #Converts any torch tensors to numpy, since Gurobi only works with numpy
    nominal_state = [to_np(x) for x in nominal_state]
    nominal_output = [to_np(y) for y in nominal_output]
    nominal_input = to_np(nominal_input)
    hover_thrust_values = to_np(hover_thrust_values)
    input_vector_current = to_np(input_vector_current)

    #If the linearization produced NaN/inf anywhere, don't build the QP, fall back to the shifted plan
    for name, matrices in [("A", A), ("B", B), ("C", C), ("D", D), ("output", nominal_output), ("state", nominal_state)]:
        for k, matrix in enumerate(matrices):
            if not np.all(np.isfinite(matrix)):
                print(f"NaN/inf in {name}[{k}], using the shifted previous plan")
                return nominal_input

    
    #Initializes the MPC optimization model using Gurobi
    build_start = timer()
    mpc = gp.Model("Aerial_Manipulator", env=gurobi_env)
    mpc.Params.OutputFlag = 0

    #Sets the time limit for the solver and specifies that the problem is non-convex
    mpc.Params.TimeLimit = 1.0
    mpc.Params.NonConvex = 0 

    #Creates the decision variables for the state and input deviations
    #Sets limits for the input deviations based on the motor constraints
    dx = mpc.addMVar((horizon_length + 1, STATE_SIZE - 1), lb=-GRB.INFINITY, name="dx")
    du = mpc.addMVar((horizon_length, INPUT_SIZE), lb=MOTOR_MIN - nominal_input, ub=MOTOR_MAX - nominal_input, name="du")
    slack_roll = mpc.addMVar((horizon_length + 1,), lb=0, name="slack_roll")
    slack_pitch = mpc.addMVar((horizon_length + 1,), lb=0, name="slack_pitch")
    slack_contact_distance = mpc.addMVar((horizon_length + 1,), lb=0, name="slack_contact_distance")
    slack_rotor_distance = mpc.addMVar((horizon_length + 1,), lb=0, name="slack_rotor_distance")

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

            lin_normal_force = linearized_output[0:1]

            #Computes the linearized input for the current time step
            linearized_input = nominal_input[k] + du[k]

            #Computes the previous linearized input for the current time step
            previous_lin_input = nominal_input[k-1] + du[k-1] if k > 0 else input_vector_current

            #Adds the linearized system dynamics constraint for the current time step
            mpc.addConstr(dx[k + 1] == A[k] @ dx[k] + B[k] @ du[k])

           

            #Input deviation cost
            cost += INPUT_WEIGHT * squared_error(mpc, linearized_input - hover_thrust_values, 4)
            #Input change cost
            cost += INPUT_CHANGE_WEIGHT * squared_error(mpc, linearized_input - previous_lin_input, 4)
            #Normal force error cost
            cost += contact_sigma * NORMAL_FORCE_WEIGHT * squared_error(mpc, lin_normal_force - normal_force_ref[k], 1)

            #Adds the constraints related to motor change limits for each rotor
            mpc.addConstr(linearized_input - previous_lin_input <= MOTOR_CHANGE_LIM)
            mpc.addConstr(linearized_input - previous_lin_input >= -MOTOR_CHANGE_LIM)


        #All terms related to the state, output, and input linearization, and use k = 1 .. N
        #dx_0 = 0, so k = 0 is a constant
        if k >= 1:

            #Computes the linearized state, output, and input for the current time step
            position, velocity, quaternion, angular_velocity, velocity_prev, angular_velocity_prev = linearize_state(nominal_state[k], dx[k])


            

            #Computes the rod orientation error with respect to the wall normal vector
            e_rod = -(lin_rod_unit[0:1] * wall_normal_vector[1] - lin_rod_unit[1:2] * wall_normal_vector[0])

            #Computes the tangent projection matrix based on the contact sigma and wall normal vector
            tangent_projection = np.eye(3) - contact_sigma * np.outer(wall_normal_vector, wall_normal_vector)

             #Slack variable cost
            cost += slack_roll[k]**2 * SLACK_WEIGHT_ROLL
            cost += slack_pitch[k]**2 * SLACK_WEIGHT_PITCH
            cost += slack_contact_distance[k]**2 * SLACK_WEIGHT_CONTACT_DISTANCE
            cost += slack_rotor_distance[k]**2 * SLACK_WEIGHT_ROTOR_DISTANCE
            
            #Position error cost
            cost += POSITION_WEIGHT * squared_error(mpc, tangent_projection @ (lin_pad_pos - pad_ref_position[k]), 3)
            #Velocity error cost
            cost += VELOCITY_WEIGHT * squared_error(mpc, tangent_projection @ (lin_pad_vel - pad_ref_velocity[k]), 3)
            #Rod orientation error cost
            cost += ROD_ORIENTATION_WEIGHT * squared_error(mpc, e_rod, 1)
            #Angular velocity error cost
            cost += ANG_VEL_WEIGHT * squared_error(mpc, angular_velocity, 3)

            #Addes constraints related to contact with the wall
            #and how the end effector should be within a certain distance from the wall during contact
            if contact_sigma >= 0.5:
                mpc.addConstr(lin_pad_dist <= CONTACT_MAX_DIST + slack_contact_distance[k])

            #Adds the constraint related to the minimum distance of the rotors from the wall
            mpc.addConstr(lin_rotor_dist >= ROTOR_MIN_DIST - slack_rotor_distance[k])

            #Adds the constraints related to roll and pitch limits
            mpc.addConstr(lin_roll <= ROLL_LIM + slack_roll[k])
            mpc.addConstr(lin_roll >= -ROLL_LIM - slack_roll[k])
            mpc.addConstr(lin_pitch <= PITCH_LIM + slack_pitch[k])
            mpc.addConstr(lin_pitch >= -PITCH_LIM - slack_pitch[k])



    #Sets the objective as minimizing the total cost
    mpc.setObjective(cost, GRB.MINIMIZE)

    #Optimizes the MPC problem to find the optimal du input sequence
    solve_start = timer()
    try:
        mpc.optimize()
    except gp.GurobiError as error:
        print("MPC solve error:", error, "- using the shifted previous plan")
        return nominal_input
    print(f"  QP build: {solve_start - build_start:.3f} s | QP solve: {timer() - solve_start:.3f} s")

    if mpc.Status == GRB.SUBOPTIMAL:
        print("MPC solve suboptimal (numerical issues), using it anyway")
    elif mpc.Status != GRB.OPTIMAL:
        print(f"MPC solve failed (status {mpc.Status}), using the shifted previous plan")
        return nominal_input

    #Takes the optimized du input sequence and adds it to the nominal input to get the actual input sequence
    input_sequence = du.X + nominal_input

    return input_sequence

#Batched QP builder: the same MPC problem as mpc_control, but every cost and constraint is added
#for the whole horizon at once using stacked (block) matrices, instead of one horizon step at a time.
#The math is identical; it just makes a handful of big Gurobi calls instead of hundreds of small ones.
#
#Stacked variables:
#   dx: (N+1)*18 numbers  [dx_0, dx_1, ..., dx_N]
#   du:  N*4     numbers  [du_0, du_1, ..., du_{N-1}]
#Stacked outputs (17 per step, k = 0..N):   y = y_nominal + C_big @ dx + D_big @ du


#Sum of weight * (Mx @ dx + Mu @ du + c)^2 over every row, through one error variable per row.
#Each row is scaled by sqrt(weight) so the objective's Q matrix stays the identity (always convex).
def stacked_squared_error(mpc, weights, Mx, dx, Mu, du, c):
    scale = sp.diags(np.sqrt(weights))
    error = mpc.addMVar(len(c), lb=-GRB.INFINITY)
    mpc.addConstr(error == (scale @ Mx) @ dx + (scale @ Mu) @ du + scale @ c)
    return error @ error


def mpc_control_batched(A, B, C, D, nominal_state, nominal_input, nominal_output, input_vector_current,
                        reference_trajectory, hover_thrust_values, horizon_length, d0, tau, wall_normal_vector):

    #Converts any torch tensors to numpy, since Gurobi only works with numpy
    nominal_state = [to_np(x) for x in nominal_state]
    nominal_output = [to_np(y) for y in nominal_output]
    nominal_input = to_np(nominal_input)
    hover_thrust_values = to_np(hover_thrust_values)
    input_vector_current = to_np(input_vector_current)

    #If the linearization produced NaN/inf anywhere, don't build the QP, fall back to the shifted plan
    for name, matrices in [("A", A), ("B", B), ("C", C), ("D", D), ("output", nominal_output), ("state", nominal_state)]:
        for k, matrix in enumerate(matrices):
            if not np.all(np.isfinite(matrix)):
                print(f"NaN/inf in {name}[{k}], using the shifted previous plan")
                return nominal_input

    build_start = timer()
    N = horizon_length
    nx, nu, ny = STATE_SIZE - 1, INPUT_SIZE, len(nominal_output[0])     # 18, 4, 17
    n = np.asarray(wall_normal_vector, dtype=float)


    # ---------------- Stacked linear model ----------------

    #Outputs at every step k = 0..N. The last step has no input, so its D block is zeros.
    C_big = sp.block_diag(C[:N + 1], format="csr")                                          # ((N+1)*17, (N+1)*18)
    D_big = sp.vstack([sp.block_diag(D[:N]), sp.csr_matrix((ny, N * nu))], format="csr")     # ((N+1)*17, N*4)
    y_nominal = np.concatenate(nominal_output[:N + 1])                                       # ((N+1)*17,)

    def output_rows(output_indices, steps):
        #Row numbers in the stacked output for the given output entries at the given steps (step-major)
        return np.array([k * ny + j for k in steps for j in output_indices])

    #Contact gate at every step, from the nominal pad distance (output 12), times the contact flag
    pad_distance = y_nominal[output_rows([12], range(N + 1))]
    contact_sigma = reference_trajectory["contact_flag"][:N + 1] / (1.0 + np.exp(-(d0 - pad_distance) / tau))

    #Steps 1..N carry the state and output costs (dx_0 = 0, so step 0 is a constant)
    later_steps = range(1, N + 1)


    # ---------------- Variables ----------------

    #State and input deviations. The input bounds combine the motor limits and the trust region.
    mpc = gp.Model("Aerial_Manipulator_batched", env=gurobi_env)
    mpc.Params.OutputFlag = 0
    mpc.Params.TimeLimit = 1.0
    mpc.Params.NonConvex = 0

    nominal_input_flat = nominal_input[:N].reshape(-1)
    du_lower = np.maximum(MOTOR_MIN - nominal_input_flat, -INPUT_TRUST_REGION)
    du_upper = np.minimum(MOTOR_MAX - nominal_input_flat, INPUT_TRUST_REGION)
    dx = mpc.addMVar((N + 1) * nx, lb=-GRB.INFINITY, name="dx")
    du = mpc.addMVar(N * nu, lb=du_lower, ub=du_upper, name="du")

    #Slacks for the softened constraints, one per step k = 1..N
    slack_roll = mpc.addMVar(N, lb=0, name="slack_roll")
    slack_pitch = mpc.addMVar(N, lb=0, name="slack_pitch")
    slack_contact_distance = mpc.addMVar(N, lb=0, name="slack_contact_distance")
    slack_rotor_distance = mpc.addMVar(N, lb=0, name="slack_rotor_distance")

    zeros_du = lambda rows: sp.csr_matrix((rows, N * nu))
    zeros_dx = lambda rows: sp.csr_matrix((rows, (N + 1) * nx))


    # ---------------- Dynamics ----------------

    #Initial state deviation is zero
    mpc.addConstr(dx[:nx] == 0)

    #dx_{k+1} = A_k dx_k + B_k du_k for every k, as one block equation:
    #   [ -A_0  I              ]          [ -B_0             ]
    #   [       -A_1  I        ] @ dx  +  [       -B_1       ] @ du  = 0
    #   [             ...  ... ]          [             ...  ]
    next_state = sp.kron(sp.eye(N, N + 1, k=1), sp.eye(nx))
    current_state = sp.hstack([sp.block_diag(A[:N]), sp.csr_matrix((N * nx, nx))])
    mpc.addConstr((next_state - current_state).tocsr() @ dx - sp.block_diag(B[:N], format="csr") @ du == 0)


    # ---------------- Input costs and constraints (k = 0..N-1) ----------------

    #Input deviation from hover
    cost = INPUT_WEIGHT * stacked_squared_error(
        mpc, np.ones(N * nu), zeros_dx(N * nu), dx, sp.eye(N * nu, format="csr"), du,
        nominal_input_flat - np.tile(hover_thrust_values, N))

    #Input change from one step to the next (step 0 is compared with the input applied right now).
    #The motor change limit is the bound on this variable.
    previous_nominal_input = np.concatenate([input_vector_current, nominal_input_flat[:-nu]])
    input_difference = sp.eye(N * nu) - sp.eye(N * nu, k=-nu)
    input_change = mpc.addMVar(N * nu, lb=-MOTOR_CHANGE_LIM, ub=MOTOR_CHANGE_LIM, name="input_change")
    mpc.addConstr(input_change == input_difference.tocsr() @ du + (nominal_input_flat - previous_nominal_input))
    cost += INPUT_CHANGE_WEIGHT * (input_change @ input_change)

    #Normal force error, weighted by the contact gate
    rows = output_rows([0], range(N))
    cost += stacked_squared_error(
        mpc, NORMAL_FORCE_WEIGHT * contact_sigma[:N], C_big[rows], dx, D_big[rows], du,
        y_nominal[rows] - reference_trajectory["normal_force"][:N])


    # ---------------- State and output costs (k = 1..N) ----------------

    #Tangent projection at each step: removes the normal direction during contact
    projection = sp.block_diag([np.eye(3) - contact_sigma[k] * np.outer(n, n) for k in later_steps], format="csr")

    #Pad position and velocity errors, projected into the wall plane during contact
    for output_indices, reference_key, weight in [([3, 4, 5], "pad_position", POSITION_WEIGHT),
                                                  ([6, 7, 8], "pad_velocity", VELOCITY_WEIGHT)]:
        rows = output_rows(output_indices, later_steps)
        reference = reference_trajectory[reference_key][1:N + 1].reshape(-1)
        cost += stacked_squared_error(
            mpc, weight * np.ones(len(rows)), projection @ C_big[rows], dx, projection @ D_big[rows], du,
            projection @ (y_nominal[rows] - reference))

    #Rod yaw error: e_rod = -(rod_x * n_y - rod_y * n_x)
    rod_x, rod_y = output_rows([9], later_steps), output_rows([10], later_steps)
    cost += stacked_squared_error(
        mpc, ROD_ORIENTATION_WEIGHT * np.ones(N),
        -n[1] * C_big[rod_x] + n[0] * C_big[rod_y], dx,
        -n[1] * D_big[rod_x] + n[0] * D_big[rod_y], du,
        -n[1] * y_nominal[rod_x] + n[0] * y_nominal[rod_y])

    #Angular velocity (state entries 10:13, error-state entries 9:12)
    angular_velocity_entries = [k * nx + i for k in later_steps for i in range(9, 12)]
    select_angular_velocity = sp.csr_matrix((np.ones(3 * N), (np.arange(3 * N), angular_velocity_entries)),
                                            shape=(3 * N, (N + 1) * nx))
    cost += stacked_squared_error(
        mpc, ANG_VEL_WEIGHT * np.ones(3 * N), select_angular_velocity, dx, zeros_du(3 * N), du,
        np.concatenate([nominal_state[k][10:13] for k in later_steps]))

    #Slack costs
    cost += SLACK_WEIGHT_ROLL * (slack_roll @ slack_roll)
    cost += SLACK_WEIGHT_PITCH * (slack_pitch @ slack_pitch)
    cost += SLACK_WEIGHT_CONTACT_DISTANCE * (slack_contact_distance @ slack_contact_distance)
    cost += SLACK_WEIGHT_ROTOR_DISTANCE * (slack_rotor_distance @ slack_rotor_distance)


    # ---------------- Output constraints (k = 1..N) ----------------

    #Pad within CONTACT_MAX_DIST of the wall, only at steps where the gate says contact
    contact_steps = [k for k in later_steps if contact_sigma[k] >= 0.5]
    if contact_steps:
        rows = output_rows([12], contact_steps)
        select_slack = sp.csr_matrix((np.ones(len(contact_steps)), (np.arange(len(contact_steps)), [k - 1 for k in contact_steps])),
                                     shape=(len(contact_steps), N))
        mpc.addConstr(C_big[rows] @ dx + D_big[rows] @ du - select_slack @ slack_contact_distance
                      <= CONTACT_MAX_DIST - y_nominal[rows])

    #Rotors at least ROTOR_MIN_DIST from the wall (one slack per step, shared by the 4 rotors)
    rows = output_rows([13, 14, 15, 16], later_steps)
    each_rotor = sp.kron(sp.eye(N), np.ones((4, 1)), format="csr")
    mpc.addConstr(C_big[rows] @ dx + D_big[rows] @ du + each_rotor @ slack_rotor_distance
                  >= ROTOR_MIN_DIST - y_nominal[rows])

    #Roll and pitch limits
    for output_index, limit, slack in [(1, ROLL_LIM, slack_roll), (2, PITCH_LIM, slack_pitch)]:
        rows = output_rows([output_index], later_steps)
        mpc.addConstr(C_big[rows] @ dx + D_big[rows] @ du - slack <= limit - y_nominal[rows])
        mpc.addConstr(C_big[rows] @ dx + D_big[rows] @ du + slack >= -limit - y_nominal[rows])


    # ---------------- Solve ----------------

    mpc.setObjective(cost, GRB.MINIMIZE)

    solve_start = timer()
    try:
        mpc.optimize()
    except gp.GurobiError as error:
        print("MPC solve error:", error, "- using the shifted previous plan")
        return nominal_input
    print(f"  QP build: {solve_start - build_start:.3f} s | QP solve: {timer() - solve_start:.3f} s")

    if mpc.Status == GRB.SUBOPTIMAL:
        print("MPC solve suboptimal (numerical issues), using it anyway")
    elif mpc.Status != GRB.OPTIMAL:
        print(f"MPC solve failed (status {mpc.Status}), using the shifted previous plan")
        return nominal_input

    #Optimized input sequence = nominal + deviation
    return du.X.reshape(N, nu) + nominal_input[:N]