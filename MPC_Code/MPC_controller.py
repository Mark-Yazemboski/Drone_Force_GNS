#The MPC controller: its weights and limits, and the QP that picks the rotor speeds.
#mpc_control builds the QP one horizon step at a time (easier to read);
#mpc_control_batched builds the exact same QP from stacked matrices (much faster).

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from utilities import quat_error_map, to_np, calculate_contact_gate, timer, quat_to_rot_matrix_np


MOTOR_MIN = 0.0 #Rad/s
MOTOR_MAX = 1000.0 #Rad/s
MOTOR_CHANGE_LIM = 100.0 #Rad/s per control step

CONTACT_MAX_DIST = 0.005 # meters
ROTOR_MIN_DIST = 0.05 # meters

ROLL_LIM = np.pi / 4 # radians, 45 degrees
PITCH_LIM = np.pi / 4 # radians, 45 degrees

POSITION_WEIGHT        = 1 / 0.02**2   # 2 cm
VELOCITY_WEIGHT        = 1 / 0.1**2    # 0.1 m/s
ROD_ORIENTATION_WEIGHT = 1 / 0.1**2    # 0.1 rad (~6 deg) off the wall normal, pitch and yaw
ANG_VEL_WEIGHT         = 1 / 0.5**2    # 0.5 rad/s
NORMAL_FORCE_WEIGHT    = 1 / 0.3**2    # 0.3 N
INPUT_WEIGHT           = 1 / 100**2    # 100 rad/s from hover
INPUT_CHANGE_WEIGHT    = 1 / 20**2     # 20 rad/s per step

#Proximal term: penalizes du, the change from the nominal (shifted) plan, not the input itself. It adds curvature in
#input directions the other costs barely see (stuck contact, where the stiff force rows dominate everything else),
#without pulling the inputs toward hover; it costs nothing once a plan stops changing. 0 turns it off.
PROXIMAL_WEIGHT        = 1 / 20**2     # 20 rad/s away from the nominal plan

#Pre-stabilized prediction (batched builder only): the QP chooses v with du_k = K_k dx_k + v_k, where K_k are LQR
#gains for the horizon's linearizations. Predictions then run on A_k + B_k K_k instead of A_k, so an unstable mode
#in the linearization (contact chatter) can't compound over the horizon. Same feasible plans, same optimum; only the
#solver's coordinates change. False gives the original formulation exactly.
PRESTABILIZE = True

#L1 (exact penalty) slacks: cost = weight * violation. Above a threshold weight the constraint holds exactly whenever
#it can, so these only need to be "big enough": 100 / tolerance makes one tolerance of violation cost as much as
#100 steps of being one tolerance off in a tracking cost. Contact distance is kept weak on purpose: enforced
#exactly, it makes the MPC tilt the pad so the closest node satisfies it (it limits the closest node, not the center).
SLACK_WEIGHT_ROLL = 100 / 0.01               # per rad over the limit
SLACK_WEIGHT_PITCH = 100 / 0.01              # per rad over the limit
SLACK_WEIGHT_CONTACT_DISTANCE = 1 / 0.1      # per m too far off the wall (weak on purpose)
SLACK_WEIGHT_ROTOR_DISTANCE = 100 / 0.002    # per m too close to the wall
SLACK_WEIGHT_APPROACH = 100 / 0.02           # per m/s over the approach speed limit

#Approach speed limit: the pad's speed into the wall must stay under gap / APPROACH_TIME + TOUCH_SPEED,
#so it slows down as it closes in and touches at about TOUCH_SPEED. Far from the wall the limit is loose.
#It applies only at steps where the planned pad isn't touching (contact gate < 0.5, from the pad distance, not the
#contact flag): once touching, the wall stops the motion and the force cost takes over.
APPROACH_TIME = 0.1             # s, about the time to close the remaining gap at the allowed speed
TOUCH_SPEED   = 0.02            # m/s, allowed speed into the wall at contact

INPUT_TRUST_REGION = 100.0 # Rad/s per control step

#Breaking the pad loose: in contact, where the pad should move in-plane faster than the plan has it moving,
#the in-plane force the drone applies along that direction is pushed toward PUSH_MARGIN * mu * F_n
#(mu is the model's learned value). The push fades out as the planned speed reaches the wanted speed.
PUSH_WEIGHT     = 1 / 0.1**2    # 0.1 N
PUSH_MARGIN     = 2.0           # over the friction limit; the learned mu (~0.33) is below MuJoCo's 0.4, and the push is one
                                # cost among several, so the drone delivers less than the target
SLIDE_MIN_SPEED = 0.01          # m/s, wanted in-plane speed above this counts as a slide
CATCH_UP_TIME   = 0.5          # s, the wanted speed includes closing the in-plane position error over this time



#Motor lag: each rotor's thrust follows its command through a first-order lag with this time constant (as in MuJoCo).
#The QP's decisions are the commands; the model takes the effective (lagged) rotor speed over each step, which is a
#weighted sum of the current command and the earlier ones, so the lag adds no QP states. Keep it equal to the
#simulator's motor_tau (the run file passes this value to the simulator). 0 turns the lag off.
MOTOR_TAU = 0.025 # s


#Sets the state and input sizes for the MPC controller
STATE_SIZE = 19 #State: [position(3), velocity(3), quaternion(4), angular_velocity(3), velocity_km1(3), angular_velocity_km1(3)]
INPUT_SIZE = 4 #Input: [Rotor1rpm, Rotor2rpm, Rotor3rpm, Rotor4rpm]


#The QP is small, so extra Gurobi threads don't help it
gurobi_env = gp.Env(empty=True)
gurobi_env.setParam("OutputFlag", 0)
gurobi_env.setParam("Threads", 2)
gurobi_env.start()


#---------------------------------- Event log ----------------------------------------
#One line per thing that went wrong (failed or suboptimal solves, numerical trouble, slacks in use, rotors at their
#limits, the trust region hit), so a run can be reviewed from one dense file instead of the console.
#open_event_log(path) once per run, set_log_step(step, time) every step, log_event(...) anywhere.
_event_log = None
_log_step = (0, 0.0)

def open_event_log(path):
    global _event_log
    _event_log = open(path, "w", buffering=1)      #line buffered, so the file is complete even if the run crashes
    _event_log.write("step,time_s,event,details\n")

def set_log_step(step, time):
    global _log_step
    _log_step = (step, time)

def log_event(event, details=""):
    details = str(details).replace(",", ";")      #keeps every line at four CSV fields
    print(f"  [{event}] {details}")
    if _event_log is not None:
        _event_log.write(f"{_log_step[0]},{_log_step[1]:.2f},{event},{details}\n")


def value_range(values):
    #"min..max" of the nonzero magnitudes
    values = np.abs(np.asarray(values, dtype=float))
    values = values[values > 0]
    return f"{values.min():.1e}..{values.max():.1e}" if len(values) else "none"

def coefficient_ranges(mpc):
    #Ranges of the QP's constraint matrix, quadratic objective, right-hand sides, and finite bounds
    objective = mpc.getObjective()
    quadratic = [objective.getCoeff(i) for i in range(objective.size())]
    bounds = np.concatenate([mpc.getAttr("LB", mpc.getVars()), mpc.getAttr("UB", mpc.getVars())])
    bounds = bounds[np.abs(bounds) < GRB.INFINITY]
    return (f"matrix {value_range(mpc.getA().data)} | quadratic objective {value_range(quadratic)} | "
            f"rhs {value_range(mpc.getAttr('RHS', mpc.getConstrs()))} | bounds {value_range(bounds)}")


def log_solution(slack_maxes, first_input, first_du):
    #Logs the slacks in use, rotors at their limits, and a first step that hits the trust region
    over = [f"{name} {scale * value:.2f} {unit}" for name, value, limit, scale, unit in [
        ("roll", slack_maxes[0], 0.01, 1, "rad"), ("pitch", slack_maxes[1], 0.01, 1, "rad"),
        ("contact", slack_maxes[2], 0.001, 1000, "mm"), ("rotor", slack_maxes[3], 0.001, 1000, "mm"),
        ("approach", slack_maxes[4], 0.01, 100, "cm/s")] if value > limit]
    if over:
        log_event("slack", "; ".join(over))
    saturated = np.flatnonzero((first_input <= MOTOR_MIN + 1.0) | (first_input >= MOTOR_MAX - 1.0))
    if len(saturated):
        log_event("rotor_saturated", " ".join(f"rotor{j + 1}={first_input[j]:.0f}" for j in saturated))
    if np.any(np.abs(first_du) >= INPUT_TRUST_REGION - 1e-3):
        log_event("trust_region_hit", "du = " + " ".join(f"{v:.0f}" for v in first_du))


#---------------------------------- Motor lag ----------------------------------------
#Over one step of length dt with the command held, the squared rotor speed (thrust / k_f) goes from s0 toward the
#squared command c: at the end it is (1 - end) c + end s0, and its average over the step is (1 - average) c + average s0.
def motor_lag_factors(dt):
    if MOTOR_TAU <= 0:
        return 0.0, 0.0
    end = np.exp(-dt / MOTOR_TAU)
    average = MOTOR_TAU / dt * (1.0 - end)
    return end, average


def motor_step(speed_squared, command, dt):
    #Runs the motors one step from their squared speeds now (4,), holding the command (4,).
    #Returns the effective rotor speed over the step (the model's input) and the squared speed at the end.
    end, average = motor_lag_factors(dt)
    command_squared = np.asarray(command, dtype=float) ** 2
    effective = np.sqrt((1.0 - average) * command_squared + average * speed_squared)
    return effective, (1.0 - end) * command_squared + end * speed_squared


def motor_map(commands, speed_squared, dt):
    #Effective rotor speeds over a horizon of commands (N, 4), from the motors' squared speeds now (4,),
    #and the matrix M (N*4, N*4) with d(effective) = M d(command), linearized about these commands.
    #M is lower triangular: a step's speed depends on its own command and the earlier ones, less the older they are.
    commands = np.asarray(commands, dtype=float)
    N, nu = commands.shape
    end, average = motor_lag_factors(dt)

    effective = np.zeros((N, nu))
    for k in range(N):
        effective[k], speed_squared = motor_step(speed_squared, commands[k], dt)

    #Weight of command j's square in step k's effective squared speed (j <= k), then the chain rule through the squares
    k, j = np.tril_indices(N)
    weight = np.where(k == j, 1.0 - average, average * (1.0 - end) * end ** np.maximum(k - j - 1, 0))
    values = weight[:, None] * commands[j] / np.maximum(effective[k], 1.0)
    rotors = np.arange(nu)
    matrix = sp.csr_matrix((values.ravel(), ((k[:, None] * nu + rotors).ravel(), (j[:, None] * nu + rotors).ravel())),
                           shape=(N * nu, N * nu))
    return effective, drop_small(matrix)


#Zeroes Jacobian entries that are float32 round-off: anything under JACOBIAN_ZERO times the largest entry in its row.
#The autograd Jacobians come out of float32 math, so entries around 1e-11 to 1e-13 are noise, but they stretch the
#QP's coefficient range across ~17 orders of magnitude, which is what Gurobi's numerical trouble comes from.
#Also zeroes anything under JACOBIAN_FLOOR outright: rows that should be exactly zero come out of autograd as pure
#round-off, and a row-relative test keeps those (the noise is its own row maximum). Nothing physical in these
#units is that small (the smallest real entries, position vs rotor speed over one step, are ~1e-6).
JACOBIAN_ZERO = 1e-6
JACOBIAN_FLOOR = 1e-9

def clean_jacobians(matrices):
    cleaned = []
    for matrix in matrices:
        matrix = np.array(matrix, dtype=float)
        row_max = np.abs(matrix).max(axis=-1, keepdims=True)
        matrix[(np.abs(matrix) < JACOBIAN_ZERO * row_max) | (np.abs(matrix) < JACOBIAN_FLOOR)] = 0.0
        cleaned.append(matrix)
    return cleaned


#Snaps gate values within GATE_SNAP of 0 or 1 to exactly 0 or 1. A sigmoid far from its center gives values like
#1e-20 or 1 - 1e-9; as cost weights (and through the tangent projection) those become tiny QP coefficients
#that only stretch the coefficient range, and change nothing a solver could resolve anyway.
GATE_SNAP = 1e-3

def snap_gate(values):
    values = np.asarray(values, dtype=float)
    return np.where(values < GATE_SNAP, 0.0, np.where(values > 1.0 - GATE_SNAP, 1.0, values))


#Drops entries smaller than JACOBIAN_ZERO times the largest entry in their row, from a sparse matrix built from
#products of the Jacobians (the projections and weightings can leave round-off behind)
def drop_small(matrix):
    matrix = sp.csr_matrix(matrix, dtype=float)
    if matrix.nnz == 0:
        return matrix
    row_max = abs(matrix).max(axis=1).toarray().ravel()
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    small = (np.abs(matrix.data) < JACOBIAN_ZERO * row_max[rows]) | (np.abs(matrix.data) < JACOBIAN_FLOOR)
    matrix.data[small] = 0.0
    matrix.eliminate_zeros()
    return matrix


#How unstable the linearized dynamics are over the horizon: the largest |eigenvalue| of any A_k (above 1 means a
#mode that grows every step), and how much the product A_{N-1} ... A_0 can amplify a deviation across the whole
#horizon (its largest singular value). A large horizon gain makes the QP badly conditioned whatever its scaling.
def horizon_growth(A, label="A"):
    radius = max(np.abs(np.linalg.eigvals(matrix)).max() for matrix in A)
    product = np.eye(A[0].shape[0])
    for matrix in A:
        product = matrix @ product
    return f"{label} eig max {radius:.3f} | horizon gain {np.linalg.norm(product, 2):.1e}"


#Gains for the pre-stabilized prediction: one K_k per horizon step, from the finite-horizon LQR (backward Riccati)
#recursion on the horizon's own linearizations, so each step is stabilized for the A_k it actually has (one gain
#from A_0 can destabilize later steps when contact changes along the horizon). The state weights are the tracking
#tolerances (history states get a token weight); the input weight is the input change weight, so the gains stay
#moderate. 50 small 4x4 solves, a few ms. An unstable mode the inputs can't reach stays in the closed-loop
#growth that the numeric_trouble log line reports.
def prestabilizing_gains(A, B):
    state_weights = np.concatenate([np.full(3, POSITION_WEIGHT), np.full(3, VELOCITY_WEIGHT),
                                    np.full(3, ROD_ORIENTATION_WEIGHT), np.full(3, ANG_VEL_WEIGHT),
                                    np.full(6, 1e-6)])
    Q = np.diag(state_weights)
    R = INPUT_CHANGE_WEIGHT * np.eye(B[0].shape[1])
    P = Q.copy()
    gains = [None] * len(A)
    for k in reversed(range(len(A))):
        K = -np.linalg.solve(R + B[k].T @ P @ B[k], B[k].T @ P @ A[k])
        gains[k] = K
        P = Q + A[k].T @ P @ (A[k] + B[k] @ K)
        P = 0.5 * (P + P.T)
    return gains


#Solves the QP. On numerical trouble (status 12), logs the model's coefficient ranges, which show what makes it
#badly scaled, and retries once with barrier in its more robust settings: the homogeneous algorithm, extra numerical
#care, and no crossover (the MPC only needs the interior solution, and crossover is a common source of trouble).
#Dual simplex was tried here and ran into the time limit (status 9) almost every time.
def solve_qp(mpc, A, label="A"):
    mpc.optimize()
    if mpc.Status == GRB.NUMERIC:
        log_event("numeric_trouble", coefficient_ranges(mpc) + " | " + horizon_growth(A, label))
        mpc.Params.Method = 2
        mpc.Params.BarHomogeneous = 1
        mpc.Params.Crossover = 0
        mpc.Params.NumericFocus = 2
        mpc.reset()
        mpc.optimize()
        log_event("numeric_retry", f"homogeneous barrier: status {mpc.Status}")

#Makes an error variable equal to the expression and returns its square.
#This keeps the objective's Q matrix diagonal, so Gurobi always sees a convex QP
def squared_error(mpc, expression, size):
    error = mpc.addMVar(size, lb=-GRB.INFINITY)
    mpc.addConstr(error == expression)
    return error @ error

#Push along the wanted slide direction at the steps where the pad should move faster than the nominal plan has it.
#Wanted in-plane velocity = reference velocity + in-plane position error / CATCH_UP_TIME, so a pad left behind
#(or stuck after the reference stops) still gets pushed. Each step's weight is scaled by how far the planned speed
#along that direction falls short of the wanted speed: 1 when stuck, 0 at the wanted speed (no on/off switching).
#The applied in-plane force is thrust + gravity (no contact or aero), so it's linear in the deviations without the model:
#   push_k = direction . (T b3 + m g) - target,   d(push)/d(du_k) = 2 k_f (direction . b3) omega,
#   d(push)/d(dtheta_k) = T (b3 x direction)   (the attitude error rotates b3 by dtheta x b3)
#While stuck, friction cancels this force, so pushing it past mu * F_n is what makes the pad slip.
#params: k_f, mass, gravity, mu from the model. Returns (k, g_theta (3,), g_u (4,), constant, scale) per push step.
def stuck_push_rows(nominal_state, nominal_input, nominal_output, reference_trajectory, contact_sigma, n, params, N):
    in_plane = np.eye(3) - np.outer(n, n)
    weight = np.array([0.0, 0.0, -params["mass"] * params["gravity"]])
    rows = []
    for k in range(N):
        position_error = reference_trajectory["pad_position"][k] - nominal_output[k][3:6]
        wanted_velocity = in_plane @ (reference_trajectory["pad_velocity"][k] + position_error / CATCH_UP_TIME)
        wanted_speed = np.linalg.norm(wanted_velocity)
        if contact_sigma[k] < 0.5 or wanted_speed < SLIDE_MIN_SPEED:
            continue

        direction = wanted_velocity / wanted_speed
        scale = np.clip(1.0 - direction @ nominal_output[k][6:9] / wanted_speed, 0.0, 1.0)
        if scale == 0.0:
            continue

        b3 = quat_to_rot_matrix_np(nominal_state[k][6:10])[:, 2]
        omega = nominal_input[k]
        thrust = params["k_f"] * np.sum(omega ** 2)
        target = PUSH_MARGIN * params["mu"] * max(nominal_output[k][0], 0.0)

        g_theta = thrust * np.cross(b3, direction)
        g_u = 2 * params["k_f"] * (direction @ b3) * omega
        constant = direction @ (thrust * b3 + weight) - target
        rows.append((k, g_theta, g_u, constant, scale))
    return rows

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
def mpc_control(A,B,C,D,nominal_state,nominal_input,nominal_output,input_vector_current,reference_trajectory,hover_thrust_values,horizon_length,d0,tau, wall_normal_vector, model_params, motor_speed_squared):

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
                log_event("nan_linearization", f"{name}[{k}] - using the shifted previous plan")
                return nominal_input

    A, B, C, D = clean_jacobians(A), clean_jacobians(B), clean_jacobians(C), clean_jacobians(D)

    #Initializes the MPC optimization model using Gurobi
    build_start = timer()
    mpc = gp.Model("Aerial_Manipulator", env=gurobi_env)
    mpc.Params.OutputFlag = 0

    #Sets the time limit for the solver and specifies that the problem is non-convex
    mpc.Params.TimeLimit = 1.0
    mpc.Params.NonConvex = 0 
    #Barrier convergence tolerance: Gurobi's default (1e-8) is far tighter than an MPC step needs, and solves
    #that stall just short of it are what come back as numerical trouble (12) or suboptimal (13)
    mpc.Params.BarConvTol = 1e-6

    #Creates the decision variables for the state and input deviations
    #Sets limits for the input deviations based on the motor constraints
    dx = mpc.addMVar((horizon_length + 1, STATE_SIZE - 1), lb=-GRB.INFINITY, name="dx")
    du = mpc.addMVar((horizon_length, INPUT_SIZE), lb=MOTOR_MIN - nominal_input, ub=MOTOR_MAX - nominal_input, name="du")

    #Motor lag: du are changes to the commands; the model sees the change in the effective rotor speed,
    #effective_du[k] = rows k of M times all the command changes
    effective_input, motor_matrix = motor_map(nominal_input[:horizon_length], motor_speed_squared, model_params["dt"])
    du_all = du.reshape(-1)
    effective_du = [motor_matrix[k * INPUT_SIZE:(k + 1) * INPUT_SIZE] @ du_all for k in range(horizon_length)]
    slack_roll = mpc.addMVar((horizon_length + 1,), lb=0, name="slack_roll")
    slack_pitch = mpc.addMVar((horizon_length + 1,), lb=0, name="slack_pitch")
    slack_contact_distance = mpc.addMVar((horizon_length + 1,), lb=0, name="slack_contact_distance")
    slack_rotor_distance = mpc.addMVar((horizon_length + 1,), lb=0, name="slack_rotor_distance")
    slack_approach = mpc.addMVar((horizon_length + 1,), lb=0, name="slack_approach")

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


    #Steps where the pad should slide but the plan has it stuck, and the push needed there
    sigma_all = [float(snap_gate(calculate_contact_gate(nominal_output[k][12], d0, tau))) * contact_flag[k]
                 for k in range(horizon_length + 1)]
    push_rows = {k: (g_theta, g_u, constant, scale) for k, g_theta, g_u, constant, scale in stuck_push_rows(
        nominal_state, effective_input, nominal_output, reference_trajectory, sigma_all,
        np.asarray(wall_normal_vector, dtype=float), model_params, horizon_length)}

    #Initializes the cost function to zero
    cost = 0
    for k in range(horizon_length + 1):

        #Calculates the contact gate for the current time step
        contact_gate = float(snap_gate(calculate_contact_gate(nominal_output[k][12], d0, tau)))

        #Computes the contact sigma by multiplying the contact gate with the contact flag
        contact_sigma = contact_gate * contact_flag[k]


        #Computes the linearized output for the current time step
        #outputs dont depend on the input deviations in this linearization
        if k < horizon_length:
            dy = C[k] @ dx[k] + D[k] @ effective_du[k]
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
            mpc.addConstr(dx[k + 1] == A[k] @ dx[k] + B[k] @ effective_du[k])

           

            #Input deviation cost
            cost += INPUT_WEIGHT * squared_error(mpc, linearized_input - hover_thrust_values, 4)
            #Proximal cost: change from the nominal plan
            if PROXIMAL_WEIGHT > 0:
                cost += PROXIMAL_WEIGHT * squared_error(mpc, du[k], 4)
            #Input change cost
            cost += INPUT_CHANGE_WEIGHT * squared_error(mpc, linearized_input - previous_lin_input, 4)
            #Normal force error cost
            #Normal force error whenever the pad is at the wall, over the whole trajectory (the reference
            #force is 0 outside the contact phase, so touching the wall when it shouldn't costs the same everywhere)
            cost += contact_gate * NORMAL_FORCE_WEIGHT * squared_error(mpc, lin_normal_force - normal_force_ref[k], 1)
            #Push to break the pad loose (only at stuck steps of a slide)
            if k in push_rows:
                g_theta, g_u, constant, scale = push_rows[k]
                cost += scale * PUSH_WEIGHT * squared_error(mpc, g_theta[None] @ dx[k][6:9] + g_u[None] @ effective_du[k] + constant, 1)

            #Adds the constraints related to motor change limits for each rotor
            mpc.addConstr(linearized_input - previous_lin_input <= MOTOR_CHANGE_LIM)
            mpc.addConstr(linearized_input - previous_lin_input >= -MOTOR_CHANGE_LIM)


        #All terms related to the state, output, and input linearization, and use k = 1 .. N
        #dx_0 = 0, so k = 0 is a constant
        if k >= 1:

            #Computes the linearized state, output, and input for the current time step
            position, velocity, quaternion, angular_velocity, velocity_prev, angular_velocity_prev = linearize_state(nominal_state[k], dx[k])


            

            #Rod orientation error: the part of the rod direction perpendicular to the wall normal
            #(pad pitch and yaw relative to the wall), zero when the rod points straight into the wall
            e_rod = (np.eye(3) - np.outer(wall_normal_vector, wall_normal_vector)) @ lin_rod_unit

            #Computes the tangent projection matrix based on the contact sigma and wall normal vector
            tangent_projection = np.eye(3) - contact_sigma * np.outer(wall_normal_vector, wall_normal_vector)

             #Slack variable cost
            cost += slack_roll[k] * SLACK_WEIGHT_ROLL
            cost += slack_pitch[k] * SLACK_WEIGHT_PITCH
            cost += slack_contact_distance[k] * SLACK_WEIGHT_CONTACT_DISTANCE
            cost += slack_rotor_distance[k] * SLACK_WEIGHT_ROTOR_DISTANCE
            cost += slack_approach[k] * SLACK_WEIGHT_APPROACH
            
            #Position error cost
            cost += POSITION_WEIGHT * squared_error(mpc, tangent_projection @ (lin_pad_pos - pad_ref_position[k]), 3)
            #Velocity error cost
            cost += VELOCITY_WEIGHT * squared_error(mpc, tangent_projection @ (lin_pad_vel - pad_ref_velocity[k]), 3)
            #Rod orientation error cost
            cost += ROD_ORIENTATION_WEIGHT * squared_error(mpc, e_rod, 3)
            #Angular velocity error cost
            cost += ANG_VEL_WEIGHT * squared_error(mpc, angular_velocity, 3)

            #Addes constraints related to contact with the wall
            #and how the end effector should be within a certain distance from the wall during contact
            if contact_sigma >= 0.5:
                mpc.addConstr(lin_pad_dist <= CONTACT_MAX_DIST + slack_contact_distance[k])

            #Adds the constraint related to the minimum distance of the rotors from the wall
            mpc.addConstr(lin_rotor_dist >= ROTOR_MIN_DIST - slack_rotor_distance[k])

            #Approach speed limit: speed into the wall under gap / APPROACH_TIME + TOUCH_SPEED, while not touching
            if contact_gate < 0.5:
                speed_into_wall = -(wall_normal_vector @ lin_pad_vel)
                gap = lin_pad_dist - model_params["contact_radius"]
                mpc.addConstr(speed_into_wall - gap / APPROACH_TIME <= TOUCH_SPEED + slack_approach[k])

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
        solve_qp(mpc, A)
    except gp.GurobiError as error:
        log_event("solver_error", f"{error} - using the shifted previous plan")
        return nominal_input
    print(f"  QP build: {solve_start - build_start:.3f} s | QP solve: {timer() - solve_start:.3f} s")

    if mpc.Status == GRB.SUBOPTIMAL:
        log_event("solve_suboptimal", "using it anyway")
    elif mpc.Status != GRB.OPTIMAL:
        log_event("solve_failed", f"status {mpc.Status} - using the shifted previous plan")
        return nominal_input

    #Takes the optimized du input sequence and adds it to the nominal input to get the actual input sequence
    input_sequence = du.X + nominal_input
    log_solution([slack.X.max() for slack in (slack_roll, slack_pitch, slack_contact_distance,
                                              slack_rotor_distance, slack_approach)],
                 input_sequence[0], du.X[0])

    return input_sequence

#Batched QP builder: the same MPC problem as mpc_control, but every cost and constraint is added
#for the whole horizon at once using stacked (block) matrices, instead of one horizon step at a time.
#The math is identical; it just makes a handful of big Gurobi calls instead of hundreds of small ones.
#
#Stacked variables:
#   dx: (N+1)*18 numbers  [dx_0, dx_1, ..., dx_N]
#   du:  N*4     numbers  [du_0, du_1, ..., du_{N-1}]
#Stacked outputs (17 per step, k = 0..N):   y = y_nominal + C_big @ dx + D_big @ du   (D_big includes the motor map)


#Empty (N*nu, (N+1)*nx) matrix: the dx part of a row that only involves the inputs
def zeros_dx_(N, nx, nu):
    return sp.csr_matrix((N * nu, (N + 1) * nx))


#Sum of weight * (Mx @ dx + Mu @ du + c)^2 over every row, through one error variable per row.
#Each row is scaled by sqrt(weight) so the objective's Q matrix stays the identity (always convex).
def stacked_squared_error(mpc, weights, Mx, dx, Mu, du, c):
    scale = sp.diags(np.sqrt(weights))
    error = mpc.addMVar(len(c), lb=-GRB.INFINITY)
    mpc.addConstr(error == drop_small(scale @ Mx) @ dx + drop_small(scale @ Mu) @ du + scale @ c)
    return error @ error


def mpc_control_batched(A, B, C, D, nominal_state, nominal_input, nominal_output, input_vector_current,
                        reference_trajectory, hover_thrust_values, horizon_length, d0, tau, wall_normal_vector,
                        model_params, motor_speed_squared, return_plan=False):
    #model_params: dict(k_f, mass, gravity, mu, contact_radius, dt) from the model, for the stuck push, the approach speed,
    #and the motor lag (dt is the MPC step)
    #motor_speed_squared: (4,) the rotors' squared speeds right now (from motor_step in the run file)
    #nominal_input: the nominal COMMANDS; A, B, C, D are linearized at the effective (lagged) speeds they give
    #return_plan=True also returns the planned outputs (N+1, 17): what the QP predicts the outputs
    #will be with its chosen inputs, y_nom + C dx + D du (the nominal outputs if the solve fails)

    #Converts any torch tensors to numpy, since Gurobi only works with numpy
    nominal_state = [to_np(x) for x in nominal_state]
    nominal_output = [to_np(y) for y in nominal_output]
    nominal_input = to_np(nominal_input)
    hover_thrust_values = to_np(hover_thrust_values)
    input_vector_current = to_np(input_vector_current)

    #What to return when the solve doesn't work: the shifted previous plan, and its nominal outputs
    def fallback():
        return (nominal_input, np.array(nominal_output[:horizon_length + 1])) if return_plan else nominal_input

    #If the linearization produced NaN/inf anywhere, don't build the QP, fall back to the shifted plan
    for name, matrices in [("A", A), ("B", B), ("C", C), ("D", D), ("output", nominal_output), ("state", nominal_state)]:
        for k, matrix in enumerate(matrices):
            if not np.all(np.isfinite(matrix)):
                log_event("nan_linearization", f"{name}[{k}] - using the shifted previous plan")
                return fallback()

    A, B, C, D = clean_jacobians(A), clean_jacobians(B), clean_jacobians(C), clean_jacobians(D)

    build_start = timer()
    N = horizon_length
    nx, nu, ny = STATE_SIZE - 1, INPUT_SIZE, len(nominal_output[0])     # 18, 4, 17
    n = np.asarray(wall_normal_vector, dtype=float)


    # ---------------- Stacked linear model ----------------

    #Motor lag: the model's input is the effective rotor speed, d(effective) = M d(command), so every place the
    #model's input enters (B, D, and the push) is multiplied by M; the input costs and limits stay on the commands
    effective_input, motor_matrix = motor_map(nominal_input[:N], motor_speed_squared, model_params["dt"])

    #Outputs at every step k = 0..N. The last step has no input, so its D block is zeros.
    C_big = sp.block_diag(C[:N + 1], format="csr")                                          # ((N+1)*17, (N+1)*18)
    D_big = sp.vstack([sp.block_diag(D[:N]), sp.csr_matrix((ny, N * nu))], format="csr")     # ((N+1)*17, N*4)
    D_big = (D_big @ motor_matrix).tocsr()
    y_nominal = np.concatenate(nominal_output[:N + 1])                                       # ((N+1)*17,)

    def output_rows(output_indices, steps):
        #Row numbers in the stacked output for the given output entries at the given steps (step-major)
        return np.array([k * ny + j for k in steps for j in output_indices])

    #Contact gate at every step, from the nominal pad distance (output 12), times the contact flag
    pad_distance = y_nominal[output_rows([12], range(N + 1))]
    contact_gate = snap_gate(1.0 / (1.0 + np.exp(-(d0 - pad_distance) / tau)))
    contact_sigma = reference_trajectory["contact_flag"][:N + 1] * contact_gate

    #Steps 1..N carry the state and output costs (dx_0 = 0, so step 0 is a constant)
    later_steps = range(1, N + 1)


    # ---------------- Variables ----------------

    #State and input deviations. The input bounds combine the motor limits and the trust region.
    mpc = gp.Model("Aerial_Manipulator_batched", env=gurobi_env)
    mpc.Params.OutputFlag = 0
    mpc.Params.TimeLimit = 1.0
    mpc.Params.NonConvex = 0
    #Barrier convergence tolerance: Gurobi's default (1e-8) is far tighter than an MPC step needs, and solves
    #that stall just short of it are what come back as numerical trouble (12) or suboptimal (13)
    mpc.Params.BarConvTol = 1e-6

    nominal_input_flat = nominal_input[:N].reshape(-1)
    du_lower = np.maximum(MOTOR_MIN - nominal_input_flat, -INPUT_TRUST_REGION)
    du_upper = np.minimum(MOTOR_MAX - nominal_input_flat, INPUT_TRUST_REGION)
    dx = mpc.addMVar((N + 1) * nx, lb=-GRB.INFINITY, name="dx")

    #Pre-stabilization: du = K_big dx + v, with K_k applied to dx_k at every step k = 0..N-1 (K_big = 0 when off).
    #The gains ignore the motor lag (they're designed as if the command acted at once); that's fine, since they only
    #change the solver's coordinates, not which plans are feasible or optimal.
    #Every cost and constraint written as Mx @ dx + Mu @ du becomes (Mx + Mu K_big) @ dx + Mu @ v.
    prestabilized = PRESTABILIZE
    gains = prestabilizing_gains(A[:N], B[:N]) if prestabilized else [np.zeros((nu, nx))] * N
    K_big = sp.hstack([sp.block_diag(gains), sp.csr_matrix((N * nu, nx))], format="csr")

    def fold(Mx, Mu):
        Mu = sp.csr_matrix(Mu)
        return (sp.csr_matrix(Mx) + Mu @ K_big).tocsr(), Mu

    def linear(Mx, Mu):
        #The linear expression Mx @ dx + Mu @ du, in terms of dx and v
        Mx, Mu = fold(Mx, Mu)
        return drop_small(Mx) @ dx + drop_small(Mu) @ v

    def cost_rows(weights, Mx, Mu, c):
        #stacked_squared_error for Mx @ dx + Mu @ du + c, in terms of dx and v
        Mx, Mu = fold(Mx, Mu)
        return stacked_squared_error(mpc, weights, Mx, dx, Mu, v, c)

    #Input bounds (motor limits and trust region): simple bounds on v when not pre-stabilized,
    #linear constraints on K dx + v when pre-stabilized
    if prestabilized:
        v = mpc.addMVar(N * nu, lb=-GRB.INFINITY, name="v")
        mpc.addConstr(linear(zeros_dx_(N, nx, nu), sp.eye(N * nu)) >= du_lower)
        mpc.addConstr(linear(zeros_dx_(N, nx, nu), sp.eye(N * nu)) <= du_upper)
    else:
        v = mpc.addMVar(N * nu, lb=du_lower, ub=du_upper, name="du")

    #Slacks for the softened constraints, one per step k = 1..N
    slack_roll = mpc.addMVar(N, lb=0, name="slack_roll")
    slack_pitch = mpc.addMVar(N, lb=0, name="slack_pitch")
    slack_contact_distance = mpc.addMVar(N, lb=0, name="slack_contact_distance")
    slack_rotor_distance = mpc.addMVar(N, lb=0, name="slack_rotor_distance")
    slack_approach = mpc.addMVar(N, lb=0, name="slack_approach")

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
    mpc.addConstr(linear(next_state - current_state, -sp.block_diag(B[:N], format="csr") @ motor_matrix) == 0)


    # ---------------- Input costs and constraints (k = 0..N-1) ----------------

    #Input deviation from hover (the weight goes into the rows, like every other cost, so the
    #quadratic objective stays the identity and its range stays small)
    cost = cost_rows(INPUT_WEIGHT * np.ones(N * nu), zeros_dx(N * nu), sp.eye(N * nu, format="csr"),
                     nominal_input_flat - np.tile(hover_thrust_values, N))

    #Proximal cost: change from the nominal plan (du itself)
    if PROXIMAL_WEIGHT > 0:
        cost += cost_rows(PROXIMAL_WEIGHT * np.ones(N * nu), zeros_dx(N * nu), sp.eye(N * nu, format="csr"),
                          np.zeros(N * nu))

    #Input change from one step to the next (step 0 is compared with the input applied right now).
    #The variable is the change scaled by sqrt(INPUT_CHANGE_WEIGHT) (so its square is the cost directly),
    #and the motor change limit, scaled the same way, is its bound.
    previous_nominal_input = np.concatenate([input_vector_current, nominal_input_flat[:-nu]])
    input_difference = sp.eye(N * nu) - sp.eye(N * nu, k=-nu)
    change_scale = np.sqrt(INPUT_CHANGE_WEIGHT)
    input_change = mpc.addMVar(N * nu, lb=-change_scale * MOTOR_CHANGE_LIM, ub=change_scale * MOTOR_CHANGE_LIM,
                               name="input_change")
    mpc.addConstr(input_change == change_scale * (linear(zeros_dx(N * nu), input_difference)
                                                  + (nominal_input_flat - previous_nominal_input)))
    cost += input_change @ input_change

    #Normal force error whenever the pad is at the wall, over the whole trajectory (the reference
    #force is 0 outside the contact phase, so touching the wall when it shouldn't costs the same everywhere)
    rows = output_rows([0], range(N))
    cost += cost_rows(NORMAL_FORCE_WEIGHT * contact_gate[:N], C_big[rows], D_big[rows],
                      y_nominal[rows] - reference_trajectory["normal_force"][:N])

    #Push to break the pad loose, at the steps where the pad should move faster than the plan has it moving
    push_rows = stuck_push_rows(nominal_state, effective_input, nominal_output, reference_trajectory,
                                contact_sigma, n, model_params, N)
    if push_rows:
        steps = [row[0] for row in push_rows]
        row_index = np.arange(len(steps))
        push_dx = sp.csr_matrix((np.concatenate([row[1] for row in push_rows]),
                                 (np.repeat(row_index, 3), [k * nx + 6 + j for k in steps for j in range(3)])),
                                shape=(len(steps), (N + 1) * nx))
        push_du = sp.csr_matrix((np.concatenate([row[2] for row in push_rows]),
                                 (np.repeat(row_index, nu), [k * nu + j for k in steps for j in range(nu)])),
                                shape=(len(steps), N * nu)) @ motor_matrix
        scales = np.array([row[4] for row in push_rows])
        cost += cost_rows(PUSH_WEIGHT * scales, push_dx, push_du, np.array([row[3] for row in push_rows]))
        print(f"  push active at {len(steps)} of {N} steps (mean scale {scales.mean():.2f})")


    # ---------------- State and output costs (k = 1..N) ----------------

    #Tangent projection at each step: removes the normal direction during contact
    projection = sp.block_diag([np.eye(3) - contact_sigma[k] * np.outer(n, n) for k in later_steps], format="csr")

    #Pad position and velocity errors, projected into the wall plane during contact
    for output_indices, reference_key, weight in [([3, 4, 5], "pad_position", POSITION_WEIGHT),
                                                  ([6, 7, 8], "pad_velocity", VELOCITY_WEIGHT)]:
        rows = output_rows(output_indices, later_steps)
        reference = reference_trajectory[reference_key][1:N + 1].reshape(-1)
        cost += cost_rows(weight * np.ones(len(rows)), projection @ C_big[rows], projection @ D_big[rows],
                          projection @ (y_nominal[rows] - reference))

    #Rod orientation error: the part of the rod direction perpendicular to the wall normal
    #(pad pitch and yaw relative to the wall), zero when the rod points straight into the wall
    rod_rows = output_rows([9, 10, 11], later_steps)
    perpendicular = sp.kron(sp.eye(N), np.eye(3) - np.outer(n, n), format="csr")
    cost += cost_rows(ROD_ORIENTATION_WEIGHT * np.ones(3 * N),
                      perpendicular @ C_big[rod_rows], perpendicular @ D_big[rod_rows],
                      perpendicular @ y_nominal[rod_rows])

    #Angular velocity (state entries 10:13, error-state entries 9:12)
    angular_velocity_entries = [k * nx + i for k in later_steps for i in range(9, 12)]
    select_angular_velocity = sp.csr_matrix((np.ones(3 * N), (np.arange(3 * N), angular_velocity_entries)),
                                            shape=(3 * N, (N + 1) * nx))
    cost += cost_rows(ANG_VEL_WEIGHT * np.ones(3 * N), select_angular_velocity, zeros_du(3 * N),
                      np.concatenate([nominal_state[k][10:13] for k in later_steps]))

    #Slack costs: linear (L1), so the quadratic objective stays the identity
    cost += SLACK_WEIGHT_ROLL * slack_roll.sum()
    cost += SLACK_WEIGHT_PITCH * slack_pitch.sum()
    cost += SLACK_WEIGHT_CONTACT_DISTANCE * slack_contact_distance.sum()
    cost += SLACK_WEIGHT_ROTOR_DISTANCE * slack_rotor_distance.sum()
    cost += SLACK_WEIGHT_APPROACH * slack_approach.sum()


    # ---------------- Output constraints (k = 1..N) ----------------

    #Pad within CONTACT_MAX_DIST of the wall, only at steps where the gate says contact
    contact_steps = [k for k in later_steps if contact_sigma[k] >= 0.5]
    if contact_steps:
        rows = output_rows([12], contact_steps)
        select_slack = sp.csr_matrix((np.ones(len(contact_steps)), (np.arange(len(contact_steps)), [k - 1 for k in contact_steps])),
                                     shape=(len(contact_steps), N))
        mpc.addConstr(linear(C_big[rows], D_big[rows]) - select_slack @ slack_contact_distance
                      <= CONTACT_MAX_DIST - y_nominal[rows])

    #Rotors at least ROTOR_MIN_DIST from the wall (one slack per step, shared by the 4 rotors)
    rows = output_rows([13, 14, 15, 16], later_steps)
    each_rotor = sp.kron(sp.eye(N), np.ones((4, 1)), format="csr")
    mpc.addConstr(linear(C_big[rows], D_big[rows]) + each_rotor @ slack_rotor_distance
                  >= ROTOR_MIN_DIST - y_nominal[rows])

    #Approach speed limit: speed into the wall - gap / APPROACH_TIME <= TOUCH_SPEED, where
    #speed into the wall = -n . pad velocity (outputs 6:9) and gap = pad distance (output 12) - contact sphere radius
    #Only at steps where the planned pad isn't touching (contact gate < 0.5)
    approach_steps = [k for k in later_steps if contact_gate[k] < 0.5]
    if approach_steps:
        m = len(approach_steps)
        velocity_rows = output_rows([6, 7, 8], approach_steps)
        distance_rows = output_rows([12], approach_steps)
        into_wall = sp.kron(sp.eye(m), -n[None, :], format="csr")
        approach_dx = drop_small(into_wall @ C_big[velocity_rows] - C_big[distance_rows] / APPROACH_TIME)
        approach_du = drop_small(into_wall @ D_big[velocity_rows] - D_big[distance_rows] / APPROACH_TIME)
        approach_nominal = (into_wall @ y_nominal[velocity_rows]
                            - (y_nominal[distance_rows] - model_params["contact_radius"]) / APPROACH_TIME)
        select_slack = sp.csr_matrix((np.ones(m), (np.arange(m), [k - 1 for k in approach_steps])), shape=(m, N))
        mpc.addConstr(linear(approach_dx, approach_du) - select_slack @ slack_approach
                      <= TOUCH_SPEED - approach_nominal)

    #Roll and pitch limits
    for output_index, limit, slack in [(1, ROLL_LIM, slack_roll), (2, PITCH_LIM, slack_pitch)]:
        rows = output_rows([output_index], later_steps)
        mpc.addConstr(linear(C_big[rows], D_big[rows]) - slack <= limit - y_nominal[rows])
        mpc.addConstr(linear(C_big[rows], D_big[rows]) + slack >= -limit - y_nominal[rows])


    # ---------------- Solve ----------------

    mpc.setObjective(cost, GRB.MINIMIZE)

    solve_start = timer()
    growth = [A[k] + B[k] @ gains[k] for k in range(N)] if prestabilized else A[:N]
    try:
        solve_qp(mpc, growth, "closed-loop A+BK" if prestabilized else "A")
    except gp.GurobiError as error:
        log_event("solver_error", f"{error} - using the shifted previous plan")
        return fallback()
    print(f"  QP build: {solve_start - build_start:.3f} s | QP solve: {timer() - solve_start:.3f} s")

    if mpc.Status == GRB.SUBOPTIMAL:
        log_event("solve_suboptimal", "using it anyway")
    elif mpc.Status != GRB.OPTIMAL:
        log_event("solve_failed", f"status {mpc.Status} - using the shifted previous plan")
        return fallback()

    #Input deviations the QP chose: du = K dx + v
    du_value = K_big @ dx.X + v.X

    print(f"  max slack: roll {slack_roll.X.max():.4f} rad, pitch {slack_pitch.X.max():.4f} rad, "
             f"contact {1000 * slack_contact_distance.X.max():.2f} mm, rotor {1000 * slack_rotor_distance.X.max():.2f} mm, "
             f"approach {100 * slack_approach.X.max():.1f} cm/s")

    #Optimized input sequence = nominal + deviation
    input_sequence = du_value.reshape(N, nu) + nominal_input[:N]
    log_solution([slack.X.max() for slack in (slack_roll, slack_pitch, slack_contact_distance,
                                              slack_rotor_distance, slack_approach)],
                 input_sequence[0], du_value[:nu])
    if not return_plan:
        return input_sequence

    #The outputs the QP planned on with those inputs
    planned_outputs = (y_nominal + C_big @ dx.X + D_big @ du_value).reshape(N + 1, ny)
    return input_sequence, planned_outputs