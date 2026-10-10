"""Known physics for the aerial manipulator. The rotor model turns measured rotor
speeds into a body-frame thrust force and torque about the COM. The analytic
aero law (quadratic body drag + rotor drag) is the anchor target for the
learned aero model. rigid_step_drone() generalizes the cube's rigid_step to a
non-isotropic body (full inertia, gyroscopic effect) by carrying the angular
momentum across the step instead of using tau / I_OVER_M. node_accel_from_wrench() gives the acceleration each node
would have under a given wrench, which is how the contact GNN sees the push.

Units follow the cube code: positions in m, everything the integrator consumes
is a SPECIFIC force or torque (divided by mass) in per-recorded-step^2 units.
  specific force  f   = F / m * dt^2        (m / step^2)
  specific torque tau = T / m * dt^2        (m^2 / step^2)
  J               = I / m                   (m^2)
so alpha = J^-1 (tau - w x J w) comes out in rad / step^2.
"""

import torch

from force_gns import so3_exp, so3_log


# Thrust force and torque about the COM from each rotor, in the body frame.
# omega: (B, M) rotor speeds in rad/s. Returns F (B,3) in N and T (B,3) in N m.
def rotor_wrench_body(omega, rotor_pos, rotor_axis, spin_dir, k_f, k_m):
    w2 = omega.pow(2)                                           # (B, M)
    F_i = (k_f * w2).unsqueeze(-1) * rotor_axis                 # (B, M, 3) thrust along each axis
    lever = torch.linalg.cross(rotor_pos.expand_as(F_i), F_i, dim=-1)
    # A rotor spinning CCW about +axis drags the body CW: reaction is -spin * k_m * omega^2 * axis.
    reaction = -(spin_dir * k_m * w2).unsqueeze(-1) * rotor_axis
    return F_i.sum(dim=1), (lever + reaction).sum(dim=1)


# Analytic aero law used as the anchor target (and as the baseline if it is ever turned on).
# u_body: (B,3) relative air velocity (wind - v) in the body frame, m/s.
# Returns a body-frame specific force in m/s^2:
#   quadratic body drag    k_body * |u| u
#   rotor drag             k_rotor * (sum_i omega_i) * u_perp
# where u_perp removes the thrust-axis component (rotor drag acts in the disk plane).
# Both point along +u: a drone moving at v through still air gets a force along -v.
def aero_law_body(u_body, omega, k_body, k_rotor, thrust_axis):
    quad = k_body * u_body.norm(dim=-1, keepdim=True) * u_body
    u_perp = u_body - (u_body * thrust_axis).sum(-1, keepdim=True) * thrust_axis
    rotor = k_rotor * omega.sum(dim=-1, keepdim=True) * u_perp
    return quad + rotor


# World-frame inertia-over-mass, J_w = R J R^T.
def world_inertia(R, J_body):
    return R @ J_body @ R.transpose(-1, -2)


# J_w^-1 applied to a vector, using J_w^-1 = R J_body^-1 R^T exactly (R is a rotation).
# Replaces torch.linalg.solve: same result, fewer kernels, and no GPU->CPU sync
# (linalg.solve checks for singular matrices on the host). inv_ex skips that check;
# J_body is a constant, physical inertia, so it is never singular.
def world_inertia_solve(R, J_body, v):
    J_inv = torch.linalg.inv_ex(J_body)[0]
    return (R @ (J_inv @ (R.transpose(-1, -2) @ v.unsqueeze(-1)))).squeeze(-1)


# Angular acceleration from a specific torque, including the gyroscopic term.
# tau, w: (B,3) world frame, per-step units. Returns alpha (B,3) in rad/step^2.
def angular_accel(tau, w, R, J_body):
    J_w = world_inertia(R, J_body)
    gyro = torch.linalg.cross(w, (J_w @ w.unsqueeze(-1)).squeeze(-1), dim=-1)
    return world_inertia_solve(R, J_body, tau - gyro)


# Acceleration of every node of the rigid body under a given wrench.
# f must already include gravity. r: (B,N,3) node lever arms from the COM, world frame.
# a_i = f + alpha x r_i + w x (w x r_i)
def node_accel_from_wrench(f, tau, w, R, J_body, r):
    alpha = angular_accel(tau, w, R, J_body).unsqueeze(1)       # (B,1,3)
    w_ = w.unsqueeze(1).expand_as(r)
    return (f.unsqueeze(1)
            + torch.linalg.cross(alpha.expand_as(r), r, dim=-1)
            + torch.linalg.cross(w_, torch.linalg.cross(w_, r, dim=-1), dim=-1))


# One Newton-Euler step of the drone in per-step units: Verlet on the COM (same
# as the cube) and a Lie-group update on the rotation written in angular-
# momentum form. The cube could use w_next = w + tau / I_OVER_M because its
# inertia is isotropic. Here the inertia rotates with the body, so the update
# carries the world angular momentum across the step instead:
#     L_prev = J_w(R_half_prev) w_prev,    L_next = L_prev + tau
#     solve J_w(R_half_next) w_next = L_next   (R_half_next depends on w_next,
#                                              so a few fixed-point passes)
# The gyroscopic effect comes out of J_w changing with attitude. The explicit
# form J_w^-1 (tau - w x J_w w) drifts ~1% in |L| per few hundred steps of
# torque-free tumbling; this form conserves it to the iteration tolerance.
# f_ext:   (B,3) total specific force EXCLUDING gravity (thrust + aero + contact)
# tau_ext: (B,3) total specific torque about the COM (thrust + aero + contact)
def rigid_step_drone(f_ext, tau_ext, com_prev, com_curr, R_prev, R_curr, J_body, g_step,
                     n_iter=3):
    com_next = 2.0 * com_curr - com_prev + f_ext + g_step

    # Angular velocity over the last step (world frame, rad/step) and the
    # angular momentum it carried, evaluated at the mid-step attitude.
    w_prev = so3_log(R_curr @ R_prev.transpose(-1, -2))
    R_half_prev = so3_exp(0.5 * w_prev) @ R_prev
    L_next = (world_inertia(R_half_prev, J_body) @ w_prev.unsqueeze(-1)).squeeze(-1) + tau_ext

    # Fixed-point solve for the next angular velocity, seeded with the explicit update.
    w_next = w_prev + angular_accel(tau_ext, w_prev, R_curr, J_body)
    for _ in range(n_iter):
        R_half_next = so3_exp(0.5 * w_next) @ R_curr
        w_next = world_inertia_solve(R_half_next, J_body, L_next)

    R_next = so3_exp(w_next) @ R_curr
    return com_next, R_next


# Converts per-step specific quantities back to Newtons / Newton-meters.
def to_newtons(x, mass, dt):
    return mass * x / (dt * dt)
