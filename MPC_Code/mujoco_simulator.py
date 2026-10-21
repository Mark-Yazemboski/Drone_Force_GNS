#Self-contained MuJoCo simulator of the aerial manipulator (drone + rod + pad) next to a wall.
#Set it up once with the drone config, wall, and wind; reset it to an initial state; then step it
#with rotor speeds. The physics match the simulator that generated the GNN's training data.

from collections import deque
import numpy as np
import mujoco
from utilities import vec_to_str, quat_to_rot_matrix_np, rot_matrix_to_quat_np, so3_log_np

def wall_tangents(wall_normal):
    #Two directions in the wall plane: horizontal along the wall, and up the wall
    along_wall = np.cross(wall_normal, [0.0, 0.0, 1.0])
    along_wall = along_wall / np.linalg.norm(along_wall)
    up_wall = np.cross(along_wall, wall_normal)
    return along_wall, up_wall


def pad_node_positions(cfg):
    #Pad center plus the ring(s) of pad nodes, in the body frame.
    #Each one becomes a small collision sphere, so the sim contact points are the GNN's pad nodes.
    tip = np.asarray(cfg.rod_tip, dtype=float)
    base = np.asarray(cfg.rod_base, dtype=float)
    rod_axis = (tip - base) / np.linalg.norm(tip - base)

    #Two directions perpendicular to the rod, spanning the pad's plane
    helper = np.array([0.0, 0.0, 1.0]) if abs(rod_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(rod_axis, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(rod_axis, u)

    nodes = [tip]
    for radius, count in cfg.pad_rings:
        for k in range(count):
            angle = 2.0 * np.pi * k / count
            nodes.append(tip + radius * (np.cos(angle) * u + np.sin(angle) * v))
    return np.asarray(nodes)


#---------------------------------- Simulator ----------------------------------------

class MujocoDroneSim:

    def __init__(self, drone_cfg, wall_normal_vector, wall_center_point, wind_vector,
                 physics_dt=0.0005, motor_tau=0.025, friction=0.4, rotor_drag_coeff=5e-5,
                 wall_solref=(0.004, 1.0), pad_solref=(0.004, 1.0), show_viewer=False):
        #drone_cfg: the DroneConfig saved in the trained model's checkpoint (model.cfg)
        #The physics settings default to the values the training data was generated with.

        self.cfg = drone_cfg
        self.physics_dt = physics_dt
        self.motor_tau = motor_tau
        self.friction = friction
        self.rotor_drag_coeff = rotor_drag_coeff
        self.wall_solref = wall_solref
        self.pad_solref = pad_solref

        #Environment, fixed for the whole run
        self.wall_normal_vector = np.asarray(wall_normal_vector, dtype=float)
        self.wall_center_point = np.asarray(wall_center_point, dtype=float)
        self.wind_vector = np.asarray(wind_vector, dtype=float)

        #Builds the MuJoCo model and its data
        self.mj_model = mujoco.MjModel.from_xml_string(self.build_xml())
        self.data = mujoco.MjData(self.mj_model)

        #Drone body and geometry, read back from the compiled model
        self.drone_id = self.mj_model.body("drone").id
        self.mass = float(self.mj_model.body_mass[self.drone_id])
        self.com_offset = self.mj_model.body_ipos[self.drone_id].copy()           # body origin -> COM, body frame
        if abs(self.mass - self.cfg.mass) > 1e-6:
            print("WARNING: sim drone mass differs from the model's config, check the drone config")

        #Rotor hubs relative to the COM and rotor axes, body frame (for rotor drag)
        self.rotor_positions = np.asarray(self.cfg.rotor_pos, dtype=float) - self.com_offset
        rotor_axes = np.asarray(self.cfg.rotor_axis, dtype=float)
        self.rotor_axes = rotor_axes / np.linalg.norm(rotor_axes, axis=1, keepdims=True)

        #Geom ids used to sort contacts: the wall, the pad spheres, and everything on the drone
        self.wall_geom_id = self.mj_model.geom("wall").id
        num_pads = len(pad_node_positions(self.cfg))
        self.pad_geom_ids = {self.mj_model.geom(f"pad{k}").id for k in range(num_pads)}
        self.drone_geom_ids = set(np.flatnonzero(self.mj_model.geom_bodyid == self.drone_id).tolist())

        #The GNN's step (0.01 s); the measured state uses poses this far apart
        self.frame_dt = self.cfg.dt
        self.substeps_per_frame = round(self.frame_dt / self.physics_dt)

        #Optional live 3D view
        self.viewer = None
        if show_viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.data)


    def build_xml(self):
        #Writes the MuJoCo model: drone body (core, arms, props, rod, pad spheres), four thrust
        #actuators with motor lag, and a wall. Masses and shapes match the training-data simulator.
        cfg = self.cfg
        tip = np.asarray(cfg.rod_tip, dtype=float)
        base = np.asarray(cfg.rod_base, dtype=float)
        rod_axis = (tip - base) / np.linalg.norm(tip - base)
        pad_disk_radius = max(radius for radius, _ in cfg.pad_rings) + cfg.pad_contact_radius

        #Arms, propeller disks, and a site at each rotor hub where the thrust acts
        rotors_xml = ""
        for i, (position, axis) in enumerate(zip(cfg.rotor_pos, cfg.rotor_axis)):
            rotors_xml += f"""
      <geom name="arm{i}" type="capsule" fromto="0 0 0 {vec_to_str(position)}" size="0.008" mass="0.04" rgba=".3 .3 .3 1"/>
      <geom name="prop{i}" type="cylinder" pos="{vec_to_str(position)}" zaxis="{vec_to_str(axis)}" size="0.06 0.005" mass="0.05" rgba=".2 .5 .9 .5"/>
      <site name="rotor{i}" pos="{vec_to_str(position)}" zaxis="{vec_to_str(axis)}"/>"""

        #One small collision sphere per pad node
        pads_xml = ""
        for k, node in enumerate(pad_node_positions(cfg)):
            pads_xml += f"""
      <geom name="pad{k}" type="sphere" pos="{vec_to_str(node)}" size="{cfg.pad_contact_radius}" mass="0.001"
            friction="{self.friction} 0.005 0.0001" condim="3" solref="{vec_to_str(self.pad_solref)}" rgba=".9 .2 .2 1"/>"""

        #Thrust actuators: input is thrust (N); the gear adds the yaw reaction torque k_m/k_f per newton,
        #and the filter gives the motor lag
        actuators_xml = ""
        for i, spin in enumerate(cfg.spin_dir):
            actuators_xml += f"""
    <general name="motor{i}" site="rotor{i}" gear="0 0 1 0 0 {-spin * cfg.k_m / cfg.k_f:.6g}"
             dyntype="filter" dynprm="{self.motor_tau}" gainprm="1" biastype="none"
             ctrllimited="true" ctrlrange="0 100"/>"""

        return f"""
<mujoco model="aerial_manipulator">
  <option timestep="{self.physics_dt}" gravity="0 0 {-cfg.gravity}" density="1.225"
          viscosity="1.8e-5" integrator="Euler" cone="elliptic"/>
  <default><geom contype="1" conaffinity="0"/></default>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="floor" type="plane" size="5 5 .1" contype="0" conaffinity="0" rgba=".8 .8 .8 1"/>
    <body name="wall_body" mocap="true" pos="0 0 -10">
      <geom name="wall" type="box" size="0.02 2 2" contype="0" conaffinity="1"
            friction="{self.friction} 0.005 0.0001" solref="{vec_to_str(self.wall_solref)}" rgba=".7 .6 .5 .6"/>
    </body>
    <body name="drone" pos="0 0 1">
      <freejoint/>
      <geom name="core" type="box" size="0.06 0.06 0.025" mass="0.9" fluidshape="ellipsoid" rgba=".1 .1 .1 1"/>{rotors_xml}
      <geom name="rod" type="capsule" fromto="{vec_to_str(base)} {vec_to_str(tip - 0.012 * rod_axis)}" size="0.006" mass="0.06"
            fluidshape="ellipsoid" rgba=".6 .6 .6 1"/>
      <geom name="sensor" type="cylinder" pos="{vec_to_str(base + 0.02 * rod_axis)}" zaxis="{vec_to_str(rod_axis)}"
            size="0.015 0.012" mass="0.05" rgba=".9 .7 .1 1"/>
      <geom name="pad_disk" type="cylinder" pos="{vec_to_str(tip - 0.002 * rod_axis)}" zaxis="{vec_to_str(rod_axis)}"
            size="{pad_disk_radius:.4f} 0.002" mass="0.01" contype="0" conaffinity="0" rgba=".9 .4 .4 .6"/>{pads_xml}
    </body>
  </worldbody>
  <actuator>{actuators_xml}
  </actuator>
</mujoco>"""


    def place_wall(self):
        #Moves the wall box so its surface passes through wall_center_point, facing wall_normal_vector.
        #The wall is a mocap body because MuJoCo doesn't re-pose fixed geoms after the model is built.
        along_wall, _ = wall_tangents(self.wall_normal_vector)
        half_thickness = self.mj_model.geom_size[self.wall_geom_id][0]
        box_rotation = np.column_stack([self.wall_normal_vector, along_wall,
                                        np.cross(self.wall_normal_vector, along_wall)])     # box x-axis = wall normal
        self.data.mocap_pos[0] = self.wall_center_point - half_thickness * self.wall_normal_vector
        self.data.mocap_quat[0] = rot_matrix_to_quat_np(box_rotation)


    def drone_pose(self):
        #COM position, rotation matrix, COM velocity (world), and body angular velocity
        R = quat_to_rot_matrix_np(self.data.qpos[3:7])
        body_angular_velocity = self.data.qvel[3:6].copy()
        com_position = self.data.qpos[0:3] + R @ self.com_offset
        com_velocity = self.data.qvel[0:3] + np.cross(R @ body_angular_velocity, R @ self.com_offset)
        return com_position, R, com_velocity, body_angular_velocity


    def apply_rotor_drag(self):
        #MuJoCo has no rotor drag, so it's added by hand: each rotor pushes with
        #k_rotor * m * omega * (local airspeed in the rotor plane), at its hub
        com_position, R, com_velocity, body_angular_velocity = self.drone_pose()

        rotor_speeds = np.sqrt(np.maximum(self.data.act, 0.0) / self.cfg.k_f)        # actual (lagged) rotor speeds
        hub_offsets = self.rotor_positions @ R.T                                      # COM -> hub, world frame
        hub_axes = self.rotor_axes @ R.T

        #Airspeed at each hub: wind minus the hub's velocity (including rotation)
        hub_velocities = com_velocity + np.cross(R @ body_angular_velocity, hub_offsets)
        hub_airspeed = self.wind_vector - hub_velocities
        hub_airspeed_in_plane = hub_airspeed - np.sum(hub_airspeed * hub_axes, axis=1, keepdims=True) * hub_axes

        hub_forces = self.mass * self.rotor_drag_coeff * rotor_speeds[:, None] * hub_airspeed_in_plane

        #xfrc_applied acts at the COM, so the hubs' moment is added as a torque
        self.data.xfrc_applied[self.drone_id, :3] = hub_forces.sum(axis=0)
        self.data.xfrc_applied[self.drone_id, 3:] = np.cross(hub_offsets, hub_forces).sum(axis=0)


    def read_contacts(self):
        #Total contact force the wall applies to the pad (world frame, N).
        #Returns None if any part of the drone other than the pad touches the wall (a crash).
        contact_force = np.zeros(3)
        force_6d = np.zeros(6)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self.wall_geom_id not in (contact.geom1, contact.geom2):
                continue

            other_geom = contact.geom2 if contact.geom1 == self.wall_geom_id else contact.geom1
            if other_geom not in self.pad_geom_ids:
                if other_geom in self.drone_geom_ids:
                    return None
                continue

            #MuJoCo gives the force in the contact frame, acting on geom2
            mujoco.mj_contactForce(self.mj_model, self.data, i, force_6d)
            force_world = contact.frame.reshape(3, 3).T @ force_6d[:3]
            if other_geom == contact.geom1:
                force_world = -force_world
            contact_force += force_world

        return contact_force


    def reset(self, initial_state, initial_rotor_speeds):
        #Puts the drone at the initial state (19-vector from the trajectory generator), at rest,
        #with the motors already spinning at initial_rotor_speeds (usually hover)
        mujoco.mj_resetData(self.mj_model, self.data)
        self.place_wall()

        #The state is about the COM, but MuJoCo's position is the body origin, so shift by the COM offset
        com_position = np.asarray(initial_state[0:3], dtype=float)
        quaternion = np.asarray(initial_state[6:10], dtype=float)
        R = quat_to_rot_matrix_np(quaternion)

        self.data.qpos[0:3] = com_position - R @ self.com_offset
        self.data.qpos[3:7] = quaternion
        self.data.qvel[:] = 0.0

        #The actuators take THRUST (N), so convert with T = k_f * w^2
        initial_thrust = self.cfg.k_f * np.asarray(initial_rotor_speeds, dtype=float) ** 2
        self.data.act[:] = initial_thrust
        self.data.ctrl[:] = initial_thrust

        #Wind for MuJoCo's body drag (rotor drag is added by hand each substep)
        self.mj_model.opt.wind[:] = self.wind_vector

        mujoco.mj_forward(self.mj_model, self.data)

        #Bookkeeping
        self.time = 0.0
        self.substep_count = 0
        self.crashed = False
        self.crash_message = ""
        self.contact_force = np.zeros(3)

        #Last three poses, 0.01 s apart (at rest: three copies of the start pose)
        com_position, R, _, _ = self.drone_pose()
        self.pose_buffer = deque([(com_position.copy(), R.copy())] * 3, maxlen=3)

        return self.get_state()


    def step(self, rotor_speeds, duration):
        #Holds the commanded rotor speeds for `duration` seconds of sim time and returns the new state.
        #duration should be a multiple of 0.01 s (for example, one MPC step).
        if self.crashed:
            return self.get_state()

        #Rotor speed command -> thrust command, inside the actuator range
        thrust_command = self.cfg.k_f * np.asarray(rotor_speeds, dtype=float) ** 2
        self.data.ctrl[:] = np.clip(thrust_command, 0.0, 100.0)

        num_substeps = round(duration / self.physics_dt)
        for _ in range(num_substeps):

            #One physics substep with rotor drag
            self.apply_rotor_drag()
            mujoco.mj_step(self.mj_model, self.data)
            self.time += self.physics_dt
            self.substep_count += 1

            #Contact force on the pad, and a crash check for anything else touching the wall
            contact_force = self.read_contacts()
            if contact_force is None:
                self.crashed = True
                self.crash_message = f"part of the drone other than the pad hit the wall at t={self.time:.2f}s"
                break
            self.contact_force = contact_force

            #Stop if the drone flips over or the sim blows up numerically
            _, R, _, _ = self.drone_pose()
            if not np.all(np.isfinite(self.data.qpos)) or (R @ [0, 0, 1])[2] < 0.5:
                self.crashed = True
                self.crash_message = f"state left the envelope at t={self.time:.2f}s"
                break

            #Every 0.01 s, record the pose for the finite-difference state
            if self.substep_count % self.substeps_per_frame == 0:
                com_position, R, _, _ = self.drone_pose()
                self.pose_buffer.append((com_position.copy(), R.copy()))
                if self.viewer is not None:
                    self.viewer.sync()

        return self.get_state()


    def get_state(self):
        #The 19-number state in the model's format, with velocities from finite differences of the
        #last three poses (0.01 s apart), the same way the GNN saw them during training:
        #[pos, vel, quat, ang_vel, vel_prev, ang_vel_prev], angular velocities in the WORLD frame
        (p2, R2), (p1, R1), (p0, R0) = self.pose_buffer

        velocity = (p0 - p1) / self.frame_dt
        velocity_prev = (p1 - p2) / self.frame_dt
        angular_velocity = so3_log_np(R0 @ R1.T) / self.frame_dt
        angular_velocity_prev = so3_log_np(R1 @ R2.T) / self.frame_dt
        quaternion = rot_matrix_to_quat_np(R0)

        return np.concatenate([p0, velocity, quaternion, angular_velocity, velocity_prev, angular_velocity_prev])


    def get_normal_force(self):
        #Normal force the wall pushes on the pad with (N), from the last substep
        return float(self.contact_force @ self.wall_normal_vector)