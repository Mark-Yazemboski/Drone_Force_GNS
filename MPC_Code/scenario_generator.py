import numpy as np


class Trajectory_Generator:
    def __init__(self, Wind_range, wall_yaw_range, Ref_traj_dt):
        #Range of the wind speed magnitude (m/s)
        self.Wind_min = Wind_range[0]
        self.Wind_max = Wind_range[1]

        #Kept for reference; the actual length comes from the segments

        #Range of the wall's yaw angle (rotation of the wall about the vertical axis)
        self.min_wall_yaw = wall_yaw_range[0]
        self.max_wall_yaw = wall_yaw_range[1]

        #Time step of the coarse reference trajectory
        self.Ref_traj_dt = Ref_traj_dt


    def set_pad_offset(self, pad_offset):
        self.pad_offset = np.asarray(pad_offset, dtype=float)
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def min_jerk(self, start, end, num_steps):
        #Smooth move from start to end: zero velocity and acceleration at both ends
        #start, end: (3,) positions or scalars.  Returns (num_steps, 3) or (num_steps, 1)
        s = np.linspace(0.0, 1.0, num_steps)[:, None]
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        return start + (end - start) * blend

    def make_segment(self, positions, forces, contact_flags):
        #Packs one segment into the reference dictionary
        #Velocity comes from differentiating the positions, so it always matches them
        if len(positions) > 1:
            velocities = np.gradient(positions, self.Ref_traj_dt, axis=0)
        else:
            velocities = np.zeros_like(positions)

        segment = {
            "pad_position": positions,                      # (n, 3)
            "pad_velocity": velocities,                     # (n, 3)
            "normal_force": np.asarray(forces).reshape(-1),         # (n,)
            "contact_flag": np.asarray(contact_flags).reshape(-1)   # (n,)
        }
        return segment

    def steps_for(self, time_range):
        #Random duration in time_range, converted to a number of reference steps (at least 2)
        duration = np.random.uniform(time_range[0], time_range[1])
        num_steps = max(int(duration / self.Ref_traj_dt), 2)
        return num_steps


    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def initilize_state(self, distance_range, yaw_offset_range):
        #Starts the drone hovering at rest, with the pad a random distance from the wall,
        #and the rod pointed at the wall (plus a small random yaw offset)

        #Yaw that points the body x-axis (the rod) into the wall
        heading_into_wall = np.arctan2(-self.Wall_normal_vector[1], -self.Wall_normal_vector[0])
        starting_yaw = heading_into_wall + np.random.uniform(yaw_offset_range[0], yaw_offset_range[1])

        #Yaw-only rotation matrix and quaternion [w, x, y, z]
        R_yaw = np.array([
            [np.cos(starting_yaw), -np.sin(starting_yaw), 0],
            [np.sin(starting_yaw),  np.cos(starting_yaw), 0],
            [0, 0, 1]
        ])
        starting_quaternion = np.array([np.cos(starting_yaw / 2), 0.0, 0.0, np.sin(starting_yaw / 2)])

        #Place the PAD at the chosen distance from the wall, then back out the COM position
        Pad_starting_distance = np.random.uniform(distance_range[0], distance_range[1])
        self.Pad_starting_position = self.Wall_center_point + Pad_starting_distance * self.Wall_normal_vector
        Drone_starting_position = self.Pad_starting_position - R_yaw @ self.pad_offset

        #Starts at rest
        starting_vel = np.zeros(3)
        starting_ang_vel = np.zeros(3)

        #State order the model expects: [pos, vel, quat, ang_vel, vel_prev, ang_vel_prev]
        Starting_state_vector = np.concatenate([Drone_starting_position, starting_vel, starting_quaternion,
                                                starting_ang_vel, starting_vel, starting_ang_vel])

        return Starting_state_vector


    # ------------------------------------------------------------------
    # Segments
    # ------------------------------------------------------------------

    def hold_mode(self, time_range):
        #Hover in place at the starting pad position
        num_steps = self.steps_for(time_range)

        positions = np.tile(self.Pad_starting_position, (num_steps, 1))
        forces = np.zeros(num_steps)
        contact_flags = np.zeros(num_steps)

        return self.make_segment(positions, forces, contact_flags)

    def move_to_wall(self, ref_trajectory, time_range, contact_depth=0.002):
        #Smoothly moves the pad straight to the wall, ending slightly BEHIND the surface
        #(contact_depth) so the position error keeps pulling the pad into contact
        num_steps = self.steps_for(time_range)
        move_steps = max(int(num_steps * 0.8), 2)
        hold_steps = max(num_steps - move_steps, 1)

        ref_pad_position_initial = ref_trajectory["pad_position"][-1]

        #Point on the wall straight in front of the pad, pushed contact_depth into the wall
        distance_to_wall = self.Wall_normal_vector @ (ref_pad_position_initial - self.Wall_center_point)
        ref_pad_position_final = (ref_pad_position_initial
                                  - distance_to_wall * self.Wall_normal_vector
                                  - contact_depth * self.Wall_normal_vector)

        #Approach (no contact yet)
        move_positions = self.min_jerk(ref_pad_position_initial, ref_pad_position_final, move_steps)

        #Short settle at the wall before pushing
        hold_positions = np.tile(ref_pad_position_final, (hold_steps, 1))

        positions = np.vstack([move_positions, hold_positions])
        forces = np.zeros(len(positions))
        contact_flags = np.zeros(len(positions))

        return self.make_segment(positions, forces, contact_flags)

    def push_ramp(self, ref_trajectory, time_range, force_range=(1.0, 3.0)):
        #Holds position at the wall while smoothly ramping the normal force from 0 to the push force
        num_steps = self.steps_for(time_range)

        #One push force for the whole contact phase (the circle uses the same value)
        self.push_force = np.random.uniform(force_range[0], force_range[1])

        positions = np.tile(ref_trajectory["pad_position"][-1], (num_steps, 1))
        forces = self.min_jerk(0.0, self.push_force, num_steps).reshape(-1)
        contact_flags = np.ones(num_steps)

        return self.make_segment(positions, forces, contact_flags)

    def draw_circle(self, ref_trajectory, time_range, radius_range=(0.1, 0.2)):
        #Slides the pad around one circle on the wall while holding the push force
        num_steps = self.steps_for(time_range)
        radius = np.random.uniform(radius_range[0], radius_range[1])

        ref_pad_position_initial = ref_trajectory["pad_position"][-1]

        #Starting point in the wall frame: [out of wall, along wall, up the wall]
        wall_frame_initial_position = self.Wall_rotation_matrix @ (ref_pad_position_initial - self.Wall_center_point)

        #Circle center is one radius above the start, so the circle begins exactly at the start point
        circle_center = wall_frame_initial_position + np.array([0.0, 0.0, radius])

        #Angle goes from -90 deg (bottom of the circle = start point) once around,
        #with a min-jerk profile so it starts and stops smoothly
        theta = self.min_jerk(-np.pi / 2, 3 * np.pi / 2, num_steps).reshape(-1)

        circle_positions = np.zeros((num_steps, 3))
        circle_positions[:, 0] = wall_frame_initial_position[0]              # same depth as the start
        circle_positions[:, 1] = circle_center[1] + radius * np.cos(theta)   # along the wall
        circle_positions[:, 2] = circle_center[2] + radius * np.sin(theta)   # up the wall

        #Back to the world frame
        positions = self.Wall_center_point + (self.Wall_rotation_matrix.T @ circle_positions.T).T

        forces = np.full(num_steps, self.push_force)
        contact_flags = np.ones(num_steps)

        return self.make_segment(positions, forces, contact_flags)

    def release(self, ref_trajectory, time_range, backoff_distance=0.3):
        #Ramps the force back to zero at the wall, then backs the pad away from the wall
        num_steps = self.steps_for(time_range)
        ramp_steps = max(num_steps // 3, 2)
        backoff_steps = max(num_steps - ramp_steps, 2)

        ref_pad_position_initial = ref_trajectory["pad_position"][-1]

        #Force ramp down, still in contact
        ramp_positions = np.tile(ref_pad_position_initial, (ramp_steps, 1))
        ramp_forces = self.min_jerk(self.push_force, 0.0, ramp_steps).reshape(-1)
        ramp_flags = np.ones(ramp_steps)

        #Back off along the wall normal, no contact
        backoff_target = ref_pad_position_initial + backoff_distance * self.Wall_normal_vector
        backoff_positions = self.min_jerk(ref_pad_position_initial, backoff_target, backoff_steps)
        backoff_forces = np.zeros(backoff_steps)
        backoff_flags = np.zeros(backoff_steps)

        positions = np.vstack([ramp_positions, backoff_positions])
        forces = np.concatenate([ramp_forces, backoff_forces])
        contact_flags = np.concatenate([ramp_flags, backoff_flags])

        return self.make_segment(positions, forces, contact_flags)


    # ------------------------------------------------------------------
    # Full trajectory
    # ------------------------------------------------------------------
    def generate_free_flight_trajectory(self, time_range=[2.0, 4.0]):
        self.initial_state = self.initilize_state(distance_range=[0.5, 1.0], yaw_offset_range=[0.0, 0.0])
    
        num_steps = self.steps_for(time_range)
        positions = np.array([self.initial_state[:3] for _ in range(num_steps)])
        forces = np.zeros(num_steps)
        contact_flags = np.zeros(num_steps)
        self.reference_time = np.arange(num_steps) * self.Ref_traj_dt
        trajectory_time_length = self.reference_time[-1]

        self.reference_trajectory = self.make_segment(positions, forces, contact_flags)
        return trajectory_time_length, self.reference_trajectory

    def generate_trajectory(self, wind_vector=None, Wall_yaw=None, Wall_center_point=None):

        

        #Initial state of the drone
        self.initial_state = self.initilize_state(distance_range=[0.5, 1.0], yaw_offset_range=[-np.radians(10), np.radians(10)])

        #Build the trajectory one segment at a time
        hold_traj    = self.hold_mode(time_range=[1.0, 2.0])
        move_traj    = self.move_to_wall(hold_traj, time_range=[3.0, 5.0])
        push_traj    = self.push_ramp(move_traj, time_range=[0.5, 1.0], force_range=[.2, .5])
        circle_traj  = self.draw_circle(push_traj, time_range=[6.0, 10.0], radius_range=[0.1, 0.2])
        release_traj = self.release(circle_traj, time_range=[1.5, 2.5])

        segments = [hold_traj, move_traj, push_traj, circle_traj, release_traj]
        self.reference_trajectory = {
            key: np.concatenate([segment[key] for segment in segments], axis=0)
            for key in hold_traj
        }

        #Time of every reference point, and the total length of the trajectory
        num_points = len(self.reference_trajectory["normal_force"])
        self.reference_time = np.arange(num_points) * self.Ref_traj_dt
        trajectory_time_length = self.reference_time[-1]

        return trajectory_time_length, self.reference_trajectory

    def generate_wall_and_wind(self, wind_vector=None, Wall_yaw=None, Wall_center_point=None):
        #One constant wind vector for the whole run (horizontal, random direction and speed)
        if wind_vector is None:
            wind_speed = np.random.uniform(self.Wind_min, self.Wind_max)
            wind_direction = np.random.uniform(0.0, 2 * np.pi)
            wind_vector = wind_speed * np.array([np.cos(wind_direction), np.sin(wind_direction), 0.0])
        self.wind_vector = np.asarray(wind_vector, dtype=float)

        if Wall_yaw is None:
            Wall_yaw = np.random.uniform(self.min_wall_yaw, self.max_wall_yaw)

        #Point on the wall surface (1.5 m up, like the MuJoCo generator)
        if Wall_center_point is None:
            Wall_center_point = np.array([0.0, 0.0, 1.5])
        self.Wall_center_point = np.asarray(Wall_center_point, dtype=float)

        #Outward wall normal (points from the wall into free space)
        self.Wall_normal_vector = np.array([-np.cos(Wall_yaw), -np.sin(Wall_yaw), 0.0])

        #Wall frame axes: out of the wall, horizontal along the wall, up the wall
        along_wall = np.cross(self.Wall_normal_vector, np.array([0.0, 0.0, 1.0]))
        along_wall = along_wall / np.linalg.norm(along_wall)
        up_wall = np.cross(along_wall, self.Wall_normal_vector)

        #From inertial to wall frame (rows are the wall axes)
        self.Wall_rotation_matrix = np.vstack([self.Wall_normal_vector, along_wall, up_wall])

        return self.wind_vector, self.Wall_normal_vector, self.Wall_center_point,
    # ------------------------------------------------------------------
    # Sampling the reference at other time steps
    # ------------------------------------------------------------------

    def sample(self, times):
        #Linearly interpolates the coarse reference at any list of times
        #Times past the end hold the last value
        sampled = {}
        for key, values in self.reference_trajectory.items():
            if values.ndim == 1:
                sampled[key] = np.interp(times, self.reference_time, values)
            else:
                columns = [np.interp(times, self.reference_time, values[:, i]) for i in range(values.shape[1])]
                sampled[key] = np.stack(columns, axis=1)

        #The contact flag is an on/off intent, so snap it back to 0 or 1
        sampled["contact_flag"] = np.round(sampled["contact_flag"])
        return sampled

    def interpolate(self, dt):
        #The whole reference resampled at a finer dt (useful for plotting)
        times = np.arange(0.0, self.reference_time[-1] + 1e-9, dt)
        return self.sample(times)

    def get_reference_window(self, current_time, MPC_dt, horizon_length):
        #The N+1 reference points the MPC needs: now, now + MPC_dt, ..., now + N * MPC_dt
        times = current_time + np.arange(horizon_length + 1) * MPC_dt
        return self.sample(times)