import numpy as np
import mujoco

class Trajectory_Generator:
    def __init__(self, Wind_max_range, Trajectory_length, max_wall_tilt):
        self.Wind_max_range = Wind_max_range
        self.Trajectory_length = Trajectory_length
        self.max_wall_tilt = max_wall_tilt

    def generate_trajectory(self, wind_vector = None, trajectory_length=None, Wall_tilt=None, Wall_center_point=None):
        if trajectory_length is None:
            trajectory_length = self.Trajectory_length
        if wind_vector is None:
            wind_vector = np.random.uniform(-self.Wind_max_range, self.Wind_max_range, size=(trajectory_length, 3))
        if Wall_tilt is None:
            Wall_tilt = np.random.uniform(-self.max_wall_tilt, self.max_wall_tilt)
        if Wall_center_point is None:
            Wall_center_point = np.random.uniform(-1.0, 1.0, size=(trajectory_length, 3))  # Example range

        Wall_normal_vector = np.array([-np.cos(Wall_tilt), -np.sin(Wall_tilt), 0]).T
        
        return wind_vector, Wall_normal_vector, Wall_center_point

    def interpolate(self, dt):

        