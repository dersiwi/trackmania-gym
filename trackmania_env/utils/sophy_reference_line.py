from trackmania_env.utils.reference_line_manager import ReferenceLineManager
import numpy as np
from scipy.interpolate import CubicSpline


class SophyReflineManager(ReferenceLineManager):
    """
        Course points describe the geometry of the track via the left and right boundaries and the
        center line. At each time step, this method computes the 3D relative positions of course 
        points ahead of the agent, based on its current velocity. These points are sampled from 
        0.1 to 6.0 seconds into the future, at 0.1-second intervals, assuming constant forward speed.
    
        The distance to each course point is dynamically computed using the agent's current speed 
        (i.e., distance = velocity x time). This results in 59 course points per line (left, center, right), 
        giving a predictive spatial representation of the upcoming track segment.
    """
    def __init__(self, filepath, lookahead_size = 120, search_recursively = True, recursive_lookahead_increase_factor = 3, max_recursion_depth = 1,lookahead_sec = 6,n_points = 60):
        super().__init__(filepath, lookahead_size, search_recursively, recursive_lookahead_increase_factor, max_recursion_depth)
        self.lookahead_sec = lookahead_sec
        self.n_points = n_points
        self.mean_ref_point_distance =  np.mean(np.linalg.norm(self.reference_line[1:]-self.refline.reference_line[:-1],axis=1),axis=0)

    def get_reference_line_points(self, begin_idx ,speed:int ,extrapolate = False):
        # we already assume that speed is in m/s
        cp_passed = (self.lookahead_sec * speed) / self.mean_ref_point_distance 
        new_end_idx = begin_idx + cp_passed
        points =  super().get_reference_line_points(begin_idx, new_end_idx, extrapolate, 1)
        interp_points = self.interpolate_points(points)
        return interp_points
    
    def interpolate_points(self,points:np.ndarray):
        # length calculation in 3D
        diffs = np.diff(points, axis=0)
        lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs, axis=1))])
        total_length = lengths[-1]

        # Create interpolation functions (x, y, z) over lengths
        fx = CubicSpline(lengths, points[:, 0])
        fy = CubicSpline(lengths, points[:, 1])
        fz = CubicSpline(lengths, points[:, 2])

        # Sample equidistant points along lengths
        uniform_s = np.linspace(0, total_length, self.n_points)
        x_sampled = fx(uniform_s)
        y_sampled = fy(uniform_s)
        z_sampled = fz(uniform_s)

        sampled_points = np.stack([x_sampled, y_sampled, z_sampled], axis=1)  # shape: (n_points, 3)

        return sampled_points