import numpy as np
from scipy.interpolate import CubicSpline

@staticmethod
def interpolate_points(n_points:int,points:np.ndarray):
    """
    Interpolates a sequence of 3D points to produce a uniform set of `n_points` sampled along the curve.

    This method computes the arc length of the polyline defined by `points`, then performs cubic spline
    interpolation over the x, y, and z coordinates independently. The result is a smooth path sampled at
    equally spaced arc lengths.

    Parameters:
        points (np.ndarray): Array of shape (N, 3) representing a sequence of 3D points (x, y, z) along a path.

    Returns:
        np.ndarray: Array of shape (self.n_points, 3) containing interpolated 3D points evenly spaced by arc length.
    """
    # length calculation in 3D
    diffs = np.diff(points, axis=0)
    lengths = np.concatenate([[0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    total_length = lengths[-1]

    # Create interpolation functions (x, y, z) over lengths
    fx = CubicSpline(lengths, points[:, 0])
    fy = CubicSpline(lengths, points[:, 1])
    fz = CubicSpline(lengths, points[:, 2])

    # Sample equidistant points along lengths
    uniform_s = np.linspace(0, total_length, n_points)
    x_sampled = fx(uniform_s)
    y_sampled = fy(uniform_s)
    z_sampled = fz(uniform_s)

    sampled_points = np.stack([x_sampled, y_sampled, z_sampled], axis=1)  # shape: (n_points, 3)

    return sampled_points