from __future__ import annotations
import numpy as np
from scipy.stats import norm
import sys, os
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!
from trackmania_gym.trackmania_env.utils.constants import MAX_LATERAL_DISTANCE

class LateralDistanceManager:

    @staticmethod
    def get_instance(distance_type : str): # TODO : parameterize.
        """Returns an instance of the LateralDistanceManager.
        Args: 
            distance_type (str) : Specifies what type of latearl distance reward normalizer"""
        if distance_type == TriangleLateralDistance.NAME:
            return TriangleLateralDistance()
            
        elif distance_type == GaussianLateralDistance.NAME:
            return GaussianLateralDistance()
        elif distance_type == GaussianEdgePunisher.NAME:
            return GaussianEdgePunisher()

        elif distance_type == "trapez":
            raise NotImplementedError("Not implemented.")

    def __init__(self):
        pass

    def clip_lateral_distance(self, absolute_dist : float) -> float:
        """Clipts lateral distance into the interval [0, MAX_LATERAL_DISTANCE]. Because the road is symmetric (pointwise it can be mirrored at the center-line; from where the lateral distance is measured.)"""
        return np.clip(absolute_dist, a_min = 0, a_max = MAX_LATERAL_DISTANCE)


    def scale_lateral_distance(self, absolute_dist : float) -> float:
        """This method scales the lateral distance into the interval [-1, 1] and assigns each lateral distance a value."""
        clipped_dist = self.clip_lateral_distance(absolute_dist)
        return self._specific_ldist_scale(clipped_dist)

    def _specific_ldist_scale(self, absolute_dist : float) -> float:
        raise NotImplementedError("Not Implemented, Use one of the subclasses via get_instance()")
    
    def draw(self):
        from matplotlib import pyplot as plt
        xs = np.linspace(0, MAX_LATERAL_DISTANCE, 200)
        ys = [self._specific_ldist_scale(x) for x in xs]

        plt.figure(figsize=(6,4))
        plt.plot(xs, ys, label="Reward / Distance from Road Center")
        plt.axhline(0, color="gray", linestyle=":")
        plt.title("Reward / Distance from Road Center")
        plt.xlabel("Distance from road center")
        plt.ylabel("Reward")
        plt.axvline(MAX_LATERAL_DISTANCE, linestyle=":")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        

    
class TriangleLateralDistance(LateralDistanceManager):
    NAME = "triangle"
    def __init__(self):
        super().__init__()
    
    def _specific_ldist_scale(clipped_dist : float) -> float:
        # inverse the distance, such that reward is bigger once distance gets smaller
        return 0.5 - clipped_dist / MAX_LATERAL_DISTANCE
    

    
class GaussianLateralDistance(LateralDistanceManager):
    """Scales reward via a gaussian curve, where the center line is the maximum, edges are 'minima'."""
    NAME = "gauss"
    def __init__(self, mean :float = 0, sigma : float = 1, multiplicator : float = 5, yshift : float = -1, dist_scale : float = 0.3):
        """
        Args:
            mean (float)  : Mean of the Gaussian function used for distance-to-centerline reward. (Only active if literal_distance_mode = "Gauss")
                         Represents the ideal centerline offset (typically 0).
            sigma (float) : Standard deviation of the Gaussian, controlling the reward falloff 
                           as the vehicle deviates from the centerline. (Only active if literal_distance_mode = "Gauss")

            yshift (float)        : Vertical shift applied to the Gaussian curve to shape the baseline reward.
            multiplicator (float) : Scales the Gaussian's amplitude; higher values amplify the reward.
            dist_scale (float)    : Scaling factor for the input distance before applying the Gaussian.
        """
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.multiplicator = multiplicator
        self.yshift = yshift
        self.dist_scale = dist_scale

    def _specific_ldist_scale(self, clipped_dist : float) -> float:
        return self.multiplicator * norm.pdf(clipped_dist * self.dist_scale, loc =self.mean, scale = self.sigma) + self.yshift
    

class GaussianEdgePunisher(GaussianLateralDistance):
    NAME = "gauss_edge"
    def __init__(self, mean = MAX_LATERAL_DISTANCE, sigma = 1, multiplicator = -2.5, yshift = 0, dist_scale = 1):
        super().__init__(mean, sigma, multiplicator, yshift, dist_scale)


if __name__ == "__main__":

    LateralDistanceManager.get_instance(GaussianEdgePunisher.NAME).draw()