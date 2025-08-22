from __future__ import annotations
import numpy as np
from scipy.stats import norm
from trackmania_env.utils.constants import MAX_LATERAL_DISTANCE

class LateralDistanceManager:

    @staticmethod
    def get_instance(distance_type : str): # TODO : parameterize.
        if distance_type == "triangle":
            return TriangleLateralDistance()
            
        elif distance_type == "gauss":
            return GaussianLateralDistance()

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
    

    
class TriangleLateralDistance(LateralDistanceManager):
    def __init__(self):
        super().__init__()
    
    def _specific_ldist_scale(clipped_dist : float) -> float:
        # inverse the distance, such that reward is bigger once distance gets smaller
        return 0.5 - clipped_dist / MAX_LATERAL_DISTANCE
    

    
class GaussianLateralDistance(LateralDistanceManager):
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