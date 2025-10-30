import numpy as np

from trackmania_env.observations.observation_manager import DictObservationManager
from trackmania_env.observations.observation_terms.img_terms import ImageObservationTerm
from trackmania_env.observations.observation_terms.sophy_terms import PropriocentricTerm,GlobalFeaturesTerm

IMAGE_SIZE = 64
"""Image size as specified in sophy paper."""

class SophyObsManager(DictObservationManager):
    def __init__(self, colorspace, convert_torch, normalize, img_width, img_height,maxlen_history:int = 3,lookahead_sec = 6,n_points = 60, img_dtype=np.float32):
        """
        Initializes the GT Sophy-style observation manager (https://arxiv.org/pdf/2406.12563v1).

        This setup is designed to match the inputs of the GT Sophy AI racing system, including
        fixed-size square image inputs and a specific prediction horizon. It prepares the agent to consume 
        perception inputs (e.g., vision and track state), maintain a short history for temporal awareness, 
        and output a trajectory of future positions.

        Parameters:
            - colorspace (str)                  : Color space of input images, typically "rgb" or "grayscale".
            - convert_torch (bool)              : Whether to convert inputs into PyTorch tensors for model compatibility.
            - img_width (int)                   : Width of input images.
            - img_height (int)                  : Height of input images. 
            - maxlen_history (int, default=3)   : Number of previous time steps to include for temporal context (e.g., past states or actions).
            - lookahead_sec (int, default=6)    : Time horizon in seconds over which the agent predicts the incomming reference line points.
            - n_points (int, default=60)        : Number of points to generate from the distance the agent traveled in lookahead_sec.
        """
        assert img_width == img_height == IMAGE_SIZE, (
            f"Sophy was trained on {IMAGE_SIZE}x{IMAGE_SIZE} images. "
            "Please use square images of this size."
        )

        super().__init__(
            observation_terms= [
                ImageObservationTerm(colorspace= colorspace, img_height=img_height, img_width= img_width, dtype=img_dtype),
                PropriocentricTerm(maxlen_history=maxlen_history),
                GlobalFeaturesTerm(lookahead_sec=lookahead_sec,n_points=n_points)
            ],
            normalize= normalize,
            convert_torch= convert_torch,
        )