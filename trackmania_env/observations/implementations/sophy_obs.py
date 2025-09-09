import numpy as np
from collections import deque
from gymnasium import spaces
import torch
from scipy.interpolate import CubicSpline
from trackmania_env.observations.observation_manager import ObservationManager, ObservationDictEntry
from trackmania_env.utils import constants
from game_interaction.ipc_fields import IPCFields
from tminterface.structs import SimStateData, HmsDynaStateStruct
from trackmania_env.observations.terms import SophyGlobalFeatures, Propriocentric_features, ImageObservationTerm

IMAGE_SIZE = 64
"""Image size as specified in sophy paper."""

class SophyObsManager(ObservationManager):
    def __init__(self, colorspace, convert_torch, img_width, img_height,maxlen_history:int = 3,lookahead_sec = 6,n_points = 60):
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
        super().__init__(colorspace, convert_torch, img_width, img_height)

        self.global_featues = ObservationDictEntry(name = "global_features",obstermlist= [SophyGlobalFeatures(lookahead_sec, n_points, normalize=False)])
        self.propfeatures = ObservationDictEntry(name = "propriocentric_features",obstermlist= [Propriocentric_features(normalize=False, maxlen_history = maxlen_history, lookahead_sec=lookahead_sec, n_points=n_points)]) # TODO fix shape n term in general
        self.imgs = ObservationDictEntry(name = "image", obstermlist=[ImageObservationTerm(img_width, img_height, 
                                                                                                         normalize=False, 
                                                                                                         colorspace=ImageObservationTerm.Colorspace.RGB, 
                                                                                                         cvr_grayscale_to_uint8=False)])
        self.observation_terms = [self.imgs, self.global_featues, self.propfeatures]