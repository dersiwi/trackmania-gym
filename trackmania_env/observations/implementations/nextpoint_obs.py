from __future__ import annotations
import traceback
from trackmania_env.observations.observation_manager import ObservationManager
from gymnasium import spaces
import numpy as np

from trackmania_env.observations.terms import NextReflinePoint, MobileStatesTerm, SurfaceFloats, SpeedTerm, ObservationTerm, LateralDistance, RelativeDistance
from trackmania_env.utils.contact_materials import physics_behavior_fromint,NUM_SURFACE_CATEGORIES

class NextPointObsManager(ObservationManager):
    def __init__(self, colorspace, convert_torch, img_width, img_height, normalize_obs):
        super().__init__(colorspace, convert_torch, img_width, img_height, normalize_obs=normalize_obs)
        self.reference_line_points_lookahead = 10
        self.refeence_line_stride = 10
        assert self.reference_line_points_lookahead % self.refeence_line_stride == 0

        self.observation_terms = [NextReflinePoint(self.reference_line_points_lookahead, self.refeence_line_stride, normalize_obs), 
                                  SpeedTerm(normalize_obs),
                                  MobileStatesTerm(normalize_obs),
                                  SurfaceFloats(normalize_obs),
                                  LateralDistance(normalize_obs),
                                  RelativeDistance(normalize_obs)]