from __future__ import annotations
import traceback
from trackmania_env.observations.observation_manager import ObservationManager, ObservationDictEntry
from gymnasium import spaces
import numpy as np

from trackmania_env.observations.terms import NextReflinePoint, MobileStatesTerm, SurfaceFloats, SpeedTerm,  LateralDistance, RelativeDistance, ImageObservationTerm

class NextPointObsManager(ObservationManager):
    def __init__(self, colorspace, convert_torch, img_width, img_height, normalize_obs):
        super().__init__(colorspace, convert_torch, img_width, img_height, normalize_obs=normalize_obs)
        self.reference_line_points_lookahead = 10
        self.refeence_line_stride = 10
        assert self.reference_line_points_lookahead % self.refeence_line_stride == 0

        statevectorentry = ObservationDictEntry(name = "state", obstermlist=[NextReflinePoint(self.reference_line_points_lookahead, self.refeence_line_stride, normalize_obs), 
                                  SpeedTerm(normalize_obs),
                                  MobileStatesTerm(normalize_obs),
                                  SurfaceFloats(normalize_obs),
                                  LateralDistance(normalize_obs),
                                  RelativeDistance(normalize_obs)], normalize=normalize_obs, axis=0, dtype= np.float32)
        
        imageentry = ObservationDictEntry(name = "image", obstermlist=[ImageObservationTerm(img_width, img_height, 
                                                                                                         normalize=normalize_obs, 
                                                                                                         colorspace=colorspace, 
                                                                                                         cvr_grayscale_to_uint8=False)])

        self.observation_terms = [statevectorentry, imageentry]