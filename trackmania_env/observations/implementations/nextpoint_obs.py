from __future__ import annotations
import traceback
from trackmania_env.observations.observation_manager import ObservationManager,DictObservationManager
from trackmania_env.observations.observation_term import Obs_Float_Stacker
from trackmania_env.observations.observation_terms.img_terms import ImageObservationTerm
from gymnasium import spaces
import numpy as np

from trackmania_env.observations.observation_terms.basic_terms import SpeedTerm, SurfaceFloats, MobileStatesTerm
from trackmania_env.observations.observation_terms.ref_line_terms import NextReflinePoint,RelativeDistance,LateralDistance

class NextPointObsManager2(ObservationManager):
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

class NextPointObsManager(DictObservationManager):
    def __init__(self,colorspace:str, convert_torch, img_width, img_height, normalize,ref_line_lookahead:int = 10, ref_line_stride:int = 10 ):
        assert ref_line_lookahead % ref_line_stride == 0
        super().__init__(
            [
                ImageObservationTerm(name="image",colorspace=colorspace, img_width=img_width, img_height=img_height,dtype=np.uint8),
                Obs_Float_Stacker(
                    name= "floats",
                    observation_terms= [
                        NextReflinePoint(ref_line_lookahead,ref_line_stride),
                        SpeedTerm(),
                        MobileStatesTerm(),
                        SurfaceFloats(),
                        LateralDistance(),
                        RelativeDistance(),
                    ]
                )
            ], convert_torch, normalize)