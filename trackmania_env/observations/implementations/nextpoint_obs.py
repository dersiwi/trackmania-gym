import numpy as np

from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.observations.observation_term import GroupedObservationTerm

from trackmania_env.observations.observation_terms.img_terms import ImageObservationTerm
from trackmania_env.observations.observation_terms.basic_terms import (
    SpeedTerm,
    SurfaceFloats,
    MobileStatesTerm,
)
from trackmania_env.observations.observation_terms.ref_line_terms import (
    NextReflinePoint,
    RelativeDistance,
    LateralDistance,
)

def make_float_obs(name:str, ref_line_lookahead:int, ref_line_stride:int) -> GroupedObservationTerm:
    """Helper method coombining Multiple Float-Like observations    
        - NextReflinePoint
        - SpeedTerm
        - MobileStatesTerm
        - LateralDistance
        - RelativeDistance
    """

    return GroupedObservationTerm(
            name= name,
            observation_terms=[
                NextReflinePoint(ref_line_lookahead, ref_line_stride),
                SpeedTerm(),
                MobileStatesTerm(),
                SurfaceFloats(),
                LateralDistance(),
                RelativeDistance(),
            ]
        )

class NextPointObsManager(ObservationManager):
    def __init__(self,colorspace:str, convert_torch, img_width, img_height, normalize,ref_line_lookahead:int = 10, ref_line_stride:int = 10,**kwargs):
        super().__init__(
            convert_torch=convert_torch,
            normalize=normalize,
            observation_terms=[
                ImageObservationTerm(name="image", colorspace=colorspace, img_width=img_width, img_height=img_height, dtype=np.float32),
                make_float_obs(name= "flaots", ref_line_stride= ref_line_stride, ref_line_lookahead= ref_line_lookahead)
            ]
        )

class VisionLessNextPointObsManager(ObservationManager):
    """ This is the Box version of the NextPointObsManager"""
    def __init__(self, convert_torch, normalize, ref_line_lookahead:int = 10, ref_line_stride:int = 10,**kwargs):
        super().__init__(
            convert_torch=convert_torch,
            normalize=normalize,
            observation_term = make_float_obs(name= "flaots", ref_line_stride= ref_line_stride, ref_line_lookahead= ref_line_lookahead)
        )

