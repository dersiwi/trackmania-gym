import numpy as np

from trackmania_env.observations.observation_manager import DictObservationManager, BoxObservationManager
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

class NextPointObsManager(DictObservationManager):
    def __init__(self,colorspace:str, convert_torch, img_width, img_height, normalize,ref_line_lookahead:int = 10, ref_line_stride:int = 10 ):
        assert ref_line_lookahead % ref_line_stride == 0

        grouped_floats_obs = GroupedObservationTerm(
            name="floats",
            observation_terms=[
                NextReflinePoint(ref_line_lookahead, ref_line_stride),
                SpeedTerm(),
                MobileStatesTerm(),
                SurfaceFloats(),
                LateralDistance(),
                RelativeDistance(),
            ]
        )

        super().__init__(
            convert_torch=convert_torch,
            normalize=normalize,
            observation_terms=[
                ImageObservationTerm(name="image", colorspace=colorspace, img_width=img_width, img_height=img_height, dtype=np.float32),
                grouped_floats_obs
            ]
        )

class BoxNextPointObsManager(BoxObservationManager):
    """ This is the Box version of the NextPointObsManager"""
    def __init__(self,colorspace:str, convert_torch, img_width, img_height, normalize,ref_line_lookahead:int = 10, ref_line_stride:int = 10 ):
        assert ref_line_lookahead % ref_line_stride == 0

        grouped_floats_obs = GroupedObservationTerm(
            name="floats",
            observation_terms=[
                NextReflinePoint(ref_line_lookahead, ref_line_stride),
                SpeedTerm(),
                MobileStatesTerm(),
                SurfaceFloats(),
                LateralDistance(),
                RelativeDistance(),
            ]
        )

        super().__init__(
            convert_torch=convert_torch,
            normalize=normalize,
            observation_term= grouped_floats_obs
        )

