import numpy as np

from trackmania_env.observations.observation_manager import DictObservationManager
from trackmania_env.observations.observation_term import GroupedObservationTerm

from trackmania_env.observations.observation_terms.img_terms import ImageObservationTerm
from trackmania_env.observations.observation_terms.basic_terms import (
    SpeedTerm,
    SurfaceFloats,
    MobileStatesTerm,
)
from trackmania_env.observations.observation_terms.ref_line_terms import (
    RelativeDistance,
    LateralDistance,
)

from trackmania_env.observations.observation_terms.sophy_terms import GlobalFeaturesTerm

class DynamicNextPointObsManager(DictObservationManager):
    def __init__(self,colorspace:str, convert_torch, img_width, img_height, normalize, lookahead_sec = 6,n_points = 60,**kwargs):
  
        grouped_floats_obs = GroupedObservationTerm(
            name="floats",
            observation_terms=[
               GlobalFeaturesTerm(lookahead_sec=lookahead_sec, n_points=n_points, name="Dynamic refline"),
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
