from trackmania_env.observations.implementations.nextpoint_obs import NextPointObsManager
from trackmania_env.observations.terms import SophyGlobalFeatures

class DynamicNextPointObsManager(NextPointObsManager):

    def __init__(self, colorspace, convert_torch, img_width, img_height, normalize_obs,lookahead_sec = 6,n_points = 60):
        super().__init__(colorspace, convert_torch, img_width, img_height, normalize_obs)

        #leave everything as is, just instead of fixed Reference line points use dynamic referencelinepoints, based on speed.
        self.observation_terms[0] = SophyGlobalFeatures(lookahead_sec=6, n_points=n_points, normalize=normalize_obs)