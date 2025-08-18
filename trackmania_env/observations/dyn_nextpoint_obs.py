from trackmania_env.observations.nextpoint_obs import NextPointObsManager,OSV
from trackmania_env.observations.sophy_obs import SophyObsManager,IMAGE_SIZE

class DynamicNextPointObsManager(NextPointObsManager):

    def __init__(self, colorspace, convert_torch, img_width, img_height, normalize_obs,lookahead_sec = 6,n_points = 60):
        super().__init__(colorspace, convert_torch, img_width, img_height, normalize_obs)

        # we will only use the sophy obs for calculating the comming refline points
        self.sophy_obsmanager = SophyObsManager(colorspace=colorspace,convert_torch=convert_torch,img_height=IMAGE_SIZE,img_width=IMAGE_SIZE,lookahead_sec=lookahead_sec,n_points=n_points)
        self.reference_line_points_lookahead = n_points
        # reinit. OSV because reference_line_points_lookahead has changed
        self.idxs = OSV(self.reference_line_points_lookahead) 
        
    def get_next_refline_points(self, next_refline_idx, car_position, car_orientation, obs = None):
        if self.sophy_obsmanager.env == None : self.sophy_obsmanager.set_env(self.env)
        return self.sophy_obsmanager.get_global_features(game_states=obs)
    