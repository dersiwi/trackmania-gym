from trackmania_env.utils.reference_line_manager import ReferenceLineManager

class SophyReflineManager(ReferenceLineManager):
    """
      Course points describe the geometry of the track via the left and right boundaries and the
        center line. At each time step, this method computes the 3D relative positions of course 
        points ahead of the agent, based on its current velocity. These points are sampled from 
        0.1 to 6.0 seconds into the future, at 0.1-second intervals, assuming constant forward speed.
    
        The distance to each course point is dynamically computed using the agent's current speed 
        (i.e., distance = velocity x time). This results in 59 course points per line (left, center, right), 
        giving a predictive spatial representation of the upcoming track segment.
    """
    def __init__(self, filepath, lookahead_size = 120, search_recursively = True, recursive_lookahead_increase_factor = 3, max_recursion_depth = 1):
        super().__init__(filepath, lookahead_size, search_recursively, recursive_lookahead_increase_factor, max_recursion_depth)
        