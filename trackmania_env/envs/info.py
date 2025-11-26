

class EnvironmentInfo:
    """This class contains keys that can be used to accesss information from the environment."""
    
    REWARDS = "rewards"
    """Contains all reward-informations (these are also stored in dictionaries). This dictionary may contain 'total'."""

    POSITION = "position"
    """Contains the position of the car at the current environment step"""

    ORIENTATION = "orientation"
    """Contains the orientation of the vehiecle in global coordinate system"""
    VELOCITY = "velocity"
    DISPLAY_SPEED = "display_speed"
    GAS = "gas"
    LAST_HAS_ANY_LATERAL_CONTACT_TIME = "last_has_any_lateral_contact_time"
    
    ROTATION_MATRIX = "rotation_matrix"
    """ssd rotation matrix"""
    DYNA_ROTATION = "dyna_rotation"
    """rotation from ssd.dyna.current_state.rotation"""

    NEXT_REFLINE_IDX = "next_refline_index"
    """Contains the next reference line index"""

    COMING_REFLINE_POINTS = "comming_refline_points"
    """Contains the 'next' couple of reference line points. This depends on the configuration."""
