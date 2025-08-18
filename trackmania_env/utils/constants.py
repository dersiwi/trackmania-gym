MS_TO_KMH = 3.6       # meters per second to kilometers per hour
MILLISECONDS_TO_SECONDS = 1000  # milliseconds to seconds
MAX_DISTANCE_TO_REFLINE = 15 # the distance after which the car is considered to be off-course
MAX_HEIGHT_DIFERENCE = 7

class NormalizationFactors:
    """Contains normalization factors for different observations.

    Usage:
    ------
        Say you want to normalize observation A, then normalized A is given by:
            A_{normalized} = A / NormlalizationFactors.A_norm

    Note
    ----
    These factors are basically all eyeballed, so its possible that not every value is strictly between [0,1] after 
    normalization.
    """
    speed_norm = 1000.0
    refline_norm = 500.0
    """normalizes reference line points"""
    gearbox_norm = 5.0
    rpm_norm = 12000.0
    lateral_dist_norm = 18
    gear_norm = 2 
    """this is basically always 1 or 0 except for rare occasions where its 2. Could go higher tho."""