import math

MS_TO_KMH = 3.6       # meters per second to kilometers per hour
MILLISECONDS_TO_SECONDS = 1000  # milliseconds to seconds
MAX_DISTANCE_TO_REFLINE = 15 # the distance after which the car is considered to be off-course
MAX_HEIGHT_DIFERENCE = 7
TYPING = False
MAX_LATERAL_DISTANCE = 12 # maximal lateral difference of the car to the road-center, this is an estimate.

MAX_SPEED = 1000.0

EPSILON = 1e-6
MAX_DEGREE_RADIANS = math.pi / 6

class ObsNormalizationFactors:

    speed_norm = 1000.0
    refline_norm = 500.0
    gearbox_norm = 5.0
    rpm_norm = 12000.0
    lateraltrack_dist_norm = 18
    gear_norm = 2
