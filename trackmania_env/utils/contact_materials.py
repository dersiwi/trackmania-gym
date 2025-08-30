"""
List of contact materials defined in the Trackmania game engine.
"""

from enum import Enum

# from Donadigo on TMInterface Discord


# Access via ContactMaterial(3).name       -->  Ice
# Access via ContactMaterial['Ice'].value  --> 3
class ContactMaterial(Enum):
    Concrete = 0
    Pavement = 1
    Grass = 2
    Ice = 3
    Metal = 4
    Sand = 5
    Dirt = 6
    Turbo = 7
    DirtRoad = 8
    Rubber = 9
    SlidingRubber = 10
    Test = 11
    Rock = 12
    Water = 13
    Wood = 14
    Danger = 15
    Asphalt = 16
    WetDirtRoad = 17
    WetAsphalt = 18
    WetPavement = 19
    WetGrass = 20
    Snow = 21
    ResonantMetal = 22
    GolfBall = 23
    GolfWall = 24
    GolfGround = 25
    Turbo2 = 26
    Bumper = 27
    NotCollidable = 28
    FreeWheeling = 29
    TurboRoulette = 30


# Group surfaces together such that:
#   0 represents broadly "Asphalt category"
#   1 represents broadly "Grass category"
#   2 represents broadly "Dirt category"
#   3 represents broadly "Turbo category"
#   4 represents broadly "anything else"

class SurfaceCategory(Enum):
    """Broad surface categories for driving physics or logic grouping."""
    ASPHALT = 0 
    GRASS = 1    
    DIRT = 2     
    TURBO = 3    
    OTHER = 4    

# TODO in linesight the variable corresponds to config_copy.n_contact_material_physics_behavior_types
"""
TODO in linesight the variable corresponds to config_copy.n_contact_material_physics_behavior_types
which is set to 4. But there are actually 5 categories. Did they do this on purpose to ignore the 
"anything else" surfaces ?
"""
NUM_SURFACE_CATEGORIES = 4
physics_group_fromstr = {
    "Concrete": SurfaceCategory.ASPHALT,
    "Pavement": SurfaceCategory.ASPHALT,
    "Asphalt": SurfaceCategory.ASPHALT,
    "WetAsphalt": SurfaceCategory.ASPHALT,
    "WetPavement": SurfaceCategory.ASPHALT,

    "Grass": SurfaceCategory.GRASS,
    "WetGrass": SurfaceCategory.GRASS,

    "Sand": SurfaceCategory.DIRT,
    "Dirt": SurfaceCategory.DIRT,
    "DirtRoad": SurfaceCategory.DIRT,
    "WetDirtRoad": SurfaceCategory.DIRT,

    "Turbo": SurfaceCategory.TURBO,
    "Turbo2": SurfaceCategory.TURBO,

    # Everything else falls under "OTHER"
    "Ice": SurfaceCategory.OTHER,
    "Metal": SurfaceCategory.OTHER,
    "Rubber": SurfaceCategory.OTHER,
    "SlidingRubber": SurfaceCategory.OTHER,
    "Test": SurfaceCategory.OTHER,
    "Rock": SurfaceCategory.OTHER,
    "Water": SurfaceCategory.OTHER,
    "Wood": SurfaceCategory.OTHER,
    "Danger": SurfaceCategory.OTHER,
    "Snow": SurfaceCategory.OTHER,
    "ResonantMetal": SurfaceCategory.OTHER,
    "GolfBall": SurfaceCategory.OTHER,
    "GolfWall": SurfaceCategory.OTHER,
    "GolfGround": SurfaceCategory.OTHER,
    "Bumper": SurfaceCategory.OTHER,
    "NotCollidable": SurfaceCategory.OTHER,
    "FreeWheeling": SurfaceCategory.OTHER,
    "TurboRoulette": SurfaceCategory.OTHER,
}

# creates the mapping from enum value to  its physics behavior group
"""
creates the mapping from enum value to  its physics behavior group
physics_behavior_fromint = {
    0: 0,  # Concrete
    1: 0,  # Pavement
    2: 1,  # Grass
    3: 4,  # Ice
    ...
}
later the wheel fields from the simstatedata object will have a field called contact_material_id
which is an integer refering to the contact material enum.
"""
physics_behavior_fromint = {
    ContactMaterial[material_string].value: physics_group for material_string, physics_group in physics_group_fromstr.items()
}