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

# TODO in linesight the variable corresponds to config_copy.n_contact_material_physics_behavior_types
"""
TODO in linesight the variable corresponds to config_copy.n_contact_material_physics_behavior_types
which is set to 4. But there are actually 5 categories. Did they do this on purpose to ignore the 
"anything else" surfaces ?
"""
NUM_SURFACE_CATEGORIES = 4
physics_group_fromstr = {
    "Concrete": 0,
    "Pavement": 0,
    "Grass": 1,
    "Ice": 4,
    "Metal": 4,
    "Sand": 2,
    "Dirt": 2,
    "Turbo": 3,
    "DirtRoad": 2,
    "Rubber": 4,
    "SlidingRubber": 4,
    "Test": 4,
    "Rock": 4,
    "Water": 4,
    "Wood": 4,
    "Danger": 4,
    "Asphalt": 0,
    "WetDirtRoad": 2,
    "WetAsphalt": 0,
    "WetPavement": 0,
    "WetGrass": 1,
    "Snow": 4,
    "ResonantMetal": 4,
    "GolfBall": 4,
    "GolfWall": 4,
    "GolfGround": 4,
    "Turbo2": 3,
    "Bumper": 4,
    "NotCollidable": 4,
    "FreeWheeling": 4,
    "TurboRoulette": 4,
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