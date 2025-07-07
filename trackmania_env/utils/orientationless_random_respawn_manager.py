

import numpy as np

class OrientationlessRespawnManager:
    """
    This reset manager works after the car has been resettet by the game; aka is in starting positoin.
    Then it chooses one of the given respawn-coordinates to teleport it there."""

    @staticmethod
    def get_respawns_for_very_long_checkpoints() -> np.ndarray:
        """TODO : Maybe put this into a different file; just want to do this quickly."""
        return np.array([
            [16.79557991027832, 9.358166694641113, 208.5802001953125],
            [208.73594665527344, 9.359643936157227, 272.6419677734375],
            [15.442355155944824, 9.359561920166016, 369.2656555175781],
            [271.31085205078125, 17.358213424682617, 495.67547607421875],
            [268.7384338378906, 9.359115600585938, 530.5123901367188], #<-- this is not actually a checkpoint but a random corner where the orientation works.
            [363.2776794433594, 9.35879898071289, 191.91046142578125], #<-- same story here
            [241.68710327148438, 17.372529983520508, 344.4925231933594], #<-- and here
            [400.3624267578125, 9.366536140441895, 145.64828491210938],
            [589.807373046875, 9.358793258666992, 178.37193298339844], #<-- also no checkpoint
            [781.3646850585938, 9.36239242553711, 189.86605834960938], #y-
            [975.7445068359375, 33.35984802246094, 112.41080474853516],
            [974.290771484375, 33.35907745361328, 291.52935791015625],  #
            [469.7168273925781, 33.370933532714844, 64.35596466064453], #
            [462.1425476074219, 25.383047103881836, 176.57432556152344], #
            [462.974365234375, 25.35723114013672, 283.20361328125],#
            [591.8837890625, 17.362228393554688, 435.260986328125],
            [590.5247192382812, 9.367033004760742, 647.839599609375],
            [682.9180908203125, 9.361037254333496, 446.05889892578125],#
            [687.2301635742188, 9.356075286865234, 689.5628662109375],
            [594.23974609375, 9.359204292297363, 783.1381225585938],#
            [399.893310546875, 9.364355087280273, 796.7860107421875],#
            [207.6233367919922, 9.3544340133667, 849.00048828125],
            [493.32989501953125, 9.359715461730957, 934.6239013671875],#
            [719.5479736328125, 17.391916275024414, 914.7745971679688],
            [911.7840576171875, 33.36442565917969, 850.7297973632812],
            [912.3789672851562, 9.358830451965332, 785.5594482421875],
            [15.999698638916016, 10.015329360961914, 9.845817565917969]
        ])

    def __init__(self, respawn_coordinates : np.ndarray, xoffset : tuple[float, float] = (0., 0.), yoffset : tuple[float, float] = (0., 0.)):
        """Parametesr
            - respawn_coordainates [N, 3] are positions at which the car can be teleported after restart.
            - x/y-offset : tuples of (min_offset, max_offset) from which an offset to the starting position is sampled and added on."""
        self.respawn_coordinates = respawn_coordinates
        print(self.respawn_coordinates.shape)
        self.xoffset = xoffset
        self.yoffset = yoffset



    def get_respawn_coordinates(self) -> tuple[np.ndarray | tuple[float, float, float], bool]:
        """Return a tuple of position in world-coordinate frame and a boolean. 
        If True, it tells the environment to teleport the car there, if false, it tells the enivronment to do nothing."""
        position_idx = np.random.randint(self.respawn_coordinates.shape[0] + 1)
        if position_idx == self.respawn_coordinates.shape[0]:
            return None, False
        
        coordinates = self.respawn_coordinates[position_idx]
        coordinates[0] += np.random.uniform(low = self.xoffset[0], high = self.xoffset[1])
        coordinates[1] += 0.05 # add this to z-coordinate because we dont want to spawn car inside road
        coordinates[2] += np.random.uniform(low = self.yoffset[0], high = self.yoffset[1])
        return coordinates, True

if __name__ == "__main__":
    orm = OrientationlessRespawnManager(np.random.randint(low = 34, size = (10, 3)))
    for i in range(100):
        p, tp = orm.get_respawn_coordinates()
        print(p, tp)