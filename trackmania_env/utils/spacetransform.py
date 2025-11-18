import numpy as np

class SpaceTransformer:
    """This class transforms a python dictionary containing images and states into a single numpy-array. It provides both way transformations."""

    _instance = None

    @staticmethod
    def get_instance():
        assert not SpaceTransformer._instance is None
        return SpaceTransformer._instance
    
    @staticmethod
    def init_instance(obsterms):
        SpaceTransformer._instance = SpaceTransformer(obsterms)
        return SpaceTransformer._instance
    

    def __init__(self, obsterms):
        from trackmania_env.observations.observation_term import ObservationTerm
        self.obsterms : list[ObservationTerm] = obsterms
        self.expected_dim : int = sum([term.get_flatten_dim() for term in self.obsterms])
        """Size of the flattened observation per environment"""
        self.expect_vectorized_obs = False
        self.n_envs = 1 # only used when expect_vectorized is True

    def expect_vectorized(self, n_envs : int) -> None:
        """Call this method, if the environment returns vectorized-observations"""
        self.n_envs = n_envs
        self.expect_vectorized_obs = True



    def dict_to_numpy(self, dictobs : dict[str, np.ndarray]) -> np.ndarray:
        """This method transforms a dictionary into a numpy array"""
        observations = []
        for term in self.obsterms:
            observations.append(term.flatten(dictobs[term.name]))
        return np.hstack([observations])
    
    def dict_to_numpy_vectorized(self, dictobs : dict[str, np.ndarray]) -> np.ndarray:
        """These the obs in dictobs are unflattened obs, so this has to be called manually depending on if its vectorized or not."""
        observations = np.zeros((self.n_envs, self.expected_dim))
        for i, term in enumerate(self.obsterms):
            observations[:, term.get_flatten_dim()] = dictobs[term.name]
        return observations
    
    def numpy_to_dict(self, array : np.ndarray) -> dict[str, np.ndarray]:
        """Trurns the given array into a dictionary."""
        if len(array) == 2:
            return self.numpy_to_dict_vectorized(array)
        
        assert array.shape[0] == self.expected_dim, f"Got unexepcted array-shape. Expected {self.expected_dim}, got {array.shape[0]}"
        obsdict = {}
        startidx = 0
        for i, term in enumerate(self.obsterms):
            obsdict[term.name] = array[startidx : startidx + term.get_flatten_dim()].reshape(term.get_native_shape())
            startidx += term.get_flatten_dim()
        return obsdict
    
    def numpy_to_dict_vectorized(self, array : np.ndarray) -> dict[str, np.ndarray]:
        """Trurns the given array into a dictionary."""
        assert array.shape[0] == self.n_envs and array.shape[0] == self.expected_dim, f"Got unexepcted array-shape. Expected {self.expected_dim}, got {array.shape[0]}"
        obsdict = {}
        startidx = 0
        for i, term in enumerate(self.obsterms):
            obsdict[term.name] = array[:, startidx : startidx + term.get_flatten_dim()].reshape(tuple([self.n_envs] + [sdim for sdim in term.get_native_shape()]))
            startidx += term.get_flatten_dim()
        return obsdict