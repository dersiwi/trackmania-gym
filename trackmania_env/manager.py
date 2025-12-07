

class ManagerTerm:
    """Baseclass for all Terms being used for Term-Based Managers"""

    def __init__(self, name : str):
        self.name = name
        """Name of the Manager-Term"""

    def set_env(self, env) -> None:
        """Sets the environment for the term."""
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = env

    def reset(self):
        """Resets the term."""
        pass

class Manager:
    """Baseclass for all term-based managers. Term-Based Managers posess terms, that calculate whatever the manager is meant for, i.e.
    Observations, Rewards, Terminations. The calculation-method is not included, as the managers should implement this themselves."""

    def __init__(self):
        self.terms : list[ManagerTerm] = []
        """Term-List for this Manager"""
        self.env = None
        """Environment-Variable of the manager"""

    def set_env(self, env) -> None:
        """Sets the environment for the manager and also propagates the environment to each term."""
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = env
        for term in self.terms:
            term.set_env(env)

    def reset(self):
        """Resets the manager and also each enviroment term."""
        for term in self.terms:
            term.reset()