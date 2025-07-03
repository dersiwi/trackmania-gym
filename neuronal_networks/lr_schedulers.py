from typing import Callable

class LR_Scheduler:

    def get_scheduler(self) -> Callable[[float], float]:
        raise NotImplementedError("Must override get_scheduler()")

class LinearScheduler(LR_Scheduler):
    def __init__(self,initial_value: float=2.5e-4):
        """
        Linear learning rate schedule.
        :param initial_value: Initial learning rate.
        """
        self.initial_value = initial_value
    
    def get_scheduler(self) -> Callable[[float], float]:
        """
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def schedule(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * self.initial_value

        return schedule

class ExponentialScheduler(LR_Scheduler):
    def __init__(self,initial_value: float= 2.5e-4, k: float = 3.0):
        """
        Exponential learning rate schedule.
        :param initial_value: Initial learning rate.
        :param k:  Exponential factor (higher = sharper decay near end).
        """
        self.initial_value = initial_value
        self.k = k

    def get_scheduler(self) -> Callable[[float], float]:
        """
        Exponential learning rate schedule.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def schedule(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            The learning rate is scaled by progress_remaining raised to power k:
            - At start (progress = 1.0): lr = initial_value
            - At end (progress = 0.0): lr = 0

            :param progress_remaining: Remaining training progress (from 1 to 0)
            :return: current learning rate
            """
            return self.initial_value * (progress_remaining ** self.k)

        return schedule
    

if __name__ ==  "__main__":
    from matplotlib import pyplot as plt
    
    lr = ExponentialScheduler(2.5*10e-4,4)
    f = lr.get_scheduler()
    r = []
    start = 500
    for i in range(start,0,-1):
        r.append(f(i/start))
    
    plt.plot(r)
    plt.show()