from typing import Callable

class LR_Scheduler:

    def __init__(self, initial_learning_rate : float):
        self.initial_value = initial_learning_rate

    def get_scheduler(self) -> Callable[[float], float]:
        raise NotImplementedError("Must override get_scheduler()")
    

class NoScheduler(LR_Scheduler):

    def __init__(self, initial_value):
        super().__init__(initial_value)

    def get_scheduler(self) -> Callable[[float], float]:
        """No Scheduling"""
        def schedule(progress_remaining: float) -> float:
            """:return: current learning rate"""
            return self.initial_value

        return schedule

class LinearScheduler(LR_Scheduler):

    def __init__(self, initial_value):
        super().__init__(initial_value)

    
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
    def __init__(self, initial_value: float= 2.5e-4, k: float = 3.0):
        """
        Exponential learning rate schedule.
        :param initial_value: Initial learning rate.
        :param k:  Exponential factor (higher = sharper decay near end).
        """
        super().__init__(initial_value)
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
    
class DropoffScheduler(LR_Scheduler):
    def __init__(self, initial_learning_rate = 2.5e-4, scheduling_beginn : float = 0.5, scheduling_type : str = "linear", propagate_progress : bool = False):
        """
        Combines linear or exponential scheduling with waiting period form [0, scheduling_beginn], durining which no scheduling happens.
        :param initial_learning_rate initial learning rate
        :param scheduling_beginn : progress-percentage at which scheduling starts
        :param scheduling_type : type of scheduling that starts at that percentage
        :param propagate_progress : If True, propagates the progress already done from [0, scheduling_beginn] to the selected schedular. If not, it suggests the scheduler, that the
        progress is at 0, although it is at scheduling_beginn (This means there is no sharp dropoff after initial phase)
        """
        super().__init__(initial_learning_rate)
        self.scheduling_beginn = scheduling_beginn
        self.scheduler : Callable = None
        if scheduling_type == "linear":
            self.scheduler = LinearScheduler(initial_learning_rate).get_scheduler()
        elif scheduling_type == "exponential":
            self.scheduler = ExponentialScheduler(initial_learning_rate).get_scheduler()
        else:
            raise ValueError(f"Scheduling-type '{scheduling_type}' not kown.")
        self.propagate_progress = propagate_progress
        

    def get_scheduler(self) -> Callable[[float], float]:
        """Returns a scheduler that does not change learning rate as long as n_steps < total_steps * scheduling_begin. Afterwards, it employs a 
        scheduler of selected type."""

        def schedule(progress_remaining : float) -> float:
            progress_remaining = 1 - progress_remaining # i actually want the progress to go from 0 (start) to 1 (end)
            
            if progress_remaining < self.scheduling_beginn:
                return self.initial_value
            else:
                progress_remaining = progress_remaining if self.propagate_progress else progress_remaining - self.scheduling_beginn
                return self.scheduler(1 - progress_remaining)
            
        return schedule

class TriangleScheduler(LR_Scheduler):
    """
    A learning rate scheduler that follows a triangular policy.

    The learning rate first increases linearly from 0 to its peak value,
    and then decreases linearly back down to 0 for the remainder of the training.
    The peak of the triangle is controlled by the `peak` parameter.
    """
     
    def __init__(self, initial_learning_rate, peak:float = 0.1):
        assert 0 <= peak <1 
        super().__init__(initial_learning_rate)
        self.peak = peak
    
    def get_scheduler(self) -> Callable[[float], float]:
        """
    
        """
        def schedule(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.
            """
            progress_remaining = 1 - progress_remaining #make it go from 0 to 1
            if progress_remaining <= self.peak:
                return progress_remaining * self.initial_value / self.peak 
    
            return (self.initial_value  / (1-self.peak) ) * (1 - progress_remaining)

        return schedule
    
    

if __name__ ==  "__main__":
    from matplotlib import pyplot as plt
    
    lr = DropoffScheduler(2.5*10e-4, 0.5, "linear",propagate_progress = True)
    f = lr.get_scheduler()
    r = []
    total_steps = 500
    for i in range(total_steps):
        r.append(f(1 - i / total_steps))
    
    plt.plot([i / total_steps for i in range(total_steps)], r)
    plt.show()