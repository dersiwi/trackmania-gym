
import numpy as np

class PositionBuffer:
    """Implements a ring buffer that stores (x,y,z)-Positions. The buffer is implemented as a numpy array of shape (N, 3) - where N is the size of the buffer.
    The main purpose of the buffer is the moved_more_than_threshold() method, which indicates if the positions given are moving.
    """
    def __init__(self, size : int):
        self.buffer : np.ndarray = np.zeros((size, 3), dtype=float)
        self.index = 0
        """Next index in the buffer to be filled"""
        self.count = 0
        """Count indicates the portion of the buffer that is filled: count € [0, size]"""
        self.size = size
        """Sizes of the buffer"""

    def add(self, position : np.ndarray) -> None:
        """Adds a position to the buffer. Overwrites N-oldest position if buffer is full."""
        self.buffer[self.index] = position
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def get_ordered(self) -> np.ndarray:
        """Return positions in the correct time order in which they were added to the buffer"""
        if self.count < self.size:
            return self.buffer[:self.count]
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]), axis=0)


    def moved_more_than_threshold(self, movement_threshold : float) -> bool:
        """
        Check if the car moved more than `movement_threshold`. 
        Parameters
        ----------
        movement_threshold: Distance threshold

        Returns
        -------
        True if car moved more than threshold, False otherwise.
        - Special case : if buffer is filled with only one position, this method returns True.

        """
        if self.count < 2:
            return True

        total_distance = self.distance_moved()
        return total_distance > movement_threshold
    
    def distance_moved(self) -> float:
        ordered = self.get_ordered()
        diffs = np.diff(ordered, axis=0)
        total_distance = np.sum(np.linalg.norm(diffs, axis=1))
        return total_distance

if __name__ == "__main__":
    def test_case(name, buffer_size, positions, threshold, expected):
        pb = PositionBuffer(buffer_size)
        for pos in positions:
            pb.add(np.array(pos))
        result = pb.moved_more_than_threshold(threshold)
        print(f"{name}: {'PASS' if result == expected else 'FAIL'}")

    test_case(
        name="Basic movement above threshold",
        buffer_size=5,
        positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        threshold=1.5,
        expected=True
    )

    test_case(
        name="No movement",
        buffer_size=5,
        positions=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        threshold=0.1,
        expected=False
    )

    test_case(
        name="Below movement threshold",
        buffer_size=5,
        positions=[[0, 0, 0], [0.1, 0.1, 0], [0.2, 0.2, 0]],
        threshold=1.0,
        expected=False
    )

    test_case(
        name="Rolling buffer wrap-around",
        buffer_size=3,
        positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        threshold=1.5,
        expected=True
    )

    test_case(
        name="Not enough data",
        buffer_size=5,
        positions=[[0, 0, 0]],
        threshold=0.5,
        expected=True
    )

    test_case(
        name="Unequal movement",
        buffer_size=5,
        positions=[[0, 0, 0], [0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0], [2.5, 0, 0]],
        threshold=2,
        expected=True
    )
