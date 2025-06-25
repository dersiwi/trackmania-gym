import numpy as np
import logging

class ReferenceLineManager:
    
    def __init__(self, filepath : str, lookahead_size : int = 10):
        """
        Parameters
        ---------
        - filepath : string for complete filepath of reference-line
        - lookahead_size : when calculating the distance to next point, looks at only this many next points.
        """
        self.reference_line : np.ndarray = np.load(filepath)
        self.n_reference_points : int = self.reference_line.shape[0]
        assert self.reference_line.shape[1] == 3, f"Expecting shape (N,3), but got {self.reference_line.shape}"
        """Reference line is a numpy array of shape (N, 3), where N is the number of points along the track. The position is interesting however.
        It seems that it is not x,y,z; as thought but rather x,z,y OR y,z,x (TODO)"""

        # calculate accumulated distances for each linesegment.
        self.segment_lengths = np.linalg.norm(self.reference_line[1:] - self.reference_line[:-1], axis=1)
        self.mean_segment_length = np.mean(self.segment_lengths)
        self.accumulated_distances = np.concatenate([[0], np.cumsum(self.segment_lengths)])
        assert self.accumulated_distances.shape[0] == self.reference_line.shape[0], f"Expected shape[0] of accumulated distnaces to be {self.reference_line.shape[0]} but got {self.accumulated_distances}"
        self.relative_accumulated_distances = self.accumulated_distances / self.accumulated_distances[-1]

        self.next_point_idx = 0
        self.lookahead : int = lookahead_size

        self.logger = logging.getLogger(self.__class__.__name__)

        self.calculate_and_step_nextpoint_return : tuple[int, float, float] = None


    def get_reference_line_points(self, begin_idx : int, end_idx : int, interpolate : bool = False) -> np.ndarray:
        """Getter for reference-line points. Basically slices refline[begin_idx : end_idx].
         If end_idx > len(refline) and interpolate == True, then this method interpolates end-points,
          such that the line is extended to the desired length 
          
          Returns
          -------
            - positions of reference-line points; shape [N, 3]"""
        
        assert begin_idx >= 0, "Cannot interpolate the beginning of the line, Begin_idx has to be greater than one"

        if end_idx >= self.n_reference_points:
            if interpolate:
                num_points = end_idx - self.reference_line + 1
                p1, p2 = self.reference_line[-2], self.reference_line[-1]
                direction = p2 - p1
                direction = direction / np.linalg.norm(direction) * self.mean_segment_length # normalize and add mean length

                new_points = [p2 + i * direction for i in range(1, num_points + 1)]
                return np.vstack([self.reference_line[begin_idx : self.n_reference_points], new_points])
            else:
                raise ValueError("")

        return self.reference_line[begin_idx : end_idx]

    def calculate_and_step_next_point(self, car_position : np.ndarray) -> tuple[int, float, float]:
        """Calculates the distance to the next point of the reference line that has not been passed. 
        The reference line has N points and this method returns the distance to the next point, as well as the index of the next point. 
        A point i has been passed, once the distance to point i+1 d(i+1) is smaller than the distance to the current point i : d(i) >= d(i+1).
        Once a point has been passed, it is irreversably passed, until reset() is called.

        After this calculation has been done, the turn-values of this method can be accessed using self.get_distance_to_next_point()
        This is done, such that the environment can do this calculation ONCE, and all other instances,
        like observation or reward-manager can access the values without continuously advancing the nextpoint.

        I.e. this method should only be called ONCE ! every environment step.
        
        
        Returns
        -------
        Tuple [i, d, drel] 
            - i : current index to next point
            - d : floatvalue of distance to next point
            - drel : relative travelled distance along centerline"""
        # calculate distances d(i) to d(i+n) where n==lookahead
        end_index = min(self.next_point_idx + self.lookahead, self.n_reference_points)
        next_points = self.reference_line[self.next_point_idx : end_index]
        distances = np.linalg.norm(next_points - car_position, axis=1, keepdims = True) # shape [N, 1] containing distances

        # find smallest distance and set index accordinly
        min_idx = np.argmin(distances)
        self.next_point_idx = self.next_point_idx + min_idx
        self.calculate_and_step_nextpoint_return = (self.next_point_idx, distances[min_idx], self.relative_accumulated_distances[self.next_point_idx])
        return self.next_point_idx, distances[min_idx], self.relative_accumulated_distances[self.next_point_idx]


    def get_distance_to_next_point(self) -> tuple[int, float, float]:
        """
        Getter for values of self.calculate_and_step_next_point. This method can be called as often one likes for one 
        environment step, AFTER calculate_and_step_next_point has been called in this environment step.
        If not, the previous values are returned.

        Returns (same as self.calculate_and_step_next_point)
        -------
        Tuple [i, d, drel] 
            - i : current index to next point
            - d : floatvalue of distance to next point
            - drel : relative travelled distance along centerline"""
        assert not self.calculate_and_step_nextpoint_return == None, "Cannot return these values, they have not ben calculated yet. "
        return self.calculate_and_step_nextpoint_return
    
    def calculate_lateral_difference(self, idx : int, car_position : np.ndarray):
        """Calculates the lateral distance the car position to the reference line, according to the current linesgement.
        Intuition: if this is very small, the car is very centered, if it's large, the car is far on the left/right on the track
        
        Parameters
        ----------
            - idx : index of next! reference-line-point
            """
        if idx == 0:
            return 0 
        distvector_d1 = car_position - self.reference_line[idx - 1]
        distance_last_point = np.linalg.norm(distvector_d1)

        #calculate scalar projection sp = (d(i-1)^T s) / |s| of car position onto line-segement s = d(i-1) - d(i)
        s = self.reference_line[idx] - self.reference_line[idx - 1]
        sp : float = np.dot(distvector_d1.T, s) / np.linalg.norm(s)

        """sp < 0 : car is before line-segemnt
           sp € [0,1] : car inside line-segement
           sp > 1 : should not happen (car after line-segment)""" 

        # pythagoras to get length of normal vector w of s, such that position of car p - w = a * s for some scalar a.
        lateral_distance = np.sqrt(distance_last_point ** 2 - sp ** 2)

        return lateral_distance
    
    def get_discrete_distance(self, refline_idx : int) -> float:
        """Say d(i) is the accumulated distance for all line-segments until i, then this method returns;
        d(i) - d(i-1), only the distance between i and i-1."""
        if refline_idx == 0:
            return 0
        return self.relative_accumulated_distances[refline_idx] - self.relative_accumulated_distances[refline_idx - 1]

    def reset(self):
        """Resets the current point-index to 0."""
        self.next_point_idx = 0
        self.calculate_and_step_nextpoint_return = None



def run_tests():
    print("Running tests...")

    # Simple 3-point straight line
    ref = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
    testfile = "test_ref.npy"
    np.save(testfile, ref)
    tracker = ReferenceLineManager(testfile, lookahead_size=3)

    # Test 1: Car near first point
    tracker.reset()
    car = np.array([1, 0, 0])
    idx, dist, reldist = tracker.get_distance_to_next_point(car)
    assert idx == 0, f"Expected index 0, got {idx}"
    print(reldist)
    assert np.isclose(dist, 1.0), f"Unexpected distance: {dist}"

    # Test 2: Car near second point
    tracker.reset()
    car = np.array([11, 0, 0])
    idx, dist, reldist  = tracker.get_distance_to_next_point(car)
    print(reldist)
    assert idx == 1, f"Expected index 1, got {idx}"

    # Test 3: Car far ahead, closest to point 2
    tracker.reset()
    car = np.array([19, 0, 0])
    idx, dist, reldist  = tracker.get_distance_to_next_point(car)
    print(reldist)
    assert idx == 2, f"Expected index 2, got {idx}"

    # Test 4: No backward motion
    tracker.next_point_idx = 2
    car = np.array([5, 0, 0])
    idx, dist, reldist  = tracker.get_distance_to_next_point(car)
    print(reldist)
    assert idx == 2, f"Expected index to remain at 2, got {idx}"

    # Test 5: Reset works
    tracker.reset()
    assert tracker.next_point_idx == 0, f"Expected reset to 0, got {tracker.next_point_idx}"
    import os
    os.remove(testfile)
    print("All tests passed!")



if __name__ == "__main__":
    level1 = "tracks/reference_line/Level1.npy"
    simple = "tracks/reference_line/simple1_validated.npy"
    refline = ReferenceLineManager(filepath=simple)
    #run_tests()
    random_array = np.random.randint(0, 1, size=(3,))
    carpos = refline.reference_line[1] - np.array([0.002, 0.0, 0.0])
    idx, dist, reldist = refline.get_distance_to_next_point(carpos)
    print(idx, dist, reldist)
