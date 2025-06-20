import numpy as np

class RandomRespawnManager:
      
    def __init__(self, filepath : str):
        """
        Parameters
        ---------
        - filepath : string for complete filepath of reference-line
        """
        self.reference_line : np.ndarray = np.load(filepath)
        self.n_reference_points : int = self.reference_line.shape[0]
        self.current_reference_line_point = self.reference_line[0]
    

    def make_simstatedata_from_pos(self,ref_point_idx:int,):
        """
        Constructs a SimStateData object given reference point index. The new position corresponds 
        to the possition fo the ref_point_idx. The orientation of the resulting state is computed 
        such that the agent faces in the direction of the track — from the current reference point
        toward the next one — effectively pointing toward the finish line.

        Parameters:
            ref_point_idx (int): Index of the reference point to teleport to. 
        """
        # Ensure the reference index is valid for lookahead (we need ref_point_idx + 1 to exist).
        assert ref_point_idx < self.n_reference_points - 1

        current_position = self.reference_line[ref_point_idx]
        next_position = self.reference_line[ref_point_idx + 1]

        # Compute the forward direction vector from the current point to the next.
        direction = next_position - current_position

