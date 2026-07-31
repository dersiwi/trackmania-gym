import numpy as np
import os
from plotting.test_environment_callbacks.core import TestEnvironmentCallback
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

class PretrainingDataCollection(TestEnvironmentCallback):
    def __init__(self, reference_line_manager : ReferenceLineManager, logging_directory : str, continuation_idx : int = -1):
        """
        Note : This Callback cannot be used in mulitprocessing (i.e. it cannot be passed as a plotter).
        Parameters
        -----------
            - reference_line_manager    : ReferenceLineManager used by environment
            - logging_directory         : Logging directory in which dataset is created or recorded
            - continueation_idx         : If this is set to some index other than -1, it is assumed there is already an existing dataset and
                                            the existing dataset is upposed to be extended. If it's -1, a new dataset is created (be careful; if it's -1 and there IS a dataset there, it'll be overwritten.)"""
        super().__init__()
        self.ref_line_manager = reference_line_manager
        self.logging_directory = logging_directory
        self.img_directory = os.path.join(self.logging_directory, "images")
        os.makedirs(self.img_directory, exist_ok = True)
        self.labels = os.path.join(self.logging_directory, "labels.csv")
        
        if not os.path.exists(self.labels) and continuation_idx == -1:
            with open(self.labels, "w") as file:
                file.write("filename,lateral_distance\n")

        if not continuation_idx == -1:
            self.n_step = continuation_idx
        else:
            assert os.path.exists(self.labels), "Expected dataset to already exists but did not find labels."


    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):

        i, d, drel = self.ref_line_manager.get_distance_to_next_point()
        if i == 0:
            self.n_step += 1
            return
        
        idx, lateral_dist = self.ref_line_manager.get_last_calculated_lateral_distance()
        assert i == idx, f"Expected indexes to be the same but got current refline idx {i} and index to which lateral distance was calculated at {idx}"
        with open(self.labels, "a") as file:
            file.write(f"img_{self.n_step}.npy,{lateral_dist}\n")

        img : np.ndarray = processed_obs["image"]
        np.save(os.path.join(self.img_directory, f"img_{self.n_step}.npy"), img)

        
        self.n_step += 1