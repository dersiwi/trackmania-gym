import numpy as np

class DTransformer:
    """This class transforms a python dictionary containing images and states into a single numpy-array. It provides both way transformations.
    """

    def __init__(self, img_height : int, img_width : int, state_dim : int, n_channels : int = 1, imgkey : str = "image", statekey : str = "state", return_without_channel_if_grayscale : bool = True):
        self.img_height = img_height
        self.img_width = img_width
        self.n_channels = n_channels
        self.state_dim = state_dim
        self.imgkey = imgkey
        self.statekey = statekey
        self._imgdim = self.img_height * self.img_width * self.n_channels
        self.return_without_channel_if_grayscale = return_without_channel_if_grayscale

    def dict_to_numpy(self, dictinoary : dict[str, np.ndarray]) -> np.ndarray:
        """This method transforms a dictionary into a numpy array.
        Args:
            dictionary (dict)   : Containing self.imgkey and self.statekey. 
                - The value of the dictionary for the self.imgkey is a numpy array of shape [self.img_height, self.img_width]
                or [self.img_height, self.img_width, self.n_chanels]
                - The value of the dictionary of the self.statekey is a numpy-array of shape [self.statedim,]"""
        # check for channel on images
        imgs, states = dictinoary[self.imgkey], dictinoary[self.statekey]
        if len(imgs.shape) == 3:
            imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1] * imgs.shape[2])
        else:
            imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1])
        assert imgs.shape[0] == self._imgdim
        return np.hstack([imgs, states])
    
    def numpy_to_dict(self, array : np.ndarray) ->dict[str, np.ndarray]:
        """Trurns the given array into a dictionary."""
        if self.n_channels == 1 and self.return_without_channel_if_grayscale:
            return {self.imgkey : array[0:self._imgdim].reshape((self.img_height, self.img_width)),
                    self.statekey : array[self._imgdim:]}
        return {self.imgkey : array[0:self._imgdim].reshape((self.img_height, self.img_width, self.n_channels)),
                    self.statekey : array[self._imgdim:]}



if __name__ == "__main__":
    h, w, c = 200, 400, 1
    sdim = 54
    img = np.zeros((h,w,c))
    tr = DTransformer(h, w, sdim, c)
    array = tr.dict_to_numpy({tr.imgkey : img, tr.statekey : np.ones(sdim,)})
    tdict = tr.numpy_to_dict(array)
    print(tdict[tr.imgkey])
    print(tdict[tr.statekey])
    print(tdict[tr.imgkey].shape)
    print(tdict[tr.statekey].shape)