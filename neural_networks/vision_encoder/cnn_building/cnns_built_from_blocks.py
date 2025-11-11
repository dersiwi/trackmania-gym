import torchvision.models as models
import torch
import torch.nn as nn
from neural_networks.vision_encoder.cnn_building.cnn_blocks import ImprovedCNNLayer, NaiveCNNLayer


class BaseCNN(nn.Module):
    def __init__(
        self,
        num_blocks,
        in_color_channels,
        image_shape,  # (H, W)
        out_dim=10,
    ):
        super(BaseCNN, self).__init__()
        self.blocks = nn.ModuleList()
        self.in_channels = in_color_channels
        self.image_shape = image_shape
        self.num_blocks = num_blocks

        self._build_blocks()

        flattened_size = self._get_cnn_dimension(in_color_channels, image_shape)

        self.output_mlp = nn.Linear(flattened_size, out_dim)

    def _get_cnn_dimension(self, in_color_channels : int, image_shape : tuple[int, int]) -> int:
        """Creates a dummy-input and passes it through the vision encoder in order to get the size of the 
        vision encoder, when flattend"""

        # Dynamically calculate flattened output size
        dummy_input = torch.zeros(1, in_color_channels, *image_shape)
        with torch.no_grad():
            x = dummy_input
            for block in self.blocks:
                x = block(x)
            flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def _build_blocks(self):
        """This method is responsible for creating the neural network itself. It does this by 
        chaining blocks together in order to create a network.
        This method sets the variable self.blocks : ModuleList"""
        raise NotImplementedError("Subclasses must implement _build_blocks")

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = torch.flatten(x, 1)
        return self.output_mlp(x)

class NaiveCNN(BaseCNN):
    def __init__(
        self,
        num_blocks,
        in_color_channels,
        out_channels,
        image_shape,
        kernel_size=3,
        stride=1,
        padding=1,
        pool_size=2,
        out_dim=10,
    ):
        self.initial_out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        super(NaiveCNN, self).__init__(num_blocks, in_color_channels, image_shape, out_dim)

    def _build_blocks(self):
        out_channels = self.initial_out_channels
        in_channels = self.in_channels
        for _ in range(self.num_blocks):
            self.blocks.append(NaiveCNNLayer(
                in_channels, out_channels,
                self.kernel_size, self.stride, self.padding, self.pool_size
            ))
            in_channels = out_channels
            out_channels *= 2

class ImprovedCNN(BaseCNN):
    def __init__(
        self,
        num_blocks,
        in_color_channels,
        image_shape,
        out_channels1 = 32,
        out_channels2 = 64,
        out_dim=10,
        use_skip=True,
    ):
        self.use_skip = use_skip
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        super(ImprovedCNN, self).__init__(num_blocks, in_color_channels, image_shape, out_dim)

        
    def _build_blocks(self):
        in_channels = self.in_channels
        out_channels1 = self.out_channels1
        out_channels2 = self.out_channels2

        # First block
        self.blocks.append(ImprovedCNNLayer(in_channels, out_channels1, out_channels2, use_skip=self.use_skip))
        in_channels = out_channels2

        # Remaining blocks
        for _ in range(1, self.num_blocks):
            out_channels1 = min(in_channels * 2, 256)
            out_channels2 = min(out_channels1 * 2, 512)
            self.blocks.append(ImprovedCNNLayer(in_channels, out_channels1, out_channels2, use_skip=self.use_skip))
            in_channels = out_channels2
