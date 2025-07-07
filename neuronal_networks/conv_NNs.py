import torch
import torch.nn as nn
import torchvision.models as models

from neuronal_networks.cnn_blocks import ImprovedCNNLayer, NaiveCNNLayer


class VisionModelSix(nn.Module): #seal team 6 very cool
    def __init__(self, in_color_channels : int, out_dim : int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_color_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # output fixed spatial size: (128, 4, 4)
        )


        self.flatten = nn.Flatten()
        self.output_layer = nn.Sequential(
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x

class VisionModelEight(nn.Module):
    def __init__(self, in_color_channels : int, out_dim=256):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            conv_block(in_color_channels, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),  # Down H,W -> H/2, W/2

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),  # H/4, W/4

            conv_block(64, 128),
            conv_block(128, 128),
        )


        with torch.no_grad():
            dummy_input = torch.zeros(1, in_color_channels, *(128, 128)) #TODO terrible to hardcode images size here.
            dummy_out = self.encoder(dummy_input)
            self.flatten_dim = dummy_out.view(1, -1).shape[1]

        self.feature_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.feature_projector(x)
        return x 




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