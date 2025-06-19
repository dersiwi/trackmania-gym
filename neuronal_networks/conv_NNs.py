import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.resnet import (
    WeightsEnum,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    ResNeXt101_64X4D_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    )

RESNET_MODELS_WEIGHTS: dict[str,WeightsEnum] = {
    "resnet18": ResNet18_Weights,
    "resnet34": ResNet34_Weights,
    "resnet50": ResNet50_Weights,
    "resnet101": ResNet101_Weights,
    "resnet152": ResNet152_Weights,
    # TODO check if the beneath models work
    "resnext50_32x4d": ResNeXt50_32X4D_Weights,
    "resnext101_32x8d": ResNeXt101_32X8D_Weights,
    "resnext101_64x4d": ResNeXt101_64X4D_Weights,
    "wide_resnet50_2": Wide_ResNet50_2_Weights,
    "wide_resnet101_2": Wide_ResNet101_2_Weights,
}


class PrebuiltResNet(nn.Module):
    """
    A  wrapper for loading pretrained ResNet models from torchvision.
    This class is intended for use cases where a standard ResNet architecture (e.g., resnet18, resnet50, etc.)
    is needed as a backbone, with the final fully connected layer replaced to match thewanted output dim.
    For more information on the available resnet variants check out:
        https://docs.pytorch.org/vision/0.21/models/resnet.html

    :param model_name: Name of the ResNet variant to load (e.g., 'resnet18', 'resnet50').
    :param in_color_channels: Number of input color channels for the cnn
    :param out_dim: dimension of the output of the model 
    :param pretrained: Whether to load pretrained weights.
    :param trainable_backbone: Whether the backbone should also be trainable
    :param use_default_weights: If True, uses torchvision's default weight configs for the specified model.
"""
    def __init__(self, 
                 model_name='resnet18', 
                 in_color_channels = 3,
                 out_dim=21, 
                 pretrained=False,
                 trainable_backbone = False,
                 weights_name = "DEFAULT"):
        
        if model_name not in RESNET_MODELS_WEIGHTS:
            raise ValueError(f"Model '{model_name}' must be one of: {RESNET_MODELS_WEIGHTS.keys()}")

        super(PrebuiltResNet, self).__init__()

        """
        Load the model (with or without pretrained weights)
        """ 
        model_fn = getattr(models, model_name)
        weights = None # not pretrained weights
        if pretrained:
            weight_class =  RESNET_MODELS_WEIGHTS.get(model_name)
            try:
                weights = getattr(weight_class,weights_name)
            except AttributeError:
                available_weights = [w for w in dir(weight_class) if not w.startswith('_')]
                raise ValueError(f"Invalid weights name: '{weights_name}' for model '{model_name}'. "
                     f"Available weights: {available_weights} or use DEFAULT as weights_name")
            
        self.model = model_fn(weights= weights)

        # Optionally freeze backbone parameters
        if not trainable_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        """
        Resnets use 3 color channels as their input layer.
        If the in_color_channels is not 3 add a conv layer before the resnet
        """
        if in_color_channels != 3:
            self.color_adjust = nn.Conv2d(
                in_channels=in_color_channels,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False)
        else:
            self.color_adjust = nn.Identity()

        # Get input features of the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features,out_dim)

    def forward(self, x):
        # assume inputs to be of shape (b,c,w,h)
        x = self.color_adjust(x)
        return self.model(x)
    
class Linesight_Vision_Model(nn.Module):
    """
    This is the same vision model which was used in linesight_rl 
    """
    def __init__(self,in_color_channels=1,img_head_channels = [1, 16, 32, 64, 32]):
        super(Linesight_Vision_Model,self).__init__()
        img_head_channels[0] = in_color_channels
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=img_head_channels[0], out_channels=img_head_channels[1], kernel_size=(4, 4), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3), stride=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Flatten(),
        )
    def forward(self,x):
        return self.model(x)

class NaiveCNNLayer(nn.Module):
    """
    A single convolutional block consisting of Conv2d -> ReLU -> MaxPool2d.

    :param in_channels: Number of input channels to the convolutional layer
    :param out_channels: Number of output channels (filters) for the convolutional layer
    :param kernel_size: Size of the convolutional kernel (default: 3)
    :param stride: Stride for the convolution operation (default: 1)
    :param padding: Padding added to both sides of the input (default: 1)
    :param pool_size: Size of the max pooling window (default: 2)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2):
        super(NaiveCNNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))

class ImprovedCNNLayer(nn.Module):
    """
    A convolutional block with two Conv2D layers, ReLU activations, optional skip connections, and MaxPooling.

    :param in_channels: Number of input channels
    :param out_channels1: Number of output channels for the first convolutional layer
    :param out_channels2: Number of output channels for the second convolutional layer
    :param use_skip: Whether to include a skip connection from input to output
    :param kernel_size: Size of the convolutional kernels (default: 3)
    :param stride: Stride for both convolutional layers (default: 1)
    :param padding: Padding method for convolution (can be 'same', 'valid', or an int)
    :param pool_kernel_size: Size of the max pooling window (default: 2)
    :param pool_stride: Stride of the max pooling operation (default: 2)


   x ────── conv ─── relu ─── conv ─── + ─── relu ─── maxpool
      │                                │
      └────────────────────────────────┘

    """

    def __init__(
        self, 
        in_channels, 
        out_channels1, 
        out_channels2, 
        use_skip, 
        kernel_size=3, 
        stride=1, 
        padding='same', 
        pool_kernel_size=2, 
        pool_stride=2
    ):
        super(ImprovedCNNLayer, self).__init__()

        self.use_skip = use_skip

        # First convolutional layer + activation
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.act1 = nn.ReLU()

        # Second convolutional layer + activation (applied after skip connection)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels1,
            out_channels=out_channels2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.act2 = nn.ReLU()

        # Max pooling operation
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Determine if a skip projection is needed
        self.skip_connection_needed = False
        if self.use_skip:
            self.skip_connection_needed = (in_channels != out_channels2) or (stride != 1)

            if self.skip_connection_needed:
                # 1x1 convolution to match dimensions for skip connection
                self.conv_skip = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels2,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                )
            else:
                self.conv_skip = None
        else:
            self.conv_skip = None

    def forward(self, x):
        # Save original input for skip connection
        identity = x

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        # Process skip connection if enabled
        if self.use_skip:
            if self.conv_skip is not None:
                identity = self.conv_skip(identity)  # Project input to match output
            out = out + identity  # Add skip connection

        # Apply second activation after skip connection
        out = self.act2(out)
        output = self.pool(out)
        return output

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

        # Dynamically calculate flattened output size
        dummy_input = torch.zeros(1, in_color_channels, *image_shape)
        with torch.no_grad():
            x = dummy_input
            for block in self.blocks:
                x = block(x)
            flattened_size = x.view(1, -1).size(1)

        self.output_mlp = nn.Linear(flattened_size, out_dim)

    def _build_blocks(self):
        """To be overridden by subclasses"""
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