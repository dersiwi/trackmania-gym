import torch
import torch.nn as nn
import torchvision.models as models


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