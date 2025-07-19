import torch
import torch.nn as nn

class VisionModel(nn.Module):
    """
    A model that combines an encoder with a feature projection head.

    This model allows for dynamic calculation of the flattened feature dimension 
    after the encoder, and uses that dimension to initialize a projection head 
    (e.g., an MLP or linear layer) for downstream tasks.

    Args:
        encoder (nn.Module): A CNN-based feature extractor module.
        feature_projector (Callable[[int], nn.Module]): A callable that takes the 
            flattened encoder output size and returns a projection module.
        img_shape (tuple): The shape of the input image tensor (C, H, W).
    """
    def __init__(self, encoder: nn.Module, feature_projector, img_shape):
        super().__init__()

        self.encoder = encoder

        # Dynamically determine the output dimension of the encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, *img_shape)
            dummy_out = self.encoder(dummy_input)
            self.flatten_dim = dummy_out.view(1, -1).shape[1]

        # Initialize the feature projector using the flattened encoder output size
        self.feature_projector = feature_projector(self.flatten_dim)

    def forward(self, x):
        """
            x: Input image tensor of shape (B, C, H, W)
        """
        x = self.encoder(x)
        x = self.feature_projector(x)
        return x

class VisionModelSix(VisionModel):  # Seal Team 6. Very cool.
    def __init__(self,out_dim: int, img_shape=(3, 64, 64)):
        encoder = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, kernel_size=5, stride=1, padding=2),
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
        )

        # Call the base constructor with flattening included in the projector
        super().__init__(
            encoder= encoder,
            feature_projector=lambda in_dim: nn.Sequential(
                nn.Flatten(),                    
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ),
            img_shape=img_shape
        )

class VisionModelEight(VisionModel):
    def __init__(self,out_dim: int = 256, img_shape=(3, 128, 128)):
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )

        encoder = nn.Sequential(
            conv_block(img_shape[0], 32),
            conv_block(32, 32),
            nn.MaxPool2d(2), 

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2), 

            conv_block(64, 128),
            conv_block(128, 128),
            nn.Dropout(0.2)
        )

        super().__init__(
            encoder=encoder,
            feature_projector=lambda in_dim: nn.Sequential(
                nn.Flatten(),              
                nn.Linear(in_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, out_dim),
                nn.ReLU(inplace=True)
            ),
            img_shape=img_shape
        )

class LinesightVisionModel(VisionModel):
    """
    Vision model used in linesight_rl
    """
    def __init__(self, out_dim=256, img_head_channels=None, img_shape=(1, 64, 64)):
        if img_head_channels is None:
            img_head_channels = [1, 16, 32, 64, 32]  # Default

        img_head_channels[0] = img_shape[0]

        # Define the encoder based on the img_head_channels sequence
        encoder = nn.Sequential(
            nn.Conv2d(img_head_channels[0], img_head_channels[1], kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(img_head_channels[1], img_head_channels[2], kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(img_head_channels[2], img_head_channels[3], kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(img_head_channels[3], img_head_channels[4], kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Use the VisionModel base class with a projector
        super().__init__(
            encoder=encoder,
            feature_projector=lambda in_dim: nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True)
            ),
            img_shape=img_shape
        )

class SophyVisionModel(VisionModel):
    def __init__(self, out_dim: int = 256, img_shape=(3, 64, 64)):
        # Define the convolutional encoder
        encoder = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),             
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),              
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Use base class to dynamically infer flattened dim
        super().__init__(
            encoder=encoder,
            feature_projector=lambda in_dim: nn.Sequential(
                nn.Flatten(),                      # Dynamically flatten output
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True)
            ),
            img_shape=img_shape
        )