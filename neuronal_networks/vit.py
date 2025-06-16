import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models import vit_b_16
VIT_MODELS =  ["vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"]

class ViTPreprocessor(nn.Module):
    """
    Preprocessor using torchvision transforms wrapped into nn.Module.
    Supports resizing, cropping, normalizing, and channel adjustment.
    """
    def __init__(self, in_color_channels:int,normalize:bool):
        super().__init__()
        self.in_color_channels = in_color_channels

        # Channel adjustment if needed
        if in_color_channels != 3:
            self.channel_adapter = nn.Conv2d(in_color_channels, 3, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.channel_adapter = nn.Identity()

        # Normalization transform (ImageNet stats)
        if normalize:
            self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, x):
        # Assume input is a torch.Tensor with shape (B, C, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.channel_adapter(x)

        # torchvision.transforms.Normalize is not nn.Module compatible directly;
        # Apply it manually
        x = self.normalize(x)
        return x



class PrebuiltViT(nn.Module):
    """
    A wrapper for Vision Transformers from torchvision.models.

    Parameters:
    -----------
    model_name: str
        One of the ViT model names in torchvision (e.g., 'vit_b_16').
    in_color_channels: int
        Number of channels in input images (default: 3).
    out_dims: int
        Output dimension of the final classification layer.
    pretrained: bool
        Whether to load pretrained ImageNet weights.
    trainable_backbone: bool
        Whether the ViT backbone is trainable.
    """
    def __init__(self,
                 model_name='vit_b_16',
                 in_color_channels=3,
                 out_dims=21,
                 pretrained=False,
                 trainable_backbone=False):
        super().__init__()

        if model_name not in VIT_MODELS:
            raise ValueError(f"Model '{model_name}' must be one of: {VIT_MODELS}")

        self.out_dims = out_dims
        weights = 'DEFAULT' if pretrained else None

        # Load model
        model_fn = getattr(models, model_name)
        self.model = model_fn(weights=weights)

        # Freeze backbone if required
        if not trainable_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Resize input to 224x224  (required for positional encoding)
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # Handle input channels
        if in_color_channels != 3:
            if pretrained:
                # For pretrained: use adapter conv to map to 3 channels
                self.color_adjust = nn.Conv2d(in_color_channels, 3, kernel_size=1, bias=False)
            else:
                # For non-pretrained: modify the model's conv_proj input layer directly
                original_conv = self.model.conv_proj
                self.model.conv_proj = nn.Conv2d(
                    in_channels=in_color_channels,
                    out_channels=original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )
                self.color_adjust = nn.Identity()
        else:
            self.color_adjust = nn.Identity()

        # Replace classification head
        hidden_dim = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(hidden_dim, out_dims)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)  # Add batch dim if needed
        x = self.color_adjust(x)
        x = self.resize(x)
        return self.model(x)
    """
    to only use the encoder of the vit replace after self.resize()
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x) [batch_size, num_tokens + 1, hidden_dim] this needs to be flattened then 
    """