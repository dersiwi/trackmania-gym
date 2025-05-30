import torch.nn as nn
import torchvision.models as models
class PrebuiltResNet(nn.Module):
    """
    A  wrapper for loading pretrained ResNet models from torchvision.
    This class is intended for use cases where a standard ResNet architecture (e.g., resnet18, resnet50, etc.)
    is needed as a backbone, with the final fully connected layer replaced to match thewanted output dim.

    :param model_name: Name of the ResNet variant to load (e.g., 'resnet18', 'resnet50').
    :param in_color_channels : Number of input color channels for the cnn
    :param out_dims: 
    :param pretrained: Whether to load pretrained weights.
    :param trainable_backbone: Whether the backbone should also be trainable
"""
    def __init__(self, 
                 model_name='resnet18', 
                 in_color_channels = 3,
                 out_dims=21, 
                 pretrained=False,
                 trainable_backbone = False):
        super(PrebuiltResNet, self).__init__()

        # Dynamically load the specified model from torchvision.models
        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' not found in torchvision.models")

        
        # Load the model (with or without pretrained weights)
        """
        resnet has a preprocessor that rescales the images and also automatically normalizes
        see https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18
        """ 
        model_fn = getattr(models, model_name)
        self.model = model_fn(weights='DEFAULT' if pretrained else None)

        # Optionally freeze backbone parameters
        if trainable_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.color_adjust = nn.Identity()
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

        """ this would override the first conv layer of the resnet 
        if in_color_channels != 3:
        self.model.conv1 = nn.Conv2d(
        in_channels=in_color_channels,
        out_channels=self.model.conv1.out_channels,
        kernel_size=self.model.conv1.kernel_size,
        stride=self.model.conv1.stride,
        padding=self.model.conv1.padding,
        bias=False
        )
        """
        # Get input features of the final fully connected layer
        in_features = self.model.fc.in_features
        # Replace the classification head
        self.model.fc = nn.Linear(in_features,out_dims)
    def forward(self, x):
        # we defined images to be of shape (w,h,c)
        # everything down there is ugly need to fix that 
        x = x.permute(2,0,1)
        x = self.color_adjust(x)
        if x.ndim == 3: x = x.unsqueeze(0)
        return self.model(x)