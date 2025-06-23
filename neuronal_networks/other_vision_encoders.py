"""In this module other vision encoders, like ResNet or the vision encoder used in the linesight-ai project
are implemented."""
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
    :param weights_name: Specify which weights should get loaded for the model
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
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[1], out_channels=img_head_channels[2], kernel_size=(4, 4), stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[2], out_channels=img_head_channels[3], kernel_size=(3, 3), stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=img_head_channels[3], out_channels=img_head_channels[4], kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
        )
    def forward(self,x):
        return self.model(x)
    