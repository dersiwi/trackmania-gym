import torch.nn as nn
import torchvision.models as models

class PrebuiltResNet(nn.Module):
    """
    A  wrapper for loading pretrained ResNet models from torchvision.
    This class is intended for use cases where a standard ResNet architecture (e.g., resnet18, resnet50, etc.)
    is needed as a backbone, with the final fully connected layer replaced to match thewanted output dim.

    :param model_name: Name of the ResNet variant to load (e.g., 'resnet18', 'resnet50').
    :param out_dims: 
    :param pretrained: Whether to load pretrained weights.
    :param trainable_backbone: Whether the backbone should also be trainable
"""
    def __init__(self, model_name='resnet18', out_dims=21, pretrained=False,trainable_backbone = False):
        super(PrebuiltResNet, self).__init__()

        # Dynamically load the specified model from torchvision.models
        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' not found in torchvision.models")

        # Load the model (with or without pretrained weights)
        model_fn = getattr(models, model_name)
        self.model = model_fn(weights='DEFAULT' if pretrained else None)

        # Optionally freeze backbone parameters
        if trainable_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get input features of the final fully connected layer
        in_features = self.model.fc.in_features
        # Replace the classification head
        self.model.fc = nn.Linear(in_features,out_dims)
    def forward(self, x):
        return self.model(x)