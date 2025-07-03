import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models

from torchvision.models.mobilenetv2 import WeightsEnum,MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights,MobileNet_V3_Small_Weights

MOBILENET_MODELS_WEIGHTS: dict[str,WeightsEnum] =  {
    "mobilenet_v2": MobileNet_V2_Weights,
    "mobilenet_v3_small": MobileNet_V3_Small_Weights,
    "mobilenet_v3_large": MobileNet_V3_Large_Weights,
}

class PrebuiltMobileNet(nn.Module):
    def __init__(self, model_name='mobilenet_v3_small',
                 in_color_channels=3,
                 out_dim=21,
                 pretrained=False,
                 trainable_backbone=False,
                weights_name = "DEFAULT"):
        super().__init__()

        if model_name not in MOBILENET_MODELS_WEIGHTS:
            raise ValueError(f"Model '{model_name}' must be one of: {MOBILENET_MODELS_WEIGHTS.keys()}")

        # Load model
        model_fn = getattr(models, model_name)
        weights = None # not pretrained weights
        if pretrained:
            weight_class =  MOBILENET_MODELS_WEIGHTS.get(model_name)
            try:
                weights = getattr(weight_class,weights_name)
            except AttributeError:
                available_weights = [w for w in dir(weight_class) if not w.startswith('_')]
                raise ValueError(f"Invalid weights name: '{weights_name}' for model '{model_name}'. "
                     f"Available weights: {available_weights} or use DEFAULT as weights_name")
            
        self.model = model_fn(weights=weights)

        # Freeze backbone if required
        if not trainable_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.resize = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

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

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, out_dim)
    
    def forward(self, x):
        x = self.color_adjust(x)
        x = self.resize(x)
        return self.model(x)
    
if __name__ ==  "__main__":
    model = PrebuiltMobileNet(model_name='mobilenet_v3_small')
    print(model.model.classifier[-1])