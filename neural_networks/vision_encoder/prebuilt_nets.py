import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models

from torchvision.models.mobilenetv2 import WeightsEnum,MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights,MobileNet_V3_Small_Weights

from torchvision.models.vision_transformer import (
    ViT_H_14_Weights,
    ViT_B_16_Weights,
    ViT_L_16_Weights,
    ViT_B_32_Weights,
    ViT_L_32_Weights
)

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

MOBILENET_MODELS_WEIGHTS: dict[str,WeightsEnum] =  {
    "mobilenet_v2": MobileNet_V2_Weights,
    "mobilenet_v3_small": MobileNet_V3_Small_Weights,
    "mobilenet_v3_large": MobileNet_V3_Large_Weights,
}
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
VIT_MODELS_WEIGHTS: dict[str,WeightsEnum] =  {
    "vit_h_14": ViT_H_14_Weights,
    "vit_b_16": ViT_B_16_Weights,
    "vit_l_16": ViT_L_16_Weights,
    "vit_b_32": ViT_B_32_Weights,
    "vit_l_32": ViT_L_32_Weights,
}

class PrebuiltNet(nn.Module):
    def __init__(self, modeltype : str):
        super().__init__()
        self.modeltype = modeltype
        if modeltype == "resnet":
            self.model_weights = RESNET_MODELS_WEIGHTS
        elif modeltype == "mobilenet":
            self.model_weights = MOBILENET_MODELS_WEIGHTS
        elif modeltype == "vit":
            self.model_weights = VIT_MODELS_WEIGHTS
    
    def adjust_fc_layer(self, out_dim : int):
        """Adjustts the output-layer of the network to match the output dimension."""
        if self.modeltype == "mobilenet": #for mobile net
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, out_dim)
        elif self.modeltype == "resnet": # for resnet
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features,out_dim)
        elif self.modeltype == "vit":
            hidden_dim = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(hidden_dim, out_dim)

VISION_MODELS_IMG_SIZES = {
    "mobilenet": (256,256),
    "resnet": (256,256),
    "vit": (224,224),
}

class PrebuiltnetImplementation(PrebuiltNet):
    """
        A  wrapper for loading pretrained ResNet models from torchvision.
        This class is intended for use cases where a standard ResNet architecture (e.g., resnet18, resnet50, etc.)
        is needed as a backbone, with the final fully connected layer replaced to match thewanted output dim.
        For more information on the available resnet variants check out:
            https://docs.pytorch.org/vision/0.21/models/resnet.html

        :param model_name: Name of the ResNet variant to load (e.g., 'resnet18', 'resnet50').
        :param modeltype : Type of prebuilt model (e.g. resnet, mobilenet...)
        :param in_color_channels: Number of input color channels for the cnn
        :param out_dim: dimension of the output of the model 
        :param pretrained: Whether to load pretrained weights.
        :param trainable_backbone: Whether the backbone should also be trainable
        :param weights_name: Specify which weights should get loaded for the model
    """
    def __init__(self, model_name='mobilenet_v3_small', modeltype = "mobilenet",
                img_shape=(3, 256, 256),
                 out_dim=21,
                 pretrained=False,
                 trainable_backbone=False,
                weights_name = "DEFAULT",**kwargs):
        super().__init__(modeltype)

        if model_name not in self.model_weights:
            raise ValueError(f"Model '{model_name}' must be one of: {self.model_weights.keys()}")

        self.model = self.load_model(model_name, pretrained, weights_name)

        # Freeze backbone if required
        if not trainable_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.img_size = VISION_MODELS_IMG_SIZES.get(modeltype) or self._raise_unknown_model_error(modeltype)
        self.resize = nn.Upsample(size= self.img_size, mode='bilinear', align_corners=False)

        self.color_adjust = self.get_color_adjust(img_shape[0])

        # super ugly way to distinguish between Models
        self.adjust_fc_layer(out_dim)

    def load_model(self, model_name : str, pretrained : bool, weights_name : str) -> nn.Module:
        """Loads the model with model_name, pretrained if set and with given names."""
        model_constructor = getattr(models, model_name)
        weights = None # not pretrained weights
        if pretrained:
            weight_class =  self.model_weights.get(model_name)
            try:
                weights = getattr(weight_class, weights_name)
            except AttributeError:
                available_weights = [w for w in dir(weight_class) if not w.startswith('_')]
                raise ValueError(f"Invalid weights name: '{weights_name}' for model '{model_name}'. "
                     f"Available weights: {available_weights} or use DEFAULT as weights_name.")
            
        return model_constructor(weights=weights)

    def get_color_adjust(self, in_color_channels : int) -> nn.Module:
        """Get the input layer to the network depending on the color channel. (Mobile net expects 3.)"""
        if in_color_channels != 3:
            return nn.Conv2d(in_channels=in_color_channels, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            return nn.Identity()
    
    def forward(self, x):
        x = self.color_adjust(x)
        x = self.resize(x)
        return self.model(x)
    
if __name__ ==  "__main__":
    model = PrebuiltnetImplementation(model_name='mobilenet_v3_small')
    print(model.model.classifier[-1])
