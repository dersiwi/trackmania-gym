import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from neuronal_networks.conv_NNs import VisionModelSix
import sys, os

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.activations = {}

        # Register hooks for all submodules
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):  # Change to nn.Module to hook all
                layer.register_forward_hook(self.save_activation(name))

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
            print(f"[Hook] {name}: {output.shape}")
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def visualize(self, num_maps=6):
        for name, activation in self.activations.items():
            print(f"Visualizing layer: {name}")
            self._plot_feature_maps(activation, name, num_maps)

    def _plot_feature_maps(self, feature_tensor, layer_name, num_maps):
        feature_tensor = feature_tensor[0]  # first item in batch
        num_maps = min(num_maps, feature_tensor.shape[0])
        fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
        for i in range(num_maps):
            axes[i].imshow(feature_tensor[i], cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f"{layer_name}\nChannel {i}")
        plt.tight_layout()
        plt.show()


