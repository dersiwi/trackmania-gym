import sys, os
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.activations = {}
        self.fig = None
        self.axes = None

        # Register hooks for all submodules
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):  # Change to nn.Module to hook all
                layer.register_forward_hook(self.save_activation(name))

    def save_activation(self, name):
        # hook signature is defined by pytorch
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
            #print(f"[Hook] {name}: {output.shape}")
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def visualize(self, num_maps=6, num_rows = 4):
        total_layers = min(num_rows,len(self.activations))

        # Create figure/axes only once
        if self.fig is None or self.axes is None:
            self.fig, self.axes = plt.subplots(total_layers, num_maps, figsize=(num_maps * 3, total_layers * 3))
            if total_layers == 1:
                self.axes = [self.axes]  # Normalize shape if 1 row
            elif num_maps == 1:
                self.axes = [[ax] for ax in self.axes]  # Normalize if 1 column

        for row_idx, (layer_name, activation) in enumerate(self.activations.items()):
            if row_idx >= num_rows: break
            feature_tensor = activation[0]  # first image in batch
            num_to_plot = min(num_maps, feature_tensor.shape[0])

            for col_idx in range(num_maps):
                ax = self.axes[row_idx][col_idx]
                ax.clear()
                if col_idx < num_to_plot:
                    ax.imshow(feature_tensor[col_idx],  origin='lower' ,cmap='viridis')
                    ax.set_title(f"{layer_name}\nC{col_idx}", fontsize=8)
                ax.axis('off')

        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # Non-blocking update


