import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from einops.layers.torch import Rearrange

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



class PatchEmbedding(nn.Module):
    """
    Converts an input image into a sequence of flattened patch embeddings.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        patch_size (int): Size of each square patch (patch_size x patch_size).
        emb_size (int): Dimensionality of the output patch embeddings.
    """
    def __init__(self, in_channels=3, patch_size=8, emb_size=128):
        super().__init__()
        self.patch_size = patch_size

        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:
            x (Tensor): Shape (batch_size, in_channels, height, width)

        Output:
            Tensor: Shape (batch_size, num_patches, emb_size)
                - num_patches = (height // patch_size) * (width // patch_size)
        """
        return self.projection(x)


class ViTMLP(nn.Module):
    """
    A feedforward multilayer perceptron (MLP) used in Vision Transformers (ViT).

    Args:
        hidden_dim (int): Size of the hidden layer.
        output_dim (int): Size of the output layer.
        dropout (float): Dropout probability applied after each linear layer.
    """
    def __init__(self, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(hidden_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(output_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x
    
class ViTBlock(nn.Module):
    """
    A Transformer block used in Vision Transformers (ViT).

    Args:
        num_hiddens (int): Dimension of the input and output features.
        norm_shape (int or tuple): Shape for LayerNorm normalization.
        mlp_num_hiddens (int): Hidden layer size in the MLP.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        use_bias (bool): Whether to use bias in attention projection layers.
    """
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attn = nn.MultiheadAttention(
            embed_dim=num_hiddens, 
            num_heads=num_heads, 
            dropout=dropout, 
            bias=use_bias,
            batch_first=True  # Ensures (B, N, E) input
        )
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X: Tensor, valid_lens=None) -> Tensor:
        X_norm = self.ln1(X)
        attn_output, _ = self.attn(X_norm, X_norm, X_norm, need_weights=False)
        X = X + self.dropout1(attn_output)
        X = X + self.mlp(self.ln2(X))
        return X
    
class ViT(nn.Module):
    """
    Vision Transformer producing a single embedding vector per input image.
    
    Args:
        in_color_channels (int): Number of image channels.
        img_size (int): Height/width of the input image (assumes square).
        patch_size (int): Patch size (assumes square).
        emb_dim (int): Embedding dimension of each patch.
        n_layers (int): Number of transformer blocks.
        dropout (float): Dropout rate.
        heads (int): Number of attention heads.
        out_dim (int): Final output embedding size.
    """
    def __init__(self, in_color_channels=3, img_size=144, patch_size=4, emb_dim=32,
                 n_layers=6, dropout=0.1, heads=2, out_dim=128):
        super().__init__()

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            in_channels=in_color_channels,
            patch_size=patch_size,
            emb_size=emb_dim
        )

        # Positional embeddings
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, emb_dim))

        # Transformer blocks
        self.layers = nn.ModuleList([
            ViTBlock(
                num_hiddens=emb_dim,
                norm_shape=emb_dim,
                mlp_num_hiddens=emb_dim * 2,
                num_heads=heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

        # Final normalization layer
        self.norm = nn.LayerNorm(emb_dim)

        # Final linear projection from pooled vector to output dimension
        self.fc_out = nn.Linear(emb_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)
        
        Returns:
            Tensor: Output embedding tensor of shape (B, out_dim)
        """
        x = self.patch_embedding(x)        # (B, N, E)
        x = x + self.pos_embedding         # Add positional encoding (B, N, E)

        for layer in self.layers:
            x = layer(x)                   # (B, N, E)

        x = self.norm(x)                   # (B, N, E)

        # Pool over patch tokens (mean pooling)
        x = x.mean(dim=1)                  # (B, E)

        # Project to final output embedding size
        x = self.fc_out(x)                 # (B, out_dim)

        return x