import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # SwiGLU expects the final dimension to be split into two chunks (L, R)
        # x_left, x_right = torch.chunk(x, chunks=2, dim=-1)
        # swish(x_left) * x_right

        # We split the last dimension explicitly
        dim = x.shape[-1]
        assert dim % 2 == 0, "Last dimension must be even for SwiGLU split."

        # Split into two equal halves
        x_left = x[..., :dim // 2]
        x_right = x[..., dim // 2:]

        # Swish activation: x * sigmoid(x)
        swish_out = x_left * torch.sigmoid(x_left)

        return swish_out * x_right

class EdgeVisionTransformerBlock(nn.Module):
    """A mock ViT block layer utilizing our custom activation function"""
    def __init__(self, in_features):
        super().__init__()
        # We expand features by 2x because SwiGLU splits the output in half
        self.fc1 = nn.Linear(in_features, in_features * 2)
        self.swiglu = SwiGLU()
        self.fc2 = nn.Linear(in_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swiglu(x)
        x = self.fc2(x)
        return x
