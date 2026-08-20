"""
Module implementing feature transforms for 
PyTorch models.
"""

import torch
from torch import nn
import torch.nn.functional as F


def get_input_features(features):
    if isinstance(features, InputFeatures):
        return features
    elif isinstance(features, nn.Module):
        # Make sure we have outdim
        with torch.no_grad():
            _xyz = torch.randn(1, 3).to(features.B.device)
            features.outdim = features(_xyz).shape[-1]
        return features
    elif features.lower() == "fourier":
        return FourierFeatures(1., 256)
    elif features.lower() in ["positional-encoding", "encoding", "pe"]:
        return PositionalEncoding(10)
    else:
        raise NotImplementedError(f"Unknown input features \"{features}\".")


class InputFeatures(nn.Module):
    """Base class for input features."""

    def __init__(self) -> None:
        super().__init__()
        self.outdim = None


class PositionalEncoding(InputFeatures):
    """Positional encoding, as proposed in NeRF, Mildenhall et al., ECCV 2020."""

    def __init__(self, L, indim=3) -> None:
        super().__init__()
        self.L = L
        
        factors = torch.tensor([[2. ** l for l in range(self.L)]]) * torch.pi
        self.register_buffer('_factors', factors)

        self.outdim = 2 * L * indim
    
    def forward(self, x):
        x_proj = (x.unsqueeze(-1) @ self._factors).flatten(-2)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def __repr__(self):
        return f"PositionalEncoding(L={self.L})"


class FourierFeatures(InputFeatures):
    """Gaussian Fourier features, as proposed in Tancik et al., NeurIPS 2020."""

    def __init__(self, scale, mapdim=256, indim=3) -> None:
        super().__init__()
        self.scale = scale
        self.mapdim = mapdim
        indim = indim

        B = torch.randn(self.mapdim, indim) * self.scale**2
        self.register_buffer('B', B)

        self.outdim = 2 * self.mapdim
    
    def forward(self, x):
        x_proj = (2. * torch.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def __repr__(self):
        return f"FourierFeatures(scale={self.scale}, mapdim={self.mapdim})"