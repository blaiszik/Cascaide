from .base import Conditioner
import torch.nn as nn

class EnergyChannelConditioner(Conditioner):
    def __init__(self, n_channels: int = 3):
        self.n_channels = n_channels

    @property
    def extra_in_channels(self):
        return self.n_channels

    @property
    def extra_embed_dim(self):
        return 0

    def spatial(self, cond, spatial_shape):
        energy = cond["energy"]                       # (B,)
        B = energy.shape[0]
        H, W = spatial_shape
        return energy[:, None, None, None].expand(B, self.n_channels, H, W)

    def vector(self, cond):
        return None


class EnergyEmbedConditioner(Conditioner):
    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @property
    def extra_in_channels(self):
        return 0

    @property
    def extra_embed_dim(self):
        return self.embed_dim

    def spatial(self, cond, spatial_shape):
        return None

    def vector(self, cond):
        return self.mlp(cond["energy"][:, None])  # (B, embed_dim)

    def parameters(self):
        return self.mlp.parameters()

    def to(self, device):
        self.mlp = self.mlp.to(device)
        return self

