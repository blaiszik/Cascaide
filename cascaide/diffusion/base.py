from abc import ABC, abstractmethod
import torch
from typing import Dict, List, Optional,Tuple
import torch.nn as nn


class NoiseSchedule(ABC):
    @abstractmethod
    def betas(self, T: int) -> torch.Tensor:
        pass


class AuxLoss(ABC, nn.Module):
    name: str = "aux"

    def __init__(self, weight: float = 1.0, apply_every: int = 1):
        super().__init__()
        self.weight = weight
        self.apply_every = apply_every
        self._step = 0

    @abstractmethod
    def compute(self, x0, x0_pred, xt, t, batch, encoder, info):
        pass

    def __call__(self, **kwargs):
        self._step += 1
        if self._step % self.apply_every != 0:
            return None, self.name
        val = self.compute(**kwargs)
        if val is None:
            return None, self.name
        return self.weight * val, self.name


class Sampler(ABC):
    @abstractmethod
    @torch.no_grad()
    def sample(self, model, diffusion, shape, cond=None, device='cuda'):
        pass



class Conditioner(ABC):
    @property
    @abstractmethod
    def extra_in_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def extra_embed_dim(self) -> int:
        pass

    @abstractmethod
    def spatial(self, cond: Dict[str, torch.Tensor],
                spatial_shape: Tuple[int, int]) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def vector(self, cond: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        pass

