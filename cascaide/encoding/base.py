from abc import ABC, abstractmethod
import torch

class CoordinateEncoder(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def output_shape(self):
        pass

    @abstractmethod
    def encode(self, vac_coords: torch.Tensor, sia_coords: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, image_tensor: torch.Tensor):
        pass

    @abstractmethod
    def differentiable_decode(self, images: torch.Tensor):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement differentiable_decode")

    @abstractmethod
    def occupancy_mask(self, image: torch.Tensor) -> torch.Tensor:
        pass



