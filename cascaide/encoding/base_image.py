import torch
import numpy as np
from .base import CoordinateEncoder

class BaseImageEncoder(CoordinateEncoder):
    name = "Base_coord_Image"

    def __init__(self, image_size=24, norm_factor=200.0, coord_range=0.9, centroid=None, empty_tol=0.05):
        self.image_size = image_size
        self.norm_factor = norm_factor
        self.coord_range = coord_range
        self.centroid = centroid if centroid is not None else np.zeros(3)
        self.empty_tol = empty_tol

    @property
    def output_shape(self):
        return (3, self.image_size, self.image_size)

    def encode(self, vac_coords, sia_coords):
        H = W = self.image_size
        image = np.ones((3, H, W), dtype=np.float32)

        vac = vac_coords.numpy() if isinstance(vac_coords, torch.Tensor) else vac_coords
        sia = sia_coords.numpy() if isinstance(sia_coords, torch.Tensor) else sia_coords

        if len(vac) > 0:
            vac = vac - self.centroid
        if len(sia) > 0:
            sia = sia - self.centroid

        n_vac, n_sia = len(vac), len(sia)
        max_per_type = H * W // 2

        if n_vac > 0:
            vac_norm = np.clip(vac / self.norm_factor,
                               -self.coord_range, self.coord_range)
            for i in range(min(n_vac, max_per_type)):
                row, col = i // W, i % W
                if row < H:
                    image[:, row, col] = vac_norm[i]

        if n_sia > 0:
            sia_norm = np.clip(sia / self.norm_factor,
                               -self.coord_range, self.coord_range)
            for i in range(min(n_sia, max_per_type)):
                idx = H * W - 1 - i
                row, col = idx // W, idx % W
                if row >= 0:
                    image[:, row, col] = sia_norm[i]

        return torch.from_numpy(image)

    def decode(self, image_tensor):

        if isinstance(image_tensor, torch.Tensor):
            img = image_tensor.detach().cpu().numpy()
        else:
            img = image_tensor

        H, W = self.image_size, self.image_size
        max_per_type = (H * W) // 2

        vac_list = []
        sia_list = []

        outer = self.coord_range + 0.15

        near_bg_dev = 0.3


        for linear_idx in range(max_per_type):
            row = linear_idx // W
            col = linear_idx % W
            pixel = img[:, row, col]

            if np.all(np.abs(pixel) <= outer):
                if np.any(np.abs(pixel - 1.0) > near_bg_dev):
                    vac_list.append(pixel)
            else:
                break  

        for i in range(max_per_type):
            linear_idx = H * W - 1 - i
            row = linear_idx // W
            col = linear_idx % W
            pixel = img[:, row, col]

            if np.all(np.abs(pixel) <= outer):
                if np.any(np.abs(pixel - 1.0) > near_bg_dev):
                    sia_list.append(pixel)
            else:
                break

        vac_coords = np.array(vac_list, dtype=np.float32) if vac_list \
            else np.zeros((0, 3), dtype=np.float32)
        sia_coords = np.array(sia_list, dtype=np.float32) if sia_list \
            else np.zeros((0, 3), dtype=np.float32)

        if len(vac_coords) > 0:
            vac_coords = vac_coords * self.norm_factor + self.centroid
        if len(sia_coords) > 0:
            sia_coords = sia_coords * self.norm_factor + self.centroid

        return (torch.from_numpy(vac_coords.astype(np.float32)),
                torch.from_numpy(sia_coords.astype(np.float32)))

    def differentiable_decode(self, images: torch.Tensor):
        B, C, H, W = images.shape
        device = images.device
        mid = H * W // 2
        temperature = 0.1
        pixels = images.permute(0, 2, 3, 1).reshape(B, H * W, 3)

        vac_pixels = pixels[:, :mid, :]  # (B, mid, 3)
        sia_idx = torch.arange(H * W - 1, mid - 1, -1, device=device)
        sia_pixels = pixels[:, sia_idx, :]  # (B, mid, 3)

        vac_emptiness = (vac_pixels - 1.0).abs().mean(dim=-1)  # (B, mid)
        sia_emptiness = (sia_pixels - 1.0).abs().mean(dim=-1)
        vac_w = torch.sigmoid((vac_emptiness - 0.3) / temperature)
        sia_w = torch.sigmoid((sia_emptiness - 0.3) / temperature)

        vac_coords = (vac_pixels / self.coord_range).clamp(-1.0, 1.0)
        sia_coords = (sia_pixels / self.coord_range).clamp(-1.0, 1.0)

        return ([vac_coords[b] for b in range(B)],
                [sia_coords[b] for b in range(B)],
                [vac_w[b] for b in range(B)],
                [sia_w[b] for b in range(B)])

    def occupancy_mask(self, image, tol=0.2):
        if image.dim() == 3:
            image = image.unsqueeze(0);
            squeeze = True
        else:
            squeeze = False
        mask = ((image[:, 0:1] - 1.0).abs() > self.empty_tol).float()  # (B,1,H,W)
        return mask.squeeze(0) if squeeze else mask


