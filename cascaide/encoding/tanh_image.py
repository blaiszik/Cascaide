import torch
import numpy as np
from .base import CoordinateEncoder


class TanhImageEncoder(CoordinateEncoder):
    """
    Raster-packed encoder with per-axis tanhP99 normalization and global
    centering (replaces the single-scalar ``norm_factor`` of BaseImageEncoder).

    Centering: a single fixed global offset G (``centroid``) is subtracted from
    every cascade, so absolute box positions are preserved and decode can add G
    back to recover ABSOLUTE coordinates -- including for generated samples.

    Normalization (per-axis 3-vectors c = tanh_center, s = tanh_scale):
        encode:  pixel = clip( tanh((coord - G - c) / s) * coord_range )
        decode:  coord = G + c + s * arctanh( clip(pixel / coord_range, +-0.999) )

    tanh squashes to (-1, 1); * coord_range keeps it inside (-0.9, 0.9) so it
    never collides with the 1.0 image background. ``differentiable_decode`` and
    ``occupancy_mask`` are identical to BaseImageEncoder because the tanh
    squashing is already baked into the pixel value.
    """
    name = "TanhP99_coord_Image"

    def __init__(self, image_size=64, coord_range=0.9, centroid=None,
                 tanh_center=None, tanh_scale=None, empty_tol=0.2):
        self.image_size = image_size
        self.coord_range = coord_range
        self.centroid = (np.asarray(centroid, dtype=np.float32).reshape(3)
                         if centroid is not None else np.zeros(3, dtype=np.float32))
        self.empty_tol = empty_tol

        c = (np.asarray(tanh_center, dtype=np.float32).reshape(3)
             if tanh_center is not None else np.zeros(3, dtype=np.float32))
        s = (np.asarray(tanh_scale, dtype=np.float32).reshape(3)
             if tanh_scale is not None else np.ones(3, dtype=np.float32))
        s = np.where(np.abs(s) < 1e-8, 1.0, s).astype(np.float32)
        self.tanh_center = c
        self.tanh_scale = s

    @property
    def output_shape(self):
        return (3, self.image_size, self.image_size)

    # ---- normalization helpers (numpy, for encode/decode) ----
    def _tanh_encode(self, coords):
        # coords: (N, 3) raw (absolute) -> (N, 3) in (-coord_range, coord_range)
        centered = coords - self.centroid
        z = np.tanh((centered - self.tanh_center) / self.tanh_scale)
        return np.clip(z * self.coord_range, -self.coord_range, self.coord_range)

    def _tanh_decode(self, pixels):
        ratio = np.clip(pixels / self.coord_range, -0.999, 0.999)
        centered = self.tanh_center + self.tanh_scale * np.arctanh(ratio)
        return centered + self.centroid

    def encode(self, vac_coords, sia_coords):
        H = W = self.image_size
        image = np.ones((3, H, W), dtype=np.float32)

        vac = vac_coords.numpy() if isinstance(vac_coords, torch.Tensor) else vac_coords
        sia = sia_coords.numpy() if isinstance(sia_coords, torch.Tensor) else sia_coords

        max_per_type = H * W // 2

        if len(vac) > 0:
            vac_norm = self._tanh_encode(vac)
            for i in range(min(len(vac_norm), max_per_type)):
                row, col = i // W, i % W
                if row < H:
                    image[:, row, col] = vac_norm[i]

        if len(sia) > 0:
            sia_norm = self._tanh_encode(sia)
            for i in range(min(len(sia_norm), max_per_type)):
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
        outer = self.coord_range + 0.15
        near_bg_dev = 0.3

        vac_list, sia_list = [], []

        for linear_idx in range(max_per_type):
            pixel = img[:, linear_idx // W, linear_idx % W]
            if np.all(np.abs(pixel) <= outer):
                if np.any(np.abs(pixel - 1.0) > near_bg_dev):
                    vac_list.append(pixel)
            else:
                break

        for i in range(max_per_type):
            linear_idx = H * W - 1 - i
            pixel = img[:, linear_idx // W, linear_idx % W]
            if np.all(np.abs(pixel) <= outer):
                if np.any(np.abs(pixel - 1.0) > near_bg_dev):
                    sia_list.append(pixel)
            else:
                break

        vac = (np.array(vac_list, dtype=np.float32) if vac_list
               else np.zeros((0, 3), dtype=np.float32))
        sia = (np.array(sia_list, dtype=np.float32) if sia_list
               else np.zeros((0, 3), dtype=np.float32))

        if len(vac) > 0:
            vac = self._tanh_decode(vac)
        if len(sia) > 0:
            sia = self._tanh_decode(sia)

        return (torch.from_numpy(vac.astype(np.float32)),
                torch.from_numpy(sia.astype(np.float32)))

    def differentiable_decode(self, images: torch.Tensor):
        # Identical to BaseImageEncoder: pixel == tanh(...)*coord_range already,
        # so pixel / coord_range == the tanh-squashed coordinate in [-1, 1].
        B, C, H, W = images.shape
        device = images.device
        mid = H * W // 2
        temperature = 0.1
        pixels = images.permute(0, 2, 3, 1).reshape(B, H * W, 3)

        vac_pixels = pixels[:, :mid, :]
        sia_idx = torch.arange(H * W - 1, mid - 1, -1, device=device)
        sia_pixels = pixels[:, sia_idx, :]

        vac_emptiness = (vac_pixels - 1.0).abs().mean(dim=-1)
        sia_emptiness = (sia_pixels - 1.0).abs().mean(dim=-1)
        vac_w = torch.sigmoid((vac_emptiness - 0.3) / temperature)
        sia_w = torch.sigmoid((sia_emptiness - 0.3) / temperature)

        vac_coords = (vac_pixels / self.coord_range).clamp(-1.0, 1.0)
        sia_coords = (sia_pixels / self.coord_range).clamp(-1.0, 1.0)

        return ([vac_coords[b] for b in range(B)],
                [sia_coords[b] for b in range(B)],
                [vac_w[b] for b in range(B)],
                [sia_w[b] for b in range(B)])

    def normalize_for_projection(self, coords, device=None):
        """Map RAW (absolute) coords into the same [-1, 1] tanh grid space the
        decoder implies, for the projection aux-loss target side:
            grid = tanh((coord - G - c) / s)
        (== applying tanh to the globally-centered coords, matching encode.)
        """
        if device is None:
            device = coords.device
        coords = coords.to(device)
        c = torch.as_tensor(self.tanh_center, device=device, dtype=torch.float32)
        s = torch.as_tensor(self.tanh_scale, device=device, dtype=torch.float32)
        g = torch.as_tensor(self.centroid, device=device, dtype=torch.float32)
        return torch.tanh((coords - g - c) / s)

    def occupancy_mask(self, image, tol=0.2):
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        mask = ((image[:, 0:1] - 1.0).abs() > self.empty_tol).float()
        return mask.squeeze(0) if squeeze else mask
