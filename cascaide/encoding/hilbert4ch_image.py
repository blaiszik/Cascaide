import numpy as np
import torch
from hilbertcurve.hilbertcurve import HilbertCurve
from .base import CoordinateEncoder

class Hilbert4ChEncoder(CoordinateEncoder):
    name = "Hilbert_Spatial_Image"

    def __init__(self, image_size=32, norm_factor=200.0, coord_range=0.9, centroid=None, p=8, empty_tol=0.5):
        self.image_size = image_size
        self.norm_factor = norm_factor
        self.coord_range = coord_range
        self.centroid = centroid if centroid is not None else np.zeros(3)
        self.p = p
        # p=8 provides a 256^3 virtual grid for high-precision spatial sorting
        self.hc = HilbertCurve(self.p, 3)
        self.empty_tol = empty_tol

    @property
    def output_shape(self):
        return (4, self.image_size, self.image_size)

    def encode(self, vac_coords: torch.Tensor, sia_coords: torch.Tensor) -> torch.Tensor:
        H = W = self.image_size
        img = np.ones((4, H, W), dtype=np.float32)
        img[3, :, :] = 0.0

        v = vac_coords.numpy() if isinstance(vac_coords, torch.Tensor) else vac_coords
        s = sia_coords.numpy() if isinstance(sia_coords, torch.Tensor) else sia_coords

        v_tagged = np.hstack([v, -np.ones((len(v), 1))]) if len(v) > 0 else np.zeros((0, 4))
        s_tagged = np.hstack([s, np.ones((len(s), 1))]) if len(s) > 0 else np.zeros((0, 4))
        all_data = np.vstack([v_tagged, s_tagged])

        if len(all_data) == 0:
            return torch.from_numpy(img)

        all_data[:, :3] -= self.centroid

        res = 2 ** self.p

        q = np.clip(((all_data[:, :3] / self.norm_factor) + 1) / 2 * (res - 1), 0, res - 1).astype(int)
        distances = self.hc.distances_from_points(q)
        all_data = all_data[np.argsort(distances)]

        n_real = min(len(all_data), H * W)
        flat_img = img.reshape(4, -1)

        flat_img[:3, :n_real] = np.clip(all_data[:n_real, :3].T / self.norm_factor,
                                        -self.coord_range, self.coord_range)
        flat_img[3, :n_real] = all_data[:n_real, 3]

        return torch.from_numpy(img)

    def decode(self, image_tensor: torch.Tensor):

        img = image_tensor.numpy() if isinstance(image_tensor, torch.Tensor) else image_tensor
        flat_img = img.reshape(4, -1)


        mask = np.abs(flat_img[3, :]) > 0.5
        valid_points = flat_img[:, mask]

        if valid_points.size == 0:
            return torch.zeros((0, 3)), torch.zeros((0, 3))

        coords = (valid_points[:3, :].T * self.norm_factor) + self.centroid
        labels = valid_points[3, :]

        vac_coords = coords[labels < 0]
        sia_coords = coords[labels > 0]

        return torch.from_numpy(vac_coords.astype(np.float32)), \
            torch.from_numpy(sia_coords.astype(np.float32))

    def differentiable_decode(self, images: torch.Tensor):

        B, C, H, W = images.shape
        flat = images.permute(0, 2, 3, 1).reshape(B, H * W, 4)  # (B, HW, 4)

        coords = (flat[..., :3] / self.coord_range).clamp(-1.0, 1.0)  # (B, HW, 3)
        label = flat[..., 3]  # (B, HW)

        # Soft mask in [0, 1]: 0 when label≈0, 1 when |label|≈1
        confidence = torch.sigmoid((label.abs() - 0.5) * 10.0)  # (B, HW)
        # Soft vac/SIA assignment: tanh on label
        vac_score = torch.sigmoid(-label * 5.0) * confidence  # (B, HW)
        sia_score = torch.sigmoid(label * 5.0) * confidence  # (B, HW)

        # Same coords, different weights — projector will down-weight non-class pixels
        return ([coords[b] for b in range(B)],
                [coords[b] for b in range(B)],
                [vac_score[b] for b in range(B)],
                [sia_score[b] for b in range(B)])

    def occupancy_mask(self, image):
        squeeze = (image.dim() == 3)
        if squeeze:
            image = image.unsqueeze(0)
        mask = (torch.abs(image[:, 3:4]) > self.empty_tol).float()
        return mask.squeeze(0) if squeeze else mask

