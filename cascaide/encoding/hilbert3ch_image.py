import numpy as np
import torch
from hilbertcurve.hilbertcurve import HilbertCurve
from .base import CoordinateEncoder

class Hilbert3ChEncoder(CoordinateEncoder):
    name = "Hilbert_3Ch_Image"

    def __init__(self, image_size=32, norm_factor=200.0, coord_range=0.9, centroid=None, p=8, EPS=1e-3):
        self.image_size = image_size
        self.norm_factor = norm_factor
        self.coord_range = coord_range
        self.centroid = centroid if centroid is not None else np.zeros(3)
        self.p = p
        self.EPS = EPS
        self.hc = HilbertCurve(self.p, 3)
        self.empty_tol = EPS / 2.0

    @property
    def output_shape(self):
        return (3, self.image_size, self.image_size)

    def encode(self, vac_coords: torch.Tensor, sia_coords: torch.Tensor) -> torch.Tensor:
        H = W = self.image_size
        img = np.ones((3, H, W), dtype=np.float32)
        img[2, :, :] = 0.0

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
        flat_img = img.reshape(3, -1)

        flat_img[:2, :n_real] = np.clip(
            all_data[:n_real, :2].T / self.norm_factor,
            -self.coord_range, self.coord_range
        )

        z_norm = np.clip(all_data[:n_real, 2] / self.norm_factor, -1.0, 1.0)
        z_pos = (z_norm + 1.0) / 2.0 * (1.0 - self.EPS) + self.EPS
        labels = all_data[:n_real, 3]
        flat_img[2, :n_real] = z_pos * labels

        return torch.from_numpy(img)

    def decode(self, image_tensor: torch.Tensor):
        img_np = image_tensor.numpy() if isinstance(image_tensor, torch.Tensor) else image_tensor

        flat = img_np.reshape(3, -1).T  # (H*W, 3)
        z_enc = flat[:, 2]

        real_mask = np.abs(z_enc) > (self.EPS / 2.0)

        if not np.any(real_mask):
            return torch.zeros((0, 3)), torch.zeros((0, 3))

        x = flat[real_mask, 0] * self.norm_factor
        y = flat[real_mask, 1] * self.norm_factor

        z_pos = np.abs(z_enc[real_mask])
        z_norm = (z_pos - self.EPS) / (1.0 - self.EPS) * 2.0 - 1.0
        z = z_norm * self.norm_factor

        labels = np.sign(z_enc[real_mask])
        coords = np.stack([x, y, z], axis=1)

        coords = coords + self.centroid

        vac_dec = coords[labels < 0]
        sia_dec = coords[labels > 0]

        return torch.from_numpy(vac_dec.astype(np.float32)), \
            torch.from_numpy(sia_dec.astype(np.float32))

    def differentiable_decode(self, images: torch.Tensor):
        B, C, H, W = images.shape
        flat = images.permute(0, 2, 3, 1).reshape(B, H * W, 3)  # (B, HW, 3)

        x = (flat[..., 0] / self.coord_range).clamp(-1.0, 1.0)
        y = (flat[..., 1] / self.coord_range).clamp(-1.0, 1.0)
        z_enc = flat[..., 2]  # signed, in [-1, 1]

        z_pos = z_enc.abs().clamp(min=self.EPS)  # in [EPS, 1]
        z_norm = (z_pos - self.EPS) / (1.0 - self.EPS) * 2.0 - 1.0  # in [-1, 1]
        coords = torch.stack([x, y, z_norm], dim=-1)  # (B, HW, 3)

        confidence = torch.sigmoid((z_enc.abs() - self.EPS * 2) * 50.0)
        vac_score = torch.sigmoid(-z_enc * 20.0) * confidence
        sia_score = torch.sigmoid(z_enc * 20.0) * confidence

        return ([coords[b] for b in range(B)],
                [coords[b] for b in range(B)],
                [vac_score[b] for b in range(B)],
                [sia_score[b] for b in range(B)])

    def occupancy_mask(self, image):
        squeeze = (image.dim() == 3)
        if squeeze:
            image = image.unsqueeze(0)
        mask = (torch.abs(image[:, 2:3]) > self.empty_tol).float()
        return mask.squeeze(0) if squeeze else mask