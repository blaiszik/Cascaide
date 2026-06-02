import torch
import torch.nn as nn
import torch.nn.functional as F
from cascaide.diffusion.base import AuxLoss
from cascaide.encoding.base_image import BaseImageEncoder
from cascaide.encoding.hilbert3ch_image import Hilbert3ChEncoder
from cascaide.encoding.hilbert4ch_image import Hilbert4ChEncoder
from cascaide.diffusion.core import GaussianDiffusion
from cascaide.encoding.base import CoordinateEncoder
from typing import Dict, List, Optional,Tuple


class DiffusionLoss(nn.Module):
    def __init__(self,
                 diffusion: GaussianDiffusion,
                 encoder: Optional["CoordinateEncoder"] = None,
                 defect_weight: float = 1.0,
                 aux_losses: Optional[list] = None):
        super().__init__()
        self.diffusion = diffusion
        self.encoder = encoder
        self.defect_weight = defect_weight
        self.aux_losses = nn.ModuleList(aux_losses) if aux_losses else nn.ModuleList()

    def _weight_map(self, x0):
        if self.defect_weight == 1.0 or self.encoder is None:
            return None
        with torch.no_grad():
            occ = self.encoder.occupancy_mask(x0)            # (B, 1, H, W)
        return occ * self.defect_weight + (1 - occ) * 1.0

    def forward(self, model, x0, cond=None, batch=None):
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.diffusion.T, (B,), device=device).long()
        noise = torch.randn_like(x0)

        xt, _ = self.diffusion.q_sample(x0, t, noise)
        target = self.diffusion.training_targets(x0, t, noise)
        model_out = model(xt, t, cond=cond)

        wmap = self._weight_map(x0)
        if wmap is not None:
            main_loss = ((model_out - target) ** 2 * wmap).mean()
        else:
            main_loss = F.mse_loss(model_out, target)

        info = {'main': main_loss.item()}
        total = main_loss

        if self.aux_losses:
            x0_pred, _ = self.diffusion.model_output_to_x0_and_eps(
                model_out, xt, t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)
            for aux_fn in self.aux_losses:
                aux_val, aux_name = aux_fn(
                    x0=x0, x0_pred=x0_pred, xt=xt, t=t,
                    info=info, batch=batch, encoder=self.encoder)
                if aux_val is not None:
                    total = total + aux_val
                    info[aux_name] = (aux_val.item() if torch.is_tensor(aux_val)
                                      else aux_val)

        info['total'] = total.item() if torch.is_tensor(total) else total
        return total, info

class DifferentiableMultiAxisProjection(nn.Module):

    def __init__(self, projection_size=128, sigma=0.5, background_value=-1.0):
        super().__init__()
        self.projection_size = projection_size
        self.quad_size = projection_size // 2
        self.sigma = sigma
        self.background_value = background_value

        # Coordinate grid for quadrants [-1, 1]
        ticks = torch.linspace(-1, 1, self.quad_size)
        grid_y, grid_x = torch.meshgrid(ticks, ticks, indexing='ij')
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)

    def _compute_local_density(self, coords_2d, sigma_dens=0.15):
        N = coords_2d.shape[0]
        if N <= 1:
            return torch.zeros(N, device=coords_2d.device)

        dist_sq = torch.cdist(coords_2d, coords_2d).pow(2)
        density = torch.exp(-dist_sq / (2 * sigma_dens ** 2)).sum(dim=1) - 1.0

        max_dens = density.max()
        if max_dens > 1e-6:
            density = (density / (max_dens + 1e-8)) * 2.0 - 1.0
        return density

    def _project_to_quadrant(self, coords, axis1, axis2, depth_axis):
        N = coords.shape[0]
        Q = self.quad_size
        device = coords.device

        if N == 0:
            return torch.full((3, Q, Q), self.background_value, device=device)

        px = coords[:, axis1]
        py = coords[:, axis2]
        depth = coords[:, depth_axis]

        coords_2d = coords[:, [axis1, axis2]]
        density = self._compute_local_density(coords_2d)

        centroid = coords.mean(dim=0)
        dist_to_c = torch.norm(coords - centroid, dim=1)
        max_dist = dist_to_c.max()
        if max_dist > 1e-6:
            dist_to_c = (dist_to_c / max_dist) * 2.0 - 1.0

        dx = px[:, None, None] - self.grid_x[None, :, :]
        dy = py[:, None, None] - self.grid_y[None, :, :]
        dist_sq = dx ** 2 + dy ** 2

        weights = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        sum_w = weights.sum(dim=0) + 1e-6

        ch0 = (weights * depth[:, None, None]).sum(dim=0) / sum_w
        ch1 = (weights * density[:, None, None]).sum(dim=0) / sum_w
        ch2 = (weights * dist_to_c[:, None, None]).sum(dim=0) / sum_w

        quad = torch.stack([ch0, ch1, ch2], dim=0)

        mask = torch.sigmoid((sum_w - 0.05) * 50)
        quad = quad * mask + (1 - mask) * self.background_value

        return quad

    def forward(self, vac_coords, sia_coords):

        v_xy = self._project_to_quadrant(vac_coords, 0, 1, 2)
        s_xy = self._project_to_quadrant(sia_coords, 0, 1, 2)
        top_row = torch.cat([v_xy, s_xy], dim=2)


        v_xz = self._project_to_quadrant(vac_coords, 0, 2, 1)
        s_xz = self._project_to_quadrant(sia_coords, 0, 2, 1)
        bot_row = torch.cat([v_xz, s_xz], dim=2)

        full_image = torch.cat([top_row, bot_row], dim=1)
        return full_image

class MultiAxisProjectionAuxLoss(AuxLoss):

    name = "proj"

    def __init__(self,
                 projection_size: int = 128,
                 sigma: float = 0.5,
                 reconstruction_weight: float = 1.0,
                 gradient_weight: float = 0.3,
                 sparsity_weight: float = 0.1,
                 weight: float = 1.0,
                 apply_every: int = 1,
                 confidence_threshold: float = 0.3,
                 start_epoch: int = 0,
                 t_threshold_frac: float = 1.0,
                 diffusion_T: int = 1000):
        super().__init__(weight=weight, apply_every=apply_every)
        self.projector = DifferentiableMultiAxisProjection(projection_size, sigma)
        self.reconstruction_weight = reconstruction_weight
        self.gradient_weight = gradient_weight
        self.sparsity_weight = sparsity_weight
        self.confidence_threshold = confidence_threshold

        self.start_epoch = start_epoch
        self.t_threshold_frac = t_threshold_frac
        self.diffusion_T = diffusion_T
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    @staticmethod
    def _filter_by_weight(coords, weights, thr):
        valid = weights > thr
        if valid.sum() < 1:
            top = torch.argmax(weights)
            valid = torch.zeros_like(weights, dtype=torch.bool)
            valid[top] = True
        return coords[valid], weights[valid]

    def _project_pred(self, vac_c, sia_c, vac_w, sia_w):
        v_c, _ = self._filter_by_weight(vac_c, vac_w, self.confidence_threshold)
        s_c, _ = self._filter_by_weight(sia_c, sia_w, self.confidence_threshold)
        return self.projector(v_c, s_c)

    def _project_target(self, vac_raw, sia_raw, encoder, device):
        # Tanh-aware encoders (e.g. TanhP99) expose normalize_for_projection so
        # predicted and target projections share one tanh grid space. Linear
        # encoders (Base/Hilbert) fall back to (coord - centroid) / scale.
        if hasattr(encoder, "normalize_for_projection"):
            if len(vac_raw) > 0:
                v = encoder.normalize_for_projection(vac_raw, device)
            else:
                v = torch.zeros(1, 3, device=device)
            if len(sia_raw) > 0:
                s = encoder.normalize_for_projection(sia_raw, device)
            else:
                s = torch.zeros(1, 3, device=device)
            return self.projector(v, s)

        scale = encoder.norm_factor * encoder.coord_range
        centroid = torch.as_tensor(encoder.centroid, device=device,
                                    dtype=torch.float32)
        if len(vac_raw) > 0:
            v = ((vac_raw.to(device) - centroid) / scale).clamp(-1.0, 1.0)
        else:
            v = torch.zeros(1, 3, device=device)
        if len(sia_raw) > 0:
            s = ((sia_raw.to(device) - centroid) / scale).clamp(-1.0, 1.0)
        else:
            s = torch.zeros(1, 3, device=device)
        return self.projector(v, s)

    def compute(self, x0, x0_pred, xt, t, batch, encoder, info):
        if self._current_epoch < self.start_epoch:
            return None

        if not hasattr(encoder, "differentiable_decode"):
            return None
        if batch is None or "vac_coords" not in batch or "sia_coords" not in batch:
            return None

        device = x0_pred.device
        B = x0_pred.shape[0]

        t_max = int(self.diffusion_T * self.t_threshold_frac)
        low_t_mask = t < t_max                                      # (B,)
        n_low_t = int(low_t_mask.sum().item())
        if n_low_t == 0:
            return None

        low_t_idx = torch.where(low_t_mask)[0].tolist()
        x0_pred_lt = x0_pred[low_t_mask]
        target_vac = [batch["vac_coords"][i] for i in low_t_idx]
        target_sia = [batch["sia_coords"][i] for i in low_t_idx]

        pred_vac, pred_sia, vac_w, sia_w = encoder.differentiable_decode(x0_pred_lt)

        recon_total = torch.zeros((), device=device)
        grad_total  = torch.zeros((), device=device)
        spars_total = torch.zeros((), device=device)
        valid = 0

        for b in range(n_low_t):
            n_vac_t = len(target_vac[b])
            n_sia_t = len(target_sia[b])
            if n_vac_t == 0 and n_sia_t == 0:
                continue
            valid += 1

            pred_proj = self._project_pred(pred_vac[b], pred_sia[b],
                                            vac_w[b], sia_w[b])
            tgt_proj  = self._project_target(target_vac[b], target_sia[b],
                                              encoder, device)

            recon_total = recon_total + F.mse_loss(pred_proj, tgt_proj)

            pdx = pred_proj[:, :, 1:] - pred_proj[:, :, :-1]
            pdy = pred_proj[:, 1:, :] - pred_proj[:, :-1, :]
            tdx = tgt_proj[:, :, 1:] - tgt_proj[:, :, :-1]
            tdy = tgt_proj[:, 1:, :] - tgt_proj[:, :-1, :]
            grad_total = grad_total + F.mse_loss(pdx, tdx) + F.mse_loss(pdy, tdy)

            pred_n_vac = vac_w[b].sum()
            pred_n_sia = sia_w[b].sum()
            spars_total = spars_total + (
                (pred_n_vac - n_vac_t).abs() / (n_vac_t + 1) +
                (pred_n_sia - n_sia_t).abs() / (n_sia_t + 1)
            )

        if valid == 0:
            return None

        recon = recon_total / valid
        grad = grad_total / valid
        spars = spars_total / valid

        info["proj_recon"] = recon.item()
        info["proj_grad"] = grad.item()
        info["proj_sparsity"] = spars.item()
        info["proj_n_low_t"] = n_low_t

        return (self.reconstruction_weight * recon +
                self.gradient_weight * grad +
                self.sparsity_weight * spars)


class OccupancyBCELoss(AuxLoss):
    name = "occ"

    def __init__(self, weight: float = 1.0, sharpness: float = 10.0,
                 apply_every: int = 1):
        super().__init__(weight=weight, apply_every=apply_every)
        self.sharpness = sharpness

    @staticmethod
    def _soft_occupancy(image: torch.Tensor, encoder, sharpness: float):

        if isinstance(encoder, BaseImageEncoder):
            # background = 1.0 on channel 0; defect → deviates from 1
            signal = (image[:, 0:1] - 1.0).abs()
            thr = encoder.empty_tol
        elif isinstance(encoder, Hilbert4ChEncoder):
            # background = 0.0 on channel 3; defect → |label| ≈ 1
            signal = image[:, 3:4].abs()
            thr = encoder.empty_tol
        elif isinstance(encoder, Hilbert3ChEncoder):
            # background = 0.0 on channel 2; defect → |z*label| ≥ EPS
            signal = image[:, 2:3].abs()
            thr = encoder.empty_tol
        else:
            # Fallback: hard mask, no gradient (rare path)
            return encoder.occupancy_mask(image)
        return torch.sigmoid((signal - thr) * sharpness)

    def compute(self, x0, x0_pred, xt, t, batch, encoder, info):
        with torch.no_grad():
            target = encoder.occupancy_mask(x0)                    # (B, 1, H, W)
        pred_soft = self._soft_occupancy(x0_pred, encoder, self.sharpness)

        loss = F.binary_cross_entropy(pred_soft.clamp(1e-6, 1 - 1e-6),
                                       target)
        info["occ_pred_density"] = pred_soft.mean().item()
        info["occ_true_density"] = target.mean().item()
        return loss



class CountLoss(AuxLoss):

    name = "count"

    def __init__(self, weight: float = 1.0,
                 use_huber: bool = True,
                 separate_classes: bool = True,
                 apply_every: int = 1):
        super().__init__(weight=weight, apply_every=apply_every)
        self.use_huber = use_huber
        self.separate_classes = separate_classes

    def _per_class_targets(self, batch, x0, encoder, device, B):
        if batch is not None and "n_vac" in batch and "n_sia" in batch:
            n_vac = torch.as_tensor(batch["n_vac"], device=device,
                                     dtype=torch.float32)
            n_sia = torch.as_tensor(batch["n_sia"], device=device,
                                     dtype=torch.float32)
            return n_vac, n_sia

        with torch.no_grad():
            total = encoder.occupancy_mask(x0).sum(dim=(1, 2, 3))
        half = total / 2
        return half, half

    def compute(self, x0, x0_pred, xt, t, batch, encoder, info):
        if not hasattr(encoder, "differentiable_decode"):
            return None

        device = x0_pred.device
        B = x0_pred.shape[0]

        _, _, vac_w, sia_w = encoder.differentiable_decode(x0_pred)

        pred_vac = torch.stack([w.sum() for w in vac_w])           # (B,)
        pred_sia = torch.stack([w.sum() for w in sia_w])           # (B,)

        tgt_vac, tgt_sia = self._per_class_targets(batch, x0, encoder, device, B)

        if self.separate_classes:
            err = torch.cat([pred_vac - tgt_vac, pred_sia - tgt_sia])
        else:
            err = (pred_vac + pred_sia) - (tgt_vac + tgt_sia)

        loss = (F.smooth_l1_loss(err, torch.zeros_like(err))
                if self.use_huber else err.pow(2).mean())

        info["count_pred_vac"] = pred_vac.mean().item()
        info["count_pred_sia"] = pred_sia.mean().item()
        info["count_true_vac"] = tgt_vac.mean().item()
        info["count_true_sia"] = tgt_sia.mean().item()
        return loss

class ClassificationLoss(AuxLoss):
    name = "cls"

    def __init__(self, weight: float = 1.0, apply_every: int = 1,
                 epsilon: float = 1e-6):
        super().__init__(weight=weight, apply_every=apply_every)
        self.eps = epsilon

    def _ground_truth_class_signal(self, x0, encoder):
        if isinstance(encoder, Hilbert4ChEncoder):
            # channel 3: -1 = vac, +1 = SIA
            label = x0[:, 3:4]                                      # (B, 1, H, W)
            vac_target = (label < -0.5).float()
            sia_target = (label >  0.5).float()
            occ = (vac_target + sia_target).clamp(max=1.0)
            return vac_target, sia_target, occ

        if isinstance(encoder, Hilbert3ChEncoder):
            # channel 2: sign carries label
            ch = x0[:, 2:3]
            vac_target = (ch < -encoder.empty_tol).float()
            sia_target = (ch >  encoder.empty_tol).float()
            occ = (vac_target + sia_target).clamp(max=1.0)
            return vac_target, sia_target, occ

        return None  # signals "use the count-based fallback"

    def compute(self, x0, x0_pred, xt, t, batch, encoder, info):
        if not hasattr(encoder, "differentiable_decode"):
            return None

        device = x0_pred.device
        B = x0_pred.shape[0]
        gt = self._ground_truth_class_signal(x0, encoder)

        _, _, vac_w, sia_w = encoder.differentiable_decode(x0_pred)

        if gt is not None:
            H = W = encoder.image_size
            vac_pred = torch.stack([w.view(1, H, W) for w in vac_w])  # (B,1,H,W)
            sia_pred = torch.stack([w.view(1, H, W) for w in sia_w])

            vac_tgt, sia_tgt, occ = gt
            stacked_pred = torch.stack([vac_pred, sia_pred], dim=0)   # (2,B,1,H,W)
            stacked_pred = F.softmax(stacked_pred / 0.5, dim=0)
            vac_prob = stacked_pred[0]
            sia_prob = stacked_pred[1]

            bce = -(vac_tgt * (vac_prob + self.eps).log()
                    + sia_tgt * (sia_prob + self.eps).log())
            loss = (bce * occ).sum() / occ.sum().clamp(min=1.0)
            info["cls_path"] = "per_pixel"
            return loss

        if batch is None or "n_vac" not in batch or "n_sia" not in batch:
            return None
        n_vac = torch.as_tensor(batch["n_vac"], device=device,
                                 dtype=torch.float32)
        n_sia = torch.as_tensor(batch["n_sia"], device=device,
                                 dtype=torch.float32)
        true_frac = n_vac / (n_vac + n_sia + self.eps)

        pred_vac_total = torch.stack([w.sum() for w in vac_w])
        pred_sia_total = torch.stack([w.sum() for w in sia_w])
        pred_frac = pred_vac_total / (pred_vac_total + pred_sia_total + self.eps)

        info["cls_path"] = "fraction"
        info["cls_pred_vac_frac"] = pred_frac.mean().item()
        info["cls_true_vac_frac"] = true_frac.mean().item()
        return F.mse_loss(pred_frac, true_frac)