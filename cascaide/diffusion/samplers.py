import torch
from .base import Sampler

class DDPMSampler(Sampler):
    def __init__(self, clip_x0: bool = True, x0_clip_range=(-1.5, 1.5)):
        self.clip_x0 = clip_x0
        self.x0_clip_range = x0_clip_range

    @torch.no_grad()
    def sample(self, model, diffusion, shape, cond=None, device='cuda',
               return_intermediates=False):
        model.eval()
        x = torch.randn(shape, device=device)
        intermediates = [x] if return_intermediates else None

        for tv in reversed(range(diffusion.T)):
            t = torch.full((shape[0],), tv, device=device, dtype=torch.long)
            model_out = model(x, t, cond=cond)
            x0_pred, _ = diffusion.model_output_to_x0_and_eps(model_out, x, t)
            if self.clip_x0:
                x0_pred = x0_pred.clamp(*self.x0_clip_range)

            mean, logvar = diffusion.q_posterior(x0_pred, x, t)
            if tv > 0:
                noise = torch.randn_like(x)
                x = mean + (0.5 * logvar).exp() * noise
            else:
                x = mean

            if return_intermediates:
                intermediates.append(x)

        return (x, intermediates) if return_intermediates else x


class DDIMSampler(Sampler):
    def __init__(self, n_steps: int = 50, eta: float = 0.0,
                 clip_x0: bool = True, x0_clip_range=(-1.5, 1.5)):
        self.n_steps = n_steps
        self.eta = eta
        self.clip_x0 = clip_x0
        self.x0_clip_range = x0_clip_range

    @torch.no_grad()
    def sample(self, model, diffusion, shape, cond=None, device='cuda',
               return_intermediates=False):
        model.eval()
        x = torch.randn(shape, device=device)
        intermediates = [x] if return_intermediates else None

        step_idx = torch.linspace(0, diffusion.T - 1, self.n_steps + 1).long()
        step_idx = list(reversed(step_idx.tolist()))

        for i in range(len(step_idx) - 1):
            t_cur, t_nxt = step_idx[i], step_idx[i + 1]
            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)

            model_out = model(x, t, cond=cond)
            x0_pred, eps = diffusion.model_output_to_x0_and_eps(model_out, x, t)
            if self.clip_x0:
                x0_pred = x0_pred.clamp(*self.x0_clip_range)
                eps = diffusion.predict_eps_from_x0(x, t, x0_pred)

            ab_cur = diffusion.alpha_bar[t_cur]
            ab_nxt = diffusion.alpha_bar[t_nxt] if t_nxt >= 0 else torch.tensor(1.0, device=device)

            sigma = self.eta * torch.sqrt(
                (1 - ab_nxt) / (1 - ab_cur) * (1 - ab_cur / ab_nxt))
            noise_term = torch.sqrt(1 - ab_nxt - sigma ** 2) * eps

            x = ab_nxt.sqrt() * x0_pred + noise_term
            if t_nxt > 0 and self.eta > 0:
                x = x + sigma * torch.randn_like(x)

            if return_intermediates:
                intermediates.append(x)

        return (x, intermediates) if return_intermediates else x
