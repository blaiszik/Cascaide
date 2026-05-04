import torch
import torch.nn as nn
from .base import NoiseSchedule

class GaussianDiffusion(nn.Module):
    PARAM_TYPES = ('eps', 'x0', 'v')

    def __init__(
        self,
        T: int = 1000,
        schedule: NoiseSchedule = None,
        parameterization: str = 'eps',
    ):
        super().__init__()
        assert parameterization in self.PARAM_TYPES, \
            f"parameterization must be one of {self.PARAM_TYPES}"

        self.T = T
        self.parameterization = parameterization
        schedule = schedule or CosineSchedule()
        betas = schedule.betas(T)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, 0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64),
                                     alpha_bar[:-1]])

        # Register all schedule tensors as buffers (auto device-handling)
        def reg(name, tensor):
            self.register_buffer(name, tensor.float())

        reg('betas',          betas)
        reg('alphas',         alphas)
        reg('alpha_bar',      alpha_bar)
        reg('alpha_bar_prev', alpha_bar_prev)
        reg('sqrt_ab',        torch.sqrt(alpha_bar))
        reg('sqrt_1m_ab',     torch.sqrt(1.0 - alpha_bar))
        reg('sqrt_recip_a',   torch.sqrt(1.0 / alphas))
        reg('sqrt_recip_ab',  torch.sqrt(1.0 / alpha_bar))
        reg('sqrt_recipm1_ab', torch.sqrt(1.0 / alpha_bar - 1))

        # Posterior q(x_{t-1} | x_t, x_0)
        post_var = betas * (1 - alpha_bar_prev) / (1 - alpha_bar)
        reg('post_var',     post_var)
        reg('post_logvar',  torch.log(post_var.clamp(min=1e-20)))
        reg('post_c1',      betas * alpha_bar_prev.sqrt() / (1 - alpha_bar))   # x0 coeff
        reg('post_c2',      (1 - alpha_bar_prev) * alphas.sqrt() / (1 - alpha_bar))  # xt coeff


    @staticmethod
    def _ext(buf, t, shape):
        out = buf.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        xt = (self._ext(self.sqrt_ab, t, x0.shape) * x0
              + self._ext(self.sqrt_1m_ab, t, x0.shape) * noise)
        return xt, noise

    def predict_x0_from_eps(self, xt, t, eps):
        return (self._ext(self.sqrt_recip_ab, t, xt.shape) * xt
                - self._ext(self.sqrt_recipm1_ab, t, xt.shape) * eps)

    def predict_eps_from_x0(self, xt, t, x0):
        return ((self._ext(self.sqrt_recip_ab, t, xt.shape) * xt - x0)
                / self._ext(self.sqrt_recipm1_ab, t, xt.shape))

    def predict_x0_from_v(self, xt, t, v):
        return (self._ext(self.sqrt_ab, t, xt.shape) * xt
                - self._ext(self.sqrt_1m_ab, t, xt.shape) * v)

    def predict_eps_from_v(self, xt, t, v):
        return (self._ext(self.sqrt_ab, t, xt.shape) * v
                + self._ext(self.sqrt_1m_ab, t, xt.shape) * xt)

    def get_v(self, x0, eps, t):
        return (self._ext(self.sqrt_ab, t, x0.shape) * eps
                - self._ext(self.sqrt_1m_ab, t, x0.shape) * x0)

    def model_output_to_x0_and_eps(self, model_out, xt, t):
        if self.parameterization == 'eps':
            eps = model_out
            x0 = self.predict_x0_from_eps(xt, t, eps)
        elif self.parameterization == 'x0':
            x0 = model_out
            eps = self.predict_eps_from_x0(xt, t, x0)
        else:  # 'v'
            x0 = self.predict_x0_from_v(xt, t, model_out)
            eps = self.predict_eps_from_v(xt, t, model_out)
        return x0, eps


    def q_posterior(self, x0, xt, t):
        mean = (self._ext(self.post_c1, t, xt.shape) * x0
                + self._ext(self.post_c2, t, xt.shape) * xt)
        logvar = self._ext(self.post_logvar, t, xt.shape)
        return mean, logvar

    def training_targets(self, x0, t, noise):
        if self.parameterization == 'eps':
            return noise
        elif self.parameterization == 'x0':
            return x0
        else:
            return self.get_v(x0, noise, t)

