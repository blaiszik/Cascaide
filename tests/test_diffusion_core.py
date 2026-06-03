import torch
import pytest

from cascaide.diffusion.core import GaussianDiffusion
from cascaide.diffusion.schedules import LinearSchedule, CosineSchedule


class DummyModel(torch.nn.Module):
    def forward(self, x, t, cond=None):
        return torch.zeros_like(x)

def get_input(batch_size=2, shape=(3, 16, 16), t_max=50):
    # t must be < T (the diffusion these tests build uses T=50); sampling t >= T overflows
    # the schedule-buffer gather.
    x0 = torch.randn(batch_size, *shape)
    t = torch.randint(0, t_max, (batch_size,))
    return x0, t


def test_diffusion_init():
    diff = GaussianDiffusion(
        T=100,
        schedule=CosineSchedule(),
        parameterization="eps"
    )

    assert diff.T == 100
    assert diff.alpha_bar.shape[0] == 100
    assert torch.all(diff.betas > 0)
    assert torch.all(diff.betas < 1)


def test_q_sample_shapes():
    diff = GaussianDiffusion(T=50)

    x0, t = get_input()
    xt, noise = diff.q_sample(x0, t)

    assert xt.shape == x0.shape
    assert noise.shape == x0.shape


def test_q_sample_reproducible():
    diff = GaussianDiffusion(T=50)

    x0, t = get_input()

    torch.manual_seed(0)
    xt1, n1 = diff.q_sample(x0, t, noise=torch.randn_like(x0))

    torch.manual_seed(0)
    xt2, n2 = diff.q_sample(x0, t, noise=torch.randn_like(x0))

    assert torch.allclose(xt1, xt2)
    assert torch.allclose(n1, n2)

def test_eps_parameterization_consistency():
    diff = GaussianDiffusion(T=50, parameterization="eps")

    x0, t = get_input()
    noise = torch.randn_like(x0)

    xt, eps = diff.q_sample(x0, t, noise)

    x0_pred = diff.predict_x0_from_eps(xt, t, eps)

    assert x0_pred.shape == x0.shape
    assert torch.isfinite(x0_pred).all()

def test_v_parameterization():
    diff = GaussianDiffusion(T=50, parameterization="v")

    x0, t = get_input()
    eps = torch.randn_like(x0)

    v = diff.get_v(x0, eps, t)
    x0_pred = diff.predict_x0_from_v(
        diff.q_sample(x0, t)[0], t, v
    )

    assert x0_pred.shape == x0.shape

def test_posterior():
    diff = GaussianDiffusion(T=50)

    x0, t = get_input()
    xt, _ = diff.q_sample(x0, t)

    mean, logvar = diff.q_posterior(x0, xt, t)

    assert mean.shape == x0.shape
    # posterior log-variance depends only on t, so it is returned broadcast-shaped
    # (B,1,1,1) and must broadcast against x0 (the sampler relies on this).
    assert logvar.shape[0] == x0.shape[0]
    assert (logvar + torch.zeros_like(x0)).shape == x0.shape
    assert torch.isfinite(mean).all()
    assert torch.isfinite(logvar).all()


def test_training_targets():
    diff = GaussianDiffusion(T=50, parameterization="eps")

    x0, t = get_input()
    noise = torch.randn_like(x0)

    target = diff.training_targets(x0, t, noise)

    assert target.shape == x0.shape
    assert torch.allclose(target, noise)