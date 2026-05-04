import torch
import math
from .base import NoiseSchedule

class LinearSchedule(NoiseSchedule):
    def __init__(self, beta_start=1e-4, beta_end=0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end

    def betas(self, T):
        return torch.linspace(self.beta_start, self.beta_end, T,
                              dtype=torch.float64)


class CosineSchedule(NoiseSchedule):
    def __init__(self, s=0.008):
        self.s = s

    def betas(self, T):
        t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
        ab = torch.cos((t + self.s) / (1 + self.s) * math.pi * 0.5) ** 2
        ab = ab / ab[0]
        betas = 1 - ab[1:] / ab[:-1]
        return betas.clamp(1e-4, 0.999)


class SigmoidSchedule(NoiseSchedule):
    def __init__(self, start=-3, end=3, tau=1.0):
        self.start = start
        self.end = end
        self.tau = tau

    def betas(self, T):
        t = torch.linspace(0, 1, T + 1, dtype=torch.float64)
        v_start = torch.tensor(self.start / self.tau).sigmoid()
        v_end   = torch.tensor(self.end   / self.tau).sigmoid()
        ab = (-((t * (self.end - self.start) + self.start) / self.tau).sigmoid()
              + v_end) / (v_end - v_start)
        ab = ab / ab[0]
        betas = 1 - ab[1:] / ab[:-1]
        return betas.clamp(1e-4, 0.999)
