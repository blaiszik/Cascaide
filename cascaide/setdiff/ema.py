"""EMA with step-count warmup (shared by the set + paired trainers).

A fixed high decay (0.9999) leaves the shadow dominated by the random init on short runs —
the bug seen 2026-06-02 (scorecard ~6 at ep100). Warmup grows the effective decay with the
step count: d_t = min(target, (1+step)/(10+step)), so early on the shadow tracks the live
model and it asymptotes to ``target``. Robust to run length without hand-tuning.
"""
import copy
import torch


class EMAWarmup:
    def __init__(self, model, decay=0.9999, warmup=True):
        self.target = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.step += 1
        d = (min(self.target, (1 + self.step) / (10 + self.step))
             if self.warmup else self.target)
        with torch.no_grad():
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.data.mul_(d).add_(p.data, alpha=1 - d)
