"""Pair-relative set diffusion: model a cascade as a SET of Frenkel PAIRS.

Each token is a 6-D pair [vac_xyz, disp_xyz] with sia = vac + disp (see pairing.py). This
exploits conservation (one count -> both populations) and models the vacancy->interstitial
OFFSET directly — the lone failing scorecard metric (vac-SIA separation) once more data
closed the g(r) gap.

Self-contained DiT (mirrors sp_cas.SetDenoiser but: 6-D in/out, no type embedding — every
token is a pair). Reuses sp_cas.CoordDiffusion for the schedule at train time (dim-agnostic
except its sampler, so we provide `sample_paired`).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pairing import to_pair_relative, from_pair_relative, pairs_to_tokens, tokens_to_pairs


# ------------------------------------------------------------------ DiT (compact)
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, d, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(h, d))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.adaLN[-1].weight); nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c, key_padding_mask=None):
        sa, ca, ga, sm, cm, gm = self.adaLN(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), sa, ca)
        a, _ = self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)
        x = x + ga.unsqueeze(1) * a
        h = modulate(self.norm2(x), sm, cm)
        x = x + gm.unsqueeze(1) * self.mlp(h)
        return x


class PairedSetDenoiser(nn.Module):
    """DiT over 6-D pair tokens. Predicts eps on [vac_xyz, disp_xyz]."""
    def __init__(self, d_model=256, n_heads=8, depth=6, mlp_ratio=4.0, in_dim=6):
        super().__init__()
        self.d = d_model
        self.in_proj = nn.Linear(in_dim, d_model)
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.e_mlp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, mlp_ratio)
                                     for _ in range(depth)])
        self.norm_f = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_f = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.proj_out = nn.Linear(d_model, in_dim)
        for m in (self.adaLN_f[-1], self.proj_out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t, energy, key_padding_mask=None):
        c = self.t_mlp(timestep_embedding(t, self.d)) + self.e_mlp(energy[:, None])
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, c, key_padding_mask=key_padding_mask)
        sh, sc = self.adaLN_f(c).chunk(2, dim=-1)
        h = modulate(self.norm_f(h), sh, sc)
        return self.proj_out(h)


# ------------------------------------------------------------------ normalization
class PairNormalizer:
    """Normalize 6-D pair tokens: vacancies by (global center, per-axis p99 scale); the
    displacement (already ~centered) by its own per-axis p99 scale — so the small vac->SIA
    offset uses the full dynamic range instead of being dwarfed by absolute positions."""
    def __init__(self, percentile=99.0):
        self.percentile = percentile
        self.Gv = np.zeros(3, np.float32); self.sv = np.ones(3, np.float32)
        self.cd = np.zeros(3, np.float32); self.sd = np.ones(3, np.float32)

    def fit(self, vac_all, disp_all):
        if len(vac_all):
            self.Gv = vac_all.mean(0).astype(np.float32)
            self.sv = np.where(np.abs(p := np.percentile(np.abs(vac_all - self.Gv),
                               self.percentile, 0)) < 1e-8, 1.0, p).astype(np.float32)
        if len(disp_all):
            self.cd = np.median(disp_all, 0).astype(np.float32)
            self.sd = np.where(np.abs(p := np.percentile(np.abs(disp_all - self.cd),
                               self.percentile, 0)) < 1e-8, 1.0, p).astype(np.float32)
        return self

    def transform(self, vac, disp):
        v = (vac - self.Gv) / self.sv
        d = (disp - self.cd) / self.sd
        return np.concatenate([v, d], 1).astype(np.float32)

    def inverse(self, tokens):
        v, d = tokens[:, :3], tokens[:, 3:]
        vac = (v * self.sv + self.Gv).astype(np.float32)
        disp = (d * self.sd + self.cd).astype(np.float32)
        return vac, (vac + disp).astype(np.float32)          # (vac, sia)

    def to_dict(self):
        return {"percentile": self.percentile, "Gv": self.Gv.tolist(),
                "sv": self.sv.tolist(), "cd": self.cd.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        o = cls(d["percentile"])
        for k in ("Gv", "sv", "cd", "sd"):
            setattr(o, k, np.asarray(d[k], np.float32))
        return o


# ------------------------------------------------------------------ dataset
def build_pair_items(samples, normalizer=None, energy_divisor=300.0, percentile=99.0):
    """samples: list of {'vac','sia','energy'(keV)}. Returns (items, count_energy,
    count_npairs, normalizer). items = [{tokens(Np,6), energy, n_pairs}] for non-empty."""
    pairs = []  # (vac_matched, disp, energy)
    for s in samples:
        vm, disp = to_pair_relative(s["vac"], s["sia"])
        pairs.append((vm, disp, s["energy"]))
    if normalizer is None:
        vac_all = np.concatenate([p[0] for p in pairs if len(p[0])] or
                                 [np.zeros((0, 3), np.float32)], 0)
        disp_all = np.concatenate([p[1] for p in pairs if len(p[1])] or
                                  [np.zeros((0, 3), np.float32)], 0)
        normalizer = PairNormalizer(percentile).fit(vac_all, disp_all)
    items, e_all, n_all = [], [], []
    for vm, disp, e in pairs:
        n = len(vm)
        e_all.append(e / energy_divisor); n_all.append(n)
        if n > 0:
            items.append({"tokens": torch.from_numpy(normalizer.transform(vm, disp)),
                          "energy": torch.tensor(e / energy_divisor, dtype=torch.float32),
                          "n_pairs": n})
    return (items,
            torch.tensor(e_all, dtype=torch.float32),
            torch.tensor(n_all, dtype=torch.float32), normalizer)


def collate_pairs(batch):
    toks, energies, npairs = zip(*[(b["tokens"], b["energy"], b["n_pairs"]) for b in batch])
    B = len(batch); N = max(t.shape[0] for t in toks)
    pad = torch.zeros(B, N, 6); mask = torch.zeros(B, N, dtype=torch.bool)
    for i, t in enumerate(toks):
        pad[i, :t.shape[0]] = t; mask[i, :t.shape[0]] = True
    return pad, mask, torch.stack(energies), torch.tensor(npairs, dtype=torch.float32)


# ------------------------------------------------------------------ sampling
def save_paired(denoiser, counthead, normalizer, cfg, path, epoch):
    """Checkpoint a paired model (denoiser + count head + PairNormalizer + config)."""
    torch.save({"epoch": epoch,
                "denoiser_state_dict": denoiser.state_dict(),
                "counthead_state_dict": counthead.state_dict(),
                "normalizer": normalizer.to_dict(),
                "config": {**cfg, "representation": "paired"}}, path)


@torch.no_grad()
def sample_paired(model, diff, energy, N, clip=4.0, device="cpu"):
    """Reverse DDPM on 6-D pair tokens. energy:[B], N tokens -> tokens [B,N,6]."""
    model.eval()
    B = energy.shape[0]
    x = torch.randn(B, N, 6, device=device)
    for tv in reversed(range(diff.T)):
        t = torch.full((B,), tv, device=device, dtype=torch.long)
        eps = model(x, t, energy, key_padding_mask=None)
        x0 = diff.predict_x0(x, t, eps).clamp(-clip, clip)
        mean = (diff._ext(diff.post_c1, t, x.shape) * x0 +
                diff._ext(diff.post_c2, t, x.shape) * x)
        if tv > 0:
            x = mean + torch.sqrt(diff._ext(diff.post_var, t, x.shape)) * torch.randn_like(x)
        else:
            x = mean
    return x
