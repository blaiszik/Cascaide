"""Model-checkpoint generators: turn a trained checkpoint into cascade samples that match
the reference energy distribution, so they can be scored by the same scorecard as baselines.

Two pipelines are supported:
- set-checkpoint   → the self-contained DiT set model in sp_cas.py
- image-checkpoint → the image-encoded 2D-UNet pipeline (cascaide package)

Each returns (samples, meta) where samples is a list of {'vac','sia','energy'} dicts.
"""
import os
import importlib.util
from collections import Counter

import numpy as np


def _energy_counts(reference, decimals=0, cap=None, energy_bin=None):
    """How many samples to generate at each energy (match the reference distribution).
    ``cap`` limits per-energy generation (sampling is the slow step). ``energy_bin`` snaps
    continuous energies to bin centers so generation targets a few bins, not ~100 values."""
    def key(e):
        return round(e / energy_bin) * energy_bin if energy_bin else round(e, decimals)
    c = Counter(key(s["energy"]) for s in reference)
    if cap is not None:
        c = {e: min(n, cap) for e, n in c.items()}
    return dict(c)


def _import_sp_cas():
    path = os.path.join(os.getcwd(), "sp_cas.py")
    if not os.path.exists(path):
        raise FileNotFoundError("sp_cas.py not found in cwd; run from the repo root.")
    spec = importlib.util.spec_from_file_location("sp_cas", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- set pipeline
def generate_set_checkpoint(checkpoint, reference, device=None, n_per_energy=48,
                            energy_bin=None, steps=None, sampler=None):
    if not checkpoint:
        raise ValueError("set-checkpoint requires --checkpoint")
    sp = _import_sp_cas()
    den, diff, ch, norm, cfg = sp.load_model(checkpoint, device=device)
    ediv = cfg.get("energy_divisor", 300.0)
    cap = cfg.get("count_cap", 1300)
    out = []
    for e_keV, n in _energy_counts(reference, cap=n_per_energy, energy_bin=energy_bin).items():
        samples = sp.generate(den, diff, ch, norm, float(e_keV), energy_divisor=ediv,
                              n_samples=n, cap=cap, device=diff.device, steps=steps, sampler=sampler)
        for vac, sia in samples:
            out.append({"vac": np.asarray(vac, np.float32),
                        "sia": np.asarray(sia, np.float32), "energy": float(e_keV)})
    return out, {"normalization": "coordnorm-p99", "energy_divisor": ediv}


# ------------------------------------------------------------------------- paired pipeline
def generate_paired_checkpoint(checkpoint, reference, device=None, n_per_energy=48):
    """Generate from a pair-relative (Frenkel-pair) checkpoint: energy -> count -> n_pairs
    6-D tokens -> denormalize -> (vac, sia). Targets the vac-SIA separation metric."""
    if not checkpoint:
        raise ValueError("paired-checkpoint requires --checkpoint")
    import torch
    sp = _import_sp_cas()
    from cascaide.setdiff.paired import PairedSetDenoiser, PairNormalizer, sample_paired

    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    den = PairedSetDenoiser(d_model=cfg["d_model"], depth=cfg["depth"],
                            n_heads=cfg["n_heads"]).to(device)
    den.load_state_dict(ck["denoiser_state_dict"]); den.eval()
    ch = sp.CountHead().to(device); ch.load_state_dict(ck["counthead_state_dict"])
    diff = sp.CoordDiffusion(T=cfg["T"], schedule=cfg["schedule"], device=device)
    norm = PairNormalizer.from_dict(ck["normalizer"])
    ediv = cfg.get("energy_divisor", 300.0); cap = cfg.get("count_cap", 400)

    out = []
    for e_keV, n in _energy_counts(reference, cap=n_per_energy).items():
        e = torch.full((n,), e_keV / ediv, device=device)
        npairs = ch.sample(e, cap=cap)
        for i in range(n):
            npi = int(npairs[i])
            if npi == 0:
                out.append({"vac": np.zeros((0, 3), np.float32),
                            "sia": np.zeros((0, 3), np.float32), "energy": float(e_keV)})
                continue
            tok = sample_paired(den, diff, e[i:i + 1], npi, device=device)[0].cpu().numpy()
            vac, sia = norm.inverse(tok)
            out.append({"vac": vac, "sia": sia, "energy": float(e_keV)})
    return out, {"normalization": "pairnorm-p99"}


# ------------------------------------------------------------------------- image pipeline
def generate_image_checkpoint(config_path, checkpoint, reference, device=None,
                              n_per_energy=48):
    """Mirror infer.py, but pass the dataset into build_encoder so TanhP99 works (the
    infer.py path is currently missing that — see handoff bug #1)."""
    if not (config_path and checkpoint):
        raise ValueError("image-checkpoint requires --config and --checkpoint")
    import torch
    from cascaide.configs.config import load_config
    from cascaide.training.trainer import (build_encoder, build_diffusion,
                                           build_sampler, build_conditioner)
    from cascaide.diffusion.model import UNet
    from cascaide.data.dataset import CascadeDataset

    cfg, _ = load_config(config_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw_ds = CascadeDataset(data_root=cfg.data.data_root,
                            max_samples=cfg.data.max_samples, compute_centroid=True)
    encoder = build_encoder(cfg, raw_ds.global_centroid, dataset=raw_ds)  # dataset= => TanhP99 ok
    diffusion = build_diffusion(cfg).to(device)
    sampler = build_sampler(cfg)
    conditioner = build_conditioner(cfg, encoder)
    model = UNet(encoder_shape=encoder.output_shape, conditioner=conditioner,
                 base_channels=cfg.model.base_channels,
                 channel_mults=tuple(cfg.model.channel_mults),
                 num_res_blocks=cfg.model.num_res_blocks,
                 attn_levels=tuple(cfg.model.attn_levels),
                 time_embed_dim=cfg.model.time_embed_dim, dropout=0.0).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"]
                          if ckpt.get("ema") is not None else ckpt["model"])
    model.eval()

    enf = cfg.data.energy_norm_factor
    out = []
    with torch.no_grad():
        for e_keV, n in _energy_counts(reference, cap=n_per_energy).items():
            energies = torch.full((n,), e_keV * 1000.0 / enf, device=device)
            shape = (n,) + encoder.output_shape
            imgs = sampler.sample(model, diffusion, shape,
                                  cond={"energy": energies}, device=device)
            for i in range(n):
                vac, sia = encoder.decode(imgs[i].cpu())
                out.append({"vac": vac.numpy().astype(np.float32),
                            "sia": sia.numpy().astype(np.float32), "energy": float(e_keV)})
    return out, {"encoder": cfg.encoder.name}
