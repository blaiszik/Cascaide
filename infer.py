"""
Inference and visualization for trained cascade diffusion models.

Usage:
        python infer.py \
          --config config.yaml \
          --checkpoint runs/hilbert4ch_v1/checkpoints/best.pt \
          --output_dir runs/hilbert4ch_v1/figures_best
"""

import argparse
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List

from cascaide.configs.config import load_config
from cascaide.training.trainer import (build_encoder, build_diffusion, build_sampler,
                      build_conditioner)
from cascaide.diffusion.model import UNet
from cascaide.data.dataset import CascadeDataset



def load_model_from_checkpoint(cfg, ckpt_path, device, encoder, use_ema=True):
    conditioner = build_conditioner(cfg, encoder)

    model = UNet(
        encoder_shape=encoder.output_shape,
        conditioner=conditioner,
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        num_res_blocks=cfg.model.num_res_blocks,
        attn_levels=tuple(cfg.model.attn_levels),
        time_embed_dim=cfg.model.time_embed_dim,
        dropout=0.0,                              # disable dropout at inference
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if use_ema and ckpt.get("ema") is not None:
        print(f"[load] using EMA weights from {ckpt_path}")
        model.load_state_dict(ckpt["ema"])
    else:
        print(f"[load] using raw model weights from {ckpt_path}")
        model.load_state_dict(ckpt["model"])

    model.eval()
    return model

def find_gt_for_energies(raw_ds, target_energies_kev, tolerance_kev=2.0):
    sample_energies_ev = np.array([
        raw_ds.samples[i][2] for i in range(len(raw_ds))
    ])
    sample_energies_kev = sample_energies_ev / 1000.0

    out = {}
    for tgt in target_energies_kev:
        idx = int(np.argmin(np.abs(sample_energies_kev - tgt)))
        actual = sample_energies_kev[idx]
        if abs(actual - tgt) > tolerance_kev:
            print(f"[warn] no sample within ±{tolerance_kev} keV of {tgt} keV "
                  f"(closest: {actual:.1f} keV)")
        sample = raw_ds[idx]
        vac = sample["vac_coords"].numpy()
        sia = sample["sia_coords"].numpy()
        out[tgt] = (vac, sia)
    return out

@torch.no_grad()
def generate_for_energies(model, diffusion, sampler, encoder,
                            target_energies_kev, energy_norm_factor,
                            device, batch=True):
    energies_ev = torch.tensor(
        [e * 1000.0 for e in target_energies_kev],
        device=device, dtype=torch.float32
    )
    energies_norm = energies_ev / energy_norm_factor

    n = len(target_energies_kev)
    shape = (n,) + encoder.output_shape

    if batch:
        cond = {"energy": energies_norm}
        imgs = sampler.sample(model, diffusion, shape, cond=cond,
                                device=device)
    else:
        imgs = []
        for i in range(n):
            cond = {"energy": energies_norm[i:i+1]}
            single_shape = (1,) + encoder.output_shape
            img = sampler.sample(model, diffusion, single_shape,
                                   cond=cond, device=device)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

    out = {}
    for i, e in enumerate(target_energies_kev):
        vac, sia = encoder.decode(imgs[i].cpu())
        out[e] = (vac.numpy(), sia.numpy())
    return out


def plot_comparison(gt_samples: Dict[float, Tuple[np.ndarray, np.ndarray]],
                     ddpm_samples: Dict[float, Tuple[np.ndarray, np.ndarray]],
                     fig_idx: int,
                     n_total: int,
                     output_dir: str,
                     elev: float = 22,
                     azim: float = 45):

    energies = sorted(gt_samples.keys())
    fig = plt.figure(figsize=(18, 11))

    for col_idx, energy in enumerate(energies):

        ax_gt = fig.add_subplot(2, 3, col_idx + 1, projection='3d')
        gt_vac, gt_sia = gt_samples[energy]
        n_gt_vac, n_gt_sia = len(gt_vac), len(gt_sia)

        if n_gt_vac > 0:
            ax_gt.scatter(gt_vac[:, 0], gt_vac[:, 1], gt_vac[:, 2],
                            c='blue', s=18, alpha=0.7, edgecolors='darkblue',
                            linewidths=0.3, label=f'Vac ({n_gt_vac})',
                            depthshade=True)
        if n_gt_sia > 0:
            ax_gt.scatter(gt_sia[:, 0], gt_sia[:, 1], gt_sia[:, 2],
                            c='red', s=18, alpha=0.7, edgecolors='darkred',
                            linewidths=0.3, label=f'SIA ({n_gt_sia})',
                            depthshade=True)

        ax_gt.set_title(f'Ground Truth — {energy:.0f} keV\n'
                         f'Vac: {n_gt_vac}  |  SIA: {n_gt_sia}',
                         fontsize=11, fontweight='bold')
        ax_gt.set_xlabel('X (Å)', fontsize=8)
        ax_gt.set_ylabel('Y (Å)', fontsize=8)
        ax_gt.set_zlabel('Z (Å)', fontsize=8)
        ax_gt.tick_params(labelsize=6)
        if n_gt_vac + n_gt_sia > 0:
            ax_gt.legend(fontsize=8, loc='upper left')
        ax_gt.view_init(elev=elev, azim=azim)

        # GT axis limits
        all_gt = []
        if n_gt_vac > 0: all_gt.append(gt_vac)
        if n_gt_sia > 0: all_gt.append(gt_sia)
        if all_gt:
            all_gt = np.vstack(all_gt)
            margin = 10.0
            for set_lim, idx in [(ax_gt.set_xlim, 0),
                                  (ax_gt.set_ylim, 1),
                                  (ax_gt.set_zlim, 2)]:
                lo = all_gt[:, idx].min() - margin
                hi = all_gt[:, idx].max() + margin
                set_lim(lo, hi)


        ax_ddpm = fig.add_subplot(2, 3, col_idx + 4, projection='3d')
        ddpm_vac, ddpm_sia = ddpm_samples[energy]
        n_ddpm_vac, n_ddpm_sia = len(ddpm_vac), len(ddpm_sia)

        if n_ddpm_vac > 0:
            ax_ddpm.scatter(ddpm_vac[:, 0], ddpm_vac[:, 1], ddpm_vac[:, 2],
                              c='blue', s=18, alpha=0.7, edgecolors='darkblue',
                              linewidths=0.3, label=f'Vac ({n_ddpm_vac})',
                              depthshade=True)
        if n_ddpm_sia > 0:
            ax_ddpm.scatter(ddpm_sia[:, 0], ddpm_sia[:, 1], ddpm_sia[:, 2],
                              c='red', s=18, alpha=0.7, edgecolors='darkred',
                              linewidths=0.3, label=f'SIA ({n_ddpm_sia})',
                              depthshade=True)

        ax_ddpm.set_title(f'DDPM Generated — {energy:.0f} keV\n'
                           f'Vac: {n_ddpm_vac}  |  SIA: {n_ddpm_sia}',
                           fontsize=11, fontweight='bold')
        ax_ddpm.set_xlabel('X (Å)', fontsize=8)
        ax_ddpm.set_ylabel('Y (Å)', fontsize=8)
        ax_ddpm.set_zlabel('Z (Å)', fontsize=8)
        ax_ddpm.tick_params(labelsize=6)
        if n_ddpm_vac + n_ddpm_sia > 0:
            ax_ddpm.legend(fontsize=8, loc='upper left')
        ax_ddpm.view_init(elev=elev, azim=azim)

        all_ddpm = []
        if n_ddpm_vac > 0: all_ddpm.append(ddpm_vac)
        if n_ddpm_sia > 0: all_ddpm.append(ddpm_sia)
        if all_ddpm:
            all_ddpm_arr = np.vstack(all_ddpm)
            margin = 10.0
            for set_lim, idx in [(ax_ddpm.set_xlim, 0),
                                  (ax_ddpm.set_ylim, 1),
                                  (ax_ddpm.set_zlim, 2)]:
                lo = all_ddpm_arr[:, idx].min() - margin
                hi = all_ddpm_arr[:, idx].max() + margin
                set_lim(lo, hi)

    fig.suptitle(f'Cascade Defect Comparison — Figure {fig_idx + 1}/{n_total}',
                  fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(output_dir, f'comparison_{fig_idx + 1:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_figures", type=int, default=100,
                          help="Number of comparison figures to generate")
    parser.add_argument("--min_energy", type=float, default=10.0, help="Minimum energy in keV")
    parser.add_argument("--max_energy", type=float, default=170.0, help="Maximum energy in keV")
    parser.add_argument("--samples_per_fig", type=int, default=3, help="How many energies to pick per figure")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--elev", type=float, default=22)
    parser.add_argument("--azim", type=float, default=45)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg, raw_cfg = load_config(args.config)
    device = torch.device(cfg.experiment.device
                            if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.output_dir, "infer_args.json"), "w") as f:
        json.dump({**vars(args),
                    "device": str(device),
                    "encoder": cfg.encoder.name}, f, indent=2)

    print("[data] loading raw dataset...")
    raw_ds = CascadeDataset(data_root=cfg.data.data_root,
                              max_samples=cfg.data.max_samples,
                              compute_centroid=True)

    encoder = build_encoder(cfg, raw_ds.global_centroid)
    diffusion = build_diffusion(cfg).to(device)
    sampler = build_sampler(cfg)
    model = load_model_from_checkpoint(cfg, args.checkpoint, device,
                                          encoder, use_ema=args.use_ema)

    print(f"[infer] encoder={encoder.name}, "
          f"sampler={cfg.sampling.sampler}, "
          f"steps={cfg.sampling.ddim_steps if cfg.sampling.sampler == 'ddim' else cfg.diffusion.T}")


    for fig_idx in range(args.n_figures):
        current_energies = np.random.uniform(args.min_energy, args.max_energy, args.samples_per_fig)
        current_energies = sorted(current_energies.tolist())
        gt_samples = find_gt_for_energies(raw_ds, current_energies, tolerance_kev=5.0)
        ddpm_samples = generate_for_energies(
            model, diffusion, sampler, encoder,
            target_energies_kev=current_energies,
            energy_norm_factor=cfg.data.energy_norm_factor,
            device=device,
            batch=True
        )
        plot_comparison(
            gt_samples=gt_samples,
            ddpm_samples=ddpm_samples,
            fig_idx=fig_idx,
            n_total=args.n_figures,
            output_dir=args.output_dir,
            elev=args.elev,
            azim=args.azim,
        )
        if (fig_idx + 1) % 10 == 0:
            print(f"  generated {fig_idx + 1}/{args.n_figures}")

    print(f"[done] {args.n_figures} figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()