#!/usr/bin/env python
"""Train the PAIR-RELATIVE set diffusion model (Frenkel pairs as 6-D tokens).

Each cascade is modeled as a set of pairs [vac_xyz, disp_xyz], sia = vac + disp. Directly
targets the lone failing scorecard metric (vac-SIA separation). Same training recipe as
train_set_v2 (warmup EMA, count head, scorecard selection, results-store write); only the
representation/model differ.

    python scripts/train_paired.py --subset data/real_subset_full.npz \
        --output_dir runs/paired_v1 --epochs 500 --device mps
"""
import argparse
import importlib.util
import os
import time

import numpy as np


def _load_sp():
    spec = importlib.util.spec_from_file_location("sp_cas", os.path.join(os.getcwd(), "sp_cas.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--output_dir", default="runs/paired_v1")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--energy_divisor", type=float, default=300.0)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--select_every", type=int, default=100)
    ap.add_argument("--select_n", type=int, default=12)
    ap.add_argument("--finalize_n", type=int, default=8)
    ap.add_argument("--results", default="results")
    ap.add_argument("--no_results", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from cascaide.eval import io
    from cascaide.setdiff.ema import EMAWarmup
    from cascaide.setdiff.paired import (PairedSetDenoiser, build_pair_items,
                                         collate_pairs, sample_paired, save_paired)
    from cascaide.setdiff.select import score_generated
    sp = _load_sp()

    dev = args.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    samples = io.load_subset(args.subset)
    items, ce, cn, norm = build_pair_items(samples, energy_divisor=args.energy_divisor)
    ce, cn = ce.to(dev), cn.to(dev)
    print(f"[paired] {len(items)} non-empty cascades | device={dev} | "
          f"Gv={norm.Gv.round(1)} sv={norm.sv.round(1)} sd={norm.sd.round(2)}")

    tl = DataLoader(items, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    collate_fn=collate_pairs)
    den = PairedSetDenoiser(d_model=args.d_model, depth=args.depth, n_heads=args.n_heads).to(dev)
    ch = sp.CountHead().to(dev)
    diff = sp.CoordDiffusion(T=1000, schedule="cosine", device=dev)
    opt = torch.optim.AdamW(den.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 1e-6)
    opt_c = torch.optim.Adam(ch.parameters(), lr=1e-3)
    ema = None if args.no_ema else EMAWarmup(den, decay=args.ema_decay)
    print(f"[paired] EMA: {'off' if ema is None else f'warmup->{args.ema_decay}'}")

    ref = samples
    cfg = {"T": diff.T, "schedule": "cosine", "d_model": args.d_model, "depth": args.depth,
           "n_heads": args.n_heads, "count_cap": 400, "energy_divisor": args.energy_divisor}
    best = float("inf"); best_path = os.path.join(args.output_dir, "best_by_scorecard.pt")

    def save(path, epoch):
        save_paired(ema.shadow if ema is not None else den, ch, norm, cfg, path, epoch)

    for epoch in range(args.epochs):
        den.train(); t0 = time.time(); losses = []
        for tokens, mask, energy, _ in tl:
            tokens, mask, energy = tokens.to(dev), mask.to(dev), energy.to(dev)
            t = torch.randint(0, diff.T, (tokens.shape[0],), device=dev).long()
            noise = torch.randn_like(tokens)
            x_t = diff.q_sample(tokens, t, noise)
            eps = den(x_t, t, energy, key_padding_mask=~mask)
            m = mask.float()
            loss = (((eps - noise) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1.0)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0); opt.step()
            if ema is not None:
                ema.update(den)
            losses.append(float(loss.detach()))
        sched.step()

        ch.train()
        for _ in range(50):
            idx = torch.randint(0, len(ce), (min(512, len(ce)),), device=dev)
            opt_c.zero_grad(set_to_none=True); ch.nll(ce[idx], cn[idx]).backward(); opt_c.step()

        msg = f"[paired] ep {epoch+1}/{args.epochs} eps {np.mean(losses):.4f} {time.time()-t0:.1f}s"

        if (epoch + 1) % args.select_every == 0:
            gen_net = ema.shadow if ema is not None else den
            gen_net.eval(); gen = []
            with torch.no_grad():
                for e_keV in sorted({round(s["energy"]) for s in ref}):
                    e = torch.full((args.select_n,), e_keV / args.energy_divisor, device=dev)
                    npairs = ch.sample(e, cap=cfg["count_cap"])
                    for i in range(args.select_n):
                        npi = int(npairs[i])
                        if npi == 0:
                            gen.append({"vac": np.zeros((0, 3), np.float32),
                                        "sia": np.zeros((0, 3), np.float32), "energy": float(e_keV)})
                            continue
                        tok = sample_paired(gen_net, diff, e[i:i+1], npi, device=dev)[0].cpu().numpy()
                        vac, sia = norm.inverse(tok)
                        gen.append({"vac": vac, "sia": sia, "energy": float(e_keV)})
            score, _ = score_generated(gen, ref, label=f"ep{epoch+1}")
            flag = ""
            if score < best:
                best = score; save(best_path, epoch); flag = " *best*"
            msg += f" | scorecard {score:.3f}{flag}"
        print(msg, flush=True)

    save(os.path.join(args.output_dir, "final_model.pt"), args.epochs - 1)
    print(f"[paired] done. best scorecard {best:.3f} -> {best_path}", flush=True)

    if args.no_results:
        return
    try:
        import shutil
        from cascaide.eval import scorecard as scm, registry, render, report
        from cascaide.eval.generators import generate_paired_checkpoint
        desc = f"set-dit paired [{'EMA' if ema is not None else 'noEMA'}, {args.epochs}ep]"
        print(f"[paired] finalizing: {args.finalize_n}/energy from {best_path}...", flush=True)
        gen, meta = generate_paired_checkpoint(best_path, ref, device=dev, n_per_energy=args.finalize_n)
        sc = scm.compute(gen, ref, label=desc)
        manifest = registry.build_manifest(
            label=desc, arch="set-dit-paired", normalization="pairnorm-p99",
            checkpoint="model.pt", energies_keV=sc["energies_keV"],
            config={**cfg, "ema": ema is not None, "epochs": args.epochs},
            tags=["set-dit-paired", "pair-relative"] + (["ema"] if ema is not None else []))
        run_dir = registry.write_run(args.results, manifest, scorecard=sc)
        shutil.copy(best_path, os.path.join(run_dir, "model.pt"))
        render.render_comparison(gen, ref, os.path.join(run_dir, "samples"))
        registry.collect(args.results); report.build_report(args.results)
        print(f"[paired] results-store run + report + weights ({run_dir}/model.pt)", flush=True)
    except Exception as e:
        print(f"[paired] results-store write skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
