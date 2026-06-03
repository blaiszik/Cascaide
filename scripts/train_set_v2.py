#!/usr/bin/env python
"""Set/point diffusion — next iteration. Composes the learnings from 2026-06-02:

  (1) substructure loss   — differentiable pairwise-distance-histogram + NN-spacing loss on
                            the predicted clean coords (low-t gated). Targets the actual gap
                            (radial is already solved; this fixes over-merging).
  (3) scorecard selection — periodically generate from the live model and keep the
                            checkpoint with the best SCORECARD (not val MSE, which plateaus).
  (4) HPC-ready           — on finish, generate + render + score + write a run to the results
                            store and rebuild the report, so cluster runs feed the dashboard.

Reuses the sp_cas.py reference model (SetDenoiser / CoordDiffusion / CountHead) so the
checkpoint loads with the existing generators. Pair-relative encoding (2) utilities live in
cascaide.setdiff.pairing; wiring the 6-D paired model is the follow-on (see handoff).

    python scripts/train_set_v2.py --subset data/real_subset_large.npz \
        --output_dir runs/set_v2 --epochs 300 --w_struct 0.5 --device mps
"""
import argparse
import copy
import importlib.util
import os
import time

import numpy as np


class EMAWarmup:
    """EMA of a model with a step-count WARMUP so the shadow isn't dominated by the random
    init early in training. (Bug seen 2026-06-02: a fixed decay=0.9999 with only ~9k steps
    by ep100 left the shadow ~40% initialization → generations scored ~6.)

    Effective decay grows with step: d_t = min(target, (1+step)/(10+step)). Early on d is
    small (shadow tracks the live model); it asymptotes to ``target``. This makes EMA robust
    to run length without hand-tuning the decay. Exposes ``.shadow`` like the simple EMA.
    """
    def __init__(self, model, decay=0.9999, warmup=True):
        import torch
        self.target = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._torch = torch

    def update(self, model):
        self.step += 1
        d = min(self.target, (1 + self.step) / (10 + self.step)) if self.warmup else self.target
        with self._torch.no_grad():
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.data.mul_(d).add_(p.data, alpha=1 - d)


def _load_sp():
    spec = importlib.util.spec_from_file_location("sp_cas", os.path.join(os.getcwd(), "sp_cas.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--output_dir", default="runs/set_v2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_sched", choices=["cosine", "cosine_warmup", "onecycle", "plateau"],
                    default="cosine", help="LR schedule for the denoiser (count head is unscheduled)")
    ap.add_argument("--warmup_frac", type=float, default=0.05,
                    help="warmup as a fraction of epochs (cosine_warmup)")
    ap.add_argument("--lr_max", type=float, default=None,
                    help="peak LR for onecycle (default 4*lr — super-convergence wants a high peak)")
    ap.add_argument("--plateau_factor", type=float, default=0.5, help="LR drop factor (plateau)")
    ap.add_argument("--plateau_patience", type=int, default=3,
                    help="plateau patience in #scorecard evals (i.e. plateau steps on the scorecard, "
                         "the true objective, every select_every epochs)")
    ap.add_argument("--energy_divisor", type=float, default=300.0)
    # per-cascade centering (required for mixed-supercell raw_data) + energy subsetting/binning
    ap.add_argument("--center", choices=["per_cascade", "global"], default="per_cascade")
    ap.add_argument("--augment_rot", action="store_true",
                    help="SO(3) rotation augmentation (per_cascade only): rotate each centered "
                         "cloud by a fresh random rotation every epoch — free data multiplication "
                         "+ enforces orientation invariance the set-DiT lacks")
    ap.add_argument("--energy_max", type=float, default=None, help="keep cascades <= this keV")
    ap.add_argument("--max_defects", type=int, default=None,
                    help="drop cascades with max(n_vac,n_sia) > this (memory cap for O(N^2) attention)")
    ap.add_argument("--energy_bin", type=float, default=None,
                    help="bin width (keV) for scoring continuous energies, e.g. 10")
    ap.add_argument("--count_cap", type=int, default=1300)
    # EMA (on by default — the single highest-confidence quality lever)
    ap.add_argument("--no_ema", action="store_true", help="disable EMA of the denoiser")
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    # Min-SNR-gamma loss weighting (ready, default OFF so it doesn't confound the EMA run)
    ap.add_argument("--min_snr", action="store_true", help="enable Min-SNR-gamma loss weighting")
    ap.add_argument("--min_snr_gamma", type=float, default=5.0)
    # substructure loss — DEMOTED: off by default (the sweep showed it hurts at w>=0.3).
    # If revisiting, use a tiny weight as a fine-tune on a converged checkpoint.
    ap.add_argument("--w_struct", type=float, default=0.0)
    ap.add_argument("--struct_t_frac", type=float, default=0.25, help="fire only for t<frac*T")
    ap.add_argument("--struct_rmax", type=float, default=6.0)
    ap.add_argument("--struct_start_epoch", type=int, default=20)
    # scorecard selection
    ap.add_argument("--select_every", type=int, default=20)
    ap.add_argument("--select_n", type=int, default=12, help="gen samples/energy for scoring")
    ap.add_argument("--results", default="results")
    ap.add_argument("--finalize_n", type=int, default=8,
                    help="samples/energy for the final results-store run (T=1000 sampling is slow)")
    ap.add_argument("--no_results", action="store_true",
                    help="skip the final generate+score+write-to-results-store block")
    ap.add_argument("--no_report", action="store_true",
                    help="write the run dir but skip collect()/build_report() (for parallel "
                         "sweeps; rebuild the index+report once after the job)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from cascaide.eval import io
    from cascaide.setdiff.losses import substructure_loss
    from cascaide.setdiff.select import score_generated
    sp = _load_sp()

    dev = args.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    samples = io.load_subset(args.subset)
    if args.energy_max is not None:
        samples = [s for s in samples if s["energy"] <= args.energy_max]
    if args.max_defects is not None:
        before = len(samples)
        samples = [s for s in samples
                   if max(len(s["vac"]), len(s["sia"])) <= args.max_defects]
        print(f"[v2] max_defects={args.max_defects}: dropped {before - len(samples)} "
              f"large cascades ({len(samples)} remain)")

    if args.center == "per_cascade":
        from cascaide.setdiff.data import PerCascadeNormalizer, SetDataset
        norm = PerCascadeNormalizer().fit(samples)
        ds = SetDataset(samples, norm, args.energy_divisor, augment_rot=args.augment_rot)
        print(f"[v2] {len(ds)} cascades (<= {args.energy_max} keV) | device={dev} | "
              f"center=per_cascade | s={norm.s.round(1)} | augment_rot={args.augment_rot}")
    else:
        if args.augment_rot:
            print("[v2] WARNING: --augment_rot is only wired for --center per_cascade; ignored.")
        raw = [(s["vac"], s["sia"]) for s in samples]
        energies = [s["energy"] for s in samples]

        class _DS(sp._PairDatasetBase):
            def __init__(self):
                self._finalize(raw, energies, normalizer=None,
                               energy_divisor=args.energy_divisor)
        ds = _DS()
        norm = ds.normalizer
        print(f"[v2] {len(ds)} cascades | device={dev} | center=global | "
              f"G={norm.G.round(1)} s={norm.s.round(1)}")

    tl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    collate_fn=sp.collate_dynamic)
    den = sp.SetDenoiser(d_model=args.d_model, depth=args.depth, n_heads=args.n_heads).to(dev)
    ch = sp.CountHead().to(dev)
    diff = sp.CoordDiffusion(T=1000, schedule="cosine", device=dev)
    opt = torch.optim.AdamW(den.parameters(), lr=args.lr, weight_decay=1e-2)
    # LR scheduler. step cadence differs by type: onecycle steps PER BATCH, cosine/warmup PER
    # EPOCH, plateau steps on the SCORECARD (the true objective) each time it is evaluated.
    L = torch.optim.lr_scheduler
    steps_per_epoch = max(1, len(tl))
    if args.lr_sched == "cosine":
        sched = L.CosineAnnealingLR(opt, args.epochs, 1e-6); sched_step = "epoch"
    elif args.lr_sched == "cosine_warmup":
        wu = max(1, round(args.warmup_frac * args.epochs))
        sched = L.SequentialLR(opt, [L.LinearLR(opt, start_factor=0.01, total_iters=wu),
                                     L.CosineAnnealingLR(opt, max(1, args.epochs - wu), 1e-6)],
                               milestones=[wu]); sched_step = "epoch"
    elif args.lr_sched == "onecycle":
        sched = L.OneCycleLR(opt, max_lr=(args.lr_max or 4 * args.lr),
                             epochs=args.epochs, steps_per_epoch=steps_per_epoch); sched_step = "batch"
    else:  # plateau — adaptive on the scorecard (lower = better)
        sched = L.ReduceLROnPlateau(opt, mode="min", factor=args.plateau_factor,
                                    patience=args.plateau_patience); sched_step = "plateau"
    print(f"[v2] lr_sched={args.lr_sched} (steps per {sched_step}) base_lr={args.lr}"
          + (f" max_lr={args.lr_max or 4*args.lr}" if args.lr_sched == "onecycle" else ""))
    opt_c = torch.optim.Adam(ch.parameters(), lr=1e-3)
    ce = ds.count_energy.to(dev); cn = ds.count_npairs.to(dev)

    # EMA (with warmup) of the denoiser — generate/select/save use the EMA weights.
    # Warmup keeps the shadow from being init-dominated on short runs (see EMAWarmup).
    ema = None if args.no_ema else EMAWarmup(den, decay=args.ema_decay, warmup=True)
    print(f"[v2] EMA: {'off' if ema is None else f'warmup->{args.ema_decay}'}")

    # reference for scorecard selection (a sample of the real distribution)
    ref = samples
    t_max = int(diff.T * args.struct_t_frac)
    best_score = float("inf"); best_path = os.path.join(args.output_dir, "best_by_scorecard.pt")

    def save(path, epoch):
        # save the EMA weights (what we deploy) as the denoiser
        net = ema.shadow if ema is not None else den
        sp._save(net, ch, norm, diff.T, "cosine", args.d_model, args.depth, args.n_heads,
                 args.count_cap, args.energy_divisor, path, epoch)

    for epoch in range(args.epochs):
        den.train(); t0 = time.time(); eps_l, st_l = [], []
        for coords, types, mask, energy, _ in tl:
            coords, types = coords.to(dev), types.to(dev)
            mask, energy = mask.to(dev), energy.to(dev)
            t = torch.randint(0, diff.T, (coords.shape[0],), device=dev).long()
            noise = torch.randn_like(coords)
            x_t = diff.q_sample(coords, t, noise)
            eps_pred = den(x_t, t, energy, types, key_padding_mask=~mask)
            m = mask.float()
            per_tok = ((eps_pred - noise) ** 2).mean(-1)                  # (B, N)
            token_mse = (per_tok * m).sum() / m.sum().clamp(min=1.0)      # raw, for logging
            if args.min_snr:
                # Min-SNR-gamma weight for eps-prediction: min(SNR,gamma)/SNR, SNR=ab/(1-ab)
                ab = diff.sqrt_ab.gather(0, t) ** 2                       # alpha_bar_t (B,)
                om = (diff.sqrt_1m_ab.gather(0, t) ** 2).clamp(min=1e-8)
                snr = ab / om
                w = torch.clamp(snr, max=args.min_snr_gamma) / snr.clamp(min=1e-8)
                per_cloud = (per_tok * m).sum(1) / m.sum(1).clamp(min=1.0)  # (B,)
                eps_mse = (per_cloud * w).mean()
            else:
                eps_mse = token_mse

            loss = eps_mse
            if args.w_struct > 0 and epoch >= args.struct_start_epoch:
                low = t < t_max
                if int(low.sum()) > 0:
                    # grad-enabled predicted clean coords (sp's predict_x0 is no_grad)
                    x0 = (diff._ext(diff.sqrt_recip_ab, t, x_t.shape) * x_t
                          - diff._ext(diff.sqrt_recipm1_ab, t, x_t.shape) * eps_pred)
                    sl, sinfo = substructure_loss(x0[low], coords[low], mask[low],
                                                  rmax=args.struct_rmax)
                    loss = loss + args.w_struct * sl
                    st_l.append(sinfo["struct_hist"] + sinfo["struct_nn"])
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0); opt.step()
            if sched_step == "batch":      # onecycle anneals per optimizer step
                sched.step()
            if ema is not None:
                ema.update(den)
            eps_l.append(float(token_mse.detach()))   # log raw token MSE (comparable across runs)
        if sched_step == "epoch":
            sched.step()

        # count head
        ch.train()
        for _ in range(50):
            idx = torch.randint(0, len(ce), (min(512, len(ce)),), device=dev)
            opt_c.zero_grad(set_to_none=True); ch.nll(ce[idx], cn[idx]).backward(); opt_c.step()

        msg = (f"[v2] ep {epoch+1}/{args.epochs} eps {np.mean(eps_l):.4f} "
               f"struct {np.mean(st_l) if st_l else 0:.4f} {time.time()-t0:.1f}s")

        # scorecard-based selection (generate from the EMA weights)
        if (epoch + 1) % args.select_every == 0:
            gen_net = ema.shadow if ema is not None else den
            gen_net.eval()
            gen = []
            ebin = (lambda x: round(x / args.energy_bin) * args.energy_bin) if args.energy_bin else round
            with torch.no_grad():
                for e_keV in sorted({ebin(s['energy']) for s in ref}):
                    e = torch.full((args.select_n,), e_keV / args.energy_divisor, device=dev)
                    npairs = ch.sample(e, cap=args.count_cap)
                    for i in range(args.select_n):
                        npi = int(npairs[i])
                        if npi == 0:
                            gen.append({"vac": np.zeros((0, 3), np.float32),
                                        "sia": np.zeros((0, 3), np.float32), "energy": float(e_keV)})
                            continue
                        types = torch.cat([torch.zeros(npi, dtype=torch.long),
                                           torch.ones(npi, dtype=torch.long)])[None].to(dev)
                        c = diff.sample(gen_net, e[i:i+1], types)[0].cpu().numpy()
                        c = norm.inverse(c)
                        gen.append({"vac": c[:npi], "sia": c[npi:], "energy": float(e_keV)})
            score, _ = score_generated(gen, ref, label=f"ep{epoch+1}", energy_bin=args.energy_bin)
            if sched_step == "plateau":    # adaptive: drop LR when the scorecard stops improving
                sched.step(score)
            flag = ""
            if score < best_score:
                best_score = score; save(best_path, epoch); flag = " *best*"
            msg += f" | scorecard {score:.3f}{flag}"
        msg += f" | lr {opt.param_groups[0]['lr']:.2e}"
        print(msg, flush=True)

    save(os.path.join(args.output_dir, "final_model.pt"), args.epochs - 1)
    print(f"[v2] done. best scorecard {best_score:.3f} -> {best_path}", flush=True)

    if args.no_results:
        print("[v2] --no_results: skipping results-store write.", flush=True)
        return

    # HPC-ready: write the best checkpoint's run into the results store + rebuild report.
    # Keep this light + instrumented: full T=1000 sampling is slow, so cap samples/energy
    # and print progress (a silent, heavy finalize looks like a hang — see 2026-06-02).
    try:
        from cascaide.eval import scorecard as scm, registry, render, report
        from cascaide.eval.generators import generate_set_checkpoint
        n_fin = args.finalize_n
        print(f"[v2] finalizing: generating {n_fin}/energy from {best_path} "
              f"(T={diff.T} sampling, may take a few min)...", flush=True)
        gen, meta = generate_set_checkpoint(best_path, ref, device=dev, n_per_energy=n_fin,
                                            energy_bin=args.energy_bin)
        print(f"[v2] generated {len(gen)} samples; scoring + rendering...", flush=True)
        import shutil
        emax = f", <{int(args.energy_max)}keV" if args.energy_max else ""
        desc = (f"set-dit v2 [{'EMA' if ema is not None else 'noEMA'}"
                f"{', minSNR' if args.min_snr else ''}{', rot-aug' if args.augment_rot else ''}"
                f", {args.lr_sched}, {args.center}, {args.epochs}ep{emax}]")
        tags = ["set-dit-v2", args.center, f"sched:{args.lr_sched}"] \
            + (["ema"] if ema is not None else []) \
            + (["min-snr"] if args.min_snr else []) \
            + (["augment-rot"] if args.augment_rot else []) \
            + (["substructure-loss"] if args.w_struct > 0 else [])
        sc = scm.compute(gen, ref, label=desc, energy_bin=args.energy_bin)
        manifest = registry.build_manifest(
            label=desc, arch="set-dit-v2",
            normalization=("percascade-p99" if args.center == "per_cascade" else "coordnorm-p99"),
            checkpoint="model.pt", energies_keV=sc["energies_keV"],  # co-located (below)
            config={"w_struct": args.w_struct, "epochs": args.epochs,
                    "d_model": args.d_model, "depth": args.depth, "center": args.center,
                    "energy_max": args.energy_max, "energy_bin": args.energy_bin,
                    "ema": ema is not None, "min_snr": args.min_snr,
                    "lr_sched": args.lr_sched, "lr": args.lr, "augment_rot": args.augment_rot},
            tags=tags)
        run_dir = registry.write_run(args.results, manifest, scorecard=sc)
        # co-locate the trained weights INSIDE the run dir so the run is self-contained
        # (this is what makes the checkpoint travel back from Modal/HPC; see 2026-06-02).
        shutil.copy(best_path, os.path.join(run_dir, "model.pt"))
        render.render_comparison(gen, ref, os.path.join(run_dir, "samples"))
        if args.no_report:
            # parallel sweep: write the run dir only; rebuild index/report ONCE after the job
            # (concurrent collect()/build_report() would race on index.json / report.html)
            print(f"[v2] run written (no_report) {run_dir}/model.pt", flush=True)
        else:
            registry.collect(args.results); report.build_report(args.results)
            print(f"[v2] results-store run + report + weights ({run_dir}/model.pt)", flush=True)
    except Exception as e:
        print(f"[v2] results-store write skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
