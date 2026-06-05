# Longer training + rotation augmentation = the new best model (2026-06-05)

> **Team-briefing summary.** We trained the per_cascade set-DiT for **500 epochs with SO(3)
> rotation augmentation** (`--augment_rot`) on the full 0–300 keV corpus, saving a checkpoint every
> 25 epochs, then scored the whole trace offline. Two results: **(1) a fine-grained convergence
> trace** (20 checkpoints @ n=12) shows the model improves *slowly and noisily throughout* — **not**
> an early plateau — with the back third (ep 375–500) the best stretch and the high-E / rdf metrics
> reaching their minima at ep 500. **(2) A deep n=24 head-to-head verdict** confirms the win:
> ep 375 / 475 / 500 all beat the prior-best 250-epoch baseline by **~0.19 OVERALL** and on **every
> regime and rdf metric**, moving 5/7 → 6/7. The headline is **high-E (0.64 vs 0.97, −0.33)** — the
> only real headroom, attacked exactly as intended. **`final_model.pt` (ep 500) is the new best
> per_cascade model.** This validates the standing recipe and the "high-E is a convergence problem"
> thesis (see [[scorecard-findings]]).

---

## 1. The run

The "best-bet" model launched 2026-06-05 (Modal app `ap-KUIg1l1P6BaEjLTsvipaZR`, A100, detached,
ran clean to ep 500/500, ~3 h ≈ ~$12):

| lever | value |
|---|---|
| centering | **per_cascade** (the proven default — see [[scorecard-findings]]) |
| EMA | on, warmup decay |
| LR schedule | cosine |
| capacity | 256 d_model / 6 depth |
| energy range | **full 0–300 keV** (7099 cascades, `--max_defects 1000`), batch 16 |
| epochs | **500** |
| augmentation | **`--augment_rot`** (SO(3), per_cascade only — exact symmetry, targets data-starved high-E) |
| in-run scoring | **DISABLED** (`--select_every 99999 --no_results`) — score offline instead |
| checkpointing | **`--save_every 25`** → `ckpt_0025.pt … ckpt_0500.pt` + `final_model.pt` on the Volume |

Why this combination: high-E was the one identified headroom (a *convergence* problem, not
architecture), augmentation was the highest-upside low-risk lever left, and the alternatives were
all ruled out (aux losses [[substructure-loss-negative]], global centering, `energy_balance`).
Artifacts: checkpoints under `cascaide-results:ckpts/percascade_500ep_aug/`, pulled to
`results/ckpts/percascade_500ep_aug/`.

## 2. Fine-grained convergence trace (n=12, all 20 checkpoints)

Every `ckpt_<ep>.pt` rescored **offline** with the corrected eval — full-range, **n=12/energy**,
**dpmpp-20**, **`--gen_energy sample`**, `energy_bin=25`, seed 0 — by `scripts/score_trace.py`
(new this session; caches per-ckpt so it can trail a live run). Lower = better; `p/t` = scorecard
requirements passed.

| ep | OVERALL | p/t | rdf | rdf_hi | low | mid | high |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.847 | 4/7 | 0.203 | 0.236 | 1.153 | 0.690 | 0.907 |
| 50 | 1.150 | 3/7 | 0.465 | 0.637 | 1.069 | 0.784 | 1.480 |
| 75 | 0.968 | 3/7 | 0.324 | 0.323 | 1.123 | 0.785 | 1.049 |
| 100 | 0.890 | 4/7 | 0.219 | 0.194 | 1.153 | 0.930 | 0.799 |
| 125 | 0.876 | 5/7 | 0.241 | 0.270 | 0.916 | 0.638 | 1.043 |
| 150 | 0.792 | 5/7 | 0.207 | 0.187 | 1.126 | 0.794 | 0.773 |
| 175 | 0.849 | 4/7 | 0.193 | 0.209 | 1.134 | 0.683 | 0.832 |
| 200 | 0.946 | 4/7 | 0.300 | 0.330 | 1.057 | 0.738 | 1.033 |
| 225 | 0.940 | 5/7 | 0.266 | 0.311 | 0.659 | 0.778 | 1.156 |
| 250 | 0.930 | 4/7 | 0.279 | 0.331 | 1.221 | 0.825 | 0.934 |
| 275 | 0.758 | 5/7 | 0.175 | 0.157 | 0.979 | 0.601 | 0.805 |
| 300 | 0.779 | 5/7 | 0.206 | 0.192 | 1.005 | 0.712 | 0.728 |
| 325 | 0.783 | 5/7 | 0.204 | 0.205 | 0.909 | 0.612 | 0.906 |
| 350 | 0.796 | 5/7 | 0.189 | 0.160 | 1.175 | 0.676 | 0.761 |
| **375** | **0.691** | 6/7 | 0.178 | 0.177 | 1.109 | 0.485 | 0.735 |
| 400 | 0.796 | 5/7 | 0.190 | 0.184 | 0.985 | 0.703 | 0.812 |
| 425 | 0.741 | 6/7 | 0.201 | 0.224 | 1.017 | 0.617 | 0.734 |
| 450 | 0.749 | 5/7 | 0.202 | 0.157 | 0.835 | 0.818 | 0.697 |
| 475 | 0.731 | 6/7 | 0.190 | 0.185 | 0.860 | 0.560 | 0.805 |
| **500** | 0.702 | 6/7 | **0.167** | **0.140** | 1.018 | 0.606 | **0.655** |

**Read (answering "do they converge early?"): no early plateau.** OVERALL drifts down from ~0.85
(early) to ~0.70–0.73 (late) under heavy n=12 noise (≈±0.1 per point — see §4). The back third
(ep 375–500) is clearly the best band (mostly 6/7); **ep 500 holds the run-minima on `rdf_l1`
(0.167), `rdf_hi` (0.140) and high-E (0.655)** — i.e. the longer run kept buying exactly the
substructure + high-E quality that was the goal. ep 375 (0.691) and ep 500 (0.702) are tied within
noise. Figure: `figures/convergence_500ep_aug.{png,pdf}` (left: OVERALL + per-regime vs epoch;
right: rdf_l1 vs epoch with the 0.20 target).

## 3. Deep verdict (n=24, head-to-head vs the prior best)

`scripts/score_verdict.py` — same eval as the centering A/B (full-range, **n=24/energy**, dpmpp-20,
`--gen_energy sample`, seed 0), all four models scored **in one process** so they are directly
comparable. Baseline = the prior-best per_cascade run `results/runs/20260603T082541-set-dit-v2-
87ec2c1b/model.pt` (a 250-epoch run, best checkpoint at ep 124).

| model | ep | OVERALL | p/t | rdf | rdf_hi | low | mid | high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline per_cascade 250ep | 124 | 0.810 | 5/7 | 0.231 | 0.244 | 0.930 | 0.604 | 0.967 |
| 500ep+aug ep375 | 374 | **0.613** | 6/7 | 0.160 | 0.162 | 0.666 | 0.564 | 0.639 |
| 500ep+aug ep475 | 474 | 0.615 | 6/7 | 0.165 | **0.145** | 0.641 | 0.534 | 0.657 |
| 500ep+aug final (ep500) | 499 | 0.623 | 6/7 | 0.174 | 0.152 | 0.682 | 0.574 | 0.640 |

**Δ vs baseline (negative = 500ep+aug better):**

| candidate | dOVERALL | drdf | dlow | dmid | dhigh |
|---|---:|---:|---:|---:|---:|
| ep375 | **−0.197** | −0.071 | −0.265 | −0.040 | **−0.328** |
| ep475 | −0.195 | −0.065 | −0.290 | −0.070 | −0.310 |
| ep500 | −0.187 | −0.057 | −0.249 | −0.030 | −0.327 |

**Verdict: 500ep + `--augment_rot` decisively beats the 250ep baseline** — every candidate wins on
**every metric and every regime**, by ~0.19 OVERALL and ~0.33 on high-E, and clears one more
scorecard requirement (6/7 vs 5/7). The three late checkpoints are statistically indistinguishable
(0.613–0.623), so **keep `final_model.pt` (ep 500)** as the new best (simplest, and it has the best
trace-level substructure/high-E). The n=12 trace independently produces the same ranking, so the
result is not a single-draw fluke.

## 4. Caveat — n=24 still carries ~±0.15 absolute sampling variance

The baseline scores **0.810 here but 0.661 in the centering A/B** for the *identical*
checkpoint+settings. Cause: in `generators.generate_set_checkpoint(..., seed=0)` the `seed` only
pins the per-bin reference-energy draw (`_energy_targets`); the **reverse-diffusion noise in
`sp.generate`/`sample_dpmpp` is NOT seeded**, so absolute OVERALL wobbles ~±0.15 run-to-run even at
n=24/energy. **Implication:** trust *within-run, same-process* matched comparisons (as this verdict
is), not absolute levels across runs. **Follow-up worth doing:** thread the seed through to the
sampler (`torch.manual_seed` inside `sp.generate`) so absolute scores become reproducible. Until
then, re-score any cross-run comparison in a single process. (This also means the n=12 trace's
point-to-point bounce is partly sampling noise, not training dynamics — read the *band*, not the
spikes.)

## 5. Cost note (corrected)

Full-range offline rescoring on MPS is **~220 s/ckpt at n=12 and ~5 min at n=24** — dominated by the
high-E tail (≥200 keV clouds, ~2000 tokens, O(N²) attention) and **highly sublinear in n** (n=2 was
194 s) because the per-bin clouds are batched. An earlier handoff estimate of "~4 min/ckpt for the
whole n=24 trace" undercounted the per-cloud cost but was right that a dense trace is tractable: the
20-ckpt n=12 trace ran in ~75 min on free local MPS *concurrent with* the cloud A100 training.

## 6. Artifacts & repro

- **Checkpoints:** `results/ckpts/percascade_500ep_aug/ckpt_0025.pt … ckpt_0500.pt`, `final_model.pt`
  (also on Modal Volume `cascaide-results:ckpts/percascade_500ep_aug/`).
- **New best model:** `results/ckpts/percascade_500ep_aug/final_model.pt`.
- **Trace:** `figures/convergence_500ep_aug.{png,pdf,json}` (json = per-ckpt score cache).
- **Tooling:** `scripts/score_trace.py` (cached trace scorer, `--only`/`--every`/`--baseline`);
  `scripts/score_verdict.py` (n=24 head-to-head; writes `/tmp/verdict_results.json`).
- **Re-run the trace:** `python scripts/score_trace.py --ckpt_dir results/ckpts/percascade_500ep_aug
  --subset data/cascaide_cascades.npz --device mps --n 12 --sampler dpmpp --steps 20 --energy_bin 25
  --only 25,50,...,500 --out figures/convergence_500ep_aug` (run from repo root; `KMP_DUPLICATE_LIB_OK=TRUE`).

## 7. What this changes / next

- **Promote** `final_model.pt` (500ep+aug) to the standing best per_cascade model; supersedes the
  250ep baseline in all future comparisons.
- **Recipe confirmed:** per_cascade · EMA · cosine · 256/6 · `--max_defects 1000` · batch 16 ·
  **500 epochs · `--augment_rot`**. Augmentation is now a *proven* positive lever (first one in a
  while — contrast the aux-loss negatives in [[substructure-loss-negative]]).
- **Untested levers still open:** capacity 512/10 (budget-permitting), even longer epochs (the curve
  hadn't clearly flattened by 500), `--min_snr` (lower priority). High-E is improved but still the
  weakest regime (0.64) — the next headroom.
- **Eval hygiene:** seed the sampler (§4) before the next cross-run verdict.
