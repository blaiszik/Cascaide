# What the scorecards tell us about the models (2026-06-04)

> Mined the 15 per-energy-regime scorecards pulled from Eagle (the 2026-06-04 A100 batch:
> `checkpoints_lt100.txt` n=48 + `checkpoints_full.txt` n=12). Two outcomes: (1) an **eval
> bug** that inflated the low-energy regime for *every* model, now fixed; (2) a clear read on
> the models' **one remaining weakness — pairwise substructure (RDF)** — which sets the next
> experiments. Numbers are screening-grade (n=12 full-range is noisy); the cross-model
> *patterns* are robust because they repeat across models and at n=48.

## TL;DR

1. **High-E is a convergence problem, not architecture.** The converged 250-epoch full-range
   per_cascade/EMA model scores **high-regime = 0.473** vs **2.3–3.6** for the un-converged
   `set-dit-v2` baselines on the identical regime. Best model overall = **0.656**. Validates
   Wave-1 cell A (train longer). Cell D (bigger model) must beat 0.473 to justify capacity.
2. **The low-regime "failure" was a SCORECARD ARTIFACT, now fixed.** See below — fixing it
   *halves* the low-regime score (1.48 → 0.69) with no model change.
3. **The one genuine model weakness is pairwise substructure (RDF).** Every other metric
   (radial, count, cluster, nn, conservation) is near-ceiling; `rdf_l1 ≈ 0.13–0.41` is an
   order of magnitude worse. This is the project's stated priority and the only lever left.
4. **per_cascade centering beats global centering ~3× on substructure** — and it's a
   normalization effect, not undertraining.

---

## The bin-0 eval artifact (fixed in this commit)

The scorer bins reference clouds by `key = round(e/bin)*bin` and **conditioned generation at
the bin center**. For every bin except the lowest, the center is the bin's midpoint, so it
matches the reference window's mean energy. But **bin-0 = `[0, 12.5)` keV has its center at
the left edge (0 keV ≈ no cascade)** — energies can't go negative — while its reference window
averages **5.55 keV**. So the scorer generated ~0-keV clouds (≈3 pairs) and compared them to
real ~5.5-keV clouds (≈7.7 pairs), reading as a 2.4× undercount.

This is **not a count-head bug.** The head is accurate:

| query | head → pairs | reference |
|---|---|---|
| E = 0.00 keV (what the scorer used) | 3.16 | — |
| E = 5.55 keV (bin-0's true mean) | **7.44** | 7.67 ✓ |

Within bin-0 the real pairs go 2.95 → 6.30 → 8.82 → 11.28 across `[0,2)→[2,5)→[5,8)→[8,12.5)`
keV and the head tracks it almost exactly. "Worsens with more training" (30ep gen=10.1 →
569ep gen=2.7 at E=0) is the head becoming *more* correct about the E→0 limit (≈0 defects),
which the artifact then penalizes. The same mismatch inflated bin-0's `nn_w1`, `cluster_l1`,
and `vac_sia_sep` (3-pair vs 7.7-pair clouds), so it contaminated the whole **low** regime —
for *every* model. The low-regime ranking we pulled is therefore partly an eval bug.

### The fix — energy-matched generation (`--gen_energy`)

`scripts/score_checkpoint.py --gen_energy {sample|bin_mean|center}` (default now **`sample`**):
- **`sample`** (default, gold standard): condition each generated cloud on a *real reference
  energy* drawn from its bin → the generated set's within-bin energy distribution equals the
  reference's. (`sp_cas.generate` now accepts a per-sample energy vector; this is free.)
- **`bin_mean`**: condition on the bin's mean reference energy (one energy/bin).
- **`center`**: the LEGACY behavior; kept to reproduce pre-2026-06-04 scores.

A/B on the 250ep model, low regime, dpmpp-20 held constant (only `gen_energy` differs):

| | LOW score | bin-0 count_mape | bin-0 nn_w1 | bin-0 cluster_l1 |
|---|---|---|---|---|
| `center` (legacy) | **1.480** (2/7) | 0.625 (5.8 vs 15.3) | 5.42 | 0.413 |
| `sample` (fixed) | **0.693** (5/7) | **0.032** (15.8 vs 15.3) | 1.36 | 0.111 |

The 250ep model's low regime (0.69) is now in line with its mid (0.76) and high (0.47) — it
was never a low-E *model* problem. **All regime comparisons should be re-run with `sample`
before being trusted; the `gen_energy` mode is recorded in each run's manifest config.**

---

## Per-metric read (n=48 `<100keV` models, mean over bins 25/50/75 — low-noise, excludes bin-0)

| model | rad_js | **rdf_l1** | clus_l1 | nn_w1 | vss_err | cnt_mape |
|---|---|---|---|---|---|---|
| per_cascade 300ep | 0.003 | **0.131** | 0.039 | 0.54 | 1.18 | 0.008 |
| per_cascade cosine | 0.003 | **0.138** | 0.037 | 0.53 | 1.81 | 0.019 |
| per_cascade 30ep | 0.005 | **0.192** | 0.050 | 0.70 | 1.45 | 0.036 |
| vignesh GLOBAL-center ep569 | 0.005 | **0.407** | 0.066 | 1.02 | 0.55 | 0.037 |

- **RDF (pairwise-distance substructure) is the only metric with headroom.** Everything else
  is at the ceiling. RDF = the histogram of inter-defect distances = the substructure the
  project cares about. → the `pairwise_hist_loss` (`w_struct`, `cascaide/setdiff/losses.py`)
  is the correct, targeted lever.
- **Radial density is solved (`rad_js ≈ 0.004`).** This is *why the radial-density aux loss
  regressed quality*: zero headroom on radial, so the term only added gradient conflict with
  the diffusion objective. (Number for the earlier "radial loss hurt" finding.)
- **per_cascade > global for substructure, ~3×.** vignesh is global-center AND fully converged
  (ep569), yet `rdf_l1` = 0.407 vs 0.131 — so it's the *normalization*, not undertraining.
  Global centering scales every cloud by a dataset-wide factor (dominated by big high-E
  clouds), washing out per-cascade pairwise geometry. **Trade-off:** global *wins* on absolute
  vac–sia separation (`vss` 0.55 vs ~1.5). Caveat: n=1 on the global side → a clean
  per_cascade-vs-global A/B (same config) would settle it. Directly relevant: we are on the
  `tanhp99-global-centering` branch.
- **Spatial fidelity is convergence.** Un-converged `set-dit-v2` baselines over-separate
  Frenkel pairs (`vac_sia_sep` gen≈22 vs real≈15) and produce too-sparse clouds (`nn`
  gen>real) at *all* energies; the converged per_cascade/EMA models fix it. Same mechanism as
  the high-E score collapse.
- **Count & conservation are free.** Conservation violation = 0 everywhere (n_vac==n_sia is
  hard-wired). `count_mape < 0.04` in the bulk; `count_w1` grows at high E only because counts
  are larger (250 keV mean 242, max 602; 300 keV max 1107) — the head models the heavy tail
  well in relative terms.

---

## What this changes for the experiment plan

- **The target metric is now RDF/substructure**, not high-E (convergence handles high-E) and
  not low-E (that was the eval bug). Next sweep should optimize `rdf_l1` directly.
- **Wave-1 cell B (`--energy_balance`) is de-motivated**: it was justified by high-E data
  starvation, which the 250ep result disproves, and the low-E bin isn't rare. Consider
  dropping/repurposing B toward a substructure lever.
- **Re-score the backlog with `--gen_energy sample`** before drawing low-regime conclusions.
- A clean **per_cascade vs global** A/B is now well-motivated by a concrete 3× RDF gap.
