# Substructure (RDF) auxiliary loss — a clean NEGATIVE result (2026-06-04)

> **Team-briefing summary.** We hypothesized that the model's pairwise-distance substructure
> (the scorecard's `rdf_l1`) could be improved by adding a differentiable g(r)-matching loss to
> training. After fixing two methodological problems and running a clean, converged, equal-epoch
> sweep on Polaris, the answer is unambiguous: **the loss hurts every metric — including
> `rdf_l1` itself — monotonically with its weight.** Moreover, the premise was off: the
> converged per-cascade `<100 keV` model **already passes the entire scorecard (7/7), with
> `rdf_l1` inside its target.** We are dropping this line of work. This is the *second*
> hand-crafted coordinate loss to degrade the model (the radial-density loss was the first),
> which points to a general lesson about this architecture.

---

## 1. The hypothesis and why it was reasonable

Mining the 2026-06-04 scorecards (see `scorecard-findings.md`), `rdf_l1` — the relative-L1
distance between the generated and real radial distribution functions g(r), i.e. the *pairwise
inter-defect spacing distribution* — was the **largest-magnitude** metric (0.13–0.41 across
models) while every other metric sat near its ceiling. g(r) *is* the "substructure" the project
cares about, and `rdf_l1` carries the scorecard's **heaviest weight (2.0)**. So targeting it
with a differentiable g(r) loss looked like the highest-value lever.

There was prior history: an earlier `--w_struct` sweep had been **demoted** ("hurts at w≥0.3").
We believed that earlier result was a *mis-configuration*, not a real negative — and we were
right about the misconfiguration, but wrong about the conclusion.

## 2. Two methodological problems we fixed first

**(a) Range mismatch (why the earlier sweep failed).** The legacy `pairwise_hist_loss` matched
the pairwise-distance histogram on a **fixed `rmax = 6 Å`** window. But in the real data,
nearest-neighbor distances are already **7–11 Å** and the *mean* pairwise distance is
**30–96 Å** (rising with energy). So the 6 Å window contained only **1–8 % of the real pairs** —
it optimized a near-empty tail while the `rdf_l1` *metric* judges g(r) out to the mean pairwise
distance. We replaced it with `rdf_hist_loss` (`--struct_mode rdf`): **per-cloud adaptive
`rmax`** (90th-percentile of that cloud's pairwise distances → energy-invariant grid) plus the
**same ideal-shell normalization** `g = counts / shell_vol` the metric uses.

**(b) Per-cloud noise (why our first corrected attempt still failed).** Our first version
computed a *per-cloud* g(r) L1 and averaged over the batch (L1-then-average). But the scorecard
computes g(r) **averaged over many clouds, then takes the L1** (average-then-L1). A single
~40-point cascade's g(r) is extremely noisy, so the per-cloud loss chased noise. We rewrote it
to **average g(r) across the batch before the relative L1**, exactly mirroring
`scorecard._per_energy`. (A per-cloud `rmax³/n²` density factor keeps each g(r) ~O(1) so
mixed-energy clouds in a batch combine as the metric intends.)

Both fixes are real improvements to the loss and are kept in the codebase, tested. They did not
change the conclusion.

## 3. The clean experiment

- **Job 7185401**, Polaris `debug-scaling`, 4 cells, 1 node (1 GPU/cell, parallel).
- Fixed recipe: per-cascade centering, EMA, `<100 keV`, 256/6, cosine LR, **500 epochs**
  (all cells reached ep500), `max_defects 1000`, pure-SGD (`--select_every 99999 --no_results`).
- **Only `--w_struct` varies**: `0` (control) / `0.3` / `1.0` / `3.0`, all `--struct_mode rdf
  --w_nn 0 --struct_start_epoch 50` (the RDF loss is gated to low-noise steps via
  `--struct_t_frac 0.25` and starts after a 50-epoch warmup, i.e. it *fine-tunes* substructure
  on a partly-converged model).
- Scored **identically and offline** with the corrected eval (`score_checkpoint.py
  --gen_energy sample`, dpmpp-20, n=48) — equal epochs, so no confound.

> Note on robustness: an earlier attempt (job 7185324, the *per-cloud* loss) was killed mid-run
> by a **node RPC timeout** (HSN infrastructure failure, not our code or walltime) at
> ep200/ep400; its partials already trended negative but were epoch-confounded. Job 7185401 is
> the clean, completed, equal-epoch replacement.

## 4. The result — decisive negative

| cell | epoch | OVERALL | `rdf_l1` (bulk 25–75 keV) | low | mid |
|---|---|---|---|---|---|
| **r0 `w_struct=0` (control)** | 499 | **0.41–0.55** | **0.13–0.16** | 0.43 | 0.39 |
| r1 `w_struct=0.3` | 499 | 1.45 | 0.52 | 1.96 | 0.98 |
| r2 `w_struct=1.0` | 499 | 1.99 | 0.76 | 2.50 | 1.54 |
| r3 `w_struct=3.0` | 499 | 2.26 | 0.75 | 2.84 | 1.63 |

(The control's OVERALL/`rdf_l1` are given as ranges because dpmpp-20/n=48 sampling noise — the
diffusion noise isn't seeded across rescores — swings OVERALL ~±0.07. The structure cells are
3–5× worse, far beyond that noise.)

- **The RDF loss degrades every metric, monotonically with weight** — overall, low, mid, and
  most damningly **`rdf_l1` itself** (the quantity it directly minimizes), which roughly
  **quadruples** even at the lightest weight (0.3) and worsens with more weight.
- The model trains *fine on the diffusion objective* during the run (eps loss ~0.20, struct
  term ~0.20 and decreasing) — yet **sampling quality collapses**. The aux gradient corrupts the
  learned score field without showing up in the training eps MSE.

## 5. The deeper reframe — we were optimizing a metric that already passes

Per-requirement breakdown of the **control** (`<100 keV`, converged):

```
requirement       value   target  weight  pass
count_mape        0.063   0.150   1.0     OK
conservation      0.000   1.000   1.0     OK
radial_js         0.006   0.050   1.5     OK
rdf_l1            0.163   0.200   2.0     OK   ← already inside target
cluster_l1        0.100   0.150   3.0     OK
nn_w1             0.719   1.000   1.0     OK
vac_sia_sep_err   2.506   3.000   1.0     OK
                                  -> 7/7 requirements PASS
```

The converged per-cascade `<100 keV` model **passes the whole scorecard.** `rdf_l1` is the
*largest-magnitude* metric relative to its target (0.82 of the way to the 0.20 limit) — which is
why it read as "the weakness" — but in absolute terms **it passes**. There was little headroom
to gain and a fully-passing model to break, and the aux loss broke it.

## 6. The general lesson (for the team)

This is the **second hand-crafted distributional coordinate loss to hurt** this set-diffusion
model:
1. **Radial-density aux loss** (earlier): hurt — the model already matched radial density
   (`radial_js ≈ 0.004`), so the term was pure gradient conflict.
2. **Pairwise / RDF g(r) loss** (this result): hurts even after fixing range + averaging, even
   at low weight, even on its own metric.

**Takeaway: do not bolt distributional auxiliary losses onto this model's coordinate objective.**
Plausible mechanism: the diffusion model learns substructure *implicitly* through the
per-token denoising (eps) objective; a g(r)/radial term is a **global, non-local** function of
all pairwise distances, so its gradient fights the local denoising signal and distorts the score
field. Because we gate it to **low-noise timesteps** (`t < 0.25 T`), it perturbs exactly the
*final* refinement steps of sampling — so even a small weight visibly corrupts generated clouds.
The clean eps objective already produces substructure that passes the scorecard.

## 7. What this means for strategy

- **Drop the substructure-loss line of work.** `--struct_mode rdf` / `--w_struct` are demoted
  to clearly-experimental (default off; "tried, hurts — don't use"). The loss code stays
  (correct + tested) for reproducibility, not for use.
- **The `<100 keV` regime is effectively solved** (7/7). Stop optimizing it.
- **Real remaining headroom is the full-range / high-E regime**, and it is a **convergence /
  capacity** story, not a loss story: converged 250ep full-range scored high-regime 0.473 vs
  2.3–3.6 for un-converged baselines. This is exactly what the **running Wave-1 preemptable
  sweep** tests (cell A: 800ep baseline; cell D: 512/10 capacity). Wait for Wave-1 before any
  new high-E experiment.
- **If substructure ever needs a structural push** (only if a *failing* metric motivates it on
  full-range/high-E), the lever is **architecture, not loss** — e.g. distance-aware /
  equivariant attention that bakes pairwise geometry into the model. Not a near-term priority.

## 8. What is preserved / still valid

- The **eval correctness fix stands**: `score_checkpoint.py --gen_energy sample` (default) —
  fixes the bin-0 left-edge artifact that inflated the low regime for every model. Unrelated to
  this negative result; keep using it (see `scorecard-findings.md`).
- The **control recipe (r0) is the best `<100 keV` model** we have. Keep it as the baseline.

## 9. Caveats / what could change this

- Tested at `<100 keV` (where the model already passes); we did **not** test the RDF loss on
  full-range/high-E, where `rdf_l1` *might* have genuine headroom. Given how badly it breaks the
  passing regime, retrying on high-E is low priority — but it's the one untested corner.
- Single-seed dpmpp-20/n=48 scoring has ~±0.07 OVERALL noise; the effect here (3–5×) dwarfs it.
- The loss fired only at low-noise `t` after a 50-epoch warmup; other gating schedules are
  untested but unlikely to flip a result this large.
