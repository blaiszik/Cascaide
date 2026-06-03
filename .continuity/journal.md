# Build journal

> Human-facing running log of model-progress tooling work. Newest entry first.
> Each entry is updated *during* the work (not just at the end) so context can be injected.

---

## 2026-06-02 — Benchmarks, interfaces & viewer for tracking model progress

**Mandate (from human):** build the interfaces, benchmarks, and viewers so a human can
easily track how close cascade models are to "meeting the requirements." Free rein over
benchmarks / architectures / normalization. Known failure mode: **models capture the
overall cascade shape but miss SUBSTRUCTURE.** A **radial density loss** was tried
elsewhere to align radial statistics but is **not yet in this repo**.

### Interpretation of "the requirements" (what the benchmarks must measure)

A generated cascade at a given energy should match the *real* distribution on:

1. **Defect count vs energy** — `n_vac`, `n_sia`, total. (Real: ~18/35/78 at 20/50/100 keV.)
2. **Frenkel conservation** — `n_vac == n_sia` (real data obeys this exactly).
3. **Radial density profile** — defect density vs distance from cascade center. The
   first-order "where are the defects" statistic; the radial loss targets this.
4. **Pair-correlation g(r) / RDF** — local ordering / short-range structure.
5. **Sub-cascade clustering** — high-energy cascades fragment into spatially separated
   sub-cascades. *This is the "substructure" the models miss.* Measured by cluster count
   and cluster-size distribution (e.g. DBSCAN at a physical linking length).
6. **vac–SIA separation** — interstitials sit at a characteristic distance from vacancies.
7. **Nearest-neighbor distance distribution** — catches unphysical overlaps / over-smoothing.

A model that "captures but misses substructure" will look fine on (1)+(3) but fail (5)
(too few clusters / one merged blob) and (4) (washed-out g(r)). So the scorecard must make
(4) and (5) first-class, not just count/shape.

### Plan (this is the live checklist; ✅ = done & verified this session)

- [ ] **`cascaide/eval/io.py`** — cache a small real subset to `.npz` via ovito (read
      once, zero-dep load after). Interface: `cache_real_subset(...)`, `load_subset(...)`.
- [ ] **`cascaide/eval/metrics.py`** — pure-numpy structural metrics: radial profile, RDF,
      NN distances, union-find clustering (sub-cascade detection), Chamfer, conservation,
      vac–SIA separation, 1-D Wasserstein for distribution comparison.
- [ ] **`cascaide/eval/scorecard.py`** — turn {generated, reference} clouds into a
      per-energy scorecard dict + `scorecard.json`. This is the autoresearch fitness fn.
- [ ] **`cascaide/eval/encoder_fidelity.py`** — encode→decode round-trip benchmark per
      encoder (Chamfer / count-recall / label accuracy): bounds best-case generation.
- [ ] **`RadialDensityAuxLoss`** in `cascaide/diffusion/loss.py` + config wiring — the
      missing radial-statistics loss, differentiable via the encoders' soft decode.
- [ ] **`viewer/dashboard.html`** — self-contained human viewer: loads `scorecard.json`,
      overlays generated-vs-real radial/RDF/cluster curves, a pass/fail requirements table,
      and a runs-over-time trend. No network deps (hand-rolled SVG charts).
- [ ] **CLI interfaces** — `scripts/build_real_subset.py`, `scripts/run_benchmark.py`,
      `scripts/encoder_report.py`.

### Environment

- `cascaide` mamba env: ovito 3.15 + numpy + (installing) torch/scipy/matplotlib/
  hilbertcurve/einops/PyYAML. See `runbook.md`.

### HPC result flow (added requirement: cluster runs feed the dashboard)

Eventually training/eval runs on an HPC cluster (SLURM/PBS) and results must "filter back"
into the dashboard. Chosen shape — a **file-based, versioned results store** that decouples
*production* (many async cluster jobs) from *viewing* (one static local dashboard):

```
results/
  index.json            # aggregator output: flat list of run records → the dashboard filters on this
  runs/<run_id>/
     run.json           # manifest: identity, provenance (git/config), model, HPC job info, status, tags
     scorecard.json     # metrics (schema_version'd), incl. gen-vs-real curves
     curves.json        # (optional) large arrays split out
     samples/           # (optional) PNGs / generated clouds
```

- `run_id` = `<UTCstamp>-<arch>-<6hash>` (sortable, unique). Each job writes its OWN dir →
  no write contention across parallel jobs.
- **The contract is the JSON schema**, not a server. Jobs `rsync` their run dir off scratch;
  `registry.collect(results_dir)` rescans `runs/*/` and rebuilds `index.json` (idempotent,
  re-runnable as results trickle in). Dashboard reloads `index.json`.
- Manifest carries SLURM/PBS `job_id`, host, gpus, walltime + git commit/branch/dirty +
  resolved config hash → full traceability from a dashboard row back to the cluster job.
- Dashboard: filter/group runs (arch, normalization, energy, tag, status, git, score, date);
  click → load that run's `scorecard.json` (curve overlays + requirements table); compare
  mode → overlay multiple runs; trend view → score vs time/commit. Also drag-drop a single
  `scorecard.json` straight off the cluster (no aggregation needed).
- Why this fits HPC: no DB/live server (scratch + rsync friendly), append-only, versioned
  (old results stay loadable), metrics(scorecard) separated from identity(manifest) so
  either can be recomputed independently.

### Delivered this session (all verified)

- **`cascaide/eval/`** — `io` (real-subset cache), `metrics` (radial profile, g(r), NN,
  **`cluster_spectrum`** = sub-cascade fragmentation curve, Chamfer, W1/JS), `scorecard`
  (per-energy gen-vs-real + requirements pass/fail + single fitness score),
  `registry` (HPC results store + index), `generators` (set/image checkpoint wrappers).
- **`scripts/`** — `build_real_subset.py`, `run_benchmark.py` (baselines + checkpoints).
- **`cascaide/diffusion/loss.py`** — `RadialDensityAuxLoss` (the missing radial loss),
  wired into `build_loss` + a `radial:` block in `config_tanhp99.yaml`.
- **`viewer/dashboard.html`** — self-contained human dashboard (runs list + filters,
  requirements table, radial/g(r)/cluster overlays gen-vs-real per energy, counts table,
  score-over-time trend; drag-drop a scorecard.json; no network deps).
- **`tests/test_eval.py`** — 6 CI-safe tests (synthetic clouds; torch test auto-skips).
- **Docs:** `docs/benchmarks.md` (how to use it all, incl. the HPC flow).

### Bugs found while testing (verify-before-claiming caught these)

1. **FIXED — `core.py` missing import.** `GaussianDiffusion(schedule=None)` used
   `CosineSchedule()` but never imported it → `NameError` on the default path. Added the
   import; the 4 schedule-agnostic core tests now pass.
2. **Pre-existing test bug (left as-is).** `tests/test_diffusion_core.py` draws
   `t ∈ [0,100)` but builds `GaussianDiffusion(T=50)`, so `t≥T` overflows the buffer
   gather (3 tests). Real bug in the *test*, not the core; previously masked because the
   whole file aborted (see #3 + bug #1). Flagged in handoff, not silently patched.
3. **Env: a pytest plugin aborts on import here.** Run tests with
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (and `KMP_DUPLICATE_LIB_OK=TRUE` for torch+ovito).
   Documented in `runbook.md`.

### Follow-up (same day): prominent Continuity view + a real local training run

Three asks: (1) results prominently browsable in Continuity, (2) a larger local training
run to see if generated cascades are well-shaped, (3) images must show **generated** (not
encoded) cascades next to real ones.

- **(1) DONE.** New `cascaide/eval/report.py` builds a single self-contained
  `results/report.html` (requirement tables + gen-vs-real curves + generated-vs-real
  cascade images, all inlined as base64). Registered in Continuity as the Cascaide
  **human entry** (`results/report.html`, `context_dirs:["results",".continuity"]`).
  Verified: the project Cover at `http://127.0.0.1:8777/p/cascaide/` embeds the full report
  via iframe (mirror route serves all 4.4 MB / 12 images).
- **(2) RUNNING.** Set/point DiT (`sp_cas.py`) training on a 1200-cascade real subset on
  **MPS** (~360 ms/step, ~15 s/epoch), 300 epochs, via `scripts/train_set_local.py`.
  Chose the set model: it outputs true coordinates (no lossy image encoding) so "well-
  shaped?" is a fair question. Checkpoints to `runs/set_local_v1/best_model.pt`.
  (Note: synthetic clouds OOM'd MPS via O(N²) attention on ~2200 tokens — real clouds are
  ≤160 tokens, fine. Yet another reason synthetic misleads.)
- **(3) DONE.** `cascaide/eval/render.py` renders generated (top) vs real (bottom) per
  energy with shared axis limits; wired into `run_benchmark.py` → run's `samples/`. Both
  the static report and the interactive dashboard display them.
- **Generation cap added** (`n_per_energy`, default 48): generating 1 sample per reference
  cascade (1200 × 2.7 s ≈ 54 min) was absurd; the scorecard only needs a few dozen/energy.
- **Real-model path validated** on an early checkpoint (~10 min training): score 1.01,
  and it already shows the textbook signature — **radial PASS (JS 0.023), but g(r)/cluster/
  NN FAIL.** i.e. captures global shape, misses substructure — the exact thing to fix.
  Expect substructure metrics to improve as training continues + with the radial loss.

### Shape assessment (interim, set-DiT ~epoch 60)

Real-model run is live in `results/` and on the Continuity cover. Scorecard:
`count 0.10 PASS · conservation PASS · radial_js 0.013 PASS · rdf_l1 0.39 FAIL ·
cluster_l1 0.16 FAIL · nn_w1 1.23 FAIL · vac_sia_sep 5.6 FAIL` (score 1.08, 3/7).
**Visual (compare_100keV.png):** generated clouds reproduce the overall blob size and
vac/SIA mixing but are **too compact / merged** vs the real cascades, which are more
dispersed with finer sub-structure. So even early, the model "captures shape, misses
substructure" — confirmed both numerically and by eye. Training continues to 300 epochs;
the radial loss (now available) + longer training should pull `cluster_l1`/`nn_w1` down.
**Next on training completion:** final MPS benchmark (48/energy) → rebuild report.

### RESULT — the 300-epoch run reframes the substructure story

Fully trained, the sp_cas DiT is **well-shaped** — and val-MSE selection would have stopped
us too early. Scorecards (full real reference, 48/energy generated):

| checkpoint | score | cluster_l1 | nn_w1 | rdf_l1 | vac_sia |
|---|---|---|---|---|---|
| **ep300 (final)** | **0.453** | 0.049 ✅ | 0.60 ✅ | 0.229 ✗ | 2.13 ✅ |
| ep~50 (best-VAL)  | 0.772 | 0.124 ✅ | 1.13 ✗ | 0.352 ✗ | 1.71 ✅ |
| ~ep60 (interim)   | 1.077 | 0.164 | 1.23 | 0.389 | 5.55 |
| baselines: copy 0.0 · jitter 0.43 · blob 1.77 | | | | | |

Takeaways (revising earlier reads):
- **The "missed substructure" was UNDERTRAINING, not the architecture.** By ep300, cluster /
  nn / radial / count / vac-sia all PASS (6/7) — so the set-DiT, fully trained, is
  **well-shaped in absolute terms**.
- **CAVEAT (corrected 2026-06-02): this does NOT establish sp_cas is "the best" architecture.**
  Every comparison above is *within the same architecture* (different checkpoints). No
  alternative (image UNet, etc.) has been scored — the image pipeline has zero scorecard runs.
  "current best" is the **researcher's claim**; our data is *consistent with* it (sp_cas is
  good and not broken) but does not *test* it. To verify "best", benchmark alternatives on the
  same scorecard. Build on sp_cas for now because it's proven-good + the researcher's pick,
  not because we measured it beats anything.
- **val coord-MSE is the wrong selector, with numbers:** the val-selected ep~50 model scores
  0.772 vs the ep300 model's 0.453. Select by scorecard. (This is why best_model.pt froze at
  ep50 yet the ep300 model is clearly better.)
- **The one remaining gap is `rdf_l1` (0.229 vs 0.20)** — short-range pair correlation —
  which is EXACTLY what the substructure (pairwise-distance-histogram) loss targets. That is
  the sharp, narrow hypothesis for the Modal sweep, on the proven architecture.

### Post-test + dashboard metrics (final state this session)

- **Pair-relative model wired** (`setdiff/paired.py` + `train_paired.py` + generator +
  `run_benchmark --generator paired-checkpoint`), tested + smoke-verified. Not trained at
  scale yet (user chose to stop new runs). Caveat: no GT Frenkel labels → Hungarian
  matching → displacement scale ~58 Å (partial range reduction).
- **New metrics added** (`metrics.py`): partial RDFs (vv/ss/vs), `cross_nn_distances`
  (Frenkel-separation distribution), `radius_of_gyration`, `largest_cluster_frac`,
  `gyration_anisotropy`. Top 3 captured as **additive scorecard diagnostics** (not weighted
  → scores stay comparable) and **displayed in the report** (`report.py:_diag_table`).
- **`scripts/posttest.py`** re-evaluated all models on the full metric set at uniform
  32/energy and refreshed the dashboard. **Final dashboard (8 runs, all with diagnostics,
  served on the Continuity cover):**
  | run | score | pass | vac_sia_nn_w1 |
  |---|---|---|---|
  | copy-real | 0.000 | 7/7 | 0.00 |
  | jitter-2Å | 0.434 | 6/7 | 0.24 |
  | **set-dit ep300** | **0.444** | **7/7** | 1.53 |
  | set-dit v2 (w=0 control) | 0.518 | 6/7 | 3.03 |
  | set-dit ep~50 best-val | 0.857 | 5/7 | 1.70 |
  | set-dit v2 (w=0.6) | 0.924 | 3/7 | 3.52 |
  | set-dit v2 (w=1.0) | 1.084 | 3/7 | 4.03 |
  | blob | 1.772 | 4/7 | 2.91 |
  The new `vac_sia_nn_w1` diagnostic confirms the substructure loss hurts vac-SIA most
  (3.5–4.0) and ep300 is the best real model (1.53).
- **EMA Modal run TIMED OUT** (Modal 2h function cap; 500ep + T=1000 selection sampling on
  T4 exceeded it; healthy at ep420 but lost — no checkpoint returned). Fixed: modal_app
  timeout → 5h. **EMA already verified working** (ep100 0.49, ep300 0.45 vs no-EMA control
  degrading to 0.67). User chose to **stop new runs** — finalize with current models.
- Budget: ~$5.3 of $30 Modal (validation + sweep + 2 lost EMA attempts).

### Learnings & implications (what the data + the run taught us)

1. **The benchmark localized the failure — and it's not where the obvious fix points.**
   The set-DiT learns count(energy) and global radial shape almost immediately (count MAPE
   ≈0.1, radial JS ≈0.013 by ~ep50) and stays there; it misses substructure (g(r), cluster
   spectrum, NN spacing) — generated clouds are **too compact/merged** vs dispersed real
   ones. ⟹ **The radial-density loss I added targets what's already solved.** The real lever
   is a **pairwise/g(r)/NN-spacing/cluster-dispersion** objective. The scorecard saved a
   wasted long run — that's the point of having it.
2. **The model's val metric is decoupled from what we care about.** `best_model.pt` froze
   at ~ep50 (val coord eps-MSE plateaued) while training ran to 300. ⟹ Select checkpoints
   by the **scorecard** (cluster_l1, nn_w1), not val loss.
3. **The failure is separable and measurable.** A merged blob (right count + radial) passes
   global metrics and fails exactly the substructure metrics (blob → 4/7, 1.9). ⟹ We have a
   fitness function that distinguishes "right shape" from "right substructure," with
   calibration anchors (copy≈0, jitter≈0.48, blob≈1.9).
4. **Real data is more correct AND cheaper here than synthetic.** Real cascades are ~10×
   smaller than synthetic assumed; synthetic's oversized clouds even OOM'd MPS (O(N²) attn).
   ⟹ Default to a cached real subset; synthetic only for CI/stress.
5. **vac–SIA placement is off (~5.5 Å sep error).** ⟹ **Pair-relative encoding** (SIA as
   displacement from its paired vacancy; Frenkel pairs guarantee a matching) is now
   data-justified, not just a hypothesis.
6. **Set/point is the right architecture** (small clouds, native variable count,
   conservation, true coords / no encoding-loss ceiling). ⟹ Promote it; iterate here.
7. **Sampling is the bottleneck** (CPU T=1000 ≈20+min/72 clouds; MPS ≈2.7s each). ⟹
   Autoresearch loops need DDIM/flow-matching or sample caps.

**Next iteration (building now):** (1) differentiable substructure loss (pairwise-distance
histogram + NN-distance), (2) pair-relative Frenkel encoding, (3) scorecard-based checkpoint
selection, (4) HPC-ready training driver that writes a run to the results store.

### Next-iteration build (items 1–4) — ready to run

- **(1) substructure loss ✅** `cascaide/setdiff/losses.py`: differentiable
  pairwise-distance-histogram (a g(r) loss) + NN-spacing loss, low-t gated. Tested
  (zero when equal, flags merged blobs, finite grads).
- **(2) pair-relative encoding ✅ (utilities)** `cascaide/setdiff/pairing.py`: Hungarian
  Frenkel matching, `to/from_pair_relative`, 6-D pair tokens. Tested (round-trip recovers
  SIA; shrinks dynamic range). Paired 6-D model wiring is the follow-on (see handoff).
- **(3) scorecard selection ✅** `cascaide/setdiff/select.py` + `scripts/train_set_v2.py`:
  generates from the live model every `select_every` epochs and keeps the best **scorecard**
  (not val MSE) → `best_by_scorecard.pt`.
- **(4) HPC-ready ✅** `train_set_v2.py` writes a run (manifest+scorecard+images) into the
  results store on finish and rebuilds the report; `scripts/submit_hpc.slurm` template;
  registry auto-captures SLURM/PBS + git. Cloud runs (incl. Modal) feed the dashboard.
- `tests/test_setdiff.py` 5/5 pass; `train_set_v2.py` smoke-verified (eps + struct loss
  both drop over a 3-epoch CPU run).

### Cloud: Modal launcher (secondary; HPC stays primary)

- `deploy/modal/modal_app.py` + `README.md`: **isolated** launcher that runs the SAME
  `scripts/train_set_v2.py` on a Modal GPU as a `w_struct` sweep. Nothing in the package
  imports modal; delete the folder and everything still works. Cheap because the cached
  `.npz` means **no ovito in the cloud** (image = torch/numpy/scipy/matplotlib only) and the
  raw dump dataset is excluded from the upload. Each run downloads back into `results/` and
  rebuilds the report. Est. ~$2–4 for a 4-run sweep on A10G. NOT yet submitted (spends
  credits — awaiting go-ahead).

### Modal validation run (single submission) — round-trip CONFIRMED

First cheap Modal run (T4, w_struct=0.6, 40 epochs, ~$0.10) to validate result capture
before any sweep. Outcome: **the full round-trip works** — image builds (no ovito), trains
on GPU (~4.6 s/epoch), substructure loss fires from ep20, scorecard selection runs, and the
run (run.json + scorecard.json + 3 gen-vs-real images) **downloads correctly** into
`results/runs/<id>/` and shows in the index + report. (40-epoch validation score 1.165 —
not meant to be well-shaped.) Bugs/gaps the validation surfaced + fixed:
- **Result-capture path bug** (doubled `runs/` in the download path) — fixed before running.
- **Silent, slow finalize looked hung** (~6 min): the in-container finalize generates clouds
  via full T=1000 sampling with no output, and the early-checkpoint count head can emit
  near-cap (800-token) clouds (O(N²) attn). Fixed: `--finalize_n` default 8 + progress prints.
- **Modal runs were tagged `scheduler=local`** (no SLURM env in the container) → added Modal
  detection (`MODAL_TASK_ID`) to `registry.capture_hpc`.
- **Artifact gap FOUND + FIXED (the important one):** the first run returned scorecard +
  manifest + images but **NOT the trained weights** (checkpoint stayed ephemeral; manifest
  even pointed at a dead container path). Fix: `train_set_v2` now co-locates the selected
  checkpoint as `results/runs/<id>/model.pt` and sets `manifest.model.checkpoint="model.pt"`,
  so the run dir is **self-contained** and the existing `results/runs/**` downloader pulls
  the weights back. Verified locally (run dir has model.pt 28 MB + scorecard + run.json +
  3 images). Benefits HPC too (rsync a run dir → has everything). The 40-epoch validation
  run was deleted from the store (low-quality + pre-fix dangling checkpoint).

### Sweep diagnostic + recipe changes (in-flight)

Live sweep (3000 cascades, 400 ep) best-by-scorecard, 12-sample in-training metric:
`w_struct=0.0 → 0.48 (ep100), worsening to 0.67 (ep300); w_struct 0.3/0.6/1.0 → 1.2–1.8`.
- **The substructure loss HURTS at w≥0.3** (control ≫ better). Hypothesis negative as configured.
- The control (`w=0`) ≈ reproduces sp_cas (~0.48 at 12 samples vs 0.453 at 48 — likely just
  small-sample bias; the in-training/finalize scores use 8–12 samples → biased UPWARD).
- **Scorecard worsens with training** → likely late-training drift (EMA should fix).

Changes made (verified locally):
- **EMA added** to `train_set_v2` (on by default, decay 0.9999; used for select + sample +
  save → `model.pt` holds EMA weights). The high-confidence quality lever; also smooths the
  drift above.
- **`scripts/rescore.py`** — re-score run-dir `model.pt` at 48/energy (× seeds) and rebuild
  the dashboard. Fixes the small-sample bias so sweep results are comparable to the 0.453
  baseline. (Run after the sweep lands.)
- **Substructure loss DEMOTED** — `--w_struct` default 0.0 (off). Revisit only as a tiny
  fine-tune weight on a converged checkpoint.

### Progress log (most recent first)

- _17:10_ — Diagnosed sweep: substructure loss hurts (w≥0.3 ≫ worse than control); control
  ≈ reproduces sp_cas (small-sample bias inflates in-training scores). Added EMA (on by
  default, used everywhere + saved), `rescore.py` (fair 48-sample scoring), demoted the loss
  to default-off. All verified via local smoke (EMA train→save→rescore round-trip).
- _16:05_ — Modal single-run validation PASSED: full train→generate→score→render→**download**
  round-trip confirmed; the run is in the dashboard. Fixed the capture-path bug + silent
  finalize + Modal provenance tagging. Ready for the real w_struct sweep once hyperparams are
  confirmed. (HPC remains the primary path; Modal stays an isolated launcher.)
- _15:50_ — Built + tested items 1–4 (substructure loss, pair-relative utils, scorecard
  selection, HPC driver `train_set_v2.py` + SLURM template). Added Learnings section above.
  Next: finalize report when set_local_v1 hits ep300; planning a small Modal w_struct sweep
  (cached npz → no ovito in cloud → cheap) to test if the substructure loss closes the gap.
- _14:46_ — Real-model interim report LIVE in Continuity (set-DiT ~ep60): 4 runs, 16
  inlined images. Visual + scorecard agree: global shape good (radial_js 0.013), substructure
  off (cluster/nn/rdf fail). CPU generation was far too slow (killed a 72-sample CPU run at
  20+ min); MPS at 8/energy lands in ~3 min even alongside training. Dashboard now shows the
  gen-vs-real images too. Training (300 ep) still running; will finalize on completion.
- _15:30_ — Built report.py (Continuity-prominent), render.py (gen-vs-real images),
  generation cap; registered report as Cascaide human entry (verified via mirror). Launched
  300-epoch set-DiT training on 1200 real cascades (MPS). Validated the full
  checkpoint→generate→render→score→report path on an early ckpt (score 1.01, substructure
  metrics failing as expected). Waiting for training to finish to produce the final report.
- _15:05_ — Radial loss verified via real overfit-a-handful (total 17.1→1.6, finite grads,
  differentiable). `tests/test_eval.py` 6/6 pass (caught + fixed a bad synthetic-blob in my
  own test — metric was right, test data was too sparse). Fixed the `CosineSchedule` import
  bug. Dashboard data-contract verified against the scorecard schema; all fetch paths 200.
  **Tooling complete.** Writing docs + handoff.
- _14:25_ — **End-to-end pipeline working ✅.** `registry.py` (HPC results store) +
  `scripts/build_real_subset.py` + `scripts/run_benchmark.py` + `eval/generators.py`
  (set/image checkpoint wrappers; the image one passes `dataset=` into `build_encoder` so
  it sidesteps infer bug #1). Installed the package editable into the env. Built a 144-
  cascade real subset and ran the 3 baselines → results store populated with
  run.json+scorecard.json+index.json. **Calibration spread:** copy-real 0.000 (7/7),
  jitter-2Å 0.483 (6/7), blob 1.867 (4/7) — exactly the ordering we want, and gives humans
  reference points ("a perfect model scores 0, a structureless blob ~1.9"). Next: the
  dashboard HTML that reads this store.
- _14:05_ — **`scorecard.py` ✅ built and validated against the failure mode.** Sanity:
  real-vs-real scores 0.000, 7/7 requirements pass. Critical test: a "merged blob"
  (correct count, correct global radial shape, destroyed substructure) scores 1.93 and
  passes count/conservation/**radial (JS 0.037)** but FAILS g(r) (L1 0.68), sub-cascade
  spectrum (L1 0.43), and NN-distance (W1 3.5 Å). i.e. the scorecard reproduces exactly
  "captures shape, misses substructure" — global metrics pass, substructure metrics fail.
  This is the discriminating power the dashboard needs. Next: `registry.py` (HPC store).
- _13:48_ — Full env ready (torch/scipy/matplotlib/hilbertcurve installed). **`eval/io.py`
  ✅ and `eval/metrics.py` ✅ verified on REAL data**: cluster counts rise with energy
  (13.5 → 29.6 → 61.6 @ 20/50/100 keV), g(r) peaks at ~2.2 Å, radial pdf peaks mid-cloud —
  all sane. Added scale-robust `cluster_spectrum` (fragmentation curve) as the headline
  substructure metric after seeing a single link length is too scale-sensitive (mean NN
  ≈7.5 Å). Designed the HPC results store (above). Next: `scorecard.py` + `registry.py`.
- _13:30_ — Mandate received. Inspected `cascade_studio_best.html` (it's a client-side
  cascade *simulator/demo*, not a progress tracker → building a separate benchmark
  dashboard). Wrote this plan. Starting on `eval/io.py` + `metrics.py`.
