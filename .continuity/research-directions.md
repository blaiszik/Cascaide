# Research directions

> Forward-looking ideas for Cascaide, grounded in the current code. Four tracks:
> (1) local benchmarks for quick screening, (2) better normalization, (3) different
> architectures, (4) autoresearch (Karpathy-style) loops. Each item notes *why it matters
> here* and *where to start*. None of these are committed work — they're a menu.

Context that shapes all of it: the data is W collision cascades; defects are **Frenkel
pairs** (`n_vac == n_sia`); counts are **heavy-tailed and rise with energy**; clouds are
**sparse, unoriented, and unbounded in size**; and the only labels are `(energy → cloud)`.
Training is slow (200–600 epochs over ~3000 cascades) and there is currently **no fast
feedback loop and no physics-aware metric** — that gap is the throughline below.

---

## 1. Local benchmarks for quick screening

The repo has 2 unit tests (`test_diffusion_core.py`, `test_base_image.py` — shapes/math
only) and no way to judge sample *quality* without eyeballing 3D scatter plots from
`infer.py`. Cheap, automatable screens, roughly in order of value:

1. **Encoder round-trip fidelity (CPU, sub-second).** For each encoder, `encode → decode`
   a known cloud and report: **Chamfer distance**, **count recall** (fraction of points
   recovered, given the `H*W/2` cap), and **label accuracy** (vac vs SIA). This bounds the
   *best case* any generator on top of that encoder can achieve — the single most
   informative cheap number, and it quantifies the lossy encoders that today are only
   shape-checked. Start from `tests/test_base_image.py`; run over Base / Hilbert3Ch /
   Hilbert4Ch / TanhP99.
2. **Overfit-a-handful smoke test on REAL data (<2 min, CPU).** Now that ovito loads
   (see `runbook.md`), cache a small real subset (§"small real data" below) and assert the
   model can memorize 4–8 real cascades — loss drops below threshold and stays finite. A
   real target makes the success criterion meaningful; `sp_cas.SyntheticCascadePairDataset`
   stays as a *zero-dependency fallback* for machines without the dumps, not the default.
   NB: synthetic cascades are ~10× oversized vs reality (`runbook.md`), so don't tune
   sizes/caps against them.
3. **Distributional scorecard (no training; any checkpoint).** Sample K clouds per energy
   and compare to held-out GT with physics-meaningful metrics:
   - **count error** (MAE / MAPE vs the GT count distribution per energy),
   - **radial distribution function (RDF / pair-correlation g(r))** — Wasserstein distance
     between generated and GT; the standard materials-science structural fingerprint,
   - **nearest-neighbor distance distribution** (are defects unphysically overlapping?),
   - **vac–SIA centroid separation (ICD)** — already computed in `sp_cas.evaluate` (`:546`),
   - **cluster-size distribution** (DBSCAN at the lattice spacing),
   - **Frenkel-conservation violation** (`|n_vac - n_sia|`) — should be ~0.
   Emit one `scorecard.json` per checkpoint. This *is* the fitness function track 4 needs.
4. **Tiny end-to-end config (`config_smoke.yaml`).** `image_size=16, T=50, base_channels=16,
   num_epochs=3, max_samples=64` → full train+sample+infer in <1 min as a real integration
   test (the current tests never exercise `Trainer`/`UNet`/sampler together).
5. **Throughput micro-benchmark.** ms/step (fwd+bwd) and sampler steps/s for UNet vs DiT at
   matched param counts — so architecture comparisons (track 3) include cost, not just loss.

**Why now:** without (1) and (3) you can't tell a normalization or architecture change from
noise, and tracks 2–4 all depend on having a trustworthy cheap metric.

### Small real data > synthetic (recommended default)

ovito now works, so the only reason synthetic existed (the dependency) is gone. Prefer a
small *real* subset as the fast-loop substrate. Concretely:

- **Cache once, load forever.** Use ovito to read N≈64–256 real cascades (stratified
  across 20/50/100 keV) and save coords/types/energy to one `.npz`/`.pt` (~a few MB,
  loads <1s, **no ovito/torch-data needed afterward**). This makes real data as convenient
  as synthetic for CI and iteration. `CascadeDataset` already `lru_cache`s in-process; a
  cached file extends that across processes.
- **Diagnostics that need a real target** (synthetic can't give these): overfit-a-handful
  (1.2); encoder round-trip on real, clustered geometry (1.1); a **real distributional
  reference** for the scorecard (1.3) — score generations against real, not fiction.
- **Small-data science questions** (only meaningful on real): a **data-efficiency curve**
  (train on 32/64/256/full → scorecard; tells you whether to buy more MD or more inductive
  bias); **leave-one-energy-out** (train 20+100 keV, test held-out 50 keV — does energy
  conditioning interpolate?); **memorization vs novelty** (min-Chamfer of each sample to
  the train set — with little data, diffusion tends to copy).
- **Augment, don't fabricate.** Bulk cascades are ~isotropic: O(h)/SO(3) rotations,
  reflections, translations, and small jitter turn 50 real cascades into many effective
  samples without inventing statistics (optionally exploit the 48 BCC point-group ops).
- **Retrieval baseline.** "Given energy, return the nearest real cascade" — any generator
  must beat resampling a real one on diversity while matching the distribution. Cheap and
  humbling.
- **Honest role for synthetic:** zero-data CI, controllable stress tests (deliberately
  huge clouds to probe the `H*W/2` cap), and scaling studies needing many samples — but
  not the default screen.

---

## 2. Better data normalization schemes

Current: image = global-center + per-axis `tanh((x-G-c)/s)`; set = per-axis `(x-mean)/p99`
(see `decisions.md`). Ideas, most promising first:

1. **Pair-relative encoding (exploit Frenkel structure).** Encode each SIA as a
   **displacement from its paired vacancy** instead of an absolute position. The
   displacement distribution is tight (interstitials sit near their vacancy), so dynamic
   range collapses and the model learns *direction/length* rather than absolute position.
   Needs a pairing step (nearest-neighbor / Hungarian) but `n_vac == n_sia` guarantees a
   perfect matching exists. Big potential win; pairs naturally with the set model.
2. **Voxel / density-field representation.** Rasterize to a 3D occupancy or smoothed-density
   grid with **separate vac and SIA channels**; normalize to `[0,1]`. Eliminates the lossy
   `H*W/2` raster cap entirely and handles variable counts for free — enables a 3D UNet
   (track 3). Decode = peak-finding / thresholding back to points.
3. **Lattice-aware normalization.** Coordinates live on a BCC W lattice (a = 3.165 Å, in
   `metadata.json`). Express positions in **lattice units** and add a **lattice-snap**
   decode post-process so generated defects land on physically allowed sites. Cheap,
   physics-grounded, and directly improves the RDF metric.
4. **Per-cascade PCA / whitening.** Cascades have no canonical orientation; align each to
   its principal axes before normalizing to remove rotational nuisance variance, then let
   the model spend capacity on shape. (Or skip it and bake rotation invariance into the
   net — see equivariance, track 3.)
5. **Size/energy-conditioned scale.** With a heavy-tailed size distribution, a single
   global p99 scale leaves small cascades using a sliver of the range. Try a log-radial
   transform or a scale that depends on energy/count.
6. **Unify + persist.** Pick one scheme for both pipelines and **store the params in the
   checkpoint** (the set pipeline already does via `CoordNormalizer.to_dict()`; the image
   pipeline does not — that's inference bug #1). Non-negotiable for reproducible inference.

---

## 3. Different architectures

1. **Promote the set/point DiT (`sp_cas.py`) into the package — highest leverage.** It
   already works: a DiT transformer (`SetDenoiser`) diffusing directly on point
   coordinates with adaLN conditioning on (timestep, energy) + a vac/SIA type embedding,
   plus a `CountHead` that models `n_pairs` so generation is `energy → count → cloud`. It
   sidesteps *both* image-encoder weaknesses (information loss + the `H*W/2` cap). Migration
   path: wrap it behind the existing `build_*` factories with its own YAML and the
   benchmark suite from track 1.
2. **E(3)/SE(3)-equivariant network** (EGNN / point-cloud transformer on relative coords).
   Cascades have no preferred orientation, so equivariance is a strong inductive bias and a
   real data-efficiency win on only ~3000 samples — likely the biggest quality lever after
   (1). Combine with SO(3) augmentation as a cheaper first step.
3. **3D voxel/SDF UNet** (pairs with normalization §2.2) — standard, well-understood 3D
   diffusion on density grids; natural variable-count handling; trivially supports
   classifier-free guidance on energy.
4. **Flow matching / rectified flow** instead of DDPM. The diffusion scaffolding
   (`GaussianDiffusion`, schedules, samplers) is clean and swappable; flow matching often
   gives better samples in far fewer steps — a cheap, high-upside experiment.
5. **Latent set-diffusion.** Encode a cloud to a fixed-size latent (PointNet/set encoder),
   diffuse in latent space, decode — decouples count from geometry and shrinks the
   diffusion problem.
6. **Conditioning upgrades for the image UNet.** It currently only adds energy to the time
   embedding or as broadcast channels; adopt **adaLN-Zero** (as the DiT already does) and
   **classifier-free guidance** on energy for sharper energy control. Also fix/restore the
   `energy_vector` path (currently broken — see `decisions.md`).

---

## 4. Autoresearch (Karpathy-style) paths

The goal: a tight, automated loop where a change is proposed, run on a cheap benchmark,
scored, and the result logged — so iteration is fast and self-documenting.

1. **The eval "fitness function" is the prerequisite.** Track 1.3's `scorecard.json`
   reduced to one (or a few) scalars is what closes the loop. Without a cheap, trustworthy
   metric, autoresearch can't run. Build this first.
2. **Experiment runner.** A `sweep.py` that takes a base YAML + an override grid, launches
   runs, and appends `(config, scorecard)` rows to a results table. The repo already has
   the substrate: per-run `config.json` snapshots and structured `train_log.jsonl`
   (`trainer.py`). Make runs comparable, then sweepable.
3. **Minimal single-file baseline as the playground.** `sp_cas.py` is already a
   nanoGPT-style hackable reference — keep it that way. Paired with a **cached small real
   subset** (preferred) or the synthetic dataset (zero-dep fallback) it gives
   seconds-per-iteration experimentation: the ideal substrate for an automated
   propose→run→measure loop.
4. **Scaling micro-study.** Sweep DiT `d_model`/`depth` (and UNet `base_channels`/levels)
   on synthetic data; plot loss vs params/compute. A Karpathy-style scaling plot tells you
   where capacity actually pays off *before* spending GPU-days on the real dumps.
5. **Agent-driven ablation loop.** An agent proposes a config delta (normalization,
   architecture, loss weights), runs the smoke benchmark, reads the scorecard, and iterates
   — logging hypothesis → result → decision. `decisions.md` + the Continuity session
   `log.md` are a natural, durable experiment journal for exactly this.
6. **Observability.** `train_log.jsonl` already exists; repurpose `cascade_studio_best.html`
   ("AI Cascade Studio") or a tiny dashboard to watch the metric live, so the loop is
   inspectable rather than a black box.

**Suggested sequence:** track 1 (benchmarks) → track 2 (normalization, with persisted
params) → track 3.1 (promote the set model) → track 4 (wrap it all in a sweep + agent
loop). Each stage makes the next one measurable.
