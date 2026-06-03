# Decisions

> Cross-cutting decisions and trade-offs, newest first.

## 2026-06-02 — Model-progress evaluation: substructure-first scorecard + file-based HPC results store

- **Decision:** Judge models with a per-energy **scorecard** whose headline metrics are
  substructure (`cluster_spectrum` fragmentation curve + g(r)), not just count/shape; emit
  a single weighted **fitness score**. Persist results as a **file-based, versioned store**
  (`results/runs/<id>/{run.json,scorecard.json}` → `registry.collect()` → `index.json`) that
  a self-contained `viewer/dashboard.html` reads. No DB, no server.
- **Why:** The known failure is "captures shape, misses substructure," so the benchmark
  must *isolate* that (verified: a merged blob passes count/radial, fails g(r)/cluster/NN).
  HPC runs are many, async, and on scratch filesystems — a file store that each job writes
  independently and that `collect()` idempotently aggregates fits rsync workflows and avoids
  write contention; a static dashboard needs no backend on a login node.
- **Rejected:** A single scalar (e.g. Chamfer) — hides which aspect failed; a database /
  live tracking server — operationally heavy on HPC and overkill; reusing
  `cascade_studio_best.html` — it's a client-side *simulator*, not a results tracker.
- **Affects:** `cascaide/eval/*`, `scripts/*`, `viewer/dashboard.html`, `docs/benchmarks.md`,
  and the new `RadialDensityAuxLoss` (training-side counterpart of the radial metric).

## 2026-06-02 — Two model families kept in parallel (image vs set/point)

- **Decision:** Keep both the image-encoding 2D-UNet pipeline (`cascaide/`) and the
  set/point DiT pipeline (`sp_cas.py`) for now, rather than collapsing to one.
- **Why:** They have complementary failure modes. The image pipeline is mature, modular,
  and YAML-driven but loses information in the 3D→2D encode and caps point count at
  `H*W/2`. The set pipeline handles variable, unbounded counts natively and avoids the
  encoding loss, but is a single untracked file with no config system or tests yet.
- **Rejected:** Committing fully to the image pipeline (caps high-energy cascades);
  deleting the image pipeline (it's the tested, configurable one).
- **Affects:** Everything downstream — `research-directions.md` §3 proposes promoting the
  set model into the package as the migration path.

## 2026-06-02 — Normalization: per-axis robust (p99) + global centering

- **Decision:** Normalize coordinates per-axis using a robust center and a 99th-percentile
  scale, with a single global offset `G` so absolute box positions are recoverable.
  Image side (`TanhImageEncoder`): `tanh((x - G - median) / p99(|x-median|)) * coord_range`.
  Set side (`CoordNormalizer`): linear `(x - mean) / p99(|x - mean|)`.
- **Why:** Defect clouds are spatially sparse with outliers; a single scalar `norm_factor`
  (the old `Base`/`Hilbert` encoders) wastes dynamic range and clips. p99 is robust to the
  heavy tail; global centering preserves absolute positions so generated samples decode to
  real-space coordinates.
- **Rejected:** Single global `norm_factor` (clips, wastes range); min-max (outlier
  sensitive); per-cascade scale (loses absolute size signal that correlates with energy).
- **Open / inconsistent:** The two pipelines disagree on details — image uses **median**
  center + **tanh** squashing; set uses **mean** center + **linear**. Unify before relying
  on cross-pipeline comparisons. Image-pipeline params are **not persisted** in checkpoints
  (set pipeline persists them) — this is the root of inference bug #1 (see handoff).

## 2026-06-02 — Energy conditioning via broadcast channels (default)

- **Decision:** Default conditioner is `energy_channel` — broadcast the scalar normalized
  energy to `C` extra input channels concatenated to the noised image. The DiT set model
  instead injects energy through adaLN alongside the timestep.
- **Why:** Broadcast channels are the simplest conditioning that reaches every spatial
  location; adaLN (set model) is the modern, stronger choice for transformers.
- **Rejected (for now):** `energy_vector` (MLP → added to time embed) is wired in config
  but **currently crashes** due to a class-name mismatch (`EnergyVectorConditioner` vs
  `EnergyEmbedConditioner`). Consider adaLN-Zero conditioning for the image UNet too
  (`research-directions.md` §3).
