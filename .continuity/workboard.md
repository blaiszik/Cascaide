# Workboard

> The live picture of scope and priorities.

_Last updated: 2026-06-02_

## Now

- [ ] **Run the new benchmark loop on a real trained checkpoint** and read the dashboard
      (`docs/benchmarks.md`). Then A/B `RadialDensityAuxLoss` on vs off via scorecards.
- [ ] Land the `tanhp99-global-centering` branch: global centering + per-axis tanhP99
      normalization for the image pipeline (encoder + config done; **inference path
      broken**, see handoff bug #1).

## Done (this session — model-progress tooling, 2026-06-02)

- [x] `cascaide/eval/` (io, metrics incl. sub-cascade `cluster_spectrum`, scorecard,
      registry, generators) — verified on real data.
- [x] `RadialDensityAuxLoss` + config wiring — verified differentiable / trains.
- [x] `viewer/dashboard.html` — human progress viewer (HPC-results aware).
- [x] `scripts/build_real_subset.py`, `scripts/run_benchmark.py`; `tests/test_eval.py` (6).
- [x] `docs/benchmarks.md`; fixed `core.py` `CosineSchedule` import bug.

## Next

- [ ] **Fix TanhP99 inference** — persist `centroid` / `tanh_center` / `tanh_scale` in the
      checkpoint and reload in `infer.py` (mirror `sp_cas.py`'s `CoordNormalizer`).
- [ ] **Fix the `energy_vector` conditioner name mismatch** (`EnergyVectorConditioner` vs
      `EnergyEmbedConditioner`) or remove the dead config option.
- [ ] **Add a fast local benchmark suite** (see `research-directions.md` §1): encoder
      round-trip fidelity, synthetic-data overfit smoke test, distributional scorecard.

## Later

- [ ] Promote `sp_cas.py` (set/point DiT diffusion) into the `cascaide` package as a
      first-class architecture alongside the image UNet (`research-directions.md` §3).
- [ ] Unify normalization across the two pipelines and persist params in checkpoints
      (`research-directions.md` §2).
- [ ] Explore alternative architectures: 3D voxel/SDF UNet, E(3)-equivariant GNN, flow
      matching, latent set-diffusion (`research-directions.md` §3).
- [ ] Stand up an experiment runner + scalar eval "fitness function" for autoresearch
      loops (`research-directions.md` §4).
- [ ] Commit / decide the fate of untracked `sp_cas.py` and `cascade_studio_best.html`.

## Done (recent)

- [x] Continuity onboarding: scaffolded `.continuity/`, registered the repo, documented
      architecture + two confirmed bugs + research directions (2026-06-02).
- [x] Added global-centering + per-axis tanhP99 normalization pipeline (commit 84b8c2e9).
