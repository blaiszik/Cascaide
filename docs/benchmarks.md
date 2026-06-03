# Cascaide benchmarks, scorecard & dashboard

Tooling to track how close a cascade model is to the **real** defect distribution — built
around the known failure mode: *models capture the overall shape but miss substructure.*

## What gets measured

Per energy, a generated set is compared to the real set on:

| metric | what it catches | substructure? |
|---|---|---|
| count (MAPE, W1) | right number of defects vs energy | no |
| Frenkel violation | `n_vac == n_sia` | no |
| radial density profile (JS) | where defects sit vs distance from center | first-order |
| **pair correlation g(r) (L1)** | short-range order | **yes** |
| **sub-cascade spectrum (L1)** | # clusters vs linking length (fragmentation) | **yes (headline)** |
| NN-distance (W1) | overlaps / over-smoothing | yes |
| vac–SIA separation | interstitial offset scale | partial |

A "merged blob" with the right count and global shape **passes** count/radial and **fails**
g(r)/cluster/NN — i.e. the scorecard isolates exactly the substructure failure. The
**score** is a weighted mean of `value/target` across requirements (lower = better; <1 on a
requirement = pass), weighted hardest on the substructure metrics. It doubles as the
fitness function for autoresearch loops.

## Quick start

```sh
# 0. one-time: dedicated env with ovito (see .continuity/runbook.md)
mamba run -n cascaide pip install -e .

# 1. cache a small real subset once (needs ovito; zero-dep to load afterward)
python scripts/build_real_subset.py \
    --data_root cascaide/dataset/defect_files_20260218/defect_files_20260218 \
    --out data/real_subset.npz --per_energy 64

# 2. benchmark. Baselines give calibration reference points on the dashboard:
python scripts/run_benchmark.py --subset data/real_subset.npz --generator all-baselines
#    copy-real -> score ~0 (ceiling) | jitter -> mid | blob -> fails substructure

# 2b. a trained model, same interface:
python scripts/run_benchmark.py --subset data/real_subset.npz \
    --generator set-checkpoint   --checkpoint path/to/best_model.pt --label "set-dit v3"
python scripts/run_benchmark.py --subset data/real_subset.npz \
    --generator image-checkpoint --config cascaide/configs/config_tanhp99.yaml \
    --checkpoint runs/.../best.pt --label "tanhp99 + radial"

# 3. view it
python -m http.server 8000           # from repo root
#    open http://localhost:8000/viewer/dashboard.html
```

The dashboard auto-loads `results/index.json`; click a run for the requirements table and
gen-vs-real curve overlays, tick boxes to compare runs, and watch the score-over-time
trend. You can also **drag a `scorecard.json`** straight onto the window (e.g. one just
copied off a cluster) with no aggregation step.

## HPC result flow

Production (many async cluster jobs) is decoupled from viewing (one static dashboard) by a
file-based, versioned results store. Each job writes its OWN run dir → no write contention.

```
results/
  index.json              # registry.collect() output — the dashboard reads this
  runs/<run_id>/
     run.json             # manifest: identity, git/config provenance, model, SLURM/PBS job, status, tags
     scorecard.json       # metrics (schema_version'd)
```

On the cluster, at the end of a job:

```python
from cascaide.eval import scorecard, registry
sc = scorecard.compute(generated, reference, label="set-dit v3")
manifest = registry.build_manifest(label="set-dit v3", arch="set-dit",
            normalization="coordnorm-p99", checkpoint=ckpt, energies_keV=sc["energies_keV"],
            config=cfg, tags=["radial-loss"])     # auto-captures git + SLURM/PBS env
registry.write_run("results", manifest, scorecard=sc)
```

Then locally, after results land (e.g. `rsync` off scratch):

```sh
python -c "from cascaide.eval import registry; registry.collect('results')"  # rebuild index.json
```

`collect()` is idempotent and re-runnable as results trickle in — that's how cluster runs
"filter back into" the dashboard. The dashboard filters/sorts on the index's structured
fields (arch, normalization, energy, tag, status, git commit, score, date), so you can,
e.g., show only `set-dit` runs on the `radial-loss` branch sorted by score.

## The radial density loss

`RadialDensityAuxLoss` (in `cascaide/diffusion/loss.py`) is the training-side counterpart of
the radial metric: a differentiable soft (RBF-binned) radial histogram of the model's
decoded coordinates vs the real coordinates, normalized to a pdf. Enable it in a config:

```yaml
loss:
  aux_losses:
    radial:
      enabled: true
      weight: 0.2
      t_threshold_frac: 0.25   # only at low noise, where x0_pred is informative
      per_class: true          # match vac & SIA radial shapes separately
```

It targets the first-order radial statistic; the scorecard's `cluster_l1` / `rdf_l1` tell
you whether higher-order substructure is also improving. A natural next step is a
differentiable **g(r)/cluster** loss if radial alone isn't enough.
```
