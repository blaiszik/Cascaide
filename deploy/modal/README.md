# Modal launcher (secondary path)

A thin, **isolated** way to run a Cascaide set-DiT v2 sweep on a cloud GPU. It wraps the
same `scripts/train_set_v2.py` that the primary **HPC** path (`scripts/submit_hpc.slurm`)
uses — so this is just a different launcher, not a fork. Nothing in the `cascaide` package
imports `modal`; delete this folder and everything else still works.

> Primary path for large runs is HPC (SLURM/PBS). Use Modal for quick, cheap ablations.

## One-time setup

```sh
pip install modal
modal token new          # opens a browser to authenticate
```

You also need a cached real subset on disk (uploaded to the cloud — the raw dumps are NOT):

```sh
# already have data/real_subset_large.npz (1200 cascades); for the full set:
python scripts/build_real_subset.py --data_root <dumps> --out data/real_subset_large.npz --per_energy -1
```

## Run

```sh
# default: w_struct sweep {0,0.3,0.6,1.0}, 500 epochs, A10G
modal run deploy/modal/modal_app.py

# cheaper GPU / smaller sweep
CASCAIDE_GPU=T4 modal run deploy/modal/modal_app.py --w-structs "0,0.6" --epochs 400
```

Each run writes a results-store run (manifest + scorecard + generated-vs-real images). The
local entrypoint downloads them into `./results/`, then rebuilds `index.json` + `report.html`
— so the runs appear in the dashboard / Continuity automatically.

## What it produces

A clean **A/B ablation of the substructure loss**: does `cluster_l1` / `nn_w1` / `rdf_l1`
drop as `w_struct` rises (and at what cost to `radial_js` / `count`)? Compared against the
baseline anchors already in the dashboard (copy≈0, blob≈1.9).

## Cost (verify current Modal rates)

Tiny model (7.4 M params); the cost driver is T=1000 sampling for scorecard selection.
Rough: **A10G ≈ $1.10/hr**, ~20–40 min/run ⇒ ~$0.4–0.7/run ⇒ **~$2–4 for the 4-run sweep**
(≈ half that on T4). Well under a $10 budget, with headroom for a larger stretch run
(`--epochs 800`, or edit `train_one` for a bigger `d_model`/`depth`).

## Notes / gotchas

- Uses the modern Modal API (`modal.App`, `Image.add_local_dir(..., ignore=...)`,
  `add_local_file`, `.starmap`). If your Modal version differs, adjust those calls.
- The dataset dir (`cascaide/dataset`, ~18k dump files) is excluded from the image via
  `ignore=["**/dataset/**"]` — only the cached `.npz` is uploaded.
- `pip install -e . --no-deps` is intentional: the set-diffusion path needs only
  torch/numpy/scipy/matplotlib, not ovito/yaml/hilbertcurve.
- This spends real credits. The sweep is bounded by `w_structs` × `epochs`; start small.
