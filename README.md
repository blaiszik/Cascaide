# Cascaide
Cascaide: Cascaded Generative Modeling with Diffusion.

## What's in here

The pipeline has four pluggable pieces, each independently swappable:

- **Encoders** — turn raw 3D coordinates into 2D images. Three implementations are included: a baseline raster-packed encoder and two Hilbert-curve-sorted variants (3-channel and 4-channel) that preserve 3D spatial locality in the 2D layout.
- **Diffusion** — Gaussian DDPM with pluggable noise schedules (linear, cosine, sigmoid) and parameterizations (eps, x0, v).
- **UNet** — channel-and-resolution-adaptive 2D UNet with timestep + energy conditioning.
- **Aux losses** — composable auxiliary losses on top of the standard diffusion MSE: occupancy, count, classification, and multi-axis projection.

Everything is configured through YAML and trained with a single command.

## Installation

Cascaide is a regular Python package. Clone, then install in editable mode:

```bash
git clone https://github.com/vigsam-coder/Cascaide
cd Cascaide
pip install -e .
```

Requires Python 3.10+. The dependency on `ovito` is for reading LAMMPS-style `.dump` files; if you only need the diffusion components and have your own data loader, it can be skipped.

## Data layout

The dataset loader expects this directory structure:

- data_root/
  - 0-10keV/
    - metadata.json
    - 0001_min_vac.dump
    - 0001_min_sia.dump
    - 0002_min_vac.dump
    - ...
  - 10-30keV/
    - ...
  - 100keV/
    - ...
## Quick start

### 1. Train

```bash
python cascaide/training/train.py --config cascaide/configs/config.yaml

```

### 2. Inference

```bash
python infer.py --config cascaide/configs/config.yaml --checkpoint runs/hilbert4ch_v1/checkpoints/best.pt --output_dir runs/hilbert4ch_v1/figures_best

```
