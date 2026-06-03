"""Next-iteration set/point diffusion improvements, built on the sp_cas.py reference.

The benchmark localized the failure: the set-DiT nails count + global radial shape but
misses SUBSTRUCTURE (g(r), sub-cascade clustering, NN spacing) — generated clouds are too
compact/merged. These modules target that directly:

- `losses`  — differentiable substructure losses (pairwise-distance histogram, NN-spacing).
- `pairing` — Frenkel pair-relative encoding (SIA as displacement from its paired vacancy).
- `select`  — score a checkpoint with the scorecard (for scorecard-based model selection).
"""
from . import losses, pairing  # noqa: F401

try:  # select pulls in eval.generators (torch); keep import resilient
    from . import select  # noqa: F401
except Exception:
    pass
