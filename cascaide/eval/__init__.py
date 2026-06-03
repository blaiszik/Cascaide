"""Evaluation / benchmarking toolkit for Cascaide.

Everything a human (or an autoresearch loop) needs to track how close a generated
cascade distribution is to the real one — with a deliberate focus on the SUBSTRUCTURE
the models tend to miss (radial density, pair-correlation g(r), sub-cascade clustering).

- `io`        — cache a small real subset to a dependency-free .npz, and load it.
- `metrics`   — pure-numpy structural metrics (radial profile, RDF, clusters, NN, Chamfer).
- `scorecard` — turn {generated, reference} clouds into a per-energy scorecard + JSON.
"""
from . import io, metrics  # noqa: F401

# Optional submodules (present once built); import lazily so partial trees still load.
try:  # pragma: no cover
    from . import scorecard  # noqa: F401
except Exception:
    pass
