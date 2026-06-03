"""Structural metrics for cascade defect point clouds (pure numpy, no scipy/sklearn).

Designed around the known failure mode: models capture the overall cascade shape but miss
SUBSTRUCTURE. So the headline metrics are the ones that expose substructure —
pair-correlation g(r) (short-range order) and clustering (sub-cascade fragmentation) —
alongside the first-order radial density profile that a radial loss targets.

Conventions: ``coords`` is an (N, 3) float array in Ångström (absolute or centered — most
metrics are translation-invariant or recentre internally). Empty clouds are handled.
"""
import numpy as np

# Linking length for sub-cascade detection. Real W cascades have mean NN ≈ 7–8 Å, so a
# sub-cascade (a spatially distinct group) only forms above that scale. ~3 lattice spacings
# (a≈3.165 Å) is a reasonable default; substructure is best probed across scales, so prefer
# ``cluster_spectrum`` over a single link length.
DEFAULT_LINK_LENGTH = 10.0  # Å
# Scales (Å) for the fragmentation curve — from sub-NN to whole-cascade.
DEFAULT_LINK_SCALES = (4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 25.0)


# ----------------------------------------------------------------- basic geometry helpers
def _as2d(coords):
    a = np.asarray(coords, np.float32)
    return a.reshape(0, 3) if a.size == 0 else a.reshape(-1, 3)


def _pdist(a):
    """Condensed-ish: full pairwise distance matrix (N,N). N is small (<~1500)."""
    d = a[:, None, :] - a[None, :, :]
    return np.sqrt(np.maximum((d * d).sum(-1), 0.0))


def centroid(coords):
    a = _as2d(coords)
    return a.mean(0) if len(a) else np.zeros(3, np.float32)


# ----------------------------------------------------------------- radial density profile
def radial_density_profile(coords, center=None, rmax=None, nbins=24):
    """Defect density as a function of distance from the cascade center.

    Returns dict with:
      r        : (nbins,) shell-center radii
      density  : (nbins,) number density per shell (counts / shell volume)
      pdf      : (nbins,) normalized radial distribution (sums to 1) — scale-free, the
                 thing to compare across clouds of different size.
      rmax     : the rmax used.
    The first-order "where are the defects" statistic; what a radial loss aligns.
    """
    a = _as2d(coords)
    if len(a) == 0:
        z = np.zeros(nbins)
        return {"r": np.zeros(nbins), "density": z, "pdf": z, "rmax": rmax or 1.0}
    c = center if center is not None else a.mean(0)
    rad = np.linalg.norm(a - c, axis=1)
    if rmax is None:
        rmax = float(rad.max()) * 1.05 + 1e-6
    edges = np.linspace(0, rmax, nbins + 1)
    counts, _ = np.histogram(rad, bins=edges)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = counts / np.maximum(shell_vol, 1e-12)
    pdf = counts / max(counts.sum(), 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    return {"r": r, "density": density, "pdf": pdf, "rmax": rmax}


# ------------------------------------------------------------- pair correlation g(r) / RDF
def rdf(coords, rmax=None, nbins=24):
    """Effective radial distribution function g(r) for a finite cluster.

    Histograms all pairwise distances, normalized by the ideal-shell expectation given the
    cloud's own mean number density. Sensitive to short-range order / substructure that a
    smooth blob washes out. Returns dict {r, g, rmax}.
    """
    a = _as2d(coords)
    if len(a) < 2:
        return {"r": np.zeros(nbins), "g": np.zeros(nbins), "rmax": rmax or 1.0}
    D = _pdist(a)
    iu = np.triu_indices(len(a), k=1)
    dists = D[iu]
    if rmax is None:
        # cap at the cloud's gyration scale so g(r) probes structure, not the boundary
        rmax = float(np.percentile(dists, 90)) + 1e-6
    edges = np.linspace(0, rmax, nbins + 1)
    counts, _ = np.histogram(dists, bins=edges)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    # mean density rho = N / volume(extent sphere of radius rmax)
    N = len(a)
    rho = N / ((4.0 / 3.0) * np.pi * rmax ** 3)
    norm = 0.5 * N * rho * shell_vol  # expected # pairs per shell for ideal gas
    g = counts / np.maximum(norm, 1e-12)
    r = 0.5 * (edges[1:] + edges[:-1])
    return {"r": r, "g": g, "rmax": rmax}


# ----------------------------------------------------------------- nearest-neighbor dists
def nn_distances(coords):
    """Per-point nearest-neighbor distance (within the same cloud). (N,) array.
    Catches unphysical overlaps (too small) and over-smoothing (too uniform)."""
    a = _as2d(coords)
    if len(a) < 2:
        return np.zeros(0, np.float32)
    D = _pdist(a)
    np.fill_diagonal(D, np.inf)
    return D.min(1).astype(np.float32)


# ----------------------------------------------------------------- clustering (substructure)
def cluster_labels(coords, link_length=DEFAULT_LINK_LENGTH):
    """Single-linkage (DBSCAN with min_samples=1) clustering via union-find at ``eps``.

    The substructure detector: high-energy cascades fragment into spatially separated
    sub-cascades; this counts and sizes them. Returns (N,) int labels (-1 only if empty).
    """
    a = _as2d(coords)
    n = len(a)
    if n == 0:
        return np.zeros(0, np.int64)
    if n == 1:
        return np.zeros(1, np.int64)
    D = _pdist(a)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where((D <= link_length) & (np.triu(np.ones_like(D, bool), 1)))
    for i, j in zip(ii.tolist(), jj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    # relabel to 0..K-1
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def cluster_stats(coords, link_length=DEFAULT_LINK_LENGTH):
    """Return dict {n_clusters, sizes (sorted desc), largest_frac} at one link length."""
    labels = cluster_labels(coords, link_length)
    if len(labels) == 0:
        return {"n_clusters": 0, "sizes": [], "largest_frac": 0.0}
    sizes = np.bincount(labels)
    sizes = np.sort(sizes)[::-1]
    return {"n_clusters": int(len(sizes)),
            "sizes": sizes.astype(int).tolist(),
            "largest_frac": float(sizes[0] / sizes.sum())}


def cluster_spectrum(coords, link_scales=DEFAULT_LINK_SCALES):
    """Fragmentation curve: number of clusters as a function of linking length.

    Scale-robust substructure fingerprint. A real fragmented cascade keeps many clusters
    out to larger scales; a single merged blob (the typical model failure) collapses to one
    cluster quickly. Comparing these curves (gen vs real) directly scores "missed
    substructure" without picking a magic link length.

    Returns dict {scales: (K,), n_clusters: (K,), largest_frac: (K,)}.
    """
    a = _as2d(coords)
    scales = np.asarray(link_scales, np.float64)
    if len(a) == 0:
        z = np.zeros(len(scales))
        return {"scales": scales, "n_clusters": z, "largest_frac": z}
    ncl, lf = [], []
    D = _pdist(a)
    n = len(a)
    for eps in scales:
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ii, jj = np.where((D <= eps) & np.triu(np.ones_like(D, bool), 1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
        _, lab = np.unique([find(i) for i in range(n)], return_inverse=True)
        sizes = np.bincount(lab)
        ncl.append(len(sizes))
        lf.append(sizes.max() / sizes.sum())
    return {"scales": scales, "n_clusters": np.array(ncl, float),
            "largest_frac": np.array(lf, float)}


# ----------------------------------------------------------------- vac/SIA relationship
def vac_sia_separation(vac, sia):
    """Characteristic vacancy–interstitial separation. Returns dict:
       centroid_dist     : ||centroid(vac) - centroid(sia)||
       mean_cross_nn     : mean over SIAs of nearest-vacancy distance (the pair scale)."""
    v, s = _as2d(vac), _as2d(sia)
    if len(v) == 0 or len(s) == 0:
        return {"centroid_dist": 0.0, "mean_cross_nn": 0.0}
    cdist = float(np.linalg.norm(v.mean(0) - s.mean(0)))
    d = np.sqrt(np.maximum(((s[:, None, :] - v[None, :, :]) ** 2).sum(-1), 0.0))
    return {"centroid_dist": cdist, "mean_cross_nn": float(d.min(1).mean())}


# ----------------------------------------------------------------- richer structure metrics
def partial_rdf(a, b, rmax=None, nbins=24):
    """Cross/partial pair-correlation between two species (e.g. vac-vac, vac-SIA).
    If a is b, this is the same-species g(r); otherwise the inter-species correlation.
    Captures class-specific structure the union g(r) washes out."""
    a, b = _as2d(a), _as2d(b)
    same = a is b
    if len(a) == 0 or len(b) == 0:
        return {"r": np.zeros(nbins), "g": np.zeros(nbins), "rmax": rmax or 1.0}
    d = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))
    dists = d[np.triu_indices(len(a), 1)] if same else d.ravel()
    if rmax is None:
        rmax = float(np.percentile(dists, 90)) + 1e-6
    edges = np.linspace(0, rmax, nbins + 1)
    counts, _ = np.histogram(dists, bins=edges)
    shell = (4 / 3) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = len(b) / ((4 / 3) * np.pi * rmax ** 3)
    norm = (0.5 if same else 1.0) * len(a) * rho * shell
    return {"r": 0.5 * (edges[1:] + edges[:-1]), "g": counts / np.maximum(norm, 1e-12),
            "rmax": rmax}


def cross_nn_distances(vac, sia):
    """For each SIA, distance to its nearest vacancy (and vice versa pooled). This is the
    Frenkel-pair separation DISTRIBUTION — a far better vac-SIA metric than a centroid
    distance, and exactly what the pair-relative model targets."""
    v, s = _as2d(vac), _as2d(sia)
    if len(v) == 0 or len(s) == 0:
        return np.zeros(0, np.float32)
    d = np.sqrt(np.maximum(((s[:, None, :] - v[None, :, :]) ** 2).sum(-1), 0.0))
    return d.min(1).astype(np.float32)          # nearest vacancy per SIA


def radius_of_gyration(coords):
    """Rg = sqrt(mean ||x - centroid||^2). Single scalar cascade-size descriptor."""
    a = _as2d(coords)
    if len(a) == 0:
        return 0.0
    return float(np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()))


def largest_cluster_frac(coords, link_length=DEFAULT_LINK_LENGTH):
    """Fraction of defects in the biggest sub-cascade at ``link_length``. ~1 = one merged
    blob; small = fragmented. A direct blob-vs-fragmented detector."""
    return cluster_stats(coords, link_length)["largest_frac"]


def gyration_anisotropy(coords):
    """Shape from the gyration-tensor eigenvalues: returns (Rg, asphericity in [0,1]).
    Asphericity 0 = spherical, ->1 = elongated/planar (sub-cascades create anisotropy)."""
    a = _as2d(coords)
    if len(a) < 2:
        return (0.0, 0.0)
    c = a - a.mean(0)
    evals = np.sort(np.linalg.eigvalsh(c.T @ c / len(a)))[::-1]
    l1, l2, l3 = evals
    tr = evals.sum()
    if tr < 1e-12:
        return (0.0, 0.0)
    b = l1 - 0.5 * (l2 + l3)                      # asphericity numerator
    return (float(np.sqrt(tr)), float(b / tr))


# ----------------------------------------------------------------- set-to-set distances
def chamfer(a, b):
    """Symmetric Chamfer distance between two point sets (mean of nearest dists both ways).
    Lower = closer. Returns float (inf-safe: empty vs nonempty -> large sentinel)."""
    a, b = _as2d(a), _as2d(b)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))
    return float(d.min(1).mean() + d.min(0).mean()) * 0.5


# ----------------------------------------------------------------- distribution distances
def wasserstein1d(a, b):
    """1-D Wasserstein-1 (earth-mover) distance between two samples. Pure numpy."""
    a = np.sort(np.asarray(a, np.float64).ravel())
    b = np.sort(np.asarray(b, np.float64).ravel())
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    q = np.linspace(0, 1, max(len(a), len(b)) * 2)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def js_divergence(p, q, eps=1e-12):
    """Jensen–Shannon divergence between two normalized histograms (same grid). [0, ln2]."""
    p = np.asarray(p, np.float64) + eps
    q = np.asarray(q, np.float64) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda x, y: np.sum(x * np.log(x / y))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def curve_l1(y_ref, y_gen):
    """Normalized L1 between two curves on the same grid (relative to ref scale)."""
    y_ref = np.asarray(y_ref, np.float64); y_gen = np.asarray(y_gen, np.float64)
    denom = np.abs(y_ref).sum() + 1e-12
    return float(np.abs(y_ref - y_gen).sum() / denom)
