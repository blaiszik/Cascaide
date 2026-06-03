"""Frenkel pair-relative encoding.

The scorecard showed vac–SIA separation is off (~5.5 Å). Interstitials sit at a small,
characteristic offset from their vacancy, so modeling SIA as a *displacement* from its
paired vacancy collapses the dynamic range and lets the model learn the offset directly.
``n_vac == n_sia`` (Frenkel conservation) guarantees a perfect matching exists; we find the
min-cost one (Hungarian if scipy is present, else greedy nearest-neighbor).

A pair token is 6-D: ``[vac_x, vac_y, vac_z, disp_x, disp_y, disp_z]`` with
``sia = vac + disp``. One token per Frenkel pair — which also exploits conservation
(generate n_pairs tokens, get both populations).
"""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HUNGARIAN = True
except Exception:  # pragma: no cover
    _HUNGARIAN = False


def frenkel_pairing(vac, sia):
    """Return (row_ind, col_ind) matching vac[row] <-> sia[col], length min(nv, ns).
    Min total vac→SIA distance (Hungarian) or greedy nearest-neighbor fallback."""
    vac = np.asarray(vac, np.float32); sia = np.asarray(sia, np.float32)
    nv, ns = len(vac), len(sia)
    if nv == 0 or ns == 0:
        return np.array([], int), np.array([], int)
    cost = np.linalg.norm(vac[:, None, :] - sia[None, :, :], axis=-1)
    if _HUNGARIAN:
        return linear_sum_assignment(cost)
    # greedy: repeatedly take the globally-closest remaining pair
    row, col = [], []
    used_v, used_s = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for i, j in order:
        if i in used_v or j in used_s:
            continue
        used_v.add(i); used_s.add(j); row.append(i); col.append(j)
        if len(row) == min(nv, ns):
            break
    return np.array(row, int), np.array(col, int)


def to_pair_relative(vac, sia):
    """(vac, sia) -> (vac_matched (Np,3), disp (Np,3)) with disp = sia - vac.
    Unmatched extras (when nv != ns) are dropped — real data is balanced."""
    r, c = frenkel_pairing(vac, sia)
    if len(r) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    vm = np.asarray(vac, np.float32)[r]
    sm = np.asarray(sia, np.float32)[c]
    return vm, (sm - vm).astype(np.float32)


def from_pair_relative(vac, disp):
    """Inverse: (vac, disp) -> (vac, sia) with sia = vac + disp."""
    vac = np.asarray(vac, np.float32)
    return vac, (vac + np.asarray(disp, np.float32)).astype(np.float32)


def pairs_to_tokens(vac, disp):
    """(Np,3),(Np,3) -> (Np,6) pair tokens [vac | disp]."""
    if len(vac) == 0:
        return np.zeros((0, 6), np.float32)
    return np.concatenate([vac, disp], axis=1).astype(np.float32)


def tokens_to_pairs(tokens):
    """(Np,6) -> (vac (Np,3), disp (Np,3))."""
    t = np.asarray(tokens, np.float32)
    return t[:, :3], t[:, 3:]
