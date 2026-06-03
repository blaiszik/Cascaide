"""Per-cascade-centered set dataset + normalizer for the mixed-supercell raw_data.

The raw_data supercell scales with energy (sc50 -> sc280), so absolute coordinate frames
differ wildly across energy ranges and a single GLOBAL centroid is meaningless. This
normalizer centers EACH cascade on its own (vac+SIA) centroid, then applies one GLOBAL
per-axis p99 scale — so cloud SHAPES are comparable across supercells while the size signal
(bigger cascades at higher energy) is preserved in normalized units.

Generated clouds come out origin-centered (absolute position is physically arbitrary). To
stay drop-in with the existing sp_cas generate/load path, ``to_dict`` serializes as a
CoordNormalizer-compatible dict with ``G = [0,0,0]`` and the global ``s`` — so
``sp_cas.CoordNormalizer.from_dict`` + ``sp_cas.generate`` decode correctly (coords*s + 0).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

TYPE_VAC, TYPE_SIA = 0, 1


def _union(s):
    parts = [c for c in (s["vac"], s["sia"]) if len(c)]
    return np.concatenate(parts, 0) if parts else np.zeros((0, 3), np.float32)


class PerCascadeNormalizer:
    """Per-cascade centering + global per-axis p99 scale."""

    def __init__(self, percentile=99.0):
        self.percentile = percentile
        self.s = np.ones(3, np.float32)
        self.mode = "per_cascade"

    def fit(self, samples):
        chunks = []
        for s in samples:
            u = _union(s)
            if len(u):
                chunks.append(u - u.mean(0))          # center on this cascade's centroid
        allc = np.vstack(chunks) if chunks else np.zeros((0, 3), np.float32)
        if len(allc):
            p = np.percentile(np.abs(allc), self.percentile, axis=0)
            self.s = np.where(np.abs(p) < 1e-8, 1.0, p).astype(np.float32)
        return self

    def encode(self, vac, sia):
        """Center vac+SIA on their SHARED centroid, then scale. Returns (vac_n, sia_n)."""
        u = np.concatenate([c for c in (vac, sia) if len(c)], 0) \
            if (len(vac) or len(sia)) else np.zeros((1, 3), np.float32)
        c = u.mean(0)
        vn = ((vac - c) / self.s).astype(np.float32) if len(vac) else np.zeros((0, 3), np.float32)
        sn = ((sia - c) / self.s).astype(np.float32) if len(sia) else np.zeros((0, 3), np.float32)
        return vn, sn

    def inverse(self, coords):
        """Decode generated (origin-centered) coords back to physical scale. No re-center —
        generated cascades have no meaningful absolute position. Named to match
        CoordNormalizer.inverse so it's drop-in in the trainer/generator."""
        return (np.asarray(coords, np.float32) * self.s).astype(np.float32)

    # CoordNormalizer-compatible serialization (G=0 so sp_cas decode = coords*s)
    def to_dict(self):
        return {"mode": self.mode, "percentile": self.percentile,
                "G": [0.0, 0.0, 0.0], "s": self.s.tolist()}

    @classmethod
    def from_dict(cls, d):
        o = cls(d.get("percentile", 99.0))
        o.s = np.asarray(d["s"], np.float32)
        return o


class SetDataset(Dataset):
    """Per-cascade-centered set dataset. ``__getitem__`` -> (coords, types, energy, n_pairs),
    compatible with sp_cas.collate_dynamic. ``count_energy``/``count_npairs`` include empty
    cascades (for the count head)."""

    def __init__(self, samples, normalizer, energy_divisor=300.0):
        self.normalizer = normalizer
        self.items = []
        e_all, n_all = [], []
        for s in samples:
            vn, sn = normalizer.encode(s["vac"], s["sia"])
            nv, ns = len(vn), len(sn)
            n_pairs = max(nv, ns)
            e_all.append(s["energy"] / energy_divisor)
            n_all.append(n_pairs)
            if nv + ns > 0:
                coords = np.concatenate([c for c in (vn, sn) if len(c)], 0).astype(np.float32)
                types = np.concatenate([np.full(nv, TYPE_VAC, np.int64),
                                        np.full(ns, TYPE_SIA, np.int64)])
                self.items.append((torch.from_numpy(coords), torch.from_numpy(types),
                                   torch.tensor(s["energy"] / energy_divisor, dtype=torch.float32),
                                   n_pairs))
        self.count_energy = torch.tensor(e_all, dtype=torch.float32)
        self.count_npairs = torch.tensor(n_all, dtype=torch.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
