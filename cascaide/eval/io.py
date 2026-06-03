"""Cache a small REAL cascade subset to a dependency-free .npz, and load it back.

Why: reading LAMMPS .dump files needs ovito (heavy), and per-file reads are slow. We pay
that cost ONCE here to materialize a small, stratified real subset as a flat .npz that
loads in <1s with numpy alone — making real data as convenient as the synthetic fallback
for fast screening, CI, and autoresearch loops.

A "sample" is one cascade: (vac coords [Nv,3], sia coords [Ns,3], energy in keV).
Storage is flat (concatenated coords + offset pointers) so NO pickle / allow_pickle is
needed — the file is portable and safe to load.
"""
import os
import glob
import json
import numpy as np


# ----------------------------------------------------------------------------- loading
def _load_dump(path):
    """Read xyz positions from a LAMMPS dump via ovito. Returns (N,3) float32.

    Mirrors cascaide.data.dataset.CascadeDataset._load_coordinates. ovito handles the
    differing column layouts of *_vac.dump (id type x y z) and *_sia.dump
    (id type x y z c_myKE c_myPE) transparently — a hand-rolled parser does not.
    """
    try:
        from ovito.io import import_file
        data = import_file(path).compute()
        if data.particles.count == 0:
            return np.zeros((0, 3), np.float32)
        return np.asarray(data.particles.positions, dtype=np.float32)
    except Exception as e:  # pragma: no cover - depends on ovito + real files
        print(f"[eval.io] error loading {path}: {e}")
        return np.zeros((0, 3), np.float32)


def _scan(data_root):
    """Return [(vac_file, sia_file, energy_keV), ...] across all *keV dirs."""
    samples = []
    for ed in sorted(glob.glob(os.path.join(data_root, "*keV"))):
        meta = os.path.join(ed, "metadata.json")
        if not os.path.exists(meta):
            continue
        with open(meta) as f:
            data = json.load(f)
        emap = {c["cascade_id"]: c["pka_energy_eV"]
                for c in data.get("cascades", [])}
        for vf in sorted(glob.glob(os.path.join(ed, "*_min_vac.dump"))):
            try:
                cid = int(os.path.basename(vf).split("_")[0])
            except ValueError:
                continue
            sf = vf.replace("_min_vac.dump", "_min_sia.dump")
            if os.path.exists(sf) and cid in emap:
                samples.append((vf, sf, emap[cid] / 1000.0))  # eV -> keV
    return samples


def cache_real_subset(data_root, out_path, per_energy=64, max_total=None,
                      seed=42, verbose=True):
    """Read a stratified real subset and save it flat to ``out_path`` (.npz).

    per_energy : cascades to sample from EACH *keV directory (stratified). Set None to
                 take everything (then ``max_total`` can cap the grand total).
    Returns a summary dict (also embedded in the .npz under 'summary_json').
    """
    rng = np.random.default_rng(seed)
    all_samples = _scan(data_root)
    if not all_samples:
        raise FileNotFoundError(f"No cascades found under {data_root}")

    # stratify by energy directory
    by_energy = {}
    for vf, sf, e in all_samples:
        by_energy.setdefault(round(e, 3), []).append((vf, sf, e))

    chosen = []
    for e, lst in sorted(by_energy.items()):
        idx = rng.permutation(len(lst))
        take = len(lst) if per_energy is None else min(per_energy, len(lst))
        chosen.extend(lst[i] for i in idx[:take])
    if max_total is not None and len(chosen) > max_total:
        idx = rng.permutation(len(chosen))[:max_total]
        chosen = [chosen[i] for i in sorted(idx)]

    if verbose:
        print(f"[eval.io] caching {len(chosen)} cascades from {len(by_energy)} "
              f"energies under {data_root}")

    vac_parts, sia_parts, energies = [], [], []
    vac_ptr, sia_ptr = [0], [0]
    for k, (vf, sf, e) in enumerate(chosen):
        v, s = _load_dump(vf), _load_dump(sf)
        vac_parts.append(v); sia_parts.append(s); energies.append(e)
        vac_ptr.append(vac_ptr[-1] + len(v))
        sia_ptr.append(sia_ptr[-1] + len(s))
        if verbose and (k + 1) % 50 == 0:
            print(f"  read {k + 1}/{len(chosen)}")

    vac_xyz = (np.concatenate(vac_parts, 0) if vac_parts
               else np.zeros((0, 3), np.float32)).astype(np.float32)
    sia_xyz = (np.concatenate(sia_parts, 0) if sia_parts
               else np.zeros((0, 3), np.float32)).astype(np.float32)
    energies = np.asarray(energies, np.float32)

    n_vac = np.diff(vac_ptr); n_sia = np.diff(sia_ptr)
    summary = {
        "data_root": data_root,
        "n_cascades": len(chosen),
        "energies_keV": sorted({float(e) for e in energies}),
        "per_energy": per_energy,
        "n_vac": {"min": int(n_vac.min()), "max": int(n_vac.max()),
                  "mean": float(n_vac.mean())} if len(n_vac) else {},
        "frenkel_violations": int(np.sum(n_vac != n_sia)),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez(out_path,
             vac_xyz=vac_xyz, sia_xyz=sia_xyz,
             vac_ptr=np.asarray(vac_ptr, np.int64),
             sia_ptr=np.asarray(sia_ptr, np.int64),
             energy=energies,
             summary_json=np.array(json.dumps(summary)))
    if verbose:
        print(f"[eval.io] wrote {out_path}  ({len(chosen)} cascades, "
              f"frenkel_violations={summary['frenkel_violations']})")
    return summary


def load_subset(path):
    """Load a cached subset. Returns a list of dicts:
        {'vac': (Nv,3) float32, 'sia': (Ns,3) float32, 'energy': float (keV)}
    No pickle needed.
    """
    z = np.load(path)
    vac_xyz, sia_xyz = z["vac_xyz"], z["sia_xyz"]
    vp, sp, energy = z["vac_ptr"], z["sia_ptr"], z["energy"]
    out = []
    for i in range(len(energy)):
        out.append({
            "vac": vac_xyz[vp[i]:vp[i + 1]].astype(np.float32),
            "sia": sia_xyz[sp[i]:sp[i + 1]].astype(np.float32),
            "energy": float(energy[i]),
        })
    return out


def group_by_energy(samples, decimals=0):
    """Bucket samples into {energy_keV (rounded): [sample, ...]}."""
    groups = {}
    for s in samples:
        groups.setdefault(round(s["energy"], decimals), []).append(s)
    return dict(sorted(groups.items()))
