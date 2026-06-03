#!/usr/bin/env python
"""Process the raw_data/ cascade dumps into a single .npz for easy, fast, dependency-free use.

raw_data/ holds 9 continuous-energy ranges (0-300 keV); each is a flat dir of LAMMPS dumps
({ID}[_min]_{vac,sia,sia_site}.dump) + metadata.json (cascade_id -> pka_energy_eV, supercell).
We read the minimized vac+sia clouds (matching the modeling pipeline) with a fast pure-numpy
parser (no ovito) and pack everything into a flat .npz.

IMPORTANT: the supercell scales with energy (sc50 -> sc280), so absolute coordinate frames
differ across ranges. We therefore also store each cascade's centroid; downstream
normalization should center PER-CASCADE (a single global centroid is meaningless here).

Output .npz keys (loadable directly by cascaide.eval.io.load_subset, which reads the first
five):  vac_xyz, sia_xyz, vac_ptr, sia_ptr, energy(keV)  + centroid, n_vac, n_sia,
energy_range, supercell, summary_json.

    python scripts/process_raw_data.py --raw_dir raw_data --out data/cascaide_cascades.npz
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np


def read_xyz(path):
    """Fast LAMMPS-dump xyz reader. Robust to extra columns (e.g. sia's c_myKE c_myPE) by
    locating x/y/z by NAME in the `ITEM: ATOMS` header. Returns (N,3) float32."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return np.zeros((0, 3), np.float32)
    for i, l in enumerate(lines):
        if l.startswith("ITEM: ATOMS"):
            cols = l.split()[2:]
            try:
                xi, yi, zi = cols.index("x"), cols.index("y"), cols.index("z")
            except ValueError:
                return np.zeros((0, 3), np.float32)
            out = np.empty((len(lines) - i - 1, 3), np.float32)
            k = 0
            for row in lines[i + 1:]:
                p = row.split()
                if len(p) > zi:
                    out[k, 0] = p[xi]; out[k, 1] = p[yi]; out[k, 2] = p[zi]; k += 1
            return out[:k]
    return np.zeros((0, 3), np.float32)


def _load_one(task):
    vf, sf = task
    return read_xyz(vf), read_xyz(sf)


def scan(raw_dir, state):
    """Return [(vac_path, sia_path, energy_keV, energy_range, supercell), ...]."""
    suffix = "_min" if state == "min" else ""
    samples = []
    for ed in sorted(glob.glob(os.path.join(raw_dir, "*keV"))):
        meta = os.path.join(ed, "metadata.json")
        if not os.path.exists(meta):
            continue
        with open(meta) as f:
            data = json.load(f)
        sm = data.get("simulation_metadata", {})
        supercell = sm.get("supercell") or "?"
        rng = os.path.basename(ed)
        emap = {c["cascade_id"]: c.get("pka_energy_eV")
                for c in data.get("cascades", [])}
        pat = "*_min_vac.dump" if state == "min" else "*_vac.dump"
        for vf in sorted(glob.glob(os.path.join(ed, pat))):
            base = os.path.basename(vf)
            if state != "min" and "_min_" in base:
                continue
            try:
                cid = int(base.split("_")[0])
            except ValueError:
                continue
            sf = vf.replace(f"{suffix}_vac.dump", f"{suffix}_sia.dump")
            e = emap.get(cid)
            if os.path.exists(sf) and e is not None:
                samples.append((vf, sf, e / 1000.0, rng, supercell))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="raw_data")
    ap.add_argument("--out", default="data/cascaide_cascades.npz")
    ap.add_argument("--state", choices=["min", "unmin"], default="min")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    samples = scan(args.raw_dir, args.state)
    if not samples:
        raise SystemExit(f"No cascades found under {args.raw_dir}")
    print(f"[process] {len(samples)} cascades ({args.state}) across "
          f"{len({s[3] for s in samples})} energy ranges; reading with {args.workers} workers...")

    tasks = [(s[0], s[1]) for s in samples]
    with Pool(args.workers) as pool:
        clouds = pool.map(_load_one, tasks, chunksize=16)

    vac_parts, sia_parts, vp, sp = [], [], [0], [0]
    energy, erange, supercell, centroid = [], [], [], []
    for (vf, sf, e, rng, sc), (vac, sia) in zip(samples, clouds):
        vac_parts.append(vac); sia_parts.append(sia)
        vp.append(vp[-1] + len(vac)); sp.append(sp[-1] + len(sia))
        energy.append(e); erange.append(rng); supercell.append(sc)
        uni = np.concatenate([c for c in (vac, sia) if len(c)], 0) if (len(vac) or len(sia)) \
            else np.zeros((1, 3), np.float32)
        centroid.append(uni.mean(0))

    vac_xyz = np.concatenate(vac_parts, 0).astype(np.float32) if vac_parts else np.zeros((0, 3), np.float32)
    sia_xyz = np.concatenate(sia_parts, 0).astype(np.float32) if sia_parts else np.zeros((0, 3), np.float32)
    energy = np.asarray(energy, np.float32)
    n_vac = np.diff(vp); n_sia = np.diff(sp)

    summary = {
        "raw_dir": args.raw_dir, "state": args.state, "n_cascades": len(samples),
        "energy_keV": {"min": float(energy.min()), "max": float(energy.max())},
        "energy_ranges": sorted(set(erange)),
        "supercells": sorted(set(supercell)),
        "n_defects": {"min": int(n_vac.min()), "max": int(n_vac.max()),
                      "mean": float(n_vac.mean())},
        "frenkel_violations": int(np.sum(n_vac != n_sia)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out,
             vac_xyz=vac_xyz, sia_xyz=sia_xyz,
             vac_ptr=np.asarray(vp, np.int64), sia_ptr=np.asarray(sp, np.int64),
             energy=energy, n_vac=n_vac.astype(np.int64), n_sia=n_sia.astype(np.int64),
             centroid=np.asarray(centroid, np.float32),
             energy_range=np.asarray(erange), supercell=np.asarray(supercell),
             summary_json=np.array(json.dumps(summary)))
    print(f"[process] wrote {args.out}")
    for k, v in summary.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
