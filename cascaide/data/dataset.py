import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from functools import lru_cache
import glob
from typing import Dict, List, Optional,Tuple
from cascaide.encoding.base import CoordinateEncoder

class CascadeDataset(Dataset):
    def __init__(self, data_root, compute_centroid=True, max_samples=None):
        self.data_root = data_root

        self.samples = self._scan_data_root()

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.global_centroid = None
        if compute_centroid:
            self._compute_global_centroid()

        print(f"Loaded {len(self.samples)} samples")
        if self.global_centroid is not None:
            print(f"Global centroid: {self.global_centroid}")

    def _scan_data_root(self):
        samples = []

        energy_dirs = sorted(glob.glob(os.path.join(self.data_root, "*keV")))

        for energy_dir in energy_dirs:

            metadata_path = os.path.join(energy_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                continue

            with open(metadata_path, "r") as f:
                data = json.load(f)

            # cascade_id → energy mapping
            energy_map = {
                c["cascade_id"]: c["pka_energy_eV"]
                for c in data["cascades"]
            }

            vac_files = sorted(glob.glob(
                os.path.join(energy_dir, "*_min_vac.dump")
            ))

            for vac_file in vac_files:
                base = os.path.basename(vac_file)

                cascade_id = int(base.split("_")[0])

                sia_file = vac_file.replace("_min_vac.dump", "_min_sia.dump")

                if os.path.exists(sia_file) and cascade_id in energy_map:
                    energy = energy_map[cascade_id]
                    samples.append((vac_file, sia_file, energy))

        return samples

    @lru_cache(maxsize=5000)
    def _load_coordinates(self, filepath):
        try:
            from ovito.io import import_file
            pipeline = import_file(filepath)
            data = pipeline.compute()
            n_atoms = data.particles.count
            if n_atoms == 0:
                return np.zeros((0, 3), dtype=np.float32)
            positions = np.array(data.particles.positions, dtype=np.float32)
            return positions
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return np.zeros((0, 3), dtype=np.float32)

    def _compute_global_centroid(self):
        print("Computing global centroid (streaming)...")

        total_sum = np.zeros(3, dtype=np.float64)
        total_count = 0

        for vac_file, sia_file, _ in self.samples:
            vac = self._load_coordinates(vac_file)
            sia = self._load_coordinates(sia_file)

            if len(vac) > 0:
                total_sum += vac.sum(axis=0)
                total_count += len(vac)

            if len(sia) > 0:
                total_sum += sia.sum(axis=0)
                total_count += len(sia)

        if total_count > 0:
            self.global_centroid = total_sum / total_count
        else:
            self.global_centroid = np.zeros(3)

    def compute_required_norm_factor(self):
        if self.global_centroid is None:
            self._compute_global_centroid()

        print("Scanning for max spatial extent...")
        max_dist = 0.0

        for vac_file, sia_file, _ in self.samples:
            vac = self._load_coordinates(vac_file)
            sia = self._load_coordinates(sia_file)

            for coords in [vac, sia]:
                if len(coords) > 0:
                    # Calculate distance from global centroid for every atom
                    dists = np.linalg.norm(coords - self.global_centroid, axis=1)
                    local_max = np.max(dists)
                    if local_max > max_dist:
                        max_dist = local_max

        print(f"Maximum atom distance found: {max_dist:.2f} Å")
        # Return a value slightly larger than the max to avoid edge clipping
        return float(np.ceil(max_dist * 1.1 / 10) * 10)

    def compute_tanh_params(self, percentile: float = 99.0):
        """Per-axis tanhP99 params on the GLOBALLY-CENTERED cloud:
            c = per-axis median of (coord - G)         (robust center)
            s = per-axis ``percentile`` of |centered - c|  (robust scale)
        Returns (c, s) as float32 (3,) arrays. Mirrors the monolith's
        _compute_tanh_params (computed on centered coords; G handled separately
        by the encoder's ``centroid``)."""
        if self.global_centroid is None:
            self._compute_global_centroid()
        G = np.asarray(self.global_centroid, dtype=np.float32)

        print(f"Computing tanhP{percentile:g} params (centered cloud)...")
        chunks = []
        for vac_file, sia_file, _ in self.samples:
            vac = self._load_coordinates(vac_file)
            sia = self._load_coordinates(sia_file)
            if len(vac) > 0:
                chunks.append(vac - G)
            if len(sia) > 0:
                chunks.append(sia - G)

        if not chunks:
            return (np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))

        allc = np.vstack(chunks).astype(np.float32)
        c = np.median(allc, axis=0).astype(np.float32)
        s = np.percentile(np.abs(allc - c), percentile, axis=0).astype(np.float32)
        s = np.where(np.abs(s) < 1e-8, 1.0, s).astype(np.float32)
        print(f"  tanh_center (median) = {c.round(3)}")
        print(f"  tanh_scale  (p{percentile:g} of |x-c|) = {s.round(3)}")
        return c, s

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vac_file, sia_file, energy = self.samples[idx]

        vac = self._load_coordinates(vac_file)
        sia = self._load_coordinates(sia_file)

        # # Center coordinates
        # if self.center_coords and self.global_centroid is not None:
        #     if len(vac) > 0:
        #         vac = vac - self.global_centroid
        #     if len(sia) > 0:
        #         sia = sia - self.global_centroid

        return {
            "vac_coords": torch.from_numpy(vac),
            "sia_coords": torch.from_numpy(sia),
            "energy": torch.tensor(energy, dtype=torch.float32),
        }


class EncodedCascadeDataset(Dataset):
    def __init__(self,
                 raw_dataset: 'CascadeDataset',
                 encoders: Dict[str, CoordinateEncoder],
                 energy_norm_factor: float = 1000.0):
        self.raw_dataset = raw_dataset
        self.encoders = encoders
        self.energy_norm_factor = energy_norm_factor
        self.global_centroid = raw_dataset.global_centroid

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]
        vac = sample["vac_coords"]
        sia = sample["sia_coords"]
        energy = sample["energy"]

        out = {}
        for name, enc in self.encoders.items():
            out[name] = enc.encode(vac, sia)

        out["energy"] = energy / self.energy_norm_factor
        out["n_vac"] = len(vac)
        out["n_sia"] = len(sia)
        out["vac_coords"] = vac
        out["sia_coords"] = sia
        return out


def collate_fn(batch):
    out = {}
    for k in ["vac_coords", "sia_coords"]:
        if k in batch[0]:
            out[k] = [item[k] for item in batch]

    for k in batch[0]:
        if k in ("vac_coords", "sia_coords"):
            continue
        v = batch[0][k]
        if torch.is_tensor(v):
            out[k] = torch.stack([item[k] for item in batch])
        else:
            out[k] = torch.tensor([item[k] for item in batch])
    return out