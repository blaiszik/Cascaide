import yaml
import random
import os
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace


def _to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(v) for v in d]
    return d


def load_config(path: str):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw), raw


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(base_dir: str, name: str) -> Path:
    run_dir = Path(base_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    return run_dir