from __future__ import annotations

import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def resolve(p: str | Path) -> Path:
    """Paths in the config are relative to the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def run_dir(cfg: dict) -> Path:
    p = resolve(cfg["paths"]["runs_dir"]) / cfg["run_name"]
    p.mkdir(parents=True, exist_ok=True)
    return p
