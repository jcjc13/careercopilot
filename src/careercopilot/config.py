from __future__ import annotations

import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """A config may name a `base:` file (relative to the repo root) and override only what differs."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        cfg = _merge(load_config(cfg.pop("base")), cfg)
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
