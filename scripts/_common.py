"""Shared bits for the scripts: import path, common CLI flags, and building components."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careercopilot.config import load_config, resolve, run_dir, set_seed  # noqa: E402,F401
from careercopilot.data import load  # noqa: E402
from careercopilot.llm import build_llm  # noqa: E402,F401


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--mock", action="store_true", help="use the offline MockLLM (no API key, outputs not meaningful)")
    p.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    return p


def pairs_path(cfg) -> Path:
    return resolve(cfg["data"]["out_dir"]) / "pairs.jsonl"


def load_pairs_or_exit(cfg, limit=None):
    p = pairs_path(cfg)
    if not p.exists():
        sys.exit(f"{p} not found -- run scripts/prepare_data.py first (or with --sample)")
    return load(p)[:limit]


try:
    from tqdm import tqdm  # noqa: F401
except ImportError:
    def tqdm(x, **_):
        return x
