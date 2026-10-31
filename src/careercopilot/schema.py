"""Shared data structures. Both members read/write these -- change with care."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Usage:
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> "Usage":
        for k in asdict(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self


@dataclass
class Pair:
    pair_id: str
    resume: str          # anonymised resume text
    job: str             # job description text
    fit: str             # good | potential | none
    bullets: list[str]   # resume split into bullets (the evidence units)
    contexts: list[str] = field(default_factory=list)  # role/company line above each bullet

    def index_texts(self, contextual: bool = False) -> list[str]:
        """What the retriever indexes: the bullet, optionally prefixed with its context."""
        if not contextual or not self.contexts:
            return self.bullets
        return [f"{c} | {b}" if c else b for b, c in zip(self.bullets, self.contexts)]


@dataclass
class Claim:
    text: str
    line_idx: int                 # which output line it came from
    supported: bool | None = None
    score: float | None = None
    verifier: str | None = None


@dataclass
class TailorResult:
    pair_id: str
    variant: str                  # A | B | C | D
    fit: str
    lines: list[str]              # final tailored output, line 0 = summary
    requirements: list[str]
    evidence: dict = field(default_factory=dict)       # requirement idx -> list of bullet idx
    citations: list[list[str]] = field(default_factory=list)  # per line, e.g. ["E2", "E5"]
    pipeline_claims: list[Claim] = field(default_factory=list)  # verifier's view (C, D only)
    removed_lines: list[str] = field(default_factory=list)
    regenerations: int = 0
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    # variant D (controller): lines sent to a person instead of output, the per-line decision trace, and counts
    escalated_lines: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    actions: dict = field(default_factory=dict)


# ---------- JSONL helpers ----------

def _to_dict(r):
    return asdict(r) if hasattr(r, "__dataclass_fields__") else r


def write_jsonl(path, rows: Iterable) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_to_dict(r), ensure_ascii=False) + "\n")


def append_jsonl(path, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_pairs(path) -> list[Pair]:
    return [Pair(**d) for d in read_jsonl(path)]


def load_results(path) -> list[TailorResult]:
    out = []
    for d in read_jsonl(path):
        d["usage"] = Usage(**d["usage"])
        d["pipeline_claims"] = [Claim(**c) for c in d["pipeline_claims"]]
        out.append(TailorResult(**d))
    return out
