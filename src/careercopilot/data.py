"""Loading and sampling resume-job pairs."""
from __future__ import annotations

import random

from .schema import Pair, load_pairs
from .text import anonymise, split_with_context


def ensure_bullets(pairs: list[Pair]) -> list[Pair]:
    for p in pairs:
        if not p.bullets or not p.contexts:
            p.bullets, p.contexts = split_with_context(p.resume)
    return pairs


def load(path) -> list[Pair]:
    return ensure_bullets(load_pairs(path))


def _norm_label(v, names=None) -> str | None:
    if isinstance(v, int) and names:
        v = names[v]
    s = str(v).lower()
    if "good" in s:
        return "good"
    if "potential" in s:
        return "potential"
    if s.startswith("no") or "no fit" in s:
        return "none"
    return None


def _pick(columns, *keys):
    for k in keys:
        for c in columns:
            if k in c.lower():
                return c
    raise KeyError(f"none of {keys} in columns {columns}")


def sample_from_hf(cfg: dict) -> list[Pair]:
    """Sample `per_label` pairs per fit level from the Hugging Face dataset."""
    from datasets import load_dataset
    d = cfg["data"]
    ds = load_dataset(d["hf_dataset"], split=d["hf_split"])
    cols = ds.column_names
    rc, jc, lc = _pick(cols, "resume"), _pick(cols, "job"), _pick(cols, "label", "fit")
    feat = ds.features.get(lc)
    names = getattr(feat, "names", None)
    print(f"[data] {len(ds)} rows; using columns resume={rc!r} job={jc!r} label={lc!r}")

    by_fit: dict[str, list[int]] = {"good": [], "potential": [], "none": []}
    for i, v in enumerate(ds[lc]):
        f = _norm_label(v, names)
        if f:
            by_fit[f].append(i)
    rng = random.Random(cfg["seed"])
    pairs = []
    for fit, idxs in by_fit.items():
        rng.shuffle(idxs)
        taken = 0
        for i in idxs:
            row = ds[i]
            resume = anonymise(row[rc])
            bullets, contexts = split_with_context(resume)
            if len(bullets) < 6:          # skip resumes we can't split into usable evidence
                continue
            pairs.append(Pair(pair_id=f"{fit}-{i}", resume=resume, job=anonymise(row[jc]),
                              fit=fit, bullets=bullets, contexts=contexts))
            taken += 1
            if taken == d["per_label"]:
                break
        print(f"[data] {fit}: sampled {taken} of {len(idxs)}")
    return pairs


def sample_other_resumes(cfg: dict, exclude: set[str], n: int) -> list[tuple[str, str]]:
    """Resumes NOT used in the test pairs, for building the verifier's fine-tuning set
    (keeps training and test resumes apart -- no leakage)."""
    from datasets import load_dataset
    d = cfg["data"]
    ds = load_dataset(d["hf_dataset"], split=d["hf_split"])
    rc = _pick(ds.column_names, "resume")
    seen, out = set(exclude), []
    order = list(range(len(ds)))
    random.Random(cfg["seed"] + 1).shuffle(order)
    for i in order:
        text = anonymise(ds[i][rc])
        if text in seen:
            continue
        seen.add(text)
        out.append((f"train-{i}", text))
        if len(out) == n:
            break
    return out
