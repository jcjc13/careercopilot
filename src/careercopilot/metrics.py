"""Metrics for E1-E3."""
from __future__ import annotations

import math
import random
import re
from collections import Counter


# ---------- E1: retrieval ----------

def recall_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    return float(bool(set(ranked[:k]) & gold))


def reciprocal_rank(ranked: list[int], gold: set[int]) -> float:
    for r, i in enumerate(ranked, 1):
        if i in gold:
            return 1 / r
    return 0.0


# ---------- E2: verifiers (positive class = fabricated = verifier says UNSUPPORTED) ----------

def prf(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


# ---------- E3: tailoring ----------

def introduced_facts(output: str, resume: str, terms: list[str]) -> dict:
    """Programmatic fabrication check (verify in code before using a judge):
    tools/technologies and numbers that appear in the tailored output but nowhere in the resume."""
    from .text import find_numbers, find_terms
    res_terms = {t.lower() for t in find_terms(resume, terms)}
    res_nums = set(find_numbers(resume))
    new_terms = sorted({t for t in find_terms(output, terms) if t.lower() not in res_terms})
    new_nums = sorted({n for n in find_numbers(output) if n not in res_nums})
    return {"new_tools": new_terms, "new_numbers": new_nums}

def bow_cosine(a: str, b: str) -> float:
    """Bag-of-words cosine; fallback when sentence-transformers is not installed."""
    ta, tb = Counter(re.findall(r"\w+", a.lower())), Counter(re.findall(r"\w+", b.lower()))
    dot = sum(ta[w] * tb[w] for w in ta)
    na, nb = math.sqrt(sum(v * v for v in ta.values())), math.sqrt(sum(v * v for v in tb.values()))
    return dot / (na * nb) if na and nb else 0.0


class Similarity:
    """Embedding cosine if available, else bag-of-words. Report which one was used."""

    def __init__(self, model_name: str | None = None):
        self.model, self.kind = None, "bow"
        if model_name:
            try:
                from .retrieval import _st_model
                self.model, self.kind = _st_model(model_name), f"embedding:{model_name}"
            except Exception as e:
                print(f"[similarity] embeddings unavailable ({e!r}); using bag-of-words cosine")

    def __call__(self, a: str, b: str) -> float:
        if self.model is None:
            return bow_cosine(a, b)
        va, vb = self.model.encode([a, b], normalize_embeddings=True)
        return float(va @ vb)


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> dict:
    """Paired test, normal approximation (fine for n >= ~10). Uses scipy when available."""
    try:
        from scipy.stats import wilcoxon
        d = [a - b for a, b in zip(x, y)]
        if all(v == 0 for v in d):
            return {"n": 0, "p": 1.0}
        r = wilcoxon(x, y)
        return {"n": len(d), "statistic": float(r.statistic), "p": float(r.pvalue)}
    except ImportError:
        pass
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n == 0:
        return {"n": 0, "p": 1.0}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:  # average ranks for ties
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    mean, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mean) / sd if sd else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"n": n, "statistic": w_plus, "p": p}


def paired_bootstrap_mean_ci(differences: list[float], replicates: int = 10000, seed: int = 469,
                             confidence: float = 0.95) -> dict:
    """Percentile CI for a paired mean difference, resampling independent pair-level differences."""
    if not differences:
        raise ValueError("at least one paired difference is required")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    n = len(differences)
    observed = sum(differences) / n
    rng = random.Random(seed)
    estimates = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates)
    )

    def percentile(q):
        index = (len(estimates) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return estimates[lower]
        weight = index - lower
        return estimates[lower] * (1 - weight) + estimates[upper] * weight

    alpha = (1 - confidence) / 2
    return {
        "n_pairs": n,
        "mean_difference": observed,
        "ci_lower": percentile(alpha),
        "ci_upper": percentile(1 - alpha),
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
    }


def roc_auc(y_true: list[bool], score: list[float]) -> float:
    """ROC-AUC of `score` (higher = more likely positive); ties count half. nan if one class is missing."""
    pos = [s for t, s in zip(y_true, score) if t]
    neg = [s for t, s in zip(y_true, score) if not t]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def mcnemar_exact(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """McNemar's test on paired per-claim correctness (exact two-sided binomial on the discordant claims)."""
    only_a = sum(a and not b for a, b in zip(correct_a, correct_b))
    only_b = sum(b and not a for a, b in zip(correct_a, correct_b))
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_a": only_a, "only_b": only_b, "p": p}
