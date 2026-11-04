"""Compare resume chunking strategies against the labeled E1 retrieval set.

    python scripts/compare_e1_chunking.py

The existing E1 gold bullet IDs refer to the sentence-split chunks. For each strategy,
those gold bullets are mapped to candidate chunks by exact text containment. The script
fails if any gold bullet cannot be mapped, rather than silently scoring an invalid set.
"""
from __future__ import annotations

import csv
import json
import re
import time

from _common import load_config, load_pairs_or_exit, parser, resolve

from careercopilot.metrics import recall_at_k, reciprocal_rank
from careercopilot.retrieval import build_retriever
from careercopilot.text import split_with_context

STRATEGIES = {
    "sentence": True,
    "preserve": False,
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _map_gold(old_bullets: list[str], chunks: list[str], gold_ids: set[int], pair_id: str) -> set[int]:
    mapped = set()
    for old_id in gold_ids:
        if old_id < 0 or old_id >= len(old_bullets):
            raise ValueError(f"{pair_id}: gold bullet id {old_id} is out of range")
        gold_text = _normalise(old_bullets[old_id])
        matches = {i for i, chunk in enumerate(chunks) if gold_text in _normalise(chunk)}
        if not matches:
            raise ValueError(
                f"{pair_id}: cannot map labeled bullet {old_id} to a chunk: "
                f"{old_bullets[old_id]!r}"
            )
        mapped.update(matches)
    return mapped


def _gold_ids(value: str) -> set[int]:
    return {int(i) for i in re.split(r"[;,]", value) if i.strip()}


def evaluate(config: str = "configs/default.yaml", limit: int | None = None) -> dict:
    cfg = load_config(config)
    pairs = {p.pair_id: p for p in load_pairs_or_exit(cfg, limit)}
    labels_path = resolve(cfg["paths"]["labels_dir"]) / "e1_labels.csv"
    with labels_path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["gold_bullet_ids"].strip()]
    if not rows:
        raise SystemExit(f"no labeled E1 rows in {labels_path}")
    missing_pairs = sorted({r["pair_id"] for r in rows} - pairs.keys())
    if missing_pairs:
        raise ValueError(f"labeled pairs missing from prepared data: {missing_pairs}")

    retriever = build_retriever(cfg)
    k = cfg["retrieval"]["top_k"]
    results = {}
    for name, split_sentences in STRATEGIES.items():
        recall1 = recallk = mrr = 0.0
        mapped_count = chunk_count = 0
        per_row = []
        started = time.perf_counter()
        for row in rows:
            pair = pairs[row["pair_id"]]
            chunks, contexts = split_with_context(pair.resume, split_sentences=split_sentences)
            gold = _map_gold(pair.bullets, chunks, _gold_ids(row["gold_bullet_ids"]), pair.pair_id)
            if cfg["retrieval"].get("contextual", False):
                texts = [f"{c} | {b}" if c else b for b, c in zip(chunks, contexts)]
            else:
                texts = chunks
            ranked = [i for i, _ in retriever.rank(row["requirement"], texts, max(k, 10))]
            r1 = recall_at_k(ranked, gold, 1)
            rk = recall_at_k(ranked, gold, k)
            rr = reciprocal_rank(ranked, gold)
            recall1 += r1
            recallk += rk
            mrr += rr
            mapped_count += len(gold)
            chunk_count += len(chunks)
            per_row.append({
                "pair_id": row["pair_id"],
                "req_id": row["req_id"],
                "gold_chunk_ids": sorted(gold),
                "top": ranked[:k],
                "recall@1": r1,
                f"recall@{k}": rk,
                "reciprocal_rank": rr,
            })
        n = len(rows)
        results[name] = {
            "n": n,
            "average_chunks_per_resume_query": chunk_count / n,
            "average_mapped_gold_chunks_per_query": mapped_count / n,
            "recall@1": recall1 / n,
            f"recall@{k}": recallk / n,
            "mrr": mrr / n,
            "ms_per_query": 1000 * (time.perf_counter() - started) / n,
            "rows": per_row,
        }

    out_dir = resolve(cfg["paths"]["runs_dir"]) / "e1-chunking-comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "e1_chunking_results.json"
    out_path.write_text(json.dumps({
        "retriever": retriever.name,
        "top_k": k,
        "contextual": cfg["retrieval"].get("contextual", False),
        "label_mapping": "exact text containment from old labeled bullets to candidate chunks",
        "strategies": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'strategy':<12}{'n':>5}{'chunks/query':>15}{'R@1':>9}{f'R@{k}':>9}{'MRR':>9}{'ms/query':>12}")
    for name, result in results.items():
        print(f"{name:<12}{result['n']:>5}{result['average_chunks_per_resume_query']:>15.1f}"
              f"{result['recall@1']:>9.3f}{result[f'recall@{k}']:>9.3f}"
              f"{result['mrr']:>9.3f}{result['ms_per_query']:>12.1f}")
    print(f"Wrote {out_path}")
    return results


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    evaluate(args.config, args.limit)
