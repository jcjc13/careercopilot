"""E1: compare evidence retrievers -- TF-IDF (Topic 1) vs BM25 vs embeddings (Topics 2, 8) vs
hybrid + cross-encoder re-ranker (Topic 8); add `+ctx` to a method for contextual retrieval
(each bullet indexed with its role/company line, Topic 8 contextual RAG).

Step 1 -- make the labelling sheet (extracts each job's requirements with the LLM):
    python scripts/run_e1_retrieval.py --make-template --pairs 12
  -> data/labels/e1_bullets.csv   every resume bullet with its id, per pair (reference while labelling)
  -> data/labels/e1_labels.csv    one row per requirement; fill `gold_bullet_ids` with the ids of
                                  bullets that support it, separated by ';' (leave empty if none),
                                  and correct the requirement text if the LLM got it wrong
Step 2 -- evaluate:
    python scripts/run_e1_retrieval.py --methods tfidf,bm25,embedding,hybrid,bm25+ctx,embedding+ctx
  -> runs/<run>/e1_results.json + a printed table (Recall@1/3, MRR, ms per query)
"""
import csv
import json
import time

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, run_dir, tqdm

from careercopilot.generation import extract_requirements
from careercopilot.metrics import recall_at_k, reciprocal_rank
from careercopilot.retrieval import build_retriever


def make_template(cfg, llm, n_pairs):
    pairs = load_pairs_or_exit(cfg)
    # spread the labelled pairs across fit levels
    by_fit = {}
    for p in pairs:
        by_fit.setdefault(p.fit, []).append(p)
    chosen = []
    while len(chosen) < n_pairs and any(by_fit.values()):
        for f in list(by_fit):
            if by_fit[f] and len(chosen) < n_pairs:
                chosen.append(by_fit[f].pop(0))
    d = resolve(cfg["paths"]["labels_dir"])
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "e1_bullets.csv", "w", newline="", encoding="utf-8") as fb, \
            open(d / "e1_labels.csv", "w", newline="", encoding="utf-8") as fl:
        wb, wl = csv.writer(fb), csv.writer(fl)
        wb.writerow(["pair_id", "bullet_id", "bullet"])
        wl.writerow(["pair_id", "req_id", "requirement", "gold_bullet_ids", "labeller", "notes"])
        for p in tqdm(chosen, desc="requirements"):
            for i, b in enumerate(p.bullets):
                wb.writerow([p.pair_id, i, b])
            reqs, _ = extract_requirements(llm, p.job, cfg["tailoring"]["n_requirements"])
            for j, r in enumerate(reqs):
                wl.writerow([p.pair_id, j, r, "", "", ""])
    print(f"wrote {d / 'e1_bullets.csv'} and {d / 'e1_labels.csv'} for {len(chosen)} pairs -- now label them")


def evaluate(cfg, methods, k=3):
    pairs = {p.pair_id: p for p in load_pairs_or_exit(cfg)}
    path = resolve(cfg["paths"]["labels_dir"]) / "e1_labels.csv"
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["gold_bullet_ids"].strip()]
    if not rows:
        raise SystemExit(f"no labelled rows in {path} (requirements with no supporting bullet are skipped)")
    results = {}
    for m in methods:
        base, contextual = m.split("+")[0], m.endswith("+ctx")
        try:
            retr = build_retriever(cfg, base)
        except Exception as e:
            print(f"skip {m}: {e!r}")
            continue
        r1 = rk = mrr = 0.0
        t0 = time.perf_counter()
        per_row = []
        for r in rows:
            p = pairs[r["pair_id"]]
            gold = {int(x) for x in r["gold_bullet_ids"].replace(",", ";").split(";") if x.strip()}
            ranked = [i for i, _ in retr.rank(r["requirement"], p.index_texts(contextual), max(k, 10))]
            r1 += recall_at_k(ranked, gold, 1)
            rk += recall_at_k(ranked, gold, k)
            mrr += reciprocal_rank(ranked, gold)
            per_row.append({"pair_id": r["pair_id"], "req_id": r["req_id"], "top": ranked[:k],
                            "gold": sorted(gold), "hit": recall_at_k(ranked, gold, k)})
        n = len(rows)
        results[m] = {"n": n, "recall@1": r1 / n, f"recall@{k}": rk / n, "mrr": mrr / n,
                      "ms_per_query": 1000 * (time.perf_counter() - t0) / n, "rows": per_row}
    out = run_dir(cfg) / "e1_results.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n{'method':<16}{'n':>5}{'R@1':>8}{f'R@{k}':>8}{'MRR':>8}{'ms/q':>9}")
    for m, r in results.items():
        print(f"{m:<16}{r['n']:>5}{r['recall@1']:>8.3f}{r[f'recall@{k}']:>8.3f}{r['mrr']:>8.3f}{r['ms_per_query']:>9.1f}")
    print(f"wrote {out}  (misses are listed per row for error analysis)")
    return results


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--make-template", action="store_true")
    p.add_argument("--pairs", type=int, default=12)
    p.add_argument("--methods", default="tfidf,bm25,embedding,hybrid,bm25+ctx,embedding+ctx")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.make_template:
        make_template(cfg, build_llm(cfg, mock=a.mock), a.pairs)
    else:
        evaluate(cfg, a.methods.split(","), cfg["retrieval"]["top_k"])
