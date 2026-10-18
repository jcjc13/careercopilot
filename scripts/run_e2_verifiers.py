"""E2: compare claim verifiers on the human-reviewed planted-fabrication set: a programmatic
rule check, a zero-shot NLI cross-encoder, the fine-tuned BERT verifier (train_verifier.py) and
an LLM judge -- the Topic 6 ladder from cheapest to dearest.

    python scripts/run_e2_verifiers.py --methods rule,nli,finetuned,llm
    python scripts/run_e2_verifiers.py --methods rule --include-unreviewed   # early dry run

Each claim is checked against ALL bullets of the resume it came from. 20% of claims (by
source bullet, so a paraphrase and its fabricated twin stay together) are a dev split used
only to tune the NLI threshold; metrics are reported on the other 80%.
  -> runs/<run>/e2_results.json, runs/<run>/e2_predictions.csv (every prediction, for error analysis)
"""
import csv
import json
import random

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, run_dir, tqdm

from careercopilot.metrics import prf
from careercopilot.schema import Usage
from careercopilot.verifiers import build_verifier


def best_threshold(scores, fabricated):
    best = (0.5, -1.0)
    for t in [i / 20 for i in range(1, 20)]:
        f1 = prf(fabricated, [s < t for s in scores])["f1"]
        if f1 > best[1]:
            best = (t, f1)
    return best[0]


def main(config, methods, include_unreviewed=False, mock=False, limit=None):
    cfg = load_config(config)
    pairs = {p.pair_id: p for p in load_pairs_or_exit(cfg)}
    path = resolve(cfg["paths"]["labels_dir"]) / "e2_claims.csv"
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if include_unreviewed or r["reviewed"].strip().lower() == "yes"]
    rows = rows[:limit] if limit else rows
    if not rows:
        raise SystemExit("no reviewed claims -- review e2_claims.csv or pass --include-unreviewed")
    sources = sorted({r["source_bullet"] for r in rows})
    random.Random(cfg["seed"]).shuffle(sources)
    dev_src = set(sources[: max(1, len(sources) // 5)])
    dev = [r for r in rows if r["source_bullet"] in dev_src]
    test = [r for r in rows if r["source_bullet"] not in dev_src]
    print(f"{len(rows)} claims: {len(dev)} dev / {len(test)} test")

    llm = build_llm(cfg, mock=mock)
    results, preds_out = {}, []
    for m in methods:
        try:
            v = build_verifier(cfg, llm, m)
        except Exception as e:
            print(f"skip {m}: {e!r}")
            continue
        if m in ("nli", "finetuned"):  # score-based verifiers: tune the threshold on dev only
            s = [v.score(r["claim"], pairs[r["pair_id"]].bullets) for r in tqdm(dev, desc=f"{m} dev")]
            v.threshold = best_threshold(s, [r["label"] == "fabricated" for r in dev])
            print(f"{m}: best dev threshold = {v.threshold}  (put it in configs/default.yaml)")
        usage, y_true, y_pred, types = Usage(), [], [], []
        for r in tqdm(test, desc=m):
            ok, score, u = v.verify(r["claim"], pairs[r["pair_id"]].bullets)
            usage.add(u)
            y_true.append(r["label"] == "fabricated")
            y_pred.append(not ok)
            types.append(r["type"])
            preds_out.append({"method": m, "claim_id": r["claim_id"], "label": r["label"], "type": r["type"],
                              "flagged": not ok, "score": round(score, 4), "claim": r["claim"],
                              "source_bullet": r["source_bullet"]})
        res = prf(y_true, y_pred)
        faithful = [p for t, p in zip(y_true, y_pred) if not t]
        res["false_flag_rate"] = sum(faithful) / len(faithful) if faithful else 0.0
        res["recall_by_type"] = {
            k: sum(p for t, p, ty in zip(y_true, y_pred, types) if t and ty == k) /
               max(1, sum(1 for t, ty in zip(y_true, types) if t and ty == k))
            for k in sorted({ty for t, ty in zip(y_true, types) if t})}
        n = len(test)
        res.update({"n": n, "ms_per_claim": 1000 * usage.latency_s / n, "usd_per_claim": usage.cost_usd / n,
                    "threshold": getattr(v, "threshold", None)})
        results[m] = res

    out = run_dir(cfg)
    json.dump(results, open(out / "e2_results.json", "w"), indent=2)
    if preds_out:
        with open(out / "e2_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(preds_out[0]))
            w.writeheader()
            w.writerows(preds_out)
    print(f"\n{'method':<8}{'P':>7}{'R':>7}{'F1':>7}{'falseflag':>11}{'ms/claim':>10}{'$/claim':>10}  recall by type")
    for m, r in results.items():
        by = ", ".join(f"{k} {v:.2f}" for k, v in r["recall_by_type"].items())
        print(f"{m:<8}{r['precision']:>7.2f}{r['recall']:>7.2f}{r['f1']:>7.2f}{r['false_flag_rate']:>11.2f}"
              f"{r['ms_per_claim']:>10.1f}{r['usd_per_claim']:>10.5f}  {by}")
    print(f"wrote {out / 'e2_results.json'} and e2_predictions.csv")
    return results


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--methods", default="rule,nli,finetuned,llm")
    p.add_argument("--include-unreviewed", action="store_true")
    a = p.parse_args()
    main(a.config, a.methods.split(","), a.include_unreviewed, a.mock, a.limit)
