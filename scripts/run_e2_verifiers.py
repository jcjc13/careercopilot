"""E2: compare claim verifiers on the human-reviewed planted-fabrication sets: a programmatic
rule check, a zero-shot NLI cross-encoder, the fine-tuned BERT and NLI verifiers (train_verifier.py)
and an LLM judge, from cheapest to dearest.

    python scripts/run_e2_verifiers.py --methods rule,nli,finetuned,finetuned_nli,llm
    python scripts/run_e2_verifiers.py --methods rule --include-unreviewed   # early dry run
    python scripts/run_e2_verifiers.py --methods nli --merge --tune dev      # rerun one method, tune on a dev split

The claims file has two sets (column `set`): standard (the original test set) and hard (subtler
fabrications and harder rewrites drafted by a different model family, make_hard_claims.py).
Each claim is checked against ALL bullets of the resume it came from. Every metric is reported per set.

Thresholds of the score-based verifiers (nli, finetuned, finetuned_nli) are tuned on the silver
validation set (data/verifier/val.jsonl, one bullet as the evidence) and every claim of each set is then
evaluated. `--tune dev` keeps the old behaviour: a 20% dev split of the standard set (by source bullet)
tunes the threshold, and the standard set is scored on the rest. The threshold is the middle of the
range of grid values that tie for the best dev F1 (the first maximum is arbitrary on a plateau).
Reported per set and verifier: P/R/F1 at the tuned threshold, F1 at 0.5, ROC-AUC (score-based only),
false-flag rate (overall and per faithful type), recall per fabrication type, scripted vs LLM-drafted
rows, and McNemar's exact test between verifiers.
  -> runs/<run>/e2_results.json  {set: {method: metrics}, "mcnemar": {set: [...]}}
     runs/<run>/e2_predictions.csv (every prediction, for error analysis)
"""
import csv
import json
import random
from itertools import combinations

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, run_dir, tqdm

from careercopilot.metrics import mcnemar_exact, prf, roc_auc
from careercopilot.schema import Usage
from careercopilot.verifiers import build_verifier

SCORE_BASED = ("nli", "finetuned", "finetuned_nli")
GRID = [i / 20 for i in range(1, 20)]


def best_threshold(scores, fabricated):
    """Middle of the longest run of grid thresholds that tie for the best F1 on the tuning data."""
    f1s = [prf(fabricated, [s < t for s in scores])["f1"] for t in GRID]
    top = max(f1s)
    runs, cur = [], []
    for t, f in zip(GRID, f1s):
        if f >= top - 1e-9:
            cur.append(t)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    best = max(runs, key=len)
    return best[len(best) // 2]


def read_claims(cfg, include_unreviewed):
    path = resolve(cfg["paths"]["labels_dir"]) / "e2_claims.csv"
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        r["set"] = r.get("set") or "standard"
        r["drafted"] = r["method"]                     # scripted | llm_draft
        if include_unreviewed or r["reviewed"].strip().lower() == "yes":
            rows.append(r)
    return rows


def silver_val(cfg, limit=None):
    path = resolve(cfg["paths"]["verifier_data"]) / "val.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found -- build the silver set (make_verifier_train.py) or use --tune dev")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def summarise(rows, y_true, y_pred, scores, usage, threshold, f1_at_half, score_based):
    res = prf(y_true, y_pred)
    faithful = [p for t, p in zip(y_true, y_pred) if not t]
    res["false_flag_rate"] = sum(faithful) / len(faithful) if faithful else 0.0
    res["f1_at_0.5"] = f1_at_half
    res["auc"] = roc_auc(y_true, [1 - s for s in scores]) if score_based else None
    types = [r["type"] for r in rows]
    res["recall_by_type"] = {
        k: sum(p for t, p, ty in zip(y_true, y_pred, types) if t and ty == k) /
           max(1, sum(1 for t, ty in zip(y_true, types) if t and ty == k))
        for k in sorted({ty for t, ty in zip(y_true, types) if t})}
    res["false_flag_by_type"] = {
        k: sum(p for t, p, ty in zip(y_true, y_pred, types) if not t and ty == k) /
           max(1, sum(1 for t, ty in zip(y_true, types) if not t and ty == k))
        for k in sorted({ty for t, ty in zip(y_true, types) if not t})}
    res["by_source"] = {}
    for src in sorted({r["drafted"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["drafted"] == src]
        yt, yp = [y_true[i] for i in idx], [y_pred[i] for i in idx]
        pos, neg = [p for t, p in zip(yt, yp) if t], [p for t, p in zip(yt, yp) if not t]
        res["by_source"][src] = {"n": len(idx), "n_fabricated": len(pos), "n_faithful": len(neg),
                                 "recall": sum(pos) / len(pos) if pos else None,
                                 "false_flag_rate": sum(neg) / len(neg) if neg else None,
                                 "f1": prf(yt, yp)["f1"]}
    n = len(rows)
    res.update({"n": n, "n_fabricated": sum(y_true), "ms_per_claim": 1000 * usage.latency_s / n,
                "usd_per_claim": usage.cost_usd / n, "threshold": threshold})
    return res


def mcnemar_table(preds):
    """Pairwise McNemar per set from the stored predictions (so it also covers --merge runs)."""
    out = {}
    for s in sorted({p["set"] for p in preds}):
        by = {}
        for p in preds:
            if p["set"] == s:
                by.setdefault(p["method"], {})[p["claim_id"]] = (str(p["flagged"]) == "True") == (p["label"] == "fabricated")
        out[s] = []
        for a, b in combinations(by, 2):
            ids = sorted(set(by[a]) & set(by[b]))
            r = mcnemar_exact([by[a][i] for i in ids], [by[b][i] for i in ids])
            out[s].append({"a": a, "b": b, "n": len(ids), **r})
    return out


def save(cfg, results, preds_out, merge):
    """Write results + predictions; called after every method so a stopped run keeps what it finished."""
    out = run_dir(cfg)
    old_json, old_csv = out / "e2_results.json", out / "e2_predictions.csv"
    if merge:  # keep earlier methods from this run (same sets), replace only the ones just rerun
        if old_json.exists():
            old = json.load(open(old_json))
            for s in set(old) - {"mcnemar"}:
                if isinstance(old[s], dict) and all(isinstance(x, dict) for x in old[s].values()):
                    results.setdefault(s, {})
                    results[s] = {**{k: x for k, x in old[s].items() if k not in results[s]}, **results[s]}
        if old_csv.exists():
            rerun = {p["method"] for p in preds_out}
            with open(old_csv, encoding="utf-8") as f:
                rd = csv.DictReader(f)
                if "set" in (rd.fieldnames or []):   # older runs have no set column; skip them
                    preds_out = [r for r in rd if r["method"] not in rerun] + preds_out
    results["mcnemar"] = mcnemar_table(preds_out)
    json.dump(results, open(old_json, "w"), indent=2)
    if preds_out:
        with open(old_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(preds_out[0]))
            w.writeheader()
            w.writerows(preds_out)
    return results


def main(config, methods, include_unreviewed=False, mock=False, limit=None, merge=False, tune="silver"):
    cfg = load_config(config)
    pairs = {p.pair_id: p for p in load_pairs_or_exit(cfg)}
    rows = read_claims(cfg, include_unreviewed)
    rows = rows[:limit] if limit else rows
    if not rows:
        raise SystemExit("no reviewed claims -- review e2_claims.csv or pass --include-unreviewed")
    sets = {s: [r for r in rows if r["set"] == s] for s in sorted({r["set"] for r in rows})}

    dev = []
    if tune == "dev":   # the old protocol: 20% of the standard set (by source bullet) tunes, the rest tests
        std = sets.get("standard", rows)
        sources = sorted({r["source_bullet"] for r in std})
        random.Random(cfg["seed"]).shuffle(sources)
        dev_src = set(sources[: max(1, len(sources) // 5)])
        dev = [r for r in std if r["source_bullet"] in dev_src]
        sets["standard"] = [r for r in std if r["source_bullet"] not in dev_src]
    print("claims per set: " + ", ".join(f"{s} {len(v)}" for s, v in sets.items()) +
          (f"  (dev {len(dev)} used for tuning)" if dev else ""))
    val = silver_val(cfg, limit) if tune == "silver" and any(m in SCORE_BASED for m in methods) else []

    llm = build_llm(cfg, mock=mock)
    results = {s: {} for s in sets}
    preds_out = []
    for m in methods:
        try:
            v = build_verifier(cfg, llm, m)
        except Exception as e:
            print(f"skip {m}: {e!r}")
            continue
        score_based = m in SCORE_BASED
        if score_based:   # tune the threshold on held-out data that is not any test claim
            if tune == "dev":
                s = [v.score(r["claim"], pairs[r["pair_id"]].bullets) for r in tqdm(dev, desc=f"{m} dev")]
                fab = [r["label"] == "fabricated" for r in dev]
            else:
                s = [v.score(r["hypothesis"], [r["premise"]]) for r in tqdm(val, desc=f"{m} silver val")]
                fab = [r["label"] == 0 for r in val]
            v.threshold = best_threshold(s, fab)
            print(f"{m}: threshold = {v.threshold}  (tuned on {tune}; put it in configs/default.yaml)")
        for sname, srows in sets.items():
            if not srows:
                continue
            usage, y_true, y_pred, scores = Usage(), [], [], []
            for r in tqdm(srows, desc=f"{m} {sname}"):
                ok, score, u = v.verify(r["claim"], pairs[r["pair_id"]].bullets)
                usage.add(u)
                y_true.append(r["label"] == "fabricated")
                y_pred.append(not ok)
                scores.append(score)
                preds_out.append({"set": sname, "method": m, "claim_id": r["claim_id"], "label": r["label"],
                                  "type": r["type"], "drafted": r["drafted"], "flagged": not ok,
                                  "score": round(score, 4), "claim": r["claim"], "source_bullet": r["source_bullet"]})
            at_half = prf(y_true, [s < 0.5 for s in scores])["f1"] if score_based else prf(y_true, y_pred)["f1"]
            results[sname][m] = summarise(srows, y_true, y_pred, scores, usage, getattr(v, "threshold", None),
                                          at_half, score_based)
        save(cfg, results, preds_out, merge)   # checkpoint after each method

    results = save(cfg, results, preds_out, merge)
    show(results)
    print(f"wrote {run_dir(cfg) / 'e2_results.json'} and e2_predictions.csv")
    return results


def fmt(x, w=7):
    return f"{'n/a':>{w}}" if x is None or x != x else f"{x:>{w}.2f}"


def show(results):
    for s, ms in results.items():
        if s == "mcnemar" or not ms:
            continue
        n = next(iter(ms.values()))
        print(f"\n=== set: {s}  ({n['n']} claims, {n['n_fabricated']} fabricated) ===")
        print(f"{'method':<14}{'P':>7}{'R':>7}{'F1':>7}{'F1@.5':>7}{'AUC':>7}{'falseflag':>11}{'thr':>6}{'ms/claim':>10}{'$/claim':>10}")
        for m, r in ms.items():
            print(f"{m:<14}{r['precision']:>7.2f}{r['recall']:>7.2f}{r['f1']:>7.2f}{fmt(r['f1_at_0.5'])}{fmt(r['auc'])}"
                  f"{r['false_flag_rate']:>11.2f}{fmt(r['threshold'], 6)}{r['ms_per_claim']:>10.1f}{r['usd_per_claim']:>10.5f}")
        for m, r in ms.items():
            print(f"  {m}: recall by type  " + ", ".join(f"{k} {v:.2f}" for k, v in r["recall_by_type"].items()))
            print(f"  {m}: false flags    " + ", ".join(f"{k} {v:.2f}" for k, v in r["false_flag_by_type"].items()))
            print(f"  {m}: by source      " + "; ".join(
                f"{k} (n={d['n']}) recall {fmt(d['recall'], 4).strip()} ff {fmt(d['false_flag_rate'], 4).strip()}"
                for k, d in r["by_source"].items()))
        mc = results.get("mcnemar", {}).get(s, [])
        if mc:
            print("  McNemar exact (p; only_a / only_b = claims only one verifier gets right):")
            for t in mc:
                print(f"    {t['a']:<14} vs {t['b']:<14} p={t['p']:.3f}  {t['only_a']}/{t['only_b']}  (n={t['n']})")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--methods", default="rule,nli,finetuned,finetuned_nli,llm")
    p.add_argument("--include-unreviewed", action="store_true")
    p.add_argument("--tune", choices=["silver", "dev"], default="silver",
                   help="where score-based thresholds are tuned: the silver val set (default) or a dev split")
    p.add_argument("--merge", action="store_true",
                   help="keep the other methods already in this run's results and replace only --methods")
    a = p.parse_args()
    main(a.config, a.methods.split(","), a.include_unreviewed, a.mock, a.limit, a.merge, a.tune)
