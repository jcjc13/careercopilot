"""E3 qualitative failure analysis (read-only: uses the saved outputs and judge results, no LLM calls).

    python scripts/analyse_e3_failures.py
  -> runs/<run>/e3_failures.csv      every claim the claim judge marked unsupported, with a category
     runs/<run>/e3_failure_summary.json   counts by variant x category, loop/removal stats, judge-error cases

Categories for an unsupported claim (first match wins; heuristics, so counts are indicative and the examples are
what to quote):
  fragment          a piece of a sentence from the claim splitter ("for reporting", "to managers") -- not a real claim
  invented_number   a number that appears nowhere in the resume
  invented_tool     a tool/technology from the skills vocabulary that appears nowhere in the resume
  self_praise       generic self-description ("proven track record", "strong knowledge of", "seasoned")
  inflated_scope    ownership/seniority verbs (managed, led, oversaw, ...) not backed by the resume
  other_duty        a duty or experience the resume does not state
Also written: the share of claims that are splitter fragments (and the unsupported rate without them), lines the
verifier removed, and the cases where the claim judge disagrees with the human labels.
"""
import csv
import json
import re
from collections import Counter, defaultdict

from _common import load_config, load_pairs_or_exit, parser, resolve, run_dir

from careercopilot.metrics import introduced_facts
from careercopilot.schema import read_jsonl
from careercopilot.text import load_vocab

CATS = ["fragment", "invented_number", "invented_tool", "self_praise", "inflated_scope", "other_duty"]
FRAGMENT = re.compile(r"^(to|for|that|between|in|on|with|over|more than|by|of|and|at|from)\b", re.I)
PRAISE = re.compile(r"\b(proven|track record|strong (?:knowledge|skills|ability)|seasoned|demonstrat\w+|skills|expert\w*|adept|"
                    r"detail[- ]oriented|motivated|highly|passionate|results[- ]driven|ability to|time[- ]management)\b", re.I)
SCOPE = re.compile(r"^(?:\W*)(managed|led|lead|oversaw|directed|spearheaded|headed|supervised|owned|ensured|"
                   r"implemented|designed|trained|facilitated|presented)\b", re.I)


def categorise(text, resume, terms):
    words = text.split()
    intro = introduced_facts(text, resume, terms)
    if len(words) < 4 or FRAGMENT.match(text.strip()) or text.strip()[:1].islower():
        return "fragment"
    if intro["new_numbers"]:
        return "invented_number"
    if intro["new_tools"]:
        return "invented_tool"
    if PRAISE.search(text):
        return "self_praise"
    if SCOPE.match(text):
        return "inflated_scope"
    return "other_duty"


def main(config):
    cfg = load_config(config)
    out = run_dir(cfg)
    pairs = {p.pair_id: p for p in load_pairs_or_exit(cfg)}
    vocab = load_vocab(resolve(cfg["paths"]["skills_vocab"]))
    terms = [t for ts in vocab.values() for t in ts]
    judged = {(d["pair_id"], d["variant"]): d for d in read_jsonl(out / "e3_judged.jsonl")}
    outputs = {(d["pair_id"], d["variant"]): d for d in read_jsonl(out / "e3_outputs.jsonl")}

    rows, counts = [], defaultdict(Counter)
    for (pid, v), j in judged.items():
        resume = pairs[pid].resume
        for i, c in enumerate(j["claims"]):
            if c["supported"]:
                continue
            cat = categorise(c["text"], resume, terms)
            counts[v][cat] += 1
            rows.append({"pair_id": pid, "fit": j["fit"], "variant": v, "claim_idx": i, "category": cat, "claim": c["text"]})
    with open(out / "e3_failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "fit", "variant", "claim_idx", "category", "claim"])
        w.writeheader()
        w.writerows(rows)

    # how much of the claim count is splitter fragments, and the unsupported rate without them
    frag = {}
    for v in "ABCD":
        cl = [(c, categorise(c["text"], "", terms) == "fragment") for (p, vv), j in judged.items() if vv == v for c in j["claims"]]
        real = [c for c, f in cl if not f]
        frag[v] = {"claims": len(cl), "fragments": sum(f for _, f in cl),
                   "unsupported_rate_all": round(sum(not c["supported"] for c, _ in cl) / len(cl), 3),
                   "unsupported_rate_without_fragments": round(sum(not c["supported"] for c in real) / len(real), 3)}
    # removals by the verifier loop
    removed = {v: Counter() for v in "CD"}
    for (pid, v), o in outputs.items():
        if v in removed:
            for line in o["removed_lines"]:
                removed[v]["summary" if line.startswith("Summary: ") else "bullet"] += 1
    # judge errors against the human labels
    errs = {"judge_missed_unsupported": [], "judge_false_alarm": []}
    sample = labels_dir = resolve(cfg["paths"]["labels_dir"])
    hs, hk = sample / "e3_human_sample.csv", sample / "e3_human_key.csv"
    if hs.exists() and hk.exists():
        key = {r["item"]: r for r in csv.DictReader(open(hk, encoding="utf-8"))}
        for r in csv.DictReader(open(hs, encoding="utf-8")):
            if not r["supported_yes_no"].strip():
                continue
            k = key[r["item"]]
            j = judged[(k["pair_id"], k["variant"])]["claims"][int(k["claim_idx"])]["supported"]
            h = r["supported_yes_no"].strip().lower().startswith("y")
            if j and not h:
                errs["judge_missed_unsupported"].append({"variant": k["variant"], "claim": r["claim"], "line": r["output_line"][:200]})
            if h and not j:
                errs["judge_false_alarm"].append({"variant": k["variant"], "claim": r["claim"], "line": r["output_line"][:200]})
    summary = {"unsupported_by_variant_and_category": {v: dict(counts[v]) for v in sorted(counts)},
               "unsupported_total": {v: sum(counts[v].values()) for v in sorted(counts)},
               "fragments": frag,
               "verifier_removed_lines": {v: dict(c) for v, c in removed.items()},
               "judge_errors_vs_human": {k: {"n": len(x), "examples": x[:8]} for k, x in errs.items()}}
    json.dump(summary, open(out / "e3_failure_summary.json", "w"), indent=2, ensure_ascii=False)
    print(f"{'variant':<8}" + "".join(f"{c:>17}" for c in CATS) + f"{'total':>8}")
    for v in sorted(counts):
        print(f"{v:<8}" + "".join(f"{counts[v][c]:>17}" for c in CATS) + f"{sum(counts[v].values()):>8}")
    print("fragments:", frag)
    print("removed lines:", summary["verifier_removed_lines"])
    print("judge errors vs human:", {k: v["n"] for k, v in summary["judge_errors_vs_human"].items()})
    print(f"wrote {out / 'e3_failures.csv'} and e3_failure_summary.json")


if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    main(a.config)
