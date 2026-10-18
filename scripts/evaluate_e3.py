"""E3 step 2: score every tailored resume, cheapest valid method first (the Topic 6 ladder):

  1. programmatic checks  tools and numbers in the output that appear nowhere in the resume
  2. LLM judge            each atomic claim supported / unsupported by the resume
                          (= RAGAS-style faithfulness), and which requirements are covered;
                          the judge is cfg.eval_judge, a stronger model than the pipeline's,
                          validated against human labels (--human-labels)
  3. pairwise LLM judge   (--pairwise) which variant a recruiter would rather receive, judged in
                          BOTH orders; a verdict counts only if both orders agree (swap-and-tie
                          mitigation for position bias), validated against human picks
  4. humans               the labelling sheets this script writes

    python scripts/evaluate_e3.py
    python scripts/evaluate_e3.py --pairwise
    python scripts/evaluate_e3.py --human-labels data/labels/e3_human_sample.csv \
                                  --human-preference data/labels/e3_preference.csv

  -> runs/<run>/e3_judged.jsonl, e3_pairwise.jsonl   cached judge outputs (re-runs are cheap)
  -> runs/<run>/e3_per_resume.csv                    one row per pair x variant
  -> runs/<run>/e3_summary.csv, e3_tests.json        means by variant x fit level, paired tests
  -> data/labels/e3_human_sample.csv                 200 claims to label (supported yes/no)
  -> data/labels/e3_preference.csv                   30 blind comparisons to rate
"""
import csv
import json
import random
from collections import defaultdict
from itertools import combinations

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, run_dir, tqdm

from careercopilot import prompts
from careercopilot.generation import split_claims
from careercopilot.llm import parse_json
from careercopilot.metrics import Similarity, cohen_kappa, introduced_facts, wilcoxon_signed_rank
from careercopilot.schema import Usage, append_jsonl, load_results, read_jsonl
from careercopilot.text import load_vocab
from careercopilot.verifiers import LLMVerifier

FITS = ["good", "potential", "none"]


def coverage(judge_llm, requirements, claims):
    if not claims or not requirements:
        return 0.0, Usage()
    reqs = "\n".join(f"[{i + 1}] {r}" for i, r in enumerate(requirements))
    cl = "\n".join(f"- {c}" for c in claims)
    out, u = judge_llm.chat(prompts.COVERAGE, f"<requirements>\n{reqs}\n</requirements>\n\n<lines>\n{cl}\n</lines>",
                            max_tokens=100)
    try:
        hit = {int(x) for x in parse_json(out) if 1 <= int(x) <= len(requirements)}
    except (ValueError, TypeError):
        hit = set()
    return len(hit) / len(requirements), u


def judge_all(cfg, results, pairs, judge_llm, splitter_llm):
    path = run_dir(cfg) / "e3_judged.jsonl"
    cache = {(r["pair_id"], r["variant"]): r for r in read_jsonl(path)}
    judge = LLMVerifier(judge_llm, name="eval_judge")
    for res in tqdm(results, desc="claim judge"):
        key = (res.pair_id, res.variant)
        if key in cache or res.error:
            continue
        bullets = pairs[res.pair_id].bullets
        claims, _ = split_claims(splitter_llm, res.lines, cfg["tailoring"]["claim_split"])
        judged = []
        for c in claims:
            ok, _, _ = judge.verify(c.text, bullets)
            judged.append({"text": c.text, "line_idx": c.line_idx, "supported": ok})
        cov, _ = coverage(judge_llm, res.requirements, [c["text"] for c in judged if c["supported"]])
        cache[key] = {"pair_id": res.pair_id, "variant": res.variant, "fit": res.fit,
                      "claims": judged, "coverage": cov}
        append_jsonl(path, cache[key])
    return cache


def pairwise_once(judge_llm, pair, first, second):
    user = (f"<job>\n{pair.job}\n</job>\n\n<resume>\n{pair.resume}\n</resume>\n\n"
            f"<resume_1>\n{first}\n</resume_1>\n\n<resume_2>\n{second}\n</resume_2>")
    out, _ = judge_llm.chat(prompts.PAIRWISE, user, max_tokens=50)
    try:
        w = str(parse_json(out).get("winner", "")).strip().lower()
    except (ValueError, AttributeError):
        w = ""
    return w if w in ("1", "2", "tie") else "tie"


def pairwise_all(cfg, results, pairs, judge_llm):
    """Every pair of variants for every resume, judged in both orders (Topic 6 lab, Part 4)."""
    path = run_dir(cfg) / "e3_pairwise.jsonl"
    cache = {(r["pair_id"], r["a"], r["b"]): r for r in read_jsonl(path)}
    outs = {(r.pair_id, r.variant): r for r in results if not r.error}
    for pid in tqdm(sorted({k[0] for k in outs}), desc="pairwise judge"):
        vs = sorted(v for (p, v) in outs if p == pid)
        for a, b in combinations(vs, 2):
            if (pid, a, b) in cache:
                continue
            ta, tb = "\n".join(outs[(pid, a)].lines), "\n".join(outs[(pid, b)].lines)
            v1 = pairwise_once(judge_llm, pairs[pid], ta, tb)      # a shown first
            v2 = pairwise_once(judge_llm, pairs[pid], tb, ta)      # b shown first
            pick1 = {"1": a, "2": b}.get(v1, "tie")
            pick2 = {"1": b, "2": a}.get(v2, "tie")
            status = ("tie" if pick1 == pick2 == "tie" else "consistent" if pick1 == pick2
                      else "mixed" if "tie" in (pick1, pick2) else "flipped")
            cache[(pid, a, b)] = {"pair_id": pid, "fit": pairs[pid].fit, "a": a, "b": b,
                                  "a_first": pick1, "b_first": pick2, "status": status,
                                  "winner": pick1 if status == "consistent" else "tie"}
            append_jsonl(path, cache[(pid, a, b)])
    return list(cache.values())


def summarise_pairwise(rows):
    wins, games = defaultdict(int), defaultdict(int)
    for r in rows:
        for v in (r["a"], r["b"]):
            games[v] += 1
        if r["winner"] != "tie":
            wins[r["winner"]] += 1
    n = len(rows)
    flipped = sum(r["status"] == "flipped" for r in rows)
    first = [r["a_first"] == r["a"] for r in rows if r["a_first"] != "tie"] + \
            [r["b_first"] == r["b"] for r in rows if r["b_first"] != "tie"]
    print(f"\npairwise judge on {n} comparisons: consistent {sum(r['status'] == 'consistent' for r in rows)}, "
          f"tie both ways {sum(r['status'] == 'tie' for r in rows)}, FLIPPED with order {flipped}, "
          f"mixed {sum(r['status'] == 'mixed' for r in rows)}; first-shown wins "
          f"{sum(first) / len(first):.2f} of non-tie verdicts (0.5 = no position bias)" if first else "")
    print("win rate (consistent wins / comparisons):",
          ", ".join(f"{v} {wins[v] / games[v]:.2f}" for v in sorted(games)))
    return {v: wins[v] / games[v] for v in games}


def main(config, mock=False, human_labels=None, human_pref=None, pairwise=False, n_human=200, n_pref=30):
    cfg = load_config(config)
    pairs = {p.pair_id: p for p in load_pairs_or_exit(cfg)}
    results = load_results(run_dir(cfg) / "e3_outputs.jsonl")
    if not results:
        raise SystemExit("no outputs -- run scripts/run_e3_tailoring.py first")
    judge_llm = build_llm(cfg, which="eval_judge", mock=mock)
    splitter_llm = build_llm(cfg, mock=mock)
    vocab = load_vocab(resolve(cfg["paths"]["skills_vocab"]))
    terms = [t for ts in vocab.values() for t in ts]
    judged = judge_all(cfg, results, pairs, judge_llm, splitter_llm)
    sim = Similarity(cfg["retrieval"]["embedding_model"])

    rows = []
    for res in results:
        j = judged.get((res.pair_id, res.variant))
        if not j:
            continue
        n = len(j["claims"])
        unsup = sum(not c["supported"] for c in j["claims"])
        p = pairs[res.pair_id]
        text = "\n".join(res.lines)
        intro = introduced_facts(text, p.resume, terms)
        rows.append({"pair_id": res.pair_id, "variant": res.variant, "fit": res.fit,
                     "new_tools": len(intro["new_tools"]), "new_numbers": len(intro["new_numbers"]),
                     "any_introduced": int(bool(intro["new_tools"] or intro["new_numbers"])),
                     "claims": n, "unsupported": unsup, "unsupported_rate": unsup / n if n else 0.0,
                     "faithfulness": 1 - unsup / n if n else 1.0,
                     "coverage": j["coverage"],
                     "similarity_gain": sim(text, p.job) - sim(p.resume, p.job),
                     "lines": len(res.lines), "removed_lines": len(res.removed_lines),
                     "regenerations": res.regenerations, "llm_calls": res.usage.llm_calls,
                     "tokens": res.usage.tokens_in + res.usage.tokens_out,
                     "latency_s": res.usage.latency_s, "cost_usd": res.usage.cost_usd,
                     "introduced": "; ".join(intro["new_tools"] + intro["new_numbers"])})
    out = run_dir(cfg)
    with open(out / "e3_per_resume.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- summary by variant x fit ----
    metrics = ["any_introduced", "new_tools", "new_numbers", "unsupported_rate", "faithfulness", "coverage",
               "similarity_gain", "removed_lines", "llm_calls", "cost_usd", "latency_s"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["fit"])].append(r)
        groups[(r["variant"], "all")].append(r)
    summary = [{"variant": v, "fit": fit, "n": len(rs), **{m: sum(r[m] for r in rs) / len(rs) for m in metrics}}
               for (v, fit), rs in sorted(groups.items())]
    by = {(r["pair_id"], r["variant"]): r for r in rows}
    tests = {}
    for v in sorted({r["variant"] for r in rows} - {"A"}):
        for fit in FITS + ["all"]:
            ids = [pid for (pid, var) in by if var == "A" and (fit == "all" or by[(pid, var)]["fit"] == fit)
                   and (pid, v) in by]
            for m in ("unsupported_rate", "coverage", "any_introduced"):
                tests[f"{v}_vs_A/{fit}/{m}"] = wilcoxon_signed_rank([by[(i, v)][m] for i in ids],
                                                                    [by[(i, "A")][m] for i in ids])
    with open(out / "e3_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    json.dump(tests, open(out / "e3_tests.json", "w"), indent=2)

    print(f"\nsimilarity = {sim.kind}")
    print(f"{'var':<4}{'fit':<10}{'n':>3}{'introd%':>8}{'unsup%':>8}{'cover%':>8}{'simgain':>9}"
          f"{'removed':>8}{'calls':>7}{'$/resume':>10}")
    for s in summary:
        print(f"{s['variant']:<4}{s['fit']:<10}{s['n']:>3}{100 * s['any_introduced']:>8.0f}"
              f"{100 * s['unsupported_rate']:>8.1f}{100 * s['coverage']:>8.1f}{s['similarity_gain']:>9.3f}"
              f"{s['removed_lines']:>8.1f}{s['llm_calls']:>7.1f}{s['cost_usd']:>10.4f}")
    print("  introd% = resumes with a tool or number not in the original (programmatic check)")
    print("\npaired tests vs A, unsupported_rate (Wilcoxon p):")
    for k, t in tests.items():
        if k.endswith("unsupported_rate"):
            print(f"  {k.rsplit('/', 1)[0]:<22} n={t['n']:<3} p={t['p']:.3f}")

    pw = pairwise_all(cfg, results, pairs, judge_llm) if pairwise else list(read_jsonl(out / "e3_pairwise.jsonl"))
    if pw:
        summarise_pairwise(pw)

    # ---- sheets for the human parts ----
    labels_dir = resolve(cfg["paths"]["labels_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    sample_path = labels_dir / "e3_human_sample.csv"
    if not sample_path.exists():
        all_claims = [(k, i, c) for k, j in judged.items() for i, c in enumerate(j["claims"])]
        pick = rng.sample(all_claims, min(n_human, len(all_claims)))
        with open(sample_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "pair_id", "variant", "claim_idx", "claim", "supported_yes_no", "labeller"])
            for n, ((pid, v), i, c) in enumerate(pick):
                w.writerow([n, pid, v, i, c["text"], "", ""])
        print(f"\nwrote {sample_path}: label each claim against its resume (hide the variant column)")
    pref_path = labels_dir / "e3_preference.csv"
    if not pref_path.exists():
        outs = {(r.pair_id, r.variant): r for r in results if not r.error}
        combos = [(pid, a, b) for pid in sorted({k[0] for k in outs})
                  for a, b in combinations(sorted(v for (p, v) in outs if p == pid), 2)]
        with open(pref_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "pair_id", "left_variant", "right_variant", "left", "right",
                        "winner_left_right_tie", "rater"])
            for n, (pid, a, b) in enumerate(rng.sample(combos, min(n_pref, len(combos)))):
                l, r = (a, b) if rng.random() < 0.5 else (b, a)
                w.writerow([n, pid, l, r, "\n".join(outs[(pid, l)].lines), "\n".join(outs[(pid, r)].lines), "", ""])
        print(f"wrote {pref_path}: hide the *_variant columns, pick the version you'd rather receive")

    # ---- validate the judges against people ----
    if human_labels:
        hl = [r for r in csv.DictReader(open(human_labels, encoding="utf-8")) if r["supported_yes_no"].strip()]
        human = [r["supported_yes_no"].strip().lower().startswith("y") for r in hl]
        judge = [judged[(r["pair_id"], r["variant"])]["claims"][int(r["claim_idx"])]["supported"] for r in hl]
        agree = sum(h == j for h, j in zip(human, judge)) / len(hl)
        print(f"\nclaim judge vs human on {len(hl)} claims: agreement {agree:.2f}, "
              f"Cohen's kappa {cohen_kappa(human, judge):.2f}")
    if human_pref and pw:
        pwmap = {(r["pair_id"], r["a"], r["b"]): r["winner"] for r in pw}
        hp = [r for r in csv.DictReader(open(human_pref, encoding="utf-8")) if r["winner_left_right_tie"].strip()]
        h, j = [], []
        for r in hp:
            a, b = sorted([r["left_variant"], r["right_variant"]])
            choice = r["winner_left_right_tie"].strip().lower()
            h.append({"left": r["left_variant"], "right": r["right_variant"]}.get(choice, "tie"))
            j.append(pwmap.get((r["pair_id"], a, b), "tie"))
        print(f"pairwise judge vs human on {len(hp)} comparisons: agreement "
              f"{sum(x == y for x, y in zip(h, j)) / len(hp):.2f}, kappa {cohen_kappa(h, j):.2f}")
    return summary


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--pairwise", action="store_true", help="run the pairwise preference judge (both orders)")
    p.add_argument("--human-labels", default=None)
    p.add_argument("--human-preference", default=None)
    a = p.parse_args()
    main(a.config, a.mock, a.human_labels, a.human_preference, a.pairwise)
