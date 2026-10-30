"""Agent-style metrics per task for E3 (read-only: needs no change to the pipeline). A task = one resume x
one variant. Variants C and D are the agentic ones: generate -> verify -> regenerate loop with a step budget.

    python scripts/agent_metrics.py          # after run_e3_tailoring.py and evaluate_e3.py
  -> runs/<run>/agent_metrics_per_task.csv, runs/<run>/agent_metrics_summary.json

Per task:
  completed        no pipeline error and a non-empty tailored resume
  success          completed AND no unsupported claim AND no introduced tool/number (needs e3_per_resume.csv)
  format_validity  share of the structured outputs that were well formed (the pipeline does not log individual
                   tool calls, so this is the closest derivable "tool-call accuracy"): the requirements list has
                   the expected length; the output has a summary and bullets (a summary the verifier removed is not a format error; see summary_removed); for grounded variants every line
                   cites evidence ids and every cited id exists among the evidence shown (E1..En)
  steps            LLM calls (+ verifier calls when the verifier is the LLM judge)
  step_efficiency  minimal steps for the variant / steps used (1.0 = no wasted calls)
  regenerations    regenerate_line calls; loop_resolved = regenerations > 0 and no line had to be removed
  tokens, latency_s, cost_usd per task; summary adds cost per successful task
"""
import csv
import json
import statistics

from _common import load_config, run_dir, parser

from careercopilot.schema import read_jsonl


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def summary_removed(r):
    return any(x.startswith("Summary: ") for x in r["removed_lines"])


def format_checks(r, n_req):
    """List of booleans, one per structured output the task produced."""
    checks = [len(r["requirements"]) == n_req] if r["variant"] != "A" else []
    lines = r["lines"]
    # a summary the verifier removed is a verification outcome (counted in summary_removed), not a format error
    has_summary = bool(lines) and (lines[0].startswith("Summary: ") or summary_removed(r))
    checks.append(has_summary and len(lines) > 1)
    if r["variant"] in ("B", "D"):   # grounded: every bullet must cite existing evidence ids
        n_ev = len({b for bs in (r.get("evidence") or {}).values() for b in bs})
        valid = {f"E{i}" for i in range(1, n_ev + 1)}
        for cites in r["citations"][1:]:
            checks.append(bool(cites) and all(c in valid for c in cites))
    return checks


def main(config, mock=False, limit=None):
    cfg = load_config(config)
    out = run_dir(cfg)
    results = list(read_jsonl(out / "e3_outputs.jsonl"))
    if not results:
        raise SystemExit("no outputs -- run scripts/run_e3_tailoring.py first")
    judged = {}
    f = out / "e3_per_resume.csv"
    if f.exists():
        judged = {(r["pair_id"], r["variant"]): r for r in csv.DictReader(open(f, encoding="utf-8"))}
    n_req = cfg["tailoring"]["n_requirements"]
    llm_verifier = cfg["verifier"]["method"] == "llm"
    rows = []
    for r in results[:limit]:
        v, u = r["variant"], r["usage"]
        j = judged.get((r["pair_id"], v))
        completed = not r.get("error") and len(r["lines"]) > 1
        success = None if j is None else bool(completed and float(j["unsupported"]) == 0 and not int(float(j["any_introduced"])))
        checks = format_checks(r, n_req)
        agentic = v in ("C", "D")
        # minimal calls: A tailor(1); B requirements + tailor(2); C/D add one claim-split call (+ one verify call per claim if LLM)
        minimal = {"A": 1, "B": 2, "C": 2, "D": 3}[v] if not agentic or not llm_verifier else None
        steps = u["llm_calls"]
        rows.append({
            "pair_id": r["pair_id"], "variant": v, "fit": r["fit"], "agentic": int(agentic),
            "completed": int(completed), "success": "" if success is None else int(success),
            "format_validity": round(sum(checks) / len(checks), 3) if checks else "",
            "steps": steps, "step_efficiency": round(min(1.0, minimal / steps), 3) if minimal and steps else "",
            "regenerations": r["regenerations"], "removed_lines": len(r["removed_lines"]),
            "summary_removed": int(summary_removed(r)),
            "loop_resolved": int(r["regenerations"] > 0 and not r["removed_lines"]) if agentic else "",
            "tokens": u["tokens_in"] + u["tokens_out"], "latency_s": round(u["latency_s"], 3),
            "cost_usd": round(u["cost_usd"], 6)})
    with open(out / "agent_metrics_per_task.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def num(rs, k):
        return [float(x[k]) for x in rs if x[k] != ""]
    summary = {}
    for v in sorted({x["variant"] for x in rows}):
        rs = [x for x in rows if x["variant"] == v]
        ok = num(rs, "success")
        cost = sum(num(rs, "cost_usd"))
        regen = [x for x in rs if x["regenerations"] > 0]
        summary[v] = {
            "tasks": len(rs), "completed": statistics.mean(num(rs, "completed")),
            "success": statistics.mean(ok) if ok else None,
            "format_validity": statistics.mean(num(rs, "format_validity")) if num(rs, "format_validity") else None,
            "summary_removed_rate": statistics.mean(num(rs, "summary_removed")),
            "steps_mean": statistics.mean(num(rs, "steps")),
            "step_efficiency": statistics.mean(num(rs, "step_efficiency")) if num(rs, "step_efficiency") else None,
            "regenerations_mean": statistics.mean(num(rs, "regenerations")),
            "tasks_needing_regen": len(regen),
            "loop_resolved_rate": (statistics.mean(num(regen, "loop_resolved")) if regen and v in ("C", "D") else None),
            "tokens_mean": statistics.mean(num(rs, "tokens")),
            "latency_mean_s": statistics.mean(num(rs, "latency_s")), "latency_p95_s": pct(num(rs, "latency_s"), 0.95),
            "cost_mean_usd": cost / len(rs), "cost_per_success_usd": cost / sum(ok) if ok and sum(ok) else None}
    json.dump(summary, open(out / "agent_metrics_summary.json", "w"), indent=2)

    def f2(x, w=8):
        return f"{'n/a':>{w}}" if x is None else f"{x:>{w}.2f}"
    print(f"{'var':<4}{'tasks':>6}{'done':>7}{'succ':>7}{'fmt':>7}{'steps':>7}{'eff':>7}{'regen':>7}{'resolved':>10}"
          f"{'lat s':>8}{'p95 s':>8}{'$/task':>9}{'$/success':>11}")
    for v, s in summary.items():
        print(f"{v:<4}{s['tasks']:>6}{f2(s['completed'], 7)}{f2(s['success'], 7)}{f2(s['format_validity'], 7)}"
              f"{f2(s['steps_mean'], 7)}{f2(s['step_efficiency'], 7)}{f2(s['regenerations_mean'], 7)}"
              f"{f2(s['loop_resolved_rate'], 10)}{f2(s['latency_mean_s'])}{f2(s['latency_p95_s'])}"
              f"{s['cost_mean_usd']:>9.4f}{f2(s['cost_per_success_usd'], 11)}")
    print(f"wrote {out / 'agent_metrics_per_task.csv'} and agent_metrics_summary.json")
    return summary


if __name__ == "__main__":
    p = parser(__doc__)
    a = p.parse_args()
    main(a.config, a.mock, a.limit)
