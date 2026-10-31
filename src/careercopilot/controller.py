"""Bounded controller for variant D's recovery loop.

Variant C keeps the fixed loop (regenerate twice, then drop). Variant D replaces it with a controller: when a
generated line fails verification, the controller looks at the line, the evidence it cites, the verifier's scores
and the attempt history, diagnoses the failure and picks ONE permitted action. Python executes the action,
enforces the limits, re-verifies, and records every step in a trace.

    ACCEPT, REWRITE_CONSERVATIVELY, RETRIEVE_MORE, USE_ORIGINAL, DROP, HUMAN_REVIEW

Rules that Python (not the model) enforces:
  * the controller may only choose an action listed in `allowed_actions` for this state; anything else is replaced by
    the rule-based router's choice and the override is logged;
  * a line is accepted only when the verifier supports every claim in it (the controller cannot override that);
  * HUMAN_REVIEW is allowed only when a failing claim scored within `review_band` of the verifier's threshold (genuine
    ambiguity); a line with no support is dropped, not escalated;
  * hard limits per line: total actions, retrievals, rewrites (cfg tailoring.controller);
  * a line still failing when the limits run out is dropped (`loop_limit`); HUMAN_REVIEW lines are not output and are
    counted as escalations;
  * the controller never writes resume text: rewrites come from the generator, USE_ORIGINAL copies a cited bullet.
The same loop can run with the deterministic `rule_router` instead of the LLM (cfg tailoring.controller.mode: rule).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import prompts
from .llm import parse_json
from .config import resolve
from .schema import Claim, Usage
from .verifiers import RuleVerifier

ACTIONS = ("ACCEPT", "REWRITE_CONSERVATIVELY", "RETRIEVE_MORE", "USE_ORIGINAL", "DROP", "HUMAN_REVIEW")
FAILURE_TYPES = ("unsupported_embellishment", "retrieval_miss", "no_evidence", "unnecessary_rewriting",
                 "ambiguous_entailment", "conflicting_evidence", "formatting_only", "other")
DEFAULTS = {"mode": "llm", "max_total": 4, "max_retrieve": 2, "max_rewrite": 2, "retrieve_k": 5, "review_band": 0.1,
            "original_max_words": 45}


@dataclass
class Decision:
    action: str
    failure_type: str = "other"
    reason: str = ""
    instructions: str | None = None
    recovery_query: str | None = None
    confidence: float | None = None
    source: str = "llm"          # llm | rule | override


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **cfg["tailoring"].get("controller", {})}


def parse_decision(text: str) -> Decision:
    """Strict: a JSON object whose action is one of ACTIONS. Raises ValueError otherwise."""
    d = parse_json(text)
    if not isinstance(d, dict):
        raise ValueError("controller reply is not a JSON object")
    action = str(d.get("action", "")).strip().upper()
    if action not in ACTIONS:
        raise ValueError(f"unknown action {d.get('action')!r}")
    ft = str(d.get("failure_type", "other")).strip().lower()
    conf = d.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    nz = lambda v: (str(v).strip() or None) if v not in (None, "null", "None") else None  # noqa: E731
    return Decision(action, ft if ft in FAILURE_TYPES else "other", str(d.get("reason", "")).strip(),
                    nz(d.get("instructions")), nz(d.get("recovery_query")), conf)


@dataclass
class LineState:
    """Everything the controller may see about one failing line (kept small and structured, so traces are
    reproducible)."""
    line_idx: int
    line: str
    is_summary: bool
    requirements: list[str]
    evidence: list[dict]                       # [{"id": "E3", "text": ...}] currently cited
    unsupported: list[dict]                    # [{"claim": ..., "score": ...}]
    rule_missing: list[str]                    # tools / numbers / seniority words the evidence does not contain
    threshold: float | None
    original_max_words: int = 45
    review_band: float = 0.1
    attempts: dict = field(default_factory=lambda: {"retrieval": 0, "rewrite": 0, "total": 0})
    limits: dict = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def allowed_actions(self) -> list[str]:
        a = self.attempts
        out = []
        if a["total"] < self.limits["total"]:
            if a["rewrite"] < self.limits["rewrite"]:
                out.append("REWRITE_CONSERVATIVELY")
            if a["retrieval"] < self.limits["retrieval"] and not self.is_summary:
                out.append("RETRIEVE_MORE")
            if self.originals() and not self.is_summary and "USE_ORIGINAL" not in self.history:
                out.append("USE_ORIGINAL")
        # a person is only asked when the verifier is genuinely unsure: a failing claim scored near its threshold
        # (a line with no support at all is simply dropped)
        return out + ["DROP"] + (["HUMAN_REVIEW"] if self.near_threshold(self.review_band) else [])

    def originals(self) -> list[dict]:
        """Cited bullets short enough to be output as they are (some resume 'bullets' are whole skills sections)."""
        return [e for e in self.evidence if len(e["text"].split()) <= self.original_max_words]

    def near_threshold(self, band: float) -> bool:
        s = [u["score"] for u in self.unsupported if u["score"] is not None]
        return bool(s and self.threshold is not None and max(s) >= self.threshold - band)

    def payload(self, band: float) -> dict:
        return {"line": self.line, "is_summary": self.is_summary, "requirements": self.requirements,
                "cited_evidence": self.evidence, "unsupported_claims": self.unsupported,
                "rule_check_missing": self.rule_missing, "verifier_threshold": self.threshold,
                "near_threshold": self.near_threshold(band), "attempts": self.attempts, "limits": self.limits,
                "previous_actions": self.history, "allowed_actions": self.allowed_actions()}


def rule_router(st: LineState, band: float = 0.1) -> Decision:
    """Deterministic router: the baseline controller, and the fallback when the LLM's choice is invalid."""
    allowed = st.allowed_actions()
    pick = lambda a, ft, why, **kw: Decision(a, ft, why, source="rule", **kw)  # noqa: E731
    if st.near_threshold(band) and "HUMAN_REVIEW" in allowed and st.attempts["total"] >= 1:
        return pick("HUMAN_REVIEW", "ambiguous_entailment", "score close to the threshold after an attempt")
    if st.rule_missing and "REWRITE_CONSERVATIVELY" in allowed:
        return pick("REWRITE_CONSERVATIVELY", "unsupported_embellishment",
                    f"not in the evidence: {', '.join(st.rule_missing)}",
                    instructions=f"Remove {', '.join(st.rule_missing)}; keep only what the evidence states.")
    if not st.evidence and "RETRIEVE_MORE" in allowed:
        return pick("RETRIEVE_MORE", "retrieval_miss", "the line cites no evidence", recovery_query=st.line)
    if st.attempts["rewrite"] >= 1 and "USE_ORIGINAL" in allowed:
        return pick("USE_ORIGINAL", "unnecessary_rewriting", "a rewrite did not fix it; fall back to the original bullet")
    if "REWRITE_CONSERVATIVELY" in allowed:
        return pick("REWRITE_CONSERVATIVELY", "unsupported_embellishment", "unsupported wording",
                    instructions="Remove or soften the unsupported claims; keep only what the evidence states.")
    return pick("DROP", "no_evidence", "no action left that is likely to help")


def llm_decide(llm, st: LineState, band: float = 0.1) -> tuple[Decision, Usage, dict]:
    """One LLM call (plus one repair call if the reply is not valid). Falls back to HUMAN_REVIEW, never to a guess."""
    usage, user = Usage(), "<state>\n" + json.dumps(st.payload(band), ensure_ascii=False, indent=1) + "\n</state>"
    note = {"invalid_replies": 0}
    for attempt in range(2):
        out, u = llm.chat(prompts.CONTROLLER, user if attempt == 0 else
                          user + "\n\nYour previous reply was not valid. Reply with ONLY the JSON object, with an "
                                 "action from allowed_actions.", max_tokens=400)
        usage.add(u)
        try:
            return parse_decision(out), usage, note
        except ValueError:
            note["invalid_replies"] += 1
    return Decision("HUMAN_REVIEW", "ambiguous_entailment", "the controller reply could not be parsed",
                    source="fallback"), usage, note


def decide(llm, st: LineState, mode: str, band: float) -> tuple[Decision, Usage, dict]:
    if mode == "rule" or llm is None:
        return rule_router(st, band), Usage(), {}
    d, u, note = llm_decide(llm, st, band)
    if d.source != "fallback" and d.action not in st.allowed_actions():   # picked something it may not (a failed parse is the exception: that always goes to a person)
        note["overridden_from"] = d.action
        r = rule_router(st, band)
        r.source = "override"
        return r, u, note
    return d, u, note


# ---------- the loop ----------

def _bullet_ids(cites: list[str], id_map: dict) -> list[int]:
    out = []
    for c in cites:
        if c in id_map:
            out.append(id_map[c])
        elif re.fullmatch(r"R\d+", c):
            out.append(int(c[1:]))
    return out


def run_controller_loop(*, lines, cites, claims, pair, requirements, evidence, id_map, llm, retriever, verifier,
                        cfg, verify_lines, regenerate, usage):
    """Process every failing line. Returns (lines, cites, claims, removed, escalated, trace, actions, regenerations).
    `verify_lines(lines, bullets, llm, verifier, claim_mode)` and `regenerate(llm, line, unsupported, evidence_text,
    instructions)` are passed in by the pipeline (keeps this module free of import cycles)."""
    cs, t = settings(cfg), cfg["tailoring"]
    texts = pair.index_texts(cfg["retrieval"].get("contextual", False))
    threshold = getattr(verifier, "threshold", None)
    limits = {"total": cs["max_total"], "retrieval": cs["max_retrieve"], "rewrite": cs["max_rewrite"]}
    lines, cites = list(lines), [list(c) for c in cites] + [[] for _ in range(len(lines) - len(cites))]
    by_line: dict[int, list[Claim]] = {}
    for c in claims:
        by_line.setdefault(c.line_idx, []).append(c)
    removed, escalated, trace, actions, regens = {}, {}, [], {}, 0
    rule = RuleVerifier(resolve(cfg["paths"]["skills_vocab"]))   # complementary guardrail: tools, numbers, seniority

    for i in sorted(i for i, cl in by_line.items() if any(not c.supported for c in cl) and 0 <= i < len(lines)):
        original = lines[i]
        tr = {"line_idx": i, "initial_line": original, "initial_cites": list(cites[i]), "steps": [], "outcome": None}
        st = LineState(
            line_idx=i, line=original, is_summary=original.startswith("Summary: "),
            requirements=[requirements[r] for r, bl in evidence.items() if set(bl) & set(_bullet_ids(cites[i], id_map))],
            evidence=[{"id": c, "text": pair.bullets[b]} for c in cites[i] for b in _bullet_ids([c], id_map)],
            unsupported=[], rule_missing=[], threshold=threshold, limits=limits,
            original_max_words=cs["original_max_words"], review_band=cs["review_band"])
        cur = by_line[i]

        def refresh(cur_claims):
            st.unsupported = [{"claim": c.text, "score": round(c.score, 3) if c.score is not None else None}
                              for c in cur_claims if not c.supported]
            ev = [pair.bullets[b] for b in _bullet_ids(cites[i], id_map)] or pair.bullets
            miss = []
            for c in cur_claims:
                if not c.supported:
                    miss += rule.check(c.text, ev)["missing"]
            st.rule_missing = sorted(set(miss))
        refresh(cur)

        outcome = None
        while outcome is None:
            if st.attempts["total"] >= st.limits["total"]:
                outcome = "loop_limit"
                break
            decision, u, note = decide(llm, st, cs["mode"], cs["review_band"])
            usage.add(u)
            step = {"step": len(tr["steps"]) + 1, "tool": "controller", "action": decision.action,
                    "failure_type": decision.failure_type, "reason": decision.reason, "source": decision.source,
                    "confidence": decision.confidence, "unsupported": list(st.unsupported), **note}
            tr["steps"].append(step)
            actions[decision.action] = actions.get(decision.action, 0) + 1
            if decision.action == "DROP":
                outcome = "dropped"
            elif decision.action == "HUMAN_REVIEW":
                outcome = "escalated"
            elif decision.action == "USE_ORIGINAL":
                cands = [b for c in st.originals() for b in _bullet_ids([c["id"]], id_map)]
                best = max(cands, key=lambda b: verifier.verify(lines[i].removeprefix("Summary: "), [pair.bullets[b]])[1])
                lines[i], cites[i] = pair.bullets[best], [f"R{best}"]
                st.attempts["total"] += 1
                st.history.append("USE_ORIGINAL")
            else:   # REWRITE_CONSERVATIVELY or RETRIEVE_MORE: both end in one generator call
                if decision.action == "RETRIEVE_MORE":
                    new = [b for b, _ in retriever.rank(decision.recovery_query or lines[i], texts, cs["retrieve_k"])]
                    have = set(_bullet_ids(cites[i], id_map))
                    new = [b for b in new if b not in have]
                    cites[i] = cites[i] + [f"R{b}" for b in new]
                    st.attempts["retrieval"] += 1
                    tr["steps"].append({"step": len(tr["steps"]) + 1, "tool": "retriever",
                                        "query": decision.recovery_query or lines[i], "new_evidence": [f"R{b}" for b in new]})
                ev_ids = _bullet_ids(cites[i], id_map)
                ev_text = "\n".join(f"- {pair.bullets[b]}" for b in ev_ids) or "\n".join(f"- {b}" for b in pair.bullets)
                prefix = "Summary: " if lines[i].startswith("Summary: ") else ""
                new_line, u = regenerate(llm, lines[i], [x["claim"] for x in st.unsupported], ev_text,
                                         decision.instructions)
                usage.add(u)
                lines[i] = prefix + new_line.removeprefix("Summary: ")
                if decision.action == "REWRITE_CONSERVATIVELY":
                    st.attempts["rewrite"] += 1
                st.attempts["total"] += 1
                st.history.append(decision.action)
                regens += 1
                tr["steps"].append({"step": len(tr["steps"]) + 1, "tool": "generator", "line": lines[i]})
            # re-verify (not for terminal actions)
            if outcome is None:
                new_claims, u = verify_lines([lines[i]], pair.bullets, llm, verifier, t["claim_split"])
                usage.add(u)
                for c in new_claims:
                    c.line_idx = i
                cur = new_claims
                ok = bool(cur) and all(c.supported for c in cur)
                tr["steps"].append({"step": len(tr["steps"]) + 1, "tool": "verifier", "supported": ok,
                                    "scores": [round(c.score, 3) if c.score is not None else None for c in cur]})
                if ok:
                    outcome = "accepted"
                else:
                    refresh(cur)
                    st.line = lines[i]
                    st.is_summary = lines[i].startswith("Summary: ")
                    st.evidence = [{"id": c, "text": pair.bullets[b]} for c in cites[i] for b in _bullet_ids([c], id_map)]
        tr["outcome"] = outcome
        tr["final_line"] = lines[i] if outcome == "accepted" else None
        tr["attempts"] = dict(st.attempts)
        trace.append(tr)
        if outcome == "accepted":
            by_line[i] = cur
        elif outcome == "escalated":
            escalated[i] = original
        else:   # dropped or loop_limit
            removed[i] = original
        if outcome == "loop_limit":
            actions["LOOP_LIMIT"] = actions.get("LOOP_LIMIT", 0) + 1

    gone = set(removed) | set(escalated)
    keep = [i for i in range(len(lines)) if i not in gone]
    remap = {old: new for new, old in enumerate(keep)}
    out_claims = []
    for i in keep:
        for c in by_line.get(i, []):
            c.line_idx = remap[i]
            out_claims.append(c)
    return ([lines[i] for i in keep], [cites[i] for i in keep], out_claims,
            [removed[i] for i in sorted(removed)], [escalated[i] for i in sorted(escalated)], trace, actions, regens)
