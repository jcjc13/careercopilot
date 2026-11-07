"""Before/after score card for one tailored resume (used by the demo app)."""
from __future__ import annotations

from . import prompts
from .llm import parse_json
from .metrics import introduced_facts
from .schema import Usage


def covered_requirements(llm, requirements: list[str], claims: list[str]) -> tuple[set[int], Usage]:
    """Indices (0-based) of the requirements that at least one claim clearly addresses."""
    if not claims or not requirements:
        return set(), Usage()
    reqs = "\n".join(f"[{i + 1}] {r}" for i, r in enumerate(requirements))
    cl = "\n".join(f"- {c}" for c in claims)
    out, u = llm.chat(prompts.COVERAGE, f"<requirements>\n{reqs}\n</requirements>\n\n<lines>\n{cl}\n</lines>",
                      max_tokens=100)
    try:
        return {int(x) - 1 for x in parse_json(out) if 1 <= int(x) <= len(requirements)}, u
    except (ValueError, TypeError):
        return set(), u


def accepted_lines(res) -> list[int]:
    """Lines kept by default: those with no claim the verifier rejected (the app's default selection)."""
    bad = {c.line_idx for c in res.pipeline_claims if not c.supported}
    return [i for i in range(len(res.lines)) if i not in bad]


def score_card(llm, pair, res, terms: list[str]) -> dict:
    """Job match before and after, how much of the tailored text the verifier supports, and invented facts."""
    kept = accepted_lines(res)
    lines = [res.lines[i].removeprefix("Summary: ") for i in kept]
    before, _ = covered_requirements(llm, res.requirements, pair.bullets)
    after, _ = covered_requirements(llm, res.requirements, lines)
    claims = [c for c in res.pipeline_claims if c.line_idx in kept]
    intro = introduced_facts("\n".join(lines), pair.resume, terms)
    return {"requirements": res.requirements, "before": sorted(before), "after": sorted(after),
            "claims": len(claims), "supported": sum(bool(c.supported) for c in claims),
            "flagged_lines": len(res.trace), "removed_lines": len(res.removed_lines),
            "new_tools": intro["new_tools"], "new_numbers": intro["new_numbers"]}
