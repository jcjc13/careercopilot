"""LLM steps: requirement extraction, tailoring (unrestricted / grounded), regeneration,
claim splitting. Each returns its result plus a Usage record."""
from __future__ import annotations

import re

from . import prompts
from .llm import parse_json
from .schema import Claim, Usage


def extract_requirements(llm, job: str, n: int = 8) -> tuple[list[str], Usage]:
    out, u = llm.chat(prompts.REQUIREMENTS.format(n=n), f"<job>\n{job}\n</job>", max_tokens=400)
    try:
        reqs = [str(r).strip() for r in parse_json(out) if str(r).strip()]
    except ValueError:
        reqs = [l.strip(" -•*0123456789.") for l in out.splitlines() if l.strip()]
    return reqs[:n], u


_CITE = re.compile(r"\s*\[(E\d+(?:\s*,\s*E\d+)*)\]\s*$")


def _parse_tailored(out: str) -> tuple[list[str], list[list[str]]]:
    """Returns (lines, citations); line 0 is the summary."""
    try:
        d = parse_json(out)
        summary, bullets = str(d.get("summary", "")).strip(), [str(b) for b in d.get("bullets", [])]
    except (ValueError, AttributeError):
        rows = [l.strip(" -•*") for l in out.splitlines() if l.strip()]
        summary, bullets = (rows[0] if rows else ""), rows[1:]
    lines, cites = ["Summary: " + summary], [[]]
    for b in bullets:
        m = _CITE.search(b)
        cites.append([c.strip() for c in m.group(1).split(",")] if m else [])
        lines.append(_CITE.sub("", b).strip())
    return lines, cites


def tailor_unrestricted(llm, resume: str, job: str, max_bullets: int = 10):
    sys = prompts.TAILOR_UNRESTRICTED.format(max_bullets=max_bullets)
    out, u = llm.chat(sys, f"<resume>\n{resume}\n</resume>\n\n<job>\n{job}\n</job>", max_tokens=1200)
    lines, cites = _parse_tailored(out)
    return lines, cites, u


def format_evidence(requirements: list[str], evidence: dict, bullets: list[str]) -> tuple[str, dict]:
    """Numbered evidence block E1..En (deduplicated across requirements) + id -> bullet idx map."""
    ids, order = {}, []
    for r_idx in range(len(requirements)):
        for b in evidence.get(r_idx, []):
            if b not in ids:
                ids[b] = f"E{len(ids) + 1}"
                order.append(b)
    req_block = "\n".join(
        f"{i + 1}. {r}  (evidence: {', '.join(ids[b] for b in evidence.get(i, [])) or 'none'})"
        for i, r in enumerate(requirements))
    ev_block = "\n".join(f"[{ids[b]}] {bullets[b]}" for b in order)
    return f"<requirements>\n{req_block}\n</requirements>\n\n<evidence>\n{ev_block}\n</evidence>", \
        {v: k for k, v in ids.items()}


def tailor_grounded(llm, requirements, evidence, bullets, job: str, max_bullets: int = 10):
    block, id_map = format_evidence(requirements, evidence, bullets)
    sys = prompts.TAILOR_GROUNDED.format(max_bullets=max_bullets)
    out, u = llm.chat(sys, f"<job>\n{job}\n</job>\n\n{block}", max_tokens=1200)
    lines, cites = _parse_tailored(out)
    return lines, cites, id_map, u


def regenerate_line(llm, line: str, unsupported: list[str], evidence_text: str) -> tuple[str, Usage]:
    user = (f"<line>\n{line}\n</line>\n\nUnsupported claims:\n" +
            "\n".join(f"- {c}" for c in unsupported) +
            f"\n\n<evidence>\n{evidence_text}\n</evidence>")
    out, u = llm.chat(prompts.REGENERATE, user, max_tokens=200)
    return out.strip().strip('"'), u


def split_claims(llm, lines: list[str], mode: str = "llm") -> tuple[list[Claim], Usage]:
    """Atomic claims for each output line. mode='line' treats each line as one claim (no LLM)."""
    if mode == "line" or llm is None:
        return [Claim(text=l.removeprefix("Summary: "), line_idx=i) for i, l in enumerate(lines) if l.strip()], Usage()
    numbered = "\n".join(f"[{i}] {l}" for i, l in enumerate(lines) if l.strip())
    out, u = llm.chat(prompts.SPLIT_CLAIMS, f"<lines>\n{numbered}\n</lines>", max_tokens=1500)
    try:
        items = parse_json(out)
        claims = [Claim(text=str(d["claim"]).strip(), line_idx=int(d["line"]))
                  for d in items if str(d.get("claim", "")).strip()]
    except (ValueError, KeyError, TypeError):
        claims = []
    if not claims:  # fall back rather than lose the resume
        claims, _ = split_claims(None, lines, "line")
    return claims, u
