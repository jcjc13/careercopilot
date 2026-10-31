"""The tailoring pipeline and its four E3 variants.

    A  unrestricted rewrite                    (baseline)
    B  grounded rewrite from retrieved evidence
    C  unrestricted rewrite + verify/regenerate loop
    D  grounded rewrite + verify + bounded controller   (full system; run v2 used the fixed loop, run v3 the controller)

Requirements are extracted ONCE per pair and shared by all variants (so all variants are
judged against the same list); their cost is charged to the variants that use them (B, D).
"""
from __future__ import annotations

from . import controller
from .generation import (extract_requirements, format_evidence, regenerate_line, rewrite_conservatively, split_claims,
                         tailor_grounded, tailor_unrestricted)
from .schema import Pair, TailorResult, Usage

VARIANTS = {
    "A": {"grounded": False, "verify": False},
    "B": {"grounded": True, "verify": False},
    "C": {"grounded": False, "verify": True},
    "D": {"grounded": True, "verify": True, "controller": True},
}


def retrieve_evidence(retriever, requirements: list[str], texts: list[str], k: int) -> dict:
    """texts = pair.index_texts(contextual): the bullets, optionally with their role line."""
    return {i: [b for b, _ in retriever.rank(r, texts, k)] for i, r in enumerate(requirements)}


def verify_lines(lines, bullets, llm, verifier, claim_mode):
    claims, u = split_claims(llm, lines, claim_mode)
    for c in claims:
        ok, score, vu = verifier.verify(c.text, bullets)
        c.supported, c.score, c.verifier = ok, score, verifier.name
        u.add(vu)
    return claims, u


def run_variant(pair: Pair, variant: str, llm, retriever, verifier, cfg: dict,
                requirements: list[str] | None = None, req_usage: Usage | None = None) -> TailorResult:
    spec, t = VARIANTS[variant], cfg["tailoring"]
    usage = Usage()
    if requirements is None:
        requirements, req_usage = extract_requirements(llm, pair.job, t["n_requirements"])
    res = TailorResult(pair_id=pair.pair_id, variant=variant, fit=pair.fit, lines=[],
                       requirements=requirements)

    # 1. generate
    if spec["grounded"]:
        usage.add(req_usage or Usage())
        texts = pair.index_texts(cfg["retrieval"].get("contextual", False))
        res.evidence = retrieve_evidence(retriever, requirements, texts, cfg["retrieval"]["top_k"])
        lines, cites, _, u = tailor_grounded(llm, requirements, res.evidence, pair.bullets,
                                             pair.job, t["max_bullets"])
    else:
        lines, cites, u = tailor_unrestricted(llm, pair.resume, pair.job, t["max_bullets"])
    usage.add(u)

    # 2. verify -> regenerate unsupported lines -> drop lines that still fail
    if spec["verify"]:
        evidence_text = "\n".join(f"- {b}" for b in pair.bullets)
        claims, u = verify_lines(lines, pair.bullets, llm, verifier, t["claim_split"])
        usage.add(u)
    if spec["verify"] and spec.get("controller"):
        _, id_map = format_evidence(requirements, res.evidence, pair.bullets)
        (lines, cites, claims, res.removed_lines, res.escalated_lines, res.trace, res.actions,
         res.regenerations) = controller.run_controller_loop(
            lines=lines, cites=cites, claims=claims, pair=pair, requirements=requirements, evidence=res.evidence,
            id_map=id_map, llm=llm, retriever=retriever, verifier=verifier, cfg=cfg, verify_lines=verify_lines,
            regenerate=rewrite_conservatively, usage=usage)
        res.pipeline_claims = claims
    elif spec["verify"]:
        for _ in range(t["max_regen"]):
            bad = sorted({c.line_idx for c in claims if not c.supported and 0 <= c.line_idx < len(lines)})
            if not bad:
                break
            for i in bad:
                unsupported = [c.text for c in claims if c.line_idx == i and not c.supported]
                prefix = "Summary: " if lines[i].startswith("Summary: ") else ""
                new, u = regenerate_line(llm, lines[i], unsupported, evidence_text)
                usage.add(u)
                lines[i] = prefix + new.removeprefix("Summary: ")
                res.regenerations += 1
            # re-check only the regenerated lines
            sub = [lines[i] for i in bad]
            new_claims, u = verify_lines(sub, pair.bullets, llm, verifier, t["claim_split"])
            usage.add(u)
            new_claims = [c for c in new_claims if 0 <= c.line_idx < len(bad)]
            for c in new_claims:
                c.line_idx = bad[c.line_idx]
            claims = [c for c in claims if c.line_idx not in bad] + new_claims
        still_bad = {c.line_idx for c in claims if not c.supported}
        res.removed_lines = [lines[i] for i in sorted(still_bad) if i < len(lines)]
        keep = [i for i in range(len(lines)) if i not in still_bad]
        remap = {old: new for new, old in enumerate(keep)}
        lines, cites = [lines[i] for i in keep], [cites[i] if i < len(cites) else [] for i in keep]
        for c in claims:
            c.line_idx = remap.get(c.line_idx, -1)
        res.pipeline_claims = [c for c in claims if c.line_idx >= 0]

    res.lines, res.citations, res.usage = lines, cites, usage
    return res
