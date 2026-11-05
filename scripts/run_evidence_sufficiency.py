"""Counterfactual evidence-sufficiency probe using manually labelled E1 evidence.

For each labelled requirement, compare the original resume with versions where all but one,
or all, E1-labelled supporting bullets are removed. This measures behavior under controlled
evidence removal; it does not establish that the remaining resume has no equivalent evidence.

    python scripts/run_evidence_sufficiency.py --variants A,D
    python scripts/run_evidence_sufficiency.py --variants A,D --limit 2 --mock
  -> runs/<run>/evidence_sufficiency.csv
"""
import csv
import re
import sys

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, run_dir, tqdm

from careercopilot.pipeline import run_variant, verify_lines
from careercopilot.retrieval import build_retriever
from careercopilot.schema import Pair
from careercopilot.text import split_with_context
from careercopilot.verifiers import build_verifier
from evaluate_e3 import coverage

FIELDS = [
    "pair_id", "req_id", "variant", "condition", "requirement",
    "original_gold_bullet_ids", "removed_bullet_ids", "retained_gold_bullet_count",
    "remaining_resume_bullet_count", "claim_count", "supported_claim_count",
    "unsupported_claim_rate", "supported_requirement_coverage", "any_supported_output",
    "abstained", "escalated_line_count", "removed_output_line_count", "latency_s",
    "cost_usd", "output", "verifier", "mock",
]


def _gold_ids(value):
    return sorted({int(x) for x in value.replace(",", ";").split(";") if x.strip()})


def _remove_bullets(pair: Pair, removed_ids: list[int]) -> Pair:
    """Remove labelled evidence from both the source text and the indexed bullet list."""
    resume = pair.resume
    for bullet_id in removed_ids:
        if not 0 <= bullet_id < len(pair.bullets):
            raise ValueError(f"{pair.pair_id}: gold bullet id {bullet_id} is out of range")
        bullet = pair.bullets[bullet_id].strip()
        pattern = re.compile(r"\s+".join(re.escape(token) for token in bullet.split()), re.I)
        matches = list(pattern.finditer(resume))
        if len(matches) != 1:
            raise ValueError(
                f"{pair.pair_id}: expected one source-text match for gold bullet {bullet_id}, "
                f"found {len(matches)}"
            )
        match = matches[0]
        resume = resume[:match.start()] + resume[match.end():]

    keep = [i for i in range(len(pair.bullets)) if i not in set(removed_ids)]
    bullets = [pair.bullets[i] for i in keep]
    contexts = [pair.contexts[i] for i in keep] if pair.contexts else []
    remaining_bullets, _ = split_with_context(resume)
    remaining = {re.sub(r"\s+", " ", b.strip().lower()) for b in remaining_bullets}
    for bullet_id in removed_ids:
        normalized = re.sub(r"\s+", " ", pair.bullets[bullet_id].strip().lower())
        if normalized in remaining:
            raise ValueError(f"{pair.pair_id}: gold bullet {bullet_id} remains in the transformed resume")

    return Pair(pair.pair_id, resume, pair.job, pair.fit, bullets, contexts)


def build_conditions(pair: Pair, gold_ids: list[int]) -> list[tuple[str, Pair, list[int]]]:
    """Create paired conditions; partial retains one labelled supporting bullet."""
    if not gold_ids:
        return []
    conditions = [("intact", pair, [])]
    if len(gold_ids) > 1:
        conditions.append(("partial", _remove_bullets(pair, gold_ids[1:]), gold_ids[1:]))
    conditions.append(("labelled_evidence_removed", _remove_bullets(pair, gold_ids), gold_ids))
    return conditions


def _read_labelled_rows(cfg, pairs):
    labels_path = resolve(cfg["paths"]["labels_dir"]) / "e1_labels.csv"
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} not found -- label E1 supporting bullet IDs first")

    by_id = {p.pair_id: p for p in pairs}
    rows, skipped = [], 0
    with labels_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("gold_bullet_ids", "").strip():
                continue
            pair = by_id.get(row.get("pair_id"))
            if pair is None:
                skipped += 1
                continue
            gold_ids = _gold_ids(row["gold_bullet_ids"])
            if any(i < 0 or i >= len(pair.bullets) for i in gold_ids):
                print(f"skip {pair.pair_id}/{row.get('req_id')}: gold bullet ID out of range", file=sys.stderr)
                skipped += 1
                continue
            rows.append((row, pair, gold_ids))
    if skipped:
        print(f"skipped {skipped} E1 label rows with missing pairs or invalid bullet IDs", file=sys.stderr)
    if not rows:
        raise SystemExit(f"no usable labelled E1 requirements found in {labels_path}")
    return rows


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(config="configs/default.yaml", variants=None, limit=None, mock=False, verifier_method=None,
         resume=False):
    cfg = load_config(config)
    if verifier_method is not None:
        if verifier_method not in {"rule", "nli", "finetuned", "finetuned_nli", "llm"}:
            raise ValueError(f"unknown verifier method: {verifier_method}")
        cfg["verifier"]["method"] = verifier_method
    pairs = load_pairs_or_exit(cfg)
    labelled = _read_labelled_rows(cfg, pairs)
    if limit is not None:
        labelled = labelled[:limit]
    variants = variants or ["A", "B", "C", "D"]
    invalid = sorted(set(variants) - {"A", "B", "C", "D"})
    if invalid:
        raise ValueError(f"unknown variants: {', '.join(invalid)}")

    llm = build_llm(cfg, mock=mock)
    judge_llm = build_llm(cfg, which="eval_judge", mock=mock)
    retriever = build_retriever(cfg)
    verifier = build_verifier(cfg, llm)
    output = run_dir(cfg) / "evidence_sufficiency.csv"
    rows = []
    completed = set()
    if resume and output.exists():
        with output.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for field in ("claim_count", "supported_claim_count", "any_supported_output"):
                row[field] = int(row[field])
            row["supported_requirement_coverage"] = float(row["supported_requirement_coverage"])
        expected_verifier = verifier.name
        if any(r["verifier"] != expected_verifier or int(r["mock"]) != int(mock) for r in rows):
            raise ValueError("existing results use a different verifier or mock setting; choose a new run_name")
        completed = {(r["pair_id"], r["req_id"], r["variant"], r["condition"]) for r in rows}

    skipped_conditions = 0
    for label, original, gold_ids in tqdm(labelled, desc="labelled requirements"):
        requirement = label["requirement"].strip()
        try:
            conditions = build_conditions(original, gold_ids)
        except ValueError as exc:
            print(f"skip {original.pair_id}/{label['req_id']}: {exc}", file=sys.stderr)
            skipped_conditions += 1
            continue
        new_rows = []
        for condition, pair, removed_ids in conditions:
            for variant in variants:
                row_key = (original.pair_id, label["req_id"], variant, condition)
                if row_key in completed:
                    continue
                result = run_variant(pair, variant, llm, retriever, verifier, cfg,
                                     requirements=[requirement])
                claims, claim_usage = verify_lines(
                    result.lines, pair.bullets, llm, verifier, cfg["tailoring"]["claim_split"])
                result.usage.add(claim_usage)
                supported = [claim.text for claim in claims if claim.supported]
                claim_count = len(claims)
                supported_count = len(supported)
                target_coverage, coverage_usage = coverage(judge_llm, [requirement], supported)
                result.usage.add(coverage_usage)
                row = {
                    "pair_id": original.pair_id,
                    "req_id": label["req_id"],
                    "variant": variant,
                    "condition": condition,
                    "requirement": requirement,
                    "original_gold_bullet_ids": ";".join(map(str, gold_ids)),
                    "removed_bullet_ids": ";".join(map(str, removed_ids)),
                    "retained_gold_bullet_count": len(set(gold_ids) - set(removed_ids)),
                    "remaining_resume_bullet_count": len(pair.bullets),
                    "claim_count": claim_count,
                    "supported_claim_count": supported_count,
                    "unsupported_claim_rate": (
                        (claim_count - supported_count) / claim_count if claim_count else ""
                    ),
                    "supported_requirement_coverage": target_coverage,
                    "any_supported_output": int(supported_count > 0),
                    "abstained": int(not any(
                        line.strip() and not line.lower().startswith("summary:")
                        for line in result.lines
                    )),
                    "escalated_line_count": len(result.escalated_lines),
                    "removed_output_line_count": len(result.removed_lines),
                    "latency_s": round(result.usage.latency_s, 3),
                    "cost_usd": round(result.usage.cost_usd, 6),
                    "output": " | ".join(result.lines),
                    "verifier": verifier.name,
                    "mock": int(mock),
                }
                rows.append(row)
                new_rows.append(row)
                completed.add(row_key)
        if new_rows:
            _write_rows(output, rows)

    if not rows:
        raise SystemExit("no valid evidence-sufficiency conditions could be evaluated")

    print(f"\n{'variant':<9}{'condition':<25}{'n':>5}{'unsupported':>14}"
          f"{'safe coverage':>15}{'supported output':>18}")
    for variant in variants:
        for condition in ("intact", "partial", "labelled_evidence_removed"):
            group = [r for r in rows if r["variant"] == variant and r["condition"] == condition]
            if not group:
                continue
            claim_n = sum(r["claim_count"] for r in group)
            unsupported_n = sum(
                r["claim_count"] - r["supported_claim_count"] for r in group
            )
            supported_output = sum(r["any_supported_output"] for r in group) / len(group)
            unsupported_rate = unsupported_n / claim_n if claim_n else 0.0
            safe_coverage = sum(r["supported_requirement_coverage"] for r in group) / len(group)
            print(f"{variant:<9}{condition:<25}{len(group):>5}{unsupported_rate:>14.3f}"
                  f"{safe_coverage:>15.3f}{supported_output:>18.3f}")
    print(f"skipped {skipped_conditions} labelled requirements with inconsistent evidence references")
    print(f"wrote {output}")


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--variants", default="A,B,C,D")
    cli.add_argument("--verifier", choices=["rule", "nli", "finetuned", "finetuned_nli", "llm"],
                     help="override verifier.method from the config")
    cli.add_argument("--resume", action="store_true", help="skip condition/variant rows already saved")
    args = cli.parse_args()
    main(args.config, args.variants.split(","), args.limit, args.mock, args.verifier, args.resume)
