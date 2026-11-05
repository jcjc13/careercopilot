"""Pair-level bootstrap confidence intervals for paired E3 metric differences.

The estimate is (variant - baseline), resampling resume/job pairs with replacement.
This quantifies uncertainty across evaluated pairs, not model randomness or label error.

    python scripts/bootstrap_e3.py --baseline A
    python scripts/bootstrap_e3.py --baseline D --variants E --config configs/matched_controller.yaml
  -> runs/<run>/e3_bootstrap_ci.json
"""
import csv
import json

from _common import load_config, parser, run_dir

from careercopilot.metrics import paired_bootstrap_mean_ci

METRICS = ("unsupported_rate", "coverage", "any_introduced", "latency_s", "cost_usd")
FITS = ("all", "good", "potential", "none")


def calculate(rows, baseline="A", variants=None, replicates=10000, seed=469, confidence=0.95):
    variants = variants or sorted({row["variant"] for row in rows} - {baseline})
    by_key = {}
    for row in rows:
        key = (row["pair_id"], row["variant"])
        if key in by_key:
            raise ValueError(f"duplicate E3 row for pair/variant {key}")
        by_key[key] = row

    output = {
        "method": "paired percentile bootstrap over resume-job pairs",
        "difference": "variant minus baseline",
        "baseline": baseline,
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
        "metrics": {},
    }
    for variant in variants:
        if variant == baseline:
            raise ValueError("baseline cannot also be listed as a comparison variant")
        for fit in FITS:
            pair_ids = sorted(
                pair_id for pair_id, var in by_key
                if var == baseline and (pair_id, variant) in by_key
                and (fit == "all" or by_key[(pair_id, baseline)]["fit"] == fit)
                and (fit == "all" or by_key[(pair_id, variant)]["fit"] == fit)
            )
            if not pair_ids:
                continue
            if any(by_key[(pair_id, baseline)]["fit"] != by_key[(pair_id, variant)]["fit"]
                   for pair_id in pair_ids):
                raise ValueError(f"{variant} and {baseline} fit labels differ for {fit} comparison")
            key = f"{variant}_vs_{baseline}/{fit}"
            output["metrics"][key] = {}
            for metric in METRICS:
                differences = [
                    float(by_key[(pair_id, variant)][metric]) -
                    float(by_key[(pair_id, baseline)][metric])
                    for pair_id in pair_ids
                ]
                output["metrics"][key][metric] = paired_bootstrap_mean_ci(
                    differences, replicates=replicates, seed=seed, confidence=confidence)
    return output


def main(config="configs/default.yaml", baseline="A", variants=None, replicates=10000, confidence=0.95):
    cfg = load_config(config)
    path = run_dir(cfg) / "e3_per_resume.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/evaluate_e3.py first")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no per-resume E3 rows in {path}")

    result = calculate(rows, baseline, variants, replicates, cfg["seed"], confidence)
    output_path = run_dir(cfg) / "e3_bootstrap_ci.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"paired bootstrap: {replicates} resamples, {confidence:.0%} percentile intervals")
    print(f"difference = variant - {baseline}; negative unsupported rate / cost / latency favors variant")
    print(f"wrote {output_path}")
    print(f"\n{'comparison/fit':<28}{'metric':<20}{'n':>4}{'mean diff':>12}{'95% CI':>26}")
    for comparison, metrics in result["metrics"].items():
        for metric, ci in metrics.items():
            print(f"{comparison:<28}{metric:<20}{ci['n_pairs']:>4}{ci['mean_difference']:>12.4f}"
                  f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]".rjust(26))
    return result


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--baseline", default="A")
    cli.add_argument("--variants", help="comma-separated comparison variants; default is all other variants")
    cli.add_argument("--replicates", type=int, default=10000)
    cli.add_argument("--confidence", type=float, default=0.95)
    args = cli.parse_args()
    main(args.config, args.baseline, args.variants.split(",") if args.variants else None,
         args.replicates, args.confidence)
