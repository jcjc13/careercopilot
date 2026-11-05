"""Run the pipeline on ONE pair and print everything -- the quickest way to see the system work
and to debug prompts.

    python scripts/tailor_one.py --pair sample-potential --variant D
    python scripts/tailor_one.py --pair sample-none --variant A --mock
"""
from _common import build_llm, load_config, load_pairs_or_exit, parser

from careercopilot.pipeline import run_variant
from careercopilot.retrieval import build_retriever
from careercopilot.verifiers import build_verifier


def main(config, pair_id, variant, mock=False, retriever=None, verifier=None):
    cfg = load_config(config)
    pairs = {p.pair_id: p for p in load_pairs_or_exit(cfg)}
    if pair_id not in pairs:
        raise SystemExit(f"unknown pair {pair_id!r}; available: {', '.join(list(pairs)[:10])}...")
    pair = pairs[pair_id]
    llm = build_llm(cfg, mock=mock)
    res = run_variant(pair, variant, llm, build_retriever(cfg, retriever),
                      build_verifier(cfg, llm, verifier), cfg)

    print(f"\n=== {pair.pair_id} ({pair.fit} fit) - variant {variant} ===")
    print("\nRequirements:")
    for i, r in enumerate(res.requirements):
        ev = res.evidence.get(i, [])
        print(f"  {i + 1}. {r}")
        for b in ev:
            print(f"       evidence: {pair.bullets[b]}")
    print("\nTailored resume:")
    for i, line in enumerate(res.lines):
        cite = f"  [{', '.join(res.citations[i])}]" if i < len(res.citations) and res.citations[i] else ""
        print(f"  {'' if i == 0 else '- '}{line}{cite}")
    if res.pipeline_claims:
        print("\nVerifier view:")
        for c in res.pipeline_claims:
            print(f"  {'OK ' if c.supported else 'BAD'} {c.text}")
    if res.removed_lines:
        print("\nRemoved (still unsupported after regeneration):")
        for l in res.removed_lines:
            print(f"  x {l}")
    u = res.usage
    print(f"\nCost: {u.llm_calls} LLM calls, {u.tokens_in}+{u.tokens_out} tokens, "
          f"{u.latency_s:.1f}s, ${u.cost_usd:.4f}; regenerations: {res.regenerations}")
    return res


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--pair", default="sample-potential")
    p.add_argument("--variant", default="D", choices=list("ABCDE"))
    p.add_argument("--retriever", default=None, help="override config: bm25 | embedding | hybrid")
    p.add_argument("--verifier", default=None, help="override config: rule | nli | llm")
    a = p.parse_args()
    main(a.config, a.pair, a.variant, a.mock, a.retriever, a.verifier)
