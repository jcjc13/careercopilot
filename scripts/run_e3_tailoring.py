"""E3 step 1: run tailoring variants A-E on every pair. Resumable (skips pair/variant combos
already in the output), so a crash or rate limit doesn't lose work.

    python scripts/run_e3_tailoring.py --variants A,B,C,D
    python scripts/run_e3_tailoring.py --variants D,E  # matched controller / fixed-loop comparison
  -> runs/<run>/requirements.jsonl   requirements per pair (shared by all variants)
  -> runs/<run>/e3_outputs.jsonl     one TailorResult per pair x variant
Set retrieval.method and verifier.method in the config to the E1/E2 winners first.
"""
from _common import build_llm, load_config, load_pairs_or_exit, parser, run_dir, set_seed, tqdm

from careercopilot.generation import extract_requirements
from careercopilot.pipeline import run_variant
from careercopilot.retrieval import build_retriever
from careercopilot.schema import TailorResult, Usage, append_jsonl, read_jsonl
from careercopilot.verifiers import build_verifier


def main(config, variants, mock=False, limit=None):
    cfg = load_config(config)
    set_seed(cfg["seed"])
    pairs = load_pairs_or_exit(cfg, limit)
    llm = build_llm(cfg, mock=mock)
    retriever, verifier = build_retriever(cfg), build_verifier(cfg, llm)
    out = run_dir(cfg)
    req_path, res_path = out / "requirements.jsonl", out / "e3_outputs.jsonl"
    reqs = {r["pair_id"]: r for r in read_jsonl(req_path)}
    done = {(r["pair_id"], r["variant"]) for r in read_jsonl(res_path)}
    print(f"retriever={retriever.name} verifier={verifier.name} model={getattr(llm, 'model', '?')}")

    for p in tqdm(pairs, desc="pairs"):
        if p.pair_id not in reqs:
            r, u = extract_requirements(llm, p.job, cfg["tailoring"]["n_requirements"])
            reqs[p.pair_id] = {"pair_id": p.pair_id, "requirements": r, "usage": vars(u)}
            append_jsonl(req_path, reqs[p.pair_id])
        rq = reqs[p.pair_id]
        for v in variants:
            if (p.pair_id, v) in done:
                continue
            try:
                res = run_variant(p, v, llm, retriever, verifier, cfg,
                                  requirements=rq["requirements"], req_usage=Usage(**rq["usage"]))
            except Exception as e:  # keep going; failures are visible in evaluation
                res = TailorResult(pair_id=p.pair_id, variant=v, fit=p.fit, lines=[],
                                   requirements=rq["requirements"], error=repr(e))
                print(f"error on {p.pair_id}/{v}: {e!r}")
            append_jsonl(res_path, res)
    print(f"wrote {res_path}")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--variants", default="A,B,C,D")
    a = p.parse_args()
    main(a.config, a.variants.split(","), a.mock, a.limit)
