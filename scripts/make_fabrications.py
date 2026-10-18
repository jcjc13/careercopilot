"""E2 test set: for ~150 real resume bullets, write one faithful paraphrase and one fabricated
copy with ONE planted error. Tool swaps and number changes are scripted; paraphrases, invented
duties and title/date changes are drafted by the LLM and MUST be reviewed by a person.

    python scripts/make_fabrications.py --n 150
  -> data/labels/e2_claims.csv  columns: claim_id, pair_id, source_bullet, claim, label, type,
                                method, reviewed
Review every row with method=llm_draft: fix it so it is exactly what its label says
(faithful = same facts; fabricated = exactly one unsupported change), then set reviewed=yes.
"""
import csv
import random

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, tqdm

from careercopilot import prompts
from careercopilot.corrupt import change_number, swap_tool
from careercopilot.text import find_numbers, find_terms, load_vocab

KINDS = ["tool", "number", "duty", "date_title"]


def main(config, n=150, mock=False):
    cfg = load_config(config)
    rng = random.Random(cfg["seed"])
    vocab = load_vocab(resolve(cfg["paths"]["skills_vocab"]))
    llm = build_llm(cfg, mock=mock)
    pairs = load_pairs_or_exit(cfg)

    # one bullet pool across all resumes, deduplicated (same resume can appear in many pairs)
    pool, seen = [], set()
    for p in pairs:
        for b in p.bullets:
            if b not in seen and len(b.split()) >= 6:
                seen.add(b)
                pool.append((p, b))
    rng.shuffle(pool)
    # prefer bullets that contain a tool or a number, so scripted corruptions apply
    all_terms = [t for c, ts in vocab.items() if c != "seniority" for t in ts]
    pool.sort(key=lambda x: -(bool(find_terms(x[1], all_terms)) + bool(find_numbers(x[1]))))
    pool = pool[:n]

    rows, cid = [], 0
    for k, (p, b) in enumerate(tqdm(pool, desc="claims")):
        para, _ = llm.chat(prompts.PARAPHRASE, f"<bullet>\n{b}\n</bullet>", max_tokens=200)
        rows.append([f"c{cid}", p.pair_id, b, para.strip(), "faithful", "paraphrase", "llm_draft", "no"])
        cid += 1
        kind = KINDS[k % len(KINDS)]
        fab, method = None, "scripted"
        if kind == "tool":
            fab = swap_tool(b, p.resume, vocab, rng)
        elif kind == "number":
            fab = change_number(b, rng)
        if fab is None:  # duty / date_title, or no tool/number to change -> LLM draft
            sys = prompts.FABRICATE.format(kind=prompts.FABRICATION_KINDS[kind], data_rule=prompts.DATA_RULE)
            fab, _ = llm.chat(sys, f"<bullet>\n{b}\n</bullet>", max_tokens=200)
            method = "llm_draft"
        rows.append([f"c{cid}", p.pair_id, b, fab.strip(), "fabricated", kind, method,
                     "yes" if method == "scripted" else "no"])
        cid += 1

    out = resolve(cfg["paths"]["labels_dir"]) / "e2_claims.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "pair_id", "source_bullet", "claim", "label", "type", "method", "reviewed"])
        w.writerows(rows)
    n_draft = sum(r[6] == "llm_draft" for r in rows)
    print(f"wrote {len(rows)} claims to {out}; {n_draft} LLM drafts need review (set reviewed=yes)")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--n", type=int, default=150, help="number of source bullets")
    a = p.parse_args()
    main(a.config, a.n, a.mock)
