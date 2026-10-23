"""E2 test set: for ~150 real resume bullets, write one faithful paraphrase and one fabricated
copy with ONE planted error. Tool swaps and quantity changes are scripted; paraphrases, invented
duties and title/date changes are drafted by the LLM and MUST be reviewed by a person.

    python scripts/make_fabrications.py --n 150
  -> data/labels/e2_claims.csv  columns: claim_id, pair_id, source_bullet, claim, label, type,
                                method, drafted_by, set, reviewed
Review every row with method=llm_draft: fix it so it is exactly what its label says
(faithful = same facts; fabricated = exactly one unsupported change), then set reviewed=yes.

Source bullets skip junk (skill lists, date/education lines, run-together fields). The four
fabrication types are balanced: a tool/number row only uses a bullet where the scripted change
applies (a real quantity such as 45% or 10+ users, never a year or date), so few rows fall back to
an LLM draft. Drafts that come back empty, unchanged or with markdown are skipped and replaced.
"""
import csv
import random

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, tqdm

from careercopilot import prompts
from careercopilot.corrupt import change_quantity, clean_llm, is_junk, swap_tool
from careercopilot.text import load_vocab

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
            if b not in seen and len(b.split()) >= 6 and not is_junk(b):
                seen.add(b)
                pool.append((p, b))
    rng.shuffle(pool)
    used = set()

    def pick(kind):
        """(index, scripted corruption) of the next unused bullet; tool/number need a bullet where
        the scripted change applies, and only fall back to an LLM draft when none is left."""
        for i, (p, b) in enumerate(pool):
            if i in used:
                continue
            fab = swap_tool(b, p.resume, vocab, rng) if kind == "tool" else \
                change_quantity(b, rng) if kind == "number" else None
            if kind in ("tool", "number") and (fab is None or fab == b):
                continue
            return i, fab
        left = [i for i in range(len(pool)) if i not in used]
        return (left[0], None) if left else (None, None)

    rows, cid, done = [], 0, 0
    bar = tqdm(total=n, desc="claims")
    while done < n:
        kind = KINDS[done % len(KINDS)]
        i, fab = pick(kind)
        if i is None:
            print(f"ran out of usable bullets after {done} items")
            break
        used.add(i)
        p, b = pool[i]
        para, _ = llm.chat(prompts.PARAPHRASE, f"<bullet>\n{b}\n</bullet>", max_tokens=200)
        para, method = clean_llm(para, b), "scripted"
        if fab is None:  # duty / date_title, or no tool/number change left -> LLM draft
            sys = prompts.FABRICATE.format(kind=prompts.FABRICATION_KINDS[kind], data_rule=prompts.DATA_RULE)
            fab, _ = llm.chat(sys, f"<bullet>\n{b}\n</bullet>", max_tokens=200)
            fab, method = clean_llm(fab, b), "llm_draft"
        if not para or not fab:   # unusable draft: skip this bullet, a different one takes its place
            continue
        rows.append([f"c{cid}", p.pair_id, b, para, "faithful", "paraphrase", "llm_draft", "no"])
        rows.append([f"c{cid + 1}", p.pair_id, b, fab, "fabricated", kind, method,
                     "yes" if method == "scripted" else "no"])
        cid += 2
        done += 1
        bar.update(1)
    bar.close()

    out = resolve(cfg["paths"]["labels_dir"]) / "e2_claims.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "pair_id", "source_bullet", "claim", "label", "type", "method", "drafted_by",
                    "set", "reviewed"])
        for r in rows:   # drafted_by: the model that wrote an llm_draft row; set: standard (hard = make_hard_claims.py)
            w.writerow(r[:7] + ["scripted" if r[6] == "scripted" else getattr(llm, "model", "llm"), "standard", r[7]])
    n_draft = sum(r[6] == "llm_draft" for r in rows)
    print(f"wrote {len(rows)} claims to {out}; {n_draft} LLM drafts need review (set reviewed=yes)")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--n", type=int, default=150, help="number of source bullets")
    a = p.parse_args()
    main(a.config, a.n, a.mock)
