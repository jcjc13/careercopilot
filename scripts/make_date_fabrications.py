"""E2 test set, extra date/title rows. The main set's date_title rows are mostly "Senior" added to
a bullet; this adds pairs where a calendar year, a duration or a job title is really changed.

    python scripts/make_date_fabrications.py --list        # show the bullets it would use, no LLM calls
    python scripts/make_date_fabrications.py --n 14
  -> data/labels/e2_claims_dates.csv  (same columns as e2_claims.csv; ids continue after the main set)
Review it like the main set (every row is an llm_draft), then append its rows to e2_claims.csv.

Source bullets come from the same test pairs: a hand-picked list (CURATED) of unused bullets that
state the candidate's own title, year or duration. Software versions ("SQL Server 2005") are tool
changes and education/coursework lines are not claims, so they are left out.
"""
import csv
import re

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, tqdm

from careercopilot import prompts
from careercopilot.corrupt import clean_llm

# Bullets that really state the candidate's own title, year or duration, hand-picked from the unused
# test-pair bullets (a regex also matched verbs like "Lead various...", tools like "Test Manager", and
# other people's titles). Matched by prefix; the cue says which kind of change to ask for.
CURATED = [
    ("Efficient Cloud Engineer with years", "role"),
    ("Technical Lead of the infrastructure of major enterprise wide Testing systems", "role"),
    ("Technical Lead of the infrastructure of major enterprise wide data warehouse", "role"),
    ("Lead on creating project QA plans", "role"),
    ("Engaging Quality Control Analyst", "role"),
    ("Highly motivated Sales Associate", "role"),
    ("Collaborate as Software Architect", "role"),
    ("Lead Architect of in-house developed applications", "role"),
    ("Meticulous Senior Software Test Analyst", "role"),
    ("Module Lead (Task Assignment", "role"),
    ("Playing the dual role of both the Product Owner", "role"),
    ("Besides, I coordinated Imagine Cup 2009", "year"),
    ("Safety Champion of Women safety initiative", "year"),
    ("I am a certified professional as MCPD web developer", "tenure"),
]

KIND_BY_CUE = {
    "year": "a different calendar year (for example 2009 -> 2007) in place of the one mentioned",
    "tenure": "a different duration (for example 8 years -> 12 years) in place of the one mentioned",
    "role": ("a different job title in place of the one mentioned (for example Analyst -> Manager, "
             "Lead -> Director, Architect -> Engineer); the new title must not just add a seniority word"),
}
FABRICATE_DATES = (
    "Task: fabricate.\n"
    "For a research test set on detecting fabricated resume claims: rewrite the bullet so it "
    "stays fluent and plausible but contains exactly ONE unsupported change: {kind}. Keep everything "
    "else the same. Do not use the words Senior, Junior or Principal. {data_rule}\n"
    "Reply with the rewritten bullet only.")


def main(config, n=14, mock=False, list_only=False):
    cfg = load_config(config)
    labels = resolve(cfg["paths"]["labels_dir"])
    existing = list(csv.DictReader(open(labels / "e2_claims.csv", encoding="utf-8")))
    used = {r["source_bullet"] for r in existing}
    next_id = max(int(r["claim_id"][1:]) for r in existing) + 1

    pool, seen = [], set()
    for p in load_pairs_or_exit(cfg):
        for b in p.bullets:
            cue = next((c for pre, c in CURATED if b.startswith(pre)), None)
            if cue and b not in used and b not in seen:
                seen.add(b)
                pool.append((p, b, cue))
    pool = pool[:n]
    print(f"{len(pool)} bullets chosen; cues: " + ", ".join(f"{k} {sum(c == k for _, _, c in pool)}" for k in KIND_BY_CUE))
    if list_only:
        for p, b, cue in pool:
            print(f"  [{cue:6}] {b[:130]}")
        return

    llm = build_llm(cfg, mock=mock)
    rows, cid = [], next_id
    for p, b, cue in tqdm(pool, desc="date/title pairs"):
        para, _ = llm.chat(prompts.PARAPHRASE, f"<bullet>\n{b}\n</bullet>", max_tokens=200)
        sys = FABRICATE_DATES.format(kind=KIND_BY_CUE[cue], data_rule=prompts.DATA_RULE)
        fab, _ = llm.chat(sys, f"<bullet>\n{b}\n</bullet>", max_tokens=200)
        para, fab = clean_llm(para, b), clean_llm(fab, b)
        if not para or not fab or re.search(r"\b(?:Senior|Junior|Principal)\b", fab) and not re.search(r"\b(?:Senior|Junior|Principal)\b", b):
            print(f"skipped (unusable draft): {b[:60]}")
            continue
        rows.append([f"c{cid}", p.pair_id, b, para, "faithful", "paraphrase", "llm_draft", "no"])
        rows.append([f"c{cid + 1}", p.pair_id, b, fab, "fabricated", "date_title", "llm_draft", "no"])
        cid += 2

    out = labels / "e2_claims_dates.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "pair_id", "source_bullet", "claim", "label", "type", "method", "reviewed"])
        w.writerows(rows)
    print(f"wrote {len(rows)} claims ({len(rows) // 2} pairs) to {out}; review every row, set reviewed=yes")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--n", type=int, default=14, help="number of source bullets")
    p.add_argument("--list", action="store_true", help="only list the chosen bullets")
    a = p.parse_args()
    main(a.config, a.n, a.mock, a.list)
