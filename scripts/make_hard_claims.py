"""E2 HARD test set (~80 claims): subtler fabrications and harder faithful rewrites than the standard
set, with the LLM-written rows drafted by a Qwen model (cfg `drafter`), a different family from the
gpt-oss model that wrote the standard set and the silver training set.

    python scripts/make_hard_claims.py --preview          # 2 rows per type, nothing written
    python scripts/make_hard_claims.py                    # 8 per fabricated type, 10 per faithful type
  -> appends to data/labels/e2_claims.csv with set=hard

Fabricated: number_small (15% -> 18%), tool_near (MySQL -> PostgreSQL), title_other (a non-"Senior"
title change), scope_inflation (team -> company, supported -> led), implied_claim (used Python ->
built ML models in Python).
Faithful: number_reformat (1+ years -> over 1 year), synonym_verb, merged (two bullets of one resume,
no new facts), reordered.
Scripted edits are preferred (method=scripted, reviewed=yes); Qwen rows are method=llm_draft,
reviewed=no and MUST be checked by a person before run_e2_verifiers uses them.
Source bullets come from the resumes of the existing E2 claims and are not reused from them.
"""
import csv
import random
import re

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, tqdm

from careercopilot import prompts
from careercopilot.corrupt import (change_quantity_small, clean_llm, is_junk, reformat_quantity,
                                   reorder_clause, scope_inflate, swap_tool_near, swap_verb)

FAB_TYPES = ["number_small", "tool_near", "title_other", "scope_inflation", "implied_claim"]
FAITH_TYPES = ["number_reformat", "synonym_verb", "merged", "reordered"]
COLUMNS = ["claim_id", "pair_id", "source_bullet", "claim", "label", "type", "method", "drafted_by", "set", "reviewed"]
TITLE_CUE = re.compile(r"^(?:\S+\s+){0,7}?\S*\b(?:Analyst|Engineer|Developer|Manager|Architect|Consultant|Administrator|Tester|"
                       r"Designer|Programmer|Specialist|Officer|Associate|Coordinator|Representative|Executive)\b")
BANNED = re.compile(r"\b(?:Senior|Junior|Principal)\b")

QWEN = {
    "title_other": ("fabricate", "change the candidate's own job title to a different, plausible title that is NOT "
                    "a seniority change (for example Analyst -> Engineer, Developer -> Architect, Associate -> "
                    "Manager). Do not use the words Senior, Junior or Principal"),
    "scope_inflation": ("fabricate", "inflate the scope or ownership of the work (for example a team becomes the whole "
                        "company, 'supported' becomes 'led', one project becomes every project)"),
    "implied_claim": ("fabricate", "add a plausible-sounding claim that the bullet does NOT state or imply and that "
                      "goes beyond it (for example 'used Python' -> 'built machine-learning models in Python'). "
                      "Add no new tool name and no number"),
    "synonym_verb": ("paraphrase", "replace the opening verb and one other verb with synonyms; keep every tool, "
                     "number, duty, title and date identical"),
    "reordered": ("paraphrase", "reorder the clauses or phrases of the bullet; keep every word's meaning and every "
                  "tool, number, duty, title and date identical, and add nothing"),
}
SYSTEM = ("Task: {task}.\nFor a research test set on detecting fabricated resume claims. Rewrite the bullet so that "
          "you {rule}. Change nothing else and keep it fluent. {data_rule}\nReply with the rewritten bullet only.")
MERGE = ("Task: paraphrase.\nMerge the two resume bullets into ONE fluent bullet. Use only facts stated in them: no "
         "new tool, number, duty, title or date, and drop nothing important. {data_rule}\n"
         "Reply with the merged bullet only.")


def qwen(llm, system, user):
    out, _ = llm.chat(system, user, max_tokens=300)
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)   # defensive: reasoning is off, but strip if present
    return re.sub(r"</?think>", "", out).strip()


def main(config, n_fab=8, n_faith=10, preview=False, force=False, mock=False):
    cfg = load_config(config)
    rng = random.Random(cfg["seed"] + 1)
    labels = resolve(cfg["paths"]["labels_dir"]) / "e2_claims.csv"
    existing = list(csv.DictReader(open(labels, encoding="utf-8")))
    if any(r.get("set") == "hard" for r in existing) and not (preview or force):
        raise SystemExit("e2_claims.csv already has set=hard rows (they may be reviewed); pass --force to add more")
    used = {r["source_bullet"] for r in existing}
    next_id = max(int(r["claim_id"][1:]) for r in existing) + 1
    resumes = {r["pair_id"] for r in existing}
    llm = build_llm(cfg, "drafter", mock=mock)
    model = getattr(llm, "model", "mock")

    # (pair, index in pair.bullets, bullet) from the resumes of the existing claims. Bullets that are whole
    # sentences only (no template placeholders, no truncated 300-char cuts). Bullets not yet used by the
    # standard set come first; the rest are reused (only 34 bullets in these 33 resumes hold a quantity).
    pool, seen = [], set()
    for p in load_pairs_or_exit(cfg):
        if p.pair_id not in resumes:
            continue
        for i, b in enumerate(p.bullets):
            if (b not in seen and len(b.split()) >= 6 and len(b) < 280 and b[0].isupper() and "[" not in b
                    and not is_junk(b)):
                seen.add(b)
                pool.append((p, i, b))
    rng.shuffle(pool)
    pool.sort(key=lambda x: x[2] in used)   # stable: unused first
    taken = {"fabricated": set(), "faithful": set()}   # a bullet may serve once as a fabrication and once as a rewrite

    scripted = {
        "number_small": lambda p, b: change_quantity_small(b, rng),
        "tool_near": lambda p, b: swap_tool_near(b, p.resume),
        "scope_inflation": lambda p, b: scope_inflate(b),
        "number_reformat": lambda p, b: reformat_quantity(b),
        "synonym_verb": lambda p, b: swap_verb(b),
        "reordered": lambda p, b: reorder_clause(b),
    }

    def pick(kind, label):
        """(pool index, scripted claim or None). Scripted types take a bullet where the edit applies and
        fall back to a Qwen draft only when none is left; title_other needs a title word in the bullet."""
        free = [i for i in range(len(pool)) if i not in taken[label]]
        for i in free:
            p, _, b = pool[i]
            if kind == "title_other":
                if TITLE_CUE.search(b) and not BANNED.search(b) and "/" not in b and b.count(",") <= 1:
                    return i, None
            elif kind in scripted:
                out = scripted[kind](p, b)
                if out and out != b:
                    return i, out
            elif kind == "merged":
                nxt = next((j for j in free if j != i and pool[j][0] is p and abs(pool[j][1] - pool[i][1]) <= 3), None)
                if nxt is not None:
                    return i, nxt
            else:   # implied_claim: any bullet
                return i, None
        return (free[0], None) if free and kind in QWEN else (None, None)

    rows, cid = [], next_id
    plan = [(k, "fabricated", n_fab) for k in FAB_TYPES] + [(k, "faithful", n_faith) for k in FAITH_TYPES]
    for kind, label, n in plan:
        made, tries = 0, 0
        while made < n and tries < 4 * n:
            tries += 1
            i, extra = pick(kind, label)
            if i is None:
                print(f"ran out of usable bullets for {kind} after {made}")
                break
            taken[label].add(i)
            p, _, b = pool[i]
            method, drafted_by, src = "scripted", "scripted", b
            if kind == "merged":
                j = extra
                taken[label].add(j)
                second = pool[j][2]
                src = f"{b} || {second}"
                claim = clean_llm(qwen(llm, MERGE.format(data_rule=prompts.DATA_RULE),
                                       f"<bullet>\n{b}\n</bullet>\n<bullet>\n{second}\n</bullet>"), "x" * 400)
                method, drafted_by = "llm_draft", model
            elif extra is not None:
                claim = extra
            else:
                task, rule = QWEN[kind]
                claim = clean_llm(qwen(llm, SYSTEM.format(task=task, rule=rule, data_rule=prompts.DATA_RULE),
                                       f"<bullet>\n{b}\n</bullet>"), b)
                method, drafted_by = "llm_draft", model
                if kind == "title_other" and BANNED.search(claim):
                    claim = ""
            if not claim or claim == b:
                continue
            rows.append([f"c{cid}", p.pair_id, src, claim, label, kind, method, drafted_by, "hard",
                         "yes" if method == "scripted" else "no"])
            cid += 1
            made += 1
            if preview and made >= 2:
                break

    if preview:
        for r in rows:
            print(f"\n[{r[5]} | {r[4]} | {r[6]}] {r[0]}\n  source: {r[2][:240]}\n  claim : {r[3][:240]}")
        print(f"\npreview only: {len(rows)} rows, nothing written (drafter: {model})")
        return rows
    with open(labels, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows([[r.get(c, "") for c in COLUMNS] for r in existing])
        w.writerows(rows)
    nq = sum(r[6] == "llm_draft" for r in rows)
    print(f"appended {len(rows)} hard claims to {labels}; {nq} Qwen rows need review (set reviewed=yes)")
    return rows


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--n-fab", type=int, default=8, help="rows per fabricated type")
    p.add_argument("--n-faith", type=int, default=10, help="rows per faithful type")
    p.add_argument("--preview", action="store_true", help="2 rows per type, print only, write nothing")
    p.add_argument("--force", action="store_true", help="add rows even though set=hard rows already exist")
    a = p.parse_args()
    main(a.config, a.n_fab, a.n_faith, a.preview, a.force, a.mock)
