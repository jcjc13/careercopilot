"""Build the SILVER training set for the fine-tuned verifier (E2), from resumes that are NOT
in the test pairs (so the verifier never trains on a resume it is later tested on).

Each example is (premise = a real resume bullet, hypothesis = a claim, label):
  supported (1):   a faithful sub-claim of the bullet (scripted), or a teacher-LLM paraphrase
  unsupported (0): a swapped tool / inflated number / added seniority (scripted), a teacher-LLM
                   invented duty or changed title/date, or a claim from ANOTHER resume paired
                   with this resume's closest bullet (BM25)
The teacher-LLM part is distillation (Topic 4): a model writes training data, no human labels.
Only the E2 TEST set (make_fabrications.py) is human-reviewed.

    python scripts/make_verifier_train.py                     # 300 other resumes from the dataset
    python scripts/make_verifier_train.py --teacher-frac 0    # scripted only, no LLM cost
    python scripts/make_verifier_train.py --resume            # continue an interrupted run
  -> data/verifier/train.jsonl, data/verifier/val.jsonl  (split by resume, 85/15)

Rows are appended to data/verifier/raw.jsonl as each resume finishes, so a crash (e.g. a Groq rate
limit) loses at most one resume; --resume skips the ones already done. Without --resume the file
is started afresh. Each resume has its own RNG seeded from (seed, resume id), so a resumed run
makes the same examples as an uninterrupted one.
"""
import json
import random
import re
from collections import Counter

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, tqdm

from careercopilot import prompts
from careercopilot.corrupt import add_seniority, change_number, sub_claim, swap_tool
from careercopilot.data import sample_other_resumes
from careercopilot.retrieval import BM25Retriever
from careercopilot.text import load_vocab, split_bullets

OTHER_RESUME_SHARE = 0.25   # cap on easy "unrelated claim" negatives, as a share of all negatives
RUN_TOGETHER = re.compile(r"[a-z]{3}[A-Z][a-z]")   # "kitsMaintained": headings/fields glued together
DATE_FIELD = re.compile(r"\d{2}/\d{4}|\d{4}\s*(?:to|-)\s*(?:\d{2,4}|present)|\bExpected in\b", re.I)


def is_junk(bullet: str) -> bool:
    """Skill lists, education/date lines and run-together fields make meaningless claims."""
    words = bullet.split()
    if bullet.count(",") > 0.2 * len(words) and bullet.count(",") >= 4:
        return True
    if DATE_FIELD.search(bullet):
        return True
    return any(len(w) >= 12 and RUN_TOGETHER.search(w) for w in words)


def clean_llm(text: str, premise: str) -> str:
    """Strip markdown the teacher sometimes adds (a model must not learn '**' means fabricated)
    and reject outputs that are empty, unchanged, or much longer than the bullet."""
    text = re.sub(r"[*`]+", "", text or "")
    text = re.sub(r"^\s*(?:[-•]\s+)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text == premise or len(text) > 2 * len(premise) + 50:
        return ""
    return text


def examples_for(rid, text, bs, all_bullets, bm25, vocab, llm, teacher_frac, per_resume, seed):
    rng = random.Random(f"{seed}-{rid}")
    rows = []
    for b in rng.sample(bs, min(per_resume, len(bs))):
        ex = []
        sc = sub_claim(b)
        if sc:
            ex.append((b, sc, 1, "sub_claim"))
        for kind, fab in (("tool", swap_tool(b, text, vocab, rng)), ("number", change_number(b, rng)),
                          ("seniority", add_seniority(b, rng))):
            if fab and fab != b:
                ex.append((b, fab, 0, kind))
        if llm is not None and rng.random() < teacher_frac:
            para, _ = llm.chat(prompts.PARAPHRASE, f"<bullet>\n{b}\n</bullet>", max_tokens=200)
            kind = rng.choice(["duty", "date_title"])
            sys = prompts.FABRICATE.format(kind=prompts.FABRICATION_KINDS[kind], data_rule=prompts.DATA_RULE)
            fab, _ = llm.chat(sys, f"<bullet>\n{b}\n</bullet>", max_tokens=200)
            para, fab = clean_llm(para, b), clean_llm(fab, b)
            if para:
                ex.append((b, para, 1, "paraphrase_llm"))
            if fab:
                ex.append((b, fab, 0, f"{kind}_llm"))
        # unrelated claim from another resume, against this resume's closest bullet
        other = [ob for orid, ob in all_bullets if orid != rid]
        if other:
            claim = rng.choice(other)
            best = bm25.rank(claim, bs, 1)[0][0]
            ex.append((bs[best], claim, 0, "other_resume"))
        for premise, hyp, label, kind in ex:
            if hyp:
                rows.append({"resume_id": rid, "premise": premise, "hypothesis": hyp, "label": label, "kind": kind})
    return rows


def main(config, teacher_frac=0.5, per_resume=8, mock=False, sample=False, n_resumes=None, resume=False):
    cfg = load_config(config)
    rng = random.Random(cfg["seed"])
    vocab = load_vocab(resolve(cfg["paths"]["skills_vocab"]))
    llm = build_llm(cfg, mock=mock) if teacher_frac > 0 else None
    test_pairs = load_pairs_or_exit(cfg)
    if sample:  # offline/demo only: reuses the sample resumes (would leak in a real run)
        resumes = [(p.pair_id, p.resume) for p in test_pairs]
    else:
        resumes = sample_other_resumes(cfg, {p.resume for p in test_pairs},
                                       n_resumes or cfg["data"]["verifier_train_resumes"])
    bullets_by = {rid: [b for b in split_bullets(t) if len(b.split()) >= 6 and not is_junk(b)]
                  for rid, t in resumes}
    all_bullets = [(rid, b) for rid, bs in bullets_by.items() for b in bs]
    bm25 = BM25Retriever()

    out = resolve(cfg["paths"]["verifier_data"])
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw.jsonl"
    done = set()
    if resume and raw_path.exists():
        done = {r["resume_id"] for r in map(json.loads, filter(str.strip, open(raw_path))) if r["kind"] == "_done"}
        # drop rows of a resume that crashed halfway, so it is redone cleanly
        keep = [l for l in open(raw_path) if l.strip() and json.loads(l)["resume_id"] in done]
        raw_path.write_text("".join(keep))
        print(f"resuming: {len(done)} resumes already in {raw_path}")
    else:
        raw_path.write_text("")
    ids = {rid for rid, _ in resumes}

    with open(raw_path, "a") as raw:
        for rid, text in tqdm(resumes, desc="resumes"):
            if rid in done or not bullets_by[rid]:
                continue
            for r in examples_for(rid, text, bullets_by[rid], all_bullets, bm25, vocab, llm,
                                  teacher_frac, per_resume, cfg["seed"]):
                raw.write(json.dumps(r) + "\n")
            raw.write(json.dumps({"resume_id": rid, "kind": "_done"}) + "\n")  # marks the resume complete
            raw.flush()
    all_rows = [json.loads(l) for l in open(raw_path) if l.strip()]
    finished = {r["resume_id"] for r in all_rows if r["kind"] == "_done"}
    rows = [r for r in all_rows if r["kind"] != "_done" and r["resume_id"] in ids and r["resume_id"] in finished]

    # balance classes: easy "other_resume" negatives are capped first, so the hard ones survive
    pos = [r for r in rows if r["label"] == 1]
    hard = [r for r in rows if r["label"] == 0 and r["kind"] != "other_resume"]
    easy = [r for r in rows if r["kind"] == "other_resume"]
    rng.shuffle(pos)
    rng.shuffle(hard)
    rng.shuffle(easy)
    neg = hard + easy[: int(len(hard) * OTHER_RESUME_SHARE / (1 - OTHER_RESUME_SHARE))]
    rng.shuffle(neg)
    n = min(len(pos), len(neg))
    rows = pos[:n] + neg[:n]
    rng.shuffle(rows)

    ids = sorted({r["resume_id"] for r in rows})
    rng.shuffle(ids)
    val_ids = set(ids[: max(1, int(0.15 * len(ids)))])
    with open(out / "train.jsonl", "w") as ft, open(out / "val.jsonl", "w") as fv:
        for r in rows:
            (fv if r["resume_id"] in val_ids else ft).write(json.dumps(r) + "\n")
    n_val = sum(r["resume_id"] in val_ids for r in rows)
    print(f"wrote {len(rows) - n_val} train / {n_val} val examples to {out}")
    print("kinds:", dict(Counter(r["kind"] for r in rows)))


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--teacher-frac", type=float, default=0.5, help="share of bullets that also get LLM-written examples")
    p.add_argument("--per-resume", type=int, default=8)
    p.add_argument("--n-resumes", type=int, default=None)
    p.add_argument("--sample", action="store_true", help="offline demo: use the sample resumes")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run (skip resumes already in raw.jsonl)")
    a = p.parse_args()
    main(a.config, a.teacher_frac, a.per_resume, a.mock, a.sample, a.n_resumes, a.resume)
