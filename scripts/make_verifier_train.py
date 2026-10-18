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
  -> data/verifier/train.jsonl, data/verifier/val.jsonl  (split by resume, 85/15)
"""
import json
import random
from collections import Counter

from _common import build_llm, load_config, load_pairs_or_exit, parser, resolve, tqdm

from careercopilot import prompts
from careercopilot.corrupt import add_seniority, change_number, sub_claim, swap_tool
from careercopilot.data import sample_other_resumes
from careercopilot.retrieval import BM25Retriever
from careercopilot.text import load_vocab, split_bullets


def main(config, teacher_frac=0.3, per_resume=8, mock=False, sample=False, n_resumes=None):
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
    bullets_by = {rid: [b for b in split_bullets(t) if len(b.split()) >= 6] for rid, t in resumes}
    all_bullets = [(rid, b) for rid, bs in bullets_by.items() for b in bs]
    bm25 = BM25Retriever()

    rows = []
    for rid, text in tqdm(resumes, desc="resumes"):
        bs = bullets_by[rid]
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
                ex.append((b, para.strip(), 1, "paraphrase_llm"))
                kind = rng.choice(["duty", "date_title"])
                sys = prompts.FABRICATE.format(kind=prompts.FABRICATION_KINDS[kind], data_rule=prompts.DATA_RULE)
                fab, _ = llm.chat(sys, f"<bullet>\n{b}\n</bullet>", max_tokens=200)
                ex.append((b, fab.strip(), 0, f"{kind}_llm"))
            # unrelated claim from another resume, against this resume's closest bullet
            other = [ob for orid, ob in all_bullets if orid != rid]
            if other:
                claim = rng.choice(other)
                best = bm25.rank(claim, bs, 1)[0][0]
                ex.append((bs[best], claim, 0, "other_resume"))
            for premise, hyp, label, kind in ex:
                if hyp:
                    rows.append({"resume_id": rid, "premise": premise, "hypothesis": hyp, "label": label, "kind": kind})

    # balance classes by downsampling the larger one
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n = min(len(pos), len(neg))
    rows = pos[:n] + neg[:n]
    rng.shuffle(rows)

    ids = sorted({r["resume_id"] for r in rows})
    rng.shuffle(ids)
    val_ids = set(ids[: max(1, int(0.15 * len(ids)))])
    out = resolve(cfg["paths"]["verifier_data"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as ft, open(out / "val.jsonl", "w") as fv:
        for r in rows:
            (fv if r["resume_id"] in val_ids else ft).write(json.dumps(r) + "\n")
    n_val = sum(r["resume_id"] in val_ids for r in rows)
    print(f"wrote {len(rows) - n_val} train / {n_val} val examples to {out}")
    print("kinds:", dict(Counter(r["kind"] for r in rows)))


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--teacher-frac", type=float, default=0.3, help="share of bullets that also get LLM-written examples")
    p.add_argument("--per-resume", type=int, default=8)
    p.add_argument("--n-resumes", type=int, default=None)
    p.add_argument("--sample", action="store_true", help="offline demo: use the sample resumes")
    a = p.parse_args()
    main(a.config, a.teacher_frac, a.per_resume, a.mock, a.sample, a.n_resumes)
