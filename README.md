# CareerCopilot: faithful resume tailoring (IS469)

Can an LLM make a resume more relevant to a job without inventing experience, and does that
depend on how well the candidate fits the job? The pipeline retrieves resume evidence for each
job requirement (RAG), rewrites the resume from it, and verifies every claim.

```
resume + job ─► anonymise + split into bullets ─► extract 8 requirements ─► retrieve evidence (E1)
                                                                                │
tailored resume ◄── verify claims (E2) ◄── split into claims ◄── tailor (E3: A/B/C/D)
                        └── unsupported: regenerate (step budget 2), then drop ──┘
```

| Variant | Grounded in retrieved evidence | Verify + regenerate |
|---|---|---|
| A (baseline) | no | no |
| B | yes | no |
| C | no | yes |
| D (full) | yes | yes |

## Where each IS469 topic is used

| Topic | In this project |
|---|---|
| 1 Text processing | TF-IDF retriever baseline (`retrieval.py`), tokenising |
| 2 Transformers / BERT | sentence-embedding retriever, cross-encoders (NLI verifier, re-ranker) |
| 3 Training | fine-tuned BERT verifier with the lab's Trainer recipe (`train_verifier.py`) |
| 4 Efficient fine-tuning | distillation: a teacher LLM writes part of the training data; LoRA as future work |
| 5 Inference | temperature 0 for reproducibility; reasoning-effort dial; optional local model via Ollama |
| 6 Evaluation | programmatic checks → LLM judge → humans; judge validated (kappa); pairwise judge in both orders |
| 7 Agents | verify-and-regenerate loop with a step budget; trajectory counts (regenerations, removals) |
| 8 RAG | BM25, hybrid + cross-encoder re-ranking, contextual retrieval, grounded generation with citations |
| 9 Ethics | prompt-injection and name-swap bias probes; PII stripping; fabrication as the core risk |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # add your GROQ_API_KEY
python tests/test_smoke.py                               # offline check, seconds
```

## Run order

```bash
# 0. data: 12 pairs per fit level from Hugging Face (or --sample for 3 offline pairs)
python scripts/prepare_data.py

# try the system on one pair
python scripts/tailor_one.py --pair <pair_id> --variant D
streamlit run app.py                                             # three tabs: tailor (live), saved E3 results, experiment tables

# E1 retrieval (Member A)
python scripts/run_e1_retrieval.py --make-template --pairs 12    # then label data/labels/e1_labels.csv
python scripts/run_e1_retrieval.py                               # tfidf, bm25, embedding, hybrid, +ctx

# E2 verifiers (Member B)
python scripts/make_fabrications.py --n 150                      # standard test set; then review data/labels/e2_claims.csv
python scripts/make_date_fabrications.py                         # +28 date/title rows (set=standard), review and append
python scripts/make_hard_claims.py --preview                     # hard test set (set=hard); drafted by the `drafter` model
python scripts/make_hard_claims.py                               #   then review the llm_draft rows, set reviewed=yes
python scripts/make_verifier_train.py                            # silver training set, other resumes
python scripts/train_verifier.py                                 # GPU / Colab, a few minutes (BERT)
python scripts/train_verifier.py --base cross-encoder/nli-deberta-v3-small --out models/verifier-nli-ft   # NLI
python scripts/run_e2_verifiers.py                               # rule, nli, finetuned, finetuned_nli, llm; both sets
                                                                 # (CPU: ~2 h; run one method at a time with --methods X --merge)

# E3 tailoring (both) -- first set retrieval.method / verifier.method to the E1/E2 winners
python scripts/run_e3_tailoring.py --variants A,B,C,D
python scripts/evaluate_e3.py --pairwise                         # then label the two sheets it writes
python scripts/evaluate_e3.py --human-labels data/labels/e3_human_sample.csv \
                              --human-preference data/labels/e3_preference.csv
python scripts/agent_metrics.py                                  # per-task success, steps, cost, latency (variants C/D loop)
python scripts/analyse_e3_failures.py                            # qualitative failure categories -> runs/<run>/e3_failures.csv
python scripts/make_e3_figure.py                                 # main exhibit -> runs/<run>/figures/e3_tradeoff.svg

# Topic 9 probes
python scripts/run_probes.py --probe injection --variants A,B,C,D
python scripts/run_probes.py --probe bias --variants A,D
```

Every script accepts `--mock` (offline fake LLM, outputs not meaningful) and `--limit N`.
Run scripts resume where they stopped; outputs go to `runs/<run_name>/`.

### Training on Colab
Upload the repo (or `git clone` it), then in a GPU runtime:
```
!pip -q install -r requirements.txt
!python scripts/train_verifier.py
```
Download `models/verifier/` (and `models/verifier-nli-ft/` for the NLI model) and keep them next to the repo;
`verifier.finetuned_path` and `verifier.finetuned_nli_path` point to them. `models/` is gitignored.

## Models

Defaults follow the class labs: Groq's `openai/gpt-oss-20b` inside the pipeline and the stronger
`openai/gpt-oss-120b` as evaluation judge (a different, stronger model, so variant D is not graded
by its own checker). The hard E2 test set is drafted by `drafter` (`qwen/qwen3.8-27b` on Groq), a different
model family from the one that wrote the training data. Any OpenAI-compatible endpoint works: set `base_url`, `api_key_env` and
`model` per role in `configs/default.yaml` -- e.g. a local `qwen3:4b` through Ollama.

## Layout

| Path | What | Owner |
|---|---|---|
| `src/careercopilot/text.py` | anonymisation, bullet + context splitting, tokenising | A |
| `src/careercopilot/retrieval.py` | TF-IDF, BM25, embedding, hybrid + re-ranker | A |
| `src/careercopilot/generation.py`, `prompts.py` | requirements, tailoring, regeneration, claim splitting | A (prompts reviewed by both) |
| `src/careercopilot/pipeline.py` | variants A-D | A |
| `src/careercopilot/verifiers.py`, `corrupt.py` | rule, NLI, fine-tuned, LLM verifiers; corruptions | B |
| `src/careercopilot/metrics.py` | all metrics and tests | B |
| `scripts/run_e1_*`, `tailor_one`, `run_probes` | E1, probes | A |
| `app.py` | demo app (built by B on the `app` branch) | B |
| `scripts/make_fabrications`, `make_date_fabrications`, `make_hard_claims`, `make_verifier_train`, `train_verifier`, `run_e2_*`, `evaluate_e3`, `agent_metrics`, `analyse_e3_failures`, `make_e3_figure`, `export_session_log` | E2, E3 scoring, process records | B |
| `configs/default.yaml` | every setting + seed | both |
| `data/labels/` | your hand labels (commit these) | both |

## Reproducibility and fairness rules

- One config, one seed (`469`). Changing `llm`, `retrieval`, `verifier` or `tailoring` → bump `run_name`.
- All variants use the same model, temperature 0, the same requirement list per pair, and the same
  "do not invent experience" instruction; only grounding and verification differ.
- Thresholds (NLI, fine-tuned BERT, fine-tuned NLI) are tuned on the silver validation set only
  (`data/verifier/val.jsonl`, resumes not in the training split or in any test pair); every claim of the
  standard and hard test sets is then scored. `run_e2_verifiers.py --tune dev` is the earlier protocol (a 20%
  dev split of the standard claims). The fine-tuned verifiers never see a test resume (their training
  resumes are sampled separately).
- E2 results are reported per set (standard / hard) with F1 at the tuned threshold and at 0.5, ROC-AUC,
  false-flag rates and McNemar tests. Run v2 is in `runs/v2/` (console logs and environment in `runs/v2/logs/`).
- All user text is wrapped in tags and every prompt says it is data, not instructions.

## Dataset

[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
on Hugging Face, labelled Good Fit / Potential Fit / No Fit. `prepare_data.py` detects the column
names and prints them; check them, the licence, and a few split resumes before use.
