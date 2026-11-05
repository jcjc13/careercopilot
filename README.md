# CareerCopilot: faithful resume tailoring (IS469)

Can an LLM make a resume more relevant to a job without inventing experience, and does that
depend on how well the candidate fits the job? The pipeline retrieves resume evidence for each
job requirement (RAG), rewrites the resume from it, and verifies every claim.

```
resume + job ─► anonymise + split into bullets ─► extract 8 requirements ─► retrieve evidence (E1)
                                                                                │
tailored resume ◄── verify claims (E2) ◄── split into claims ◄── tailor (E3: A/B/C/D)
                        └── unsupported: C regenerates (budget 2) then drops; D uses a bounded controller ──┘
```

| Variant | Grounded in retrieved evidence | Verify + regenerate |
|---|---|---|
| A (baseline) | no | no |
| B | yes | no |
| C | no | yes |
| D (full) | yes | yes, with a controller (run v3; the earlier fixed loop is run v2) |

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
streamlit run app.py                                             # two tabs: tailor (variant D, hybrid retrieval, fine-tuned NLI verifier) and experiment results

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
python scripts/run_e3_tailoring.py --variants A,B,C,D           # D uses the controller (tailoring.controller in the config; mode: llm|rule)
python scripts/evaluate_e3.py --pairwise                         # then label the two sheets it writes
python scripts/evaluate_e3.py --human-labels data/labels/e3_human_sample.csv \
                              --human-preference data/labels/e3_preference.csv
python scripts/agent_metrics.py                                  # per-task success, steps, cost, latency (C loop; D controller metrics: recovery success, escalation rate, actions by fit)
python scripts/analyse_e3_failures.py                            # qualitative failure categories -> runs/<run>/e3_failures.csv
python scripts/make_e3_figure.py                                 # main exhibit -> runs/<run>/figures/e3_tradeoff.svg

# responsible-AI probes
python scripts/run_probes.py --probe injection --variants A,B,C,D
python scripts/run_probes.py --probe bias --variants A,D

# evidence sufficiency: E1-labelled requirements under intact / partially removed /
# removed labelled-support conditions; use the same pairs across variants
python scripts/run_evidence_sufficiency.py --variants A,B,C,D
```

Every script accepts `--mock` (offline fake LLM, outputs not meaningful) and `--limit N`.
Run scripts resume where they stopped; outputs go to `runs/<run_name>/`.
The evidence-sufficiency probe also accepts `--verifier METHOD` to override the configured
verifier; record that choice when comparing results, since it changes the system under test.
Completed requirement rows are checkpointed to CSV; `--resume` skips rows already saved after
an interrupted or rate-limited run.
The evidence-sufficiency probe requires manually labelled `gold_bullet_ids` in
`data/labels/e1_labels.csv`. It records verifier-based unsupported-claim rate, coverage of the
target requirement by supported output, and abstention/removal/review behavior. “Removed” means
the E1-labelled supporting bullets were removed from the resume; it does not prove that the
remaining resume contains no equivalent evidence. Treat these paired stress-test results as
pilot evidence, not an independent generalization test.

### Training on Colab
Upload the repo (or `git clone` it), then in a GPU runtime:
```
!pip -q install -r requirements.txt
!python scripts/train_verifier.py
```
Download `models/verifier/` (and `models/verifier-nli-ft/` for the NLI model) and keep them next to the repo;
`verifier.finetuned_path` and `verifier.finetuned_nli_path` point to them. `models/` is gitignored.
`notebooks/BERT_Verifier.ipynb` and `notebooks/NLI_Verifier(fine_tuned).ipynb` are the Colab runs of these two training commands.

## Models

Defaults: Groq's `openai/gpt-oss-20b` inside the pipeline and the stronger
`openai/gpt-oss-120b` as evaluation judge (a different, stronger model, so variant D is not graded
by its own checker). The hard E2 test set is drafted by `drafter` (`qwen/qwen3.8-27b` on Groq), a different
model family from the one that wrote the training data. Any OpenAI-compatible endpoint works: set `base_url`, `api_key_env` and
`model` per role in `configs/default.yaml` -- e.g. a local `qwen3:4b` through Ollama.

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

## Controller (variant D, run v3)

`src/careercopilot/controller.py` replaces D's fixed regenerate-then-drop loop. For each line the verifier
rejects, the model proposes a JSON decision (ACCEPT, REWRITE_CONSERVATIVELY, RETRIEVE_MORE, USE_ORIGINAL, DROP,
HUMAN_REVIEW); Python enforces allowed actions, per-line limits (`tailoring.controller` in `configs/default.yaml`),
acceptance only when the verifier supports every claim, HUMAN_REVIEW only near the verifier threshold, and a
rule-based router as override/fallback. The controller was tuned on 9 development pairs (`configs/dev.yaml`,
`data/dev/`, runs `v3dev*`), disjoint from the 36 test pairs. Run v3 reuses A to C from run v2 and re-runs only D.
Result: no better than the fixed loop on faithfulness or coverage; the gain is bounded, logged recovery
(see `CareerCopilot — Process Record` 3.16). The old D outputs stay in `runs/v2/`.

## Dataset

[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
on Hugging Face, labelled Good Fit / Potential Fit / No Fit. `prepare_data.py` detects the column
names and prints them; check them, the licence, and a few split resumes before use.
