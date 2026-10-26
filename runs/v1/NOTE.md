# Run v1 (superseded by v2)

Kept for comparison only. v1 used the earlier E2 protocol, which v2 replaced:
- test set: 320 claims (standard set only); no hard set;
- thresholds tuned on a 20% dev split of those claims (64), metrics reported on the other 256;
- results file is flat (`{method: metrics}`), without per-set results, F1 at 0.5, ROC-AUC, per-type false-flag
  rates or McNemar tests; zero-shot NLI, fine-tuned BERT, fine-tuned NLI, rule and LLM judge were compared.

Run v2 (`runs/v2/`) evaluates the standard (320) and hard (80) sets, tunes thresholds on the silver validation
set (`data/verifier/val.jsonl`) and writes `{set: {method: metrics}, "mcnemar": ...}`. See
`~/Documents/IS469/CareerCopilot — E2 Error Analysis (v2).md` for the v2 analysis.
