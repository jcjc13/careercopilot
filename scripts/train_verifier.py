"""Fine-tune a BERT-family cross-encoder as the claim verifier -- the Topic 3 lab recipe
(AutoModelForSequenceClassification + Trainer + DataCollatorWithPadding, evaluate per epoch on a
held-out validation set, then inspect predictions qualitatively).

Input pairs are (resume bullet, claim) -> supported (1) / unsupported (0), from
scripts/make_verifier_train.py. Run on a GPU (Colab is fine; a few minutes for bert-base):

    python scripts/train_verifier.py
    python scripts/train_verifier.py --base cross-encoder/nli-deberta-v3-small --out models/verifier-nli-ft
  -> models/verifier/ (model + tokenizer + metrics.json); point verifier.finetuned_path at it

Compare in E2: the zero-shot NLI model vs this fine-tuned model = base vs fine-tuned.
"""
import json
import time

import numpy as np
from _common import load_config, parser, resolve


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def metrics_from(logits, labels):
    pred = np.argmax(logits, axis=-1)
    acc = float((pred == labels).mean())
    # report F1 for catching FABRICATED claims (label 0), the class the project cares about
    tp = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 0) & (labels == 1)).sum())
    fn = int(((pred == 1) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "fab_precision": prec, "fab_recall": rec, "f1": f1}


def main(config, base=None, out=None, epochs=None):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    cfg = load_config(config)
    ft = cfg["finetune"]
    base = base or ft["base_model"]
    out = resolve(out or cfg["verifier"]["finetuned_path"])
    set_seed(cfg["seed"])

    d = resolve(cfg["paths"]["verifier_data"])
    train_rows, val_rows = load_rows(d / "train.jsonl"), load_rows(d / "val.jsonl")
    print(f"train {len(train_rows)} / val {len(val_rows)} examples; base model {base}")

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, id2label={0: "unsupported", 1: "supported"},
        label2id={"unsupported": 0, "supported": 1},
        ignore_mismatched_sizes=True)   # an NLI checkpoint has a 3-way head: replace it
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f}M (full fine-tuning; LoRA is a future-work option)")

    def tokenize(batch):  # pair input: [CLS] premise [SEP] hypothesis [SEP]; no padding, the collator pads
        return tok(batch["premise"], batch["hypothesis"], truncation=True, max_length=ft["max_length"])

    cols = ["premise", "hypothesis", "label"]
    train_ds = Dataset.from_list([{k: r[k] for k in cols} for r in train_rows]).map(tokenize, batched=True)
    val_ds = Dataset.from_list([{k: r[k] for k in cols} for r in val_rows]).map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=str(resolve(cfg["paths"]["models_dir"]) / "checkpoints"),
        report_to="none",
        num_train_epochs=epochs or ft["epochs"],
        learning_rate=ft["learning_rate"],
        per_device_train_batch_size=ft["batch_size"],
        per_device_eval_batch_size=ft["batch_size"],
        weight_decay=ft["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        seed=cfg["seed"],
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=lambda p: metrics_from(p.predictions, p.label_ids),
                      data_collator=DataCollatorWithPadding(tokenizer=tok))
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    val = trainer.evaluate()
    print("validation:", {k: round(v, 3) for k, v in val.items() if isinstance(v, float)})

    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    json.dump({"base_model": base, "train_examples": len(train_rows), "val_examples": len(val_rows),
               "train_seconds": round(train_s, 1), "params_millions": round(n_params / 1e6, 1),
               "val": val, "log_history": trainer.state.log_history},
              open(out / "metrics.json", "w"), indent=2)
    print(f"saved to {out} ({train_s:.0f}s training)")

    # qualitative inspection, as in the lab: a few validation pairs, predicted vs true
    preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    names = {0: "UNSUPPORTED", 1: "SUPPORTED"}
    for i in range(min(8, len(val_rows))):
        r = val_rows[i]
        mark = "OK " if preds[i] == r["label"] else "XX "
        print(f"{mark} PRED {names[int(preds[i])]:<11} TRUE {names[r['label']]:<11} ({r['kind']})")
        print(f"    bullet: {r['premise'][:110]}\n    claim : {r['hypothesis'][:110]}")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--base", default=None, help="override finetune.base_model")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args()
    main(a.config, a.base, a.out, a.epochs)
