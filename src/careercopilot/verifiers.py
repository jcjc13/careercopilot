"""Claim verifiers (E2). Each checks ONE claim against the original resume's bullets and
returns (supported, score, usage). score is a confidence in [0, 1] that the claim is supported.

- RuleVerifier  every tool/technology, number and seniority word in the claim must appear
                in the resume. Free and fast; blind to invented duties phrased in new words.
- NLIVerifier   a zero-shot cross-encoder NLI model; supported if max P(entailment) over the
                resume's bullets >= threshold (tune the threshold on the E2 dev split).
- FinetunedVerifier  a BERT-family cross-encoder fine-tuned on our silver data (Topic 3).
- LLMVerifier   asks an LLM SUPPORTED / UNSUPPORTED.
This is the Topic 6 ladder in one file: programmatic check -> trained classifier -> LLM judge.
"""
from __future__ import annotations

import time

from . import prompts
from .schema import Usage
from .text import find_numbers, find_terms, load_vocab


class RuleVerifier:
    name = "rule"

    def __init__(self, vocab_path):
        vocab = load_vocab(vocab_path)
        self.terms = [t for ts in vocab.values() for t in ts]

    def check(self, claim: str, evidence: list[str]) -> dict:
        """Which tools/numbers in the claim are missing from the evidence (for error analysis)."""
        ev = " ".join(evidence)
        ev_nums = set(find_numbers(ev))
        ev_terms = {t.lower() for t in find_terms(ev, self.terms)}
        terms = find_terms(claim, self.terms)
        nums = find_numbers(claim)
        missing = [t for t in terms if t.lower() not in ev_terms] + [n for n in nums if n not in ev_nums]
        return {"checked": len(terms) + len(nums), "missing": missing}

    def verify(self, claim: str, evidence: list[str]):
        t0 = time.perf_counter()
        r = self.check(claim, evidence)
        score = 1.0 if r["checked"] == 0 else 1 - len(r["missing"]) / r["checked"]
        return not r["missing"], score, Usage(latency_s=time.perf_counter() - t0)


class NLIVerifier:
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.5):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.threshold = threshold
        # find the 'entailment' output column from the model's own label map (order differs by model)
        config = getattr(self.model, "config", None) or getattr(getattr(self.model, "model", None), "config", None)
        id2label = getattr(config, "id2label", None) or {}
        self.entail_idx = next((int(i) for i, l in id2label.items() if "entail" in str(l).lower()), 1)

    def score(self, claim: str, evidence: list[str]) -> float:
        import numpy as np
        if not evidence:
            return 0.0
        logits = np.asarray(self.model.predict([(e, claim) for e in evidence]))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return float(probs[:, self.entail_idx].max())

    def verify(self, claim: str, evidence: list[str]):
        t0 = time.perf_counter()
        s = self.score(claim, evidence)
        return s >= self.threshold, s, Usage(latency_s=time.perf_counter() - t0)


class FinetunedVerifier:
    """The BERT-family cross-encoder fine-tuned by scripts/train_verifier.py (Topic 3 recipe).
    Scores (bullet, claim) for every resume bullet; supported if max P(supported) >= threshold."""
    name = "finetuned"

    def __init__(self, path, threshold: float = 0.5, max_length: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
        self.threshold, self.max_length = threshold, max_length
        self.sup_idx = self.model.config.label2id.get("supported", 1)

    def score(self, claim: str, evidence: list[str]) -> float:
        if not evidence:
            return 0.0
        enc = self.tok(evidence, [claim] * len(evidence), truncation=True, max_length=self.max_length,
                       padding=True, return_tensors="pt")
        with self.torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1)[:, self.sup_idx]
        return float(probs.max())

    def verify(self, claim: str, evidence: list[str]):
        t0 = time.perf_counter()
        s = self.score(claim, evidence)
        return s >= self.threshold, s, Usage(latency_s=time.perf_counter() - t0)


class LLMVerifier:
    def __init__(self, llm, name: str = "llm"):
        self.llm, self.name = llm, name

    def verify(self, claim: str, evidence: list[str]):
        resume = "\n".join(f"- {e}" for e in evidence)
        out, u = self.llm.chat(prompts.VERIFY, f"<resume>\n{resume}\n</resume>\n\n<claim>\n{claim}\n</claim>",
                               max_tokens=5)
        supported = out.strip().upper().startswith("SUPPORTED")
        return supported, (1.0 if supported else 0.0), u


def build_verifier(cfg: dict, llm=None, method: str | None = None):
    from .config import resolve
    v = cfg["verifier"]
    method = method or v["method"]
    if method == "rule":
        return RuleVerifier(resolve(cfg["paths"]["skills_vocab"]))
    if method == "nli":
        return NLIVerifier(v["nli_model"], v["nli_threshold"])
    if method == "finetuned":
        return FinetunedVerifier(resolve(v["finetuned_path"]), v["finetuned_threshold"])
    if method == "finetuned_nli":   # the NLI checkpoint fine-tuned on our data: base vs fine-tuned, same model
        fv = FinetunedVerifier(resolve(v.get("finetuned_nli_path", "models/verifier-nli-ft")),
                               v.get("finetuned_nli_threshold", 0.5))
        fv.name = "finetuned_nli"
        return fv
    if method == "llm":
        return LLMVerifier(llm)
    raise ValueError(f"unknown verifier {method!r}")
