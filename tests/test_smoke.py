"""Offline smoke test: runs every stage and every script on the 3 sample pairs with the
MockLLM and the dependency-free components (BM25, rule verifier).

    pytest -q            or      python tests/test_smoke.py
"""
from __future__ import annotations

import csv
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from careercopilot.data import load  # noqa: E402
from careercopilot.llm import MockLLM, parse_json  # noqa: E402
from careercopilot.metrics import cohen_kappa, prf, wilcoxon_signed_rank  # noqa: E402
from careercopilot.pipeline import run_variant  # noqa: E402
from careercopilot.retrieval import BM25Retriever  # noqa: E402
from careercopilot.text import anonymise, split_bullets  # noqa: E402
from careercopilot.verifiers import RuleVerifier  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cfg(tmp: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    cfg["data"]["out_dir"] = str(tmp / "processed")
    cfg["paths"]["runs_dir"] = str(tmp / "runs")
    cfg["paths"]["labels_dir"] = str(tmp / "labels")
    cfg["paths"]["verifier_data"] = str(tmp / "verifier")
    cfg["retrieval"]["embedding_model"] = None   # forces bag-of-words similarity offline
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_text_and_metrics():
    assert anonymise("mail a.b@x.com or +65 9123 4567") == "mail [EMAIL] or [PHONE]"
    b = split_bullets("SKILLS\n- Built Power BI dashboards for 12 managers\n• Wrote SQL in PostgreSQL daily\nshort")
    assert b == ["Built Power BI dashboards for 12 managers", "Wrote SQL in PostgreSQL daily"]
    assert parse_json('```json\n["a", "b"]\n```') == ["a", "b"]
    assert prf([True, False, True], [True, True, False])["precision"] == 0.5
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) - 1) < 1e-9
    assert wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])["p"] == 1.0


def test_contextual_split_and_tfidf():
    from careercopilot.retrieval import TfidfRetriever
    from careercopilot.text import split_with_context
    b, c = split_with_context("Data Analyst, Retail Co (2023-2025)\n- Built Power BI dashboards for 12 managers\n"
                              "Intern, Logistics Ltd (2022)\n- Analysed delivery delays in Excel")
    assert c[b.index("Built Power BI dashboards for 12 managers")] == "Data Analyst, Retail Co (2023-2025)"
    assert c[b.index("Analysed delivery delays in Excel")] == "Intern, Logistics Ltd (2022)"
    top = TfidfRetriever().rank("Excel analysis of delays", b, 1)
    assert b[top[0][0]].startswith("Analysed")


def test_rule_verifier():
    v = RuleVerifier(ROOT / "data/skills.txt")
    ev = ["Built Power BI dashboards for 12 regional store managers"]
    assert v.verify("Built Power BI dashboards for 12 managers", ev)[0]
    assert not v.verify("Built Tableau dashboards for 12 managers", ev)[0]
    assert not v.verify("Built Power BI dashboards for 500 managers", ev)[0]


def test_pipeline_variants():
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    pair = load(ROOT / "data/sample/pairs.jsonl")[1]
    llm, verifier = MockLLM(), RuleVerifier(ROOT / "data/skills.txt")
    a = run_variant(pair, "A", llm, BM25Retriever(), verifier, cfg)
    d = run_variant(pair, "D", llm, BM25Retriever(), verifier, cfg, requirements=a.requirements)
    assert any("Tableau" in l for l in a.lines)          # mock baseline plants a fabrication
    assert not any("Tableau" in l for l in d.lines)      # full system removes it
    assert d.evidence and all(d.citations[1:])           # grounded lines cite evidence
    c = run_variant(pair, "C", llm, BM25Retriever(), verifier, cfg, requirements=a.requirements)
    assert c.regenerations >= 1 and not any("Tableau" in l for l in c.lines)


def test_all_scripts_end_to_end():
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = str(_cfg(tmp))
        _script("prepare_data").main(cfg, sample=True)
        _script("tailor_one").main(cfg, "sample-none", "D", mock=True)

        e1 = _script("run_e1_retrieval")
        from careercopilot.config import load_config
        c = load_config(cfg)
        e1.make_template(c, MockLLM(), 3)
        rows = list(csv.DictReader(open(tmp / "labels/e1_labels.csv", encoding="utf-8")))
        for r in rows:  # fake labels: bullet 0 supports every requirement
            r["gold_bullet_ids"] = "0"
        with open(tmp / "labels/e1_labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        res1 = e1.evaluate(c, ["tfidf", "bm25", "bm25+ctx", "embedding"])
        assert {"tfidf", "bm25", "bm25+ctx"} <= set(res1)

        _script("make_fabrications").main(cfg, n=10, mock=True)
        _script("make_verifier_train").main(cfg, teacher_frac=1.0, mock=True, sample=True)
        assert (tmp / "verifier/train.jsonl").exists()
        res2 = _script("run_e2_verifiers").main(cfg, ["rule", "llm"], include_unreviewed=True, mock=True)
        assert res2["rule"]["recall_by_type"]

        _script("run_e3_tailoring").main(cfg, ["A", "B", "C", "D"], mock=True)
        summary = _script("evaluate_e3").main(cfg, mock=True, pairwise=True)
        assert {s["variant"] for s in summary} == {"A", "B", "C", "D"}
        a_all = next(s for s in summary if s["variant"] == "A" and s["fit"] == "all")
        d_all = next(s for s in summary if s["variant"] == "D" and s["fit"] == "all")
        assert a_all["any_introduced"] > d_all["any_introduced"]   # programmatic check sees the planted Tableau
        assert (tmp / "labels/e3_preference.csv").exists()
        pr = _script("run_probes")
        pr.main(cfg, "injection", ["A", "D"], n=3, mock=True)
        pr.main(cfg, "bias", ["A"], n=2, mock=True)
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
