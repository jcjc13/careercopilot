"""Offline smoke test: runs every stage and every script on the 3 sample pairs with the
MockLLM and the dependency-free components (BM25, rule verifier).

    pytest -q            or      python tests/test_smoke.py
"""
from __future__ import annotations

import csv
import importlib.util
import os
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
from careercopilot.metrics import (cohen_kappa, paired_bootstrap_mean_ci, prf,
                                   wilcoxon_signed_rank)  # noqa: E402
from careercopilot.pipeline import run_variant  # noqa: E402
from careercopilot.retrieval import BM25Retriever  # noqa: E402
from careercopilot.text import anonymise, split_bullets, split_with_context  # noqa: E402
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
    cfg["retrieval"]["method"] = "bm25"          # the default (hybrid) needs embedding + re-ranker models
    cfg["verifier"]["method"] = "rule"           # the default (finetuned_nli) needs the trained model files
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_text_and_metrics():
    assert anonymise("mail a.b@x.com or +65 9123 4567") == "mail [EMAIL] or [PHONE]"
    b = split_bullets("SKILLS\n- Built Power BI dashboards for 12 managers\n• Wrote SQL in PostgreSQL daily\nshort")
    assert b == ["Built Power BI dashboards for 12 managers", "Wrote SQL in PostgreSQL daily"]
    b, _ = split_with_context(
        "- Built weekly dashboards for sales leaders. Presented the results to senior stakeholders."
    )
    assert b == ["Built weekly dashboards for sales leaders. Presented the results to senior stakeholders."]
    sentence_chunks, _ = split_with_context(
        "- Built weekly dashboards for sales leaders. Presented the results to senior stakeholders.",
        split_sentences=True,
    )
    assert sentence_chunks == [
        "Built weekly dashboards for sales leaders.",
        "Presented the results to senior stakeholders.",
    ]
    b, _ = split_with_context(
        "SummaryHighly organized and detail-oriented professional. "
        "Career OverviewSenior data analyst with experience. "
        "Work ExperienceManaged supply chain reporting.Automated new dashboards for regional teams. "
        + "Created monthly reports for stakeholders. " * 12
    )
    assert any("Highly organized" in bullet for bullet in b)
    assert any("Senior data analyst" in bullet for bullet in b)
    assert any("Automated new dashboards" in bullet for bullet in b)
    assert all(len(bullet) <= 300 for bullet in b)
    assert parse_json('```json\n["a", "b"]\n```') == ["a", "b"]
    assert prf([True, False, True], [True, True, False])["precision"] == 0.5
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) - 1) < 1e-9
    assert wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])["p"] == 1.0


def test_contextual_split_and_tfidf():
    from careercopilot.retrieval import BM25Retriever, TfidfRetriever
    from careercopilot.text import split_with_context
    b, c = split_with_context("Data Analyst, Retail Co (2023-2025)\n- Built Power BI dashboards for 12 managers\n"
                              "Intern, Logistics Ltd (2022)\n- Analysed delivery delays in Excel")
    assert c[b.index("Built Power BI dashboards for 12 managers")] == "Data Analyst, Retail Co (2023-2025)"
    assert c[b.index("Analysed delivery delays in Excel")] == "Intern, Logistics Ltd (2022)"
    top = TfidfRetriever().rank("Excel analysis of delays", b, 1)
    assert b[top[0][0]].startswith("Analysed")
    unrelated = ["Built a Java service", "Designed database schemas"]
    assert BM25Retriever().rank("Kubernetes", unrelated, 3) == []
    assert TfidfRetriever().rank("Kubernetes", unrelated, 3) == []


def test_embedding_retriever_caches_resume_embeddings(monkeypatch):
    import numpy as np
    import careercopilot.retrieval as retrieval

    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, normalize_embeddings):
            self.calls.append(tuple(texts))
            if texts == ["alpha", "beta"]:
                return np.array([[1.0, 0.0], [0.0, 1.0]])
            return np.array([[1.0, 0.0]])

    model = FakeModel()
    monkeypatch.setattr(retrieval, "_st_model", lambda _: model)
    retriever = retrieval.EmbeddingRetriever("test-model")

    first = retriever.rank("alpha", ["alpha", "beta"], 1)
    second = retriever.rank("alpha", ["alpha", "beta"], 1)

    assert first == second == [(0, 1.0)]
    assert model.calls == [("alpha",), ("alpha", "beta"), ("alpha",)]


def test_parse_tailored_salvages_malformed_json():
    from careercopilot.generation import _parse_tailored
    broken = ('{"summary":"Data analyst.","bullets":[ "Built dashboards for 12 managers. [E1]",'
              '" Wrote SQL reports. [E2, E3]"}')      # closing bracket missing, as the model sometimes writes it
    lines, cites = _parse_tailored(broken)
    assert lines == ["Summary: Data analyst.", "Built dashboards for 12 managers.", "Wrote SQL reports."]
    assert cites == [[], ["E1"], ["E2", "E3"]]


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
    e = run_variant(pair, "E", llm, BM25Retriever(), verifier, cfg, requirements=a.requirements)
    assert e.evidence and all(e.citations[1:])
    assert e.pipeline_claims and not e.trace
    assert not any("Tableau" in l for l in e.lines)


def test_paired_bootstrap_and_e3_calculation():
    ci = paired_bootstrap_mean_ci([-1.0, -0.5, 0.0, 0.5], replicates=1000, seed=469)
    assert ci == paired_bootstrap_mean_ci([-1.0, -0.5, 0.0, 0.5], replicates=1000, seed=469)
    assert ci["n_pairs"] == 4 and ci["mean_difference"] == -0.25
    assert ci["ci_lower"] <= ci["mean_difference"] <= ci["ci_upper"]

    bootstrap = _script("bootstrap_e3")
    rows = [
        {"pair_id": "p1", "variant": "D", "fit": "good", "unsupported_rate": "0.2",
         "coverage": "0.5", "any_introduced": "0", "latency_s": "10", "cost_usd": "0.1"},
        {"pair_id": "p1", "variant": "E", "fit": "good", "unsupported_rate": "0.4",
         "coverage": "0.7", "any_introduced": "1", "latency_s": "8", "cost_usd": "0.08"},
        {"pair_id": "p2", "variant": "D", "fit": "none", "unsupported_rate": "0.6",
         "coverage": "0.1", "any_introduced": "1", "latency_s": "12", "cost_usd": "0.2"},
        {"pair_id": "p2", "variant": "E", "fit": "none", "unsupported_rate": "0.2",
         "coverage": "0.2", "any_introduced": "0", "latency_s": "9", "cost_usd": "0.12"},
    ]
    result = bootstrap.calculate(rows, "D", ["E"], replicates=500, seed=469)
    overall = result["metrics"]["E_vs_D/all"]
    assert overall["unsupported_rate"]["n_pairs"] == 2
    assert abs(overall["unsupported_rate"]["mean_difference"] + 0.1) < 1e-12
    assert abs(overall["coverage"]["mean_difference"] - 0.15) < 1e-12
    assert result["metrics"]["E_vs_D/good"]["unsupported_rate"]["n_pairs"] == 1


def test_evidence_sufficiency_conditions():
    from careercopilot.schema import Pair

    probe = _script("run_evidence_sufficiency")
    pair = Pair(
        "evidence-test",
        "Analyst\n- Built weekly Power BI dashboards\n- Presented findings to managers",
        "Data analyst role",
        "potential",
        ["Built weekly Power BI dashboards", "Presented findings to managers"],
        ["Analyst", "Analyst"],
    )
    conditions = probe.build_conditions(pair, [0, 1])
    assert [condition for condition, _, _ in conditions] == [
        "intact", "partial", "labelled_evidence_removed"
    ]
    assert conditions[1][1].bullets == [pair.bullets[0]]
    assert conditions[1][2] == [1]
    assert conditions[2][1].bullets == []
    assert conditions[2][2] == [0, 1]
    assert pair.bullets == ["Built weekly Power BI dashboards", "Presented findings to managers"]


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
        _script("run_evidence_sufficiency").main(cfg, ["A", "D"], limit=1, mock=True)
        sufficiency_files = list((tmp / "runs").rglob("evidence_sufficiency.csv"))
        assert len(sufficiency_files) == 1
        sufficiency = list(csv.DictReader(open(sufficiency_files[0], encoding="utf-8")))
        assert len(sufficiency) == 4
        assert {r["condition"] for r in sufficiency} == {"intact", "labelled_evidence_removed"}
        assert all(r["supported_requirement_coverage"] != "" for r in sufficiency)
        _script("run_evidence_sufficiency").main(
            cfg, ["A", "D"], limit=1, mock=True, resume=True)
        resumed = list(csv.DictReader(open(sufficiency_files[0], encoding="utf-8")))
        assert len(resumed) == len(sufficiency)

        _script("make_fabrications").main(cfg, n=10, mock=True)
        _script("make_verifier_train").main(cfg, teacher_frac=1.0, mock=True, sample=True)
        assert (tmp / "verifier/train.jsonl").exists()
        _script("make_hard_claims").main(cfg, n_fab=1, n_faith=1, mock=True)   # appends set=hard rows
        res2 = _script("run_e2_verifiers").main(cfg, ["rule", "llm"], include_unreviewed=True, mock=True)
        assert res2["standard"]["rule"]["recall_by_type"] and "mcnemar" in res2
        assert set(res2) - {"mcnemar"} <= {"standard", "hard"}

        _script("run_e3_tailoring").main(cfg, ["A", "B", "C", "D", "E"], mock=True)
        summary = _script("evaluate_e3").main(cfg, mock=True, pairwise=True)
        assert {s["variant"] for s in summary} == {"A", "B", "C", "D", "E"}
        a_all = next(s for s in summary if s["variant"] == "A" and s["fit"] == "all")
        d_all = next(s for s in summary if s["variant"] == "D" and s["fit"] == "all")
        assert a_all["any_introduced"] > d_all["any_introduced"]   # programmatic check sees the planted Tableau
        assert (tmp / "labels/e3_preference.csv").exists()
        am = _script("agent_metrics").main(cfg, mock=True)
        assert {"A", "B", "C", "D", "E"} <= set(am) and am["D"]["steps_mean"] >= am["A"]["steps_mean"]
        _script("analyse_e3_failures").main(cfg)
        _script("make_e3_figure").main(cfg)
        assert list((tmp / "runs").rglob("e3_tradeoff.svg"))
        pr = _script("run_probes")
        pr.main(cfg, "injection", ["A", "D"], n=3, mock=True)
        pr.main(cfg, "bias", ["A"], n=2, mock=True)
    finally:
        shutil.rmtree(tmp)


def test_app_runs_offline():
    """The Streamlit app starts, tailors a sample with the offline mock and renders both tabs."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        return   # streamlit not installed: skip
    os.environ["CAREERCOPILOT_MOCK"] = "1"
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.selectbox[0].select(next(o for o in at.selectbox[0].options if o != "(none)")).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert any(c.label == "keep" for c in at.checkbox)   # one accept box per tailored line
    assert at.session_state["last"][2] == "D"   # the app always runs variant D
    assert len(at.tabs) == 2


class _ScriptedLLM:
    """Replies to the controller from a fixed list; rewrites return a clean line. For testing the loop's rules."""
    model = "scripted"

    def __init__(self, decisions, rewrite_out="Built reports in SQL"):
        self.decisions, self.rewrite_out, self.calls = list(decisions), rewrite_out, []

    def chat(self, system, user, max_tokens=800):
        from careercopilot.llm import task_of
        from careercopilot.schema import Usage
        task = task_of(system)
        self.calls.append(task)
        out = self.decisions.pop(0) if task == "controller" else self.rewrite_out
        return out, Usage(llm_calls=1)


class _FlagBad:
    """Verifier stub: a claim containing 'BAD' is unsupported (score 0.2); everything else supported (0.9)."""
    name, threshold = "flagbad", 0.7

    def verify(self, claim, evidence):
        from careercopilot.schema import Usage
        bad = "BAD" in claim
        return (not bad), (0.2 if bad else 0.9), Usage()


def _controller_run(decisions, bad_line="Led BAD enterprise reporting [E1]", mode="llm", max_total=4,
                    rewrite_out="Built reports in SQL", review_band=0.1):
    from careercopilot import controller
    from careercopilot.generation import format_evidence
    from careercopilot.pipeline import verify_lines
    from careercopilot.generation import rewrite_conservatively
    from careercopilot.schema import Claim, Pair, Usage
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    cfg["tailoring"]["claim_split"] = "line"
    cfg["tailoring"]["controller"].update(mode=mode, max_total=max_total, review_band=review_band)
    bullets = ["Built reports in SQL for the finance team", "Managed a small team of analysts", "Used Tableau dashboards"]
    pair = Pair("t", "\n".join(bullets), "job", "good", bullets, ["", "", ""])
    evidence = {0: [0, 1]}
    _, id_map = format_evidence(["SQL reporting"], evidence, bullets)
    lines = ["Summary: Analyst", "Built reports in SQL", bad_line.replace(" [E1]", "")]
    cites = [[], ["E1"], ["E1"]]
    llm, usage = _ScriptedLLM(decisions, rewrite_out), Usage()
    claims, _ = verify_lines(lines, bullets, llm, _FlagBad(), "line")
    out = controller.run_controller_loop(
        lines=lines, cites=cites, claims=claims, pair=pair, requirements=["SQL reporting"], evidence=evidence,
        id_map=id_map, llm=llm, retriever=BM25Retriever(), verifier=_FlagBad(), cfg=cfg, verify_lines=verify_lines,
        regenerate=rewrite_conservatively, usage=usage)
    return out, llm


def test_controller_decisions_and_limits():
    from careercopilot import controller
    d = controller.parse_decision('{"action": "drop", "failure_type": "no_evidence", "reason": "x", "confidence": "0.8"}')
    assert d.action == "DROP" and d.confidence == 0.8
    for bad in ('{"action": "FABRICATE"}', "no json here", "[1, 2]"):
        try:
            controller.parse_decision(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError:
            pass
    st = controller.LineState(0, "x", False, [], [{"id": "E1", "text": "t"}], [], [], 0.7,
                              attempts={"retrieval": 2, "rewrite": 2, "total": 4},
                              limits={"total": 4, "retrieval": 2, "rewrite": 2})
    assert st.allowed_actions() == ["DROP"]                    # every corrective budget is spent; no claim is near the threshold
    st.unsupported = [{"claim": "c", "score": 0.65}]           # 0.65 is within 0.1 of the 0.7 threshold
    assert st.allowed_actions() == ["DROP", "HUMAN_REVIEW"]
    long = controller.LineState(0, "x", False, [], [{"id": "E1", "text": "word " * 60}], [], [], 0.7,
                                limits={"total": 4, "retrieval": 2, "rewrite": 2})
    assert "USE_ORIGINAL" not in long.allowed_actions()        # a 60-word 'bullet' is a skills block, not a bullet


def test_controller_loop_actions():
    fix = '{"action": "REWRITE_CONSERVATIVELY", "failure_type": "unsupported_embellishment", "reason": "r", "instructions": "remove BAD"}'
    (lines, cites, claims, removed, escalated, trace, actions, regens), _ = _controller_run([fix])
    assert lines[-1] == "Built reports in SQL" and not removed and not escalated and regens == 1
    assert trace[0]["outcome"] == "accepted" and all(c.supported for c in claims)

    (lines, _, _, removed, escalated, trace, actions, _), _ = _controller_run(
        ['{"action": "USE_ORIGINAL", "failure_type": "unnecessary_rewriting", "reason": "r"}'])
    assert lines[-1] in ("Built reports in SQL for the finance team", "Managed a small team of analysts")
    assert trace[0]["outcome"] == "accepted" and actions == {"USE_ORIGINAL": 1}

    hr = '{"action": "HUMAN_REVIEW", "failure_type": "ambiguous_entailment", "reason": "r"}'
    (lines, _, _, removed, escalated, trace, actions, _), _ = _controller_run([hr], review_band=0.6)   # 0.2 is "near" 0.7
    assert escalated and not removed and len(lines) == 2
    # the same request with a score far from the threshold is not allowed: the rule router takes over
    (_, _, _, _, escalated, trace, _, _), _ = _controller_run([hr])
    assert not escalated and trace[0]["steps"][0]["source"] == "override"

    (lines, _, _, removed, escalated, trace, _, _), _ = _controller_run(['{"action": "DROP", "reason": "r"}'])
    assert removed and not escalated and len(lines) == 2

    # an action the state does not allow (ACCEPT) is replaced by the rule router and logged
    (lines, _, _, _, _, trace, _, _), _ = _controller_run(['{"action": "ACCEPT", "reason": "r"}'])
    assert trace[0]["steps"][0]["source"] == "override" and trace[0]["steps"][0]["overridden_from"] == "ACCEPT"

    # a reply that never parses ends in HUMAN_REVIEW, not a guess
    (_, _, _, _, escalated, trace, _, _), llm = _controller_run(["nonsense", "still nonsense"])
    assert escalated and trace[0]["steps"][0]["source"] == "fallback" and llm.calls.count("controller") == 2

    # loop limit: rewrites that keep failing are dropped once the total budget is spent
    rw = '{"action": "REWRITE_CONSERVATIVELY", "reason": "r"}'
    (lines, _, _, removed, escalated, trace, actions, _), _ = _controller_run([rw, rw], max_total=2, rewrite_out="still BAD")
    assert removed and trace[0]["outcome"] == "loop_limit" and actions["LOOP_LIMIT"] == 1 and len(lines) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
