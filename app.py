"""Streamlit demo:  streamlit run app.py
Paste a resume and a job description (or load a sample), pick a variant, and see the
requirements, the evidence found for each, the tailored resume, and the verifier's verdicts."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from careercopilot.config import load_config, resolve  # noqa: E402
from careercopilot.data import load  # noqa: E402
from careercopilot.llm import build_llm  # noqa: E402
from careercopilot.pipeline import VARIANTS, run_variant  # noqa: E402
from careercopilot.retrieval import build_retriever  # noqa: E402
from careercopilot.schema import Pair  # noqa: E402
from careercopilot.text import anonymise, split_with_context  # noqa: E402
from careercopilot.verifiers import build_verifier  # noqa: E402

st.set_page_config(page_title="CareerCopilot", layout="wide")
cfg = load_config()
samples = {p.pair_id: p for p in load(resolve(cfg["data"]["sample_pairs"]))}

st.title("CareerCopilot")
st.caption("Tailor a resume to a job without inventing experience.")

with st.sidebar:
    sample = st.selectbox("Load a sample", ["(none)"] + list(samples))
    variant = st.radio("Variant", list(VARIANTS), index=3,
                       format_func=lambda v: {"A": "A - unrestricted", "B": "B - grounded",
                                              "C": "C - unrestricted + verifier", "D": "D - full system"}[v])
    retr = st.selectbox("Retriever", ["bm25", "tfidf", "embedding", "hybrid"])
    verif = st.selectbox("Verifier", ["rule", "nli", "finetuned", "llm"])
    mock = st.checkbox("Offline mock LLM", value=False, help="no API key needed; outputs are not meaningful")

pre = samples.get(sample)
c1, c2 = st.columns(2)
resume = c1.text_area("Resume", pre.resume if pre else "", height=320)
job = c2.text_area("Job description", pre.job if pre else "", height=320)

if st.button("Tailor", type="primary", disabled=not (resume.strip() and job.strip())):
    resume_a = anonymise(resume)
    bullets, contexts = split_with_context(resume_a)
    pair = Pair(pair_id="ui", resume=resume_a, job=anonymise(job), fit="unknown", bullets=bullets, contexts=contexts)
    with st.spinner("Running pipeline..."):
        llm = build_llm(cfg, mock=mock)
        res = run_variant(pair, variant, llm, build_retriever(cfg, retr), build_verifier(cfg, llm, verif), cfg)

    left, right = st.columns(2)
    with left:
        st.subheader("Requirements and evidence")
        for i, r in enumerate(res.requirements):
            st.markdown(f"**{i + 1}. {r}**")
            ev = res.evidence.get(i, [])
            for b in ev:
                st.caption(f"↳ {pair.bullets[b]}")
            if not ev and VARIANTS[variant]["grounded"]:
                st.caption("↳ no evidence retrieved")
    with right:
        st.subheader("Tailored resume")
        verdict = {}
        for c in res.pipeline_claims:
            verdict.setdefault(c.line_idx, []).append(c)
        for i, line in enumerate(res.lines):
            cites = res.citations[i] if i < len(res.citations) else []
            claims = verdict.get(i, [])
            mark = "" if not claims else (" ✅" if all(c.supported for c in claims) else " ⚠️")
            st.markdown(f"{'' if i == 0 else '- '}{line}{mark}" + (f"  `{', '.join(cites)}`" if cites else ""))
        if res.removed_lines:
            st.warning("Removed because they could not be supported:\n\n" +
                       "\n".join(f"- {l}" for l in res.removed_lines))
        u = res.usage
        st.caption(f"{u.llm_calls} LLM calls · {u.tokens_in + u.tokens_out} tokens · "
                   f"{u.latency_s:.1f}s · ${u.cost_usd:.4f} · {res.regenerations} regenerations")
