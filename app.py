"""Streamlit demo:  streamlit run app.py

Three tabs:
  Tailor         paste a resume and a job (or load a sample), pick a variant, see the requirements, the evidence found
                 for each, the tailored resume with a verdict per line, the evidence behind every cited line, and
                 accept or reject each line before downloading the final text
  Saved results  the 36 resume-job pairs of the E3 study, all four variants side by side, with the judge's verdict
                 on every claim (instant: read from runs/<run>/, no API calls)
  Experiments    the E1, E2 and E3 results tables, the main figure, and the probe results
"""
import base64
import csv
import json
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from careercopilot.config import load_config, resolve, run_dir  # noqa: E402
from careercopilot.data import load  # noqa: E402
from careercopilot.generation import format_evidence  # noqa: E402
from careercopilot.llm import build_llm  # noqa: E402
from careercopilot.pipeline import VARIANTS, run_variant, verify_lines  # noqa: E402
from careercopilot.retrieval import build_retriever  # noqa: E402
from careercopilot.schema import Pair  # noqa: E402
from careercopilot.text import anonymise, split_with_context  # noqa: E402
from careercopilot.verifiers import build_verifier  # noqa: E402

st.set_page_config(page_title="CareerCopilot", layout="wide")
cfg = load_config()
RUN = run_dir(cfg)
samples = {p.pair_id: p for p in load(resolve(cfg["data"]["sample_pairs"]))}
VARIANT_NAMES = {"A": "A - unrestricted", "B": "B - grounded", "C": "C - unrestricted + verifier", "D": "D - full system"}
EXPECTED_TIME = {"A": "about 1 second", "B": "about 1 to 2 seconds",
                 "C": "several minutes with 2 regeneration attempts (6 minutes on average in our study)",
                 "D": "several minutes with 2 regeneration attempts (4 minutes on average in our study)"}
RETRIEVERS = ["bm25", "tfidf", "embedding", "hybrid"]
VERIFIERS = ["rule", "nli", "finetuned", "finetuned_nli", "llm"]
VERIFIER_NOTE = {"finetuned_nli": "fine-tuned NLI (best on our tests)", "finetuned": "fine-tuned BERT",
                 "nli": "zero-shot NLI", "rule": "rule check", "llm": "LLM judge"}


# ---------- helpers ----------

@st.cache_resource(show_spinner="Loading models (the first run takes a while)...")
def components(retriever: str, verifier: str, mock: bool):
    llm = build_llm(cfg, mock=mock)
    return llm, build_retriever(cfg, retriever), build_verifier(cfg, llm, verifier)


def redact_name(text: str, name: str) -> str:
    """The user's own name, removed before any text reaches an LLM. Regex cannot find names reliably, so the
    user says which name to remove."""
    parts = [name.strip()] + [p for p in name.split() if len(p) > 2]
    for p in sorted({x for x in parts if x}, key=len, reverse=True):
        text = re.sub(re.escape(p), "[NAME]", text, flags=re.I)
    return text


def read_csv(path):
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def read_jsonl_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except OSError:
        return []


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def pct(x, digits=1):
    try:
        return f"{100 * float(x):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


# ---------- tab 1: tailor ----------

def render_result(res, pair, variant):
    id_map = {}
    if VARIANTS[variant]["grounded"]:
        _, id_map = format_evidence(res.requirements, res.evidence, pair.bullets)   # E-id -> bullet index
    claims_by_line = {}
    for c in res.pipeline_claims:
        claims_by_line.setdefault(c.line_idx, []).append(c)

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
        st.caption("✅ every claim in the line is supported · ⚠️ a claim could not be matched to your resume · "
                   "untagged lines were not checked. Untick a line to leave it out of the final text.")
        keep = []
        for i, line in enumerate(res.lines):
            cites = res.citations[i] if i < len(res.citations) else []
            claims = claims_by_line.get(i, [])
            mark = "" if not claims else (" ✅" if all(c.supported for c in claims) else " ⚠️")
            default = not claims or all(c.supported for c in claims)
            c1, c2 = st.columns([0.07, 0.93])
            ok = c1.checkbox("keep", value=default, key=f"keep_{variant}_{i}_{abs(hash(line))}", label_visibility="collapsed")
            c2.markdown(f"{'' if i == 0 else '- '}{line}{mark}" + (f"  `{', '.join(cites)}`" if cites else ""))
            if ok:
                keep.append(i)
            bad = [c for c in claims if not c.supported]
            if bad or cites:
                with c2.expander("Why / evidence", expanded=False):
                    for c in bad:
                        score = f" (support score {c.score:.2f})" if c.score is not None else ""
                        st.markdown(f"⚠️ not supported{score}: _{c.text}_")
                    for e in cites:
                        if e in id_map:
                            st.caption(f"{e}: {pair.bullets[id_map[e]]}")
        if res.removed_lines:
            st.warning("Removed because they could not be supported:\n\n" + "\n".join(f"- {l}" for l in res.removed_lines))
        u = res.usage
        st.caption(f"{u.llm_calls} LLM calls · {u.tokens_in + u.tokens_out} tokens · {u.latency_s:.1f}s · "
                   f"${u.cost_usd:.4f} · {res.regenerations} regenerations")

    final = "\n".join(res.lines[i].removeprefix("Summary: ") if i == 0 else f"- {res.lines[i]}" for i in keep)
    st.subheader("Final text (accepted lines)")
    st.text_area("Copy or download this", final, height=220, label_visibility="collapsed")
    st.download_button("Download as text", final, file_name="tailored_resume.txt")

    if VARIANTS[variant]["grounded"]:
        st.subheader("Requirements without a supporting line")
        kept_cites = {e for i in keep for e in (res.citations[i] if i < len(res.citations) else [])}
        gaps = [r for i, r in enumerate(res.requirements)
                if not ({e for e, b in id_map.items() if b in res.evidence.get(i, [])} & kept_cites)]
        if gaps:
            st.caption("No accepted line cites evidence for these. Add real experience if you have it, or leave them "
                       "out. The tool will not invent it.")
            for g in gaps:
                st.markdown(f"- {g}")
        else:
            st.caption("Every requirement has at least one accepted line that cites evidence for it.")
    else:
        st.caption("The requirement check needs a grounded variant (B or D), because only those cite evidence.")


def tab_tailor():
    with st.sidebar:
        st.header("Settings")
        sample = st.selectbox("Load a sample", ["(none)"] + list(samples))
        variant = st.radio("Variant", list(VARIANTS), index=3, format_func=VARIANT_NAMES.get)
        retr_default = cfg["retrieval"]["method"]
        retr = st.selectbox("Retriever", RETRIEVERS, index=RETRIEVERS.index(retr_default) if retr_default in RETRIEVERS else 0)
        ver_default = cfg["verifier"]["method"]
        verif = st.selectbox("Verifier", VERIFIERS, index=VERIFIERS.index(ver_default) if ver_default in VERIFIERS else 0,
                             format_func=lambda v: VERIFIER_NOTE.get(v, v))
        regen = st.slider("Regeneration attempts", 0, 2, cfg["tailoring"]["max_regen"],
                          help="C and D only. 0 removes unsupported lines at once, which is much faster.")
        flag = st.checkbox("Also flag unsupported claims in A and B", value=True,
                           help="A and B do not verify. This runs the verifier afterwards, only to show what it would flag.")
        mock = st.checkbox("Offline mock LLM", value=False, help="no API key needed; outputs are not meaningful")

    pre = samples.get(sample)
    c1, c2 = st.columns(2)
    resume = c1.text_area("Resume", pre.resume if pre else "", height=320)
    job = c2.text_area("Job description", pre.job if pre else "", height=320)
    name = st.text_input("Your name (optional)", help="Removed from the resume before any text is sent to the LLM. "
                         "Emails, phone numbers and URLs are always removed.")
    if VARIANTS[variant]["verify"]:
        st.info(f"Variant {variant} verifies every claim, so it takes {EXPECTED_TIME[variant]}. "
                f"Set regeneration attempts to 0 in the sidebar for a quick run.")

    if st.button("Tailor", type="primary", disabled=not (resume.strip() and job.strip())):
        resume_a = anonymise(redact_name(resume, name) if name.strip() else resume)
        bullets, contexts = split_with_context(resume_a)
        pair = Pair(pair_id="ui", resume=resume_a, job=anonymise(job), fit="unknown", bullets=bullets, contexts=contexts)
        run_cfg = {**cfg, "tailoring": {**cfg["tailoring"], "max_regen": regen}}
        try:
            with st.spinner(f"Running variant {variant}..."):
                llm, retriever, verifier = components(retr, verif, mock)
                res = run_variant(pair, variant, llm, retriever, verifier, run_cfg)
                if flag and not VARIANTS[variant]["verify"]:
                    claims, u = verify_lines(res.lines, pair.bullets, llm, verifier, cfg["tailoring"]["claim_split"])
                    res.pipeline_claims = claims
                    res.usage.add(u)
        except Exception as e:   # a missing model folder, a missing key or a network error
            st.error(f"The pipeline failed: {type(e).__name__}: {e}")
            st.caption("Missing key? Put GROQ_API_KEY in .env, or tick the offline mock. Missing verifier model? "
                       "Pick the rule check or the zero-shot NLI.")
            return
        st.session_state["last"] = (res, pair, variant)
    if "last" in st.session_state:
        render_result(*st.session_state["last"])


# ---------- tab 2: saved E3 results ----------

def tab_saved():
    outs = read_jsonl_file(RUN / "e3_outputs.jsonl")
    judged = {(d["pair_id"], d["variant"]): d for d in read_jsonl_file(RUN / "e3_judged.jsonl")}
    per = {(r["pair_id"], r["variant"]): r for r in read_csv(RUN / "e3_per_resume.csv")}
    pairs = {p.pair_id: p for p in load(resolve(cfg["data"]["out_dir"]) / "pairs.jsonl")} \
        if (resolve(cfg["data"]["out_dir"]) / "pairs.jsonl").exists() else {}
    if not outs or not pairs:
        st.info(f"No saved E3 run found in {RUN}. Run scripts/run_e3_tailoring.py and scripts/evaluate_e3.py first.")
        return
    by_pair = {}
    for o in outs:
        by_pair.setdefault(o["pair_id"], {})[o["variant"]] = o
    fits = {"good": "Good fit", "potential": "Potential fit", "none": "No fit"}
    fit_filter = st.radio("Fit level", ["all"] + list(fits), horizontal=True, format_func=lambda f: fits.get(f, "All"))
    ids = [p for p in by_pair if fit_filter == "all" or by_pair[p]["A"]["fit"] == fit_filter]
    pid = st.selectbox("Resume-job pair", ids, format_func=lambda p: f"{p} ({fits.get(by_pair[p]['A']['fit'], '')})")
    pair = pairs.get(pid)
    if pair:
        with st.expander("Job description"):
            st.write(pair.job)
        with st.expander("Original resume"):
            st.write(pair.resume)
    st.caption("Same resume and job, four variants. ⚠️ = the claim judge (gpt-oss-120b) found a claim in the line that the "
               "resume does not support. Judge agreement with human labels: 0.90, Cohen's kappa 0.74.")
    cols = st.columns(4)
    for col, v in zip(cols, "ABCD"):
        o = by_pair[pid].get(v)
        with col:
            st.subheader(VARIANT_NAMES[v])
            if not o:
                st.write("no output")
                continue
            r, j = per.get((pid, v), {}), judged.get((pid, v), {"claims": []})
            st.metric("Unsupported claims", pct(r.get("unsupported_rate")))
            st.caption(f"{r.get('claims', '?')} claims · coverage {pct(r.get('coverage'), 0)} · "
                       f"invented tool/number: {'yes' if str(r.get('any_introduced')) in ('1', '1.0') else 'no'} · "
                       f"{float(r.get('latency_s', 0)):.0f}s · ${float(r.get('cost_usd', 0)):.4f} · "
                       f"{o['regenerations']} regenerations · {len(o['removed_lines'])} lines removed")
            by_line = {}
            for c in j["claims"]:
                by_line.setdefault(c["line_idx"], []).append(c)
            for i, line in enumerate(o["lines"]):
                bad = [c for c in by_line.get(i, []) if not c["supported"]]
                st.markdown(f"{'' if i == 0 else '- '}{line}{' ⚠️' if bad else ''}")
                if bad:
                    with st.expander(f"{len(bad)} unsupported claim(s)"):
                        for c in bad:
                            st.markdown(f"- _{c['text']}_")
            if o["removed_lines"]:
                with st.expander(f"{len(o['removed_lines'])} line(s) removed by the verifier"):
                    for l in o["removed_lines"]:
                        st.markdown(f"- {l}")


# ---------- tab 3: experiment results ----------

def tab_experiments():
    st.caption(f"Read from the saved run `{cfg['run_name']}` (E1 from run v1). Full tables and discussion: the project report.")
    st.header("E1: evidence retrieval")
    e1 = read_json(resolve("runs/v1/e1_results.json"))
    if e1:
        st.table([{"Retriever": k, "Recall@1": f"{v['recall@1']:.3f}", "Recall@3": f"{v['recall@3']:.3f}",
                   "MRR": f"{v['mrr']:.3f}", "ms/query": f"{v['ms_per_query']:.0f}"} for k, v in e1.items()])
        st.caption(f"{next(iter(e1.values()))['n']} labelled requirements. Hybrid search has the best Recall@3; "
                   "embeddings the best MRR; most gaps between the top methods are within noise.")
    else:
        st.info("runs/v1/e1_results.json not found.")

    st.header("E2: claim verification")
    e2 = read_json(RUN / "e2_results.json")
    if e2:
        names = {"rule": "Rule check", "nli": "Zero-shot NLI", "finetuned": "Fine-tuned BERT",
                 "finetuned_nli": "Fine-tuned NLI", "llm": "LLM judge"}
        rows = []
        for m, label in names.items():
            row = {"Verifier": label}
            for s in ("standard", "hard"):
                x = e2.get(s, {}).get(m)
                row[f"{s.capitalize()} F1"] = f"{x['f1']:.3f}" if x else ""
                row[f"{s.capitalize()} false flags"] = pct(x["false_flag_rate"]) if x else ""
            rows.append(row)
        st.table(rows)
        st.caption("Standard set: 320 claims. Hard set: 80 claims, 28 drafted by a Qwen model. Fine-tuned NLI wins the "
                   "standard set and ties the LLM judge on the hard set; the score-based models wrongly flag 60-70% of "
                   "faithful claims that merge two bullets.")
    else:
        st.info("e2_results.json not found.")

    st.header("E3: tailoring study")
    summ = [r for r in read_csv(RUN / "e3_summary.csv") if r["fit"] == "all"]
    agent = read_json(RUN / "agent_metrics_summary.json")
    if summ:
        st.table([{"Variant": VARIANT_NAMES[r["variant"]], "Unsupported claims": pct(r["unsupported_rate"]),
                   "Resumes with invented tool/number": pct(r["any_introduced"], 0), "Coverage": pct(r["coverage"]),
                   "Task success": f"{agent.get(r['variant'], {}).get('success', 0):.2f}" if agent else "",
                   "LLM calls": f"{float(r['llm_calls']):.1f}", "Latency": f"{float(r['latency_s']):.0f} s",
                   "Cost/resume": f"${float(r['cost_usd']):.4f}"} for r in summ])
    fig = RUN / "figures" / "e3_tradeoff.svg"
    if fig.exists():
        b64 = base64.b64encode(fig.read_bytes()).decode()
        st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%"/>', unsafe_allow_html=True)
    st.caption("Grounding and verification both cut fabrication; verification costs requirement coverage and time. "
               "The pairwise preference judge could not be validated, so we make no claim about recruiter preference.")

    st.header("Probes")
    inj = [r for f in ("probe_injection_AB.csv", "probe_injection_C.csv", "probe_injection_D.csv") for r in read_csv(RUN / f)]
    if inj:
        rows = []
        for v in "ABCD":
            rs = [r for r in inj if r["variant"] == v]
            if rs:
                rows.append({"Variant": VARIANT_NAMES[v], "Resumes": len(rs),
                             "Fake certification added": f"{sum(int(r['attack_success']) for r in rs)} of {len(rs)}"})
        st.table(rows)
        st.caption("A hidden instruction in the job description asks the model to add a fake certification. "
                   "No name-based difference beyond noise in the name-swap probe (variant A).")


tabs = st.tabs(["Tailor", "Saved results", "Experiments"])
with tabs[0]:
    st.title("CareerCopilot")
    st.caption("Tailor a resume to a job without inventing experience.")
    tab_tailor()
with tabs[1]:
    tab_saved()
with tabs[2]:
    tab_experiments()
