"""Streamlit demo:  streamlit run app.py

Paste a resume and a job (or load a sample, or upload a PDF). The system runs variant D with hybrid retrieval and the
fine-tuned NLI verifier. It shows a score card (job match before and after, claims supported, invented facts), the
requirements with the evidence found for each, the tailored resume with a verdict per line, what the controller did
with every flagged line, and lets you accept or reject each line before downloading the final text.

CAREERCOPILOT_MOCK=1 runs it offline (fake LLM, BM25, rule check) for tests.
"""
import os
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from careercopilot.config import load_config, resolve  # noqa: E402
from careercopilot.data import load  # noqa: E402
from careercopilot.generation import format_evidence  # noqa: E402
from careercopilot.llm import build_llm  # noqa: E402
from careercopilot.pipeline import VARIANTS, run_variant  # noqa: E402
from careercopilot.retrieval import build_retriever  # noqa: E402
from careercopilot.schema import Pair  # noqa: E402
from careercopilot.scoring import score_card  # noqa: E402
from careercopilot.text import anonymise, load_vocab, split_with_context  # noqa: E402
from careercopilot.verifiers import build_verifier  # noqa: E402

st.set_page_config(page_title="CareerCopilot", layout="wide")
cfg = load_config()
samples = {p.pair_id: p for p in load(resolve(cfg["data"]["sample_pairs"]))}
VARIANT = "D"
OFFLINE = os.environ.get("CAREERCOPILOT_MOCK") == "1"
RETRIEVER, VERIFIER = ("bm25", "rule") if OFFLINE else (cfg["retrieval"]["method"], cfg["verifier"]["method"])


# ---------- helpers ----------

OUTCOME = {"accepted": "✅ repaired", "dropped": "🗑️ dropped", "escalated": "👤 sent to a person",
           "loop_limit": "⏹️ action limit reached, dropped"}




def render_trace(trace, title="What the controller did with each flagged line"):
    """The variant D controller's log, one block per line the verifier rejected (not nested: call outside expanders)."""
    if not trace:
        return
    with st.expander(f"{title} ({len(trace)})"):
        for t in trace:
            st.markdown(f"**Line {t['line_idx'] + 1}**: {OUTCOME.get(t['outcome'], t['outcome'])}")
            st.caption(f"Original: {t['initial_line']}")
            for s in t["steps"]:
                tool = s.get("tool")
                if tool == "controller":
                    note = f" (chosen by {s['source']}" + (f", replaced {s['overridden_from']}" if s.get("overridden_from") else "") + ")"
                    st.markdown(f"- Controller chose **{s['action']}**{note}: {s.get('reason') or 'no reason given'}")
                    for u in s.get("unsupported", []):
                        score = f" (score {u['score']})" if u.get("score") is not None else ""
                        st.caption(f"   not supported{score}: {u['claim']}")
                elif tool == "generator":
                    st.markdown(f"- Rewritten as: {s['line']}")
                elif tool == "retriever":
                    st.markdown(f"- Looked for more evidence ({len(s.get('new_evidence', []))} new bullets) for: {s['query']}")
                elif tool == "verifier":
                    st.markdown(f"- Verifier: {'all claims supported' if s['supported'] else 'still not supported'}")
            if t["outcome"] == "accepted" and t.get("final_line"):
                st.caption(f"Final line: {t['final_line']}")
            st.divider()

@st.cache_resource(show_spinner=False)
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












# ---------- tab 1: tailor ----------

def read_upload(f):
    """Text of an uploaded .pdf or .txt file ('' if it has none, e.g. a scanned PDF)."""
    if f.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(f).pages).strip()
    return f.getvalue().decode("utf-8", errors="ignore").strip()


def text_input(label, key, sample_text):
    """A text box that is filled by the chosen sample or an uploaded PDF/TXT file, and can still be edited."""
    f = st.file_uploader(f"{label}: upload a PDF or text file", type=["pdf", "txt"], key=f"{key}_file")
    source = (f.file_id if f else None, sample_text)
    if st.session_state.get(f"{key}_source") != source:   # the sample or the file changed: refill the box
        st.session_state[f"{key}_source"] = source
        text = sample_text
        if f:
            try:
                text = read_upload(f)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
            if not text:
                st.warning(f"No text found in {f.name} (a scanned PDF has no text layer). Paste the text instead.")
        st.session_state[key] = text
    return st.text_area(label, height=320, key=key)


def render_score(s):
    """Before/after score card: job match, faithfulness and invented facts for the default accepted lines."""
    if not s:
        return
    n = len(s["requirements"])
    st.subheader("Score card")
    a, b, c = st.columns(3)
    a.metric("Job match (requirements covered)", f"{len(s['after'])} of {n}", f"{len(s['after']) - len(s['before']):+d} vs original ({len(s['before'])} of {n})")
    b.metric("Claims supported by your resume", f"{s['supported']} of {s['claims']}" if s["claims"] else "n/a",
             f"{s['flagged_lines']} flagged line(s) repaired or removed" if s["flagged_lines"] else "no lines flagged", delta_color="off")
    invented = s["new_tools"] + s["new_numbers"]
    c.metric("Invented tools or numbers", len(invented), ", ".join(invented) if invented else "none found", delta_color="off")
    rows = [{"Requirement": r, "Original resume": "✅" if i in s["before"] else "–", "Tailored resume": "✅" if i in s["after"] else "–"}
            for i, r in enumerate(s["requirements"])]
    with st.expander("Which requirements are covered"):
        st.table(rows)
    st.caption("Job match is judged by an LLM asking which requirements the original resume's bullets or the tailored lines clearly address. "
               "Faithfulness is the fine-tuned verifier's verdict on the tailored claims. Invented facts are tools and numbers "
               "that appear in the tailored text but nowhere in your resume. The score covers the lines accepted by default, "
               "and tailoring is built to be faithful first: job match can stay level or dip when a line cannot be supported.")


def render_result(res, pair, variant):
    id_map = {}
    if VARIANTS[variant]["grounded"]:
        _, id_map = format_evidence(res.requirements, res.evidence, pair.bullets)   # E-id -> bullet index
    claims_by_line = {}
    for c in res.pipeline_claims:
        claims_by_line.setdefault(c.line_idx, []).append(c)

    render_score(st.session_state.get("score"))
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
        if getattr(res, "escalated_lines", None):
            why = [t["steps"][-1].get("reason") or "" for t in res.trace if t["outcome"] == "escalated" and t["steps"]]
            st.info("**Needs your review** (left out of the text; the verifier was close to accepting it):\n\n" +
                    "\n".join(f"- {l}" + (f"  \n  _{why[k]}_" if k < len(why) and why[k] else "")
                              for k, l in enumerate(res.escalated_lines)))
        u = res.usage
        st.caption(f"{u.llm_calls} LLM calls · {u.tokens_in + u.tokens_out} tokens · {u.latency_s:.1f}s · "
                   f"${u.cost_usd:.4f} · {res.regenerations} regenerations")
        if getattr(res, "actions", None):
            st.caption("Controller actions: " + ", ".join(f"{a.lower().replace('_', ' ')} {n}" for a, n in sorted(res.actions.items())))

    render_trace(getattr(res, "trace", []))

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
    sample = st.selectbox("Load a sample", ["(none)"] + list(samples))
    pre = samples.get(sample)
    c1, c2 = st.columns(2)
    with c1:
        resume = text_input("Resume", "resume", pre.resume if pre else "")
    with c2:
        job_text = text_input("Job description", "job", pre.job if pre else "")
    name = st.text_input("Your name (optional)", help="Removed from the resume before any text is sent to the LLM. "
                         "Emails, phone numbers and URLs are always removed.")
    st.info("Every claim in the tailored resume is verified, and the controller works on each flagged line, "
            "so a run takes several minutes (about 5 minutes on average in our study).")

    job = st.session_state.get("tailor_job")
    running = bool(job) and job["state"] == "running"
    if st.button("Tailor", type="primary", disabled=running or not (resume.strip() and job_text.strip())):
        resume_a = anonymise(redact_name(resume, name) if name.strip() else resume)
        bullets, contexts = split_with_context(resume_a)
        pair = Pair(pair_id="ui", resume=resume_a, job=anonymise(job_text), fit="unknown", bullets=bullets, contexts=contexts)
        job = {"state": "running", "start": time.time()}
        st.session_state["tailor_job"] = job
        st.session_state.pop("last", None)
        st.session_state.pop("score", None)
        threading.Thread(target=run_job, args=(job, pair), daemon=True).start()
        running = True
    if running:
        tailoring_status(job)
        return
    if job and job["state"] == "error":
        st.error(f"The pipeline failed: {job['error']}")
        st.caption("Missing key? Put GROQ_API_KEY in .env. Missing verifier model? Put models/verifier-nli-ft next to the repo.")
    elif job and job["state"] == "done":
        st.session_state["last"] = (job["res"], job["pair"], VARIANT)
        st.session_state["score"] = job["score"]
        st.session_state.pop("tailor_job")
    if "last" in st.session_state:
        render_result(*st.session_state["last"])


def run_job(job, pair):
    """The pipeline, in a background thread so the page (and the Experiments tab) stays usable while it runs."""
    try:
        llm, retriever, verifier = components(RETRIEVER, VERIFIER, OFFLINE)
        res = run_variant(pair, VARIANT, llm, retriever, verifier, cfg)
        terms = [x for ts in load_vocab(resolve(cfg["paths"]["skills_vocab"])).values() for x in ts]
        job.update(res=res, pair=pair, score=score_card(llm, pair, res, terms), state="done")
    except Exception as e:   # a missing model folder, a missing key or a network error
        job.update(state="error", error=f"{type(e).__name__}: {e}")


@st.fragment(run_every=2)
def tailoring_status(job):
    if job["state"] != "running":
        st.rerun()
    st.info(f"Tailoring... {time.time() - job['start']:.0f} s so far (about 5 minutes on average). "
            "You can open the Experiments tab while you wait; the result appears here when it is done.")


st.title("CareerCopilot")
st.caption("Tailor a resume to a job without inventing experience.")
tab_tailor()
