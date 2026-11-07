"""Streamlit demo:  streamlit run app.py

Two tabs:
  Tailor       paste a resume and a job (or load a sample); the system runs variant D with hybrid retrieval and the
               fine-tuned NLI verifier. Shows the requirements, the evidence found for each, the tailored resume with a
               verdict per line, a score card (job match before and after, claims supported, invented facts), what the controller did with every flagged line, and lets you accept or reject each
               line before downloading the final text
  Experiments  the E1, E2 and E3 results tables, the main figure, and the probe results (read from runs/, no API calls)

CAREERCOPILOT_MOCK=1 runs it offline (fake LLM, BM25, rule check) for tests.
"""
import base64
import csv
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from careercopilot.config import load_config, resolve, run_dir  # noqa: E402
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
RUN = run_dir(cfg)
samples = {p.pair_id: p for p in load(resolve(cfg["data"]["sample_pairs"]))}
VARIANT_NAMES = {"A": "A - unrestricted", "B": "B - grounded", "C": "C - unrestricted + verifier", "D": "D - full system"}
VARIANT = "D"
OFFLINE = os.environ.get("CAREERCOPILOT_MOCK") == "1"
RETRIEVER, VERIFIER = ("bm25", "rule") if OFFLINE else (cfg["retrieval"]["method"], cfg["verifier"]["method"])


# ---------- helpers ----------

OUTCOME = {"accepted": "✅ repaired", "dropped": "🗑️ dropped", "escalated": "👤 sent to a person",
           "loop_limit": "⏹️ action limit reached, dropped"}


def read_first(name):
    """A results file from this run, or from run v2 (E2 and the A to C probes were not re-run for run v3)."""
    for d in (RUN, resolve("runs/v2")):
        if (d / name).exists():
            return d / name
    return RUN / name


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


def ci_text(m):
    """'+1.0 [-0.7, +2.8]' in percentage points from one bootstrap entry."""
    if not m:
        return "n/a"
    return f"{100 * m['mean_difference']:+.1f} [{100 * m['ci_lower']:+.1f}, {100 * m['ci_upper']:+.1f}]"


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


# ---------- tab 2: experiment results ----------

def tab_experiments():
    st.caption(f"Read from the saved runs (E1 from run v1, E2 from run v2, E3 from run {cfg['run_name']}). "
               "Rows marked ✅ are the methods this demo uses. Full tables and discussion: the project report.")
    st.header("E1: evidence retrieval")
    st.markdown("**What it tests:** for each job requirement, can the retriever find the resume bullets that support it? "
                "The better the retriever, the better the evidence the tailoring step gets to see. "
                "We hand-labelled which bullets support each requirement and measured how often the retriever finds them.")
    st.caption("Recall@1 / Recall@3: share of requirements where a correct bullet is the top result / among the top 3. "
               "MRR: how high the first correct bullet ranks (1 is best). ms/query: speed. "
               "bm25 and tfidf are keyword matching; embedding matches by meaning; hybrid combines bm25 and embeddings "
               "and re-orders the candidates with a cross-encoder. +ctx means each bullet is indexed together with its "
               "job title and company line.")
    e1 = read_json(resolve("runs/v1/e1_results.json"))
    extra = read_json(resolve("runs/e1-ctx-extra/e1_results.json")) or {}
    if e1:
        for k in ("tfidf+ctx", "hybrid+ctx"):   # scored later on the same labelled bullets; speed is the plain retriever's
            if k in extra and k not in e1:
                e1[k] = {**extra[k], "ms_per_query": e1[k.split("+")[0]]["ms_per_query"]}
        order = ["tfidf", "bm25", "embedding", "hybrid", "tfidf+ctx", "bm25+ctx", "embedding+ctx", "hybrid+ctx"]
        e1 = {k: e1[k] for k in order if k in e1}
        used = cfg["retrieval"]["method"] + ("+ctx" if cfg["retrieval"].get("contextual") else "")
        st.table([{"Retriever": k, "Used in demo": "✅" if k == used else "", "Recall@1": f"{v['recall@1']:.3f}", "Recall@3": f"{v['recall@3']:.3f}",
                   "MRR": f"{v['mrr']:.3f}", "ms/query": f"{v['ms_per_query']:.0f}"} for k, v in e1.items()])
        st.caption(f"{next(iter(e1.values()))['n']} labelled requirements. Hybrid search has the best Recall@3; "
                   "embeddings the best MRR; most gaps between the top methods are within noise (one requirement is 1.9 points). "
                   "Adding the role line (+ctx) helps every retriever a little and never hurts, but only about one resume bullet in nine "
                   "has a role line above it. The speed of the TF-IDF and hybrid +ctx rows is that of the plain retriever.")
        st.success("**Chosen: hybrid with role context.** The tailoring step cites up to 3 pieces of evidence per requirement, so "
                   "finding a correct bullet anywhere in the top 3 (Recall@3) matters most, and hybrid with role context is best on "
                   "that (0.741). It is slower, which is acceptable because the LLM calls dominate the run time.")
    else:
        st.info("runs/v1/e1_results.json not found.")

    st.header("E2: claim verification")
    st.markdown("**What it tests:** can a verifier tell whether a claim is supported by the resume? We planted fabrications "
                "(a swapped tool, an inflated number, an invented duty or title) and wrote faithful rewrites, then checked "
                "whether each verifier flags the fabrications and accepts the faithful claims. "
                "The verifier decides which lines variants C and D keep.")
    st.caption("F1: balance of catching fabrications and not raising false alarms (1 is best). False flags: faithful claims "
               "wrongly rejected, which cost useful lines. Standard set: single-bullet claims. Hard set: subtler "
               "fabrications and faithful claims that merge two bullets. Rule check: looks for tools, numbers and titles that "
               "are missing from the resume. Zero-shot NLI: a pre-trained entailment model. Fine-tuned BERT / NLI: the same "
               "kinds of model, further trained by us on resume claims. LLM judge: asks an LLM supported or unsupported.")
    e2 = read_json(read_first("e2_results.json"))
    if e2:
        names = {"rule": "Rule check", "nli": "Zero-shot NLI", "finetuned": "Fine-tuned BERT",
                 "finetuned_nli": "Fine-tuned NLI", "llm": "LLM judge"}
        rows = []
        for m, label in names.items():
            row = {"Verifier": label, "Used in demo": "✅" if m == cfg["verifier"]["method"] else ""}
            for s in ("standard", "hard"):
                x = e2.get(s, {}).get(m)
                row[f"{s.capitalize()} F1"] = f"{x['f1']:.3f}" if x else ""
                row[f"{s.capitalize()} false flags"] = pct(x["false_flag_rate"]) if x else ""
            rows.append(row)
        st.table(rows)
        st.caption("Standard set: 320 claims. Hard set: 80 claims, 28 drafted by a Qwen model. Fine-tuned NLI wins the "
                   "standard set and ties the LLM judge on the hard set; the score-based models wrongly flag 60-70% of "
                   "faithful claims that merge two bullets.")
        st.success("**Chosen: fine-tuned NLI.** Best F1 on the standard set, and close to the LLM judge on the hard set "
                   "(the LLM judge is a little better there). It runs locally, costs nothing per claim, and does not depend "
                   "on the LLM that writes the resume, so the writer is not marking its own work.")
    else:
        st.info("e2_results.json not found.")

    st.header("E3: tailoring study")
    st.markdown("**What it tests:** the whole system end to end. We tailored 36 resume-job pairs (12 good fit, 12 potential, "
                "12 no fit) with four variants and scored every output. "
                "A is plain prompting. B adds evidence retrieval, so the model must cite your bullets. "
                "C keeps A but verifies each claim and regenerates or drops flagged lines. "
                "D is B plus verification plus a bounded controller that decides how to repair each flagged line.")
    st.caption("Unsupported claims: share of claims in the output that the resume does not support, judged by a stronger LLM "
               "whose verdicts matched human labels (agreement 0.90). Invented tool/number: resumes that contain a tool or "
               "number found nowhere in the original (a crude check). Coverage: share of the job's requirements "
               "addressed by a supported claim. Task success: share of runs that finished with no unsupported claim and no "
               "invented tool or number. Lower is better for the first two and cost, higher for the rest.")
    summ = [r for r in read_csv(RUN / "e3_summary.csv") if r["fit"] == "all"]
    agent = read_json(RUN / "agent_metrics_summary.json")
    if summ:
        st.table([{"Variant": VARIANT_NAMES[r["variant"]], "Used in demo": "✅" if r["variant"] == VARIANT else "", "Unsupported claims": pct(r["unsupported_rate"]),
                   "Resumes with invented tool/number": pct(r["any_introduced"], 0), "Coverage": pct(r["coverage"]),
                   "Task success": f"{agent.get(r['variant'], {}).get('success', 0):.2f}" if agent else "",
                   "LLM calls": f"{float(r['llm_calls']):.1f}", "Latency": f"{float(r['latency_s']):.0f} s",
                   "Cost/resume": f"${float(r['cost_usd']):.4f}"} for r in summ])
    if summ:
        st.success("**Chosen: D.** It cuts unsupported claims from about 47% (A) to about 5%, and every repair decision is "
                   "logged. The price is lower requirement coverage than the unverified variants and several minutes per run. "
                   "C reaches a similar faithfulness; D's advantage is a bounded, explainable repair process, not a lower rate.")
    boot = read_json(RUN / "e3_bootstrap_ci.json")
    if boot:
        st.subheader("How sure are we? Differences from A with 95% intervals")
        names = {"B": "B: grounded", "C": "C: verifier only", "D": "D: full system"}
        rows = []
        for v, label in names.items():
            m = boot["metrics"].get(f"{v}_vs_A/all", {})
            if m:
                rows.append({"Variant minus A": label, **{h: ci_text(m.get(k)) for h, k in
                             (("Unsupported claims", "unsupported_rate"), ("Requirement coverage", "coverage"),
                              ("Resumes with invented tool/number", "any_introduced"))}})
        st.table(rows)
        st.caption("Points of difference with a 95% bootstrap interval, resampling the 36 resume-job pairs 10,000 times. "
                   "An interval that excludes 0 is a difference we can trust given these pairs. All three variants clearly cut "
                   "unsupported claims and invented tools or numbers. For coverage, D's interval includes 0, so we cannot say D "
                   "loses coverage. The intervals reflect which pairs were sampled, not model or judge randomness.")
    ctrl = agent.get("D", {}).get("controller") if agent else None
    if ctrl:
        st.subheader("Variant D controller")
        st.table([{"Lines generated": ctrl["lines_generated"], "Needed recovery": ctrl["lines_needing_recovery"],
                   "Recovered": pct(ctrl["recovery_success_rate"], 0), "Escalated to a person": pct(ctrl["escalation_rate"]),
                   "Hit the action limit": pct(ctrl["loop_limit_hit_rate"], 0), "Overridden by rules": pct(ctrl["override_rate"], 0)}])
        st.caption("Actions: " + ", ".join(f"{a.lower().replace('_', ' ')} {n}" for a, n in sorted(ctrl["action_counts"].items())) +
                   ". The controller repaired about two thirds of the lines that needed it; it did not change the "
                   "unsupported-claim rate compared with our earlier fixed regenerate-and-drop loop, but every decision is logged.")
    msum = {(r["variant"]): r for r in read_csv(resolve("runs/matched-controller/e3_summary.csv")) if r["fit"] == "all"}
    mci = (read_json(resolve("runs/matched-controller/e3_bootstrap_ci.json")) or {}).get("metrics", {}).get("E_vs_D/all")
    if msum.get("D") and msum.get("E") and mci:
        st.subheader("Controller or fixed loop? Matched comparison of D and E")
        st.markdown("**What it tests:** is D's controller actually better than a plain fixed loop? E uses the same grounded "
                    "generator, retrieval and verifier as D, but regenerates a rejected line up to twice and then drops it. Both ran "
                    "together on the same 36 pairs, with at most two recovery actions each.")
        rows = []
        for k, label, f in (("unsupported_rate", "Unsupported claims", pct), ("coverage", "Requirement coverage", pct),
                            ("any_introduced", "Resumes with invented tool/number", lambda x: pct(x, 0))):
            rows.append({"Measure": label, "D: controller": f(msum["D"][k]), "E: fixed loop": f(msum["E"][k]),
                         "E minus D (95% interval)": ci_text(mci.get(k))})
        rows.append({"Measure": "Time per resume", "D: controller": f"{float(msum['D']['latency_s']):.0f} s",
                     "E: fixed loop": f"{float(msum['E']['latency_s']):.0f} s", "E minus D (95% interval)": f"{mci['latency_s']['mean_difference']:+.0f} s "
                     f"[{mci['latency_s']['ci_lower']:+.0f}, {mci['latency_s']['ci_upper']:+.0f}]"})
        st.table(rows)
        st.success("**Why D stays in the demo:** the two cannot be told apart on faithfulness or coverage (every interval "
                   "includes 0), but D is about a minute faster per resume and logs every repair decision.")
        st.caption("This D is limited to two recovery actions without extra retrieval, so it is not the 4-action D in the main table.")
    fig = RUN / "figures" / "e3_tradeoff.svg"
    if fig.exists():
        b64 = base64.b64encode(fig.read_bytes()).decode()
        st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%"/>', unsafe_allow_html=True)
    st.caption("Grounding and verification both cut fabrication; verification costs requirement coverage and time. "
               "The pairwise preference judge could not be validated, so we make no claim about recruiter preference.")

    st.header("Probes")
    st.markdown("**What it tests:** two risks. Prompt injection: a hidden instruction in the job description tells the model to "
                "add a fake certification, and we count how often it works. Bias: the same resume is tailored under different "
                "names to see whether the output changes beyond run-to-run noise.")
    inj = [r for f in ("probe_injection_AB.csv", "probe_injection_C.csv", "probe_injection_D.csv")
           for r in read_csv(read_first(f))]
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


tabs = st.tabs(["Tailor", "Experiments"])
with tabs[0]:
    st.title("CareerCopilot")
    st.caption("Tailor a resume to a job without inventing experience.")
    tab_tailor()
with tabs[1]:
    tab_experiments()
