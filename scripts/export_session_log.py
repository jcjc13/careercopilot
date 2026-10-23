"""Turn a Claude Code session transcript (.jsonl) into a readable markdown log of the process: every user
message, assistant reply, tool call (name + input) and tool result, with timestamps. Long tool inputs and
results are truncated (--max-chars) so the log stays readable; the .jsonl keeps everything.

    python scripts/export_session_log.py <session.jsonl> <out.md> [--max-chars 1500]
Redacts anything that looks like an API key/token. Review the output before sharing it.
"""
import argparse
import json
import re

SECRET = re.compile(r"(ghp_|github_pat_|gsk_|sk-|hf_[A-Za-z0-9]{20})[A-Za-z0-9_\-]{10,}")


def clip(s, n):
    s = SECRET.sub("[REDACTED]", str(s))
    return s if len(s) <= n else s[:n] + f"\n... [{len(s) - n} more chars truncated]"


def block(text, n, lang=""):
    return f"```{lang}\n{clip(text, n)}\n```"


def result_text(c):
    if isinstance(c, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return str(c)


def session_metrics(src):
    """Per-task metrics of the working session. A task = one user message; its steps are the tool calls that
    follow until the next user message. Errors = tool results flagged is_error. Time = wall clock to next task."""
    from datetime import datetime

    def t(ts):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    tasks, cur, names = [], None, {}
    for l in open(src, encoding="utf-8"):
        d = json.loads(l)
        ty, msg = d.get("type"), d.get("message") or {}
        c = msg.get("content")
        if ty == "user":
            texts = [c] if isinstance(c, str) else [x.get("text", "") for x in c or [] if x.get("type") == "text"]
            for x in (c if isinstance(c, list) else []):
                if x.get("type") == "tool_result" and cur is not None:
                    cur["results"] += 1
                    cur["errors"] += bool(x.get("is_error"))
                    cur["denied"] += "denied" in result_text(x.get("content")).lower()
            if any(x.strip() for x in texts) and not d.get("isSidechain"):
                cur = {"start": d["timestamp"], "end": d["timestamp"], "prompt": " ".join(texts)[:80], "calls": 0,
                       "results": 0, "errors": 0, "denied": 0, "tokens_in": 0, "tokens_out": 0, "tools": {}}
                tasks.append(cur)
        elif ty == "assistant" and cur is not None:
            cur["end"] = d["timestamp"]
            u = msg.get("usage") or {}
            cur["tokens_in"] += u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
            cur["tokens_out"] += u.get("output_tokens", 0)
            for x in c or []:
                if x.get("type") == "tool_use":
                    cur["calls"] += 1
                    cur["tools"][x["name"]] = cur["tools"].get(x["name"], 0) + 1
    for x in tasks:
        x["seconds"] = round((t(x["end"]) - t(x["start"])).total_seconds(), 1)
    return tasks


def metrics_md(tasks):
    calls = sum(x["calls"] for x in tasks)
    errs = sum(x["errors"] for x in tasks)
    tools = {}
    for x in tasks:
        for k, v in x["tools"].items():
            tools[k] = tools.get(k, 0) + v
    out = ["\n# Session metrics (per task = per user message)\n",
           f"Tasks: {len(tasks)}; tool calls: {calls}; tool errors: {errs} "
           f"(tool-call success rate {1 - errs / calls:.1%}); permission denials: {sum(x['denied'] for x in tasks)}; "
           f"tokens in/out (summed over assistant turns, cache reads included): "
           f"{sum(x['tokens_in'] for x in tasks):,} / {sum(x['tokens_out'] for x in tasks):,}\n",
           "Tool calls by tool: " + ", ".join(f"{k} {v}" for k, v in sorted(tools.items(), key=lambda kv: -kv[1])) + "\n",
           "| # | task (first 80 chars) | tool calls | errors | seconds | tokens out |", "|---|---|---|---|---|---|"]
    for i, x in enumerate(tasks, 1):
        out.append(f"| {i} | {x['prompt'].replace('|', '/').replace(chr(10), ' ')} | {x['calls']} | {x['errors']} | "
                   f"{x['seconds']} | {x['tokens_out']} |")
    return "\n".join(out) + "\n"


def main(src, out, n):
    lines = ["# Session log\n", f"Source: `{src}`\n"]
    calls = 0
    for l in open(src, encoding="utf-8"):
        d = json.loads(l)
        t, ts = d.get("type"), (d.get("timestamp") or "")[:19].replace("T", " ")
        msg = d.get("message") or {}
        content = msg.get("content")
        if t == "user":
            if isinstance(content, str):
                lines.append(f"\n## {ts} USER\n\n{clip(content, n * 3)}\n")
            else:
                for c in content or []:
                    if c.get("type") == "text":
                        lines.append(f"\n## {ts} USER\n\n{clip(c['text'], n * 3)}\n")
                    elif c.get("type") == "tool_result":
                        lines.append(f"**result**\n{block(result_text(c.get('content')), n)}\n")
        elif t == "assistant":
            for c in content or []:
                if c.get("type") == "text" and c["text"].strip():
                    lines.append(f"\n## {ts} ASSISTANT\n\n{clip(c['text'], n * 3)}\n")
                elif c.get("type") == "tool_use":
                    calls += 1
                    inp = c.get("input", {})
                    lines.append(f"\n### {ts} TOOL CALL #{calls}: `{c['name']}`\n")
                    for k, v in inp.items():
                        lines.append(f"- **{k}**:\n{block(v if isinstance(v, str) else json.dumps(v, indent=1), n)}")
    tasks = session_metrics(src)
    lines.append(metrics_md(tasks))
    json.dump(tasks, open(out.rsplit(".", 1)[0] + "_metrics.json", "w"), indent=1)
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"wrote {out}: {calls} tool calls")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--max-chars", type=int, default=1500)
    a = p.parse_args()
    main(a.src, a.out, a.max_chars)
