"""OpenAI-compatible chat client that records tokens, latency and cost, plus a
deterministic MockLLM so every script can be dry-run offline with --mock.

Works with Groq, OpenAI, OpenRouter, Together, vLLM and a local
Ollama server -- set base_url / api_key_env per model in the config.
"""
from __future__ import annotations

import json
import os
import re
import time

from .schema import Usage


class LLM:
    def __init__(self, model: str, temperature: float = 0.0, price_in: float = 0.0,
                 price_out: float = 0.0, seed: int | None = None, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", reasoning_effort: str | None = None):
        from openai import OpenAI
        key = os.getenv(api_key_env) or ("ollama" if base_url and "11434" in base_url else None)
        if not key:
            raise SystemExit(f"{api_key_env} is not set -- add it to .env (see .env.example)")
        self.client = OpenAI(base_url=base_url or None, api_key=key)
        self.model, self.temperature, self.seed = model, temperature, seed
        self.price_in, self.price_out = price_in, price_out
        self.reasoning_effort = reasoning_effort

    def chat(self, system: str, user: str, max_tokens: int = 800) -> tuple[str, Usage]:
        kwargs = dict(model=self.model, temperature=self.temperature, seed=self.seed,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}])
        if "gpt-oss" in self.model or self.reasoning_effort:
            # reasoning models spend hidden tokens before answering: a tiny cap gives an empty reply
            kwargs["max_tokens"] = max(max_tokens, 1024)
            if self.reasoning_effort:
                kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        else:
            kwargs["max_tokens"] = max_tokens
        t0 = time.perf_counter()
        for attempt in range(5):  # retry rate limits (Groq free tier) and transient errors
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt * 3
                print(f"[llm] {type(e).__name__}: retrying in {wait}s")
                time.sleep(wait)
        dt = time.perf_counter() - t0
        u = resp.usage
        tin, tout = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
        cost = (tin * self.price_in + tout * self.price_out) / 1e6
        return ((resp.choices[0].message.content or "").strip(),
                Usage(llm_calls=1, tokens_in=tin, tokens_out=tout, latency_s=dt, cost_usd=cost))


def build_llm(cfg: dict, which: str = "llm", mock: bool = False):
    if mock:
        return MockLLM()
    c = cfg[which]
    return LLM(c["model"], c.get("temperature", 0.0), c.get("price_in", 0.0), c.get("price_out", 0.0),
               cfg["seed"], c.get("base_url"), c.get("api_key_env", "OPENAI_API_KEY"),
               c.get("reasoning_effort"))


# ---------------- parsing helpers ----------------

def parse_json(text: str):
    """Pull the first JSON object/array out of an LLM reply (tolerates ``` fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    starts = sorted((i, c) for c in "[{" if (i := text.find(c)) != -1)  # outermost bracket first
    for start, opener in starts:
        closer = "]" if opener == "[" else "}"
        end = text.rfind(closer)
        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in reply: {text[:200]!r}")


def task_of(system: str) -> str:
    """Every prompt in this repo starts with 'Task: <name>.' -- used by MockLLM and logs."""
    m = re.match(r"Task:\s*([\w-]+)", system)
    return m.group(1) if m else "unknown"


# ---------------- offline mock ----------------

class MockLLM:
    """Deterministic, rule-based stand-in for an LLM. Its outputs are NOT meaningful --
    it only lets you run every script end to end without an API key. Its unrestricted
    rewrite deliberately adds one fabrication so the verifiers have something to catch."""

    model = "mock"

    def chat(self, system: str, user: str, max_tokens: int = 800) -> tuple[str, Usage]:
        task = task_of(system)
        out = getattr(self, f"_{task.replace('-', '_')}", self._unknown)(user)
        return out, Usage(llm_calls=1, tokens_in=len((system + user).split()),
                          tokens_out=len(out.split()))

    @staticmethod
    def _section(user: str, header: str) -> str:
        m = re.search(rf"<{header}>\n(.*?)\n</{header}>", user, re.S)
        return m.group(1) if m else ""

    def _requirements(self, user):
        job = self._section(user, "job")
        lines = [l.strip(" -•*") for l in re.split(r"[\n.;]", job) if len(l.split()) >= 3]
        return json.dumps(lines[:8] or ["general experience"])

    def _tailor_grounded(self, user):
        ev = re.findall(r"^\[(E\d+)\] (.+)$", self._section(user, "evidence"), re.M)
        bullets = [f"{t} [{e}]" for e, t in ev[:6]]
        return json.dumps({"summary": "Analyst with experience matching the role.", "bullets": bullets})

    def _tailor_unrestricted(self, user):
        resume = self._section(user, "resume")
        lines = [l.strip(" -•*") for l in resume.splitlines() if len(l.split()) >= 4][:5]
        lines.append("Built Tableau dashboards used by 500 stakeholders")  # planted fabrication
        return json.dumps({"summary": "Experienced analyst ready for this role.", "bullets": lines})

    def _regenerate(self, user):
        return self._section(user, "line").replace("Tableau", "Power BI").replace(" used by 500 stakeholders", "")

    def _rewrite_conservative(self, user):
        return self._regenerate(user)

    def _controller(self, user):
        """Mock controller: rewrite while allowed, else drop (so the loop and its limits are exercised offline)."""
        m = re.search(r'"allowed_actions":\s*\[(.*?)\]', user, re.S)
        allowed = re.findall(r'"(\w+)"', m.group(1)) if m else []
        action = next((a for a in ("REWRITE_CONSERVATIVELY", "DROP") if a in allowed), "HUMAN_REVIEW")
        return json.dumps({"action": action, "failure_type": "unsupported_embellishment", "reason": "mock",
                           "instructions": "remove unsupported details", "recovery_query": None, "confidence": 0.5})

    def _split_claims(self, user):
        lines = re.findall(r"^\[(\d+)\] (.+)$", self._section(user, "lines"), re.M)
        return json.dumps([{"line": int(i), "claim": t} for i, t in lines])

    def _verify(self, user):
        claim = self._section(user, "claim").lower()
        resume = self._section(user, "resume").lower()
        words = [w for w in re.findall(r"[a-z0-9]+", claim) if len(w) > 3]
        hit = sum(w in resume for w in words) / max(len(words), 1)
        return "SUPPORTED" if hit >= 0.85 else "UNSUPPORTED"

    def _coverage(self, user):
        n = len(re.findall(r"^\[\d+\]", self._section(user, "requirements"), re.M))
        return json.dumps(list(range(1, n // 2 + 1)))

    def _pairwise(self, user):
        first, second = self._section(user, "resume_1"), self._section(user, "resume_2")
        if "Tableau" in first and "Tableau" not in second:
            return '{"winner": "2"}'
        if "Tableau" in second and "Tableau" not in first:
            return '{"winner": "1"}'
        return '{"winner": "1"}' if len(first) >= len(second) else '{"winner": "2"}'  # a biased mock, on purpose

    def _paraphrase(self, user):
        return "In my role, " + self._section(user, "bullet").rstrip(".") + "."

    def _fabricate(self, user):
        return self._section(user, "bullet").rstrip(".") + ", leading a team of 6 engineers."

    def _unknown(self, user):
        return ""
