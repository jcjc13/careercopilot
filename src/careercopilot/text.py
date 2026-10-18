"""Text utilities: anonymisation, splitting a resume into evidence bullets, tokenising."""
from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S*", re.I)
_BULLET_CHARS = "•●▪◦‣∙·–-*"


def anonymise(text: str) -> str:
    """Strip emails, phone numbers and URLs. Names are NOT reliably removable by regex --
    the public dataset is already largely anonymised; check a sample by eye."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    text = _PHONE.sub("[PHONE]", text)
    return text


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s*(?=[A-Z])", text) if s.strip()]


_SECTION_HEADINGS = (
    "Career Overview", "Professional Summary", "Work Experience",
    "Professional Experience", "Employment History", "Technical Skills",
    "Education", "Certifications", "Projects", "Summary", "Experience",
    "Skills", "Awards",
)


def _separate_headings(text: str) -> str:
    headings = "|".join(re.escape(heading) for heading in
                        sorted(_SECTION_HEADINGS, key=len, reverse=True))
    # case-sensitive and only when glued to a capital ("SummaryHighly..."); a case-insensitive match
    # also deleted ordinary words like "experience" and "skills" from sentences
    return re.sub(rf"(?<![A-Za-z])({headings})(?=[A-Z])", r"\n\1\n", text)


def _limit_chars(text: str, max_chars: int) -> list[str]:
    """Split oversized text at word boundaries without exceeding max_chars."""
    words = text.split()
    chunks, current = [], ""
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def _is_header(line: str) -> bool:
    """A role/company/education line such as 'Data Analyst, Retail Co (2023-2025)'."""
    return bool(_YEAR.search(line)) and len(line.split()) <= 12


def split_with_context(resume: str, min_words: int = 4, max_chars: int = 300) -> tuple[list[str], list[str]]:
    """Split a resume into short evidence units, each with its context line.

    Uses line breaks and bullet characters first; long lines (e.g. resumes stored as one
    blob of text) are further split into sentences. Very short lines are dropped.
    The context of a bullet is the most recent role/company line above it (a short line
    with a year) -- prepended at retrieval time for contextual RAG (Topic 8): 'Built
    dashboards' alone loses the job it was done in. Deterministic, so ids are stable.
    """
    pieces, ctx = [], ""
    resume = _separate_headings(resume)
    for line in re.split(r"[\n\r]+", resume):
        clean = re.sub(r"\s+", " ", line).strip(" \t" + _BULLET_CHARS)
        if _is_header(clean):
            ctx = clean
            pieces.append((clean, ""))
            continue
        for part in re.split(rf"\s[{re.escape(_BULLET_CHARS)}]\s|^[{re.escape(_BULLET_CHARS)}]\s*", line):
            part = (part or "").strip(" \t" + _BULLET_CHARS)
            if not part:
                continue
            for sentence in _sentences(part):
                pieces.extend((chunk, ctx) for chunk in _limit_chars(sentence, max_chars))
    bullets, contexts, seen = [], [], set()
    for p, c in pieces:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p.split()) < min_words or p.isupper():
            continue
        key = p.lower()
        if key not in seen:
            seen.add(key)
            bullets.append(p)
            contexts.append(c)
    return bullets, contexts


def split_bullets(resume: str, min_words: int = 4, max_chars: int = 300) -> list[str]:
    return split_with_context(resume, min_words, max_chars)[0]


_TOKEN = re.compile(r"[a-z0-9+#.]+")
_STOP = set("a an the of in on at to for and or is was were are be by with as from that this "
            "which who what when where how it its our we you your i my me their they".split())


def tokenize(text: str) -> list[str]:
    return [t.strip(".") for t in _TOKEN.findall(text.lower()) if t.strip(".") and t not in _STOP]


def load_vocab(path) -> dict[str, list[str]]:
    """skills.txt: one category per line, `category: term, term, ...`."""
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            cat, terms = line.split(":", 1)
            vocab[cat.strip()] = [t.strip() for t in terms.split(",") if t.strip()]
    return vocab


def find_terms(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    found = []
    for t in terms:
        if re.search(rf"(?<![\w+#]){re.escape(t.lower())}(?![\w+#])", low):
            found.append(t)
    return found


NUMBER = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?(?:\s?%|\+|(?:k|m|bn)\b)?", re.I)


def find_numbers(text: str) -> list[str]:
    return [n.strip().lower().replace(",", "").replace(" ", "") for n in NUMBER.findall(text)]
