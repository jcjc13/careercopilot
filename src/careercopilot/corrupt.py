"""Scripted claim corruptions and faithful sub-claims, shared by the E2 test set
(make_fabrications.py) and the verifier's fine-tuning set (make_verifier_train.py)."""
from __future__ import annotations

import re

from .text import NUMBER, find_terms


def swap_tool(bullet: str, resume: str, vocab: dict, rng) -> str | None:
    """Replace one known tool with another from the same category that the resume never mentions."""
    for cat, terms in vocab.items():
        if cat == "seniority":
            continue
        present = find_terms(bullet, terms)
        if present:
            old = present[0]
            options = [t for t in terms if t not in find_terms(resume, terms)]
            if options:
                new = rng.choice(options)
                return re.sub(rf"(?<![\w+#]){re.escape(old)}(?![\w+#])", new, bullet, count=1, flags=re.I)
    return None


def change_number(bullet: str, rng) -> str | None:
    """Inflate the first number in the bullet (x3, x5 or x10)."""
    m = NUMBER.search(bullet)
    if not m:
        return None
    raw = m.group(0)
    digits = re.sub(r"[^\d.]", "", raw)
    try:
        val = float(digits)
    except ValueError:
        return None
    new = val * rng.choice([3, 5, 10]) if val > 0 else 25
    new_s = str(int(new)) if new == int(new) else f"{new:.1f}"
    # swap the digits inside the matched span; "1,200" has no "1200" substring, so build it by position
    lead = raw[:len(raw) - len(raw.lstrip("$"))]
    tail = raw[len(lead) + len(re.match(r"[\d,]*(?:\.\d+)?", raw[len(lead):]).group(0)):]
    return bullet[:m.start()] + lead + new_s + tail + bullet[m.end():]


def add_seniority(bullet: str, rng) -> str | None:
    """Scripted title/scope inflation, e.g. 'Built ...' -> 'Led a team of 8 to build ...'."""
    m = re.match(r"^(Built|Developed|Created|Designed|Wrote|Analysed|Analyzed|Automated|Implemented|"
                 r"Presented|Prepared|Conducted|Delivered|Maintained)\b\s*(.*)$", bullet)
    if not m:
        return None
    return f"Led a team of {rng.choice([4, 6, 8, 12])} that {m.group(1).lower()} {m.group(2)}"


RUN_TOGETHER = re.compile(r"[a-z]{3}[A-Z][a-z]")   # "kitsMaintained": headings/fields glued together
DATE_FIELD = re.compile(r"\d{2}/\d{4}|\d{4}\s*(?:to|-)\s*(?:\d{2,4}|present)|\bExpected in\b", re.I)
# a number that is a real quantity (not a year, date or version): %, $, "10+", or a count with a unit
QUANTITY = re.compile(
    r"\$\d[\d,]*(?:\.\d+)?\s?(?:k|m|mm|million|billion|bn)?\b|\b\d[\d,]*(?:\.\d+)?\s?%|\b\d[\d,]*\+"
    r"|\b\d[\d,]*(?:\.\d+)?\s(?:years?|months?|users|customers|clients|employees|people|members|team|teams|"
    r"projects|applications|servers|records|accounts|reports|stores|locations|branches|countries)\b", re.I)


def is_junk(bullet: str) -> bool:
    """Skill lists, education/date lines and run-together fields make meaningless claims."""
    words = bullet.split()
    if bullet.count(",") > 0.2 * len(words) and bullet.count(",") >= 4:
        return True
    if DATE_FIELD.search(bullet):
        return True
    return any(len(w) >= 12 and RUN_TOGETHER.search(w) for w in words)


def clean_llm(text: str, premise: str) -> str:
    """Strip markdown the teacher sometimes adds (a model must not learn '**' means fabricated)
    and reject outputs that are empty, unchanged, or much longer than the bullet."""
    text = re.sub(r"[*`]+", "", text or "")
    text = re.sub(r"^\s*(?:[-•]\s+)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text == premise or len(text) > 2 * len(premise) + 50:
        return ""
    return text


def change_quantity(bullet: str, rng) -> str | None:
    """Like change_number, but only on a real quantity (45%, $2M, 10+ users), never a year or date."""
    m = QUANTITY.search(bullet)
    if not m:
        return None
    new = change_number(m.group(0), rng)
    return bullet[:m.start()] + new + bullet[m.end():] if new else None


def sub_claim(bullet: str) -> str | None:
    """A faithful atomic claim: the first clause of the bullet, if it is a full claim on its own."""
    for sep in [", ", "; ", " and ", " to "]:
        if sep in bullet:
            head = bullet.split(sep)[0].strip()
            if len(head.split()) >= 4:
                return head
    return None
