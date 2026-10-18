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
    return bullet[:m.start()] + raw.replace(digits, new_s, 1) + bullet[m.end():]


def add_seniority(bullet: str, rng) -> str | None:
    """Scripted title/scope inflation, e.g. 'Built ...' -> 'Led a team of 8 to build ...'."""
    m = re.match(r"^(Built|Developed|Created|Designed|Wrote|Analysed|Analyzed|Automated|Implemented|"
                 r"Presented|Prepared|Conducted|Delivered|Maintained)\b\s*(.*)$", bullet)
    if not m:
        return None
    return f"Led a team of {rng.choice([4, 6, 8, 12])} that {m.group(1).lower()} {m.group(2)}"


def sub_claim(bullet: str) -> str | None:
    """A faithful atomic claim: the first clause of the bullet, if it is a full claim on its own."""
    for sep in [", ", "; ", " and ", " to "]:
        if sep in bullet:
            head = bullet.split(sep)[0].strip()
            if len(head.split()) >= 4:
                return head
    return None
