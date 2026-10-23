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


# ---------------- hard E2 test set: small number edits, scope inflation, faithful rewrites ----------------

def change_quantity_small(bullet: str, rng) -> str | None:
    """A small edit to a real quantity (15% -> 18%, 10+ -> 12+): hard because the bullet barely changes."""
    m = QUANTITY.search(bullet)
    if not m:
        return None
    raw = m.group(0)
    num = re.search(r"\d[\d,]*(?:\.\d+)?", raw)
    txt = num.group(0).replace(",", "")
    val, dec = float(txt), (len(txt.split(".")[1]) if "." in txt else 0)
    unit = 10 ** -dec if dec else 1
    step = max(unit, round(val * rng.choice([0.1, 0.15, 0.2]), dec))
    new = round(val + rng.choice([-1, 1]) * step, dec)
    if new <= 0 or new == val or ("%" in raw and new > 100):
        new = round(val + step, dec)
    if "%" in raw and new > 100:
        return None
    new_s = f"{new:,.{dec}f}" if "," in num.group(0) else f"{new:.{dec}f}"
    return bullet[:m.start()] + raw[:num.start()] + new_s + raw[num.end():] + bullet[m.end():]


NEAR_TOOLS = {   # a plausible neighbour of the same kind: the swap looks natural, so it is a hard negative
    "MySQL": "PostgreSQL", "PostgreSQL": "MySQL", "Oracle": "MySQL", "MongoDB": "PostgreSQL",
    "Redshift": "BigQuery", "BigQuery": "Snowflake", "Snowflake": "Redshift",
    "Tableau": "Power BI", "Power BI": "Tableau", "Looker": "Tableau", "Excel": "Google Sheets",
    "AWS": "Azure", "Azure": "AWS", "GCP": "AWS",
    "Java": "C#", "C#": "Java", "Python": "R", "JavaScript": "TypeScript", "TypeScript": "JavaScript",
    "TensorFlow": "PyTorch", "PyTorch": "TensorFlow", "Keras": "PyTorch", "Hadoop": "Spark", "Spark": "Hadoop",
    "React": "Angular", "Angular": "Vue", "Vue": "React", "Django": "Flask", "Flask": "Django",
    "Salesforce": "HubSpot", "HubSpot": "Salesforce", "Jenkins": "Terraform", "Docker": "Kubernetes",
}


def swap_tool_near(bullet: str, resume: str) -> str | None:
    """Replace a tool with its natural neighbour (MySQL -> PostgreSQL) that the resume never mentions."""
    for old, new in NEAR_TOOLS.items():
        if re.search(rf"(?<![\w+#]){re.escape(old)}(?![\w+#])", bullet) and \
                not re.search(rf"(?<![\w+#]){re.escape(new)}(?![\w+#])", resume, re.I):
            return re.sub(rf"(?<![\w+#]){re.escape(old)}(?![\w+#])", new, bullet, count=1)
    return None


SCOPE_UP = [   # (pattern, replacement): the claimed scope or role grows, nothing else changes
    (r"\b(Supported|Assisted|Helped)\s+(?:with |in )?(?=(?:the|a|an|all|our|multiple|various)\s)", "Led "),
    (r"\b(supported|assisted|helped)\s+(?:with |in )?(?=(?:the|a|an|all|our|multiple|various)\s)", "led "),
    (r"\b(Led|led|Managed|managed|Mentored|mentored|mentoring|managing|leading|supervised|supervising|trained)"
     r" (?:the|my|our) (?:\w+ )?team\b", r"\1 the entire company"),
]


def scope_inflate(bullet: str) -> str | None:
    """Scripted scope inflation (team -> company, supported -> led); None if no cue is in the bullet."""
    for pat, new in SCOPE_UP:
        out = re.sub(pat, new, bullet, count=1)
        if out != bullet:
            return out
    return None


def reformat_quantity(bullet: str) -> str | None:
    """A faithful rewrite of a quantity's format: 10+ -> over 10, 45% -> 45 percent, $2M -> $2 million."""
    rules = [
        (r"\b(\d[\d,]*)\+\s*(years?)", r"over \1 \2"),
        (r"\b(\d[\d,]*)\+", r"over \1"),
        (r"\b(\d[\d,.]*)\s?%", r"\1 percent"),
        (r"\$(\d[\d,.]*)\s?(?:M|MM|m)\b", r"$\1 million"),
        (r"\$(\d[\d,.]*)\s?(?:K|k)\b", r"$\1 thousand"),
        (r"\b(\d[\d,.]*)\s?(?:mn|MM)\b", r"\1 million"),
    ]
    for pat, rep in rules:
        out = re.sub(pat, rep, bullet, count=1)
        if out != bullet:
            return out
    return None


VERB_SYNONYMS = {
    "developed": "Built", "built": "Developed", "created": "Built", "designed": "Devised",
    "implemented": "Introduced", "improved": "Enhanced", "enhanced": "Improved", "analyzed": "Examined", "analysed": "Examined", "wrote": "Authored",
    "automated": "Streamlined", "reduced": "Cut", "increased": "Raised", "prepared": "Compiled",
    "delivered": "Provided", "monitored": "Tracked", "identified": "Pinpointed", "tested": "Validated",
}


def swap_verb(bullet: str) -> str | None:
    """Faithful rewrite: replace the opening verb with a near-synonym (Developed -> Built)."""
    first, _, rest = bullet.partition(" ")
    new = VERB_SYNONYMS.get(first.lower())
    return f"{new} {rest}" if new and rest else None


def reorder_clause(bullet: str) -> str | None:
    """Faithful rewrite: move a trailing 'using/with/by X' phrase to the front ('Using X, deploy ...')."""
    m = re.match(r"^(?P<main>[A-Z][^,;]*?)\s+(?P<kw>using|via|through)\s+(?P<tail>[^,;]+?)\.?$", bullet)
    if not m or len(m["main"].split()) < 3 or re.search(r"\b(?:and|or)\b", m["main"]):   # one clause only
        return None
    main = m["main"]
    if not (main.split()[0].isupper() and len(main.split()[0]) > 1):   # keep acronyms capitalised
        main = main[0].lower() + main[1:]
    return f"{m['kw'].capitalize()} {m['tail']}, {main}."
