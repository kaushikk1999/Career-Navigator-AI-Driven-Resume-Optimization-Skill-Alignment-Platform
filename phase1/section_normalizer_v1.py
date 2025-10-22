"""
Deterministic, idempotent post-processor for EDUCATION and SKILLS sections.

Goals
- EDUCATION: one-line entries -> Institution — Program — Dates — Location
- SKILLS: inline categories -> "Category: item1, item2, …" (one row per)
- Punctuation: space after commas, normalize ranges to en dash with spaces
- Parentheses: balance missing right paren on logical line (skills/tools)

Returns normalized text and a small audit JSON with edit counters.
Pure Python, CPU-light, no I/O, no network.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple


_MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December"
_DATE_RANGE = re.compile(rf"(?i)\b(?:{_MONTHS})?\s*\d{{4}}\b.*?(?:-|—|–).*?\b(?:{_MONTHS})?\s*(?:\d{{4}}|present)\b")
_DEG = re.compile(r"(?i)\b(BA|BSc|B\.?(?:Tech|E|Sc|CA)|M\.?(?:Tech|E|Sc|CA)|Master|Bachelor|Diploma|Data Science)\b")
_LOC_HINT = re.compile(r"(?i)\b(India|Bengaluru|Bangalore|Pune|Kolkata|Kokata|Maharashtra|Delhi|Mumbai)\b")
_HEADER = re.compile(r"^[A-Z][A-Z /&-]{3,}$")
_CAT = re.compile(r"(?i)^(Programming Languages|Libraries/Frameworks|Tools ?/ ?Platforms|Databases)\b")


def _norm_dash(s: str) -> str:
    return re.sub(r"\s*(?:-|—)\s*", " – ", s)


def inline_education(block: str, audit: Dict) -> str:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    out: List[str] = []
    i = 0
    while i < len(lines):
        inst = lines[i]
        prog = dates = loc = ""
        j = i + 1
        while j < len(lines) and j <= i + 4:
            s = lines[j].strip()
            if not prog and _DEG.search(s):
                prog = s
            elif not dates and (_DATE_RANGE.search(s) or re.search(r"\b\d{4}\b\s*(?:-|—|–)\s*(?:\d{4}|present)\b", s, re.I)):
                dates = s
            elif not loc and (_LOC_HINT.search(s) or re.search(r"^[A-Za-z][A-Za-z .-]+(?:,\s*[A-Za-z .-]+)*$", s)):
                loc = s
            j += 1
        present = sum(1 for x in (inst, prog, dates, loc) if x)
        if present >= 2:
            row = " — ".join([x for x in [inst, prog, _norm_dash(dates), loc] if x]).strip(" —")
            row = re.sub(r"[ \t]{2,}", " ", row)
            out.append(row)
            audit["education_inlined"] += 1
            i = j
        else:
            out.append(inst)
            i += 1
    return "\n".join(out)


def inline_skills(block: str, audit: Dict, notes: List[str]) -> str:
    lines = [ln.rstrip() for ln in block.splitlines()]
    out: List[str] = []
    i = 0
    while i < len(lines):
        m = _CAT.match(lines[i]) if lines[i] else None
        if not m:
            out.append(lines[i])
            i += 1
            continue
        cat = m.group(1)
        items: List[str] = []
        i += 1
        while i < len(lines) and not _CAT.match(lines[i] or "") and lines[i].strip() and not _HEADER.match(lines[i].strip()):
            items.append(lines[i].strip())
            i += 1
        items_txt = ", ".join(items)
        items_txt = re.sub(r"\s*,\s*", ", ", items_txt)
        items_txt = re.sub(r",\s*,+", ", ", items_txt)
        items_txt = re.sub(r"[ \t]{2,}", " ", items_txt).strip()
        # Balance paren in Tools/Platforms
        if re.match(r"(?i)^Tools ?/ ?Platforms$", cat) and items_txt.count("(") > items_txt.count(")"):
            # Capture the dangling token near end
            m2 = re.search(r"([A-Za-z0-9+./ -]+\([^)]*)$", items_txt)
            if m2:
                notes.append(f"balanced_parenthesis: {m2.group(1)})")
            items_txt = items_txt + ")"
            audit["parentheses_balanced"] += 1
        out.append(f"{cat}: {items_txt}" if items_txt else lines[i-1])
        audit["skills_inlined"] += 1 if items_txt else 0
        while i < len(lines) and not lines[i].strip():
            i += 1
    return "\n".join([ln for ln in out if ln is not None])


def _punctuation_normalize(text: str, audit: Dict) -> str:
    t, n1 = re.subn(r",(\S)", r", \1", text)           # space after comma
    t, n2 = re.subn(r"\s*(?:-|—)\s*", " – ", t)          # en dash with spaces
    t, n3 = re.subn(r"[ \t]{2,}", " ", t)                # collapse spaces
    audit["spaces_after_commas"] += n1
    audit["dash_normalized"] += n2
    return t


def normalize_resume(text: str) -> Tuple[str, Dict]:
    base = unicodedata.normalize("NFC", text or "")
    edits = {"education_inlined": 0, "skills_inlined": 0, "spaces_after_commas": 0, "dash_normalized": 0, "parentheses_balanced": 0}
    notes: List[str] = []

    def _apply_block(label: str, func):
        pat = re.compile(rf"(?s)(^|\n){label}\n(.*?)(?=\n[A-Z][A-Z ]{{3,}}\n|\Z)")
        def _cb(m: re.Match) -> str:
            head = m.group(0).splitlines()[0]
            body = m.group(2)
            try:
                new_body = func(body)
            except Exception:
                new_body = body
            return head + "\n" + new_body
        return pat.sub(_cb, base)

    # EDUCATION block
    base = re.sub(r"(?s)(^|\n)EDUCATION\n(.*?)(?=\n[A-Z][A-Z ]{3,}\n|\Z)",
                  lambda m: m.group(0).splitlines()[0] + "\n" + inline_education(m.group(2), edits), base)
    # SKILLS block
    base = re.sub(r"(?s)(^|\n)SKILLS\n(.*?)(?=\n[A-Z][A-Z ]{3,}\n|\Z)",
                  lambda m: m.group(0).splitlines()[0] + "\n" + inline_skills(m.group(2), edits, notes), base)

    # Global punctuation last
    base = _punctuation_normalize(base, edits)

    audit = {
        "normalizer": "resume_v1",
        "edits": edits,
        "notes": notes,
    }
    return base, audit

