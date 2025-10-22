"""
Deterministic, non-generative Document Formatter

Normalizes and restructures OCR'd plain text while preserving content:
- Section adjacency (EDUCATION/ACADEMICS): fold program + institute + city + dates
- SKILLS layout: normalize headings -> values into bullets or two-column rows
- Punctuation: close unmatched parentheses, normalize commas and pipes
- Micro-typos: common location fixes (configurable, region-aware)
- Whitespace compaction with section structure preserved

Public API:
    normalize_resume_layout(text: str, region: str | None = None, opts: dict | None = None) -> str

No external deps; deterministic; idempotent on repeated runs.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple, Dict, Any, Optional


# ---------------------- Regex & small dictionaries ----------------------
RE_HEADER = re.compile(r"^(EDUCATION|ACADEMICS|EXPERIENCE|SKILLS|PROJECTS|CERTIFICATIONS|AWARDS)\s*$", re.I)
RE_SECTION_EDU = re.compile(r"^(EDUCATION|ACADEMICS)\s*$", re.I)
RE_SECTION_SKILLS = re.compile(r"^SKILLS\s*$", re.I)

RE_DATE = re.compile(
    r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b.*?\b\d{4}\b(?:\s*[–-]\s*\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b.*?\b(?:\d{4}|present)\b)?|\b\d{4}\b\s*[–-]\s*\b(?:\d{4}|present)\b"
)

RE_PROGRAM = re.compile(r"(?i)\b(m\.?tech|b\.?tech|m\.?sc|b\.?sc|m\.?e|b\.?e|phd|mba|mca|bca|master|bachelor)\b")
RE_INSTITUTE = re.compile(r"(?i)\b(university|college|institute|school|academy|iit|nit)\b")
RE_TITLECASE = re.compile(r"^[A-Z][a-z]+(?:[ '-][A-Za-z]+)*$")
RE_UPPER = re.compile(r"^[A-Z][A-Z &/.'-]+$")

RE_WS = re.compile(r"[ \t]+")
RE_PIPE = re.compile(r"\s*\|\s*")
RE_MULTI_COMMA = re.compile(r"\s*,\s*,+")
RE_COMMA_NO_SPACE = re.compile(r",(\S)")
RE_MULTI_SPACE = re.compile(r"\s{2,}")
RE_TRIPLE_NL = re.compile(r"\n{3,}")
RE_EMPTY = re.compile(r"^\s*$")
RE_URL = re.compile(r"(?i)\bhttps?://\S+")

SKILL_HEADS = [
    (re.compile(r"(?i)^programming( languages)?\s*:?$"), "Programming"),
    (re.compile(r"(?i)^libraries(/frameworks)?\s*:?$"), "Libraries/Frameworks"),
    (re.compile(r"(?i)^tools( / platforms)?\s*:?$"), "Tools/Platforms"),
    (re.compile(r"(?i)^databases?\s*:?$"), "Databases"),
]

CITY_FIXES_DEFAULT = {
    "kokata": "kolkata",
    "manglore": "mangalore",
    "bangaluru": "bengaluru",
    # Optional modernization per style guide (kept off by default)
    # "bombay": "mumbai",
}


# ------------------------------ Utilities ------------------------------
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _lines(s: str) -> List[str]:
    return s.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _join(lines: List[str]) -> str:
    return "\n".join(lines)


def _case_preserve(repl: str, original: str) -> str:
    if original.isupper():
        return repl.upper()
    if original[:1].isupper() and original[1:].islower():
        return repl[:1].upper() + repl[1:]
    if original.islower():
        return repl.lower()
    return repl


def _fix_commas_pipes(line: str) -> str:
    s = line
    s = RE_PIPE.sub(" | ", s)
    s = RE_MULTI_COMMA.sub(",", s)
    s = RE_COMMA_NO_SPACE.sub(r", \1", s)
    s = RE_MULTI_SPACE.sub(" ", s)
    return s.strip()


def _fix_parens(line: str) -> str:
    if line.count("(") > line.count(")"):
        return line + ")"
    return line


def _fix_city_typos(line: str, city_map: Dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        w = m.group(0)
        fixed = city_map.get(w.lower())
        return _case_preserve(fixed, w) if fixed else w

    s = re.sub(r"\b[A-Za-z]+\b", repl, line)
    s = re.sub(r"([A-Za-z])\,([A-Za-z])", r"\1, \2", s)
    return s


def _compact_blanklines(lines: List[str]) -> List[str]:
    txt = _join(lines)
    txt = RE_TRIPLE_NL.sub("\n\n", txt)
    return _lines(txt)


def _ensure_header_spacing(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        if RE_HEADER.match(lines[i].strip()):
            j = i + 1
            empties = 0
            while j < len(lines) and RE_EMPTY.match(lines[j]):
                empties += 1
                j += 1
            if empties:
                out.append("")
            i = j
            continue
        i += 1
    return out


def _remove_blank_between_role_and_dates(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(lines):
        if i + 2 < len(lines) and lines[i].strip() and RE_EMPTY.match(lines[i + 1]) and RE_DATE.search(lines[i + 2]):
            out.append(lines[i])
            out.append(lines[i + 2])
            i += 3
            continue
        out.append(lines[i])
        i += 1
    return out


# ---------------------- Education folding logic ------------------------
def _block_clusters(lines: List[str]) -> List[List[str]]:
    clusters: List[List[str]] = []
    cur: List[str] = []
    for ln in lines + [""]:  # sentinel blank
        if not RE_EMPTY.match(ln):
            cur.append(ln)
            continue
        if cur:
            clusters.append(cur)
            cur = []
        else:
            clusters.append([""])
    if clusters and clusters[-1] == [""]:
        clusters.pop()
    return clusters


def _fold_education(lines: List[str]) -> List[str]:
    out: List[str] = []
    for cluster in _block_clusters(lines):
        if cluster == [""]:
            out.append("")
            continue
        program = None
        institute = None
        city = None
        dates = None
        used = set()
        for idx, ln in enumerate(cluster):
            t = ln.strip()
            if dates is None and RE_DATE.search(t):
                dates = t
                used.add(idx)
                continue
            if program is None and (RE_PROGRAM.search(t) or RE_UPPER.match(t) or RE_TITLECASE.match(t)):
                if not RE_DATE.search(t) and not RE_URL.search(t):
                    program = t
                    used.add(idx)
                    continue
        for idx, ln in enumerate(cluster):
            if idx in used:
                continue
            t = ln.strip()
            if institute is None and (RE_INSTITUTE.search(t) or RE_UPPER.match(t)) and not RE_DATE.search(t):
                institute = t
                used.add(idx)
                continue
        for idx, ln in enumerate(cluster):
            if idx in used:
                continue
            t = ln.strip()
            if city is None and t and not any(ch.isdigit() for ch in t) and not RE_URL.search(t):
                # Accept simple City or City, State/Country patterns
                if RE_TITLECASE.match(t) or RE_UPPER.match(t) or re.match(r"^[A-Za-z][A-Za-z .-]+(?:, [A-Za-z .-]+)?$", t):
                    city = t
                    used.add(idx)
                    break
        if program or institute or city or dates:
            parts = [p for p in [program, institute, city, dates] if p]
            out.append(" ".join(parts))
            leftovers = [cluster[i] for i in range(len(cluster)) if i not in used and cluster[i].strip()]
            out.extend(leftovers)
            out.append("")
        else:
            out.extend(cluster)
            out.append("")
    while out and RE_EMPTY.match(out[-1]):
        out.pop()
    # Replace placeholder with middle dot + spaces
    out = [ln.replace(" \u007f", " · ") for ln in out]
    return out


# ------------------------ Skills formatting ----------------------------
def _parse_skills(lines: List[str]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        matched_name: Optional[str] = None
        for pat, canon in SKILL_HEADS:
            if pat.match(ln):
                matched_name = canon
                break
        if matched_name is None:
            i += 1
            continue
        vals: List[str] = []
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt:
                j += 1
                continue
            if RE_HEADER.match(nxt) or any(pat.match(nxt) for pat, _ in SKILL_HEADS):
                break
            # Split on bullets/semicolons/pipes/commas, but NOT on hyphens inside tokens
            parts = re.split(r"\s*[•;|]\s*|,\s*", nxt)
            vals.extend([p for p in parts if p])
            j += 1
        seen = set()
        dedup = []
        for v in [RE_WS.sub(" ", v.strip()) for v in vals if v.strip()]:
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(v)
        if dedup:
            normalized[matched_name] = dedup
        i = j
    return normalized


def _render_skills_bullets(sk: Dict[str, List[str]]) -> List[str]:
    order = ["Programming", "Libraries/Frameworks", "Tools/Platforms", "Databases"]
    lines: List[str] = []
    for key in order:
        if key in sk:
            lines.append(f"• {key}: {', '.join(sk[key])}")
    return lines


def _render_skills_two_col(sk: Dict[str, List[str]]) -> List[str]:
    left_keys = ["Programming", "Libraries/Frameworks"]
    right_keys = ["Tools/Platforms", "Databases"]
    left_lines = [f"{k}: {', '.join(sk[k])}" for k in left_keys if k in sk]
    right_lines = [f"{k}: {', '.join(sk[k])}" for k in right_keys if k in sk]
    rows: List[str] = []
    n = max(len(left_lines), len(right_lines))
    pad = 40
    for i in range(n):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        rows.append(left.ljust(pad) + ("  " if right else "") + right)
    return rows


def _format_skills(lines: List[str]) -> List[str]:
    sk = _parse_skills(lines)
    if not sk:
        return [ln for ln in lines]
    bullets = _render_skills_bullets(sk)
    if bullets:
        avg = sum(len(ln) for ln in bullets) / len(bullets)
        if avg > 100:
            return _render_skills_two_col(sk)
        return bullets
    return _render_skills_two_col(sk)


# ---------------------------- Core routine -----------------------------
def _split_blocks(text: str) -> List[Tuple[str, List[str]]]:
    lines = _lines(text)
    blocks: List[Tuple[str, List[str]]] = []
    cur_head = ""
    cur_body: List[str] = []
    for ln in lines:
        if RE_HEADER.match(ln.strip()):
            if cur_head or cur_body:
                blocks.append((cur_head, cur_body))
            cur_head = ln.strip()
            cur_body = []
        else:
            cur_body.append(ln)
    blocks.append((cur_head, cur_body))
    return blocks


def _join_blocks(blocks: List[Tuple[str, List[str]]]) -> str:
    parts: List[str] = []
    for head, body in blocks:
        if head:
            parts.append(head)
        parts.extend(body)
        parts.append("")
    s = "\n".join(parts)
    s = RE_TRIPLE_NL.sub("\n\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    lines = [ln.rstrip() for ln in s.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _join_heading_dates(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(lines):
        t = lines[i].strip()
        if t and RE_UPPER.match(t):
            j = i + 1
            if j < len(lines) and RE_EMPTY.match(lines[j]):
                j += 1
            if j < len(lines) and RE_DATE.search(lines[j] or ""):
                out.append(f"{t} — {lines[j].strip()}")
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return out


def normalize_resume_layout(text: str, region: Optional[str] = None, opts: Optional[Dict[str, Any]] = None) -> str:
    """Normalize and restructure resume text deterministically (no fabrication)."""
    opts = opts or {}
    city_map = dict(CITY_FIXES_DEFAULT)
    city_map.update(opts.get("city_map", {}))
    t = _nfc(text)
    lines = [_fix_commas_pipes(RE_WS.sub(" ", ln.strip())) for ln in _lines(t)]
    lines = [_fix_parens(ln) for ln in lines]
    lines = [_fix_city_typos(ln, city_map) for ln in lines]
    lines = _compact_blanklines(lines)
    lines = _ensure_header_spacing(lines)
    lines = _remove_blank_between_role_and_dates(lines)

    blocks = _split_blocks(_join(lines))
    new_blocks: List[Tuple[str, List[str]]] = []
    for head, body in blocks:
        body = [ln.strip() for ln in body]
        if RE_SECTION_EDU.match(head or ""):
            body = _fold_education(_join_heading_dates(body))
        elif RE_SECTION_SKILLS.match(head or ""):
            body = _format_skills(body)
        else:
            body = _join_heading_dates(body)
        new_blocks.append((head, body))

    out = _join_blocks(new_blocks)
    return out


if __name__ == "__main__":
    sample = "\n".join([
        "EDUCATION",
        "M.Tech Data Science",
        "Christ University",
        "Bengaluru",
        "Aug 2024 - Aug 2026",
        "",
        "B.Sc. Computer Science",
        "XYZ College",
        "Kokata,India",
        "2020 - 2023",
        "",
        "SKILLS",
        "Programming",
        "Python, SQL",
        "Libraries/Frameworks",
        "TensorFlow; Scikit-learn | Keras",
        "Tools / Platforms",
        "Tableau, Docker, Django, Git, GitHub, AWS (S3",
        "Databases",
        "MySQL, MongoDB, Oracle",
    ])
    print(normalize_resume_layout(sample))
