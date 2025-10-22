"""
Deterministic resume/CV text normalizer for OCR/PDF extracted pages.

Public API:
    fix_resume_text(raw_pages: list[str], mode: str = "readability", opts: dict | None = None) -> dict

Goals
- Readability-first normalization with optional source-faithful mode
- No fabrication; only reformatting and trivial syntax fixes
- Stable ordering, deterministic, idempotent
- O(total_chars) runtime, standard library only

This module does not touch I/O. It transforms plain-text pages.
"""
from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any, Iterable, Optional


# -------------------------- Compiled patterns ---------------------------

_RE_WS = re.compile(r"[ \t]+")
_RE_URL = re.compile(r"(?i)\bhttps?://[^\s)]+")
_RE_BULLET = re.compile(r"(?m)^\s*[-*•·–—]\s+")
_RE_DUP_BLANKS = re.compile(r"\n{3,}")
_RE_ARTIFACT_URL = re.compile(r"(?mi)^(https?://)?[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/?\s*$")

_MONTHS = (
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
)
_RE_DATE_RANGE = re.compile(
    r"(?i)\b(?:(?:" + "|".join(_MONTHS) + r")[\s-]*)?(?:19|20)\d{2}\b\s*[–-]\s*(?:(?:" + "|".join(_MONTHS) + r")[\s-]*)?(?:19|20)\d{2}|present\b"
)

_RE_TITLECASE_TOKEN = re.compile(r"^[A-Z][a-z]+(?:[-'][A-Za-z]+)?$")
_RE_ALLCAPS = re.compile(r"^[A-Z0-9][A-Z0-9 '&/.-]*[A-Z0-9)]$")
_RE_HEADING_SHORT = re.compile(r"^[A-Z][A-Z &/]+$")

_RE_DEGREE = re.compile(
    r"(?i)\b((?:master|bachelor)\s+of\s+(?:technology|science|engineering))\b|\b(m\.?tech|b\.?tech|m\.?sc|b\.?sc|m\.?e|b\.?e|phd|mba|mca|bca)\b"
)

_KNOWN_PAREN_TOKENS = {"S3", "EC2", "EKS", "GKE", "RDS", "SNS", "SQS", "ELB", "GCS", "BigQuery", "PostgreSQL", "MySQL"}

_CANON_BRANDS = {
    "linkedin": "LinkedIn",
    "github": "GitHub",
    "gitlab": "GitLab",
    "coursera": "Coursera",
    "streamlit": "Streamlit",
    "aws": "AWS",
    "gcp": "GCP",
}

_CANON_DEGREES = {
    "master of technology": "Master of Technology",
    "bachelor of technology": "Bachelor of Technology",
    "master of science": "Master of Science",
    "bachelor of science": "Bachelor of Science",
    "master of engineering": "Master of Engineering",
    "bachelor of engineering": "Bachelor of Engineering",
}


# ------------------------------ Utilities -------------------------------

def _normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return _RE_WS.sub(" ", s.strip())


def _split_pages(raw_pages: List[str]) -> List[List[str]]:
    return [p.split("\n") if p else [] for p in raw_pages]


def _join_pages(pages: List[List[str]]) -> str:
    return "\f".join("\n".join(pg) for pg in pages)


def _title_case(token: str) -> str:
    if not token:
        return token
    if token.isupper() or token.isdigit():
        return token
    small = {"and", "or", "the", "of", "in", "on", "for", "to", "with", "a", "an"}
    parts = re.split(r"(\s+|[,&/])", token)
    out: List[str] = []
    for p in parts:
        if not p or p.isspace() or p in {",", "&", "/"}:
            out.append(p)
        elif p.lower() in small:
            out.append(p.lower())
        else:
            out.append(p[:1].upper() + p[1:].lower())
    return "".join(out)


def _looks_institution(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if _RE_ALLCAPS.match(t) and len(t.split()) >= 2:
        return True
    words = t.split()
    if 1 <= len(words) <= 8 and all(_RE_TITLECASE_TOKEN.match(w) or w.isupper() for w in words):
        return True
    return False


def _contains_degree(s: str) -> bool:
    return _RE_DEGREE.search(s) is not None


def _extract_degree_and_field(line: str, mode: str) -> Tuple[str, Optional[str]]:
    """Return canonical degree string and optional field before it.

    Example: "Data Science Master of technology" -> ("Master of Technology", "Data Science")
    For source_faithful mode, preserves original casing on field and degree phrase.
    """
    m = _RE_DEGREE.search(line)
    if not m:
        return (line.strip(), None)
    span = m.span()
    deg_raw = line[span[0]:span[1]]
    before = line[: span[0]].strip(" ,-/|")
    # Normalize degree phrase
    deg_norm = deg_raw.lower().replace(".", "")
    if deg_norm in _CANON_DEGREES:
        deg = _CANON_DEGREES[deg_norm] if mode == "readability" else deg_raw
    else:
        deg = _title_case(deg_raw) if mode == "readability" else deg_raw
    field = _title_case(before) if (before and mode == "readability") else (before or None)
    return deg.strip(), (field if field else None)


def _find_contact_block(lines: List[str]) -> Tuple[int, int]:
    """Heuristic: contact block spans from first non-empty line until the last of the
    next 5 lines containing contact separators ("|", ",") or keywords (Email/Mobile/Phone).
    Returns (start_idx, end_idx_exclusive)."""
    n = len(lines)
    start = 0
    while start < n and not lines[start].strip():
        start += 1
    end = min(n, start + 1)
    for i in range(start, min(n, start + 6)):
        t = lines[i].strip().lower()
        if any(k in t for k in ("email", "mobile", "phone")) or "|" in t or "," in t:
            end = i + 1
    return (start, end)


def _readability_score_estimate(text: str) -> float:
    """Proxy similar to tests: reward average line length within ~60-110 and
    a bit for headings and bullets. Returns 0..1.
    """
    lines = [ln for ln in text.replace("\r", "\n").split("\n")]
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return 0.0
    avg_len = sum(len(ln) for ln in non_empty) / max(1, len(non_empty))
    if 60 <= avg_len <= 110:
        base = 1.0
    elif avg_len < 60:
        base = max(0.0, avg_len / 60.0)
    else:
        base = max(0.0, (200.0 - min(avg_len, 200.0)) / 90.0)
    bullets = sum(1 for ln in non_empty if ln.lstrip().startswith("• "))
    headings = sum(1 for ln in non_empty if (ln.strip().isupper() and len(ln.strip().split()) <= 4))
    bonus = min(0.1, 0.5 * (bullets / max(1, len(non_empty))) + 0.5 * (headings / max(1, len(non_empty))))
    return min(1.0, 0.9 * base + bonus)


# --------------------------- Transformations ----------------------------

def _dedupe_headers(pages: List[List[str]], edits: List[Dict[str, Any]]) -> List[List[str]]:
    if len(pages) <= 1:
        return pages
    edges: List[str] = []
    for pg in pages:
        top = [ln.strip() for ln in pg[:5] if ln.strip()]
        bot = [ln.strip() for ln in pg[-5:] if ln.strip()]
        edges.extend(set(top + bot))
    threshold = max(2, int(0.6 * len(pages)))
    common = {ln for ln in set(edges) if edges.count(ln) >= threshold and len(ln) <= 80}
    if not common:
        return pages
    out: List[List[str]] = []
    for pg in pages:
        tmp: List[str] = []
        for idx, ln in enumerate(pg):
            is_edge = idx < 5 or idx >= len(pg) - 5
            if is_edge and ln.strip() in common:
                edits.append({"type": "dedupe_header_footer", "line": ln})
                continue
            tmp.append(ln)
        out.append(tmp)
    return out


def _norm_bullets(lines: List[str], edits: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        new = _RE_BULLET.sub("• ", ln)
        if new != ln:
            edits.append({"type": "bullet_normalize", "before": ln, "after": new})
        out.append(new)
    return out


def _fix_parens(lines: List[str], edits: List[Dict[str, Any]], warnings: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        s = ln
        # Drop malformed URL artifacts entirely (safety rule)
        if _RE_ARTIFACT_URL.match(s.strip()):
            edits.append({"type": "drop_malformed_url_artifact", "line": s})
            continue
        opens = s.count("(")
        closes = s.count(")")
        if opens > closes:
            tail = s.rsplit("(", 1)[-1]
            token = tail.strip()
            # Same-line token within parentheses
            if token and any(token.startswith(t) for t in _KNOWN_PAREN_TOKENS):
                new = s + ")"
                edits.append({"type": "paren_fix", "line_before": s, "line_after": new})
                s = new
        elif closes > opens:
            warnings.append("unmatched_right_paren_ignored")
        out.append(s)
    return out


def _apply_city_map(lines: List[str], mode: str, city_map: Dict[str, str], edits: List[Dict[str, Any]], warnings: List[str]) -> List[str]:
    if mode != "readability" or not city_map:
        # In source-faithful mode, record warnings if any token is matched
        if mode == "source_faithful":
            lower = {k.lower(): v for k, v in city_map.items()}
            for ln in lines:
                for tok in re.findall(r"[A-Za-z]+", ln):
                    if tok.lower() in lower:
                        warnings.append("city_normalization_skipped")
                        break
        return lines
    out: List[str] = []
    lower = {k.lower(): v for k, v in city_map.items()}
    for ln in lines:
        def repl(m: re.Match) -> str:
            word = m.group(0)
            fixed = lower.get(word.lower())
            if fixed and fixed != word:
                edits.append({"type": "city_normalization", "from": word, "to": fixed, "mode": mode})
                return fixed
            return word
        new = re.sub(r"\b[A-Za-z]+\b", repl, ln)
        out.append(new)
    return out


def _case_pass(lines: List[str], mode: str, section_names: List[str], edits: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Normalize casing for headings and known brands/degrees in readability mode.
    Returns (new_lines, detected_sections)."""
    detected: List[str] = []
    if mode != "readability":
        # Still detect sections without mutating
        for ln in lines:
            if _is_heading_like(ln, section_names):
                detected.append(_canonical_heading(ln))
        return lines, detected

    # Determine heading style
    headings = [ln for ln in lines if _is_heading_like(ln, section_names)]
    upper_count = sum(1 for h in headings if h.strip().isupper()) if headings else 0
    style_upper = upper_count >= max(1, len(headings) // 2)

    out: List[str] = []
    for ln in lines:
        new = ln
        if _is_heading_like(ln, section_names):
            detected.append(_canonical_heading(ln))
            head = ln.strip()
            fixed = head.upper() if style_upper else _title_case(head)
            if fixed != head:
                new = ln.replace(head, fixed)
                edits.append({"type": "casing_fix", "where": "heading", "before": head, "after": fixed})
        else:
            # Brand casing and degree canonicalization (lightweight)
            def _brand_repl(m: re.Match) -> str:
                w = m.group(0)
                canon = _CANON_BRANDS.get(w.lower())
                if canon and canon != w:
                    edits.append({"type": "casing_fix", "where": "brand", "before": w, "after": canon})
                    return canon
                return w

            new2 = re.sub(r"\b[a-z]{2,}\b", _brand_repl, new)

            def _deg_repl(m: re.Match) -> str:
                deg = m.group(0)
                canon = _CANON_DEGREES.get(deg.lower())
                if canon and canon != deg:
                    edits.append({"type": "casing_fix", "where": "degree", "before": deg, "after": canon})
                    return canon
                return _title_case(deg)

            new3 = re.sub(r"(?i)\b(master of technology|bachelor of technology|master of science|bachelor of science|master of engineering|bachelor of engineering)\b", _deg_repl, new2)
            new = new3
        out.append(new)
    return out, detected


def _is_heading_like(ln: str, section_names: List[str]) -> bool:
    t = ln.strip()
    if not t:
        return False
    if t.upper() in {n.upper() for n in section_names}:
        return True
    if _RE_HEADING_SHORT.match(t):
        return True
    return False


def _canonical_heading(ln: str) -> str:
    t = ln.strip()
    return t.upper()


def _join_education_block(lines: List[str], mode: str, edits: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if _looks_institution(ln):
            # Look ahead up to 5 lines for degree, city (optional), and date range
            window_end = min(n, i + 6)
            degree_j = city_j = date_j = -1
            degree_line = city_line = date_line = None
            for j in range(i + 1, window_end):
                t = lines[j].strip()
                tl = t.lower()
                if degree_j == -1 and (_contains_degree(t) or ("bachelor" in tl or "master" in tl)):
                    degree_j, degree_line = j, t
                if city_j == -1 and t and not any(ch.isdigit() for ch in t) and len(t.split()) <= 4 and not _RE_URL.search(t):
                    # Exclude institution/degree-like tokens from being mistaken as city
                    if not (_contains_degree(t) or any(k in tl for k in ("university","college","institute","school","data","science","engineering","technology"))):
                        if re.match(r"^[A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,2}$", t):
                            city_j, city_line = j, t
                if date_j == -1 and _RE_DATE_RANGE.search(t):
                    date_j, date_line = j, t
                if degree_line and date_line and (city_line is not None or city_j == -1):
                    # Found enough
                    break
            if degree_line and date_line:
                inst = _normalize_ws(ln)
                deg, field = _extract_degree_and_field(degree_line, mode)
                # Assemble degree+field
                deg_part = deg
                if field:
                    deg_part = f"{field}, {deg}" if mode == "readability" else f"{field} {deg}"
                city_part = f" ({_title_case(city_line) if (city_line and mode=='readability') else city_line})" if city_line else ""
                cluster = f"{inst} — {deg_part}{city_part} — {_normalize_ws(date_line)}"
                before = "\n".join([ln] + [x for x in [degree_line, city_line, date_line] if x])
                edits.append({"type": "join_education_block", "before": before, "after": cluster})
                out.append(cluster)
                # Skip used lines
                used = {i, degree_j, date_j}
                if city_j != -1:
                    used.add(city_j)
                i += 1
                while i < n and i in used:
                    i += 1
                continue
        out.append(ln)
        i += 1
    return out


def _lift_links(pages: List[List[str]], link_hints: List[str], edits: List[Dict[str, Any]], warnings: List[str]) -> List[List[str]]:
    # Gather URLs and map by host hint
    urls: List[str] = []
    for pg in pages:
        for ln in pg:
            urls.extend(_RE_URL.findall(ln))
    urls = list(dict.fromkeys(urls))  # stable unique

    found: Dict[str, str] = {}
    for u in urls:
        lu = u.lower()
        if "github.com" in lu:
            found.setdefault("github", u)
        if "linkedin.com" in lu:
            found.setdefault("linkedin", u)
        if "gitlab.com" in lu:
            found.setdefault("gitlab", u)
        if "portfolio" in lu:
            found.setdefault("portfolio", u)

    if not pages:
        return pages

    # Heuristic: lift into the first page contact block when labels present but URLs missing
    p0 = list(pages[0])
    s, e = _find_contact_block(p0)
    block = "\n".join(p0[s:e])
    lowered = block.lower()
    lifted_any = False
    for hint in link_hints:
        if hint in ("github", "linkedin", "gitlab", "portfolio"):
            label = _CANON_BRANDS.get(hint, hint.title())
            has_label = hint in lowered or label.lower() in lowered
            has_url = any(hint in u.lower() for u in urls)
            if has_label and has_url:
                url = found.get(hint)
                # Check if already present in the contact block
                present = any((hint in ln.lower() and _RE_URL.search(ln)) for ln in p0[s:e])
                # Presence anywhere in page to ensure idempotency
                present_anywhere = any((hint in ln.lower() and _RE_URL.search(ln)) or (ln.strip().lower().startswith(label.lower()+":") and _RE_URL.search(ln)) for ln in p0)
                if not (present or present_anywhere) and url:
                    insert_line = f"{label}: {url}"
                    p0.insert(e, insert_line)
                    e += 1
                    lifted_any = True
                    edits.append({"type": "link_lift", "label": label, "url": url})
            elif has_label and not has_url:
                warnings.append(f"missing_url_for_{hint}")
    if lifted_any:
        pages = [p0] + pages[1:]
    return pages


# ------------------------------ Public API ------------------------------

def fix_resume_text(raw_pages: List[str], mode: str = "readability", opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize OCR/PDF-extracted resume text.

    Inputs
    - raw_pages: list of page strings (with \n line breaks)
    - mode: "readability" or "source_faithful"
    - opts: optional dict with keys:
        * known_city_map: dict[str,str]
        * section_names: list[str]
        * link_hints: list[str]

    Returns dict with keys: text, edits, metrics, warnings
    """
    assert mode in {"readability", "source_faithful"}
    opts = opts or {}
    city_map = {
        "kokata": "kolkata",
        "manglore": "mangalore",
    }
    city_map.update(opts.get("known_city_map", {}))
    section_names = opts.get("section_names", ["EDUCATION", "EXPERIENCE", "SKILLS", "PROJECTS", "CERTIFICATIONS"])
    link_hints = [h.lower() for h in opts.get("link_hints", ["github", "linkedin", "portfolio"]) if h]

    # Normalize WS per page early; split to lines
    pages = [[_normalize_ws(ln) for ln in p.split("\n")] for p in raw_pages]
    edits: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # F) Header/footer dedup (safety)
    pages = _dedupe_headers(pages, edits)

    # Per page transformations
    for pi in range(len(pages)):
        lines = pages[pi]
        # G) bullets + whitespace
        lines = _norm_bullets(lines, edits)
        # C) Parenthesis repair and malformed URL artifact drop
        lines = _fix_parens(lines, edits, warnings)
        # A) Education block joiner (applies where pattern matches)
        lines = _join_education_block(lines, mode, edits)
        # B) City normalization
        lines = _apply_city_map(lines, mode, city_map, edits, warnings)
        pages[pi] = lines

    # E) Link lifting using whole document context
    pages = _lift_links(pages, link_hints, edits, warnings)

    # D) Casing normalization (readability) + section detection
    sections_detected: List[str] = []
    for pi in range(len(pages)):
        new_lines, detected = _case_pass(pages[pi], mode, section_names, edits)
        pages[pi] = new_lines
        for h in detected:
            if h not in sections_detected:
                sections_detected.append(h)

    # Cleanup whitespace (>=3 blank lines -> 1)
    text = _join_pages(pages)
    text = _RE_DUP_BLANKS.sub("\n", text)

    # Metrics
    metrics = {
        "chars": len(text),
        "lines": sum(len(p.split("\n")) for p in text.split("\f")),
        "sections_detected": sections_detected,
        "readability_score_estimate": _readability_score_estimate(text),
        "changes": len(edits),
    }

    return {"text": text, "edits": edits, "metrics": metrics, "warnings": warnings}


# ------------------------------- Selftest -------------------------------

def selftest() -> None:
    def _fix(pages, mode="readability"):
        return fix_resume_text(pages, mode=mode)

    # Paren repair
    out = _fix(["AWS (S3"]) 
    assert "AWS (S3)" in out["text"], "Paren repair failed"

    # Education join (readability)
    edu = [
        "CHRIST UNIVERSITY",
        "Data Science Master of technology",
        "Bangalore",
        "August 2024 - August 2026",
    ]
    out2 = _fix(["\n".join(edu)])
    assert "CHRIST UNIVERSITY — Data Science, Master of Technology (Bangalore) — August 2024 - August 2026" in out2["text"], "Education join failed"

    # City map only in readability
    out3 = _fix(["kokata, india"], mode="readability")
    assert "kolkata, india" in out3["text"], "City normalization (readability) failed"
    out3b = _fix(["kokata, india"], mode="source_faithful")
    assert "kokata, india" in out3b["text"] and out3b["edits"] == [], "City normalization should be skipped in source_faithful"

    # Link lift from page 2 into contact block on page 1
    p1 = "\n".join(["JOHN DOE", "Email: john@example.com | GitHub | LinkedIn"]) 
    p2 = "See more: https://linkedin.com/in/kaushik99 and https://github.com/kaushik" 
    out4 = _fix([p1, p2])
    assert "LinkedIn:" in out4["text"] and "GitHub:" in out4["text"], "Link lift failed"

    # Idempotency
    once = _fix([p1, p2])
    twice = fix_resume_text(once["text"].split("\f"), mode="readability")
    assert twice["text"] == once["text"] and len(twice["edits"]) == 0, "Idempotency failed"

    # Print one-line metrics summary
    print(f"OK: chars={once['metrics']['chars']} lines={once['metrics']['lines']} changes={once['metrics']['changes']}")


if __name__ == "__main__":
    selftest()
