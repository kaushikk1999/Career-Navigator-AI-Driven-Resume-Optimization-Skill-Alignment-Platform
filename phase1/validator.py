"""
Document normalization & fidelity validator.

Transforms raw OCR/PDF extracted text into a clean, readable, and source-faithful
rendition while preserving content and ordering. Runs a deterministic pipeline,
logs changes, computes readability and content match metrics, and returns a JSON
object per spec.

Public API:
    normalize_validate(pages: list[str], raw_text: str, region: str | None = None, opts: dict | None = None) -> dict

No network calls. Standard library only.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional


# ---------------- Patterns & small maps ----------------
_RE_WS = re.compile(r"[ \t]+")
_RE_URL = re.compile(r"(?i)\bhttps?://[^\s)]+")
_RE_PIPE = re.compile(r"\s*\|\s*")
_RE_DEHYPH = re.compile(r"(\w)-\n(\w)")
_RE_TRIPLE_BLANKS = re.compile(r"\n{3,}")
_RE_HANGING_URL = re.compile(r"(?mi)^(https?://)?[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/?\s*$")
_RE_UPPER_HEADING = re.compile(r"^[A-Z][A-Z &/()'\-]+$")
_RE_DATE_RANGE = re.compile(
    r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)?\s*\d{4}\s*[–-]\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)?\s*(?:\d{4}|present)\b"
)

_LIGS = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "“": '"', "”": '"', "‘": "'", "’": "'",
}

_DEFAULT_SECTIONS = ["EDUCATION", "EXPERIENCE", "SKILLS", "PROJECTS", "CERTIFICATIONS"]


# ---------------- Utility metrics ----------------
def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _wer(reference: str, hypothesis: str) -> float:
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    m, n = len(ref), len(hyp)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / float(m)


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union


def _readability_score(s: str) -> float:
    lines = s.split("\n")
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return 0.0
    avg_len = sum(len(ln) for ln in non_empty) / max(1, len(non_empty))
    base = 1.0 if 60 <= avg_len <= 110 else (avg_len / 60.0 if avg_len < 60 else max(0.0, (200.0 - min(avg_len, 200.0)) / 90.0))
    penalties = 0.0
    # unmatched parens
    opens = s.count("(")
    closes = s.count(")")
    if opens != closes:
        penalties += 0.1
    if _RE_TRIPLE_BLANKS.search(s):
        penalties += 0.05
    if any(len(ln) > 180 for ln in non_empty):
        penalties += 0.05
    if any("|" in ln and not _RE_PIPE.search(ln) for ln in non_empty):
        penalties += 0.05
    return max(0.0, min(1.0, 0.9 * base - penalties))


# ---------------- Transformations ----------------
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _clean_base(text: str) -> str:
    t = _nfc(text.replace("\r\n", "\n").replace("\r", "\n"))
    t = t.translate({0x00AD: None})
    for src, dst in _LIGS.items():
        t = t.replace(src, dst)
    t = _RE_DEHYPH.sub(r"\1\2", t)
    return t


def _standardize_pipes(line: str) -> str:
    # Avoid touching URLs (pipes rarely appear in URLs)
    parts = line.split("http")
    if len(parts) > 1:
        head = parts[0]
        fixed = _RE_PIPE.sub(" | ", head)
        return fixed + "http" + "http".join(parts[1:])
    return _RE_PIPE.sub(" | ", line)


def _dedupe_header_footer(pages: List[List[str]], changes: List[Dict[str, Any]]) -> List[List[str]]:
    if len(pages) <= 1:
        return pages
    edges: List[str] = []
    for pg in pages:
        top = [ln.strip() for ln in pg[:5] if ln.strip()]
        bot = [ln.strip() for ln in pg[-5:] if ln.strip()]
        edges.extend(set(top + bot))
    threshold = max(2, int(0.6 * len(pages)))
    common = {ln for ln in set(edges) if edges.count(ln) >= threshold and len(ln) <= 80}
    out: List[List[str]] = []
    for pg in pages:
        tmp: List[str] = []
        for i, ln in enumerate(pg):
            if (i < 5 or i >= len(pg) - 5) and ln.strip() in common:
                changes.append({"rule": "tidy_layout", "before_excerpt": ln[:160], "after_excerpt": "", "confidence": 0.8})
                continue
            if _RE_HANGING_URL.match(ln.strip()):
                changes.append({"rule": "tidy_layout", "before_excerpt": ln[:160], "after_excerpt": "", "confidence": 0.8})
                continue
            tmp.append(ln)
        out.append(tmp)
    return out


def _close_parens(lines: List[str], changes: List[Dict[str, Any]], flags: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        s = ln
        opens = s.count("(")
        closes = s.count(")")
        if opens > closes and s.rstrip().endswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")):
            # Append a closing paren safely (only if it looks like a truncated token)
            s2 = s + ")"
            changes.append({"rule": "close_parens", "before_excerpt": s[:160], "after_excerpt": s2[:160], "confidence": 0.7})
            s = s2
        elif closes > opens:
            flags.append("unmatched_right_paren_ignored")
        out.append(s)
    return out


def _join_heading_dates(lines: List[str], changes: List[Dict[str, Any]], section_headers: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        t = ln.strip()
        if t and (_RE_UPPER_HEADING.match(t) or t.upper() in {h.upper() for h in section_headers}):
            # Look ahead up to 2 lines for date range
            found = False
            for j in range(i + 1, min(n, i + 3)):
                nxt = lines[j].strip()
                if _RE_DATE_RANGE.search(nxt):
                    joined = f"{t} — {nxt}"
                    changes.append({"rule": "join_heading_dates", "before_excerpt": (t + "\n" + nxt)[:160], "after_excerpt": joined[:160], "confidence": 0.85})
                    out.append(joined)
                    # Skip the date line
                    i = j + 1
                    found = True
                    break
            if found:
                continue
        out.append(ln)
        i += 1
    return out


def _apply_city_map(lines: List[str], region: Optional[str], opts: Dict[str, Any], changes: List[Dict[str, Any]], flags: List[str]) -> List[str]:
    m = {"kokata": "kolkata", "manglore": "mangalore"}
    m.update(opts.get("known_city_map", {}))
    if region and region.upper() == "IN":
        # same map for now; kept for extensibility
        pass
    out: List[str] = []
    for ln in lines:
        def repl(match: re.Match) -> str:
            w = match.group(0)
            tgt = m.get(w.lower())
            if tgt and tgt != w:
                changes.append({"rule": "fix_cities", "before_excerpt": w, "after_excerpt": tgt, "confidence": 0.9})
                return tgt
            return w
        new = re.sub(r"\b[A-Za-z]+\b", repl, ln)
        out.append(new)
    return out


def _normalize_bullets(lines: List[str], changes: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        new = re.sub(r"(?m)^\s*[-*•·–—]\s+", "• ", ln)
        if new != ln:
            changes.append({"rule": "whitespace", "before_excerpt": ln[:160], "after_excerpt": new[:160], "confidence": 0.7})
        out.append(new)
    return out


def _normalize_whitespace(lines: List[str], section_headers: List[str], changes: List[Dict[str, Any]]) -> List[str]:
    # collapse spaces/tabs, standardize pipes and ensure blank line after section headers
    out: List[str] = []
    for ln in lines:
        s = _RE_WS.sub(" ", ln.strip())
        s = _standardize_pipes(s)
        out.append(s)
    # Ensure at most one blank line after top-level headings
    i = 0
    final: List[str] = []
    while i < len(out):
        final.append(out[i])
        t = out[i].strip()
        if t and (t.upper() in {h.upper() for h in section_headers} or _RE_UPPER_HEADING.match(t)):
            # collapse subsequent empty lines to exactly one
            j = i + 1
            empties = 0
            while j < len(out) and not out[j].strip():
                empties += 1
                j += 1
            if empties >= 2:
                changes.append({"rule": "tidy_layout", "before_excerpt": "\\n".join(out[i+1:i+1+empties])[:160], "after_excerpt": "", "confidence": 0.7})
            final.append("") if empties else None
            i = j
            continue
        i += 1
    return final


def _rebuild(pages: List[List[str]]) -> str:
    return "\f".join("\n".join(pg) for pg in pages)


# ---------------- Core API ----------------
def normalize_validate(pages: List[str], raw_text: str, region: Optional[str] = None, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    opts = opts or {}
    changes: List[Dict[str, Any]] = []
    flags: List[str] = []

    # Base normalization of inputs
    pages_clean = [_clean_base(p) for p in pages]
    raw_clean = _clean_base(raw_text)

    # Page -> lines
    lines_pages: List[List[str]] = [p.split("\n") for p in pages_clean]

    # Pass 1: page-aware cleanup
    lines_pages = _dedupe_header_footer(lines_pages, changes)
    for idx in range(len(lines_pages)):
        lines = lines_pages[idx]
        lines = _normalize_bullets(lines, changes)
        lines = _close_parens(lines, changes, flags)
        lines = _apply_city_map(lines, region, opts, changes, flags)
        lines_pages[idx] = lines

    # Document-level transforms
    doc_lines = [ln for pg in lines_pages for ln in (pg + ["\f"])][:-1]
    # join heading + dates across doc-level lines while preserving page breaks
    # operate page by page for better locality
    sec = opts.get("section_names", _DEFAULT_SECTIONS)
    for pi in range(len(lines_pages)):
        lines_pages[pi] = _join_heading_dates(lines_pages[pi], changes, sec)
        lines_pages[pi] = _normalize_whitespace(lines_pages[pi], sec, changes)

    # Collapse triple blanks globally per page
    for pi in range(len(lines_pages)):
        page_text = "\n".join(lines_pages[pi])
        new_text = _RE_TRIPLE_BLANKS.sub("\n\n", page_text)
        if new_text != page_text:
            changes.append({"rule": "tidy_layout", "before_excerpt": page_text[:160], "after_excerpt": new_text[:160], "confidence": 0.6})
        lines_pages[pi] = new_text.split("\n")

    normalized_text = _rebuild(lines_pages)

    # Pass 2 (idempotency check): run the same pipeline once more and accept if no change
    pages2 = normalized_text.split("\f")
    again = normalize_validate_once(pages2, region, opts)
    if again["text"] != normalized_text:
        # Accept the second pass result but merge change logs
        normalized_text = again["text"]
        changes.extend(again["changes"])  # deterministic order

    # Metrics
    readability = int(round(_readability_score(normalized_text) * 100))
    content_match = int(round((1.0 - _wer(raw_clean, normalized_text)) * 100))
    vocab_overlap = int(round(_jaccard(raw_clean, normalized_text) * 100))

    # Validators and flags
    if _readability_score(normalized_text) < 0.9:
        flags.append("readability_below_target")
    if content_match < 95:
        flags.append("content_match_below_95")
    if vocab_overlap < 97:
        flags.append("vocab_overlap_below_97")
    # structural checks
    if normalized_text.count("(") != normalized_text.count(")"):
        flags.append("unmatched_parentheses")
    if re.search(r"\|\s*$", normalized_text, re.M):
        flags.append("hanging_pipe")

    return {
        "normalized_text": normalized_text,
        "readability_score_pct": readability,
        "content_match_score_pct": max(0, min(100, content_match)),
        "vocab_overlap_pct": max(0, min(100, vocab_overlap)),
        "changes": changes,
        "flags": sorted(set(flags)),
    }


def normalize_validate_once(pages: List[str], region: Optional[str], opts: Dict[str, Any]) -> Dict[str, Any]:
    """Internal single-pass (no recursive pass) used for idempotency check."""
    changes: List[Dict[str, Any]] = []
    flags: List[str] = []
    lines_pages: List[List[str]] = [p.split("\n") for p in pages]
    lines_pages = _dedupe_header_footer(lines_pages, changes)
    for i in range(len(lines_pages)):
        lns = lines_pages[i]
        lns = _normalize_bullets(lns, changes)
        lns = _close_parens(lns, changes, flags)
        lns = _apply_city_map(lns, region, opts, changes, flags)
        lines_pages[i] = lns
    sec = opts.get("section_names", _DEFAULT_SECTIONS)
    for i in range(len(lines_pages)):
        lines_pages[i] = _join_heading_dates(lines_pages[i], changes, sec)
        lines_pages[i] = _normalize_whitespace(lines_pages[i], sec, changes)
        txt = "\n".join(lines_pages[i])
        lines_pages[i] = _RE_TRIPLE_BLANKS.sub("\n\n", txt).split("\n")
    return {"text": _rebuild(lines_pages), "changes": changes}


# --------------- Quick self-check (optional) ---------------
if __name__ == "__main__":
    sample_pages = [
        "JOHN DOE\nEmail: john@example.com | GitHub | LinkedIn\n\nEDUCATION\nCHRIST UNIVERSITY\nData Science Master of technology\nBangalore\nAugust 2024 - August 2026\n\nAWS (S3",
        "See: https://linkedin.com/in/john and https://github.com/john\n\nEXPERIENCE\nACME LTD\nJan 2020 - Feb 2023",
    ]
    raw_text = "\f".join(sample_pages)
    out = normalize_validate(sample_pages, raw_text, region="IN")
    print({k: out[k] for k in ("readability_score_pct", "content_match_score_pct", "vocab_overlap_pct", "flags")})
