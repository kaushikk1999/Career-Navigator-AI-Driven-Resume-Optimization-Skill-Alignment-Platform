"""
Canonical Post-Processing & Normalization Module

Implements a deterministic, language-agnostic pipeline with the exact order:
1) NFC + whitespace collapse (per page)
2) Header/footer repeat removal (top/bottom-K lines, ≥60% support)
3) De-hyphenate line wraps
4) Paragraph healing (preserve bullets/headings)
5) Email/URL normalization (email-as-URL fix)
6) Balance simple pairs (line-local parentheses)
7) Bullets & ligatures normalization
8) Placeholder cleanup (bare language tags)
9) Assemble pages and compute metrics/flags

Returns the JSON contract in the user spec.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Tuple, Optional


# ------------------------------- Utilities --------------------------------

_WS = re.compile(r"[\t ]+")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _normalize_lines_keep_breaks(page: str) -> Tuple[str, int]:
    """Collapse horizontal whitespace per line, trim ends; keep newlines intact.
    Returns (normalized_page, changed_count)."""
    changed = 0
    lines = page.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: List[str] = []
    for ln in lines:
        new = _WS.sub(" ", ln.rstrip())
        if new != ln:
            changed += 1
        out.append(new)
    return "\n".join(out), changed


def _norm_key(s: str) -> str:
    s = _WS.sub(" ", s.strip()).lower()
    s = re.sub(r"\d+", "", s)
    return s


def _is_sentence_like(s: str) -> bool:
    if "." not in s:
        return False
    words = re.findall(r"\b\w+\b", s)
    return len(words) >= 6


def _looks_header_footer_candidate(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if len(t) > 120:
        return False
    if _is_sentence_like(t):
        return False
    if "@" in t or "http" in t or "|" in t:
        return False
    return True


def _is_heading_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if t.isupper() and len(t.split()) <= 4 and len(t) <= 64 and not t.endswith(('.', ',', ';', ':')):
        return True
    if re.match(r"^[A-Z][A-Za-z &/]+$", t) and not t.endswith((':', ';', '.')):
        return True
    return False


def _readability_proxy(text: str) -> float:
    lines = text.split("\n")
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
    bullets = sum(1 for ln in non_empty if ln.lstrip().startswith("• ") or ln.lstrip().startswith("- ") or ln.lstrip().startswith("* "))
    headings = sum(1 for ln in non_empty if _is_heading_line(ln))
    bonus = min(0.1, 0.5 * (bullets / max(1, len(non_empty))) + 0.5 * (headings / max(1, len(non_empty))))
    return round(min(1.0, 0.9 * base + bonus), 3)


# --------------------------- Core transforms -----------------------------

def _strip_repeats(pages: List[str], K: int = 5, support: float = 0.60, cap_unique: int = 8) -> Tuple[List[str], List[str], int, int]:
    split = [p.split("\n") for p in pages]
    if len(split) < 2:
        return pages, [], 0, sum(len(ls) for ls in split)
    counts: Dict[str, int] = {}
    exemplar: Dict[str, str] = {}
    for lines in split:
        top = [ln for ln in lines[:K] if ln.strip()]
        bot = [ln for ln in lines[-K:] if ln.strip()]
        seen = set()
        for ln in top + bot:
            if not _looks_header_footer_candidate(ln):
                continue
            key = _norm_key(ln)
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
            exemplar.setdefault(key, ln.strip())
    n_pages = len(split)
    keys = [k for k, c in counts.items() if (c / n_pages) >= support]
    # Cap distinct boilerplate lines
    keys = keys[:cap_unique]
    removed_names = [exemplar[k] for k in keys]
    total_lines_before = sum(len(ls) for ls in split)
    out_pages: List[str] = []
    for lines in split:
        before_n = len(lines)
        new_lines: List[str] = []
        removed_this = 0
        for ln in lines:
            if _norm_key(ln) in keys:
                removed_this += 1
                continue
            new_lines.append(ln)
        # Per-page guard: avoid stripping >25% of lines on a page unless page is very short
        if before_n > 0 and removed_this / before_n > 0.25:
            # revert this page
            new_lines = lines
        out_pages.append("\n".join(new_lines))
    removed_total = total_lines_before - sum(len(p.split("\n")) for p in out_pages)
    return out_pages, removed_names, removed_total, total_lines_before


_HYPH_JOINS = re.compile(r"(\w)-\n(\w)")


def _dehyphenate(page: str) -> Tuple[str, int]:
    return _HYPH_JOINS.subn(r"\1\2", page)


def _heal_paragraphs(page: str) -> Tuple[str, int]:
    # replace single newline not preceded by terminal punctuation and not followed by bullet/number/header
    pattern = re.compile(r"(?<![.!?:])\n(?!\s*(?:[-*•]|\d+\.|[A-Z][A-Z ]{2,}$))")
    out, n = pattern.subn(" ", page)
    # collapse 3+ blanklines -> 2
    out2 = re.sub(r"\n{3,}", "\n\n", out)
    return out2, n


_URL_EMAIL = re.compile(r"(?i)\b((?:https?|ftp)://)([^/\s]+)([^\s]*)")
_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def _normalize_email_urls(text: str) -> Tuple[str, bool]:
    changed = False
    def _cb(m: re.Match) -> str:
        nonlocal changed
        scheme, auth, path = m.group(1), m.group(2), m.group(3)
        if "@" in auth:
            # extract email part from auth
            email = auth
            if _EMAIL.fullmatch(email):
                changed = True
                return email
            # ambiguous: drop full token
            changed = True
            return ""
        return m.group(0)
    out = _URL_EMAIL.sub(_cb, text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out, changed


_PAREN_ABBR = {"S3", "GPU", "CPU", "API", "IAM", "SQL", "NLP", "GCP", "EC2", "EKS"}


def _balance_parens_line(line: str) -> Tuple[str, int]:
    # Only heal when exactly one unmatched '('
    opens, closes = line.count("("), line.count(")")
    if opens == closes + 1:
        m = re.search(r"\((?P<x>[A-Za-z0-9]{1,6})\s*$", line)
        if m:
            token = m.group("x")
            if token.upper() in _PAREN_ABBR:
                return line + ")", 1
    return line, 0


def _balance_parens(text: str) -> Tuple[str, int]:
    lines = text.split("\n")
    out: List[str] = []
    count = 0
    for ln in lines:
        new, c = _balance_parens_line(ln)
        count += c
        out.append(new)
    return "\n".join(out), count


def _normalize_bullets_ligatures(text: str, bullet: str = "•") -> Tuple[str, int, bool]:
    # bullets at line start
    before = text
    text = re.sub(r"(?m)^[ \t]*[-*•][ \t]+", f"{bullet} ", text)
    # ligatures and quotes
    repls = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
        "“": '"', "”": '"', "‘": "'", "’": "'",
    }
    changed = 0
    for a, b in repls.items():
        if a in text:
            changed += text.count(a)
            text = text.replace(a, b)
    return text, (0 if before == text else 1), bool(changed)


_PLACEHOLDER_CODES = {"hi","bn","ar","he","ta","gu","te","kn","ml","mr","pa","ur","fa","zh","ja","ko"}


def _cleanup_placeholders(text: str) -> Tuple[str, int]:
    # remove bare (xx) placeholders when isolated by whitespace or start/end
    pattern = re.compile(r"(?<!\w)\((%s)\)(?!\w)" % "|".join(sorted(_PLACEHOLDER_CODES)))
    out, n = pattern.subn("", text)
    # collapse spaces that might be left
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out, n


def canonical_postproc(pages: List[str], bullet_symbol: str = "•") -> Dict:
    # Raw input snapshot and baseline metrics
    raw_text = "\f".join(pages or [])
    raw_len = len(raw_text)
    raw_lines_total = sum(len((_nfc(p) or "").replace("\r\n","\n").replace("\r","\n").split("\n")) for p in (pages or []))

    # 1) NFC + per-line whitespace collapse
    norm_pages: List[str] = []
    ws_changes = 0
    for p in (pages or []):
        s = _nfc(p)
        s2, c = _normalize_lines_keep_breaks(s)
        ws_changes += c
        norm_pages.append(s2)

    # 2) Header/footer stripping
    stripped_pages, removed_names, removed_total, total_lines_before = _strip_repeats(norm_pages)

    # 3) De-hyphenate wraps (per page)
    dehyph_total = 0
    stage3_pages: List[str] = []
    for p in stripped_pages:
        s3, joins = _dehyphenate(p)
        dehyph_total += joins
        stage3_pages.append(s3)

    # 4) Paragraph healing
    healed_total = 0
    stage4_pages: List[str] = []
    for p in stage3_pages:
        s4, nheal = _heal_paragraphs(p)
        healed_total += nheal
        stage4_pages.append(s4)

    # 5) Email/URL normalization
    email_fixed_any = False
    stage5_pages: List[str] = []
    for p in stage4_pages:
        s5, ch = _normalize_email_urls(p)
        email_fixed_any = email_fixed_any or ch
        stage5_pages.append(s5)

    # 6) Balance simple pairs
    paren_delta_total = 0
    stage6_pages: List[str] = []
    for p in stage5_pages:
        s6, delta = _balance_parens(p)
        paren_delta_total += delta
        stage6_pages.append(s6)

    # 7) Bullets & ligatures
    bullets_norm_flag = False
    ligatures_flag = False
    stage7_pages: List[str] = []
    for p in stage6_pages:
        s7, bullets_changed, ligs = _normalize_bullets_ligatures(p, bullet_symbol)
        bullets_norm_flag = bullets_norm_flag or bool(bullets_changed)
        ligatures_flag = ligatures_flag or ligs
        stage7_pages.append(s7)

    # 8) Placeholder cleanup
    placeholders_removed = 0
    stage8_pages: List[str] = []
    for p in stage7_pages:
        s8, nrm = _cleanup_placeholders(p)
        placeholders_removed += nrm
        stage8_pages.append(s8)

    # Assemble final
    final_text = "\f".join(stage8_pages)

    # Flags & metrics
    headers_removed_flag = bool(removed_names)
    pct_lines_removed = (100.0 * removed_total / total_lines_before) if total_lines_before else 0.0
    pct_lines_healed = (100.0 * healed_total / max(1, raw_lines_total))
    length_ratio = (100.0 * len(final_text) / max(1, raw_len))
    readability = _readability_proxy(final_text)

    out = {
        "version": "postproc-1.0",
        "pages": len(pages or []),
        "text": final_text,
        "flags": {
            "headers_removed": headers_removed_flag,
            "email_url_fixed": bool(email_fixed_any),
            "paren_balanced_lines": int(paren_delta_total),
            "dehyphenated_joins": int(dehyph_total),
            "bullets_normalized": bullet_symbol if bullets_norm_flag else bullet_symbol,
            "ligatures_replaced": bool(ligatures_flag),
            "placeholders_removed": int(placeholders_removed),
        },
        "metrics": {
            "pct_lines_removed_as_headers": round(pct_lines_removed, 1),
            "pct_lines_healed": round(pct_lines_healed, 1),
            "paren_balance_delta": int(paren_delta_total),
            "length_ratio_vs_raw": round(length_ratio, 1),
            "readability_proxy": readability,
        },
        "errors": [],
    }
    return out

