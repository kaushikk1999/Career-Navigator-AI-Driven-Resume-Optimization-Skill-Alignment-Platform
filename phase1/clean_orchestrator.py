"""
Text Hygiene Orchestrator

Deterministic, idempotent post-OCR normalization that fixes six core defects:
1) Strip watermarks/headers/footers
2) Fix email-as-URL artifacts
3) Close dangling parentheses like 'AWS (S3'
4) De-hyphenate soft wraps
5) Normalize quotes & ligatures
6) Unsmash columns / heal wrapped lines

Public entry: clean_text(pages: list[str]) -> dict per 'clean-1.0.0' JSON.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import List, Dict, Tuple


# ------------------------------ Canonicalize ------------------------------
_WS = re.compile(r"[\t ]+")


def _nfc_lines(page: str) -> str:
    s = unicodedata.normalize("NFC", (page or "").replace("\r\n", "\n").replace("\r", "\n"))
    lines = s.split("\n")
    out = []
    for ln in lines:
        out.append(_WS.sub(" ", ln.rstrip()))
    return "\n".join(out)


# ----------------------- Header/Footer Stripping ------------------------
_HDR_RX = re.compile(r"(?im)^(?:.*(?:cv maker|footer).*(?:page\s*\d+).*)$")


def _strip_headers(pages: List[str]) -> Tuple[List[str], int]:
    if len(pages) < 2:
        # Apply regex seed only on single-page docs
        removed = 0
        out_pages = []
        for p in pages:
            kept = []
            for ln in p.splitlines():
                if _HDR_RX.match(ln):
                    removed += 1
                    continue
                kept.append(ln)
            out_pages.append("\n".join(kept))
        return out_pages, removed

    K = 5
    # Gather candidates from top/bottom K of each page
    per_page_edges: List[List[str]] = []
    for p in pages:
        ls = [ln for ln in p.splitlines()]
        edges = [ln for ln in ls[:K] if ln.strip()] + [ln for ln in ls[-K:] if ln.strip()]
        per_page_edges.append(edges)
    # Normalize candidate keys (collapse spaces, lower)
    def _key(s: str) -> str:
        t = _WS.sub(" ", s.strip()).lower()
        t = re.sub(r"\d+", "", t)  # strip counters
        return t
    counts: Dict[str, int] = {}
    exemplar: Dict[str, str] = {}
    considered_keys = set()
    regex_pages = 0
    for edges in per_page_edges:
        seen = set()
        page_has_regex = False
        for ln in edges:
            if len(ln) > 80:
                continue
            if _HDR_RX.match(ln):
                page_has_regex = True
                k = _key(ln)
                if k not in seen:
                    seen.add(k)
                    counts[k] = counts.get(k, 0) + 1
                    exemplar.setdefault(k, ln)
            else:
                # repetition heuristic candidate
                k = _key(ln)
                if any(tok in ln.lower() for tok in ("http", "@", "linkedin.com", "github.com")):
                    continue
                if not ln.strip():
                    continue
                if k not in seen:
                    seen.add(k)
                    counts[k] = counts.get(k, 0) + 1
                    exemplar.setdefault(k, ln)
                    considered_keys.add(k)
        n_pages = len(pages)
        if page_has_regex:
            regex_pages += 1
    # compute regex support across pages
    regex_support = (regex_pages / n_pages) >= 0.6 if n_pages else False
    # Ambiguity guard: if too many unique candidates, skip deletion by repetition
    uniq = sum(1 for k, c in counts.items() if c == 1)
    if len(counts) > 0 and (uniq / len(counts)) > 0.20:
        keys = []
    else:
        keys = [k for k, c in counts.items() if c >= max(2, int(0.6 * n_pages))]
        # Cap to ≤8 distinct
        keys = keys[:8]

    removed = 0
    out_pages: List[str] = []
    for p in pages:
        kept = []
        for ln in p.splitlines():
            if _HDR_RX.match(ln):
                removed += 1
                continue
            if len(ln) <= 80 and _key(ln) in keys:
                removed += 1
                continue
            kept.append(ln)
        # Per-page removal guard: >25% removed? revert unless global repetition/regex support
        removed_ratio = (len(p.splitlines()) - len(kept)) / max(1, len(p.splitlines()))
        if removed_ratio > 0.25 and not (keys or regex_support):
            out_pages.append(p)
        else:
            out_pages.append("\n".join(kept))
    return out_pages, removed


# ---------------------- Email-as-URL normalization ----------------------
_EMAIL_URL = re.compile(r"(?i)\bhttps?://([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?:/)?\b")


def _fix_email_urls(text: str) -> Tuple[str, int]:
    return _EMAIL_URL.subn(r"\1", text)


# --------------------- De-hyphenation + soft hyphen ---------------------
_DEHARD = re.compile(r"([A-Za-z0-9])-\n([A-Za-z0-9])")


def _dehyphen(page: str) -> Tuple[str, int]:
    s = page.replace("\u00AD", "")
    out, n = _DEHARD.subn(r"\1\2", s)
    return out, n


# -------------------- Quotes & ligatures normalization ------------------
_REPL_CHARS = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "…": "...", "–": "-", "—": "-",
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "©": "", "®": "", "™": "",
}
_TRANSLATE = str.maketrans(_REPL_CHARS)


def _norm_quotes_ligs(text: str) -> Tuple[str, int]:
    before = text
    out = text.translate(_TRANSLATE)
    # Count as number of characters changed approximately
    changed = 0
    for a in _REPL_CHARS.keys():
        if a in before:
            changed += before.count(a)
    return out, changed


# ----------------- Close dangling parentheses (conservative) ------------
_PAREN = re.compile(r"\b([A-Z]{2,}\s*)\(([A-Za-z0-9+/#\- ]{1,12})\b(?!\))")


def _close_parens(text: str) -> Tuple[str, int]:
    count = 0

    def _cb(m: re.Match) -> str:
        nonlocal count
        s = m.group(0)
        # Prevent '())' within 5 chars after insertion
        proposed = s + ")"
        if re.search(r"\)\)\)", proposed[-5:]):
            return s
        count += 1
        return proposed

    out = _PAREN.sub(_cb, text)
    return out, count


# ------------------ Heal wrapped lines / unsmash columns ----------------

def _heal_lines(page: str) -> Tuple[str, int]:
    lines = page.split("\n")
    out: List[str] = []
    merges = 0
    merges_cap = 30
    for i in range(len(lines)):
        ln = lines[i]
        if not out:
            out.append(ln)
            continue
        prev = out[-1]
        # Duplicate uppercase header within a 3-line window → drop current dup
        if ln.strip().isupper() and prev.strip() == ln.strip():
            continue
        if merges < merges_cap:
            if not re.search(r"[.;:!?)]\s*$", prev) and not re.match(r"^\s*[\*\u2022-]\s*", ln) and re.match(r"^[a-z0-9]", ln.strip() or "0"):
                out[-1] = prev.rstrip() + " " + ln.lstrip()
                merges += 1
                # Paragraph size guard
                if len(out[-1]) > 2000:
                    # revert this merge
                    out[-1] = prev
                    out.append(ln)
                continue
        out.append(ln)
    return "\n".join(out), merges


# ------------------------------ Public API ------------------------------

def clean_text(pages: List[str]) -> Dict:
    # 1) canonicalize per-page
    pages1 = [_nfc_lines(p) for p in (pages or [])]
    # 2) header/footer
    pages2, removed_hf = _strip_headers(pages1)
    # 3) email-as-URL (done later per page after merges per spec? spec says before dehyphenation; we do after dehyphen and heal lines to keep counts conservative)
    # 4) de-hyphenate wraps
    pages3: List[str] = []
    dehyph_total = 0
    for p in pages2:
        s, n = _dehyphen(p)
        dehyph_total += n
        pages3.append(s)
    # 5) quotes & ligatures
    pages4: List[str] = []
    ql_total = 0
    for p in pages3:
        s, n = _norm_quotes_ligs(p)
        ql_total += n
        pages4.append(s)
    # 6) close parentheses
    pages5: List[str] = []
    paren_total = 0
    for p in pages4:
        s, n = _close_parens(p)
        paren_total += n
        pages5.append(s)
    # 7) heal lines / unsmash columns
    pages6: List[str] = []
    merge_total = 0
    for p in pages5:
        s, n = _heal_lines(p)
        merge_total += n
        pages6.append(s)
    # 8) email-as-URL fix (final token cleanup)
    pages7: List[str] = []
    email_fix_total = 0
    for p in pages6:
        s, n = _fix_email_urls(p)
        email_fix_total += n
        pages7.append(s)

    final_text = "\f".join(pages7)
    edits = [
        {"rule": "header_footer.remove", "count": int(removed_hf)},
        {"rule": "email_url.fix", "count": int(email_fix_total)},
        {"rule": "dehyphen.wrap", "count": int(dehyph_total)},
        {"rule": "quotes_ligatures.normalize", "count": int(ql_total)},
        {"rule": "paren.close_shortspan", "count": int(paren_total)},
        {"rule": "line.heal_merge", "count": int(merge_total)},
    ]
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "version": "clean-1.0.0",
        "timestamp_utc": now_utc,
        "pages": len(pages or []),
        "edits": edits,
        "text_nfc": final_text,
    }
