"""
Finisher (Spec-Aligned): header/footer stripping, email-URL cleanup,
dangling parenthesis patching, and column-merge hints with combined output.

All logic is deterministic, idempotent, CPU-light, and non-generative.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple, Optional


# ------------------------------- Utilities --------------------------------

_WS = re.compile(r"[\t ]+")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _norm_token(s: str) -> str:
    s = _nfc(s).strip()
    s = _WS.sub(" ", s)
    s = s.lower()
    s = re.sub(r"\d+", "", s)  # strip dynamic counters
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
    if "@" in t or "http" in t:
        return False
    if "|" in t:
        return False
    if _is_sentence_like(t):
        return False
    return True


def _is_header(s: str) -> bool:
    t = s.strip()
    return bool(t.isupper() and len(t) >= 6 and len(t.split()) <= 3)


def _is_label(s: str) -> bool:
    return bool(re.match(r"(?i)^\s*[A-Za-z ]{2,20}:", s))


# ---------------------- 1) Header/Footer Stripping ------------------------

def strip_boilerplate(
    pages: List[str], K: int = 5, threshold: float = 0.60, max_unique: int = 8
) -> Dict:
    pages = [(_nfc(p) if p else "") for p in pages]
    split = [p.split("\n") for p in pages]
    if len(split) < 2:
        return {
            "pages_clean": pages,
            "boilerplate_removed": [],
            "threshold": threshold,
            "lines_considered": min(K, len(pages[0].split("\n"))) * 2 if pages else 0,
        }
    counts: Dict[str, int] = {}
    exemplar: Dict[str, str] = {}
    considered = 0
    for lines in split:
        top = [ln for ln in lines[:K] if ln.strip()]
        bot = [ln for ln in lines[-K:] if ln.strip()]
        considered += len(top) + len(bot)
        seen = set()
        for ln in top + bot:
            if not _looks_header_footer_candidate(ln):
                continue
            key = _norm_token(ln)
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
            exemplar.setdefault(key, ln.strip())
    n_pages = len(split)
    # pick boiler by support
    boiler = [(k, c) for k, c in counts.items() if (c / n_pages) >= threshold]
    # sort by freq desc then by first appearance for stability
    boiler.sort(key=lambda x: (-x[1], list(counts.keys()).index(x[0])))
    # cap unique removals
    boiler_keys = [k for k, _ in boiler[:max_unique]]
    removed_names = [exemplar[k] for k in boiler_keys]

    # Apply with global fail-safe: do not remove >20% of lines overall
    total_lines = sum(len(ls) for ls in split)
    removed_total = 0
    cleaned_pages: List[str] = []
    for lines in split:
        new_lines: List[str] = []
        # code-fence awareness
        fenced = False
        for ln in lines:
            if ln.strip().startswith("```"):
                fenced = not fenced
            key = _norm_token(ln)
            if (not fenced) and ("<" not in ln and ">" not in ln) and key in boiler_keys:
                removed_total += 1
                continue
            new_lines.append(ln)
        cleaned_pages.append("\n".join(new_lines))

    # Fail-safe: only abort if many distinct boiler lines AND excessive removals
    if removed_names and (len(boiler_keys) > 2) and (removed_total > max(1, int(0.20 * total_lines))):
        # Abort stripping, return original pages
        return {
            "pages_clean": pages,
            "boilerplate_removed": [],
            "threshold": threshold,
            "lines_considered": min(K * 2, considered),
        }

    return {
        "pages_clean": cleaned_pages,
        "boilerplate_removed": removed_names,
        "threshold": threshold,
        "lines_considered": min(K * 2, considered),
    }


# ---------------------- 2) Email-as-URL Cleaner --------------------------

_URL_WITH_AUTH = re.compile(r"(?i)\b((?:https?|ftp)://([^/\s]+)([^\s]*))")


def clean_email_url(text: str) -> Dict:
    removed: List[str] = []

    def _cb(m: re.Match) -> str:
        full = m.group(1)
        auth = m.group(2)
        if "@" in auth:
            removed.append(full)
            return ""
        return full

    out = _URL_WITH_AUTH.sub(_cb, text)
    # Collapse any double spaces introduced by removals
    out = re.sub(r"[ \t]{2,}", " ", out)
    return {"text": out, "removed": removed}


# ------------------- 3) Dangling Parenthesis Patcher --------------------

_KNOWN_ABBR = {"S3", "EC2", "EKS", "GPU", "CPU", "API", "IAM", "SQL", "NLP"}
_KNOWN_BRANDS = {"AWS", "GCP", "Azure", "Apache", "Google", "Hadoop", "Spark", "Kafka", "Linux", "Windows"}


def patch_dangling_parens(text: str, max_patches: int = 5) -> Dict:
    patched: List[str] = []
    count = 0

    def _cb(m: re.Match) -> str:
        nonlocal count
        brand = m.group(1)
        token = m.group(2)
        if count >= max_patches:
            return m.group(0)
        if brand.upper() in _KNOWN_BRANDS and token.upper() in _KNOWN_ABBR:
            count += 1
            patched.append(f"({token} → ({token}))")
            return f"{brand} ({token})"
        return m.group(0)

    # Heal only when preceded by brand and not already closed
    out = re.sub(r"\b([A-Za-z][A-Za-z0-9+/.-]{1,20})\s*\(([A-Za-z0-9]{1,6})(?![^)]*\))", _cb, text)
    return {"text": out, "patched": patched}


# ---------------------- 4) Column-Merge Hints ---------------------------

LABELS = {"programming", "frameworks", "tools", "platforms", "databases", "languages", "certifications", "education"}


def merge_label_rows(lines: List[str], delimiter: str = "|") -> Dict:
    norm_delim = delimiter
    merges = 0
    out: List[str] = []
    i = 0
    cap_consecutive = 4
    while i < len(lines):
        cur = lines[i].rstrip()
        # normalize trailing delimiter spacing
        cur = re.sub(r"\s*\|\s*$", " | ", cur)
        if (_is_label(cur) or cur.endswith("|")) and i + 1 < len(lines):
            nxt = lines[i + 1].lstrip()
            if _is_header(nxt):
                out.append(cur)
                i += 1
                continue
            if _is_label(nxt) or re.match(r"^[A-Z0-9*]", nxt):
                buf = cur.rstrip()
                used = 1
                while i + used < len(lines) and used <= cap_consecutive:
                    nn = lines[i + used].strip()
                    if not nn:
                        break
                    if _is_header(nn):
                        break
                    if _is_label(nn) or re.match(r"^[A-Z0-9*]", nn):
                        buf = re.sub(r"\s*\|\s*$", " | ", buf)
                        buf = buf + " " + nn
                        merges += 1
                        used += 1
                        # stop if too long
                        if len(buf) > 2000:
                            break
                        continue
                    break
                # normalize hard separators into chosen delimiter
                buf2 = re.sub(r"\s*[•\-]\s*", f" {norm_delim} ", buf)
                buf2 = re.sub(r"\s*\|\s*", f" {norm_delim} ", buf2)
                out.append(buf2)
                i += used
                continue
        out.append(cur)
        i += 1
    return {"lines": out, "merges": merges, "normalized_delimiter": norm_delim}


# -------------------- Combined end-to-end finisher ----------------------

def finish_spec(pages: List[str], opts: Optional[Dict] = None) -> Dict:
    opts = opts or {}
    # 1) Normalize whitespace lightly (no title-casing)
    base_pages = [re.sub(r"[\t ]+", " ", _nfc(p)).strip("\f") for p in pages]

    # 2) Header/Footer stripping
    h = strip_boilerplate(base_pages)
    pages2 = h["pages_clean"]

    # 3) Email-as-URL removal per page (token-level)
    removed_tokens: List[str] = []
    pages3: List[str] = []
    for p in pages2:
        res = clean_email_url(p)
        pages3.append(res["text"])
        removed_tokens.extend(res["removed"])

    # 4) Dangling paren patch per page
    patched_total: List[str] = []
    pages4: List[str] = []
    for p in pages3:
        res = patch_dangling_parens(p)
        pages4.append(res["text"])
        patched_total.extend(res["patched"])

    # 5) Column merge pass per page
    merges_total = 0
    pages5: List[str] = []
    for p in pages4:
        lines = p.split("\n")
        # skip fenced blocks
        fenced = False
        safe_lines: List[str] = []
        for ln in lines:
            if ln.strip().startswith("```"):
                fenced = not fenced
            if fenced:
                safe_lines.append(ln)
                continue
            res = merge_label_rows([ln]) if False else None
            # merging needs context; apply on whole list instead
        res_all = merge_label_rows(lines)
        pages5.append("\n".join(res_all["lines"]))
        merges_total += res_all["merges"]

    # 6) Final tidy: de-hyphenate wraps, ligature/quotes, collapse blanklines
    ZERO_WIDTH = dict.fromkeys([0x00AD, 0x200B, 0x200C, 0x200D], None)
    def _final_tidy(s: str) -> str:
        s = s.translate(ZERO_WIDTH)
        s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s
    pages_final = [_final_tidy(p) for p in pages5]
    text = "\f".join(pages_final)

    return {
        "text": text,
        "audit": {
            "boilerplate_removed": h.get("boilerplate_removed", []),
            "email_url_removed": removed_tokens,
            "paren_patched": patched_total,
            "column_merges": merges_total,
        },
    }
