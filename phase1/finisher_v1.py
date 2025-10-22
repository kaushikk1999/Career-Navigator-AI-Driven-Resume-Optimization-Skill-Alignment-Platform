"""
Text Normalization & QA Finisher (spec-driven, deterministic, idempotent)

Implements the ordered rules from the user's spec:
1) Page segmentation preserved; output joins pages with \f
2) Header/Footer repeat-line filter (>=60% pages, first/last 5 lines)
3) Email-as-URL line artifacts removal (http(s)://...@...)
4) Hyphenation healing for wrapped words; remove soft/zero-width chars
5) Quote/ligature normalization; bullet unification; horizontal space folding
6) Optional parenthesis healing for known acronyms (guarded)
7) Paragraph healing: join wrapped lines conservatively
8) Script safety: no transliteration/reordering (implicit)
9) No fabrication guard and audit of skipped rules

Public entry:
    finish(pages: list[str], opts: dict | None = None) -> dict (JSON-compatible)
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Tuple, Optional


# ------------------------ Compiled regex and constants ------------------------
_EMAIL_URL_ART = re.compile(r"(?mi)^(?:https?://)\S+@\S+/?\s*$")
_WS = re.compile(r"[\t ]+")
_HYPH_WRAP = re.compile(r"(\w)-\n(\w)")
_BULLET_LINE = re.compile(r"(?m)^[ \t]*[-*•][ \t]+")
_ALL_CAPS_SHORT = re.compile(r"^[A-Z][A-Z &/.'-]{1,63}$")

_QUOTE_MAP = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}
_LIG_MAP = {"ﬁ": "fi", "ﬂ": "fl"}

_ZERO_WIDTH = dict.fromkeys([0x00AD, 0x200B, 0x200C, 0x200D], None)

_KNOWN_ACRONYMS = {
    "S3", "EC2", "EKS", "ECS", "RDS", "SNS", "SQS", "GPU", "NLP", "GKE", "ELB",
}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _split_lines_per_page(pages: List[str]) -> List[List[str]]:
    return [(_nfc(p) if p else "").replace("\r\n", "\n").replace("\r", "\n").split("\n") for p in pages]


def _join_pages(pages_lines: List[List[str]]) -> str:
    return "\f".join("\n".join(ls) for ls in pages_lines)


def _edges(lines: List[str]) -> List[str]:
    top = [ln.strip() for ln in lines[:5] if ln.strip()]
    bot = [ln.strip() for ln in lines[-5:] if ln.strip()]
    # de-duplicate within page
    out = []
    seen = set()
    for ln in top + bot:
        if ln not in seen:
            out.append(ln)
            seen.add(ln)
    return out


def _detect_repeated_edges(pages_lines: List[List[str]]) -> List[str]:
    if len(pages_lines) < 2:
        return []
    freq: Dict[str, int] = {}
    for lines in pages_lines:
        for ln in _edges(lines):
            if len(ln) <= 120:
                freq[ln] = freq.get(ln, 0) + 1
    threshold = max(2, int(0.6 * len(pages_lines) + 0.9999))
    return [ln for ln, c in freq.items() if c >= threshold]


def _is_heading(ln: str) -> bool:
    t = ln.strip()
    if not t:
        return False
    if _ALL_CAPS_SHORT.match(t):
        # Heuristic: 1-4 words considered headings
        return len(t.split()) <= 6
    return False


def finish(pages: List[str], opts: Optional[Dict] = None) -> Dict:
    assert isinstance(pages, list), "pages must be a list of strings"
    opts = opts or {}
    prefer_bullets = opts.get("normalize_bullets", "•|*")
    bullet_symbol = (prefer_bullets.split("|")[0] or "•").strip() or "•"
    heal_paren = bool(opts.get("heal_dangling_paren", True))
    drop_email_url = bool(opts.get("drop_email_url_artifacts", True))

    # Snapshot input for no-op detection
    input_text = "\f".join(pages)

    audit = {
        "removed_headers_footers": [],
        "removed_email_urls": [],
        "normalized_parens": [],
        "hyphen_joins": 0,
        "quote_ligature_normalizations": 0,
        "bullet_style": bullet_symbol,
        "skipped_rules": [],
    }
    errors: List[str] = []

    # 1) Normalize to NFC and split lines per page
    pages_lines = _split_lines_per_page(pages)

    # 2) Header/Footer repeat filter
    repeated = _detect_repeated_edges(pages_lines)
    if repeated:
        # record unique removed
        audit["removed_headers_footers"] = list(repeated)
        for i, lines in enumerate(pages_lines):
            pages_lines[i] = [ln for ln in lines if ln.strip() not in repeated]

    # 3) Email-as-URL artifact filter
    if drop_email_url:
        removed_email: List[str] = []
        for i, lines in enumerate(pages_lines):
            kept: List[str] = []
            for ln in lines:
                if _EMAIL_URL_ART.match(ln.strip()):
                    removed_email.append(ln.strip())
                    continue
                kept.append(ln)
            pages_lines[i] = kept
        if removed_email:
            # Deduplicate while preserving order
            seen = set()
            out = []
            for x in removed_email:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            audit["removed_email_urls"] = out

    # 4) Hyphenation healing and zero-width removal (per page)
    for i, lines in enumerate(pages_lines):
        text = "\n".join(lines)
        text = text.translate(_ZERO_WIDTH)
        text, n_hyp = _HYPH_WRAP.subn(r"\1\2", text)
        audit["hyphen_joins"] += n_hyp
        pages_lines[i] = text.split("\n")

    # 5) Quotes/ligatures normalization, bullet unification, spacing
    def _norm_line(ln: str) -> Tuple[str, int, bool]:
        count = 0
        s = ln
        # quotes
        for a, b in _QUOTE_MAP.items():
            if a in s:
                count += s.count(a)
                s = s.replace(a, b)
        # ligatures
        for a, b in _LIG_MAP.items():
            if a in s:
                count += s.count(a)
                s = s.replace(a, b)
        # bullets
        s2 = _BULLET_LINE.sub(f"{bullet_symbol} ", s)
        is_bullet = bool(s2 is not s and s2.lstrip().startswith(f"{bullet_symbol} ")) or s2.lstrip().startswith(f"{bullet_symbol} ")
        # whitespace collapse (horizontal only)
        s2 = _WS.sub(" ", s2.rstrip())
        return s2, count, is_bullet

    bullet_lines: List[Tuple[int, int]] = []  # (page_idx, line_idx)
    for i, lines in enumerate(pages_lines):
        new_lines: List[str] = []
        for j, ln in enumerate(lines):
            s, c, is_bullet = _norm_line(ln)
            audit["quote_ligature_normalizations"] += c
            if is_bullet:
                bullet_lines.append((i, len(new_lines)))
            new_lines.append(s)
        pages_lines[i] = new_lines

    # 6) Optional paren healing (dangling)
    if heal_paren:
        healed: List[Tuple[str, str]] = []
        pat = re.compile(r"([A-Za-z0-9+/_-]+)\s*\(([A-Z0-9]{1,6})(?![^)]*\))")
        for i, lines in enumerate(pages_lines):
            for j, ln in enumerate(lines):
                def _cb(m: re.Match) -> str:
                    before = m.group(0)
                    acro = m.group(2)
                    if acro.upper() in _KNOWN_ACRONYMS:
                        after = before + ")"
                        healed.append((before, after))
                        return after
                    return before
                new_ln = pat.sub(_cb, ln)
                pages_lines[i][j] = new_ln
        if healed:
            # dedupe but keep insertion order
            seen = set()
            out = []
            for a, b in healed:
                key = (a, b)
                if key in seen:
                    continue
                out.append({"from": a, "to": b})
                seen.add(key)
            audit["normalized_parens"] = out

    # 7) Paragraph healing (conservative)
    def _should_join(prev: str, nxt: str) -> bool:
        if not prev or not nxt:
            return False
        if _BULLET_LINE.match(nxt) or _is_heading(nxt) or _is_heading(prev):
            return False
        if prev.rstrip().endswith((".", "!", "?", ":", ";")):
            return False
        # Likely wrap when next starts lowercase or punctuation
        t = nxt.lstrip()
        return bool(t and (t[:1].islower() or t[:1] in ",.;:)]}"))

    for i, lines in enumerate(pages_lines):
        out: List[str] = []
        k = 0
        while k < len(lines):
            cur = lines[k]
            if out and _should_join(out[-1], cur):
                out[-1] = out[-1].rstrip() + " " + cur.lstrip()
                k += 1
                continue
            out.append(cur)
            k += 1
        # Ensure headings are surrounded by blank lines (lightly)
        spaced: List[str] = []
        for idx, ln in enumerate(out):
            if _is_heading(ln):
                if spaced and spaced[-1] != "":
                    spaced.append("")
                spaced.append(ln.strip())
                if idx + 1 < len(out) and out[idx + 1].strip() != "":
                    spaced.append("")
            else:
                spaced.append(ln.rstrip())
        pages_lines[i] = spaced

    # Compose final text
    final_text = _join_pages(pages_lines)

    # Metrics
    per_page = [{"index": i, "chars": len("\n".join(ls))} for i, ls in enumerate(pages_lines)]
    length_chars = len(final_text)
    length_tokens = len(re.findall(r"\S+", final_text))

    # No-op guard
    if (not audit["removed_headers_footers"] and not audit["removed_email_urls"]
        and not audit["normalized_parens"] and audit["hyphen_joins"] == 0
        and audit["quote_ligature_normalizations"] == 0 and final_text == input_text):
        audit["skipped_rules"].append("no_op")

    return {
        "phase": "1.0.0",
        "inputs": {
            "pages": pages,
            "opts": {
                "normalize_bullets": prefer_bullets,
                "heal_dangling_paren": heal_paren,
                "drop_email_url_artifacts": drop_email_url,
            },
        },
        "result": {
            "text_normalized": final_text,
            "per_page": per_page,
        },
        "audit": audit,
        "metrics": {
            "length_chars": length_chars,
            "length_tokens": length_tokens,
        },
        "errors": errors,
    }

