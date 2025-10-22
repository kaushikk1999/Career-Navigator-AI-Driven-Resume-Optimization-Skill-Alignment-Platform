"""
Spec-accurate post-OCR cleanup utilities and orchestrator.

Implements six core fixes and a deterministic, idempotent pipeline:
 1) strip_repeated_headers_footers
 2) drop_email_url_lines
 3) heal_dangling_parens (AWS tokens)
 4) heal_wraps (de-hyphen + paragraph heal)
 5) normalize_glyphs (quotes/ligatures; NFC)
 6) dedupe_headings (duplicate headings + spacing)

Provides:
 - clean_pages(pages) -> (text, meta)
 - make_output_json(pages) -> dict (phase/results/metrics/errors)
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import List, Tuple, Dict


# ------------------------- 1) Header/Footer Strip -------------------------

SECTION_RE = re.compile(r"^(OBJECTIVE|EDUCATION|EXPERIENCE|SKILLS|PROJECTS|CREDENTIALS)\b", re.I)


def strip_repeated_headers_footers(pages: List[str], freq: float = 0.6) -> Tuple[List[str], List[str], bool]:
    bands: List[set[str]] = []
    for p in pages:
        L = [ln.strip() for ln in p.splitlines()]
        bands.append(set([*L[:5], *L[-5:]]))
    cnt = Counter(ln for s in bands for ln in s if ln)
    thresh = max(2, int(freq * max(1, len(pages))))
    common = {ln for ln, c in cnt.items() if c >= thresh and not SECTION_RE.search(ln)}
    cleaned: List[str] = []
    for p in pages:
        keep = [ln for ln in p.splitlines() if ln.strip() not in common]
        cleaned.append("\n".join(keep))
    return cleaned, sorted(common), bool(common)


# ------------------------- 2) Email-as-URL Cleaner ------------------------

EMAIL_URL_RE = re.compile(r"^\s*https?://\S+@\S+/?\s*$", re.I)


def drop_email_url_lines(p: str) -> Tuple[str, List[str]]:
    kept, removed = [], []
    for ln in p.splitlines():
        if EMAIL_URL_RE.match(ln):
            removed.append(ln)
        else:
            kept.append(ln)
    return "\n".join(kept), removed


# ---------------------- 3) Dangling Parenthesis Healer --------------------

DANGLING_RE = re.compile(r"\bAWS\s*\((S3|EC2|RDS|ECS|EKS)\b(?!\))", re.I)


def heal_dangling_parens(s: str) -> str:
    healed: List[str] = []
    for line in s.splitlines():
        m = DANGLING_RE.search(line)
        if m and ")" not in line[m.start() : m.end() + 20]:
            line = line[: m.end()] + ")" + line[m.end() :]
        healed.append(line)
    return "\n".join(healed)


# --------------------- 4) Hyphenation & Wrap Healing ----------------------

SOFT_HYPHEN = "\u00AD"
DEHYPH = re.compile(r"(\w)-\n(\w)")
LINE_HEAL = re.compile(r"(?<![.!?:])\n(?!\s*[-*•])")


def heal_wraps(s: str) -> str:
    s = s.replace(SOFT_HYPHEN, "")
    s = DEHYPH.sub(r"\1\2", s)
    s = LINE_HEAL.sub(" ", s)
    return s


# --------------------- 5) Ligatures & Smart Quotes -----------------------

LIG_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


def normalize_glyphs(s: str) -> str:
    for k, v in LIG_MAP.items():
        s = s.replace(k, v)
    return unicodedata.normalize("NFC", s)


# --------- 6) Column Smash Remnants (duplicate headings & spaces) ---------

HEAD_RE = re.compile(
    r"^(OBJECTIVE|EDUCATION|EXPERIENCE|SKILLS|PROJECTS(?:\s*&\s*OPEN-?SOURCE)?|CREDENTIALS)\b",
    re.I,
)


def dedupe_headings(s: str) -> Tuple[str, int]:
    lines = s.splitlines()
    out, dedup = [], 0
    i = 0
    while i < len(lines):
        out.append(lines[i])
        if HEAD_RE.search(lines[i]) and i + 1 < len(lines) and HEAD_RE.search(lines[i + 1]):
            dedup += 1
            i += 2
            continue
        i += 1
    # Normalize internal spaces except bullet starters
    norm: List[str] = []
    for ln in out:
        if re.match(r"^\s*[*\-•]\s", ln):
            norm.append(re.sub(r"\s+", " ", ln).strip())
        else:
            norm.append(re.sub(r"[ \t]+", " ", ln).strip())
    return "\n".join(norm), dedup


# ------------------------------- Orchestrator ------------------------------

def clean_pages(pages: List[str]) -> Tuple[str, Dict]:
    meta: Dict[str, object] = {}
    # 1) strip repeated headers/footers
    p1, removed, wm = strip_repeated_headers_footers(pages)
    # Over-deletion guard: skip if >30% of non-empty lines would be removed
    total_nonempty = sum(1 for p in pages for ln in p.splitlines() if ln.strip())
    after_nonempty = sum(1 for p in p1 for ln in p.splitlines() if ln.strip())
    removed_count = total_nonempty - after_nonempty
    if total_nonempty and (removed_count / total_nonempty) > 0.30:
        meta["skip_repeats"] = True
        p1 = list(pages)
        removed, wm = [], False
    meta["removed_repeats"] = removed
    meta["watermark_removed"] = wm
    # 2) per-page email-as-URL removal
    p2, removed_urls = [], []
    for pg in p1:
        pg2, rm = drop_email_url_lines(pg)
        p2.append(pg2)
        removed_urls += rm
    meta["removed_email_urls"] = removed_urls
    # 3) join pages → normalize glyphs → heal wraps → heal dangling parens
    joined = "\f".join(p2)
    joined = normalize_glyphs(joined)
    before_heal = joined
    joined = heal_wraps(joined)
    healed_hyphens = (joined != before_heal)
    joined = heal_dangling_parens(joined)
    # 4) dedupe headings & normalize spaces
    final, dedup = dedupe_headings(joined)
    meta["dedup_headings"] = dedup
    # Done
    return final, meta


def make_output_json(pages: List[str]) -> Dict:
    text, meta = clean_pages(pages)
    out = {
        "phase": "1.0.0",
        "results": {
            "text": text,
            "meta": meta,
        },
        "metrics": {
            "chars": len(text),
            "lines": sum(len(pg.split("\n")) for pg in text.split("\f")),
            "healed_hyphens": any("\u00AD" in p or "-\n" in p for p in pages),
        },
        "errors": [],
    }
    return out

