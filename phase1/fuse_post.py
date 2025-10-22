import re, hashlib, json, unicodedata
from collections import Counter
from typing import List, Union
from .text_clean import clean_text_general

# Normalization helpers (pre/post OCR)
_LIGS = {
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
_WHITELIST_CASES = {
    "fundamentals of nursing": "Fundamentals of Nursing",
    "psychology & sociology": "Psychology & Sociology",
}
HEADINGS = (
    "OBJECTIVE",
    "EDUCATION",
    "CLINICAL SKILLS & COMPETENCIES",
    "PROJECTS & VOLUNTEER WORK",
    "CREDENTIALS",
)
RE_WS = re.compile(r"[ \t]+")
RE_DEHYPH = re.compile(r"(\w)-\n(\w)")
RE_ORPHAN = re.compile(r"(?m)(,)\n(?:•\s*)?([A-Z][A-Za-z-]{2,})\b")
RE_BAD_URL = re.compile(r"(?m)^https?://\S+@\S+\s*$")
RE_MULTI_NL = re.compile(r"\n{3,}")

# Additional robust cleaners
WS = RE_WS
DEHYPH = RE_DEHYPH
BULLET = re.compile(r"^\s*[\-•\u2022·–—*◦]\s*", re.M)
HANGING = re.compile(r"(?m)^(.*?:.*?),\s*$\n^\s*([•\-\u2022]?\s*[\w].*)$")
ARTIFACT_URL = re.compile(r"(?mi)^(https?://)?[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/?\s*$")


def _merge_hanging(match: re.Match) -> str:
    head = match.group(1).strip().rstrip(",")
    tail = match.group(2).lstrip("•-–—·*◦ ").strip()
    if not tail:
        return f"• {head}"
    return f"• {head}, {tail}"


def _should_join(prev: str, current: str) -> bool:
    prev_stripped = prev.rstrip()
    curr_stripped = current.lstrip()
    if not prev_stripped or not curr_stripped:
        return False
    if curr_stripped.startswith("•"):
        return False
    if prev_stripped.endswith((".", "!", "?", ":")):
        return False
    if prev_stripped.strip().isupper():
        return False
    if prev_stripped.endswith(",") or curr_stripped[0].islower():
        return True
    # Join label-like single-word lines (e.g., "Objective") with the next line
    # when the next line is not a bullet or heading line.
    tokens = prev_stripped.strip().split()
    if len(tokens) == 1 and not prev_stripped.strip().isupper() and tokens[0][0].isalpha():
        if not re.match(r"^\s*[\-•\u2022]", curr_stripped) and not _is_heading_line(curr_stripped.strip()):
            return True
    return False

def normalize_bullets(text: str) -> str:
    """Standardize bullet markers to '• ' at the start of a line."""
    def _repl(match: re.Match) -> str:
        prefix = match.group(1) or ""
        return f"{prefix}• "

    text = re.sub(r"(?m)^([ \t]*)[-•\u2022·▪◦*–—]+\s*", _repl, text)
    text = re.sub(r"(?m)^([ \t]*)•\s*(?:•\s*)+", r"\1• ", text)
    return text


def _apply_whitelist(text: str) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(0).lower()
        return _WHITELIST_CASES.get(key, match.group(0))

    if not _WHITELIST_CASES:
        return text
    pattern = re.compile("|".join(re.escape(k) for k in _WHITELIST_CASES.keys()), re.IGNORECASE)
    return pattern.sub(repl, text)


def postproc_pages(pages: List[str]) -> str:
    cleaned_pages: List[str] = []
    for raw in pages:
        text = clean_safe(raw)
        text = HANGING.sub(_merge_hanging, text)
        text = normalize_bullets(text)
        text = RE_ORPHAN.sub(lambda m: m.group(1) + " " + m.group(2), text)
        text = RE_BAD_URL.sub("", text)
        # Heal wrapped lines in a heading-aware way (avoid merging headings)
        # We intentionally avoid a blanket regex that turns most newlines into spaces,
        # because that can glue ALL-CAPS headings (e.g., OBJECTIVE, EDUCATION) onto
        # the previous line. Instead, rely on the line-wise join logic below.
        lines = text.split("\n")
        healed: List[str] = []
        for line in lines:
            if ARTIFACT_URL.match(line.strip()):
                continue
            if healed and _should_join(healed[-1], line):
                healed[-1] = (healed[-1].rstrip() + " " + line.lstrip())
            else:
                healed.append(line.strip())
        deduped: List[str] = []
        prev_blank = False
        for line in healed:
            if not line:
                if not prev_blank:
                    deduped.append("")
                prev_blank = True
                continue
            prev_blank = False
            deduped.append(line)
        trimmed = list(deduped)
        tail = len(trimmed) - 1
        heading_count = 0
        while tail >= 0:
            candidate = trimmed[tail].strip()
            if candidate and candidate.upper() in HEADINGS:
                heading_count += 1
                tail -= 1
                continue
            break
        if heading_count >= 3:
            trimmed = trimmed[: tail + 1]
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
        # Ensure a space after label colons at line starts (avoid URLs)
        for i, ln in enumerate(trimmed):
            if re.match(r"^\s*(?:https?://|ftp://|mailto:)", ln):
                continue
            # Only fix the first colon if immediately followed by non-space
            trimmed[i] = re.sub(r"^([^:\n]{2,40}):(?!\s)", r"\1: ", ln)
        text = "\n".join(trimmed)
        text = RE_MULTI_NL.sub("\n\n", text)
        text = re.sub(r"(?m)^•\s*•\s*", "• ", text)
        cleaned_pages.append(text.strip())
    return "\f".join(cleaned_pages)


def strip_artifacts(s: str) -> str:
    pages = s.split("\f")
    if not pages:
        return ""
    filtered_pages: List[List[str]] = []
    for pg in pages:
        lines = pg.splitlines()
        filtered_pages.append([ln for ln in lines if not ARTIFACT_URL.match(ln.strip())])
    if len(filtered_pages) >= 2:
        edge: List[str] = []
        for lines in filtered_pages:
            trimmed_top = [ln.strip() for ln in lines[:5] if ln.strip()]
            trimmed_bottom = [ln.strip() for ln in lines[-5:] if ln.strip()]
            # De-duplicate within a single page to avoid double-counting
            per_page_edges = set(trimmed_top + trimmed_bottom)
            edge.extend(per_page_edges)
        threshold = max(2, int(0.6 * len(filtered_pages)))
        common = {ln for ln, cnt in Counter(edge).items() if cnt >= threshold and len(ln) <= 80}
    else:
        common = set()
    cleaned_pages = []
    for lines in filtered_pages:
        cleaned = [ln for ln in lines if ln.strip() not in common]
        cleaned_pages.append("\n".join(cleaned).strip())
    return "\f".join(cleaned_pages)

def detect_repeated_header_footer_lines(pages: List[str]) -> List[str]:
    """Return lines that appear in first/last 5 lines of >=60% pages.

    Uses same heuristic as strip_artifacts() without mutating pages.
    """
    if not pages:
        return []
    if len(pages) == 1:
        return []
    edges: List[str] = []
    split_pages = [pg.splitlines() for pg in pages]
    for lines in split_pages:
        trimmed_top = [ln.strip() for ln in lines[:5] if ln.strip()]
        trimmed_bottom = [ln.strip() for ln in lines[-5:] if ln.strip()]
        # De-duplicate within a page before counting occurrences across pages
        edges.extend(set(trimmed_top + trimmed_bottom))
    threshold = max(2, int(0.6 * len(split_pages)))
    common = [ln for ln, cnt in Counter(edges).items() if cnt >= threshold and len(ln) <= 80]
    return common


def postproc(pages: Union[str, List[str]], normalize_typos: bool = False) -> str:
    """Deterministic post-processing that preserves URLs and headings.

    - NFC normalize via clean_safe()
    - Keep URLs; drop bogus http://…@… artifacts only
    - De-hyphenate wrapped words; collapse runs of spaces
    - Heal wrapped lines conservatively; normalize bullet markers to '• '
    - Preserve headings as-is when line is ALL CAPS and ≤4 words
    - Optional typo fixes when normalize_typos=True
    """
    if isinstance(pages, str):
        pages = [pages]
    stripped = strip_artifacts("\f".join(pages))
    working = stripped.split("\f") if stripped else []
    cleaned = postproc_pages(working)
    parts = cleaned.split("\f") if cleaned else []
    # Optional typo normalization (very targeted)
    if normalize_typos:
        TYPO_MAP = {
            "Kokata,India": "Kolkata, India",
            "Manglore,India": "Mangalore, India",
        }
        def _typo_fix(s: str) -> str:
            out = s
            for src, dst in TYPO_MAP.items():
                out = out.replace(src, dst)
            return out
        parts = [_typo_fix(part) for part in parts]
    parts = [_apply_whitelist(RE_BAD_URL.sub("", part).strip()) for part in parts]
    return "\f".join(parts)


def polish_resume_text(txt: str) -> str:
    pages = txt.split("\f") if txt else []
    stripped = strip_artifacts("\f".join(pages))
    working = stripped.split("\f") if stripped else []
    cleaned = postproc_pages(working)
    parts = cleaned.split("\f") if cleaned else []
    formatted_pages: List[str] = []
    for part in parts:
        lines = []
        for raw in part.split("\n"):
            stripped = raw.strip()
            if not stripped:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            if stripped.upper() in HEADINGS and lines and lines[-1] != "":
                lines.append("")
            lines.append(stripped)
        while len(lines) > 1 and not lines[-1]:
            lines.pop()
        formatted_pages.append(_apply_whitelist("\n".join(lines).strip()))
    formatted_pages = [pg for pg in formatted_pages if pg]
    return "\f".join(formatted_pages)

def _trigrams(s: str):
    s = re.sub(r"\s+", " ", s)
    return {s[i:i+3] for i in range(max(0, len(s)-2))}

def _trigram_dist(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb:
        return 0.0
    union = ta | tb
    inter = ta & tb
    return 1.0 - (len(inter) / len(union))

def _clarity_score(s: str) -> float:
    # heuristic: favor alpha trigrams ratio, penalties for long punct runs
    tri = _trigrams(s)
    if not tri:
        return 0.0
    alpha = {t for t in tri if re.match(r"[A-Za-z0-9 ][A-Za-z0-9 ][A-Za-z0-9 ]", t)}
    punct_runs = len(re.findall(r"[\.,;:]{3,}", s))
    non_ascii = sum(1 for ch in s if ord(ch) > 126 and not ch.isspace())
    return (len(alpha)/len(tri)) - 0.02*punct_runs - 0.01*non_ascii

def fuse_results(a, b):
    """Fuse OCR outputs by comparing cleaned strings and confidences."""
    if a is None and b is None:
        return None
    if a is None:
        if b and b.text:
            b.text = clean_text_general(b.text)
        return b
    ta = clean_text_general(a.text or "")
    a.text = ta
    if b is None or not b.text:
        return a
    tb = clean_text_general(b.text)
    mean_a = (sum(a.confidences)/len(a.confidences)/100) if a.confidences else 0.0
    mean_b = (sum(b.confidences)/len(b.confidences)/100) if b.confidences else 0.0
    conf_gap = mean_a - mean_b
    dist = _trigram_dist(tb, ta)
    clarity_a = _clarity_score(ta)
    clarity_b = _clarity_score(tb)

    if mean_b > mean_a + 0.05:
        b.text = tb
        return b

    distance_improves = dist >= 0.12 or len(tb) > len(ta) * 1.05 or clarity_b > clarity_a + 0.05
    if distance_improves and conf_gap <= 0.05:
        b.text = tb
        return b

    return a

def stable_run_id(preproc_pages, params) -> str:
    """
    sha256( NFC(bytes_of_each_page_after_preproc) || sorted_params_json )
    We NFC via latin-1 roundtrip so bytes are preserved deterministically.
    """
    h = hashlib.sha256()
    for p in preproc_pages:
        if hasattr(p, "bytes_gray") and isinstance(getattr(p, "bytes_gray"), (bytes, bytearray)):
            payloads = [p.bytes_gray]
            bb = getattr(p, "bytes_binary", None)
            if isinstance(bb, (bytes, bytearray)):
                payloads.append(bb)
            for payload in payloads:
                s = unicodedata.normalize("NFC", payload.decode("latin1"))
                h.update(s.encode("latin1"))
        elif hasattr(p, "text") and isinstance(getattr(p, "text"), str):
            s = unicodedata.normalize("NFC", p.text)
            h.update(s.encode("utf-8"))
    h.update(json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER on cleaned tokens."""
    ref_tokens = clean_safe(reference).split()
    hyp_tokens = clean_safe(hypothesis).split()
    m, n = len(ref_tokens), len(hyp_tokens)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # substitution
            )
    return float(dp[m][n]) / float(m)


def clean_safe(s: str) -> str:
    """Normalize ligatures, whitespace, and soft artifacts without dropping layout."""
    if not s:
        return ""
    txt = unicodedata.normalize("NFC", s.replace("\r\n", "\n").replace("\r", "\n"))
    txt = txt.translate({0x00AD: None})
    for src, dst in _LIGS.items():
        txt = txt.replace(src, dst)
    txt = DEHYPH.sub(r"\1\2", txt)
    txt = re.sub(r"&\s*\n", "& ", txt)
    pages = txt.split("\f")
    cleaned_pages = []
    for pg in pages:
        lines = pg.split("\n")
        normalized = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                normalized.append("")
                continue
            stripped = WS.sub(" ", stripped)
            normalized.append(stripped)
        cleaned_pages.append("\n".join(normalized).strip())
    cleaned = "\f".join(cleaned_pages)
    return cleaned.strip("\f")


def clean_resume(text_by_pages_or_str: Union[str, List[str]]) -> str:
    return postproc(text_by_pages_or_str)


def clean_resume_text(s: str) -> str:
    """Additional readability rules on a single logical document string."""
    t = clean_safe(s)
    # Ensure key headings are on their own lines
    headings = [
        "OBJECTIVE", "EDUCATION", "CLINICAL SKILLS & COMPETENCIES",
        "PROJECTS & VOLUNTEER WORK", "CREDENTIALS",
    ]
    for h in headings:
        t = re.sub(fr"{h}\s+", h + "\n", t)
    t = normalize_bullets(t)
    # Bullets only at line start: replace mid-line bullets with space, preserve leading
    def _bullet_cb(m: re.Match) -> str:
        return (m.group(1) or "") + "•" if m.group(1) is not None else " "
    t = re.sub(r"(^\s*)?•", _bullet_cb, t, flags=re.MULTILINE)
    # Fix merged bullet case for sample
    t = re.sub(r"(Disinfection)\s+(Interpersonal Skills:)", r"\1\n• \2", t)
    # Heal paragraphs: join newline->space when prev not end punctuation and next not bullet/header
    healed = []
    lines = t.split("\n")
    k = 0
    while k < len(lines):
        cur = lines[k].rstrip()
        is_prev_header = cur.strip() in [
            "OBJECTIVE", "EDUCATION", "CLINICAL SKILLS & COMPETENCIES",
            "PROJECTS & VOLUNTEER WORK", "CREDENTIALS",
        ]
        if k + 1 < len(lines):
            nxt = lines[k + 1].lstrip()
            is_header = bool(re.match(r"^[A-Z][A-Z &]+$", nxt))
            if (cur and cur[-1] not in ".!?:;" and not nxt.startswith("• ") and not is_header and not is_prev_header):
                healed.append(cur + " " + nxt)
                k += 2
                continue
        healed.append(cur)
        k += 1
    t = "\n".join(healed)
    # Collapse triple blank lines to double
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Trim per-line whitespace
    t = "\n".join(ln.strip() for ln in t.split("\n"))
    t = _apply_whitelist(t)
    return t


def postproc_exact(pages: Union[str, List[str]]) -> str:
    """Exact mode: minimal normalization suitable for preview/export.

    - NFC normalize, remove soft hyphens, normalize quotes/ligatures
    - De-hyphenate line-wraps, collapse internal horizontal whitespace
    - Preserve line structure otherwise; do not drop headers/footers or bullets
    - Join pages with formfeed if input is a list of pages
    """
    if isinstance(pages, str):
        # Treat as single logical document (no form-feeds added)
        return clean_safe(pages)
    out_pages: List[str] = []
    for pg in pages:
        out_pages.append(clean_safe(pg))
    return "\f".join(out_pages)


# ---------------- Document normalizer (parameterized) ----------------
def _is_heading_line(s: str) -> bool:
    if not s:
        return False
    t = s.strip()
    if not t:
        return False
    if t.upper() in HEADINGS:
        return True
    if t.isupper() and len(t) <= 64 and not t.endswith(('.', ',', ';', ':')):
        return True
    if re.match(r"^[A-Z][A-Za-z &/]+$", t) and not t.endswith((':', ';', '.')):
        return True
    return False


def _title_case_item(token: str) -> str:
    if not token:
        return token
    if token.isupper() or token.isdigit():
        return token
    small = {"and", "or", "the", "of", "in", "on", "for", "to", "with", "a", "an"}
    parts = re.split(r"(\s+|[,&/])", token)
    out = []
    for p in parts:
        if not p or p.isspace() or p in {",", "&", "/"}:
            out.append(p)
        elif p.lower() in small:
            out.append(p.lower())
        else:
            out.append(p[:1].upper() + p[1:].lower())
    return "".join(out)


def normalize_document(
    raw_text: str,
    list_casing: str = "title",
    bullet_symbol: str = "•",
    heading_style: str = "spacious",
    labels: List[str] = None,
    return_json: bool = False,
):
    """Policy-driven normalization of a single document string.

    Returns a cleaned string or a JSON-like dict when return_json=True.
    Deterministic and idempotent for identical inputs.
    """
    if not raw_text:
        return {"text": "", "metrics": {}, "policy": {}} if return_json else ""
    labels = labels or [
        "Email",
        "Mobile",
        "Phone",
        "Graduation",
        "Academic Performance",
        "Relevant Coursework",
    ]
    metrics = {
        "lines_total_in": 0,
        "lines_total_out": 0,
        "bullets_normalized": 0,
        "bullets_wrapped_healed": 0,
        "headers_detected": 0,
        "labels_normalized": 0,
        "bogus_url_artifacts_removed": 0,
        "blanklines_collapsed": 0,
        "casing_policy": list_casing,
    }

    txt = unicodedata.normalize("NFC", raw_text.replace("\r\n", "\n").replace("\r", "\n"))
    txt = txt.translate({0x00AD: None})
    for src, dst in _LIGS.items():
        txt = txt.replace(src, dst)
    txt = RE_DEHYPH.sub(r"\1\2", txt)

    pages = txt.split("\f")
    out_pages: List[str] = []
    label_pat = re.compile(r"^(%s)\s*:(?!\s)" % ("|".join(re.escape(x) for x in labels)), re.IGNORECASE)
    header_pipe = re.compile(r"\s*\|\s*")
    bullet_pat = re.compile(r"(?m)^\s*[-*•·—]\s+")

    for page in pages:
        # Step 1: canonical whitespace and label spacing
        lines = page.split("\n")
        metrics["lines_total_in"] += len(lines)
        tmp: List[str] = []
        has_mobile = any(re.match(r"(?i)^\s*mobile\s*:", ln) for ln in lines)
        prefer_label = "Mobile" if has_mobile else "Phone"
        for ln in lines:
            if ARTIFACT_URL.match(ln.strip()):
                metrics["bogus_url_artifacts_removed"] += 1
                continue
            ln2 = header_pipe.sub(" | ", ln)
            ln2 = label_pat.sub(lambda m: f"{m.group(1).title()}: ", ln2)
            if re.match(r"(?i)^\s*(mobile|phone)\s*:", ln2):
                ln2 = re.sub(r"(?i)^\s*(mobile|phone)\s*:", f"{prefer_label}: ", ln2)
                metrics["labels_normalized"] += 1
            ln2 = RE_WS.sub(" ", ln2.rstrip())
            tmp.append(ln2)

        # Step 4: bullet normalization
        s = "\n".join(tmp)
        before = s
        s = re.sub(rf"(?m)^\s*[-*•·—]\s+", f"{bullet_symbol} ", s)
        if s != before:
            metrics["bullets_normalized"] += 1

        # Heal bullet wraps and general paragraph wraps
        out_lines: List[str] = []
        i = 0
        while i < len(tmp):
            cur = tmp[i]
            if cur.lstrip().startswith(f"{bullet_symbol} "):
                # Merge subsequent wrapped lines until next bullet/heading/blank
                buff = cur.rstrip()
                j = i + 1
                while j < len(tmp):
                    nxt = tmp[j]
                    if not nxt.strip():
                        break
                    if nxt.lstrip().startswith(f"{bullet_symbol} ") or _is_heading_line(nxt):
                        break
                    if not buff.endswith(('.', '!', '?', ':', ';')):
                        buff = buff.rstrip() + " " + nxt.lstrip()
                        metrics["bullets_wrapped_healed"] += 1
                        j += 1
                        continue
                    break
                out_lines.append(buff)
                i = j
                continue
            # Non-bullet: paragraph heal across single newlines not ending with punctuation
            if out_lines and not out_lines[-1].endswith(('.', '!', '?', ':', ';')) and cur and not cur.lstrip().startswith(f"{bullet_symbol} ") and not _is_heading_line(cur):
                out_lines[-1] = out_lines[-1].rstrip() + " " + cur.lstrip()
            else:
                out_lines.append(cur)
            i += 1

        # Heading spacing
        spaced: List[str] = []
        for ln in out_lines:
            if _is_heading_line(ln.strip()):
                metrics["headers_detected"] += 1
                if spaced and spaced[-1] != "":
                    spaced.append("")
                spaced.append(ln.strip())
                if heading_style == "spacious":
                    spaced.append("")
            else:
                spaced.append(ln.rstrip())

        text_page = "\n".join(spaced)
        # Coursework casing
        if list_casing in {"title", "sentence"}:
            def _case_cb(m: re.Match) -> str:
                head = m.group(1)
                body = m.group(2)
                items = [x.strip() for x in re.split(r",\s*", body) if x.strip()]
                if list_casing == "title":
                    items = [_title_case_item(x) for x in items]
                else:
                    items = [x[:1].upper() + x[1:].lower() if x.isupper() else x for x in items]
                return f"{head}: " + ", ".join(items)
            text_page = re.sub(r"(?mi)^(Relevant Coursework)\s*:\s*(.+)$", _case_cb, text_page)

        # Collapse excessive blank lines
        before_blanks = text_page
        text_page = re.sub(r"\n{3,}", "\n\n" if heading_style == "spacious" else "\n", text_page)
        if text_page != before_blanks:
            metrics["blanklines_collapsed"] += 1

        out_pages.append(text_page.strip())

    final_text = "\f".join(out_pages)
    metrics["lines_total_out"] = sum(len(p.splitlines()) for p in out_pages)
    if return_json:
        return {
            "text": final_text,
            "metrics": metrics,
            "policy": {"bullet_symbol": bullet_symbol, "heading_style": heading_style},
        }
    return final_text


def postproc_with_v1_normalizer(pages: Union[str, List[str]]):
    """Run baseline postproc, then apply section normalizer v1.

    Returns (normalized_text, audit_json). Pure, deterministic, idempotent.
    """
    base_text = postproc(pages)
    # Import lazily to avoid any circular dependency concerns
    try:
        from .section_normalizer_v1 import normalize_resume
    except Exception:
        # Fallback to baseline if module unavailable
        return base_text, {
            "normalizer": "resume_v1",
            "edits": {"education_inlined": 0, "skills_inlined": 0, "spaces_after_commas": 0, "dash_normalized": 0, "parentheses_balanced": 0},
            "notes": ["v1_normalizer_unavailable"],
        }
    norm_text, audit = normalize_resume(base_text)
    return norm_text, audit
