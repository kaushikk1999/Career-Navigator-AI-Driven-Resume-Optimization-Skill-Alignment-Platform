import re
import unicodedata
from typing import List

# Typographic ligatures and quotes → ASCII
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

_RE_WS = re.compile(r"[\t ]+")
_RE_EMAIL_WRAPPER = re.compile(r"https?://[^\s]*@[^\s/]+(?:/[^\s]*)?", re.IGNORECASE)
_RE_DEHYPH = re.compile(r"(?<=\w)-\n(?=\w)")
_RE_BULLET_LINE = re.compile(r"(?m)^\s*([\-\*•●▪])\s+")
_RE_NUMBERED = re.compile(r"(?m)^\s*\d+[\.)]\s+")


def _is_heading(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if t.isupper() and len(t) <= 64 and not t.endswith(('.', ',', ';', ':')):
        return True
    if re.match(r"^[A-Z][A-Za-z &/]+$", t) and not t.endswith(('.', ',', ';', ':')):
        return True
    return False


def _nfc(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFC", s)
    # Remove soft hyphen
    s = s.translate({0x00AD: None})
    for src, dst in _LIGS.items():
        s = s.replace(src, dst)
    return s


def clean_text_general(
    text: str,
    *,
    join_paragraphs: bool = True,
    normalize_bullets: bool = True,
    strip_email_wrappers: bool = True,
) -> str:
    """General-purpose cleaner for OCR/text-layer output.

    - NFC normalize, remove soft hyphen, replace ligatures and curly quotes
    - Remove stray http(s)://...@... wrappers around emails (optional)
    - Safe de-hyphenation across line breaks (word-\nword)
    - Collapse runs of spaces/tabs to single spaces and rstrip lines
    - Normalize bullets to '• ' for leading -, •, ●, ▪, *
    - Heal paragraph wraps where previous line does not end with .:;?! and the next is not a bullet/numbered/heading
    - Preserve page breaks (\f) and reduce >2 blank lines to 2
    """
    if not text:
        return ""
    s = _nfc(text)
    if strip_email_wrappers:
        s = _RE_EMAIL_WRAPPER.sub("", s)
    # De-hyphenation on explicit line-wrapped hyphen
    s = _RE_DEHYPH.sub("", s)

    pages = s.split("\f")
    out_pages: List[str] = []
    for pg in pages:
        lines = pg.split("\n")
        # Trim trailing spaces, collapse internal spaces
        lines = [_RE_WS.sub(" ", ln.rstrip()) for ln in lines]
        if normalize_bullets:
            lines = [_RE_BULLET_LINE.sub("• ", ln) for ln in lines]
        healed: List[str] = []
        i = 0
        while i < len(lines):
            cur = lines[i]
            if not cur.strip():
                # keep at most two consecutive blanks
                if len(healed) >= 2 and healed[-1] == "" and healed[-2] == "":
                    i += 1
                    continue
                healed.append("")
                i += 1
                continue
            if not join_paragraphs:
                healed.append(cur)
                i += 1
                continue
            # Decide whether to join with the next line
            if (i + 1) < len(lines):
                nxt = lines[i + 1]
                is_bullet_or_num = bool(_RE_BULLET_LINE.match(nxt) or _RE_NUMBERED.match(nxt))
                # Never merge lines that are bullets/numbered themselves
                if _RE_BULLET_LINE.match(cur) or _RE_NUMBERED.match(cur):
                    healed.append(cur)
                    i += 1
                    continue
                if (cur and cur[-1] not in ".:;?!" and nxt.strip() and not is_bullet_or_num and not _is_heading(nxt)):
                    healed.append(cur.rstrip() + " " + nxt.lstrip())
                    i += 2
                    continue
            healed.append(cur)
            i += 1
        # Collapse triple+ blanks to double just in case
        pg_text = "\n".join(healed)
        pg_text = re.sub(r"\n{3,}", "\n\n", pg_text)
        out_pages.append(pg_text.strip())
    return "\f".join(out_pages)


def normalize_section_case(text: str, style: str = "title") -> str:
    """Optionally title/sentence-case list items for generic 'Header: a, b, c' lines.

    This helper is conservative and only adjusts comma-separated items after a label.
    It does not detect specific section names; callers should pre-filter the lines they want to transform.
    """
    if not text:
        return ""

    def _title(s: str) -> str:
        small = {"and", "or", "the", "of", "in", "on", "for", "to", "with", "a", "an"}
        tokens = re.split(r"(\s+|[,&/])", s)
        out = []
        for t in tokens:
            if not t or t.isspace() or t in {",", "&", "/"}:
                out.append(t)
            elif t.lower() in small:
                out.append(t.lower())
            else:
                out.append(t[:1].upper() + t[1:].lower())
        return "".join(out)

    def _cb(m: re.Match) -> str:
        head = m.group(1)
        body = m.group(2)
        items = [x.strip() for x in re.split(r",\s*", body) if x.strip()]
        if style == "title":
            items = [_title(x) for x in items]
        elif style == "sentence":
            items = [x[:1].upper() + x[1:].lower() if x else x for x in items]
        else:
            items = [x for x in items]
        return f"{head}: " + ", ".join(items)

    pattern = re.compile(r"(?mi)^([^:\n]{2,40})\s*:\s*(.+)$")
    return pattern.sub(_cb, _nfc(text))


def format_contact_line(s: str) -> str:
    """Normalize a header/contact line spacing and label capitalization.

    - Standardize labels Email:/Mobile:/Phone:
    - Normalize separator spacing around |, •, — to ' | '
    - Do not fabricate or drop content
    """
    if not s:
        return ""
    line = _nfc(s).strip()
    # Heuristic: only touch if looks like a contact line
    looks_contact = ("@" in line) or bool(re.search(r"\b(Email|Mobile|Phone)\b", line, re.IGNORECASE))
    if not looks_contact:
        return line
    # Normalize separators
    line = re.sub(r"\s*[\|•—]\s*", " | ", line)
    # Collapse spaces
    line = _RE_WS.sub(" ", line)
    # Normalize labels (single space after colon)
    line = re.sub(r"(?i)\bemail\s*:\s*", "Email: ", line)
    line = re.sub(r"(?i)\bmobile\s*:\s*", "Mobile: ", line)
    line = re.sub(r"(?i)\bphone\s*:\s*", "Phone: ", line)
    return line.strip()
