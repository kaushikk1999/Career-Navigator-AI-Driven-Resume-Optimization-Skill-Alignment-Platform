from io import BytesIO
from typing import Optional

try:
    from pdfminer.high_level import extract_text
except Exception:  # pragma: no cover - optional dependency resolution handled by requirements
    extract_text = None


def extract_pdf_text_bytes(pdf_bytes: bytes, max_pages: int = 10) -> Optional[str]:
    if extract_text is None:
        return None
    try:
        raw = extract_text(BytesIO(pdf_bytes), maxpages=max_pages) or ""
    except Exception:
        return None
    cleaned = raw.replace("\r\n", "\n").replace("\r", "\n")
    if len(cleaned.strip()) <= 30:
        return None
    try:
        from . import fuse_post
        normalized = fuse_post.postproc(cleaned)
    except Exception:
        normalized = cleaned.strip()
    return normalized if normalized and len(normalized.strip()) > 30 else None
