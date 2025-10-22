from pdf2image import convert_from_bytes
from PIL import Image
from io import BytesIO
from typing import Optional, List, Union
from .contracts import Page, error

# Optional: pdfminer for embedded text bypass
try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
except Exception:
    extract_pages = None
    LTTextContainer = None

MAX_MB, MAX_PAGES = 15, 10

def enumerate_pages(b: bytes, mime: str) -> Union[List[Page], dict]:
    if len(b) > MAX_MB * 1024 * 1024:
        return error("FILE_TOO_LARGE")
    if mime == "application/pdf":
        # Try to rasterize all pages (≤ MAX_PAGES). Also attach PDF text layer per page if available.
        try:
            imgs = convert_from_bytes(
                b, dpi=330, grayscale=True, use_pdftocairo=True, fmt="png"
            )
        except Exception:
            return error("UNSUPPORTED_PDF")
        if len(imgs) > MAX_PAGES:
            return error("TOO_MANY_PAGES")
        # Extract per-page text if pdfminer is available
        text_pages = pdf_extract_text_pages(b)
        out: List[Page] = []
        for i, im in enumerate(imgs):
            buf = BytesIO()
            im.save(buf, format="PNG")
            meta = {}
            if text_pages and i < len(text_pages):
                t = (text_pages[i] or "")
                # Heuristic: consider meaningful only if long enough; avoids false positives
                if len(t.strip()) > 200:
                    meta["pdf_text"] = t
            out.append(Page(bytes=buf.getvalue(), mime="image/png", index=i, meta=meta))
        return out
    if mime.startswith("image/"):
        try:
            Image.open(BytesIO(b)).verify()
            return [Page(bytes=b, mime=mime, index=0)]
        except Exception:
            return error("BAD_MIME")
    return error("BAD_MIME")


def pdf_extract_text_pages(b: bytes):
    """Extract per-page text using pdfminer if available. Returns list[str]."""
    if extract_pages is None:
        return []
    texts = []
    try:
        for page_layout in extract_pages(BytesIO(b)):
            chunks = []
            for element in page_layout:
                if hasattr(element, "get_text"):
                    try:
                        chunks.append(element.get_text())
                    except Exception:
                        continue
            texts.append("".join(chunks))
    except Exception:
        return []
    return texts

def pdf_text_is_meaningful(text_pages):
    if not text_pages:
        return False
    # Consider meaningful if any page contains at least ~30 characters of text
    return any(len((t or "").strip()) >= 30 for t in text_pages)


# Compatibility wrapper for the bytes-level extractor mentioned in the plan
def extract_pdf_text_bytes(pdf_bytes: bytes, max_pages: int = 10) -> Optional[str]:
    """Delegate to phase1.pdf_text.extract_pdf_text_bytes if available.

    Keeps this function in io_pages for discoverability based on the plan,
    without duplicating implementation.
    """
    try:
        from .pdf_text import extract_pdf_text_bytes as _impl
    except Exception:
        return None
    return _impl(pdf_bytes, max_pages=max_pages)
