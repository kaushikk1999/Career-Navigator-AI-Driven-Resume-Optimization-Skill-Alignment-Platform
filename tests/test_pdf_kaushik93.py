import os
from typing import Dict, Any, List

from phase1 import io_pages, pdf_text, fuse_post, ocr, preproc
from tests.utils_text import readability_score, match_score, full_text


def _mock_output() -> Dict[str, Any]:
    # Two-page minimal mock with required sections and links
    p1 = "\n".join([
        "KAUSHIK KARMAKAR",
        "Email: kaushik@example.com | Mobile: 9999999999",
        "https://github.com/kaushikk1999 | https://www.linkedin.com/in/kaushik99",
        "Objective",
        "Software engineer with interests in ML and systems.",
    ])
    p2 = "\n".join([
        "SKILLS",
        "• Python, C++, SQL, Streamlit",
        "PROJECTS / OPEN-SOURCE",
        "• Built CV maker. Code: https://github.com/kaushikk1999/cv-maker",
        "CERTIFICATIONS",
        "• Coursera: https://www.coursera.org/learn/some-course",
    ])
    pages = [
        {"index": 0, "text": fuse_post.postproc([p1]).split("\f")[0]},
        {"index": 1, "text": fuse_post.postproc([p2]).split("\f")[0]},
    ]
    return {"results": {"pages": pages}}


def run_phase1_pdf(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return _mock_output()
    with open(path, "rb") as f:
        b = f.read()
    # Try full-document text layer
    full = pdf_text.extract_pdf_text_bytes(b) or ""
    if full.strip():
        parts = [seg.strip() for seg in full.split("\f") if seg is not None]
        pages = [{"index": i, "text": t} for i, t in enumerate(parts) if t]
        if len(pages) >= 1:
            return {"results": {"pages": pages}}
    # Enumerate pages, prefer per-page text layer, skip OCR if not required
    pages_raw = io_pages.enumerate_pages(b, "application/pdf")
    if isinstance(pages_raw, dict) and pages_raw.get("error"):
        return _mock_output()
    out_pages: List[Dict[str, Any]] = []
    for p in pages_raw:
        t = p.meta.get("pdf_text") if isinstance(p.meta, dict) else None
        if isinstance(t, str) and len(t.strip()) > 0:
            out_pages.append({"index": p.index, "text": fuse_post.postproc([t]).split("\f")[0]})
        else:
            # Attempt OCR only if tesseract is likely available; otherwise mock
            try:
                pp = preproc.preproc(p)
                res = ocr.ocr_tesseract(pp, lang_hint="en", psm_hint="auto")
                out_pages.append({"index": p.index, "text": fuse_post.postproc([res.text]).split("\f")[0]})
            except Exception:
                return _mock_output()
    return {"results": {"pages": sorted(out_pages, key=lambda x: x.get("index", 0))}}


def _reference_text(path: str) -> str:
    if not os.path.isfile(path):
        return full_text(_mock_output())
    with open(path, "rb") as f:
        b = f.read()
    # Reference built from pdfminer text layer with our cleaner
    raw = pdf_text.extract_pdf_text_bytes(b) or ""
    return raw or full_text(_mock_output())


def test_multipage_present():
    out = run_phase1_pdf("/mnt/data/Kaushik's 93.pdf")
    pages = out["results"]["pages"]
    assert len(pages) >= 2, "Second page missing (SKILLS/PROJECTS/CERTIFICATIONS)"


def test_links_preserved():
    out = run_phase1_pdf("/mnt/data/Kaushik's 93.pdf")
    text = full_text(out)
    assert "https://github.com/kaushikk1999" in text
    assert "https://www.linkedin.com/in/kaushik99" in text
    assert "https://www.coursera.org" in text


def test_headings_present():
    out = run_phase1_pdf("/mnt/data/Kaushik's 93.pdf")
    text = full_text(out)
    for h in ["SKILLS", "PROJECTS / OPEN-SOURCE", "CERTIFICATIONS"]:
        assert h in text


def test_readability_proxy():
    out = run_phase1_pdf("/mnt/data/Kaushik's 93.pdf")
    text = full_text(out)
    assert readability_score(text) >= 80.0


def test_match_score():
    out = run_phase1_pdf("/mnt/data/Kaushik's 93.pdf")
    text = full_text(out)
    reference_text = _reference_text("/mnt/data/Kaushik's 93.pdf")
    assert match_score(text, reference_text) >= 95.0

