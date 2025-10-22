import os
from types import SimpleNamespace

from phase1 import fuse_post, ocr, contracts


def test_lang_mapping_osd_default(monkeypatch):
    # No TESSDATA_PREFIX -> assume packs present
    if "TESSDATA_PREFIX" in os.environ:
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    code = ocr.resolve_tess_lang("bn", allow_osd=True)
    assert "+osd" in code
    assert code.startswith("ben") or code.startswith("eng")


def test_lang_mapping_missing_pack(monkeypatch, tmp_path):
    # Empty tessdata dir should force fallback to eng+osd
    td = tmp_path / "tessdata"
    td.mkdir()
    monkeypatch.setenv("TESSDATA_PREFIX", str(td))
    code = ocr.resolve_tess_lang("bn", allow_osd=True)
    assert code.startswith("eng") and "+osd" in code


def test_resolve_lang_preserves_pre_resolved(monkeypatch):
    monkeypatch.setattr(ocr, "have_traineddata", lambda code: True)
    monkeypatch.setenv("TESSERACT_INCLUDE_OSD", "1")
    code = ocr.resolve_tess_lang("hin+osd")
    assert code == "hin+osd"


def test_psm_auto_wide_vs_dense():
    from PIL import Image
    # Create dummy grayscale image
    im_wide = Image.new("L", (2000, 900), color=255)
    p_wide = contracts.PreprocPage(bytes_gray=b"x", bytes_binary=b"y", index=0, meta={"w":2000,"h":900,"dpi":300}, artifacts={"line_count": 10, "fill_ratio": 0.2, "aspect_ratio": 2000/900})
    p_dense = contracts.PreprocPage(bytes_gray=b"x", bytes_binary=b"y", index=0, meta={"w":900,"h":1200,"dpi":300}, artifacts={"line_count": 40, "fill_ratio": 0.75, "aspect_ratio": 900/1200})
    # Access internal chooser
    assert ocr._choose_psm(p_wide, im_wide, None) in (4,6)
    # Dense should pick 6
    im_dense = Image.new("L", (900, 1200), color=255)
    assert ocr._choose_psm(p_dense, im_dense, None) == 6


def test_postproc_bullets():
    s = "\n".join(["•", "Disinfection of surfaces with 0.5% chlorine solution."])
    out = fuse_post.clean_resume(s)
    assert "• Disinfection" in out


def test_header_footer_strip():
    body = "Experience\n- X\n- Y\n"
    tail = "\n".join([
        "OBJECTIVE", "EDUCATION", "CLINICAL SKILLS & COMPETENCIES",
        "PROJECTS & VOLUNTEER WORK", "CREDENTIALS",
    ])
    out = fuse_post.clean_resume(body + "\n" + tail)
    assert "CREDENTIALS" not in out.splitlines()[-1]


def test_pdf_text_layer_flag():
    texts = ["This is a digital page with selectable text." * 1, ""]
    from phase1.io_pages import pdf_text_is_meaningful
    assert pdf_text_is_meaningful(texts) is True


def test_dehyphenation():
    assert fuse_post.clean_safe("medi-\ncal") == "medical"


def test_headings_block():
    out = fuse_post.clean_resume_text("OBJECTIVE A motivated student")
    assert "OBJECTIVE\nA motivated" in out


def test_bullets_split_and_merge_fix():
    s = "• Safety & Infection Control: PPE Usage, Disinfection Interpersonal Skills: Empathy"
    o = fuse_post.clean_resume_text(s)
    assert "\n• Interpersonal Skills" in o


def test_preproc_outputs(monkeypatch):
    import cv2
    import numpy as np
    # tiny synthetic image
    arr = np.full((120, 200), 255, dtype=np.uint8)
    ok, enc = cv2.imencode('.png', arr)
    assert ok
    pg = contracts.Page(bytes=enc.tobytes(), mime='image/png', index=0)
    pp = __import__('phase1.preproc', fromlist=['preproc']).preproc.preproc(pg)
    assert isinstance(pp.bytes_gray, (bytes, bytearray)) and len(pp.bytes_gray) > 0
    assert isinstance(pp.bytes_binary, (bytes, bytearray)) and len(pp.bytes_binary) > 0


def test_stable_run_id_determinism():
    pages = [
        SimpleNamespace(bytes_gray=b"gray", bytes_binary=b"bin"),
        SimpleNamespace(text="embedded"),
    ]
    params = {"engine": "tesseract", "lang": "eng"}
    first = fuse_post.stable_run_id(pages, params)
    second = fuse_post.stable_run_id(pages, dict(params))
    assert first == second
    pages_changed = [
        SimpleNamespace(bytes_gray=b"Gray", bytes_binary=b"bin"),
        SimpleNamespace(text="embedded"),
    ]
    assert fuse_post.stable_run_id(pages_changed, params) != first
