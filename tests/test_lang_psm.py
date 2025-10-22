import warnings
from types import SimpleNamespace

from PIL import Image

from phase1 import ocr


def test_resolve_lang_mapping(monkeypatch):
    monkeypatch.setattr(
        ocr,
        "have_traineddata",
        lambda code: all(seg in {"hin", "ben", "eng"} for seg in code.split("+") if seg and seg != "osd"),
    )
    monkeypatch.setattr(ocr, "_WARNED_LANGS", set())
    code_hi = ocr.resolve_tess_lang("hi")
    code_bn = ocr.resolve_tess_lang("bn")
    assert code_hi.startswith("hin")
    assert code_bn.startswith("ben")
    assert getattr(ocr.resolve_tess_lang, "last_fallback", False) is False


def test_resolve_lang_missing_warns(monkeypatch):
    monkeypatch.setattr(ocr, "have_traineddata", lambda code: False)
    monkeypatch.setattr(ocr, "_WARNED_LANGS", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code = ocr.resolve_tess_lang("hi")
    assert code == "eng+osd"
    assert any("falling back" in str(w.message).lower() for w in caught)
    assert getattr(ocr.resolve_tess_lang, "last_fallback", False) is True


def test_choose_psm_respects_override(monkeypatch):
    page = SimpleNamespace(artifacts={"line_count": 12})
    im = Image.new("L", (2200, 1000))
    assert ocr._choose_psm(page, im, override=11) == 11
    assert ocr._choose_psm(page, im, override=None) in {4, 6, 11}
