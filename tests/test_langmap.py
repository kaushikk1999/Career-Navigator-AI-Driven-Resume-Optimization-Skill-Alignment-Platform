import warnings
from phase1 import ocr


def test_lang_mapping_and_fallback(monkeypatch):
    # Pretend only eng and osd present
    monkeypatch.setattr(ocr, "have_traineddata", lambda code: all(seg in {"eng", "osd"} for seg in code.split("+") if seg))
    monkeypatch.setattr(ocr, "_WARNED_LANGS", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code_en = ocr.resolve_tess_lang("en")
        code_hi = ocr.resolve_tess_lang("hi")
    assert code_en.startswith("eng") and "+osd" in code_en
    assert code_hi == "eng+osd"
    assert getattr(ocr.resolve_tess_lang, "last_fallback", False) is True

