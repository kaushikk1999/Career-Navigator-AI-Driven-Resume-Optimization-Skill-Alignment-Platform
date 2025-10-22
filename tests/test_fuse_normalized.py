from phase1 import fuse_post as fp
from phase1.contracts import OcrResult


def test_fuse_prefers_b_when_cleaned_better():
    # a has ligatures and shorter text; b is longer and cleans similarly
    a = OcrResult(text="ABCD of\ufb03ce data", confidences=[80, 80, 80], meta={"engine": "tesseract", "params": {}})
    b = OcrResult(text="ABCD office data and extras", confidences=[80, 80, 80], meta={"engine": "vision", "params": {}})
    out = fp.fuse_results(a, b)
    assert isinstance(out, OcrResult)
    # Expect b chosen due to length/distance improvement with similar confidence
    assert out.meta.get("engine") == "vision"

