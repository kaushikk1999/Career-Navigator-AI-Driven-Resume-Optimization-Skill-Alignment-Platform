from phase1 import ocr


def test_should_retry_thresholds():
    assert ocr._should_retry(4, 65.0) is True
    assert ocr._should_retry(6, 69.9) is True
    assert ocr._should_retry(4, 80.0) is False
    assert ocr._should_retry(11, 40.0) is False
