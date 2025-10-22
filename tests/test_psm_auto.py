from phase1 import ocr


def test_auto_psm_heuristics():
    # Sparse content -> 11
    assert ocr._auto_psm(1200, 1600, lines=2) == 11
    # Wide page, moderate lines -> 4
    assert ocr._auto_psm(2200, 1000, lines=20) == 4
    # Dense standard page -> 6
    assert ocr._auto_psm(1200, 1600, lines=30) == 6

