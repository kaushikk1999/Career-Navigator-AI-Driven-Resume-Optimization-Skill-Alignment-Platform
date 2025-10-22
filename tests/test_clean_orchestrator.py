from phase1.clean_orchestrator import clean_text


def test_orchestrator_basic_rules():
    pages = [
        "CV MAKER demo watermark page 1\nThis is itera-\ntion with “fi” and ‘test’.\nAWS (S3",
        "CV MAKER demo watermark page 2\nContact: http://user@domain.com/\n- item two",
    ]
    out = clean_text(pages)
    txt = out["text_nfc"]
    # Header/footer removed on both pages
    assert "watermark" not in txt.lower()
    # Soft hyphen + dehyphenation + quotes normalization
    assert 'iteration with "fi" and \'test\'' in txt
    # Paren closed
    assert "AWS (S3)" in txt
    # Email-as-URL fixed
    assert "user@domain.com" in txt and "http://user@domain.com/" not in txt
    # Bullet normalized to ' - ' may be left as '-' or '•' depending mapping; we ensure merge didn't break bullet line
    assert "item two" in txt
    # JSON envelope version
    assert out["version"] == "clean-1.0.0"

