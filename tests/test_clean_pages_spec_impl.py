from phase1.clean_pages import (
    strip_repeated_headers_footers,
    drop_email_url_lines,
    heal_dangling_parens,
    heal_wraps,
    normalize_glyphs,
    dedupe_headings,
    clean_pages,
    make_output_json,
)


def test_strip_repeated_headers_footers():
    pages = [
        "CV MAKER demo watermark\nA\nB",
        "CV MAKER demo watermark\nC\nD",
        "CV MAKER demo watermark\nE\nF",
    ]
    cleaned, removed, flag = strip_repeated_headers_footers(pages)
    assert flag is True and "CV MAKER demo watermark" in removed
    assert all("CV MAKER demo watermark\n" not in p for p in cleaned)


def test_drop_email_url_lines():
    p = "Contact\nhttp://user@domain.com/\nuser@domain.com\nhttps://example.com/path"
    out, removed = drop_email_url_lines(p)
    assert "http://user@domain.com/" in removed
    assert "user@domain.com" in out and "https://example.com/path" in out


def test_heal_dangling_parens():
    s = "Using AWS (S3 in project"
    out = heal_dangling_parens(s)
    assert "AWS (S3)" in out


def test_heal_wraps_and_glyphs():
    s = "itera-\ntion with “fi” and soft hy\u00ADphen"
    out = heal_wraps(normalize_glyphs(s))
    assert 'iteration with "fi" and soft hyphen' in out


def test_dedupe_headings():
    s = "EDUCATION\nEDUCATION\nLine"
    out, n = dedupe_headings(s)
    assert n == 1 and out.splitlines()[0] == "EDUCATION"


def test_clean_pages_json():
    pages = [
        "WATER\nThis is itera-\ntion\nAWS (S3\nhttp://user@domain.com/",
        "WATER\nNext page",
        "WATER\nEnd",
    ]
    final, meta = clean_pages(pages)
    assert "WATER\n" not in final
    assert "AWS (S3)" in final
    J = make_output_json(pages)
    assert J["phase"] == "1.0.0" and "results" in J and "metrics" in J

