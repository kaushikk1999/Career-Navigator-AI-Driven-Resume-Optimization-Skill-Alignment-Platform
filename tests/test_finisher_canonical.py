from phase1.finisher_canonical import canonical_postproc


def test_canonical_basic_flow():
    pages = [
        "HDR\nThis is itera-\ntion test.\n• Item one\nHDR",
        "HDR\nContact: http://user@domain.com/\nAWS (S3\nHDR",
    ]
    out = canonical_postproc(pages)
    txt = out["text"]
    assert "HDR\n" not in txt  # header removed
    assert "iteration" in txt  # dehyphenated
    assert "user@domain.com" in txt and "http://user@domain.com/" not in txt  # email-as-URL fixed
    assert "AWS (S3)" in txt  # paren balanced
    assert out["flags"]["dehyphenated_joins"] >= 1
    assert out["metrics"]["length_ratio_vs_raw"] > 0

