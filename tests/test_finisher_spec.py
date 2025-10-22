from phase1.finisher_spec import strip_boilerplate, clean_email_url, patch_dangling_parens, merge_label_rows, finish_spec


def test_header_footer_basic():
    pages = [
        "WATERMARK\nA\nB",
        "WATERMARK\nC\nD",
        "WATERMARK\nE\nF",
        "Unique\nG\nH",
    ]
    res = strip_boilerplate(pages)
    assert "WATERMARK" in res["boilerplate_removed"]
    assert all("WATERMARK\n" not in p for p in res["pages_clean"][:3])


def test_email_as_url_token_removed():
    t = "Contact: http://user@domain.com/ and email user@domain.com"
    res = clean_email_url(t)
    assert "http://user@domain.com/" in res["removed"]
    assert "user@domain.com" in res["text"]


def test_paren_patcher_guarded():
    t = "Tools: AWS (S3 and Apache (GPU next"
    res = patch_dangling_parens(t)
    assert any(x.startswith("(S3") for x in res["patched"]) or any("(GPU" in x for x in res["patched"]) 
    assert "AWS (S3)" in res["text"] or "Apache (GPU)" in res["text"]


def test_column_merge():
    lines = [
        "Programming: Python",
        "Frameworks: TensorFlow",
        "Tools: Tableau",
        "Databases: MySQL",
        "EXPERIENCE",
    ]
    out = merge_label_rows(lines)
    merged_str = "\n".join(out["lines"])
    assert "Programming: Python" in merged_str
    assert out["merges"] >= 2


def test_finish_spec_combined():
    pages = [
        "HDR\nAWS (S3\nTools: Tableau\nDatabases: MySQL\nHDR",
        "HDR\nBody two\nHDR",
        "HDR\nContact http://x@y/\nHDR",
    ]
    res = finish_spec(pages)
    t = res["text"]
    assert "HDR\n" not in t
    assert "AWS (S3)" in t
    assert res["audit"]["column_merges"] >= 1
    assert any("http://x@y/" in r for r in res["audit"]["email_url_removed"]) or res["audit"]["email_url_removed"]

