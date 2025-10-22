from phase1.finisher_v1 import finish


def test_email_url_artifact_removed_and_audited():
    pages = [
        "Header watermark\nhttp://kaushikk1999@gmail.com/\nBody line",
        "Header watermark\nRegular email: kaushikk1999@gmail.com\nBody 2",
    ]
    out = finish(pages)
    txt = out["result"]["text_normalized"]
    assert "http://kaushikk1999@gmail.com/" not in txt
    assert "kaushikk1999@gmail.com" in txt
    assert out["audit"]["removed_email_urls"] == ["http://kaushikk1999@gmail.com/"]


def test_repeated_header_footer_removed():
    pages = [
        "WATER\nA\nB",
        "WATER\nC\nD",
        "WATER\nE\nF",
    ]
    out = finish(pages)
    txt = out["result"]["text_normalized"]
    assert "WATER\n" not in txt
    assert "WATER" in out["audit"]["removed_headers_footers"]


def test_hyphen_join_and_paren_heal():
    pages = [
        "This is itera-\ntion in progress\nTools: AWS (S3\nNext",
    ]
    out = finish(pages, opts={"heal_dangling_paren": True})
    t = out["result"]["text_normalized"]
    assert "iteration" in t
    assert any(x["to"].endswith(")") for x in out["audit"]["normalized_parens"])

