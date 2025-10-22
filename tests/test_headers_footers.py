from phase1 import fuse_post as fp


def test_detect_and_strip_headers():
    pages = [
        "Header Line\nBody A\nFooter Line",
        "Header Line\nBody B\nFooter Line",
    ]
    reps = set(fp.detect_repeated_header_footer_lines(pages))
    assert "Header Line" in reps and "Footer Line" in reps
    out = fp.postproc(pages)
    # Headers/footers removed from each page
    assert "Header Line" not in out and "Footer Line" not in out

