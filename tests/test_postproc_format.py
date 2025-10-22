from phase1 import fuse_post as fp


def test_resume_postproc_merges_disinfection():
    src = "Safety & Infection Control: Universal Precautions, Waste Segregation, PPE Usage,\n• Disinfection"
    out = fp.postproc(src)
    assert "PPE Usage, Disinfection" in out
    assert "\n• Disinfection" not in out


def test_postproc_heals_wrapped_words():
    src = "Experience\nLine with hyphen-\nbreak next"
    out = fp.postproc(src)
    assert "hyphenbreak" in out


def test_normalize_bullets_helper():
    src = "- Item one\n\u2022 Item two"
    out = fp.normalize_bullets(src)
    lines = out.splitlines()
    assert lines[0].startswith("• ") and lines[1].startswith("• ")
