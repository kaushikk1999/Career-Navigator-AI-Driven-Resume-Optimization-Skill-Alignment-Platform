import pytest

from phase1 import fuse_post as fp


def test_bullet_midline_removed():
    s = "second • year\n• Skill"
    out = fp.clean_resume_text(s)
    assert "second • year" not in out
    assert any(line.startswith("• ") for line in out.splitlines())


def test_dehyphenation_wrap_only():
    assert fp.clean_safe("patient-\ncare") == "patientcare"
    assert fp.clean_safe("Sree-Venkateswara") == "Sree-Venkateswara"


def test_paragraph_heal():
    s = "This is a line\nthat continues.\nEnds with period.\nNext line"
    out = fp.clean_resume_text(s)
    assert "This is a line that continues." in out
    # After period we keep newline
    assert "Ends with period.\nNext line" in out


def test_merge_orphan_bullet():
    src = "Safety & Infection Control: Universal Precautions, Waste Segregation, PPE Usage,\n• Disinfection"
    out = fp.postproc(src)
    assert "PPE Usage, Disinfection" in out
    assert "\n• Disinfection" not in out


def test_drop_bogus_email_url():
    src = "Footer\nhttp://deepakumari1817@gmail.com/\nEnd"
    out = fp.postproc(src)
    assert "http://deepakumari1817@gmail.com/" not in out


def test_bullets_normalized_and_paragraphs_joined():
    src = "- Item one\n-  Item two\n\n\nLine with hyphen-\nbreak next"
    out = fp.postproc(src)
    assert "• Item one" in out and "• Item two" in out
    assert "\n\n\n" not in out
    assert "hyphenbreak" in out


def test_clean_safe_basic():
    s = "inter-\nnational “fi” ﬂow"
    out = fp.clean_safe(s)
    assert out.count("international") == 1
    assert '"' in out


def test_polish_headings():
    text = "OBJECTIVE\nline\nEDUCATION\nnext\nhttp://demo@example.com/"
    out = fp.polish_resume_text(text)
    assert "\n\nEDUCATION" in out
    assert "http://demo@example.com/" not in out


def test_clean_safe_ampersand_join():
    assert "Pulse & Respiration" in fp.clean_safe("Pulse &\nRespiration")


def test_word_error_rate_basic():
    ref = "patient assessment and care"
    hyp = "patient care"
    wer = fp.word_error_rate(ref, hyp)
    assert wer == pytest.approx(0.5, abs=1e-6)


def test_postproc_collapses_duplicate_bullets():
    src = "• • Skilled task\n- Another bullet"
    out = fp.postproc(src)
    lines = [ln for ln in out.splitlines() if ln]
    assert lines[0].startswith("• Skilled task")
    assert lines[1].startswith("• Another bullet")


def test_label_colon_spacing():
    from phase1 import fuse_post as fp
    src = "Email:abc@example.com\nWebsite: https://example.com"
    out = fp.postproc(src)
    assert "Email: abc@example.com" in out
    assert "Website: https://example.com" in out
