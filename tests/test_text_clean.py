import time
from phase1.text_clean import clean_text_general, normalize_section_case, format_contact_line


def test_email_wrapper_removed_and_urls_kept():
    s = "Visit https://example.com/page and http://user.name@example.org/ for more."
    out = clean_text_general(s)
    assert "https://example.com/page" in out
    assert "user.name@example.org" not in out  # wrapper removed entirely


def test_paragraph_heal_skips_bullets_and_punct():
    s = "This is a line\nthat continues\n• item one\nnext line after bullet\nEnds here.\nNext line"
    out = clean_text_general(s)
    # First two lines joined
    assert "This is a line that continues" in out
    # Bullet normalized and not merged with next bullet or paragraph line
    lines = out.splitlines()
    idx = lines.index(next(ln for ln in lines if ln.strip().startswith("• ")))
    # Ensure the line after bullet did not merge into bullet because previous line ended with punctuation? (no punctuation) -> we still avoid joining onto a new bullet
    assert lines[idx].startswith("• ")
    assert not lines[idx].endswith(" next line after bullet")
    # Sentence end keeps newline
    assert "Ends here.\nNext line" in out


def test_bullet_normalization():
    s = "- one\n* two\n• three\n● four\n▪ five"
    out = clean_text_general(s)
    lines = [ln for ln in out.splitlines() if ln]
    assert all(ln.startswith("• ") for ln in lines)


def test_dehyphenation_positive_and_negative():
    assert clean_text_general("medi-\ncal") == "medical"
    assert "Sree-Venkateswara" in clean_text_general("Sree-Venkateswara")


def test_ligatures_and_quotes():
    s = "office \ufb01le “test” and ‘ok’"
    out = clean_text_general(s)
    assert "office file \"test\" and 'ok'" in out


def test_idempotent():
    s = "Line one\nline two-\nwrap\n- bullet\nitem"
    o1 = clean_text_general(s)
    o2 = clean_text_general(o1)
    assert o1 == o2


def test_performance_guard():
    sample = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 150).strip()
    sample = "\n".join(sample[i:i+80] for i in range(0, len(sample), 80))
    t0 = time.time()
    out = clean_text_general(sample)
    dt = time.time() - t0
    assert isinstance(out, str)
    assert dt < 0.5  # soft guard


def test_contact_line_format():
    s = "Email:deepa@x.com|Mobile:  7679 211817"
    out = format_contact_line(s)
    assert out == "Email: deepa@x.com | Mobile: 7679 211817"


def test_normalize_section_case_generic():
    s = "Topics: anatomy & physiology, fundamentals of nursing, PSYCHOLOGY & SOCIOLOGY"
    out = normalize_section_case(s, style="title")
    assert "Topics: Anatomy & Physiology, Fundamentals of Nursing, Psychology & Sociology" in out
