from phase1 import fuse_post as fp


def test_clean_safe_paragraph_and_ligatures():
    p1 = "This is a para with arti\nficial wrap and soft hy\u00ADphen. Smart quotes: “fi” and ‘test’."
    p2 = "Second page bullet\n- item two"
    out = fp.postproc([p1, p2])
    # Retains a single form feed between pages
    assert out.count("\f") == 1
    # Ligatures replaced and quotes normalized
    assert '"fi"' in out and "'test'" in out
    # Paragraph healed (newline -> space) and soft hyphen removed
    assert "arti ficial wrap" in out and "soft hyphen" in out
    # Bullets normalized
    assert "\n• item two" in out or "\n• Item two" in out
