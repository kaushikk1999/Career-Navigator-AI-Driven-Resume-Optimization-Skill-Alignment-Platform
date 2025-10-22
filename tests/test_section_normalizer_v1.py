import re
from phase1.section_normalizer_v1 import normalize_resume


def test_education_inline_and_dash():
    src = "\n".join([
        "EDUCATION",
        "CHRIST UNIVERSITY",
        "Data Science Master of technology",
        "Bangalore",
        "August 2024 - August 2026",
        "",
        "EXPERIENCE",
        "Role X",
    ])
    out, audit = normalize_resume(src)
    edu_block = re.search(r"EDUCATION\n(.+)", out)
    assert edu_block, "EDUCATION block missing"
    line = edu_block.group(1)
    assert "CHRIST UNIVERSITY" in line
    assert "Data Science Master of technology" in line
    assert "August 2024 – August 2026" in line  # en dash
    assert line.endswith("Bangalore") or " — Bangalore" in line
    assert audit["edits"]["education_inlined"] >= 1


def test_skills_inline_and_parenthesis_balance():
    src = "\n".join([
        "SKILLS",
        "Programming Languages",
        "Python",
        "C++",
        "Libraries/Frameworks",
        "TensorFlow",
        "Keras",
        "Tools / Platforms",
        "AWS (S3",
        "Databases",
        "MySQL,PostgreSQL",
    ])
    out, audit = normalize_resume(src)
    assert "Programming Languages: Python, C++" in out
    assert "Libraries/Frameworks: TensorFlow, Keras" in out
    assert "Tools / Platforms: AWS (S3)" in out
    assert "Databases: MySQL, PostgreSQL" in out
    assert audit["edits"]["skills_inlined"] >= 3
    assert audit["edits"]["parentheses_balanced"] >= 1
    assert any(n.startswith("balanced_parenthesis: AWS (S3)") for n in audit["notes"])


def test_global_punctuation_spacing_and_dash():
    src = "Objective line\nKokata,India March 2024 - June 2024"
    out, audit = normalize_resume(src)
    assert "Kokata, India March 2024 – June 2024" in out
    assert audit["edits"]["spaces_after_commas"] >= 1
    assert audit["edits"]["dash_normalized"] >= 1

