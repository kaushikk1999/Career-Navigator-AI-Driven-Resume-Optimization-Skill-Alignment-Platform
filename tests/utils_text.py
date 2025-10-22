from typing import Dict, Any
from phase1 import fuse_post


def full_text(out: Dict[str, Any]) -> str:
    pages = out.get("results", {}).get("pages", [])
    # Ensure order by index
    pages_sorted = sorted(pages, key=lambda x: x.get("index", 0))
    return "\f".join([p.get("text", "") for p in pages_sorted])


def readability_score(s: str) -> float:
    if not s:
        return 0.0
    lines = [ln for ln in s.replace("\r", "\n").split("\n")]
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return 0.0
    avg_len = sum(len(ln) for ln in non_empty) / max(1, len(non_empty))
    # Score 100 when avg line length is within [60, 110], linearly decay outside
    if 60 <= avg_len <= 110:
        base = 100.0
    elif avg_len < 60:
        base = max(0.0, 100.0 * (avg_len / 60.0))
    else:
        # avg_len > 110, decay inversely up to ~200
        base = max(0.0, 100.0 * (200.0 - min(avg_len, 200.0)) / 90.0)

    # Bonus for normalized bullets and clear headings
    bullets = sum(1 for ln in non_empty if ln.lstrip().startswith("• "))
    headings = sum(1 for ln in non_empty if (ln.strip().isupper() and len(ln.strip().split()) <= 4))
    frac_bullets = bullets / max(1, len(non_empty))
    frac_headings = headings / max(1, len(non_empty))
    bonus = min(10.0, 100.0 * (0.5 * frac_bullets + 0.5 * frac_headings))
    return min(100.0, base * 0.9 + bonus)


def match_score(hypothesis: str, reference: str) -> float:
    if reference is None or hypothesis is None:
        return 0.0
    wer = fuse_post.word_error_rate(reference, hypothesis)
    return max(0.0, 100.0 * (1.0 - wer))

