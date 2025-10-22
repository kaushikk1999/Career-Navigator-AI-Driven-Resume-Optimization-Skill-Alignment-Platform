"""Lightweight observability helpers for per-page OCR metrics.

These functions are optional and kept import-light to avoid cold-start costs.
The Streamlit app may use them to summarize timings and parameters per page.
"""
from typing import Dict, Any, List


def summarize_page_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact summary from an OCR result's meta dict.

    Expected keys: engine, t_ms, params.psm, params.lang, char_conf_mean
    """
    if not isinstance(meta, dict):
        return {}
    params = meta.get("params", {}) if isinstance(meta.get("params"), dict) else {}
    return {
        "engine": meta.get("engine"),
        "t_ocr_ms": meta.get("t_ms"),
        "psm": params.get("psm"),
        "lang": params.get("lang"),
        "char_conf_mean": meta.get("char_conf_mean"),
    }


def summarize_run(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute minimal rollups for a run given page-level meta dicts."""
    engines = {}
    times: List[float] = []
    for m in results:
        if not isinstance(m, dict):
            continue
        e = m.get("engine")
        if e:
            engines[e] = engines.get(e, 0) + 1
        t = m.get("t_ms")
        if isinstance(t, (int, float)):
            times.append(float(t))
    total = len(results)
    p95 = None
    if times:
        times_sorted = sorted(times)
        idx = min(int(0.95 * len(times_sorted)), len(times_sorted) - 1)
        p95 = times_sorted[idx]
    return {"pages": total, "engines": engines, "t_ocr_ms_p95": p95}

