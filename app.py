import streamlit as st, json
from datetime import datetime, timezone

st.set_page_config(page_title="Phase 1 · OCR Ingest", layout="wide")
st.title("Phase 1 · OCR Ingest")

# --- UI (render first) ---
files = st.file_uploader("Upload PDF/Images (≤15MB each)", type=["pdf","png","jpg","jpeg","webp"], accept_multiple_files=True, key="w_upload")
consent_proc = st.checkbox("I consent to on-device processing (required)", key="w_consent_process")
consent_api  = st.checkbox("I consent to external OCR API (Google Vision)", key="w_consent_external_api")
engine = st.radio("Engine", ["auto","tesseract","vision"], index=0, horizontal=True, key="w_engine")
prefer_pdf_text = st.checkbox("Prefer embedded PDF text layer (if present)", value=True, key="w_prefer_pdf_text")
preview_mode = st.radio("Preview mode", ["clean","exact"], index=0, horizontal=True, key="w_preview_mode")
lang = st.selectbox("Language hint", ["en","hi","bn","hr","und"], index=0, key="w_lang_hint")
psm_choice = st.selectbox(
    "Tesseract page segmentation (PSM)",
    ["auto","4","6","11"],
    index=0,
    key="w_psm_hint",
    help="auto picks 4/6/11; use 4 for multi-column layouts",
)
psm_arg = None if psm_choice == "auto" else psm_choice
psm_override = None if psm_choice == "auto" else int(psm_choice)
skip_deskew = st.checkbox("Skip deskew (document already looks aligned)", value=False, key="w_skip_deskew")
show_previews = st.checkbox("Show preprocessing previews (first few pages)", value=False, key="w_show_previews")
normalize_typos = st.checkbox("Normalize output (typo fixes)", value=False, key="w_normalize_typos")
run = st.button("Run OCR", type="primary", use_container_width=True, key="w_run")
st.caption(f"Vision consent: {consent_api} • Engine: {engine} • PSM: {psm_choice}")

# small helper to show exceptions nicely
def show_exc(prefix: str, e: Exception):
    st.error(f"{prefix}: {type(e).__name__}: {e}")

# --- lazy imports so UI appears even if deps missing ---
try:
    from phase1 import io_pages, preproc, ocr, fuse_post, pdf_text
except Exception as e:
    show_exc("Import error (check files under phase1/ and requirements)", e)
    st.stop()

if run:
    if not files or not consent_proc:
        st.error("Please upload at least one file and accept processing consent.")
        st.stop()

    try:
        # enumerate pages with per-page text-layer preference
        pages_to_ocr=[]
        direct_results=[]
        file_summaries=[]
        for f in files:
            b = f.getvalue()
            if f.type == "application/pdf":
                direct_blob = None
                if prefer_pdf_text:
                    # Try full-document text extraction for fastest bypass
                    direct_blob = pdf_text.extract_pdf_text_bytes(b)
                if prefer_pdf_text and direct_blob:
                    parts = direct_blob.split("\f")
                    if parts and not parts[-1].strip():
                        parts = parts[:-1]
                    if any(len(seg.strip()) > 30 for seg in parts):
                        if len(parts) > io_pages.MAX_PAGES:
                            st.error("TOO_MANY_PAGES"); st.stop()
                        from phase1.contracts import OcrResult
                        for idx, seg in enumerate(parts):
                            direct_results.append(OcrResult(text=seg.strip(), confidences=[], meta={"engine":"pdf_text","params":{"source":"embedded","page":idx}}))
                        file_summaries.append({"name": f.name, "bytes": f.size, "mime": f.type, "pages": len(parts)})
                        continue
                # Rasterize all pages (<=10), attach per-page text if available
                raster = io_pages.enumerate_pages(b, f.type)
                if isinstance(raster, dict) and "error" in raster:
                    st.error(raster["error"]); st.stop()
                used = 0
                from phase1.contracts import OcrResult
                for p in raster:
                    t = None
                    if prefer_pdf_text and isinstance(getattr(p, "meta", None), dict):
                        t = p.meta.get("pdf_text")
                    if prefer_pdf_text and isinstance(t, str) and len(t.strip()) >= 30:
                        direct_results.append(OcrResult(text=t, confidences=[], meta={"engine":"pdf_text","params":{"source":"embedded","page":p.index}}))
                        used += 1
                    else:
                        pages_to_ocr.append(p)
                file_summaries.append({"name": f.name, "bytes": f.size, "mime": f.type, "pages": len(raster)})
                continue
            res = io_pages.enumerate_pages(b, f.type)
            if isinstance(res, dict) and "error" in res:
                st.error(res["error"]); st.stop()
            pages_to_ocr += res
            file_summaries.append({"name": f.name, "bytes": f.size, "mime": f.type, "pages": len(res)})

        # Resolve language to tessdata code and warn if missing
        lang_code = ocr.resolve_tess_lang(lang)
        if getattr(ocr.resolve_tess_lang, "last_fallback", False):
            st.warning("Requested language pack missing. Falling back to 'eng+osd'.")
        for dr in direct_results:
            if isinstance(dr.meta, dict):
                params = dr.meta.setdefault("params", {})
                params.setdefault("lang", lang_code)

        # preproc (only for pages that need OCR) with timing per page
        import time as _time
        pps = []
        _pp_ms = []
        for p in pages_to_ocr:
            _t0 = _time.time()
            pp = preproc.preproc(p, enable_deskew=not skip_deskew)
            _pps_ms = int((_time.time() - _t0) * 1000)
            _pp_ms.append(_pps_ms)
            pps.append(pp)

        # id (include both preproc bytes and embedded text pages)
        class _TextForHash:
            def __init__(self, text):
                self.text = text
        hash_inputs = list(pps) + [_TextForHash(r.text) for r in direct_results]
        run_id = fuse_post.stable_run_id(
            hash_inputs,
            {
                "engine": engine,
                "lang": lang,
                "lang_code": lang_code,
                "psm": psm_override or "auto",
                "skip_deskew": skip_deskew,
            },
        )

        def _quality_flags(pp):
            flags = []
            if pp.artifacts.get("laplacian_var", 0.0) < 25.0:
                flags.append("low_sharpness")
            if pp.artifacts.get("mean_intensity", 0.0) < 0.18 or pp.artifacts.get("mean_intensity", 1.0) > 0.92:
                flags.append("low_contrast")
            if abs(pp.artifacts.get("skew_deg", 0.0)) > 6.0:
                flags.append("high_skew")
            return flags

        issues = [f"Page {pp.index+1}: {', '.join(flags)}" for pp in pps if (flags := _quality_flags(pp))]
        if issues:
            st.warning("Image quality issues detected:\n" + "\n".join(issues))

        # Optional visual previews for quick QA
        if show_previews and pps:
            with st.expander("Preprocessing previews", expanded=False):
                max_preview = min(len(pps), 6)
                for i in range(max_preview):
                    pp = pps[i]
                    cols = st.columns(2)
                    cols[0].image(pp.bytes_gray, caption=f"Page {pp.index+1} · Gray {int(pp.meta.get('w',0))}×{int(pp.meta.get('h',0))}", use_column_width=True)
                    cols[1].image(pp.bytes_binary, caption=f"Page {pp.index+1} · Binary", use_column_width=True)
                    flags = _quality_flags(pp)
                    if flags:
                        st.caption(f"Flags: {', '.join(flags)} | Skew: {pp.artifacts.get('skew_deg', 0.0):.2f}°")

        # OCR per page for scanned/images, then append digital results
        results=[]
        page_quality=[]
        for p in pps:
            flags = _quality_flags(p)
            page_quality.append(flags)
            a = ocr.ocr_tesseract(p, lang_code, psm_arg)
            b = ocr.ocr_vision(p, lang) if (engine in ["auto","vision"] and consent_api) else None
            r = fuse_post.fuse_results(a,b)
            results.append(r)
        # Append direct (embedded) results with empty quality flags
        results = direct_results + results
        direct_count = len(direct_results)
        page_quality = ([[] for _ in direct_results] + page_quality)

        # High-readability post-processing across pages
        page_texts = [r.text for r in results]
        # Compute repeated header/footer candidates for metrics
        try:
            rep_candidates = fuse_post.detect_repeated_header_footer_lines(page_texts)
        except Exception:
            rep_candidates = []
        import time as _time
        _t_post0 = _time.time()
        cleaned_joined = (
            fuse_post.postproc(page_texts, normalize_typos=normalize_typos)
            if preview_mode == "clean"
            else fuse_post.postproc_exact(page_texts)
        )
        _t_post_ms = int((_time.time() - _t_post0) * 1000)
        cleaned_pages = cleaned_joined.split("\f") if cleaned_joined else []
        if cleaned_pages and len(cleaned_pages) == len(results):
            for i, txt in enumerate(cleaned_pages):
                results[i].text = txt
        elif cleaned_pages:
            for i, txt in enumerate(cleaned_pages):
                if i < len(results):
                    results[i].text = txt

        # Spec-aligned finisher (header/footer, email-URL, paren heal, column merges)
        finisher_payload = None
        try:
            from phase1.finisher_spec import finish_spec as _finish_spec
            finisher_payload = _finish_spec([r.text for r in results])
        except Exception:
            finisher_payload = {
                "text": "\f".join([r.text or "" for r in results]),
                "audit": {"error": "finisher_spec_unavailable"},
            }

        # Orchestrator (six-rule) clean output
        orchestrator_payload = None
        try:
            from phase1.clean_orchestrator import clean_text as _clean_text
            orchestrator_payload = _clean_text([r.text for r in results])
        except Exception:
            orchestrator_payload = {
                "version": "clean-1.0.0",
                "pages": len(results),
                "edits": [],
                "text_nfc": "\f".join([r.text or "" for r in results]),
                "error": "clean_orchestrator_unavailable",
            }

        def _conf_mean(vals):
            return (sum(vals) / len(vals) / 100.0) if vals else 0.0

        def _conf_p05(vals):
            if not vals:
                return 0.0
            sorted_vals = sorted(vals)
            idx = max(int(len(sorted_vals) * 0.05) - 1, 0)
            return sorted_vals[idx] / 100.0

        # Aggregate per-run metrics
        total_chars = sum(len(r.text or "") for r in results)
        headers_stripped_count = 0
        if rep_candidates:
            for pg in page_texts:
                headers_stripped_count += sum(1 for ln in pg.splitlines() if ln.strip() in rep_candidates)

        out = {
          "phase": "1.0.0",
          "run_id": run_id,
          "timestamp_utc": datetime.now(timezone.utc).isoformat(),
          "inputs": {
              "files": file_summaries,
              "opts": {"engine": engine, "lang_hint": lang, "lang_code": lang_code, "psm_hint": psm_choice, "skip_deskew": skip_deskew},
          },
          "results": {
              "pages": [
                  {
                      "index": i,
                      "text": r.text,
                      "lang": (r.meta.get("params", {}).get("lang", "und") if isinstance(r.meta, dict) else "und"),
                      "char_conf_mean": _conf_mean(r.confidences),
                      "char_conf_p05": _conf_p05(r.confidences),
                      "quality_flags": page_quality[i] if i < len(page_quality) else [],
                      "tesseract": r.meta,
                      "preproc_artifacts": ({} if i < direct_count else {**pps[i - direct_count].artifacts, "t_preproc_ms": (_pp_ms[i - direct_count] if (i - direct_count) < len(_pp_ms) else None)}),
                      "psm_used": (r.meta.get("params", {}).get("psm") if isinstance(r.meta, dict) else None),
                      "lang_code": (r.meta.get("params", {}).get("lang") if isinstance(r.meta, dict) else None),
                      "WxH": (None if i < direct_count else [int(pps[i-direct_count].meta.get("w",0)), int(pps[i-direct_count].meta.get("h",0))]),
                      "skew_deg": (None if i < direct_count else float(pps[i-direct_count].artifacts.get("skew_deg",0.0))),
                      "t_postproc_ms": _t_post_ms,
                  }
                  for i, r in enumerate(results)
              ],
              "wer_eval": {"is_bench": False, "wer": None},
              "finisher": finisher_payload,
              "orchestrator": orchestrator_payload,
          },
          "provenance": {
              "tesseract": {"version": "5.x", "params": {"oem": 3, "psm": psm_override or "auto"}},
              "vision": {"model": "latest", "called": bool(consent_api and engine in ['auto','vision'])},
              "preproc": {"dpi": 300, "deskew": not skip_deskew, "binarize": "clahe+otsu"},
          },
          "metrics": {"total_chars": total_chars, "headers_stripped_count": headers_stripped_count},
          "errors": [],
        }

        st.success("Ready")
        if results:
            # Show per-page tabs for preview
            tabs = st.tabs([f"Page {i+1}" for i in range(len(results))])
            for i, tab in enumerate(tabs):
                with tab:
                    st.text_area(f"Preview (page {i})", results[i].text, height=240)
            with st.expander("Finisher (spec) output", expanded=False):
                st.text_area("Normalized text (spec)", finisher_payload.get("text", ""), height=200)
                st.json(finisher_payload.get("audit", {}))
            with st.expander("Orchestrator (six-rule) output", expanded=False):
                st.text_area("Normalized text (orchestrator)", orchestrator_payload.get("text_nfc", ""), height=200)
                st.json({
                    "version": orchestrator_payload.get("version"),
                    "edits": orchestrator_payload.get("edits", []),
                    "flags": orchestrator_payload.get("flags", {}),
                    "metrics": orchestrator_payload.get("metrics", {}),
                })
            with st.expander("Debug: per-page OCR details", expanded=False):
                for i, r in enumerate(results):
                    if not isinstance(r.meta, dict):
                        continue
                    params = r.meta.get("params", {}) if isinstance(r.meta, dict) else {}
                    t_ms = r.meta.get("t_ms") if isinstance(r.meta, dict) else None
                    st.write({
                        "page": i,
                        "psm_used": params.get("psm"),
                        "lang_code": params.get("lang"),
                        "t_ocr_ms": t_ms,
                        "char_conf_mean": (sum(r.confidences)/len(r.confidences)/100.0) if r.confidences else 0.0,
                        "WxH": (results[i].meta.get("size", {}) if isinstance(results[i].meta, dict) else {}),
                        "t_postproc_ms": _t_post_ms,
                    })
        st.download_button("Export JSON", data=json.dumps(out, ensure_ascii=False),
                           file_name=f"{run_id}.json", mime="application/json")
    except Exception as e:
        show_exc("Runtime error", e)
        st.stop()
