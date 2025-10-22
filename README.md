# Phase 1 · OCR Ingest (Streamlit)

Lightweight OCR ingest for resumes and similar documents. Accepts PDFs and images, preprocesses pages (deskew, enhance), runs Tesseract locally with optional Google Vision fallback, and exports a clean JSON bundle. Digital PDFs skip OCR and use the exact text layer for fidelity.

## Prerequisites

- Python 3.9+
- System tools (install via your OS package manager):
  - Tesseract OCR (v5+ recommended)
  - Poppler (provides `pdftocairo` used by `pdf2image`)

Examples:
- macOS (Homebrew)
  - `brew install tesseract poppler`
- Ubuntu/Debian
  - `sudo apt-get update`
  - `sudo apt-get install -y tesseract-ocr poppler-utils`
  - Optional languages: `sudo apt-get install -y tesseract-ocr-ben tesseract-ocr-hin tesseract-ocr-hrv`
- Windows
  - Install Tesseract (UB Mannheim build recommended). Ensure `tesseract.exe` is on `PATH` or set `TESSERACT_CMD`.
  - Install Poppler for Windows and add `pdftocairo.exe` to `PATH`.

## Install Python deps

- Runtime
  - `pip install -r requirements.txt`
- Dev/Test
  - `pip install -r requirements-dev.txt`

This project uses only light dependencies. New runtime dependency: `pdfminer.six` for digital PDF text layer.

## Environment configuration

- Tesseract binary (override if not autodetected):
  - `TESSERACT_CMD` → absolute path to tesseract binary
    - macOS (Apple Silicon default in code): `/opt/homebrew/bin/tesseract`
    - macOS (Intel): `/usr/local/bin/tesseract`
    - Windows example: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
- Tesseract language data (recommended to set explicitly):
  - `TESSDATA_PREFIX` → folder containing `*.traineddata` (must include `eng` and `osd`. Optional: `ben`, `hin`, `hrv`)
    - Common paths:
      - macOS Homebrew: `/opt/homebrew/share/tessdata`
      - Debian/Ubuntu: `/usr/share/tesseract-ocr/5/tessdata` (or `/usr/share/tesseract-ocr/4.00/tessdata`)
- Google Vision (optional cloud fallback):
  - API key: `GCP_VISION_KEY`
  - or Service account JSON: `GOOGLE_APPLICATION_CREDENTIALS` → `/absolute/path/to/credentials.json`
  - Tunables (increase for slower networks):
    - `VISION_CONNECT_TIMEOUT_S` (default 1.2)
    - `VISION_READ_TIMEOUT_S` (default 5.0)
    - `VISION_RETRIES` (default 2)
    - `VISION_RETRY_BACKOFF_S` (default 0.4)

## Credentials & Secrets (No Keys in Git)

- Never commit cloud keys or service-account JSON files.
- Copy `.env.example` to `.env`, then point `GCP_SA_KEY_PATH` to your local credential file (keep it outside the repo).
- `app.py` sets `GOOGLE_APPLICATION_CREDENTIALS` automatically from `GCP_SA_KEY_PATH` so the Google client libraries work without code edits.
- Streamlit Cloud: store the JSON payload under `st.secrets["gcp"]["service_account_json"]` and write it to a temp file before launching the app, for example:
  ```python
  import os, tempfile
  from pathlib import Path

  if "gcp" in st.secrets and "service_account_json" in st.secrets["gcp"]:
      tmp_key = Path(tempfile.gettempdir()) / "gcp-sa.json"
      tmp_key.write_text(st.secrets["gcp"]["service_account_json"])
      os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp_key)
  ```
- Rotate leaked keys immediately and prefer Workload Identity Federation when possible.

## Run

- Launch the app
  - `streamlit run app.py`
- UI options
  - Engine: `auto` (Tesseract + optional Vision), `tesseract`, or `vision`
  - Language hint: `en`, `hi`, `bn`, `hr`, `und` (mapped to Tesseract codes with OSD)
  - PSM (page segmentation): `auto`, `4`, `6`, `11`
  - Skip deskew: disable rotation adjustment for already aligned pages
  - Normalize output (typo fixes): off by default; when enabled applies targeted replacements (e.g., `Kokata,India → Kolkata, India`)
- Export: download a JSON including per-page text, confidences (OCR pages), quality flags, and provenance.

## Behavior summary

- Digital PDFs: use embedded text per page (exact fidelity) when available; otherwise rasterize and OCR.
- Scans/Images: preprocess to grayscale for OCR and binarized image for QA; Tesseract LSTM (`--oem 3`) with heuristic/overrideable PSM; Vision used if enabled and beneficial.
- Post-processing: cleans ligatures/quotes, heals paragraph wraps, normalizes bullets, removes bogus `http://…@…` lines, and keeps page breaks as `\f`.
  - Pagination is preserved: per-page texts are always joined with a single form-feed (`\f`).
  - Soft hyphens (U+00AD) are removed; common ligatures (ﬀ, ﬁ, ﬂ, ﬃ, ﬄ) are expanded.
  - A space is enforced after label colons at line start (e.g., `Email:abc` → `Email: abc`).

### Quick test run (Kaushik’s 93.pdf acceptance)

Run the unit tests; these include acceptance checks for multi-page completeness, link preservation, headings, readability, and match score. If the sample PDF is not available at `/mnt/data/Kaushik's 93.pdf`, the tests use a mock text that exercises the same constraints.

- `pytest -q tests/test_pdf_kaushik93.py`

Toggles: enable “Normalize output (typo fixes)” in the UI to apply optional known typo corrections.

## Testing

- `pytest -q`
  - Language mapping fallback
  - PSM auto-selection heuristics
  - Bullet normalization and header-tail trimming
  - Digital PDF text-layer detection
  - Fusion decision accounts for cleaned text similarity and confidence

## Troubleshooting

- Tesseract not found
  - Set `TESSERACT_CMD` to the absolute path; verify `tesseract --version` works in your shell.
- Missing language packs
  - Install the specific packs and set `TESSDATA_PREFIX` to the tessdata directory.
- PDF rasterization fails
  - Ensure `pdftocairo` from Poppler is on `PATH`.
- Vision timeouts
  - Increase Vision env timeouts/retries or disable Vision consent to stay on-device.

## Notes

- CPU-only, low RAM target. No background daemons are started.
- All functions in the pipeline are pure/deterministic; errors are surfaced as structured codes and guarded in the UI.
 
