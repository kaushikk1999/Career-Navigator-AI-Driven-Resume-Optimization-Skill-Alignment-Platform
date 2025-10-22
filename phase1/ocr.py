import os, base64, time, warnings
from io import BytesIO
from pathlib import Path
from typing import Optional
from PIL import Image
import pytesseract

from .contracts import PreprocPage, OcrResult

# Prefer Homebrew Tesseract unless overridden
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/opt/homebrew/bin/tesseract")

LANG_HINT_TO_TESS = {
    "en": "eng",
    "hi": "hin",
    "bn": "ben",
    "hr": "hrv",
    "und": "eng",
}
_WARNED_LANGS = set()

# ---------- helpers ----------
def _rotate_by_osd(im: Image.Image) -> Image.Image:
    """Use Tesseract OSD to rotate 0/90/180/270 before OCR (best-effort)."""
    try:
        osd = pytesseract.image_to_osd(im)
        for line in osd.splitlines():
            if line.strip().lower().startswith("rotate:"):
                deg = int(line.split(":")[1].strip())
                if deg in (90, 180, 270):
                    return im.rotate(-deg, expand=True)
    except Exception:
        pass
    return im

def have_traineddata(code: str) -> bool:
    """Check tessdata availability for every segment in a composite code."""
    segments = [seg.strip() for seg in (code or "").split("+") if seg.strip()]
    if not segments:
        return False
    candidates = []
    prefix = os.environ.get("TESSDATA_PREFIX")
    if prefix:
        candidates.append(Path(prefix))
    tess_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
    if tess_cmd:
        tess_path = Path(tess_cmd).expanduser().resolve()
        candidates.append(tess_path.parent / "../share/tessdata")
    seen = {}
    for seg in segments:
        if seg == "osd":
            continue
        if seg in seen:
            continue
        found = False
        for root in candidates:
            try:
                root_path = Path(root).expanduser().resolve()
            except OSError:
                continue
            if (root_path / f"{seg}.traineddata").is_file():
                found = True
                break
        if not found:
            return False
        seen[seg] = True
    return True


def resolve_tess_lang(lang_hint: str, allow_osd: bool = True) -> str:
    orig = (lang_hint or "und").lower()
    segments = [seg.strip() for seg in orig.split("+") if seg.strip()]
    include_osd = allow_osd and os.getenv("TESSERACT_INCLUDE_OSD", "1") != "0"

    if segments and all(len(seg) == 3 or seg == "osd" for seg in segments):
        base_parts = [seg for seg in segments if seg != "osd"]
        if not base_parts:
            base_parts = ["eng"]
        candidate_parts = base_parts[:]
        if include_osd or "osd" in segments:
            if "osd" not in candidate_parts:
                candidate_parts.append("osd")
        candidate = "+".join(candidate_parts)
    else:
        base = LANG_HINT_TO_TESS.get(orig, LANG_HINT_TO_TESS["und"])
        candidate_parts = [base]
        if include_osd and "osd" not in base:
            candidate_parts.append("osd")
        candidate = "+".join(candidate_parts)

    if not have_traineddata(candidate):
        if orig not in _WARNED_LANGS:
            warnings.warn(f"Missing tessdata for '{candidate}'. Falling back to 'eng+osd'.", RuntimeWarning)
            _WARNED_LANGS.add(orig)
        resolve_tess_lang.last_fallback = True
        return "eng+osd"

    resolve_tess_lang.last_fallback = False
    return candidate


resolve_tess_lang.last_fallback = False

def _auto_psm(w: int, h: int, lines: float = 0.0) -> int:
    aspect = float(w / h) if h else 1.0
    if lines <= 3:
        return 11
    if aspect > 1.55 and lines < 35:
        return 4
    return 6 if lines >= 8 else 4

def _choose_psm(page: PreprocPage, im: Image.Image, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    # Feature flag: allow disabling auto PSM via env (rollback)
    if os.getenv("PHASE1_PSM_AUTO", "1").lower() in {"0", "false", "no"}:
        return 6
    artifacts = page.artifacts or {}
    lines = float(artifacts.get("line_count", 0.0))
    return _auto_psm(im.width, im.height, lines)


def _should_retry(psm: int, mean_conf: float) -> bool:
    return psm in (4, 6) and mean_conf < 70.0

# ---------- Tesseract OCR ----------
def _pil_from_bytes(img_bytes: bytes) -> Image.Image:
    im = Image.open(BytesIO(img_bytes))
    if im.mode not in ("L", "RGB"):
        im = im.convert("RGB")
    return im

def _mean_conf(conf):
    return (sum(conf)/len(conf)) if conf else 0.0

def _build_cfg(psm: int) -> str:
    tessdir = os.environ.get("TESSDATA_PREFIX")
    base = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    return (f'--tessdata-dir "{tessdir}" ' + base) if tessdir else base


def _run_tess(im: Image.Image, lang: str, cfg: str):
    txt = pytesseract.image_to_string(im, lang=lang, config=cfg)
    data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, lang=lang, config=cfg)
    conf = [float(c) for c in data.get("conf", []) if c != "-1"]
    return txt, conf, _mean_conf(conf)


def ocr_tesseract(p: PreprocPage, lang_hint: str = "en", psm_hint: Optional[object] = None) -> OcrResult:
    t0 = time.time()
    im = _rotate_by_osd(_pil_from_bytes(p.bytes_gray))
    im.info["dpi"] = (300, 300)
    override: Optional[int]
    if psm_hint is None or psm_hint == "auto":
        override = None
    else:
        try:
            override = int(psm_hint)
        except (ValueError, TypeError):
            override = None
    psm = _choose_psm(p, im, override)
    lang_code = resolve_tess_lang(lang_hint)
    cfg = _build_cfg(psm)
    txt, conf, mean1 = _run_tess(im, lang_code, cfg)
    tried = False
    if _should_retry(psm, mean1):
        tried = True
        alt_psm = 6 if psm == 4 else 4
        alt_cfg = _build_cfg(alt_psm)
        alt_txt, alt_conf, mean2 = _run_tess(im, lang_code, alt_cfg)
        if mean2 > mean1:
            txt, conf, mean1, psm, cfg = alt_txt, alt_conf, mean2, alt_psm, alt_cfg
    meta = {
        "engine": "tesseract",
        "t_ms": int((time.time() - t0) * 1000),
        "params": {"oem": 3, "psm": psm, "lang": lang_code, "retry": tried},
        "char_conf_mean": float(mean1),
        "cfg": cfg,
        "size": p.meta if hasattr(p, "meta") else {},
    }
    return OcrResult(text=txt, confidences=conf, meta=meta)

# ---------- Vision OCR (service-account bearer OR API key) ----------
def _get_api_key() -> Optional[str]:
    try:
        import streamlit as st
        return st.secrets.get("GCP_VISION_KEY") or os.getenv("GCP_VISION_KEY")
    except Exception:
        return os.getenv("GCP_VISION_KEY")

def _get_bearer_from_service_account() -> Optional[str]:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path or not os.path.isfile(cred_path):
        return None
    try:
        from google.oauth2 import service_account
        import google.auth.transport.requests as gar
        creds = service_account.Credentials.from_service_account_file(
            cred_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(gar.Request())
        return creds.token
    except Exception:
        return None

def ocr_vision(p: PreprocPage, lang: str = "en") -> Optional[OcrResult]:
    import requests, time as _time
    t0 = time.time()

    # Tunables (env): defaults are conservative for local
    # Defaults target ~900ms with one quick retry
    conn_timeout = float(os.getenv("VISION_CONNECT_TIMEOUT_S", "0.3"))
    read_timeout = float(os.getenv("VISION_READ_TIMEOUT_S", "0.6"))
    retries      = int(os.getenv("VISION_RETRIES", "1"))
    backoff_s    = float(os.getenv("VISION_RETRY_BACKOFF_S", "0.1"))

    bearer = _get_bearer_from_service_account()
    api_key = _get_api_key()
    if not bearer and not api_key:
        return None

    url = "https://vision.googleapis.com/v1/images:annotate"
    payload = {
        "requests": [{
            "image": {"content": base64.b64encode(p.bytes_gray).decode()},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            "imageContext": {"languageHints": [lang]},
        }]
    }
    headers = {"Content-Type": "application/json"}; params = {}
    if bearer: headers["Authorization"] = f"Bearer {bearer}"
    else:      params["key"] = api_key

    attempt = 0
    while True:
        try:
            r = requests.post(url, headers=headers, params=params, json=payload,
                              timeout=(conn_timeout, read_timeout))
            r.raise_for_status()
            txt = r.json()["responses"][0].get("fullTextAnnotation", {}).get("text", "")
            return OcrResult(text=txt, confidences=[], meta={"engine":"vision","t_ms":int((time.time()-t0)*1000)})
        except (requests.ReadTimeout, requests.ConnectTimeout):
            attempt += 1
            if attempt > retries:
                return None  # graceful degrade to Tesseract
            _time.sleep(backoff_s)
        except requests.RequestException:
            return None
