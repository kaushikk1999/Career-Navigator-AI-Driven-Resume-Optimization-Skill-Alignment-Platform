import cv2, numpy as np
from .contracts import Page, PreprocPage


def _estimate_line_stats(binary_img: np.ndarray):
    if binary_img.ndim != 2:
        return 0.0, 0.0
    row_fill = (binary_img == 0).mean(axis=1)
    mask = row_fill > 0.12
    if not np.any(mask):
        return 0.0, float((binary_img == 0).mean())
    transitions = np.diff(mask.astype(np.int8))
    line_count = int(np.sum(transitions == 1) + (1 if mask[0] else 0))
    return float(line_count), float((binary_img == 0).mean())


def _autocrop_gray(gray: np.ndarray, pad: int = 6) -> np.ndarray:
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thr)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    h_lim, w_lim = gray.shape
    x0 = max(x - pad, 0); y0 = max(y - pad, 0)
    x1 = min(x + w + pad, w_lim); y1 = min(y + h + pad, h_lim)
    return gray[y0:y1, x0:x1]


def _deskew(gray: np.ndarray, max_deg: float = 5.0):
    edges = cv2.Canny(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angs = [(theta - np.pi / 2.0) for _, theta in lines[:, 0]]
        angle = float(np.clip(np.degrees(np.median(angs)), -max_deg, max_deg))
        center = (gray.shape[1] / 2, gray.shape[0] / 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, mat, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return gray, angle


def _enhance_gray(gray: np.ndarray) -> np.ndarray:
    """Improve local contrast using CLAHE without destroying gradients."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _resize_to_band(img: np.ndarray, min_long=1800, max_long=3200) -> np.ndarray:
    h, w = img.shape
    long_edge = max(h, w)
    if long_edge < min_long:
        scale = min_long / long_edge
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    if long_edge > max_long:
        scale = max_long / long_edge
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _make_binary(gray: np.ndarray) -> np.ndarray:
    """Create a QA binary view while keeping text strokes intact."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def _encode_png(gray: np.ndarray, binary: np.ndarray):
    ok_g, enc_g = cv2.imencode(".png", gray)
    ok_b, enc_b = cv2.imencode(".png", binary)
    if not ok_g or not ok_b:
        raise ValueError("PNG encode failed")
    return enc_g.tobytes(), enc_b.tobytes()


def _unsharp(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return np.clip(cv2.addWeighted(gray, 1.6, blur, -0.6, 0), 0, 255).astype(np.uint8)


def preproc(page: Page, long_edge_cap: int = 3200, enable_deskew: bool = True):
    img = cv2.imdecode(np.frombuffer(page.bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _autocrop_gray(gray)
    gray = _resize_to_band(gray, max_long=long_edge_cap)
    gray = _enhance_gray(gray)
    if enable_deskew:
        gray, angle = _deskew(gray)
    else:
        angle = 0.0
    ocr_gray = _unsharp(gray)
    binary = _make_binary(gray)
    h, w = ocr_gray.shape
    gray_bytes, bin_bytes = _encode_png(ocr_gray, binary)
    line_count, fill_ratio = _estimate_line_stats(binary)
    lap_var = float(cv2.Laplacian(ocr_gray, cv2.CV_64F).var()) if ocr_gray.size else 0.0
    mean_intensity = float(np.mean(ocr_gray) / 255.0) if ocr_gray.size else 0.0
    artifacts = {
        "skew_deg": float(angle),
        "line_count": line_count,
        "fill_ratio": fill_ratio,
        "aspect_ratio": float(w / h) if h else 1.0,
        "laplacian_var": lap_var,
        "mean_intensity": mean_intensity,
    }
    meta = {"w": float(w), "h": float(h), "dpi": 300.0}
    return PreprocPage(bytes_gray=gray_bytes, bytes_binary=bin_bytes, index=page.index, meta=meta, artifacts=artifacts)

# Allow tests to call __import__('phase1.preproc', fromlist=['preproc']).preproc.preproc(...)
try:
    preproc.preproc = preproc  # type: ignore[attr-defined]
except Exception:
    pass
