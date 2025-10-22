from dataclasses import dataclass, field
from typing import Literal, Dict, List, Any

ErrorCode = Literal[
    "BAD_MIME",
    "FILE_TOO_LARGE",
    "TOO_MANY_PAGES",
    "PAGE_TOO_LARGE",
    "UNSUPPORTED_PDF",
    "OCR_ENGINE_UNAVAILABLE",
    "EXTERNAL_API_DISABLED",
    "TIMEOUT",
    "UNKNOWN",
]

def error(code: ErrorCode) -> Dict[str, str]:
    return {"error": code}

@dataclass
class Page:
    bytes: bytes
    mime: str
    index: int
    # Optional metadata for upstream/external extraction (e.g., PDF text layer)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PreprocPage:
    bytes_gray: bytes  # grayscale PNG tuned for OCR
    bytes_binary: bytes  # high-contrast PNG for QA/diagnostics
    index: int
    meta: Dict[str, float]
    artifacts: Dict[str, float]

    @property
    def bytes(self) -> bytes:
        """Backward compatible alias for grayscale bytes."""
        return self.bytes_gray

@dataclass
class OcrResult:
    text: str
    confidences: List[float]
    meta: Dict[str, object]
