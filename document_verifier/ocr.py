from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import TesseractNotFoundError

from .preprocess import preprocess_document_scan


def _candidate_tesseract_paths() -> list[Path]:
    env_cmd = __import__("os").environ.get("TESSERACT_CMD", "").strip()
    paths = []
    if env_cmd:
        paths.append(Path(env_cmd))
    paths.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
    )
    return paths


def configure_tesseract_cmd() -> str | None:
    current = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
    if current and Path(current).exists():
        return current
    for candidate in _candidate_tesseract_paths():
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)
    return None


def get_tesseract_version_status() -> tuple[str, str]:
    configure_tesseract_cmd()
    try:
        version = str(pytesseract.get_tesseract_version())
        return "ok", version
    except Exception as exc:
        return "unavailable", str(exc)


def pdf_first_page_to_image(pdf_path: Path, output_path: Path) -> Path:
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        raise ValueError("PDF has no pages")
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(output_path)
    doc.close()
    return output_path


def load_image_for_processing(path: Path, processed_dir: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        page_path = pdf_first_page_to_image(path, processed_dir / f"{path.stem}_page1_raw.png")
        return preprocess_document_scan(page_path, processed_dir / f"{path.stem}_page1_scanned.png")

    scanned_path = processed_dir / f"{path.stem}_scanned.png"
    return preprocess_document_scan(path, scanned_path)


def extract_text(image_path: Path) -> tuple[str, str]:
    image = Image.open(image_path)
    configure_tesseract_cmd()
    try:
        text = pytesseract.image_to_string(image)
        return text.strip(), "ok"
    except TesseractNotFoundError as exc:
        return (
            "OCR unavailable: install the native Tesseract binary and add it to PATH. "
            f"pytesseract error: {exc}",
            "ocr_setup_error",
        )


def deskew_and_threshold(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image for OCR preprocessing: {image_path}")
    image = cv2.resize(image, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
