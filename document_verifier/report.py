from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def build_report(document, output_path: Path) -> Path:
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    y = height - inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, y, "Document Verification Report")
    y -= 0.35 * inch

    c.setFont("Helvetica", 10)
    c.drawString(inch, y, f"Document ID: {document.id}")
    y -= 0.22 * inch
    c.drawString(inch, y, f"Original filename: {document.original_filename}")
    y -= 0.22 * inch
    c.drawString(inch, y, f"Source: {document.source_type}")
    y -= 0.35 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, "Explanation")
    y -= 0.22 * inch
    c.setFont("Helvetica", 9)
    for line in _wrap(document.explanation.summary if document.explanation else "No explanation stored.", 92):
        c.drawString(inch, y, line)
        y -= 0.18 * inch

    y -= 0.12 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, "Detected Issues")
    y -= 0.22 * inch
    c.setFont("Helvetica", 9)
    if document.detections:
        for detection in document.detections:
            line = (
                f"{detection.issue_type} | confidence {detection.confidence:.2f} | "
                f"box x={detection.x}, y={detection.y}, w={detection.width}, h={detection.height}"
            )
            for wrapped in _wrap(line, 92):
                c.drawString(inch, y, wrapped)
                y -= 0.18 * inch
            if y < inch:
                c.showPage()
                y = height - inch
    else:
        c.drawString(inch, y, "No issues detected.")
        y -= 0.18 * inch

    y -= 0.15 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, "Extracted Text")
    y -= 0.22 * inch
    c.setFont("Helvetica", 8)
    text = document.extracted_data.full_text if document.extracted_data else ""
    for line in _wrap(text or "(empty OCR output)", 105):
        c.drawString(inch, y, line)
        y -= 0.15 * inch
        if y < inch:
            c.showPage()
            y = height - inch
            c.setFont("Helvetica", 8)

    c.save()
    return output_path


def _wrap(text: str, limit: int):
    for raw_line in text.splitlines() or [""]:
        line = raw_line.strip()
        while len(line) > limit:
            split_at = line.rfind(" ", 0, limit)
            if split_at < 20:
                split_at = limit
            yield line[:split_at]
            line = line[split_at:].strip()
        yield line
