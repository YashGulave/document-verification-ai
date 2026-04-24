import base64
import json
import mimetypes
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_file, url_for
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .detection import detect_tampering, draw_highlights
from .models import DetectionResult, Document, Explanation, ExtractedData, db
from .ocr import extract_text, get_tesseract_version_status, load_image_for_processing
from .reasoning import generate_explanation
from .report import build_report


bp = Blueprint("document_verifier", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


@bp.app_errorhandler(ValueError)
def value_error(error):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"error": str(error)}), 400
    return render_template("index.html", error=str(error)), 400


@bp.get("/health")
def health():
    ocr_status, tesseract_version = get_tesseract_version_status()
    return jsonify({"status": "ok", "ocr_status": ocr_status, "tesseract": tesseract_version})


@bp.post("/upload")
def upload():
    file = request.files.get("document")
    if file is None or file.filename == "":
        return render_template("index.html", error="Choose an image or PDF to verify."), 400
    document = _save_and_process_file(file, source_type="upload")
    return _render_or_json(document)


@bp.post("/capture")
def capture():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data or "," not in image_data:
        return jsonify({"error": "Camera image data is missing."}), 400

    header, encoded = image_data.split(",", 1)
    extension = "jpg" if "jpeg" in header or "jpg" in header else "png"
    filename = f"camera_capture.{extension}"
    raw = base64.b64decode(encoded)
    storage = FileStorage(stream=_BytesIO(raw), filename=filename, content_type=f"image/{extension}")
    document = _save_and_process_file(storage, source_type="camera")
    return jsonify(_document_payload(document))


@bp.post("/api/verify")
def api_verify():
    file = request.files.get("document")
    if file is None or file.filename == "":
        return jsonify({"error": "Attach a file with form field name 'document'."}), 400
    document = _save_and_process_file(file, source_type="api")
    return jsonify(_document_payload(document))


@bp.get("/documents/<int:document_id>")
def result(document_id):
    document = Document.query.get_or_404(document_id)
    return render_template("result.html", document=document, payload=_document_payload(document))


@bp.get("/api/documents/<int:document_id>")
def api_document(document_id):
    document = Document.query.get_or_404(document_id)
    return jsonify(_document_payload(document))


@bp.get("/documents/<int:document_id>/report")
def report(document_id):
    document = Document.query.get_or_404(document_id)
    output_path = Path(current_app.config["PROCESSED_DIR"]) / f"document_{document.id}_report.pdf"
    build_report(document, output_path)
    return send_file(output_path, as_attachment=True, download_name=f"verification_report_{document.id}.pdf")


@bp.get("/storage/<path:folder>/<path:filename>")
def storage_file(folder, filename):
    base = Path(current_app.config["STORAGE_DIR"]).resolve()
    target = (base / folder / filename).resolve()
    if not str(target).startswith(str(base)):
        return jsonify({"error": "Invalid path"}), 400
    return send_file(target)


def _save_and_process_file(file: FileStorage, source_type: str) -> Document:
    original = secure_filename(file.filename or "document.png")
    extension = original.rsplit(".", 1)[-1].lower() if "." in original else "png"
    if extension not in current_app.config["ALLOWED_EXTENSIONS"]:
        raise ValueError(f"Unsupported file extension: {extension}")

    unique = f"{uuid.uuid4().hex}_{original}"
    upload_path = Path(current_app.config["UPLOAD_DIR"]) / unique
    file.save(upload_path)

    display_path = load_image_for_processing(upload_path, Path(current_app.config["PROCESSED_DIR"]))
    issues = detect_tampering(
        display_path,
        current_app.config.get("DOCUMENT_TAMPER_MODEL_PATH"),
        current_app.config.get("DOCUMENT_TAMPER_MODEL_THRESHOLD", 0.65),
    )
    highlighted_path = Path(current_app.config["PROCESSED_DIR"]) / f"{display_path.stem}_highlighted.png"
    draw_highlights(display_path, issues, highlighted_path)
    extracted_text, ocr_status = extract_text(display_path)
    issue_dicts = [issue.as_dict() for issue in issues]
    groq_api_key = "" if current_app.testing else current_app.config.get("GROQ_API_KEY", "")
    summary, structured = generate_explanation(
        issue_dicts,
        extracted_text,
        ocr_status,
        groq_api_key=groq_api_key,
        groq_model=current_app.config.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_api_url=current_app.config.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"),
        timeout_seconds=current_app.config.get("GROQ_TIMEOUT_SECONDS", 20),
    )

    mime_type = file.content_type or mimetypes.guess_type(original)[0] or "application/octet-stream"
    document = Document(
        original_filename=original,
        stored_filename=upload_path.name,
        display_filename=display_path.name,
        highlighted_filename=highlighted_path.name,
        source_type=source_type,
        mime_type=mime_type,
    )
    db.session.add(document)
    db.session.flush()

    db.session.add(ExtractedData(document_id=document.id, full_text=extracted_text, ocr_engine="pytesseract", status=ocr_status))
    for issue in issues:
        db.session.add(
            DetectionResult(
                document_id=document.id,
                issue_type=issue.issue_type,
                confidence=float(issue.confidence),
                x=issue.x,
                y=issue.y,
                width=issue.width,
                height=issue.height,
                details=issue.details,
            )
        )
    db.session.add(Explanation(document_id=document.id, summary=summary, structured_json=structured))
    db.session.commit()
    return document


def _document_payload(document: Document) -> dict:
    issues = [_present_issue(detection.as_dict()) for detection in document.detections]
    explanation_json = {}
    if document.explanation and document.explanation.structured_json:
        explanation_json = json.loads(document.explanation.structured_json)
    return {
        "document_id": document.id,
        "source_type": document.source_type,
        "original_filename": document.original_filename,
        "image_url": url_for("document_verifier.storage_file", folder="processed", filename=document.display_filename),
        "highlighted_url": url_for("document_verifier.storage_file", folder="processed", filename=document.highlighted_filename),
        "report_url": url_for("document_verifier.report", document_id=document.id),
        "ocr": {
            "status": document.extracted_data.status if document.extracted_data else "missing",
            "engine": document.extracted_data.ocr_engine if document.extracted_data else "pytesseract",
            "text": document.extracted_data.full_text if document.extracted_data else "",
        },
        "issues": issues,
        "explanation": {
            "summary": document.explanation.summary if document.explanation else "",
            "structured": explanation_json,
        },
    }


def _present_issue(issue: dict) -> dict:
    names = {
        "edge_inconsistency": "Possible pasted or retyped content",
        "noise_variance_anomaly": "Possible edited background field",
        "localized_blur_anomaly": "Possible erased or softened field",
        "trained_model_tamper_signal": "AI tamper-classifier signal",
    }
    meanings = {
        "edge_inconsistency": "Text, border, or stamp edges in this area do not visually match the surrounding document.",
        "noise_variance_anomaly": "The paper texture or compression pattern changes around this area.",
        "localized_blur_anomaly": "This area is softer than nearby content and may have been cleaned or rewritten.",
        "trained_model_tamper_signal": "The trained classifier marked the document-level pattern as suspicious.",
    }
    issue_type = issue.get("issue_type", "")
    issue["display_name"] = names.get(issue_type, issue_type.replace("_", " ").title())
    issue["review_meaning"] = meanings.get(issue_type, issue.get("details", "This area should be reviewed."))
    return issue


def _render_or_json(document: Document):
    if request.accept_mimetypes.best == "application/json":
        return jsonify(_document_payload(document))
    return render_template("result.html", document=document, payload=_document_payload(document))


class _BytesIO:
    def __init__(self, data: bytes):
        from io import BytesIO

        self._buffer = BytesIO(data)

    def read(self, *args):
        return self._buffer.read(*args)

    def seek(self, *args):
        return self._buffer.seek(*args)
