from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    display_filename = db.Column(db.String(255), nullable=False)
    highlighted_filename = db.Column(db.String(255), nullable=False)
    source_type = db.Column(db.String(30), nullable=False)
    mime_type = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    extracted_data = db.relationship("ExtractedData", back_populates="document", uselist=False, cascade="all, delete-orphan")
    detections = db.relationship("DetectionResult", back_populates="document", cascade="all, delete-orphan")
    explanation = db.relationship("Explanation", back_populates="document", uselist=False, cascade="all, delete-orphan")


class ExtractedData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("document.id"), nullable=False)
    full_text = db.Column(db.Text, nullable=False)
    ocr_engine = db.Column(db.String(80), nullable=False)
    status = db.Column(db.String(40), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    document = db.relationship("Document", back_populates="extracted_data")


class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("document.id"), nullable=False)
    issue_type = db.Column(db.String(80), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    x = db.Column(db.Integer, nullable=False)
    y = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    details = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    document = db.relationship("Document", back_populates="detections")

    def as_dict(self):
        return {
            "issue_type": self.issue_type,
            "confidence": round(self.confidence, 3),
            "box": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "details": self.details,
        }


class Explanation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("document.id"), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    structured_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    document = db.relationship("Document", back_populates="explanation")
