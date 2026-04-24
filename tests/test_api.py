from pathlib import Path

from document_verifier import create_app
from document_verifier.config import Config
from document_verifier.sample_data import create_sample


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


def test_api_verify_end_to_end():
    runtime = Path(__file__).resolve().parent / "runtime" / "api"
    runtime.mkdir(parents=True, exist_ok=True)
    TestConfig.STORAGE_DIR = runtime / "storage"
    TestConfig.UPLOAD_DIR = TestConfig.STORAGE_DIR / "uploads"
    TestConfig.PROCESSED_DIR = TestConfig.STORAGE_DIR / "processed"
    TestConfig.SAMPLE_DIR = TestConfig.STORAGE_DIR / "samples"
    TestConfig.INSTANCE_DIR = runtime / "instance"
    app = create_app(TestConfig)
    sample = create_sample(runtime / "tampered.png")

    with app.test_client() as client:
        with sample.open("rb") as handle:
            response = client.post("/api/verify", data={"document": (handle, "tampered.png")}, content_type="multipart/form-data")
        assert response.status_code == 200
        data = response.get_json()
        assert data["document_id"] >= 1
        assert data["ocr"]["engine"] == "pytesseract"
        assert len(data["issues"]) >= 2
        assert data["explanation"]["structured"]["basis"]
        assert data["explanation"]["structured"]["reference_datasets"]
