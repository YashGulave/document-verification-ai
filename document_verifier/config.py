import os
from pathlib import Path


def _load_local_env(base_dir: Path):
    env_path = base_dir / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    _load_local_env(BASE_DIR)

    STORAGE_DIR = BASE_DIR / "storage"
    UPLOAD_DIR = STORAGE_DIR / "uploads"
    PROCESSED_DIR = STORAGE_DIR / "processed"
    SAMPLE_DIR = STORAGE_DIR / "samples"
    INSTANCE_DIR = BASE_DIR / "instance"
    MODEL_DIR = BASE_DIR / "models"
    DOCUMENT_TAMPER_MODEL_PATH = Path(os.getenv("DOCUMENT_TAMPER_MODEL_PATH", MODEL_DIR / "document_tamper_model.joblib"))
    DOCUMENT_TAMPER_MODEL_THRESHOLD = float(os.getenv("DOCUMENT_TAMPER_MODEL_THRESHOLD", "0.65"))

    SECRET_KEY = "local-hackathon-secret"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{INSTANCE_DIR / 'verifier.sqlite3'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"}

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    GROQ_TIMEOUT_SECONDS = int(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))

    @classmethod
    def ensure_directories(cls):
        for path in (cls.UPLOAD_DIR, cls.PROCESSED_DIR, cls.SAMPLE_DIR, cls.INSTANCE_DIR, cls.MODEL_DIR):
            path.mkdir(parents=True, exist_ok=True)
