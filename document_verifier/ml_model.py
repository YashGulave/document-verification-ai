from functools import lru_cache
from pathlib import Path

import joblib

from .detection import Issue
from .ml_features import extract_image_features


@lru_cache(maxsize=2)
def _load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


def detect_with_trained_model(image_path: Path, model_path: Path, threshold: float) -> Issue | None:
    model = _load_model(str(model_path))
    if model is None:
        return None

    features = extract_image_features(image_path).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][1])
    else:
        probability = float(model.predict(features)[0])

    if probability < threshold:
        return None

    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image for ML model detection: {image_path}")
    height, width = image.shape[:2]
    return Issue(
        "trained_model_tamper_signal",
        probability,
        0,
        0,
        width,
        height,
        (
            "The trained document-tamper classifier scored this document as suspicious "
            f"with probability {probability:.2f}."
        ),
    )
