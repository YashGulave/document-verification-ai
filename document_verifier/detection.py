from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Issue:
    issue_type: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    details: str

    def as_dict(self):
        return {
            "issue_type": self.issue_type,
            "confidence": round(float(self.confidence), 3),
            "box": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "details": self.details,
        }


def _tile_boxes(width: int, height: int, tile: int = 96):
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            yield x, y, min(tile, width - x), min(tile, height - y)


def _score_to_confidence(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def detect_edge_inconsistency(gray: np.ndarray) -> list[Issue]:
    edges = cv2.Canny(gray, 80, 180)
    h, w = gray.shape
    densities = []
    boxes = []
    for x, y, bw, bh in _tile_boxes(w, h):
        roi = edges[y : y + bh, x : x + bw]
        densities.append(float(np.mean(roi > 0)))
        boxes.append((x, y, bw, bh))

    if len(densities) < 4:
        return []

    arr = np.array(densities)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) + 1e-6
    issues = []
    for density, (x, y, bw, bh) in zip(densities, boxes):
        robust_z = abs(density - median) / mad
        if robust_z > 4.5 and density > median * 1.6:
            confidence = _score_to_confidence(robust_z, 4.5, 12.0)
            issues.append(
                Issue(
                    "edge_inconsistency",
                    max(confidence, 0.55),
                    x,
                    y,
                    bw,
                    bh,
                    f"Local edge density {density:.4f} is unusually high compared with document median {median:.4f}.",
                )
            )
    return _merge_similar(issues)


def detect_noise_variance(gray: np.ndarray) -> list[Issue]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.absdiff(gray, blurred)
    h, w = gray.shape
    variances = []
    boxes = []
    for x, y, bw, bh in _tile_boxes(w, h):
        roi = residual[y : y + bh, x : x + bw]
        variances.append(float(np.var(roi)))
        boxes.append((x, y, bw, bh))

    arr = np.array(variances)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = float(q3 - q1) + 1e-6
    threshold = float(q3 + 2.2 * iqr)
    issues = []
    for variance, (x, y, bw, bh) in zip(variances, boxes):
        if variance > threshold and variance > 8.0:
            confidence = _score_to_confidence(variance, threshold, threshold * 2.5)
            issues.append(
                Issue(
                    "noise_variance_anomaly",
                    max(confidence, 0.52),
                    x,
                    y,
                    bw,
                    bh,
                    f"Residual noise variance {variance:.2f} exceeds local threshold {threshold:.2f}.",
                )
            )
    return _merge_similar(issues)


def detect_blur_anomaly(gray: np.ndarray) -> list[Issue]:
    h, w = gray.shape
    scores = []
    boxes = []
    for x, y, bw, bh in _tile_boxes(w, h):
        roi = gray[y : y + bh, x : x + bw]
        scores.append(float(cv2.Laplacian(roi, cv2.CV_64F).var()))
        boxes.append((x, y, bw, bh))

    arr = np.array(scores)
    median = float(np.median(arr))
    low_threshold = max(float(np.percentile(arr, 12)), median * 0.22)
    issues = []
    for score, (x, y, bw, bh) in zip(scores, boxes):
        if score < low_threshold and median > 45.0:
            confidence = _score_to_confidence(median - score, median * 0.35, median)
            issues.append(
                Issue(
                    "localized_blur_anomaly",
                    max(confidence, 0.50),
                    x,
                    y,
                    bw,
                    bh,
                    f"Sharpness score {score:.2f} is much lower than median document sharpness {median:.2f}.",
                )
            )
    return _merge_similar(issues)


def _merge_similar(issues: list[Issue]) -> list[Issue]:
    if not issues:
        return []

    grouped: dict[str, list[Issue]] = {}
    for issue in issues:
        grouped.setdefault(issue.issue_type, []).append(issue)

    merged: list[Issue] = []
    for issue_type, group in grouped.items():
        group = sorted(group, key=lambda i: i.confidence, reverse=True)[:8]
        for issue in group:
            merged.append(issue)
    return sorted(merged, key=lambda i: i.confidence, reverse=True)


def detect_tampering(image_path: Path, model_path: Path | None = None, model_threshold: float = 0.65) -> list[Issue]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image for detection: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    issues = []
    issues.extend(detect_edge_inconsistency(gray))
    issues.extend(detect_noise_variance(gray))
    issues.extend(detect_blur_anomaly(gray))
    if model_path is not None:
        from .ml_model import detect_with_trained_model

        model_issue = detect_with_trained_model(image_path, model_path, model_threshold)
        if model_issue is not None:
            issues.append(model_issue)
    return sorted(issues, key=lambda i: i.confidence, reverse=True)[:18]


def draw_highlights(image_path: Path, issues: list[Issue], output_path: Path) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image for highlight drawing: {image_path}")

    palette = {
        "edge_inconsistency": (0, 84, 255),
        "noise_variance_anomaly": (0, 180, 255),
        "localized_blur_anomaly": (255, 72, 72),
    }
    for issue in issues:
        color = palette.get(issue.issue_type, (0, 255, 0))
        x, y, w, h = issue.x, issue.y, issue.width, issue.height
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        label = f"{issue.issue_type.replace('_', ' ')} {issue.confidence:.2f}"
        cv2.putText(image, label, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    cv2.imwrite(str(output_path), image)
    return output_path
