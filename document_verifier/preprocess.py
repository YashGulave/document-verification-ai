from pathlib import Path

import cv2
import numpy as np


def preprocess_document_scan(image_path: Path, output_path: Path) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unsupported or unreadable document image: {image_path.name}")

    page = _perspective_correct(image)
    cleaned = _normalize_scan(page)
    cv2.imwrite(str(output_path), cleaned)
    return output_path


def _perspective_correct(image: np.ndarray) -> np.ndarray:
    ratio = image.shape[0] / 700.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 700), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_contour = None
    image_area = resized.shape[0] * resized.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > image_area * 0.18:
            page_contour = approx.reshape(4, 2).astype(np.float32)
            break

    if page_contour is None:
        return image

    points = page_contour * ratio
    ordered = _order_points(points)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    width = int(max(top_width, bottom_width))
    height = int(max(left_height, right_height))

    if width < 200 or height < 200:
        return image

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, transform, (width, height))


def _normalize_scan(image: np.ndarray) -> np.ndarray:
    if max(image.shape[:2]) > 1800:
        scale = 1800 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    balanced = cv2.merge((lightness, channel_a, channel_b))
    balanced = cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(gray, 31)
    normalized = cv2.divide(gray, background, scale=255)
    sharpened = cv2.addWeighted(normalized, 1.45, cv2.GaussianBlur(normalized, (0, 0), 1.2), -0.45, 0)
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 8, 7, 21)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(point_sum)]
    rect[2] = points[np.argmax(point_sum)]
    rect[1] = points[np.argmin(point_diff)]
    rect[3] = points[np.argmax(point_diff)]
    return rect
