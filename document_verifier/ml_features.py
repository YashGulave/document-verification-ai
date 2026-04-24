from pathlib import Path

import cv2
import numpy as np


def extract_image_features(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image for ML features: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(gray)

    edges = cv2.Canny(equalized, 80, 180)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    residual = cv2.absdiff(equalized, blurred)
    laplacian = cv2.Laplacian(equalized, cv2.CV_64F)

    features: list[float] = []
    features.extend(_stats(equalized))
    features.extend(_stats(edges.astype(np.float32)))
    features.extend(_stats(residual.astype(np.float32)))
    features.extend(_stats(np.abs(laplacian).astype(np.float32)))
    features.extend(_tile_stats(edges, residual, laplacian))
    features.extend(_frequency_features(equalized))
    return np.array(features, dtype=np.float32)


def _stats(values: np.ndarray) -> list[float]:
    flat = values.astype(np.float32).ravel()
    q10, q25, q50, q75, q90 = np.percentile(flat, [10, 25, 50, 75, 90])
    return [
        float(np.mean(flat)),
        float(np.std(flat)),
        float(q10),
        float(q25),
        float(q50),
        float(q75),
        float(q90),
        float(np.max(flat) - np.min(flat)),
    ]


def _tile_stats(edges: np.ndarray, residual: np.ndarray, laplacian: np.ndarray, tile: int = 64) -> list[float]:
    edge_scores = []
    noise_scores = []
    blur_scores = []
    height, width = edges.shape
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            edge_roi = edges[y : y + tile, x : x + tile]
            residual_roi = residual[y : y + tile, x : x + tile]
            laplacian_roi = laplacian[y : y + tile, x : x + tile]
            edge_scores.append(float(np.mean(edge_roi > 0)))
            noise_scores.append(float(np.var(residual_roi)))
            blur_scores.append(float(np.var(laplacian_roi)))

    features = []
    for scores in (edge_scores, noise_scores, blur_scores):
        arr = np.array(scores, dtype=np.float32)
        features.extend(_stats(arr))
        features.append(float(np.max(arr) - np.median(arr)))
        features.append(float(np.median(arr) - np.min(arr)))
    return features


def _frequency_features(gray: np.ndarray) -> list[float]:
    dct = cv2.dct(gray.astype(np.float32) / 255.0)
    low = dct[:32, :32]
    mid = dct[32:128, 32:128]
    high = dct[128:, 128:]
    return [
        float(np.mean(np.abs(low))),
        float(np.mean(np.abs(mid))),
        float(np.mean(np.abs(high))),
        float(np.std(np.abs(mid))),
        float(np.std(np.abs(high))),
    ]
