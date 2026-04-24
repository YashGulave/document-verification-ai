import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from document_verifier.ml_features import extract_image_features


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Train a real document-tamper classifier from labeled images.")
    parser.add_argument("--dataset-dir", type=Path, help="Folder with authentic/ and tampered/ subfolders.")
    parser.add_argument("--manifest", type=Path, help="CSV with columns image_path,label where label is 0/1 or authentic/tampered.")
    parser.add_argument("--output", type=Path, default=Path("models/document_tamper_model.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = _load_samples(args.dataset_dir, args.manifest)
    if len(samples) < 10:
        raise SystemExit("Need at least 10 labeled images to train a useful prototype model.")

    features = []
    labels = []
    for image_path, label in samples:
        features.append(extract_image_features(image_path))
        labels.append(label)

    x = np.vstack(features)
    y = np.array(labels, dtype=np.int32)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=1,
        ),
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    print(classification_report(y_test, predictions, target_names=["authentic", "tampered"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved trained model to {args.output}")


def _load_samples(dataset_dir: Path | None, manifest: Path | None) -> list[tuple[Path, int]]:
    if manifest:
        return _load_manifest(manifest)
    if dataset_dir:
        return _load_directory_dataset(dataset_dir)
    raise SystemExit("Provide --dataset-dir or --manifest.")


def _load_manifest(manifest: Path) -> list[tuple[Path, int]]:
    samples = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (manifest.parent / image_path).resolve()
            samples.append((image_path, _parse_label(row["label"])))
    return samples


def _load_directory_dataset(dataset_dir: Path) -> list[tuple[Path, int]]:
    samples = []
    for folder_name, label in (("authentic", 0), ("tampered", 1), ("real", 0), ("fake", 1)):
        folder = dataset_dir / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    return samples


def _parse_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"1", "tampered", "fake", "forged", "fraud"}:
        return 1
    if normalized in {"0", "authentic", "real", "genuine", "clean"}:
        return 0
    raise ValueError(f"Unsupported label: {value}")


if __name__ == "__main__":
    main()
