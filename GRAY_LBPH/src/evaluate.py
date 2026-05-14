from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .dataset import SUPPORTED_IMAGE_SUFFIXES
from .predict import LBPHPredictor


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({str(item["true_label"]) for item in results})
    total = len(results)
    correct = sum(1 for item in results if item.get("predicted_label") == item.get("true_label"))
    confusion: dict[str, dict[str, int]] = {label: defaultdict(int) for label in labels}
    per_identity: dict[str, float] = {}
    error_cases: list[dict[str, Any]] = []
    preprocess_statuses = Counter(_preprocess_status(item) for item in results)

    for item in results:
        true_label = str(item["true_label"])
        predicted = item.get("predicted_label")
        predicted_key = str(predicted) if predicted is not None else "__none__"
        confusion.setdefault(true_label, defaultdict(int))[predicted_key] += 1
        if predicted != item.get("true_label"):
            error_cases.append(dict(item))

    precisions = []
    recalls = []
    f1_scores = []
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        actual = sum(confusion.get(label, {}).values())
        predicted_count = sum(row.get(label, 0) for row in confusion.values())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_identity[label] = recall

    return {
        "overall_accuracy": correct / total if total else 0.0,
        "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "num_test_images": total,
        "num_successful_predictions": sum(1 for item in results if item.get("status") == "ok"),
        "num_failed_preprocess": sum(
            1 for item in results if _preprocess_status(item) in {"read_failed", "face_not_found"}
        ),
        "num_non_ok_predictions": sum(1 for item in results if item.get("status") != "ok"),
        "preprocess_statuses": dict(sorted(preprocess_statuses.items())),
        "per_identity_accuracy": per_identity,
        "confusion_matrix": {key: dict(value) for key, value in confusion.items()},
        "error_cases": error_cases,
    }


def evaluate_directory(
    *,
    test_dir: str | Path,
    algorithm_dir: str | Path,
    reports_dir: str | Path,
    threshold: float | None = None,
) -> dict[str, Any]:
    test_root = Path(test_dir)
    predictor = LBPHPredictor(algorithm_dir=algorithm_dir, threshold=threshold)
    rows: list[dict[str, Any]] = []
    for identity_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        for image_path in sorted(
            path
            for path in identity_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ):
            prediction = predictor.predict_image(image_path)
            rows.append(
                {
                    "image_path": str(image_path),
                    "true_label": identity_dir.name,
                    "predicted_label": prediction.get("label"),
                    "confidence": prediction.get("confidence"),
                    "status": prediction.get("status"),
                    "preprocess_status": prediction.get("preprocess_status") or prediction.get("status"),
                    "face_detected": prediction.get("face_detected"),
                    "face_rect": prediction.get("face_rect"),
                }
            )
    metrics = compute_metrics(rows)
    write_reports(metrics, reports_dir, rows)
    return metrics


def write_reports(metrics: dict[str, Any], reports_dir: str | Path, rows: list[dict[str, Any]] | None = None) -> None:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_nested_csv(root / "confusion_matrix.csv", metrics.get("confusion_matrix", {}))
    _write_simple_csv(root / "per_identity_accuracy.csv", metrics.get("per_identity_accuracy", {}))
    _write_error_cases(root / "error_cases.csv", metrics.get("error_cases", []))
    if rows is not None:
        _write_prediction_results(root / "prediction_results.csv", rows)


def _write_nested_csv(path: Path, matrix: dict[str, dict[str, int]]) -> None:
    columns = sorted({column for row in matrix.values() for column in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *columns])
        for label in sorted(matrix):
            writer.writerow([label, *[matrix[label].get(column, 0) for column in columns]])


def _write_simple_csv(path: Path, values: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["identity", "accuracy"])
        for key in sorted(values):
            writer.writerow([key, values[key]])


def _write_error_cases(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        "status",
        "preprocess_status",
        "face_detected",
        "face_rect",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_prediction_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        "status",
        "preprocess_status",
        "face_detected",
        "face_rect",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _preprocess_status(item: dict[str, Any]) -> str:
    return str(item.get("preprocess_status") or item.get("status") or "")
