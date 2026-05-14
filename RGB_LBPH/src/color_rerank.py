from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".ppm"}


@dataclass(frozen=True)
class ColorRerankConfig:
    size: tuple[int, int] = (400, 450)
    detect_face: bool = False
    equalization: str = "clahe"
    margin_ratio: float = 0.15
    min_face_size: tuple[int, int] = (40, 40)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    fallback_to_full_image: bool = True
    input_adapter: str = "score2026_framework"
    radius: int = 2
    neighbors: int = 8
    grid_x: int = 10
    grid_y: int = 11
    color_bins: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ColorRerankConfig":
        data = data or {}
        return cls(
            size=_parse_pair(data.get("size", cls.size), cls.size),
            detect_face=_parse_bool(data.get("detect_face", cls.detect_face)),
            equalization=str(data.get("equalization", cls.equalization)),
            margin_ratio=float(data.get("margin_ratio", cls.margin_ratio)),
            min_face_size=_parse_pair(data.get("min_face_size", cls.min_face_size), cls.min_face_size),
            scale_factor=float(data.get("scale_factor", cls.scale_factor)),
            min_neighbors=int(data.get("min_neighbors", cls.min_neighbors)),
            fallback_to_full_image=_parse_bool(data.get("fallback_to_full_image", cls.fallback_to_full_image)),
            input_adapter=str(data.get("input_adapter", cls.input_adapter)),
            radius=int(data.get("radius", cls.radius)),
            neighbors=int(data.get("neighbors", cls.neighbors)),
            grid_x=int(data.get("grid_x", cls.grid_x)),
            grid_y=int(data.get("grid_y", cls.grid_y)),
            color_bins=int(data.get("color_bins", cls.color_bins)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": [int(self.size[0]), int(self.size[1])],
            "detect_face": bool(self.detect_face),
            "equalization": self.equalization,
            "margin_ratio": float(self.margin_ratio),
            "min_face_size": [int(self.min_face_size[0]), int(self.min_face_size[1])],
            "scale_factor": float(self.scale_factor),
            "min_neighbors": int(self.min_neighbors),
            "fallback_to_full_image": bool(self.fallback_to_full_image),
            "input_adapter": self.input_adapter,
            "radius": int(self.radius),
            "neighbors": int(self.neighbors),
            "grid_x": int(self.grid_x),
            "grid_y": int(self.grid_y),
            "color_bins": int(self.color_bins),
        }


@dataclass(frozen=True)
class GrayPrediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class RerankPrediction:
    label: str
    original_label: str
    confidence: float
    reranked: bool
    color_label: str | None
    color_distance: float | None
    gray_label_color_distance: float | None


class ColorRerankLBPHModel:
    def __init__(
        self,
        labels: Iterable[str],
        sample_labels: np.ndarray,
        color_features: np.ndarray,
        *,
        config: ColorRerankConfig,
        recognizer: Any | None = None,
    ) -> None:
        self.labels = [str(label) for label in labels]
        self.sample_labels = np.asarray(sample_labels).astype(str)
        self.color_features = np.asarray(color_features, dtype=np.float32)
        self.config = config
        self.recognizer = recognizer
        if self.color_features.ndim != 2:
            raise ValueError("color_features must be a 2D array")
        if self.color_features.shape[0] != len(self.sample_labels):
            raise ValueError("sample_labels count must match color_features")

    @classmethod
    def load(cls, model_dir: str | Path) -> "ColorRerankLBPHModel":
        root = Path(model_dir)
        config = ColorRerankConfig.from_dict(
            json.loads((root / "rerank_config.json").read_text(encoding="utf-8"))
        )
        payload = np.load(root / "color_index.npz")
        recognizer = _create_recognizer(config)
        recognizer.read(str(root / "gray_model.xml"))
        return cls(
            labels=[str(label) for label in payload["labels"].tolist()],
            sample_labels=payload["sample_labels"],
            color_features=payload["color_features"],
            config=config,
            recognizer=recognizer,
        )

    def save(self, output_dir: str | Path) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "color_index.npz",
            labels=np.array(self.labels),
            sample_labels=self.sample_labels.astype(str),
            color_features=self.color_features.astype(np.float32, copy=False),
        )
        (root / "rerank_config.json").write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def predict_image(
        self,
        image_path: str | Path,
        *,
        confidence_gate: float,
        margin_ratio: float,
        top_k: int | None = None,
    ) -> RerankPrediction:
        if self.recognizer is None:
            raise RuntimeError("recognizer is required for image prediction")
        gray = preprocess_gray_for_lbph(image_path, self.config)
        gray_candidates = collect_gray_candidates(self.recognizer, gray, self.labels)
        label, raw_confidence = gray_candidates[0]
        candidate_labels = [label for label, _dist in gray_candidates[:top_k]] if top_k else None
        return self.rerank(
            GrayPrediction(label=label, confidence=float(raw_confidence)),
            extract_color_feature(image_path, self.config),
            confidence_gate=confidence_gate,
            margin_ratio=margin_ratio,
            candidate_labels=candidate_labels,
        )

    def rerank(
        self,
        gray_prediction: GrayPrediction,
        color_feature: np.ndarray,
        *,
        confidence_gate: float,
        margin_ratio: float,
        candidate_labels: Iterable[str] | None = None,
    ) -> RerankPrediction:
        original = str(gray_prediction.label)
        if float(gray_prediction.confidence) < float(confidence_gate):
            return RerankPrediction(
                label=original,
                original_label=original,
                confidence=float(gray_prediction.confidence),
                reranked=False,
                color_label=None,
                color_distance=None,
                gray_label_color_distance=None,
            )
        allowed_labels = _normalize_candidate_labels(candidate_labels, original)
        per_label = self._label_color_distances(color_feature, candidate_labels=allowed_labels)
        if not per_label:
            return RerankPrediction(
                label=original,
                original_label=original,
                confidence=float(gray_prediction.confidence),
                reranked=False,
                color_label=None,
                color_distance=None,
                gray_label_color_distance=None,
            )
        color_label, color_distance = min(per_label.items(), key=lambda item: item[1])
        original_distance = per_label.get(original)
        should_switch = (
            color_label != original
            and original_distance is not None
            and color_distance <= original_distance * (1.0 - float(margin_ratio))
        )
        return RerankPrediction(
            label=color_label if should_switch else original,
            original_label=original,
            confidence=float(gray_prediction.confidence),
            reranked=bool(should_switch),
            color_label=color_label,
            color_distance=float(color_distance),
            gray_label_color_distance=float(original_distance) if original_distance is not None else None,
        )

    def _label_color_distances(
        self,
        color_feature: np.ndarray,
        *,
        candidate_labels: Iterable[str] | None = None,
    ) -> dict[str, float]:
        distances = chi_square_many(self.color_features, np.asarray(color_feature, dtype=np.float32))
        per_label: dict[str, float] = {}
        labels = (
            _dedupe_labels(candidate_labels)
            if candidate_labels is not None
            else [str(label) for label in np.unique(self.sample_labels)]
        )
        for label in labels:
            values = distances[self.sample_labels == label]
            if values.size:
                per_label[str(label)] = float(values.min())
        return per_label


def train_directory(
    *,
    train_dir: str | Path,
    output_dir: str | Path,
    config: ColorRerankConfig,
) -> dict[str, Any]:
    train_root = Path(train_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    samples = _collect_samples(train_root)
    if not samples:
        raise ValueError(f"no training samples found: {train_root}")
    labels = sorted({label for label, _path in samples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    faces = []
    numeric_labels = []
    sample_labels = []
    color_features = []
    valid_per_identity = defaultdict(int)
    for label, image_path in samples:
        faces.append(preprocess_gray_for_lbph(image_path, config))
        numeric_labels.append(label_to_id[label])
        sample_labels.append(label)
        color_features.append(extract_color_feature(image_path, config))
        valid_per_identity[label] += 1
    recognizer = _create_recognizer(config)
    recognizer.train(faces, np.array(numeric_labels, dtype=np.int32))
    recognizer.write(str(output_root / "gray_model.xml"))
    model = ColorRerankLBPHModel(
        labels=labels,
        sample_labels=np.array(sample_labels),
        color_features=np.stack(color_features).astype(np.float32),
        config=config,
        recognizer=recognizer,
    )
    model.save(output_root)
    mapping = {
        "name_to_id": {label: int(index) for label, index in label_to_id.items()},
        "id_to_name": {str(index): label for label, index in label_to_id.items()},
    }
    (output_root / "label_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "GRAY-LBPH with color rerank",
        "num_samples": len(samples),
        "num_valid_faces": len(samples),
        "num_identities": len(labels),
        "valid_per_identity": dict(sorted(valid_per_identity.items())),
        "config": config.to_dict(),
        "artifacts": {
            "gray_model": str(output_root / "gray_model.xml"),
            "color_index": str(output_root / "color_index.npz"),
            "config": str(output_root / "rerank_config.json"),
            "label_mapping": str(output_root / "label_mapping.json"),
        },
    }
    (output_root / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def evaluate_directory(
    *,
    test_dir: str | Path,
    model_dir: str | Path,
    reports_dir: str | Path,
    confidence_gates: list[float] | None = None,
    margin_ratios: list[float] | None = None,
    candidate_top_ks: list[int] | None = None,
) -> dict[str, Any]:
    gates = confidence_gates or [float("inf"), 55.0, 56.0, 60.0, 62.0, 65.0, 70.0]
    margins = margin_ratios or [0.0, 0.02, 0.05, 0.1]
    top_ks = candidate_top_ks or [2, 3]
    model = ColorRerankLBPHModel.load(model_dir)
    test_root = Path(test_dir)
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    base_items = _predict_base_items(test_root, model)
    metrics_by_key = {}
    best_key = ""
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_num_reranked = 10**9
    best_rows: list[dict[str, Any]] = []
    for top_k in top_ks:
        for gate in gates:
            for margin in margins:
                rows = _rows_for_params(
                    base_items,
                    model,
                    confidence_gate=gate,
                    margin_ratio=margin,
                    top_k=top_k,
                )
                metrics = compute_metrics(rows)
                key = _param_key(top_k, gate, margin)
                metrics["params"] = {
                    "candidate_top_k": int(top_k),
                    "confidence_gate": _json_gate(gate),
                    "margin_ratio": float(margin),
                }
                metrics_by_key[key] = metrics
                _write_prediction_results(root / f"prediction_results_{key}.csv", rows)
                _write_error_cases(root / f"error_cases_{key}.csv", metrics["error_cases"])
                if _is_better(metrics, best_accuracy, best_macro_f1, best_num_reranked):
                    best_key = key
                    best_accuracy = float(metrics["overall_accuracy"])
                    best_macro_f1 = float(metrics["macro_f1"])
                    best_num_reranked = int(metrics["num_reranked"])
                    best_rows = rows
    payload = {"params": metrics_by_key, "best_key": best_key, "best": metrics_by_key[best_key]}
    _write_prediction_results(root / "prediction_results.csv", best_rows)
    _write_error_cases(root / "error_cases.csv", payload["best"]["error_cases"])
    _write_summary_csv(root / "rerank_param_summary.csv", metrics_by_key)
    (root / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({str(row["true_label"]) for row in rows})
    total = len(rows)
    correct = sum(1 for row in rows if row["predicted_label"] == row["true_label"])
    confusion: dict[str, dict[str, int]] = {label: defaultdict(int) for label in labels}
    error_cases = []
    for row in rows:
        true_label = str(row["true_label"])
        predicted = str(row["predicted_label"])
        confusion[true_label][predicted] += 1
        if predicted != true_label:
            error_cases.append(dict(row))
    precisions = []
    recalls = []
    f1_scores = []
    per_identity = {}
    for label in labels:
        tp = confusion[label].get(label, 0)
        actual = sum(confusion[label].values())
        predicted_count = sum(row.get(label, 0) for row in confusion.values())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
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
        "correct": correct,
        "wrong": len(error_cases),
        "num_reranked": sum(1 for row in rows if row.get("reranked")),
        "error_cases": error_cases,
        "per_identity_accuracy": per_identity,
        "confusion_matrix": {label: dict(values) for label, values in confusion.items()},
    }


def extract_color_feature(image: str | Path | np.ndarray, config: ColorRerankConfig) -> np.ndarray:
    rgb = read_rgb_image(image, input_adapter=config.input_adapter) if isinstance(image, (str, Path)) else image
    if rgb is None:
        raise ValueError(f"failed to read image: {image}")
    prepared = preprocess_rgb_for_color(np.asarray(rgb, dtype=np.uint8), config)
    features = []
    height, width = prepared.shape[:2]
    for gy in range(config.grid_y):
        y0 = gy * height // config.grid_y
        y1 = (gy + 1) * height // config.grid_y
        for gx in range(config.grid_x):
            x0 = gx * width // config.grid_x
            x1 = (gx + 1) * width // config.grid_x
            cell = prepared[y0:y1, x0:x1, :]
            for channel in range(3):
                values = cell[:, :, channel].astype(np.int16, copy=False).ravel()
                hist = np.bincount(
                    np.minimum(values * config.color_bins // 256, config.color_bins - 1),
                    minlength=config.color_bins,
                ).astype(np.float32)
                total = float(hist.sum())
                if total:
                    hist /= total
                features.append(hist)
    return np.concatenate(features).astype(np.float32, copy=False)


def preprocess_gray_for_lbph(image: str | Path, config: ColorRerankConfig) -> np.ndarray:
    rgb = read_rgb_image(image, input_adapter=config.input_adapter)
    if rgb is None:
        raise ValueError(f"failed to read image: {image}")
    prepared = preprocess_rgb_for_color(rgb, config)
    gray = rgb_to_gray(prepared)
    return equalize_channel(gray, config.equalization)


def preprocess_rgb_for_color(rgb: np.ndarray, config: ColorRerankConfig) -> np.ndarray:
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("expected RGB image")
    rgb = rgb[:, :, :3].astype(np.uint8, copy=False)
    if config.detect_face:
        gray = rgb_to_gray(rgb)
        faces = _detect_faces(gray, config)
        if faces:
            rect = max(faces, key=lambda item: int(item[2]) * int(item[3]))
            rgb = _crop_rgb_with_margin(rgb, rect, config.margin_ratio)
        elif not config.fallback_to_full_image:
            raise ValueError("face_not_found")
    return resize_rgb(rgb, config.size)


def read_rgb_image(image: str | Path | np.ndarray, *, input_adapter: str = "score2026_framework") -> np.ndarray | None:
    if not isinstance(image, (str, Path)):
        return np.asarray(image, dtype=np.uint8)
    path = Path(image)
    try:
        import cv2

        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except ModuleNotFoundError:
        pass
    try:
        from PIL import Image

        with Image.open(path) as loaded:
            return np.array(loaded.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    except ModuleNotFoundError:
        values = (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        )
        return np.clip(values, 0, 255).astype(np.uint8)


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(rgb, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        from PIL import Image

        return np.array(Image.fromarray(rgb).resize((int(size[0]), int(size[1]))), dtype=np.uint8)


def equalize_channel(channel: np.ndarray, equalization: str) -> np.ndarray:
    method = (equalization or "none").lower()
    if method in {"none", "off", "false"}:
        return channel.astype(np.uint8, copy=False)
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for equalization") from exc
    if method in {"hist", "equalizehist", "equalize_hist"}:
        return cv2.equalizeHist(channel.astype(np.uint8, copy=False))
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(channel.astype(np.uint8, copy=False))
    raise ValueError(f"unsupported equalization method: {equalization}")


def chi_square_many(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    numerator = (features - target) ** 2
    denominator = features + target + 1e-12
    return 0.5 * np.sum(numerator / denominator, axis=1)


def _predict_base_items(test_dir: Path, model: ColorRerankLBPHModel) -> list[dict[str, Any]]:
    items = []
    for identity_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        for image_path in _iter_images(identity_dir):
            gray = preprocess_gray_for_lbph(image_path, model.config)
            gray_candidates = collect_gray_candidates(model.recognizer, gray, model.labels)
            label, raw_confidence = gray_candidates[0]
            items.append(
                {
                    "image_path": str(image_path),
                    "true_label": identity_dir.name,
                    "gray_prediction": GrayPrediction(
                        label=label,
                        confidence=float(raw_confidence),
                    ),
                    "gray_candidates": gray_candidates,
                    "color_feature": extract_color_feature(image_path, model.config),
                }
            )
    return items


def _rows_for_params(
    base_items: list[dict[str, Any]],
    model: ColorRerankLBPHModel,
    *,
    confidence_gate: float,
    margin_ratio: float,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for item in base_items:
        candidate_labels = [label for label, _dist in item["gray_candidates"][:top_k]]
        prediction = model.rerank(
            item["gray_prediction"],
            item["color_feature"],
            confidence_gate=confidence_gate,
            margin_ratio=margin_ratio,
            candidate_labels=candidate_labels,
        )
        rows.append(
            {
                "image_path": item["image_path"],
                "true_label": item["true_label"],
                "predicted_label": prediction.label,
                "gray_label": prediction.original_label,
                "confidence": prediction.confidence,
                "reranked": prediction.reranked,
                "color_label": prediction.color_label,
                "color_distance": prediction.color_distance,
                "gray_label_color_distance": prediction.gray_label_color_distance,
                "candidate_top_k": top_k,
                "confidence_gate": _json_gate(confidence_gate),
                "margin_ratio": margin_ratio,
            }
        )
    return rows


def _collect_samples(root: Path) -> list[tuple[str, Path]]:
    rows = []
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in _iter_images(identity_dir):
            rows.append((identity_dir.name, image_path))
    return rows


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _create_recognizer(config: ColorRerankConfig) -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required; install opencv-contrib-python") from exc
    try:
        return cv2.face.LBPHFaceRecognizer_create(
            radius=int(config.radius),
            neighbors=int(config.neighbors),
            grid_x=int(config.grid_x),
            grid_y=int(config.grid_y),
        )
    except AttributeError as exc:
        raise ModuleNotFoundError("OpenCV LBPH requires opencv-contrib-python") from exc


def _detect_faces(gray: np.ndarray, config: ColorRerankConfig) -> list[tuple[int, int, int, int]]:
    try:
        import cv2
    except ModuleNotFoundError:
        return []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return []
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=config.scale_factor,
        minNeighbors=config.min_neighbors,
        minSize=(int(config.min_face_size[0]), int(config.min_face_size[1])),
    )
    return [tuple(map(int, rect)) for rect in faces]


def _crop_rgb_with_margin(
    rgb: np.ndarray, rect: tuple[int, int, int, int], margin_ratio: float
) -> np.ndarray:
    x, y, width, height = rect
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(rgb.shape[1], x + width + margin_x)
    bottom = min(rgb.shape[0], y + height + margin_y)
    return rgb[top:bottom, left:right, :]


def _write_prediction_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image_path",
        "true_label",
        "predicted_label",
        "gray_label",
        "confidence",
        "reranked",
        "color_label",
        "color_distance",
        "gray_label_color_distance",
        "candidate_top_k",
        "confidence_gate",
        "margin_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_error_cases(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_prediction_results(path, rows)


def _write_summary_csv(path: Path, metrics_by_key: dict[str, Any]) -> None:
    fields = [
        "key",
        "candidate_top_k",
        "confidence_gate",
        "margin_ratio",
        "overall_accuracy",
        "macro_f1",
        "correct",
        "wrong",
        "num_reranked",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, metrics in metrics_by_key.items():
            params = metrics["params"]
            writer.writerow(
                {
                    "key": key,
                    "candidate_top_k": params["candidate_top_k"],
                    "confidence_gate": params["confidence_gate"],
                    "margin_ratio": params["margin_ratio"],
                    "overall_accuracy": metrics["overall_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "correct": metrics["correct"],
                    "wrong": metrics["wrong"],
                    "num_reranked": metrics["num_reranked"],
                }
            )


def collect_gray_candidates(recognizer: Any, gray: np.ndarray, labels: list[str]) -> list[tuple[str, float]]:
    if hasattr(recognizer, "predict_collect"):
        try:
            import cv2

            collector = cv2.face.StandardCollector_create(1e9)
            recognizer.predict_collect(gray, collector)
            by_label: dict[str, float] = {}
            for raw_label, distance in collector.getResults():
                label = labels[int(raw_label)]
                by_label[label] = min(by_label.get(label, float("inf")), float(distance))
            if by_label:
                return sorted(by_label.items(), key=lambda item: item[1])
        except Exception:
            pass
    raw_label, distance = recognizer.predict(gray)
    return [(labels[int(raw_label)], float(distance))]


def _is_better(
    metrics: dict[str, Any],
    best_accuracy: float,
    best_macro_f1: float,
    best_num_reranked: int,
) -> bool:
    accuracy = float(metrics["overall_accuracy"])
    macro_f1 = float(metrics["macro_f1"])
    num_reranked = int(metrics["num_reranked"])
    if accuracy > best_accuracy:
        return True
    if accuracy < best_accuracy:
        return False
    if macro_f1 > best_macro_f1:
        return True
    if macro_f1 < best_macro_f1:
        return False
    return num_reranked < best_num_reranked


def _param_key(candidate_top_k: int, confidence_gate: float, margin_ratio: float) -> str:
    top_k = f"topk_{int(candidate_top_k)}"
    gate = "inf" if math.isinf(confidence_gate) else f"{confidence_gate:g}".replace(".", "_")
    margin = f"{margin_ratio:g}".replace(".", "_")
    return f"{top_k}_gate_{gate}_margin_{margin}"


def _json_gate(confidence_gate: float) -> str | float:
    return "inf" if math.isinf(confidence_gate) else float(confidence_gate)


def _parse_pair(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, str):
        left, _, right = value.lower().partition("x")
        if left and right:
            return int(left), int(right)
        return int(fallback[0]), int(fallback[1])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return int(fallback[0]), int(fallback[1])


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _normalize_candidate_labels(candidate_labels: Iterable[str] | None, original_label: str) -> list[str] | None:
    if candidate_labels is None:
        return None
    labels = _dedupe_labels(candidate_labels)
    if original_label not in labels:
        labels.insert(0, original_label)
    return labels


def _dedupe_labels(labels: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for label in labels:
        value = str(label)
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
