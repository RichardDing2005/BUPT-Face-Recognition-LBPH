from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dataset import SUPPORTED_IMAGE_SUFFIXES
from .preprocess import PreprocessConfig, preprocess_image
from .train import build_label_mapping


@dataclass(frozen=True)
class ConfusionRerankConfig:
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
    aux_size: tuple[int, int] = (200, 200)
    aux_radius: int = 2
    aux_neighbors: int = 8
    aux_grid_x: int = 7
    aux_grid_y: int = 7
    color_bins: int = 8
    texture_bins: int = 8
    texture_grid_x: int = 5
    texture_grid_y: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ConfusionRerankConfig":
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
            aux_size=_parse_pair(data.get("aux_size", cls.aux_size), cls.aux_size),
            aux_radius=int(data.get("aux_radius", cls.aux_radius)),
            aux_neighbors=int(data.get("aux_neighbors", cls.aux_neighbors)),
            aux_grid_x=int(data.get("aux_grid_x", cls.aux_grid_x)),
            aux_grid_y=int(data.get("aux_grid_y", cls.aux_grid_y)),
            color_bins=int(data.get("color_bins", cls.color_bins)),
            texture_bins=int(data.get("texture_bins", cls.texture_bins)),
            texture_grid_x=int(data.get("texture_grid_x", cls.texture_grid_x)),
            texture_grid_y=int(data.get("texture_grid_y", cls.texture_grid_y)),
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
            "aux_size": [int(self.aux_size[0]), int(self.aux_size[1])],
            "aux_radius": int(self.aux_radius),
            "aux_neighbors": int(self.aux_neighbors),
            "aux_grid_x": int(self.aux_grid_x),
            "aux_grid_y": int(self.aux_grid_y),
            "color_bins": int(self.color_bins),
            "texture_bins": int(self.texture_bins),
            "texture_grid_x": int(self.texture_grid_x),
            "texture_grid_y": int(self.texture_grid_y),
        }

    def primary_preprocess_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            size=self.size,
            detect_face=self.detect_face,
            equalization=self.equalization,
            margin_ratio=self.margin_ratio,
            min_face_size=self.min_face_size,
            scale_factor=self.scale_factor,
            min_neighbors=self.min_neighbors,
            fallback_to_full_image=self.fallback_to_full_image,
            input_adapter=self.input_adapter,
        )

    def aux_preprocess_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            size=self.aux_size,
            detect_face=self.detect_face,
            equalization=self.equalization,
            margin_ratio=self.margin_ratio,
            min_face_size=self.min_face_size,
            scale_factor=self.scale_factor,
            min_neighbors=self.min_neighbors,
            fallback_to_full_image=self.fallback_to_full_image,
            input_adapter=self.input_adapter,
        )


@dataclass(frozen=True)
class RerankRuntimeConfig:
    candidate_top_k: int = 4
    confidence_gate: float = 60.0
    gray_margin_gate: float = 65.0
    switch_margin: float = 0.05
    primary_weight: float = 0.15
    aux_weight: float = 0.35
    color_weight: float = 0.30
    texture_weight: float = 0.20
    color_reliability_floor: float = 0.35

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RerankRuntimeConfig":
        data = data or {}
        return cls(
            candidate_top_k=int(data.get("candidate_top_k", cls.candidate_top_k)),
            confidence_gate=float(data.get("confidence_gate", cls.confidence_gate)),
            gray_margin_gate=float(data.get("gray_margin_gate", cls.gray_margin_gate)),
            switch_margin=float(data.get("switch_margin", cls.switch_margin)),
            primary_weight=float(data.get("primary_weight", cls.primary_weight)),
            aux_weight=float(data.get("aux_weight", cls.aux_weight)),
            color_weight=float(data.get("color_weight", cls.color_weight)),
            texture_weight=float(data.get("texture_weight", cls.texture_weight)),
            color_reliability_floor=float(data.get("color_reliability_floor", cls.color_reliability_floor)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_top_k": int(self.candidate_top_k),
            "confidence_gate": float(self.confidence_gate),
            "gray_margin_gate": float(self.gray_margin_gate),
            "switch_margin": float(self.switch_margin),
            "primary_weight": float(self.primary_weight),
            "aux_weight": float(self.aux_weight),
            "color_weight": float(self.color_weight),
            "texture_weight": float(self.texture_weight),
            "color_reliability_floor": float(self.color_reliability_floor),
        }


@dataclass(frozen=True)
class GrayCandidate:
    label: str
    confidence: float


@dataclass(frozen=True)
class ConfusionEvidenceFeature:
    color: np.ndarray
    texture: np.ndarray
    quality: dict[str, float]


@dataclass(frozen=True)
class RerankPrediction:
    label: str
    original_label: str
    confidence: float
    reranked: bool
    trigger_reason: str
    candidate_scores: dict[str, float]
    secondary_label: str | None
    original_score: float | None
    secondary_score: float | None


class ConfusionRerankLBPHModel:
    def __init__(
        self,
        labels: Iterable[str],
        sample_labels: np.ndarray,
        color_features: np.ndarray,
        texture_features: np.ndarray,
        quality_features: np.ndarray,
        *,
        config: ConfusionRerankConfig,
        recognizer: Any | None = None,
        aux_recognizer: Any | None = None,
    ) -> None:
        self.labels = [str(label) for label in labels]
        self.sample_labels = np.asarray(sample_labels).astype(str)
        self.color_features = np.asarray(color_features, dtype=np.float32)
        self.texture_features = np.asarray(texture_features, dtype=np.float32)
        self.quality_features = np.asarray(quality_features, dtype=np.float32)
        self.config = config
        self.recognizer = recognizer
        self.aux_recognizer = aux_recognizer
        self._validate_index()

    @classmethod
    def load(cls, model_dir: str | Path) -> "ConfusionRerankLBPHModel":
        root = Path(model_dir)
        config = ConfusionRerankConfig.from_dict(_read_json(root / "rerank_config.json"))
        payload = np.load(root / "evidence_index.npz")
        recognizer = _create_recognizer(config.radius, config.neighbors, config.grid_x, config.grid_y)
        recognizer.read(str(root / "gray_model.xml"))
        aux_recognizer = _create_recognizer(
            config.aux_radius,
            config.aux_neighbors,
            config.aux_grid_x,
            config.aux_grid_y,
        )
        aux_recognizer.read(str(root / "gray_aux_model.xml"))
        return cls(
            labels=[str(label) for label in payload["labels"].tolist()],
            sample_labels=payload["sample_labels"],
            color_features=payload["color_features"],
            texture_features=payload["texture_features"],
            quality_features=payload["quality_features"],
            config=config,
            recognizer=recognizer,
            aux_recognizer=aux_recognizer,
        )

    def save(self, output_dir: str | Path) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "evidence_index.npz",
            labels=np.array(self.labels),
            sample_labels=self.sample_labels.astype(str),
            color_features=self.color_features.astype(np.float32, copy=False),
            texture_features=self.texture_features.astype(np.float32, copy=False),
            quality_features=self.quality_features.astype(np.float32, copy=False),
        )
        _write_json(root / "rerank_config.json", self.config.to_dict())

    def predict_image(
        self,
        image_path: str | Path,
        *,
        runtime_config: RerankRuntimeConfig | None = None,
    ) -> dict[str, Any]:
        if self.recognizer is None or self.aux_recognizer is None:
            raise RuntimeError("recognizers are required for image prediction")
        primary = preprocess_image(image_path, self.config.primary_preprocess_config())
        if primary.status == "read_failed":
            return {"label": None, "status": "read_failed", "confidence": None}
        aux = preprocess_image(image_path, self.config.aux_preprocess_config())
        evidence = extract_evidence_feature(image_path, self.config)
        gray_candidates = collect_gray_candidates(self.recognizer, primary.face, self.labels)
        aux_candidates = collect_gray_candidates(self.aux_recognizer, aux.face, self.labels)
        prediction = self.rerank(
            gray_candidates=gray_candidates,
            evidence_feature=evidence,
            aux_candidates=aux_candidates,
            runtime_config=runtime_config,
        )
        return {
            "label": prediction.label,
            "status": "ok",
            "confidence": prediction.confidence,
            "gray_label": prediction.original_label,
            "reranked": prediction.reranked,
            "trigger_reason": prediction.trigger_reason,
            "secondary_label": prediction.secondary_label,
            "original_score": prediction.original_score,
            "secondary_score": prediction.secondary_score,
        }

    def rerank(
        self,
        *,
        gray_candidates: Iterable[GrayCandidate | tuple[str, float]],
        evidence_feature: ConfusionEvidenceFeature,
        aux_candidates: Iterable[GrayCandidate | tuple[str, float]] | None = None,
        runtime_config: RerankRuntimeConfig | None = None,
    ) -> RerankPrediction:
        runtime = runtime_config or RerankRuntimeConfig()
        candidates = _normalize_candidates(gray_candidates)
        if not candidates:
            return RerankPrediction(
                label="unknown",
                original_label="unknown",
                confidence=math.inf,
                reranked=False,
                trigger_reason="no_gray_candidates",
                candidate_scores={},
                secondary_label=None,
                original_score=None,
                secondary_score=None,
            )
        original = candidates[0].label
        confidence = float(candidates[0].confidence)
        trigger_reason = _trigger_reason(candidates, runtime)
        if trigger_reason == "not_triggered":
            return RerankPrediction(
                label=original,
                original_label=original,
                confidence=confidence,
                reranked=False,
                trigger_reason=trigger_reason,
                candidate_scores={},
                secondary_label=None,
                original_score=None,
                secondary_score=None,
            )
        candidate_labels = _dedupe_labels([candidate.label for candidate in candidates[: runtime.candidate_top_k]])
        if original not in candidate_labels:
            candidate_labels.insert(0, original)
        scores = self._candidate_scores(
            candidate_labels,
            candidates,
            evidence_feature,
            _normalize_candidates(aux_candidates or []),
            runtime,
        )
        if not scores or original not in scores:
            return RerankPrediction(
                label=original,
                original_label=original,
                confidence=confidence,
                reranked=False,
                trigger_reason="no_secondary_scores",
                candidate_scores=scores,
                secondary_label=None,
                original_score=scores.get(original),
                secondary_score=None,
            )
        secondary_label, secondary_score = min(scores.items(), key=lambda item: item[1])
        original_score = float(scores[original])
        should_switch = (
            secondary_label != original
            and float(secondary_score) <= original_score * (1.0 - float(runtime.switch_margin))
        )
        return RerankPrediction(
            label=secondary_label if should_switch else original,
            original_label=original,
            confidence=confidence,
            reranked=bool(should_switch),
            trigger_reason=trigger_reason,
            candidate_scores={key: float(value) for key, value in scores.items()},
            secondary_label=secondary_label,
            original_score=original_score,
            secondary_score=float(secondary_score),
        )

    def _candidate_scores(
        self,
        candidate_labels: list[str],
        gray_candidates: list[GrayCandidate],
        evidence_feature: ConfusionEvidenceFeature,
        aux_candidates: list[GrayCandidate],
        runtime: RerankRuntimeConfig,
    ) -> dict[str, float]:
        primary_raw = {candidate.label: candidate.confidence for candidate in gray_candidates}
        aux_raw = {candidate.label: candidate.confidence for candidate in aux_candidates}
        color_raw = self._label_feature_distances(
            self.color_features,
            evidence_feature.color,
            candidate_labels=candidate_labels,
        )
        texture_raw = self._label_feature_distances(
            self.texture_features,
            evidence_feature.texture,
            candidate_labels=candidate_labels,
        )
        primary = _normalize_distance_map(primary_raw, candidate_labels)
        aux = _normalize_distance_map(aux_raw, candidate_labels)
        color = _normalize_distance_map(color_raw, candidate_labels)
        texture = _normalize_distance_map(texture_raw, candidate_labels)

        reliability = float(evidence_feature.quality.get("color_reliability", 1.0))
        reliability = min(1.0, max(float(runtime.color_reliability_floor), reliability))
        primary_weight = max(0.0, float(runtime.primary_weight))
        aux_weight = max(0.0, float(runtime.aux_weight))
        color_weight = max(0.0, float(runtime.color_weight)) * reliability
        texture_weight = max(0.0, float(runtime.texture_weight)) + max(0.0, float(runtime.color_weight)) * (1.0 - reliability)
        total = primary_weight + aux_weight + color_weight + texture_weight
        if total <= 0:
            total = 1.0
        return {
            label: (
                primary[label] * primary_weight
                + aux[label] * aux_weight
                + color[label] * color_weight
                + texture[label] * texture_weight
            )
            / total
            for label in candidate_labels
        }

    def _label_feature_distances(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        *,
        candidate_labels: Iterable[str],
    ) -> dict[str, float]:
        per_label: dict[str, float] = {}
        target_array = np.asarray(target, dtype=np.float32)
        for label in candidate_labels:
            subset = matrix[self.sample_labels == str(label)]
            if subset.size:
                per_label[str(label)] = float(chi_square_many(subset, target_array).min())
        return per_label

    def _validate_index(self) -> None:
        if self.color_features.ndim != 2:
            raise ValueError("color_features must be a 2D array")
        if self.texture_features.ndim != 2:
            raise ValueError("texture_features must be a 2D array")
        if self.quality_features.ndim != 2:
            raise ValueError("quality_features must be a 2D array")
        sample_count = len(self.sample_labels)
        if self.color_features.shape[0] != sample_count:
            raise ValueError("sample_labels count must match color_features")
        if self.texture_features.shape[0] != sample_count:
            raise ValueError("sample_labels count must match texture_features")
        if self.quality_features.shape[0] != sample_count:
            raise ValueError("sample_labels count must match quality_features")


def train_directory(
    *,
    train_dir: str | Path,
    output_dir: str | Path,
    config: ConfusionRerankConfig,
) -> dict[str, Any]:
    train_root = Path(train_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    samples = _collect_samples(train_root)
    if not samples:
        raise ValueError(f"no training samples found: {train_root}")

    labels = sorted({label for label, _path in samples})
    mapping = build_label_mapping(labels)
    label_to_id = {label: int(index) for label, index in mapping["name_to_id"].items()}
    primary_faces = []
    aux_faces = []
    numeric_labels = []
    sample_labels = []
    color_features = []
    texture_features = []
    quality_features = []
    statuses: Counter[str] = Counter()
    valid_per_identity: Counter[str] = Counter()
    for label, image_path in samples:
        primary = preprocess_image(image_path, config.primary_preprocess_config())
        statuses[primary.status] += 1
        if primary.status == "read_failed":
            continue
        aux = preprocess_image(image_path, config.aux_preprocess_config())
        evidence = extract_evidence_feature(image_path, config)
        primary_faces.append(primary.face)
        aux_faces.append(aux.face)
        numeric_labels.append(label_to_id[label])
        sample_labels.append(label)
        color_features.append(evidence.color)
        texture_features.append(evidence.texture)
        quality_features.append(_quality_vector(evidence.quality))
        valid_per_identity[label] += 1

    if not primary_faces:
        raise ValueError("no valid training faces found")

    recognizer = _create_recognizer(config.radius, config.neighbors, config.grid_x, config.grid_y)
    recognizer.train(primary_faces, np.array(numeric_labels, dtype=np.int32))
    recognizer.write(str(output_root / "gray_model.xml"))
    aux_recognizer = _create_recognizer(
        config.aux_radius,
        config.aux_neighbors,
        config.aux_grid_x,
        config.aux_grid_y,
    )
    aux_recognizer.train(aux_faces, np.array(numeric_labels, dtype=np.int32))
    aux_recognizer.write(str(output_root / "gray_aux_model.xml"))

    model = ConfusionRerankLBPHModel(
        labels=labels,
        sample_labels=np.array(sample_labels),
        color_features=np.stack(color_features).astype(np.float32),
        texture_features=np.stack(texture_features).astype(np.float32),
        quality_features=np.stack(quality_features).astype(np.float32),
        config=config,
        recognizer=recognizer,
        aux_recognizer=aux_recognizer,
    )
    model.save(output_root)
    _write_json(output_root / "label_mapping.json", mapping)
    _write_json(output_root / "preprocess_config.json", config.primary_preprocess_config().to_dict())
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "CA-ME-LBPH",
        "num_samples": len(samples),
        "num_valid_faces": len(primary_faces),
        "num_identities": len(labels),
        "valid_per_identity": dict(sorted(valid_per_identity.items())),
        "preprocess_statuses": dict(sorted(statuses.items())),
        "config": config.to_dict(),
        "runtime_defaults": RerankRuntimeConfig().to_dict(),
        "artifacts": {
            "gray_model": str(output_root / "gray_model.xml"),
            "gray_aux_model": str(output_root / "gray_aux_model.xml"),
            "evidence_index": str(output_root / "evidence_index.npz"),
            "label_mapping": str(output_root / "label_mapping.json"),
            "config": str(output_root / "rerank_config.json"),
        },
    }
    _write_json(output_root / "training_report.json", report)
    return report


def evaluate_directory(
    *,
    test_dir: str | Path,
    model_dir: str | Path,
    reports_dir: str | Path,
    runtime_config: RerankRuntimeConfig | None = None,
) -> dict[str, Any]:
    model = ConfusionRerankLBPHModel.load(model_dir)
    rows = []
    for identity_dir in sorted(path for path in Path(test_dir).iterdir() if path.is_dir()):
        for image_path in _iter_images(identity_dir):
            prediction = model.predict_image(image_path, runtime_config=runtime_config)
            rows.append(
                {
                    "image_path": str(image_path),
                    "true_label": identity_dir.name,
                    "predicted_label": prediction.get("label"),
                    "gray_label": prediction.get("gray_label"),
                    "confidence": prediction.get("confidence"),
                    "status": prediction.get("status"),
                    "reranked": prediction.get("reranked", False),
                    "trigger_reason": prediction.get("trigger_reason"),
                    "secondary_label": prediction.get("secondary_label"),
                    "original_score": prediction.get("original_score"),
                    "secondary_score": prediction.get("secondary_score"),
                }
            )
    metrics = compute_metrics(rows)
    write_reports(metrics, reports_dir, rows)
    return metrics


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({str(row["true_label"]) for row in rows})
    total = len(rows)
    correct = sum(1 for row in rows if row.get("predicted_label") == row.get("true_label"))
    confusion: dict[str, dict[str, int]] = {label: defaultdict(int) for label in labels}
    error_cases = []
    for row in rows:
        true_label = str(row["true_label"])
        predicted = str(row.get("predicted_label"))
        confusion[true_label][predicted] += 1
        if row.get("predicted_label") != row.get("true_label"):
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
    effect = count_rerank_effect(rows)
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
        **effect,
    }


def count_rerank_effect(rows: list[dict[str, Any]]) -> dict[str, int]:
    help_count = 0
    harm_count = 0
    for row in rows:
        true_label = str(row.get("true_label"))
        gray_label = str(row.get("gray_label"))
        predicted = str(row.get("predicted_label"))
        gray_correct = gray_label == true_label
        rerank_correct = predicted == true_label
        if not gray_correct and rerank_correct:
            help_count += 1
        elif gray_correct and not rerank_correct:
            harm_count += 1
    return {"rerank_help": help_count, "rerank_harm": harm_count, "net_gain": help_count - harm_count}


def write_reports(metrics: dict[str, Any], reports_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "metrics.json", metrics)
    _write_prediction_results(root / "prediction_results.csv", rows)
    _write_prediction_results(root / "error_cases.csv", metrics.get("error_cases", []))


def extract_evidence_feature(image: str | Path | np.ndarray, config: ConfusionRerankConfig) -> ConfusionEvidenceFeature:
    rgb = read_rgb_image(image, input_adapter=config.input_adapter) if isinstance(image, (str, Path)) else image
    if rgb is None:
        raise ValueError(f"failed to read image: {image}")
    prepared = preprocess_rgb_for_evidence(np.asarray(rgb, dtype=np.uint8), config, size=config.size)
    gray = rgb_to_gray(prepared)
    quality = compute_quality(gray)
    color = extract_color_feature_from_rgb(prepared, config)
    texture = extract_texture_feature_from_gray(gray, config)
    return ConfusionEvidenceFeature(color=color, texture=texture, quality=quality)


def extract_color_feature_from_rgb(rgb: np.ndarray, config: ConfusionRerankConfig) -> np.ndarray:
    lab = _convert_color(rgb, "lab")
    hsv = _convert_color(rgb, "hsv")
    rgb_float = rgb.astype(np.float32)
    total = np.sum(rgb_float, axis=2, keepdims=True) + 1e-6
    chroma = np.concatenate([rgb_float[:, :, 0:1] / total, rgb_float[:, :, 1:2] / total], axis=2)
    planes = [
        (lab[:, :, 1], 0.0, 255.0),
        (lab[:, :, 2], 0.0, 255.0),
        (hsv[:, :, 0], 0.0, 180.0),
        (hsv[:, :, 1], 0.0, 255.0),
        (chroma[:, :, 0], 0.0, 1.0),
        (chroma[:, :, 1], 0.0, 1.0),
    ]
    features = []
    height, width = rgb.shape[:2]
    for gy in range(config.grid_y):
        y0 = gy * height // config.grid_y
        y1 = (gy + 1) * height // config.grid_y
        for gx in range(config.grid_x):
            x0 = gx * width // config.grid_x
            x1 = (gx + 1) * width // config.grid_x
            for plane, low, high in planes:
                features.append(_histogram(plane[y0:y1, x0:x1], bins=config.color_bins, low=low, high=high))
    return np.concatenate(features).astype(np.float32, copy=False)


def extract_texture_feature_from_gray(gray: np.ndarray, config: ConfusionRerankConfig) -> np.ndarray:
    gray_float = gray.astype(np.float32)
    try:
        import cv2

        dx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    except ModuleNotFoundError:
        dy, dx = np.gradient(gray_float)
    magnitude = np.sqrt(dx * dx + dy * dy)
    features = []
    height, width = gray.shape[:2]
    for gy in range(config.texture_grid_y):
        y0 = gy * height // config.texture_grid_y
        y1 = (gy + 1) * height // config.texture_grid_y
        for gx in range(config.texture_grid_x):
            x0 = gx * width // config.texture_grid_x
            x1 = (gx + 1) * width // config.texture_grid_x
            features.append(_histogram(gray[y0:y1, x0:x1], bins=config.texture_bins, low=0.0, high=255.0))
            features.append(_histogram(magnitude[y0:y1, x0:x1], bins=config.texture_bins, low=0.0, high=255.0))
    return np.concatenate(features).astype(np.float32, copy=False)


def compute_quality(gray: np.ndarray) -> dict[str, float]:
    values = gray.astype(np.float32)
    brightness = float(np.mean(values) / 255.0)
    contrast = float(np.std(values) / 128.0)
    try:
        import cv2

        blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except ModuleNotFoundError:
        gy, gx = np.gradient(values)
        blur = float(np.var(gx) + np.var(gy))
    exposure = 1.0 - min(1.0, abs(brightness - 0.5) * 2.0)
    blur_score = min(1.0, blur / 120.0)
    contrast_score = min(1.0, contrast)
    reliability = max(0.0, min(1.0, 0.45 * exposure + 0.35 * contrast_score + 0.20 * blur_score))
    return {
        "brightness": brightness,
        "contrast": contrast,
        "laplacian_var": blur,
        "color_reliability": reliability,
    }


def preprocess_rgb_for_evidence(
    rgb: np.ndarray,
    config: ConfusionRerankConfig,
    *,
    size: tuple[int, int],
) -> np.ndarray:
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("expected RGB image")
    rgb = rgb[:, :, :3].astype(np.uint8, copy=False)
    if config.detect_face:
        faces = _detect_faces(rgb_to_gray(rgb), config)
        if faces:
            rect = max(faces, key=lambda item: int(item[2]) * int(item[3]))
            rgb = _crop_rgb_with_margin(rgb, rect, config.margin_ratio)
        elif not config.fallback_to_full_image:
            raise ValueError("face_not_found")
    return resize_rgb(rgb, size)


def read_rgb_image(image: str | Path | np.ndarray, *, input_adapter: str = "score2026_framework") -> np.ndarray | None:
    if not isinstance(image, (str, Path)):
        return np.asarray(image, dtype=np.uint8)
    path = Path(image)
    try:
        import cv2

        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        if input_adapter == "score2026_framework":
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ok, buffer = cv2.imencode(".jpg", image_rgb)
            if not ok:
                return None
            decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            return decoded[:, :, :3] if decoded is not None else None
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


def chi_square_many(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    numerator = (features - target) ** 2
    denominator = features + target + 1e-12
    return 0.5 * np.sum(numerator / denominator, axis=1)


def collect_gray_candidates(recognizer: Any, gray: np.ndarray, labels: list[str]) -> list[GrayCandidate]:
    if hasattr(recognizer, "predict_collect"):
        try:
            import cv2

            collector = cv2.face.StandardCollector_create(1e9)
            recognizer.predict_collect(gray, collector)
            by_label: dict[str, float] = {}
            for raw_label, distance in collector.getResults():
                index = int(raw_label)
                if index < 0 or index >= len(labels):
                    continue
                label = labels[index]
                by_label[label] = min(by_label.get(label, float("inf")), float(distance))
            if by_label:
                return [GrayCandidate(label, distance) for label, distance in sorted(by_label.items(), key=lambda item: item[1])]
        except Exception:
            pass
    raw_label, distance = recognizer.predict(gray)
    return [GrayCandidate(labels[int(raw_label)], float(distance))]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ConfusionRerankConfig(
        size=_parse_pair(args.resize, ConfusionRerankConfig.size),
        detect_face=bool(args.detect_face),
        equalization=args.equalization,
        input_adapter=args.input_adapter,
        radius=args.radius,
        neighbors=args.neighbors,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        aux_size=_parse_pair(args.aux_resize, ConfusionRerankConfig.aux_size),
        aux_grid_x=args.aux_grid_x,
        aux_grid_y=args.aux_grid_y,
    )
    report = train_directory(train_dir=args.train_dir, output_dir=args.output_dir, config=config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CA-ME-LBPH confusion-aware reranker.")
    parser.add_argument("--train-dir", default="datasets/score2026/Faces_raw")
    parser.add_argument("--output-dir", default="Algorithm_score2026_confusion_rerank_full")
    parser.add_argument("--resize", default="400x450")
    parser.add_argument("--equalization", default="clahe", choices=["none", "hist", "equalizeHist", "clahe"])
    parser.set_defaults(detect_face=False)
    parser.add_argument("--detect-face", dest="detect_face", action="store_true")
    parser.add_argument("--no-detect-face", dest="detect_face", action="store_false")
    parser.add_argument("--input-adapter", default="score2026_framework")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--grid-x", type=int, default=10)
    parser.add_argument("--grid-y", type=int, default=11)
    parser.add_argument("--aux-resize", default="200x200")
    parser.add_argument("--aux-grid-x", type=int, default=7)
    parser.add_argument("--aux-grid-y", type=int, default=7)
    return parser


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


def _create_recognizer(radius: int, neighbors: int, grid_x: int, grid_y: int) -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required; install opencv-contrib-python") from exc
    try:
        return cv2.face.LBPHFaceRecognizer_create(
            radius=int(radius),
            neighbors=int(neighbors),
            grid_x=int(grid_x),
            grid_y=int(grid_y),
        )
    except AttributeError as exc:
        raise ModuleNotFoundError("OpenCV LBPH requires opencv-contrib-python") from exc


def _normalize_candidates(candidates: Iterable[GrayCandidate | tuple[str, float]]) -> list[GrayCandidate]:
    by_label: dict[str, float] = {}
    for item in candidates:
        if isinstance(item, GrayCandidate):
            label = item.label
            confidence = item.confidence
        else:
            label, confidence = item
        value = str(label)
        by_label[value] = min(by_label.get(value, float("inf")), float(confidence))
    return [GrayCandidate(label, confidence) for label, confidence in sorted(by_label.items(), key=lambda item: item[1])]


def _trigger_reason(candidates: list[GrayCandidate], runtime: RerankRuntimeConfig) -> str:
    if float(candidates[0].confidence) >= float(runtime.confidence_gate):
        return "low_confidence"
    if len(candidates) >= 2:
        margin = float(candidates[1].confidence) - float(candidates[0].confidence)
        if margin <= float(runtime.gray_margin_gate):
            return "small_gray_margin"
    return "not_triggered"


def _normalize_distance_map(values: dict[str, float], labels: list[str]) -> dict[str, float]:
    finite = [float(values[label]) for label in labels if label in values and math.isfinite(float(values[label]))]
    if not finite:
        return {label: 1.0 for label in labels}
    low = min(finite)
    high = max(finite)
    if math.isclose(low, high):
        return {label: 0.0 if label in values else 1.0 for label in labels}
    return {
        label: ((float(values[label]) - low) / (high - low)) if label in values else 1.0
        for label in labels
    }


def _quality_vector(quality: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            float(quality.get("brightness", 0.0)),
            float(quality.get("contrast", 0.0)),
            float(quality.get("color_reliability", 0.0)),
        ],
        dtype=np.float32,
    )


def _convert_color(rgb: np.ndarray, target: str) -> np.ndarray:
    try:
        import cv2

        code = cv2.COLOR_RGB2LAB if target == "lab" else cv2.COLOR_RGB2HSV
        return cv2.cvtColor(rgb, code)
    except ModuleNotFoundError:
        return rgb.astype(np.uint8, copy=False)


def _histogram(values: np.ndarray, *, bins: int, low: float, high: float) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32), low, high)
    scaled = np.floor((clipped - low) * int(bins) / max(high - low, 1e-6)).astype(np.int32)
    scaled = np.clip(scaled, 0, int(bins) - 1)
    hist = np.bincount(scaled.ravel(), minlength=int(bins)).astype(np.float32)
    total = float(hist.sum())
    if total:
        hist /= total
    return hist


def _detect_faces(gray: np.ndarray, config: ConfusionRerankConfig) -> list[tuple[int, int, int, int]]:
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
    rgb: np.ndarray,
    rect: tuple[int, int, int, int],
    margin_ratio: float,
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
        "status",
        "reranked",
        "trigger_reason",
        "secondary_label",
        "original_score",
        "secondary_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


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


def _dedupe_labels(labels: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for label in labels:
        value = str(label)
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
