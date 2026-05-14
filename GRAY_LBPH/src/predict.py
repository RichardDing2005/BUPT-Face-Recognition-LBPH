from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .preprocess import PreprocessConfig, preprocess_image


@dataclass(frozen=True)
class LabelMapping:
    name_to_id: dict[str, int]
    id_to_name: dict[int, str]


def load_label_mapping(path: str | Path) -> LabelMapping:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    name_to_id = {str(name): int(label) for name, label in data.get("name_to_id", {}).items()}
    id_to_name = {int(label): str(name) for label, name in data.get("id_to_name", {}).items()}
    if not id_to_name and name_to_id:
        id_to_name = {label: name for name, label in name_to_id.items()}
    return LabelMapping(name_to_id=name_to_id, id_to_name=id_to_name)


def confidence_matches(confidence: float, threshold: float | None) -> bool:
    return True if threshold is None else float(confidence) <= float(threshold)


class LBPHPredictor:
    def __init__(
        self,
        *,
        algorithm_dir: str | Path,
        model_path: str | Path | None = None,
        mapping_path: str | Path | None = None,
        preprocess_config_path: str | Path | None = None,
        threshold: float | None = None,
    ) -> None:
        self.algorithm_dir = Path(algorithm_dir)
        self.model_path = self._artifact_path(model_path, "face_recognizer_model.xml")
        self.mapping_path = self._artifact_path(mapping_path, "label_mapping.json")
        self.preprocess_config_path = (
            self._artifact_path(preprocess_config_path, "preprocess_config.json")
            if preprocess_config_path
            else self.algorithm_dir / "preprocess_config.json"
        )
        self.threshold = threshold
        self.mapping = load_label_mapping(self.mapping_path)
        self.preprocess_config = self._load_preprocess_config()
        self.recognizer = self._load_recognizer()

    def predict_image(self, image_path: str | Path) -> dict[str, Any]:
        result = preprocess_image(image_path, self.preprocess_config)
        preprocess_payload = _preprocess_payload(result)
        if result.status == "read_failed":
            return {"label": None, "confidence": None, "status": "read_failed", **preprocess_payload}
        if result.status == "face_not_found" and not self.preprocess_config.fallback_to_full_image:
            return {"label": None, "confidence": None, "status": "face_not_found", **preprocess_payload}
        try:
            raw_label, raw_confidence = self.recognizer.predict(result.face)
        except Exception as exc:
            return {
                "label": None,
                "confidence": None,
                "status": str(exc) or exc.__class__.__name__,
                **preprocess_payload,
            }
        confidence = float(raw_confidence)
        label = self.mapping.id_to_name.get(int(raw_label))
        if label is None:
            return {"label": None, "confidence": confidence, "status": "label_not_found", **preprocess_payload}
        if not confidence_matches(confidence, self.threshold):
            return {
                "label": None,
                "confidence": confidence,
                "status": "unknown",
                "candidate_label": label,
                **preprocess_payload,
            }
        return {"label": label, "confidence": confidence, "status": "ok", **preprocess_payload}

    def _load_preprocess_config(self) -> PreprocessConfig:
        if not self.preprocess_config_path.exists():
            return PreprocessConfig()
        data = json.loads(self.preprocess_config_path.read_text(encoding="utf-8"))
        return PreprocessConfig.from_dict(data)

    def _load_recognizer(self) -> Any:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("OpenCV is required; install opencv-contrib-python") from exc
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError as exc:
            raise ModuleNotFoundError("OpenCV LBPH requires opencv-contrib-python") from exc
        recognizer.read(str(self.model_path))
        return recognizer

    def _artifact_path(self, path: str | Path | None, default_name: str) -> Path:
        if path is None:
            return self.algorithm_dir / default_name
        value = Path(path)
        return value if value.is_absolute() else self.algorithm_dir / value


def predict_image(
    image_path: str | Path,
    *,
    algorithm_dir: str | Path,
    threshold: float | None = None,
) -> dict[str, Any]:
    return LBPHPredictor(algorithm_dir=algorithm_dir, threshold=threshold).predict_image(image_path)


def _preprocess_payload(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", {}) or {}
    return {
        "preprocess_status": getattr(result, "status", None),
        "face_detected": metadata.get("face_detected"),
        "face_rect": metadata.get("face_rect"),
    }
