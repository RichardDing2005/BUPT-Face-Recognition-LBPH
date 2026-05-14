from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from Algorithm.Interface.AlgorithmInterface import AlgorithmInterface
from Algorithm.Interface.Model.ReportModel import ReportModel


class AlgorithmImplement(AlgorithmInterface):
    def __init__(self) -> None:
        super().__init__()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ColorRerankLBPHScore2026")
        self.algorithm_dir = Path(__file__).resolve().parent
        self.preprocess_config = self._load_preprocess_config()
        self.rerank_config = self._load_rerank_config()
        self.label_mapping = self._load_label_mapping()
        self.model = self._load_model()
        self.color_index = self._load_color_index()
        self.face_detector = self._load_face_detector() if self.preprocess_config["detect_face"] else None

    def run(self) -> None:
        end_flag = False
        while not end_flag:
            data_model = self._problemInterface.getData()
            if data_model is None:
                continue

            image = getattr(data_model, "data", None)
            if image is not None:
                result = self._predict_image(image)
                report_model = ReportModel()
                report_model.result_label = result
                self._problemInterface.report(report_model)

            end_flag = bool(getattr(data_model, "finishedFlag", False))

    def _load_model(self) -> Any:
        model_path = self.algorithm_dir / "face_recognizer_model.xml"
        if not model_path.is_file():
            raise FileNotFoundError(f"LBPH model not found: {model_path}")
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError as exc:
            raise RuntimeError("OpenCV LBPH requires opencv-contrib-python") from exc
        recognizer.read(str(model_path))
        return recognizer

    def _load_color_index(self) -> dict[str, Any]:
        index_path = self.algorithm_dir / "color_index.npz"
        if not index_path.is_file():
            raise FileNotFoundError(f"color index not found: {index_path}")
        payload = np.load(index_path)
        sample_labels = payload["sample_labels"].astype(str)
        color_features = payload["color_features"].astype(np.float32, copy=False)
        if color_features.ndim != 2 or color_features.shape[0] != len(sample_labels):
            raise ValueError("invalid color_index.npz")
        return {"sample_labels": sample_labels, "color_features": color_features}

    def _load_label_mapping(self) -> dict[str, str]:
        mapping_path = self.algorithm_dir / "label_mapping.json"
        if not mapping_path.is_file():
            raise FileNotFoundError(f"label mapping not found: {mapping_path}")
        with mapping_path.open("r", encoding="utf-8-sig") as file:
            payload = json.load(file)
        id_to_name = payload.get("id_to_name") or {}
        if not id_to_name and payload.get("name_to_id"):
            id_to_name = {str(label): str(name) for name, label in payload["name_to_id"].items()}
        return {str(label): str(name) for label, name in id_to_name.items()}

    def _load_preprocess_config(self) -> dict[str, Any]:
        config_path = self.algorithm_dir / "preprocess_config.json"
        default = {
            "size": [400, 450],
            "detect_face": False,
            "equalization": "clahe",
            "margin_ratio": 0.15,
            "min_face_size": [40, 40],
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "fallback_to_full_image": True,
            "input_adapter": "score2026_framework",
        }
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8-sig") as file:
                default.update(json.load(file))
        default["size"] = self._parse_pair(default["size"], (400, 450))
        default["min_face_size"] = self._parse_pair(default["min_face_size"], (40, 40))
        default["detect_face"] = self._parse_bool(default["detect_face"])
        default["fallback_to_full_image"] = self._parse_bool(default["fallback_to_full_image"])
        default["margin_ratio"] = float(default["margin_ratio"])
        default["scale_factor"] = float(default["scale_factor"])
        default["min_neighbors"] = int(default["min_neighbors"])
        default["equalization"] = str(default["equalization"])
        default["input_adapter"] = str(default.get("input_adapter") or "score2026_framework")
        return default

    def _load_rerank_config(self) -> dict[str, Any]:
        config_path = self.algorithm_dir / "rerank_runtime_config.json"
        default = {
            "candidate_top_k": 2,
            "confidence_gate": 60.0,
            "rerank_margin_ratio": 0.1,
            "grid_x": 10,
            "grid_y": 11,
            "color_bins": 8,
        }
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8-sig") as file:
                default.update(json.load(file))
        default["candidate_top_k"] = int(default["candidate_top_k"])
        default["confidence_gate"] = float(default["confidence_gate"])
        default["rerank_margin_ratio"] = float(default["rerank_margin_ratio"])
        default["grid_x"] = int(default["grid_x"])
        default["grid_y"] = int(default["grid_y"])
        default["color_bins"] = int(default["color_bins"])
        return default

    def _load_face_detector(self) -> Any:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            raise RuntimeError("failed to load OpenCV Haar cascade")
        return detector

    def _predict_image(self, image: np.ndarray) -> str:
        prepared_rgb = self._prepare_rgb(image)
        if prepared_rgb is None:
            return "unknown"
        gray = self._equalize(self._to_gray(prepared_rgb)).astype(np.uint8, copy=False)
        try:
            gray_candidates = self._collect_gray_candidates(gray)
        except Exception as exc:
            self.logger.exception("LBPH prediction failed: %s", exc)
            return "unknown"
        if not gray_candidates:
            return "unknown"

        original_label, confidence = gray_candidates[0]
        if confidence < self.rerank_config["confidence_gate"]:
            return original_label

        candidate_labels = [label for label, _dist in gray_candidates[: self.rerank_config["candidate_top_k"]]]
        color_feature = self._extract_color_feature(prepared_rgb)
        return self._rerank_label(
            original_label=original_label,
            candidate_labels=candidate_labels,
            color_feature=color_feature,
        )

    def _collect_gray_candidates(self, gray: np.ndarray) -> list[tuple[str, float]]:
        if hasattr(self.model, "predict_collect"):
            collector = cv2.face.StandardCollector_create(1e9)
            self.model.predict_collect(gray, collector)
            by_label: dict[str, float] = {}
            for raw_label, distance in collector.getResults():
                label = self.label_mapping.get(str(int(raw_label)))
                if label is None:
                    continue
                by_label[label] = min(by_label.get(label, float("inf")), float(distance))
            if by_label:
                return sorted(by_label.items(), key=lambda item: item[1])
        raw_label, distance = self.model.predict(gray)
        label = self.label_mapping.get(str(int(raw_label)), "unknown")
        return [(label, float(distance))]

    def _rerank_label(
        self,
        *,
        original_label: str,
        candidate_labels: Iterable[str],
        color_feature: np.ndarray,
    ) -> str:
        allowed_labels = self._dedupe_labels([original_label, *candidate_labels])
        per_label = self._label_color_distances(color_feature, allowed_labels)
        if not per_label:
            return original_label
        color_label, color_distance = min(per_label.items(), key=lambda item: item[1])
        original_distance = per_label.get(original_label)
        if original_distance is None or color_label == original_label:
            return original_label
        margin = self.rerank_config["rerank_margin_ratio"]
        if color_distance <= original_distance * (1.0 - margin):
            return color_label
        return original_label

    def _label_color_distances(self, color_feature: np.ndarray, labels: list[str]) -> dict[str, float]:
        sample_labels = self.color_index["sample_labels"]
        color_features = self.color_index["color_features"]
        distances = self._chi_square_many(color_features, color_feature)
        per_label: dict[str, float] = {}
        for label in labels:
            values = distances[sample_labels == label]
            if values.size:
                per_label[label] = float(values.min())
        return per_label

    def _prepare_rgb(self, image: np.ndarray) -> np.ndarray | None:
        rgb = self._to_rgb(image)
        if rgb is None:
            return None
        if self.preprocess_config["detect_face"]:
            gray = self._to_gray(rgb)
            if self.face_detector is None:
                return None
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=self.preprocess_config["scale_factor"],
                minNeighbors=self.preprocess_config["min_neighbors"],
                minSize=tuple(self.preprocess_config["min_face_size"]),
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: int(rect[2]) * int(rect[3]))
                rgb = self._crop_rgb_with_margin(rgb, int(x), int(y), int(w), int(h))
            elif not self.preprocess_config["fallback_to_full_image"]:
                return None
        width, height = self.preprocess_config["size"]
        return cv2.resize(rgb, (int(width), int(height)), interpolation=cv2.INTER_AREA)

    def _to_rgb(self, image: np.ndarray) -> np.ndarray | None:
        if image is None:
            return None
        frame = np.asarray(image)
        if frame.size == 0:
            return None
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            if self.preprocess_config.get("input_adapter") == "score2026_framework":
                rgb = frame
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            if self.preprocess_config.get("input_adapter") == "score2026_framework":
                rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            return None
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb[:, :, :3]

    @staticmethod
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _crop_rgb_with_margin(self, rgb: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        margin_ratio = self.preprocess_config["margin_ratio"]
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(rgb.shape[1], x + w + margin_x)
        bottom = min(rgb.shape[0], y + h + margin_y)
        return rgb[top:bottom, left:right, :]

    def _equalize(self, gray: np.ndarray) -> np.ndarray:
        method = (self.preprocess_config["equalization"] or "none").lower()
        if method in {"none", "off", "false"}:
            return gray
        if method in {"hist", "equalizehist", "equalize_hist"}:
            return cv2.equalizeHist(gray)
        if method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        raise ValueError(f"unsupported equalization method: {method}")

    def _extract_color_feature(self, rgb: np.ndarray) -> np.ndarray:
        features = []
        height, width = rgb.shape[:2]
        grid_y = self.rerank_config["grid_y"]
        grid_x = self.rerank_config["grid_x"]
        bins = self.rerank_config["color_bins"]
        for gy in range(grid_y):
            y0 = gy * height // grid_y
            y1 = (gy + 1) * height // grid_y
            for gx in range(grid_x):
                x0 = gx * width // grid_x
                x1 = (gx + 1) * width // grid_x
                cell = rgb[y0:y1, x0:x1, :]
                for channel in range(3):
                    values = cell[:, :, channel].astype(np.int16, copy=False).ravel()
                    hist = np.bincount(
                        np.minimum(values * bins // 256, bins - 1),
                        minlength=bins,
                    ).astype(np.float32)
                    total = float(hist.sum())
                    if total:
                        hist /= total
                    features.append(hist)
        return np.concatenate(features).astype(np.float32, copy=False)

    @staticmethod
    def _chi_square_many(features: np.ndarray, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        numerator = (features - target) ** 2
        denominator = features + target + 1e-12
        return 0.5 * np.sum(numerator / denominator, axis=1)

    @staticmethod
    def _dedupe_labels(labels: Iterable[str]) -> list[str]:
        result = []
        seen = set()
        for label in labels:
            value = str(label)
            if value not in seen:
                result.append(value)
                seen.add(value)
        return result

    @staticmethod
    def _parse_pair(value: Any, fallback: tuple[int, int]) -> list[int]:
        if isinstance(value, str):
            left, _, right = value.lower().partition("x")
            if left and right:
                return [int(left), int(right)]
            return [int(fallback[0]), int(fallback[1])]
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return [int(value[0]), int(value[1])]
        return [int(fallback[0]), int(fallback[1])]

    @staticmethod
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
