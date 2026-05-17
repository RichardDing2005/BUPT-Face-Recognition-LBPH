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
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("CAMELBPHScore2026")
        self.algorithm_dir = Path(__file__).resolve().parent
        self.preprocess_config = self._load_preprocess_config()
        self.rerank_config = self._load_rerank_config()
        self.label_mapping = self._load_label_mapping()
        self.labels = [self.label_mapping[str(index)] for index in sorted(int(key) for key in self.label_mapping)]
        self.model = self._load_model("face_recognizer_model.xml")
        self.aux_model = self._load_model("gray_aux_model.xml")
        self.index = self._load_evidence_index()
        self.face_detector = self._load_face_detector() if self.preprocess_config["detect_face"] else None

    def run(self) -> None:
        end_flag = False
        while not end_flag:
            data_model = self._problemInterface.getData()
            if data_model is None:
                continue
            image = getattr(data_model, "data", None)
            if image is not None:
                report_model = ReportModel()
                report_model.result_label = self._predict_image(image)
                self._problemInterface.report(report_model)
            end_flag = bool(getattr(data_model, "finishedFlag", False))

    def _predict_image(self, image: Any) -> str:
        source_rgb = self._to_rgb(image)
        if source_rgb is None:
            return "unknown"
        gray = self._prepare_gray_for_lbph(source_rgb, self.preprocess_config["size"])
        primary_rgb = self._prepare_rgb(source_rgb, tuple(self.preprocess_config["size"]))
        if primary_rgb is None:
            return "unknown"
        try:
            gray_candidates = self._collect_gray_candidates(self.model, gray)
        except Exception as exc:
            self.logger.exception("primary LBPH prediction failed: %s", exc)
            return "unknown"
        if not gray_candidates:
            return "unknown"
        original_label, confidence = gray_candidates[0]
        if self._trigger_reason(gray_candidates) == "not_triggered":
            return original_label

        candidate_labels = self._dedupe_labels(
            [label for label, _distance in gray_candidates[: self.rerank_config["candidate_top_k"]]]
        )
        if original_label not in candidate_labels:
            candidate_labels.insert(0, original_label)

        aux_rgb = self._prepare_rgb(source_rgb, tuple(self.rerank_config["aux_size"]))
        aux_candidates = []
        if aux_rgb is not None:
            aux_gray = self._prepare_gray_for_lbph(source_rgb, self.rerank_config["aux_size"])
            aux_candidates = self._collect_gray_candidates(self.aux_model, aux_gray)

        color_feature, texture_feature, quality = self._extract_evidence(primary_rgb)
        scores = self._candidate_scores(
            candidate_labels=candidate_labels,
            gray_candidates=gray_candidates,
            aux_candidates=aux_candidates,
            color_feature=color_feature,
            texture_feature=texture_feature,
            quality=quality,
        )
        if not scores or original_label not in scores:
            return original_label
        secondary_label, secondary_score = min(scores.items(), key=lambda item: item[1])
        original_score = scores[original_label]
        if (
            secondary_label != original_label
            and float(secondary_score) <= float(original_score) * (1.0 - self.rerank_config["switch_margin"])
        ):
            return secondary_label
        return original_label

    def _load_model(self, name: str) -> Any:
        model_path = self.algorithm_dir / name
        if not model_path.is_file():
            raise FileNotFoundError(f"LBPH model not found: {model_path}")
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError as exc:
            raise RuntimeError("OpenCV LBPH requires opencv-contrib-python") from exc
        recognizer.read(str(model_path))
        return recognizer

    def _load_evidence_index(self) -> dict[str, Any]:
        index_path = self.algorithm_dir / "evidence_index.npz"
        if not index_path.is_file():
            raise FileNotFoundError(f"evidence index not found: {index_path}")
        payload = np.load(index_path)
        sample_labels = payload["sample_labels"].astype(str)
        color_features = payload["color_features"].astype(np.float32, copy=False)
        texture_features = payload["texture_features"].astype(np.float32, copy=False)
        if color_features.shape[0] != len(sample_labels) or texture_features.shape[0] != len(sample_labels):
            raise ValueError("invalid evidence_index.npz")
        return {
            "sample_labels": sample_labels,
            "color_features": color_features,
            "texture_features": texture_features,
        }

    def _load_label_mapping(self) -> dict[str, str]:
        mapping_path = self.algorithm_dir / "label_mapping.json"
        with mapping_path.open("r", encoding="utf-8-sig") as file:
            payload = json.load(file)
        id_to_name = payload.get("id_to_name") or {}
        if not id_to_name and payload.get("name_to_id"):
            id_to_name = {str(label): str(name) for name, label in payload["name_to_id"].items()}
        return {str(label): str(name) for label, name in id_to_name.items()}

    def _load_preprocess_config(self) -> dict[str, Any]:
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
        config_path = self.algorithm_dir / "preprocess_config.json"
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
        default = {
            "candidate_top_k": 4,
            "confidence_gate": 60.0,
            "gray_margin_gate": 65.0,
            "switch_margin": 0.05,
            "primary_weight": 0.15,
            "aux_weight": 0.35,
            "color_weight": 0.30,
            "texture_weight": 0.20,
            "color_reliability_floor": 0.35,
            "aux_size": [200, 200],
            "color_bins": 8,
            "texture_bins": 8,
            "grid_x": 10,
            "grid_y": 11,
            "texture_grid_x": 5,
            "texture_grid_y": 5,
        }
        config_path = self.algorithm_dir / "rerank_runtime_config.json"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8-sig") as file:
                default.update(json.load(file))
        default["candidate_top_k"] = int(default["candidate_top_k"])
        for key in (
            "confidence_gate",
            "gray_margin_gate",
            "switch_margin",
            "primary_weight",
            "aux_weight",
            "color_weight",
            "texture_weight",
            "color_reliability_floor",
        ):
            default[key] = float(default[key])
        default["aux_size"] = self._parse_pair(default["aux_size"], (200, 200))
        for key in ("color_bins", "texture_bins", "grid_x", "grid_y", "texture_grid_x", "texture_grid_y"):
            default[key] = int(default[key])
        return default

    def _load_face_detector(self) -> Any:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            raise RuntimeError("failed to load OpenCV Haar cascade")
        return detector

    def _prepare_rgb(self, image: Any, size: tuple[int, int]) -> np.ndarray | None:
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
        return cv2.resize(rgb, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)

    def _prepare_gray_for_lbph(self, image: Any, size: tuple[int, int] | list[int]) -> np.ndarray:
        rgb = self._to_rgb(image)
        if rgb is None:
            raise ValueError("image conversion failed")
        gray = self._to_gray(rgb)
        if self.preprocess_config["detect_face"]:
            if self.face_detector is None:
                raise ValueError("face detector missing")
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=self.preprocess_config["scale_factor"],
                minNeighbors=self.preprocess_config["min_neighbors"],
                minSize=tuple(self.preprocess_config["min_face_size"]),
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda rect: int(rect[2]) * int(rect[3]))
                gray = self._crop_gray_with_margin(gray, int(x), int(y), int(w), int(h))
            elif not self.preprocess_config["fallback_to_full_image"]:
                raise ValueError("face_not_found")
        resized = cv2.resize(gray, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
        return self._equalize(resized).astype(np.uint8, copy=False)

    def _to_rgb(self, image: Any) -> np.ndarray | None:
        if image is None:
            return None
        if isinstance(image, (bytes, bytearray)):
            nparr = np.frombuffer(image, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
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

    def _collect_gray_candidates(self, recognizer: Any, gray: np.ndarray) -> list[tuple[str, float]]:
        if hasattr(recognizer, "predict_collect"):
            collector = cv2.face.StandardCollector_create(1e9)
            recognizer.predict_collect(gray, collector)
            by_label: dict[str, float] = {}
            for raw_label, distance in collector.getResults():
                label = self.label_mapping.get(str(int(raw_label)))
                if label is None:
                    continue
                by_label[label] = min(by_label.get(label, float("inf")), float(distance))
            if by_label:
                return sorted(by_label.items(), key=lambda item: item[1])
        raw_label, distance = recognizer.predict(gray)
        return [(self.label_mapping.get(str(int(raw_label)), "unknown"), float(distance))]

    def _trigger_reason(self, candidates: list[tuple[str, float]]) -> str:
        if float(candidates[0][1]) >= self.rerank_config["confidence_gate"]:
            return "low_confidence"
        if len(candidates) >= 2:
            margin = float(candidates[1][1]) - float(candidates[0][1])
            if margin <= self.rerank_config["gray_margin_gate"]:
                return "small_gray_margin"
        return "not_triggered"

    def _candidate_scores(
        self,
        *,
        candidate_labels: list[str],
        gray_candidates: list[tuple[str, float]],
        aux_candidates: list[tuple[str, float]],
        color_feature: np.ndarray,
        texture_feature: np.ndarray,
        quality: dict[str, float],
    ) -> dict[str, float]:
        primary_raw = {label: distance for label, distance in gray_candidates}
        aux_raw = {label: distance for label, distance in aux_candidates}
        color_raw = self._label_feature_distances(self.index["color_features"], color_feature, candidate_labels)
        texture_raw = self._label_feature_distances(self.index["texture_features"], texture_feature, candidate_labels)
        primary = self._normalize_distance_map(primary_raw, candidate_labels)
        aux = self._normalize_distance_map(aux_raw, candidate_labels)
        color = self._normalize_distance_map(color_raw, candidate_labels)
        texture = self._normalize_distance_map(texture_raw, candidate_labels)
        reliability = min(
            1.0,
            max(self.rerank_config["color_reliability_floor"], float(quality.get("color_reliability", 1.0))),
        )
        primary_weight = self.rerank_config["primary_weight"]
        aux_weight = self.rerank_config["aux_weight"]
        color_weight = self.rerank_config["color_weight"] * reliability
        texture_weight = self.rerank_config["texture_weight"] + self.rerank_config["color_weight"] * (1.0 - reliability)
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
        candidate_labels: Iterable[str],
    ) -> dict[str, float]:
        per_label: dict[str, float] = {}
        target = np.asarray(target, dtype=np.float32)
        sample_labels = self.index["sample_labels"]
        for label in candidate_labels:
            subset = matrix[sample_labels == str(label)]
            if subset.size:
                per_label[str(label)] = float(self._chi_square_many(subset, target).min())
        return per_label

    def _extract_evidence(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        gray = self._to_gray(rgb)
        return self._extract_color_feature(rgb), self._extract_texture_feature(gray), self._compute_quality(gray)

    def _extract_color_feature(self, rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
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
        for gy in range(self.rerank_config["grid_y"]):
            y0 = gy * height // self.rerank_config["grid_y"]
            y1 = (gy + 1) * height // self.rerank_config["grid_y"]
            for gx in range(self.rerank_config["grid_x"]):
                x0 = gx * width // self.rerank_config["grid_x"]
                x1 = (gx + 1) * width // self.rerank_config["grid_x"]
                for plane, low, high in planes:
                    features.append(self._histogram(plane[y0:y1, x0:x1], self.rerank_config["color_bins"], low, high))
        return np.concatenate(features).astype(np.float32, copy=False)

    def _extract_texture_feature(self, gray: np.ndarray) -> np.ndarray:
        gray_float = gray.astype(np.float32)
        dx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(dx * dx + dy * dy)
        features = []
        height, width = gray.shape[:2]
        for gy in range(self.rerank_config["texture_grid_y"]):
            y0 = gy * height // self.rerank_config["texture_grid_y"]
            y1 = (gy + 1) * height // self.rerank_config["texture_grid_y"]
            for gx in range(self.rerank_config["texture_grid_x"]):
                x0 = gx * width // self.rerank_config["texture_grid_x"]
                x1 = (gx + 1) * width // self.rerank_config["texture_grid_x"]
                features.append(self._histogram(gray[y0:y1, x0:x1], self.rerank_config["texture_bins"], 0.0, 255.0))
                features.append(
                    self._histogram(magnitude[y0:y1, x0:x1], self.rerank_config["texture_bins"], 0.0, 255.0)
                )
        return np.concatenate(features).astype(np.float32, copy=False)

    @staticmethod
    def _compute_quality(gray: np.ndarray) -> dict[str, float]:
        values = gray.astype(np.float32)
        brightness = float(np.mean(values) / 255.0)
        contrast = float(np.std(values) / 128.0)
        blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        exposure = 1.0 - min(1.0, abs(brightness - 0.5) * 2.0)
        blur_score = min(1.0, blur / 120.0)
        contrast_score = min(1.0, contrast)
        reliability = max(0.0, min(1.0, 0.45 * exposure + 0.35 * contrast_score + 0.20 * blur_score))
        return {"brightness": brightness, "contrast": contrast, "laplacian_var": blur, "color_reliability": reliability}

    @staticmethod
    def _histogram(values: np.ndarray, bins: int, low: float, high: float) -> np.ndarray:
        clipped = np.clip(values.astype(np.float32), low, high)
        scaled = np.floor((clipped - low) * int(bins) / max(high - low, 1e-6)).astype(np.int32)
        scaled = np.clip(scaled, 0, int(bins) - 1)
        hist = np.bincount(scaled.ravel(), minlength=int(bins)).astype(np.float32)
        total = float(hist.sum())
        if total:
            hist /= total
        return hist

    @staticmethod
    def _chi_square_many(features: np.ndarray, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        numerator = (features - target) ** 2
        denominator = features + target + 1e-12
        return 0.5 * np.sum(numerator / denominator, axis=1)

    @staticmethod
    def _normalize_distance_map(values: dict[str, float], labels: list[str]) -> dict[str, float]:
        finite = [float(values[label]) for label in labels if label in values and np.isfinite(float(values[label]))]
        if not finite:
            return {label: 1.0 for label in labels}
        low = min(finite)
        high = max(finite)
        if abs(high - low) < 1e-12:
            return {label: 0.0 if label in values else 1.0 for label in labels}
        return {label: ((float(values[label]) - low) / (high - low)) if label in values else 1.0 for label in labels}

    def _crop_rgb_with_margin(self, rgb: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        margin_ratio = self.preprocess_config["margin_ratio"]
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(rgb.shape[1], x + w + margin_x)
        bottom = min(rgb.shape[0], y + h + margin_y)
        return rgb[top:bottom, left:right, :]

    def _crop_gray_with_margin(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        margin_ratio = self.preprocess_config["margin_ratio"]
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(gray.shape[1], x + w + margin_x)
        bottom = min(gray.shape[0], y + h + margin_y)
        return gray[top:bottom, left:right]

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
