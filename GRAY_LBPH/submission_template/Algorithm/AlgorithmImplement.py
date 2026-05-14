from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
        self.logger = logging.getLogger("LBPHScore2026")
        self.algorithm_dir = Path(__file__).resolve().parent
        self.preprocess_config = self._load_preprocess_config()
        self.label_mapping = self._load_label_mapping()
        self.model = self._load_model()
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

    def _load_label_mapping(self) -> dict[str, str]:
        mapping_path = self.algorithm_dir / "label_mapping.json"
        if not mapping_path.is_file():
            raise FileNotFoundError(f"label mapping not found: {mapping_path}")
        with mapping_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        id_to_name = payload.get("id_to_name") or {}
        if not id_to_name and payload.get("name_to_id"):
            id_to_name = {str(label): str(name) for name, label in payload["name_to_id"].items()}
        return {str(label): str(name) for label, name in id_to_name.items()}

    def _load_preprocess_config(self) -> dict[str, Any]:
        config_path = self.algorithm_dir / "preprocess_config.json"
        default = {
            "size": [200, 200],
            "detect_face": True,
            "equalization": "clahe",
            "margin_ratio": 0.15,
            "min_face_size": [40, 40],
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "fallback_to_full_image": True,
            "input_adapter": "image_file",
        }
        if not config_path.is_file():
            return default
        with config_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        default.update(payload)
        default["size"] = self._parse_pair(default["size"], (200, 200))
        default["min_face_size"] = self._parse_pair(default["min_face_size"], (40, 40))
        default["detect_face"] = self._parse_bool(default["detect_face"])
        default["fallback_to_full_image"] = self._parse_bool(default["fallback_to_full_image"])
        default["margin_ratio"] = float(default["margin_ratio"])
        default["scale_factor"] = float(default["scale_factor"])
        default["min_neighbors"] = int(default["min_neighbors"])
        default["equalization"] = str(default["equalization"])
        default["input_adapter"] = str(default.get("input_adapter") or "image_file")
        return default

    def _load_face_detector(self) -> Any:
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if detector.empty():
            raise RuntimeError("failed to load OpenCV Haar cascade")
        return detector

    def _predict_image(self, image: np.ndarray) -> str:
        face = self._preprocess_image(image)
        if face is None:
            return "unknown"
        try:
            raw_label, _confidence = self.model.predict(face)
        except Exception as exc:
            self.logger.exception("LBPH prediction failed: %s", exc)
            return "unknown"
        return self.label_mapping.get(str(int(raw_label)), "unknown")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray | None:
        gray = self._to_gray(image)
        if gray is None:
            return None

        crop = gray
        if self.preprocess_config["detect_face"]:
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
                crop = self._crop_with_margin(gray, int(x), int(y), int(w), int(h))
            elif not self.preprocess_config["fallback_to_full_image"]:
                return None

        width, height = self.preprocess_config["size"]
        resized = cv2.resize(crop, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        return self._equalize(resized).astype(np.uint8, copy=False)

    def _to_gray(self, image: np.ndarray) -> np.ndarray | None:
        if image is None:
            return None
        frame = np.asarray(image)
        if frame.size == 0:
            return None
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            if self.preprocess_config.get("input_adapter") == "score2026_framework":
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            if self.preprocess_config.get("input_adapter") == "score2026_framework":
                gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            return None
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    def _crop_with_margin(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        margin_ratio = self.preprocess_config["margin_ratio"]
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(gray.shape[1], x + w + margin_x)
        bottom = min(gray.shape[0], y + h + margin_y)
        return gray[top:bottom, left:right]

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
