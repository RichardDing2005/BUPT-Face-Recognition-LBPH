from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    size: tuple[int, int] = (200, 200)
    detect_face: bool = True
    equalization: str = "clahe"
    margin_ratio: float = 0.15
    min_face_size: tuple[int, int] = (40, 40)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    fallback_to_full_image: bool = True
    input_adapter: str = "image_file"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PreprocessConfig":
        data = data or {}
        size = data.get("size", cls.size)
        if isinstance(size, str):
            left, _, right = size.lower().partition("x")
            size = (int(left), int(right))
        return cls(
            size=(int(size[0]), int(size[1])),
            detect_face=_parse_bool(data.get("detect_face", cls.detect_face)),
            equalization=str(data.get("equalization", cls.equalization)),
            margin_ratio=float(data.get("margin_ratio", cls.margin_ratio)),
            min_face_size=tuple(data.get("min_face_size", cls.min_face_size)),
            scale_factor=float(data.get("scale_factor", cls.scale_factor)),
            min_neighbors=int(data.get("min_neighbors", cls.min_neighbors)),
            fallback_to_full_image=_parse_bool(
                data.get("fallback_to_full_image", cls.fallback_to_full_image)
            ),
            input_adapter=str(data.get("input_adapter", cls.input_adapter)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": [int(self.size[0]), int(self.size[1])],
            "detect_face": self.detect_face,
            "equalization": self.equalization,
            "margin_ratio": self.margin_ratio,
            "min_face_size": [int(self.min_face_size[0]), int(self.min_face_size[1])],
            "scale_factor": self.scale_factor,
            "min_neighbors": self.min_neighbors,
            "fallback_to_full_image": self.fallback_to_full_image,
            "input_adapter": self.input_adapter,
        }


BACKEND_COMPAT_PREPROCESS_PROFILE_ID = "lbph-backend-compatible-v1"
BACKEND_COMPAT_PREPROCESS_CONFIG = PreprocessConfig(
    size=(200, 200),
    detect_face=True,
    equalization="clahe",
    margin_ratio=0.15,
    min_face_size=(40, 40),
    scale_factor=1.1,
    min_neighbors=5,
    fallback_to_full_image=True,
)


@dataclass(frozen=True)
class PreprocessResult:
    face: np.ndarray
    status: str
    metadata: dict[str, Any]


def preprocess_image(image_path: str | Path, config: PreprocessConfig | None = None) -> PreprocessResult:
    config = config or PreprocessConfig()
    path = Path(image_path)
    frame = _read_gray(path, config.input_adapter)
    if frame is None:
        return PreprocessResult(
            face=np.zeros((config.size[1], config.size[0]), dtype=np.uint8),
            status="read_failed",
            metadata={"path": str(path), "size": list(config.size)},
        )

    status = "ok"
    crop = frame
    face_rect = None
    if config.detect_face:
        detected = _detect_faces(frame, config)
        if detected:
            face_rect = max(detected, key=lambda rect: int(rect[2]) * int(rect[3]))
            crop = _crop_with_margin(frame, face_rect, config.margin_ratio)
        elif config.fallback_to_full_image:
            status = "face_not_found"
        else:
            return PreprocessResult(
                face=np.zeros((config.size[1], config.size[0]), dtype=np.uint8),
                status="face_not_found",
                metadata={"path": str(path), "size": list(config.size), "face_detected": False},
            )

    resized = _resize_gray(crop, config.size)
    normalized = _equalize(resized, config.equalization)
    return PreprocessResult(
        face=normalized.astype(np.uint8, copy=False),
        status=status,
        metadata={
            "path": str(path),
            "size": [int(config.size[0]), int(config.size[1])],
            "face_detected": face_rect is not None,
            "face_rect": list(map(int, face_rect)) if face_rect is not None else None,
            "equalization": config.equalization,
            "input_adapter": config.input_adapter,
        },
    )


def _read_gray(path: Path, input_adapter: str = "image_file") -> np.ndarray | None:
    if input_adapter == "score2026_framework":
        return _read_gray_score2026_framework(path)
    if input_adapter not in {"image_file", "", "default"}:
        raise ValueError(f"unsupported input adapter: {input_adapter}")
    try:
        import cv2

        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            return frame
    except ModuleNotFoundError:
        pass
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.array(image.convert("L"), dtype=np.uint8)
    except Exception:
        return None


def _read_gray_score2026_framework(path: Path) -> np.ndarray | None:
    try:
        import cv2

        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, buffer = cv2.imencode(".jpg", image_rgb)
        if not ok:
            return None
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            return None
        return cv2.cvtColor(decoded, cv2.COLOR_RGB2GRAY)
    except ModuleNotFoundError:
        return _read_gray(path, "image_file")


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


def _detect_faces(gray: np.ndarray, config: PreprocessConfig) -> list[tuple[int, int, int, int]]:
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


def _crop_with_margin(gray: np.ndarray, rect: tuple[int, int, int, int], margin_ratio: float) -> np.ndarray:
    x, y, width, height = rect
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(gray.shape[1], x + width + margin_x)
    bottom = min(gray.shape[0], y + height + margin_y)
    return gray[top:bottom, left:right]


def _resize_gray(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(gray, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        from PIL import Image

        image = Image.fromarray(gray.astype(np.uint8), mode="L")
        return np.array(image.resize((int(size[0]), int(size[1]))), dtype=np.uint8)


def _equalize(gray: np.ndarray, equalization: str) -> np.ndarray:
    method = (equalization or "none").lower()
    if method in {"none", "off", "false"}:
        return gray
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for LBPH equalization") from exc
    if method in {"hist", "equalizehist", "equalize_hist"}:
        return cv2.equalizeHist(gray)
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    raise ValueError(f"unsupported equalization method: {equalization}")
