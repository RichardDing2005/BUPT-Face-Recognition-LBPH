from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SOURCE_IMAGE = Path("datasets/score2026/Faces_raw/2024210809/1.jpg")
DEFAULT_OUTPUT_DIR = Path(r"C:\tmp\lbph_augmentation_preview_2024210809")


@dataclass(frozen=True)
class AugmentationPreviewConfig:
    source_image: str | Path = DEFAULT_SOURCE_IMAGE
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
    noise_sigma: float = 5.0
    noise_seed: int = 42
    blur_kernel_size: int = 3
    blur_sigma: float = 0.8
    jpeg_quality: int = 95


def add_gaussian_noise(image: np.ndarray, *, sigma: float = 5.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, float(sigma), size=image.shape)
    noisy = image.astype(np.float32) + noise.astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_gaussian_blur(image: np.ndarray, *, kernel_size: int = 3, sigma: float = 0.8) -> np.ndarray:
    cv2 = _require_cv2()
    kernel = _odd_kernel_size(kernel_size)
    return cv2.GaussianBlur(image, (kernel, kernel), float(sigma)).astype(np.uint8, copy=False)


def create_augmentation_preview(config: AugmentationPreviewConfig) -> dict[str, Any]:
    cv2 = _require_cv2()
    source = Path(config.source_image).resolve()
    output_dir = Path(config.output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source image not found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read source image: {source}")

    original_path = output_dir / "original.jpg"
    noise_path = output_dir / f"noise_gaussian_sigma{_number_token(config.noise_sigma)}.jpg"
    blur_path = output_dir / (
        f"blur_gaussian_k{_odd_kernel_size(config.blur_kernel_size)}_"
        f"sigma{_number_token(config.blur_sigma)}.jpg"
    )
    manifest_path = output_dir / "preview_manifest.json"

    _write_original(source, original_path, image, quality=int(config.jpeg_quality))
    _write_jpeg(
        noise_path,
        add_gaussian_noise(image, sigma=float(config.noise_sigma), seed=int(config.noise_seed)),
        quality=int(config.jpeg_quality),
    )
    _write_jpeg(
        blur_path,
        apply_gaussian_blur(
            image,
            kernel_size=int(config.blur_kernel_size),
            sigma=float(config.blur_sigma),
        ),
        quality=int(config.jpeg_quality),
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "manual_review_only",
        "source_image": str(source),
        "output_dir": str(output_dir),
        "parameters": {
            "noise": {
                "type": "gaussian",
                "sigma": float(config.noise_sigma),
                "seed": int(config.noise_seed),
            },
            "blur": {
                "type": "gaussian",
                "kernel_size": int(_odd_kernel_size(config.blur_kernel_size)),
                "sigma": float(config.blur_sigma),
            },
            "jpeg_quality": int(config.jpeg_quality),
        },
        "outputs": {
            "original": str(original_path),
            "noise": str(noise_path),
            "blur": str(blur_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = create_augmentation_preview(
        AugmentationPreviewConfig(
            source_image=args.source_image,
            output_dir=args.output_dir,
            noise_sigma=args.noise_sigma,
            noise_seed=args.noise_seed,
            blur_kernel_size=args.blur_kernel_size,
            blur_sigma=args.blur_sigma,
            jpeg_quality=args.jpeg_quality,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create light blur/noise preview images for manual audit.")
    parser.add_argument("--source-image", default=str(DEFAULT_SOURCE_IMAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--noise-sigma", type=float, default=5.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--blur-kernel-size", type=int, default=3)
    parser.add_argument("--blur-sigma", type=float, default=0.8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def _write_original(source: Path, output: Path, image: np.ndarray, *, quality: int) -> None:
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source, output)
        return
    _write_jpeg(output, image, quality=quality)


def _write_jpeg(path: Path, image: np.ndarray, *, quality: int) -> None:
    cv2 = _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError(f"failed to write image: {path}")


def _odd_kernel_size(value: int) -> int:
    kernel = max(1, int(value))
    return kernel if kernel % 2 == 1 else kernel + 1


def _number_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "_").replace("-", "m")


def _require_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for augmentation preview") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
