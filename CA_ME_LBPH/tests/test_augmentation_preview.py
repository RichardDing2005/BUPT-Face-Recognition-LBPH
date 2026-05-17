from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gaussian_noise_is_seeded_and_keeps_shape_dtype() -> None:
    from src.augmentation_preview import add_gaussian_noise

    image = np.full((12, 10, 3), 120, dtype=np.uint8)

    first = add_gaussian_noise(image, sigma=5.0, seed=42)
    second = add_gaussian_noise(image, sigma=5.0, seed=42)

    assert np.array_equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == np.uint8
    assert not np.array_equal(first, image)


def test_light_gaussian_blur_keeps_shape_dtype_and_changes_pixels() -> None:
    from src.augmentation_preview import apply_gaussian_blur

    image = np.zeros((11, 11, 3), dtype=np.uint8)
    image[5, 5] = [255, 255, 255]

    blurred = apply_gaussian_blur(image, kernel_size=3, sigma=0.8)

    assert blurred.shape == image.shape
    assert blurred.dtype == np.uint8
    assert blurred[5, 5, 0] < 255
    assert blurred[5, 4, 0] > 0


def test_preview_writes_expected_files_manifest_and_preserves_source(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    from src.augmentation_preview import AugmentationPreviewConfig, create_augmentation_preview

    source = tmp_path / "source.jpg"
    image = np.zeros((16, 18, 3), dtype=np.uint8)
    image[:, :, 0] = 50
    image[:, :, 1] = 100
    image[:, :, 2] = 150
    assert cv2.imwrite(str(source), image)
    before = source.read_bytes()

    output_dir = tmp_path / "preview"
    manifest = create_augmentation_preview(
        AugmentationPreviewConfig(
            source_image=source,
            output_dir=output_dir,
            noise_sigma=5.0,
            noise_seed=42,
            blur_kernel_size=3,
            blur_sigma=0.8,
            jpeg_quality=95,
        )
    )

    assert source.read_bytes() == before
    assert (output_dir / "original.jpg").is_file()
    assert (output_dir / "noise_gaussian_sigma5.jpg").is_file()
    assert (output_dir / "blur_gaussian_k3_sigma0_8.jpg").is_file()
    manifest_path = output_dir / "preview_manifest.json"
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload == manifest
    assert payload["source_image"] == str(source.resolve())
    assert payload["parameters"]["noise"]["sigma"] == 5.0
    assert payload["parameters"]["blur"]["sigma"] == 0.8
    assert payload["outputs"]["noise"].endswith("noise_gaussian_sigma5.jpg")


def test_preview_names_custom_strength_files_from_parameters(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    from src.augmentation_preview import AugmentationPreviewConfig, create_augmentation_preview

    source = tmp_path / "source.jpg"
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)

    output_dir = tmp_path / "preview"
    manifest = create_augmentation_preview(
        AugmentationPreviewConfig(
            source_image=source,
            output_dir=output_dir,
            noise_sigma=7.0,
            blur_kernel_size=3,
            blur_sigma=1.0,
        )
    )

    assert (output_dir / "noise_gaussian_sigma7.jpg").is_file()
    assert (output_dir / "blur_gaussian_k3_sigma1.jpg").is_file()
    assert manifest["outputs"]["noise"].endswith("noise_gaussian_sigma7.jpg")
    assert manifest["outputs"]["blur"].endswith("blur_gaussian_k3_sigma1.jpg")
