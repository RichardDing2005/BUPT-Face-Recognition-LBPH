from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, value: int) -> None:
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((18, 20, 3), value, dtype=np.uint8)
    image[:, :, 1] = np.arange(20, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_augment_dataset_refuses_output_equal_to_source(tmp_path: Path) -> None:
    from src.dataset_augmentation import DatasetAugmentationConfig, augment_dataset

    source = tmp_path / "Faces_raw"
    _write_image(source / "2024210809" / "1.jpg", 120)

    with pytest.raises(ValueError, match="output_dir must be separate"):
        augment_dataset(
            DatasetAugmentationConfig(
                source_dir=source,
                output_dir=source,
                manifest_path=tmp_path / "augmentation_manifest.csv",
                report_path=tmp_path / "dataset_report.json",
            )
        )


def test_augment_dataset_preserves_labels_and_writes_manifest_report(tmp_path: Path) -> None:
    from src.dataset_augmentation import DatasetAugmentationConfig, augment_dataset

    source = tmp_path / "score2026" / "Faces_raw"
    _write_image(source / "2024210809" / "1.jpg", 120)
    _write_image(source / "2024210809" / "2.jpg", 130)
    _write_image(source / "2024210776" / "person_named_differently.jpg", 140)

    output = tmp_path / "score2026_aug_light" / "Faces_raw"
    manifest_path = tmp_path / "score2026_aug_light" / "augmentation_manifest.csv"
    report_path = tmp_path / "score2026_aug_light" / "dataset_report.json"
    report = augment_dataset(
        DatasetAugmentationConfig(
            source_dir=source,
            output_dir=output,
            manifest_path=manifest_path,
            report_path=report_path,
            noise_sigma=7.0,
            base_seed=42,
            blur_kernel_size=3,
            blur_sigma=1.0,
            jpeg_quality=95,
        )
    )

    assert sorted(path.name for path in output.iterdir() if path.is_dir()) == [
        "2024210776",
        "2024210809",
    ]
    assert (output / "2024210809" / "1.jpg").is_file()
    assert (output / "2024210809" / "1__noise_gaussian_sigma7.jpg").is_file()
    assert (output / "2024210809" / "1__blur_gaussian_k3_sigma1.jpg").is_file()
    assert (output / "2024210776" / "person_named_differently.jpg").is_file()
    assert (output / "2024210776" / "person_named_differently__noise_gaussian_sigma7.jpg").is_file()
    assert (output / "2024210776" / "person_named_differently__blur_gaussian_k3_sigma1.jpg").is_file()

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert {row["identity"] for row in rows} == {"2024210809", "2024210776"}
    assert {row["augmentation_type"] for row in rows} == {"original", "noise_gaussian", "blur_gaussian"}
    assert all(row["jpeg_quality"] == "95" for row in rows)
    assert all(row["noise_sigma"] == "7.0" for row in rows)
    assert all(row["blur_sigma"] == "1.0" for row in rows)

    per_identity = report["per_identity"]
    assert report["num_identities"] == 2
    assert report["num_source_images"] == 3
    assert report["num_output_images"] == 9
    assert report["valid"] is True
    assert per_identity["2024210809"]["source_images"] == 2
    assert per_identity["2024210809"]["original"] == 2
    assert per_identity["2024210809"]["noise_gaussian"] == 2
    assert per_identity["2024210809"]["blur_gaussian"] == 2
    assert per_identity["2024210776"]["source_images"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_seed_derivation_and_noise_outputs_are_reproducible(tmp_path: Path) -> None:
    from src.dataset_augmentation import (
        DatasetAugmentationConfig,
        augment_dataset,
        derive_image_seed,
    )

    source = tmp_path / "source" / "Faces_raw"
    _write_image(source / "2024210809" / "1.jpg", 110)

    assert derive_image_seed(Path("2024210809/1.jpg"), base_seed=42) == derive_image_seed(
        Path("2024210809/1.jpg"), base_seed=42
    )
    assert derive_image_seed(Path("2024210809/1.jpg"), base_seed=42) != derive_image_seed(
        Path("2024210809/2.jpg"), base_seed=42
    )

    first_output = tmp_path / "first" / "Faces_raw"
    second_output = tmp_path / "second" / "Faces_raw"
    for output in (first_output, second_output):
        augment_dataset(
            DatasetAugmentationConfig(
                source_dir=source,
                output_dir=output,
                manifest_path=output.parent / "augmentation_manifest.csv",
                report_path=output.parent / "dataset_report.json",
                noise_sigma=7.0,
                base_seed=42,
            )
        )

    first_noise = first_output / "2024210809" / "1__noise_gaussian_sigma7.jpg"
    second_noise = second_output / "2024210809" / "1__noise_gaussian_sigma7.jpg"
    assert first_noise.read_bytes() == second_noise.read_bytes()
