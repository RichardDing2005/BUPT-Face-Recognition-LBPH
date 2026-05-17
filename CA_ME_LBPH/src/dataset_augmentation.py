from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.augmentation_preview import (
    add_gaussian_noise,
    apply_gaussian_blur,
    _number_token,
    _odd_kernel_size,
    _write_jpeg,
)
from src.dataset import SUPPORTED_IMAGE_SUFFIXES


DEFAULT_SOURCE_DIR = Path("datasets/score2026/Faces_raw")
DEFAULT_OUTPUT_DIR = Path("datasets/score2026_aug_light/Faces_raw")
DEFAULT_MANIFEST_PATH = Path("datasets/score2026_aug_light/augmentation_manifest.csv")
DEFAULT_REPORT_PATH = Path("datasets/score2026_aug_light/dataset_report.json")

MANIFEST_FIELDS = [
    "source_path",
    "output_path",
    "identity",
    "source_filename",
    "output_filename",
    "augmentation_type",
    "noise_sigma",
    "noise_seed",
    "blur_kernel_size",
    "blur_sigma",
    "jpeg_quality",
]


@dataclass(frozen=True)
class DatasetAugmentationConfig:
    source_dir: str | Path = DEFAULT_SOURCE_DIR
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH
    report_path: str | Path = DEFAULT_REPORT_PATH
    noise_sigma: float = 7.0
    base_seed: int = 42
    blur_kernel_size: int = 3
    blur_sigma: float = 1.0
    jpeg_quality: int = 95
    overwrite: bool = True


def augment_dataset(config: DatasetAugmentationConfig) -> dict[str, Any]:
    cv2 = _require_cv2()
    source_dir = Path(config.source_dir).resolve()
    output_dir = Path(config.output_dir).resolve()
    manifest_path = Path(config.manifest_path).resolve()
    report_path = Path(config.report_path).resolve()
    _validate_paths(source_dir, output_dir, manifest_path, report_path)

    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"output_dir already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    per_identity: dict[str, dict[str, int]] = {}
    output_count = 0

    identity_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    for identity_dir in identity_dirs:
        identity = identity_dir.name
        target_identity_dir = output_dir / identity
        target_identity_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {
            "source_images": 0,
            "original": 0,
            "noise_gaussian": 0,
            "blur_gaussian": 0,
            "total": 0,
        }

        for source_image in _iter_identity_images(identity_dir):
            relative_source = source_image.relative_to(source_dir)
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read source image: {source_image}")

            stem = source_image.stem
            seed = derive_image_seed(relative_source, base_seed=int(config.base_seed))
            original_path = target_identity_dir / source_image.name
            noise_path = target_identity_dir / (
                f"{stem}__noise_gaussian_sigma{_number_token(float(config.noise_sigma))}.jpg"
            )
            blur_path = target_identity_dir / (
                f"{stem}__blur_gaussian_k{_odd_kernel_size(int(config.blur_kernel_size))}_"
                f"sigma{_number_token(float(config.blur_sigma))}.jpg"
            )

            shutil.copy2(source_image, original_path)
            manifest_rows.append(
                _manifest_row(
                    source_image=source_image,
                    output_image=original_path,
                    identity=identity,
                    augmentation_type="original",
                    noise_sigma=config.noise_sigma,
                    noise_seed=seed,
                    blur_kernel_size=config.blur_kernel_size,
                    blur_sigma=config.blur_sigma,
                    jpeg_quality=config.jpeg_quality,
                )
            )
            counts["original"] += 1
            output_count += 1

            _write_jpeg(
                noise_path,
                add_gaussian_noise(image, sigma=float(config.noise_sigma), seed=seed),
                quality=int(config.jpeg_quality),
            )
            manifest_rows.append(
                _manifest_row(
                    source_image=source_image,
                    output_image=noise_path,
                    identity=identity,
                    augmentation_type="noise_gaussian",
                    noise_sigma=config.noise_sigma,
                    noise_seed=seed,
                    blur_kernel_size=config.blur_kernel_size,
                    blur_sigma=config.blur_sigma,
                    jpeg_quality=config.jpeg_quality,
                )
            )
            counts["noise_gaussian"] += 1
            output_count += 1

            _write_jpeg(
                blur_path,
                apply_gaussian_blur(
                    image,
                    kernel_size=int(config.blur_kernel_size),
                    sigma=float(config.blur_sigma),
                ),
                quality=int(config.jpeg_quality),
            )
            manifest_rows.append(
                _manifest_row(
                    source_image=source_image,
                    output_image=blur_path,
                    identity=identity,
                    augmentation_type="blur_gaussian",
                    noise_sigma=config.noise_sigma,
                    noise_seed=seed,
                    blur_kernel_size=config.blur_kernel_size,
                    blur_sigma=config.blur_sigma,
                    jpeg_quality=config.jpeg_quality,
                )
            )
            counts["blur_gaussian"] += 1
            output_count += 1

            counts["source_images"] += 1
            counts["total"] += 3

        per_identity[identity] = counts

    _write_manifest(manifest_path, manifest_rows)
    source_count = sum(item["source_images"] for item in per_identity.values())
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "parameters": {
            "noise": {
                "type": "gaussian",
                "sigma": float(config.noise_sigma),
                "base_seed": int(config.base_seed),
            },
            "blur": {
                "type": "gaussian",
                "kernel_size": int(_odd_kernel_size(int(config.blur_kernel_size))),
                "sigma": float(config.blur_sigma),
            },
            "jpeg_quality": int(config.jpeg_quality),
        },
        "num_identities": len(per_identity),
        "num_source_images": int(source_count),
        "num_output_images": int(output_count),
        "expected_output_images": int(source_count * 3),
        "per_identity": per_identity,
        "failures": [],
        "valid": True,
    }
    report["failures"] = _report_failures(report)
    report["valid"] = not report["failures"]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["failures"]:
        raise ValueError(f"dataset augmentation integrity check failed: {report['failures']}")
    return report


def derive_image_seed(relative_path: str | Path, *, base_seed: int = 42) -> int:
    normalized = Path(relative_path).as_posix()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (int(base_seed) + offset) % (2**32)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = augment_dataset(
        DatasetAugmentationConfig(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            report_path=args.report_path,
            noise_sigma=args.noise_sigma,
            base_seed=args.base_seed,
            blur_kernel_size=args.blur_kernel_size,
            blur_sigma=args.blur_sigma,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a label-preserving light-augmented score2026 dataset.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--noise-sigma", type=float, default=7.0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--blur-kernel-size", type=int, default=3)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_paths(source_dir: Path, output_dir: Path, manifest_path: Path, report_path: Path) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source dataset directory not found: {source_dir}")
    if output_dir == source_dir or _is_relative_to(output_dir, source_dir):
        raise ValueError("output_dir must be separate from source_dir")
    for path, name in ((manifest_path, "manifest_path"), (report_path, "report_path")):
        if path == source_dir or _is_relative_to(path, source_dir):
            raise ValueError(f"{name} must not be inside source_dir")


def _iter_identity_images(identity_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in identity_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _manifest_row(
    *,
    source_image: Path,
    output_image: Path,
    identity: str,
    augmentation_type: str,
    noise_sigma: float,
    noise_seed: int,
    blur_kernel_size: int,
    blur_sigma: float,
    jpeg_quality: int,
) -> dict[str, str]:
    return {
        "source_path": str(source_image.resolve()),
        "output_path": str(output_image.resolve()),
        "identity": identity,
        "source_filename": source_image.name,
        "output_filename": output_image.name,
        "augmentation_type": augmentation_type,
        "noise_sigma": str(float(noise_sigma)),
        "noise_seed": str(int(noise_seed)),
        "blur_kernel_size": str(int(_odd_kernel_size(int(blur_kernel_size)))),
        "blur_sigma": str(float(blur_sigma)),
        "jpeg_quality": str(int(jpeg_quality)),
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _report_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if report["num_output_images"] != report["expected_output_images"]:
        failures.append("total_output_count_mismatch")
    per_identity: dict[str, dict[str, int]] = report["per_identity"]
    for identity, counts in sorted(per_identity.items()):
        expected = counts["source_images"]
        if counts["original"] != expected:
            failures.append(f"{identity}:original_count_mismatch")
        if counts["noise_gaussian"] != expected:
            failures.append(f"{identity}:noise_count_mismatch")
        if counts["blur_gaussian"] != expected:
            failures.append(f"{identity}:blur_count_mismatch")
        if counts["total"] != expected * 3:
            failures.append(f"{identity}:total_count_mismatch")
    return failures


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _require_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for dataset augmentation") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
