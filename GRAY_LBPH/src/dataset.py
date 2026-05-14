from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}

BASE_MANIFEST_FIELDS = [
    "relative_path",
    "identity",
    "split",
    "quality_flag",
    "face_status",
    "width",
    "height",
    "notes",
]

PROVENANCE_FIELDS = [
    "image_id",
    "source_image",
    "source_path",
    "output_path",
    "effect_type",
    "severity_level",
    "effect_params_json",
    "source_manifest",
]

MANIFEST_FIELDS = BASE_MANIFEST_FIELDS + PROVENANCE_FIELDS


def workspace_root(workspace: str | Path | None = None) -> Path:
    return Path(workspace or Path(__file__).resolve().parents[1]).resolve()


def default_manifest_path(workspace: str | Path | None = None) -> Path:
    return workspace_root(workspace) / "metadata" / "manifest.csv"


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "metadata" / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return [_with_default_fields(row) for row in rows]


def write_manifest(path: str | Path, rows: Iterable[dict[str, str]]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [_with_default_fields(row) for row in rows]
    fields = _field_order(normalized)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)
    return manifest_path


def scan_faces_raw(
    workspace: str | Path | None = None,
    *,
    raw_dir: str | Path = "TestData/Faces_raw",
    out_manifest: str | Path = "metadata/manifest.csv",
) -> list[dict[str, str]]:
    root = workspace_root(workspace)
    raw_root = _resolve_under(root, raw_dir)
    rows: list[dict[str, str]] = []
    if not raw_root.exists():
        raise FileNotFoundError(f"raw dataset directory not found: {raw_root}")

    for identity_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        identity = identity_dir.name
        for image_path in sorted(_iter_images(identity_dir)):
            width, height = image_size(image_path)
            rows.append(
                _with_default_fields(
                    {
                        "relative_path": image_path.relative_to(root).as_posix(),
                        "identity": identity,
                        "split": "",
                        "quality_flag": "normal",
                        "face_status": "unprocessed",
                        "width": str(width),
                        "height": str(height),
                        "notes": "",
                    }
                )
            )

    write_manifest(_resolve_under(root, out_manifest), rows)
    return rows


def import_image_generate_manifest(
    source_manifest: str | Path,
    *,
    workspace: str | Path | None = None,
    raw_dir: str | Path = "TestData/Faces_raw",
    out_manifest: str | Path = "metadata/manifest.csv",
    copy_mode: str = "copy",
    identity_from: str = "source_image",
) -> list[dict[str, str]]:
    root = workspace_root(workspace)
    source_manifest_path = Path(source_manifest).resolve()
    raw_root = _resolve_under(root, raw_dir)
    rows: list[dict[str, str]] = []

    with source_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for source_row in reader:
            output_path = Path(source_row.get("output_path", "")).expanduser()
            if not output_path.is_absolute():
                output_path = (source_manifest_path.parent / output_path).resolve()
            if not output_path.exists():
                raise FileNotFoundError(f"generated image not found: {output_path}")

            identity = _identity_from_source(source_row, identity_from)
            image_id = source_row.get("image_id") or output_path.stem
            suffix = output_path.suffix.lower() or ".jpg"
            target_path = raw_root / identity / f"{_safe_name(image_id)}{suffix}"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if copy_mode == "copy":
                shutil.copy2(output_path, target_path)
            elif copy_mode == "reference":
                target_path = output_path
            else:
                raise ValueError("copy_mode must be 'copy' or 'reference'")

            width = source_row.get("width") or ""
            height = source_row.get("height") or ""
            if not width or not height:
                width_value, height_value = image_size(target_path)
                width = str(width_value)
                height = str(height_value)
            relative_path = (
                target_path.relative_to(root).as_posix()
                if _is_relative_to(target_path, root)
                else str(target_path)
            )
            rows.append(
                _with_default_fields(
                    {
                        "relative_path": relative_path,
                        "identity": identity,
                        "split": "",
                        "quality_flag": source_row.get("effect_type") or "normal",
                        "face_status": "unprocessed",
                        "width": str(width),
                        "height": str(height),
                        "notes": "",
                        "image_id": image_id,
                        "source_image": source_row.get("source_image", ""),
                        "source_path": source_row.get("source_path", ""),
                        "output_path": source_row.get("output_path", ""),
                        "effect_type": source_row.get("effect_type", ""),
                        "severity_level": source_row.get("severity_level", "single"),
                        "effect_params_json": source_row.get("effect_params_json", ""),
                        "source_manifest": str(source_manifest_path),
                    }
                )
            )

    write_manifest(_resolve_under(root, out_manifest), rows)
    return rows


def stratified_split(
    workspace: str | Path | None = None,
    *,
    manifest: str | Path = "metadata/manifest.csv",
    train_dir: str | Path = "TestData/Faces_train",
    val_dir: str | Path = "TestData/Faces_val",
    test_dir: str | Path = "TestData/Faces_test",
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
    test_ratio: float = 0.2,
    seed: int = 42,
    copy_files: bool = True,
) -> list[dict[str, str]]:
    root = workspace_root(workspace)
    manifest_path = _resolve_under(root, manifest)
    rows = read_manifest(manifest_path)
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if copy_files:
        _clean_split_dirs(root, train_dir, val_dir, test_dir)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["identity"]].append(row)

    rng = random.Random(seed)
    split_rows: list[dict[str, str]] = []
    for identity in sorted(groups):
        items = sorted(groups[identity], key=lambda item: item["relative_path"])
        rng.shuffle(items)
        counts = _split_counts(len(items), train_ratio, val_ratio, test_ratio)
        splits = (
            ["train"] * counts["train"]
            + ["val"] * counts["val"]
            + ["test"] * counts["test"]
        )
        for row, split in zip(items, splits):
            updated = dict(row)
            updated["split"] = split
            split_rows.append(updated)
            if copy_files:
                _copy_split_file(root, updated, split, train_dir, val_dir, test_dir)

    split_rows.sort(key=lambda item: (item["identity"], item["split"], item["relative_path"]))
    write_manifest(manifest_path, split_rows)
    _write_split_summary(root / "metadata" / "split_summary.json", split_rows, seed)
    return split_rows


def image_size(path: str | Path) -> tuple[int, int]:
    image_path = Path(path)
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


def _iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            yield path


def _with_default_fields(row: dict[str, str]) -> dict[str, str]:
    normalized = {field: "" for field in MANIFEST_FIELDS}
    normalized.update({str(key): "" if value is None else str(value) for key, value in row.items()})
    return normalized


def _field_order(rows: list[dict[str, str]]) -> list[str]:
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in MANIFEST_FIELDS and key not in extras:
                extras.append(key)
    return MANIFEST_FIELDS + extras


def _resolve_under(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _identity_from_source(row: dict[str, str], mode: str) -> str:
    if mode == "source_image":
        source_image = row.get("source_image") or row.get("image_id") or "unknown"
        return _safe_name(Path(source_image).stem)
    if mode == "image_id":
        image_id = row.get("image_id") or "unknown"
        parts = image_id.split("_")
        return _safe_name("_".join(parts[:2]) if len(parts) >= 2 and parts[0] == "person" else parts[0])
    if mode in row:
        return _safe_name(Path(row[mode]).stem)
    raise ValueError(f"unsupported identity source: {mode}")


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value.strip())
    return safe[:120] or "unknown"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    if any(value < 0 for value in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("split ratios must not be negative")


def _split_counts(size: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    if size <= 0:
        return {"train": 0, "val": 0, "test": 0}
    total = train_ratio + val_ratio + test_ratio
    train_count = int(round(size * train_ratio / total))
    val_count = int(round(size * val_ratio / total))
    test_count = size - train_count - val_count

    if size >= 2:
        train_count = max(1, train_count)
        test_count = max(1, test_count)
    if val_ratio > 0 and size >= 3:
        val_count = max(1, val_count)

    while train_count + val_count + test_count > size:
        if val_count > 0:
            val_count -= 1
        elif train_count > test_count and train_count > 1:
            train_count -= 1
        else:
            test_count -= 1
    while train_count + val_count + test_count < size:
        train_count += 1
    return {"train": train_count, "val": val_count, "test": test_count}


def _copy_split_file(
    root: Path,
    row: dict[str, str],
    split: str,
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
) -> None:
    source = Path(row["relative_path"])
    if not source.is_absolute():
        source = root / source
    if not source.exists():
        raise FileNotFoundError(f"manifest image not found: {source}")
    output_root = {
        "train": _resolve_under(root, train_dir),
        "val": _resolve_under(root, val_dir),
        "test": _resolve_under(root, test_dir),
    }[split]
    target = output_root / _safe_name(row["identity"]) / _relative_identity_path(root, source, row["identity"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _clean_split_dirs(root: Path, *dirs: str | Path) -> None:
    for directory in dirs:
        target = _resolve_under(root, directory)
        if not _is_relative_to(target, root):
            raise ValueError(f"refusing to clean split directory outside workspace: {target}")
        if target.exists():
            shutil.rmtree(target)


def _relative_identity_path(root: Path, source: Path, identity: str) -> Path:
    try:
        relative = source.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(source.name)
    parts = list(relative.parts)
    try:
        identity_index = parts.index(identity)
    except ValueError:
        return Path(source.name)
    tail = parts[identity_index + 1 :]
    return Path(*tail) if tail else Path(source.name)


def _write_split_summary(path: Path, rows: list[dict[str, str]], seed: int) -> None:
    import json

    summary: dict[str, dict[str, int] | int] = {"seed": int(seed)}
    counts: dict[str, int] = defaultdict(int)
    per_identity: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["split"]] += 1
        per_identity[row["identity"]][row["split"]] += 1
    summary["counts"] = dict(counts)
    summary["per_identity"] = {key: dict(value) for key, value in per_identity.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
