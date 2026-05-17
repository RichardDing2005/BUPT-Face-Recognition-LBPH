from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import SUPPORTED_IMAGE_SUFFIXES, read_manifest, workspace_root
from .preprocess import BACKEND_COMPAT_PREPROCESS_CONFIG, PreprocessConfig, preprocess_image


def build_label_mapping(identities: list[str]) -> dict[str, dict[str, int | str]]:
    names = sorted({str(identity) for identity in identities})
    name_to_id = {name: index for index, name in enumerate(names)}
    id_to_name = {str(index): name for name, index in name_to_id.items()}
    return {"name_to_id": name_to_id, "id_to_name": id_to_name}


def train_lbph(
    *,
    workspace: str | Path | None = None,
    manifest: str | Path | None = None,
    train_dir: str | Path = "TestData/Faces_train",
    algorithm_dir: str | Path = "Algorithm",
    model_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    preprocess_config: PreprocessConfig | None = None,
    radius: int = 2,
    neighbors: int = 8,
    grid_x: int = 7,
    grid_y: int = 7,
    threshold: float | None = None,
) -> dict[str, Any]:
    root = workspace_root(workspace)
    algorithm_root = _resolve_under(root, algorithm_dir)
    algorithm_root.mkdir(parents=True, exist_ok=True)
    model_file = _resolve_under(root, model_path) if model_path else algorithm_root / "face_recognizer_model.xml"
    mapping_file = _resolve_under(root, mapping_path) if mapping_path else algorithm_root / "label_mapping.json"
    config = preprocess_config or BACKEND_COMPAT_PREPROCESS_CONFIG

    samples = _training_samples(root, manifest, train_dir)
    if not samples:
        raise ValueError("no training samples found")

    mapping = build_label_mapping([sample["identity"] for sample in samples])
    faces = []
    labels = []
    statuses: Counter[str] = Counter()
    valid_per_identity: Counter[str] = Counter()
    for sample in samples:
        result = preprocess_image(_resolve_image(root, sample["relative_path"]), config)
        statuses[result.status] += 1
        if result.status == "read_failed":
            continue
        faces.append(result.face)
        label = int(mapping["name_to_id"][sample["identity"]])
        labels.append(label)
        valid_per_identity[sample["identity"]] += 1

    if not faces:
        raise ValueError("no valid training faces found")

    recognizer = _create_lbph_recognizer(radius, neighbors, grid_x, grid_y, threshold)
    recognizer.train(faces, _labels_array(labels))
    _write_recognizer(recognizer, model_file)
    _write_json(mapping_file, mapping)
    config_file = algorithm_root / "preprocess_config.json"
    _write_json(config_file, config.to_dict())
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(samples),
        "num_valid_faces": len(faces),
        "num_identities": len(mapping["name_to_id"]),
        "valid_per_identity": dict(valid_per_identity),
        "preprocess_statuses": dict(statuses),
        "lbph_params": {
            "radius": int(radius),
            "neighbors": int(neighbors),
            "grid_x": int(grid_x),
            "grid_y": int(grid_y),
            "threshold": threshold,
        },
        "model_path": str(model_file),
        "label_mapping_path": str(mapping_file),
        "preprocess_config_path": str(config_file),
    }
    report_file = algorithm_root / "training_report.json"
    _write_json(report_file, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PreprocessConfig(
        size=_parse_size(args.resize),
        equalization=args.equalization,
        detect_face=not args.no_detect_face,
        input_adapter=args.input_adapter,
    )
    report = train_lbph(
        workspace=args.workspace,
        manifest=args.manifest,
        train_dir=args.train_dir,
        algorithm_dir=args.algorithm_dir,
        model_path=args.model,
        mapping_path=args.mapping,
        preprocess_config=config,
        radius=args.radius,
        neighbors=args.neighbors,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        threshold=args.threshold,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _training_samples(root: Path, manifest: str | Path | None, train_dir: str | Path) -> list[dict[str, str]]:
    if manifest:
        rows = read_manifest(_resolve_under(root, manifest))
        selected = [row for row in rows if (row.get("split") or "train") == "train"]
        return selected

    train_root = _resolve_under(root, train_dir)
    rows: list[dict[str, str]] = []
    if not train_root.exists():
        return rows
    for identity_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in identity_dir.rglob("*") if path.is_file()):
            if image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                rows.append(
                    {
                        "relative_path": _path_for_manifest(root, image_path),
                        "identity": identity_dir.name,
                    }
                )
    return rows


def _create_lbph_recognizer(radius: int, neighbors: int, grid_x: int, grid_y: int, threshold: float | None) -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required; install opencv-contrib-python") from exc
    try:
        kwargs = {
            "radius": int(radius),
            "neighbors": int(neighbors),
            "grid_x": int(grid_x),
            "grid_y": int(grid_y),
        }
        if threshold is not None:
            kwargs["threshold"] = float(threshold)
        return cv2.face.LBPHFaceRecognizer_create(**kwargs)
    except AttributeError as exc:
        raise ModuleNotFoundError("OpenCV LBPH requires opencv-contrib-python") from exc


def _labels_array(labels: list[int]) -> Any:
    try:
        import numpy as np

        return np.array(labels, dtype="int32")
    except ModuleNotFoundError:
        return labels


def _write_recognizer(recognizer: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(recognizer, "write"):
        recognizer.write(str(path))
    else:
        recognizer.save(str(path))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_under(root: Path, path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("path is required")
    value = Path(path)
    return value if value.is_absolute() else root / value


def _resolve_image(root: Path, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _path_for_manifest(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _parse_size(value: str) -> tuple[int, int]:
    left, _, right = value.lower().partition("x")
    if not left or not right:
        raise ValueError("resize must use WIDTHxHEIGHT format")
    return int(left), int(right)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an OpenCV LBPH face recognizer.")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--train-dir", default="TestData/Faces_train")
    parser.add_argument("--algorithm-dir", default="Algorithm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mapping", default=None)
    default_width, default_height = BACKEND_COMPAT_PREPROCESS_CONFIG.size
    parser.add_argument("--resize", default=f"{default_width}x{default_height}")
    parser.add_argument(
        "--equalization",
        default=BACKEND_COMPAT_PREPROCESS_CONFIG.equalization,
        choices=["none", "hist", "equalizeHist", "clahe"],
    )
    parser.add_argument("--no-detect-face", action="store_true")
    parser.add_argument("--input-adapter", default=BACKEND_COMPAT_PREPROCESS_CONFIG.input_adapter)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--grid-x", type=int, default=7)
    parser.add_argument("--grid-y", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
