from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


REQUIRED_MODEL_ARTIFACTS = (
    "gray_model.xml",
    "color_index.npz",
    "label_mapping.json",
    "rerank_config.json",
)


def build_submission_package(
    *,
    model_dir: str | Path,
    output_dir: str | Path,
    template_dir: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
    tar_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(model_dir)
    output_root = Path(output_dir)
    template = Path(template_dir) if template_dir is not None else _default_template_dir()
    _validate_model_dir(source)
    if not template.is_dir():
        raise FileNotFoundError(f"submission template not found: {template}")

    algorithm_dir = output_root / "Algorithm"
    if algorithm_dir.exists():
        shutil.rmtree(algorithm_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, algorithm_dir, ignore=_ignore_template_noise)

    shutil.copy2(source / "gray_model.xml", algorithm_dir / "face_recognizer_model.xml")
    shutil.copy2(source / "color_index.npz", algorithm_dir / "color_index.npz")
    shutil.copy2(source / "label_mapping.json", algorithm_dir / "label_mapping.json")
    if (source / "training_report.json").is_file():
        shutil.copy2(source / "training_report.json", algorithm_dir / "training_report.json")

    config = _read_json(source / "rerank_config.json")
    preprocess_config = _preprocess_config_from_rerank_config(config)
    _write_json(algorithm_dir / "preprocess_config.json", preprocess_config)
    runtime = _runtime_config_from_rerank_config(config, runtime_config)
    _write_json(algorithm_dir / "rerank_runtime_config.json", runtime)

    archive_path = Path(tar_path) if tar_path is not None else None
    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(algorithm_dir, arcname="Algorithm")

    manifest = {
        "algorithm": "RGB-LBPH",
        "algorithm_dir": str(algorithm_dir),
        "artifacts": sorted(path.name for path in algorithm_dir.iterdir()),
        "archive": str(archive_path) if archive_path is not None else None,
    }
    _write_json(output_root / "submission_manifest.json", manifest)
    return manifest


def _default_template_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "submission_template" / "Algorithm"


def _validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    missing = [name for name in REQUIRED_MODEL_ARTIFACTS if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"model directory is missing required artifacts: {', '.join(missing)}")


def _ignore_template_noise(_dir: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in {"__pycache__", ".pytest_cache"}
        or name.endswith(".pyc")
        or name.endswith(".pyo")
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _preprocess_config_from_rerank_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
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
    return {key: config.get(key, value) for key, value in defaults.items()}


def _runtime_config_from_rerank_config(
    config: dict[str, Any],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    runtime = {
        "candidate_top_k": 2,
        "confidence_gate": 70.0,
        "rerank_margin_ratio": 0.0,
        "grid_x": config.get("grid_x", 10),
        "grid_y": config.get("grid_y", 11),
        "color_bins": config.get("color_bins", 8),
    }
    if override:
        runtime.update(override)
    runtime["candidate_top_k"] = int(runtime["candidate_top_k"])
    runtime["confidence_gate"] = float(runtime["confidence_gate"])
    runtime["rerank_margin_ratio"] = float(runtime["rerank_margin_ratio"])
    runtime["grid_x"] = int(runtime["grid_x"])
    runtime["grid_y"] = int(runtime["grid_y"])
    runtime["color_bins"] = int(runtime["color_bins"])
    return runtime
