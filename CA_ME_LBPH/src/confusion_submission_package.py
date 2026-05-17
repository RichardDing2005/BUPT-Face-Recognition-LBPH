from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .benchmark_pack import create_score2026_submission_tar
from .confusion_rerank import ConfusionRerankConfig, RerankRuntimeConfig


REQUIRED_MODEL_ARTIFACTS = (
    "gray_model.xml",
    "gray_aux_model.xml",
    "evidence_index.npz",
    "label_mapping.json",
    "rerank_config.json",
)


def build_submission_package(
    *,
    model_dir: str | Path,
    output_dir: str | Path,
    template_dir: str | Path | None = None,
    runtime_config: dict[str, Any] | RerankRuntimeConfig | None = None,
    tar_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(model_dir).resolve()
    output_root = Path(output_dir).resolve()
    template = Path(template_dir).resolve() if template_dir is not None else _default_template_dir()
    _validate_model_dir(source)
    if not template.is_dir():
        raise FileNotFoundError(f"submission template not found: {template}")

    algorithm_dir = output_root / "Algorithm"
    if algorithm_dir.exists():
        shutil.rmtree(algorithm_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, algorithm_dir, ignore=_ignore_template_noise)

    shutil.copy2(source / "gray_model.xml", algorithm_dir / "face_recognizer_model.xml")
    shutil.copy2(source / "gray_aux_model.xml", algorithm_dir / "gray_aux_model.xml")
    shutil.copy2(source / "evidence_index.npz", algorithm_dir / "evidence_index.npz")
    shutil.copy2(source / "label_mapping.json", algorithm_dir / "label_mapping.json")
    if (source / "training_report.json").is_file():
        shutil.copy2(source / "training_report.json", algorithm_dir / "training_report.json")

    config = ConfusionRerankConfig.from_dict(_read_json(source / "rerank_config.json"))
    _write_json(algorithm_dir / "preprocess_config.json", config.primary_preprocess_config().to_dict())
    runtime = _runtime_config(runtime_config)
    runtime_payload = runtime.to_dict()
    runtime_payload.update(
        {
            "aux_size": [int(config.aux_size[0]), int(config.aux_size[1])],
            "equalization": config.equalization,
            "color_bins": int(config.color_bins),
            "texture_bins": int(config.texture_bins),
            "grid_x": int(config.grid_x),
            "grid_y": int(config.grid_y),
            "texture_grid_x": int(config.texture_grid_x),
            "texture_grid_y": int(config.texture_grid_y),
        }
    )
    _write_json(algorithm_dir / "rerank_runtime_config.json", runtime_payload)

    archive_target = Path(tar_path) if tar_path is not None else output_root / "Algorithm.tar.gz"
    archive_path = create_score2026_submission_tar(output_root, output=archive_target)
    manifest = {
        "algorithm": "CA-ME-LBPH",
        "algorithm_dir": str(algorithm_dir),
        "artifacts": sorted(path.name for path in algorithm_dir.iterdir()),
        "archive": str(archive_path) if archive_path is not None else None,
    }
    _write_json(output_root / "submission_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime = RerankRuntimeConfig(
        candidate_top_k=args.candidate_top_k,
        confidence_gate=args.confidence_gate,
        gray_margin_gate=args.gray_margin_gate,
        switch_margin=args.switch_margin,
    )
    manifest = build_submission_package(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        template_dir=args.template_dir,
        runtime_config=runtime,
        tar_path=args.tar_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CA-ME-LBPH score2026 submission package.")
    parser.add_argument("--model-dir", default="Algorithm_score2026_confusion_rerank_full")
    parser.add_argument(
        "--output-dir",
        default="submission_build/confusion_rerank",
    )
    parser.add_argument("--template-dir", default=None)
    parser.add_argument("--tar-path", default=None)
    parser.add_argument("--candidate-top-k", type=int, default=4)
    parser.add_argument("--confidence-gate", type=float, default=60.0)
    parser.add_argument("--gray-margin-gate", type=float, default=65.0)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    return parser


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
        if name in {"__pycache__", ".pytest_cache"} or name.endswith(".pyc") or name.endswith(".pyo")
    }


def _runtime_config(value: dict[str, Any] | RerankRuntimeConfig | None) -> RerankRuntimeConfig:
    if isinstance(value, RerankRuntimeConfig):
        return value
    return RerankRuntimeConfig.from_dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
