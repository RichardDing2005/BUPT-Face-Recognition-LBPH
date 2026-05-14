from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import read_manifest, workspace_root, write_manifest
from .evaluate import evaluate_directory
from .predict import load_label_mapping
from .preprocess import BACKEND_COMPAT_PREPROCESS_CONFIG, PreprocessConfig, preprocess_image
from .train import (
    _create_lbph_recognizer,
    _labels_array,
    _resolve_image,
    _training_samples,
    _write_json,
    _write_recognizer,
    build_label_mapping,
    train_lbph,
)
from .training_tools import build_quality_report, write_stage_comparison


ARTIFACT_FILES = [
    "face_recognizer_model.xml",
    "label_mapping.json",
    "preprocess_config.json",
    "training_report.json",
]


def run_training_stage(
    *,
    workspace: str | Path | None = None,
    stage_name: str,
    stage_manifest: str | Path | None = None,
    stage_train_dir: str | Path | None = None,
    algorithm_dir: str | Path = "Algorithm",
    resume_mode: str = "rebuild",
    run_id: str | None = None,
    preprocess_config: PreprocessConfig | None = None,
    radius: int = 2,
    neighbors: int = 8,
    grid_x: int = 7,
    grid_y: int = 7,
    threshold: float | None = None,
    evaluate_after_stage: bool = False,
    test_dir: str | Path = "TestData/Faces_test",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    if resume_mode not in {"rebuild", "update"}:
        raise ValueError("resume_mode must be 'rebuild' or 'update'")
    if stage_manifest is None and stage_train_dir is None:
        raise ValueError("stage_manifest or stage_train_dir is required")

    root = workspace_root(workspace)
    algorithm_root = _resolve_under(root, algorithm_dir)
    algorithm_root.mkdir(parents=True, exist_ok=True)
    state_path = algorithm_root / "training_state.json"
    state = load_training_state(state_path)
    now = _utc_now()
    if not state:
        state = {
            "run_id": run_id or now.replace(":", "").replace("-", ""),
            "created_at": now,
            "updated_at": now,
            "resume_mode": resume_mode,
            "stages": [],
            "active_stage": None,
            "latest_checkpoint": None,
            "label_mapping": {},
            "preprocess_config": {},
            "lbph_params": {},
        }
    elif run_id:
        state["run_id"] = run_id

    config = preprocess_config or BACKEND_COMPAT_PREPROCESS_CONFIG
    lbph_params = {
        "radius": int(radius),
        "neighbors": int(neighbors),
        "grid_x": int(grid_x),
        "grid_y": int(grid_y),
        "threshold": threshold,
    }
    samples = _stage_samples(root, stage_manifest, stage_train_dir)
    if not samples:
        raise ValueError("no stage training samples found")

    stage_index = len(state.get("stages", [])) + 1
    checkpoint_name = f"{stage_index:03d}_{_safe_name(stage_name)}"
    checkpoint_dir = algorithm_root / "checkpoints" / checkpoint_name
    quality_report = build_quality_report(samples)

    try:
        if resume_mode == "rebuild":
            _prepare_checkpoint_dir(checkpoint_dir)
            cumulative_samples = _dedupe_samples(_previous_samples(state) + samples)
            manifest_path = _write_cumulative_manifest(algorithm_root, checkpoint_name, cumulative_samples)
            training_report = train_lbph(
                workspace=root,
                manifest=manifest_path,
                algorithm_dir=checkpoint_dir,
                preprocess_config=config,
                radius=radius,
                neighbors=neighbors,
                grid_x=grid_x,
                grid_y=grid_y,
                threshold=threshold,
            )
            mapping = _read_mapping(checkpoint_dir / "label_mapping.json")
            training_mode = "rebuild"
        else:
            previous_checkpoint = _previous_checkpoint(root, algorithm_root, state)
            _validate_update_compatibility(state, previous_checkpoint, config, lbph_params)
            _prepare_checkpoint_dir(checkpoint_dir)
            mapping = _extend_mapping(_read_mapping(previous_checkpoint / "label_mapping.json"), samples)
            cumulative_samples = _dedupe_samples(_previous_samples(state) + samples)
            training_report = _train_update_stage(
                root=root,
                previous_checkpoint=previous_checkpoint,
                checkpoint_dir=checkpoint_dir,
                samples=samples,
                mapping=mapping,
                preprocess_config=config,
                lbph_params=lbph_params,
            )
            training_mode = "update"

        evaluation = None
        if evaluate_after_stage:
            resolved_test_dir = _resolve_under(root, test_dir)
            if resolved_test_dir.exists():
                evaluation = evaluate_directory(
                    test_dir=resolved_test_dir,
                    algorithm_dir=checkpoint_dir,
                    reports_dir=_resolve_under(root, reports_dir) / "stages" / checkpoint_name,
                    threshold=threshold,
                )
    except Exception:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        raise

    stage_record = {
        "stage_index": stage_index,
        "stage_name": stage_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint": f"checkpoints/{checkpoint_name}",
        "created_at": now,
        "training_mode": training_mode,
        "resume_mode": resume_mode,
        "data_source": {
            "stage_manifest": str(stage_manifest) if stage_manifest is not None else None,
            "stage_train_dir": str(stage_train_dir) if stage_train_dir is not None else None,
        },
        "stage_num_samples": len(samples),
        "cumulative_num_samples": len(cumulative_samples),
        "samples": samples,
        "quality_report": quality_report,
        "training_report": training_report,
        "evaluation": evaluation,
    }
    _write_json(checkpoint_dir / "stage_report.json", stage_record)
    _promote_checkpoint(checkpoint_dir, algorithm_root)

    state["updated_at"] = _utc_now()
    state["resume_mode"] = resume_mode
    state["active_stage"] = stage_name
    state["latest_checkpoint"] = f"checkpoints/{checkpoint_name}"
    state["label_mapping"] = mapping
    state["preprocess_config"] = config.to_dict()
    state["lbph_params"] = lbph_params
    state.setdefault("stages", []).append(stage_record)
    _write_json(state_path, state)
    write_stage_comparison(state["stages"], _resolve_under(root, reports_dir) / "stage_comparison.csv")
    return {
        "run_id": state["run_id"],
        "stage_name": stage_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint_dir": str(checkpoint_dir),
        "training_mode": training_mode,
        "training_report": training_report,
        "quality_report": quality_report,
        "evaluation": evaluation,
        "label_mapping": mapping,
        "state_path": str(state_path),
    }


def load_training_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _train_update_stage(
    *,
    root: Path,
    previous_checkpoint: Path,
    checkpoint_dir: Path,
    samples: list[dict[str, str]],
    mapping: dict[str, dict[str, int | str]],
    preprocess_config: PreprocessConfig,
    lbph_params: dict[str, Any],
) -> dict[str, Any]:
    recognizer = _create_lbph_recognizer(
        int(lbph_params["radius"]),
        int(lbph_params["neighbors"]),
        int(lbph_params["grid_x"]),
        int(lbph_params["grid_y"]),
        lbph_params.get("threshold"),
    )
    model_path = previous_checkpoint / "face_recognizer_model.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"previous LBPH model not found: {model_path}")
    recognizer.read(str(model_path))
    if not hasattr(recognizer, "update"):
        raise RuntimeError("OpenCV LBPH recognizer does not support update")

    faces = []
    labels = []
    statuses: Counter[str] = Counter()
    valid_per_identity: Counter[str] = Counter()
    for sample in samples:
        result = preprocess_image(_resolve_image(root, sample["relative_path"]), preprocess_config)
        statuses[result.status] += 1
        if result.status == "read_failed":
            continue
        faces.append(result.face)
        labels.append(int(mapping["name_to_id"][sample["identity"]]))
        valid_per_identity[sample["identity"]] += 1

    if not faces:
        raise ValueError("no valid update faces found")

    recognizer.update(faces, _labels_array(labels))
    _write_recognizer(recognizer, checkpoint_dir / "face_recognizer_model.xml")
    _write_json(checkpoint_dir / "label_mapping.json", mapping)
    _write_json(checkpoint_dir / "preprocess_config.json", preprocess_config.to_dict())
    report = {
        "created_at": _utc_now(),
        "training_mode": "update",
        "num_samples": len(samples),
        "num_valid_faces": len(faces),
        "num_identities": len(mapping["name_to_id"]),
        "valid_per_identity": dict(valid_per_identity),
        "preprocess_statuses": dict(statuses),
        "lbph_params": lbph_params,
        "previous_model_path": str(model_path),
        "model_path": str(checkpoint_dir / "face_recognizer_model.xml"),
        "label_mapping_path": str(checkpoint_dir / "label_mapping.json"),
        "preprocess_config_path": str(checkpoint_dir / "preprocess_config.json"),
    }
    _write_json(checkpoint_dir / "training_report.json", report)
    return report


def _stage_samples(
    root: Path,
    stage_manifest: str | Path | None,
    stage_train_dir: str | Path | None,
) -> list[dict[str, str]]:
    rows = _training_samples(root, stage_manifest, stage_train_dir or "TestData/Faces_train")
    normalized = []
    for row in rows:
        updated = dict(row)
        updated["split"] = "train"
        normalized.append(updated)
    return normalized


def _write_cumulative_manifest(algorithm_root: Path, checkpoint_name: str, samples: list[dict[str, str]]) -> Path:
    manifest_path = algorithm_root / "staged_manifests" / f"{checkpoint_name}_train_manifest.csv"
    rows = []
    for sample in samples:
        row = dict(sample)
        row["split"] = "train"
        rows.append(row)
    write_manifest(manifest_path, rows)
    return manifest_path


def _previous_samples(state: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for stage in state.get("stages", []):
        rows.extend(dict(sample) for sample in stage.get("samples", []))
    return rows


def _dedupe_samples(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for sample in samples:
        key = (sample.get("relative_path"), sample.get("identity"))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(sample))
    return result


def _previous_checkpoint(root: Path, algorithm_root: Path, state: dict[str, Any]) -> Path:
    relative = state.get("latest_checkpoint")
    if not relative:
        raise ValueError("update mode requires an existing checkpoint")
    checkpoint = Path(relative)
    if not checkpoint.is_absolute():
        checkpoint = algorithm_root / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"previous checkpoint not found: {checkpoint}")
    return checkpoint


def _validate_update_compatibility(
    state: dict[str, Any],
    previous_checkpoint: Path,
    preprocess_config: PreprocessConfig,
    lbph_params: dict[str, Any],
) -> None:
    required = ["face_recognizer_model.xml", "label_mapping.json", "preprocess_config.json"]
    missing = [name for name in required if not (previous_checkpoint / name).exists()]
    if missing:
        raise FileNotFoundError(f"previous checkpoint missing artifacts: {', '.join(missing)}")
    previous_config = state.get("preprocess_config") or json.loads(
        (previous_checkpoint / "preprocess_config.json").read_text(encoding="utf-8")
    )
    if previous_config != preprocess_config.to_dict():
        raise ValueError("update mode requires identical preprocess config")
    previous_params = state.get("lbph_params") or {}
    if _normalized_params(previous_params) != _normalized_params(lbph_params):
        raise ValueError("update mode requires identical LBPH params")


def _extend_mapping(
    mapping: dict[str, dict[str, int | str]],
    samples: list[dict[str, str]],
) -> dict[str, dict[str, int | str]]:
    name_to_id = {str(name): int(label) for name, label in mapping.get("name_to_id", {}).items()}
    next_id = max(name_to_id.values(), default=-1) + 1
    for identity in sorted({sample["identity"] for sample in samples}):
        if identity not in name_to_id:
            name_to_id[identity] = next_id
            next_id += 1
    id_to_name = {str(label): name for name, label in sorted(name_to_id.items(), key=lambda item: item[1])}
    return {"name_to_id": name_to_id, "id_to_name": id_to_name}


def _read_mapping(path: Path) -> dict[str, dict[str, int | str]]:
    loaded = load_label_mapping(path)
    name_to_id = dict(sorted(loaded.name_to_id.items(), key=lambda item: item[1]))
    id_to_name = {str(label): name for label, name in sorted(loaded.id_to_name.items())}
    if not name_to_id and id_to_name:
        name_to_id = {name: int(label) for label, name in id_to_name.items()}
    if not id_to_name and name_to_id:
        id_to_name = {str(label): name for name, label in name_to_id.items()}
    return {"name_to_id": name_to_id, "id_to_name": id_to_name}


def _promote_checkpoint(checkpoint_dir: Path, algorithm_root: Path) -> None:
    for name in ARTIFACT_FILES:
        source = checkpoint_dir / name
        if source.exists():
            shutil.copy2(source, algorithm_root / name)


def _prepare_checkpoint_dir(checkpoint_dir: Path) -> None:
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


def _resolve_under(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe or "stage"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "radius": int(params.get("radius", 2)),
        "neighbors": int(params.get("neighbors", 8)),
        "grid_x": int(params.get("grid_x", 7)),
        "grid_y": int(params.get("grid_y", 7)),
        "threshold": params.get("threshold"),
    }


def _parse_size(value: str) -> tuple[int, int]:
    left, _, right = value.lower().partition("x")
    if not left or not right:
        raise ValueError("resize must use WIDTHxHEIGHT format")
    return int(left), int(right)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PreprocessConfig(
        size=_parse_size(args.resize),
        equalization=args.equalization,
        detect_face=not args.no_detect_face,
        input_adapter=args.input_adapter,
    )
    result = run_training_stage(
        workspace=args.workspace,
        stage_name=args.stage_name,
        stage_manifest=args.stage_manifest,
        stage_train_dir=args.stage_train_dir,
        algorithm_dir=args.algorithm_dir,
        resume_mode=args.resume_mode,
        run_id=args.run_id,
        preprocess_config=config,
        radius=args.radius,
        neighbors=args.neighbors,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        threshold=args.threshold,
        evaluate_after_stage=args.evaluate_after_stage,
        test_dir=args.test_dir,
        reports_dir=args.reports_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a staged LBPH training step.")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--stage-manifest", default=None)
    parser.add_argument("--stage-train-dir", default=None)
    parser.add_argument("--algorithm-dir", default="Algorithm")
    parser.add_argument("--resume-mode", choices=["rebuild", "update"], default="rebuild")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--evaluate-after-stage", action="store_true")
    parser.add_argument("--test-dir", default="TestData/Faces_test")
    parser.add_argument("--reports-dir", default="reports")
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
