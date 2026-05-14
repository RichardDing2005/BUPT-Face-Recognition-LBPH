from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.evaluate import evaluate_directory
from src.preprocess import PreprocessConfig
from src.train import train_lbph


BASELINE_ACCURACY = 0.9617137648131268
BASELINE_MACRO_F1 = 0.9618999170520551
DEFAULT_TRAIN_DIR = Path("datasets/score2026/Faces_train")
DEFAULT_TEST_DIR = Path("datasets/score2026/Faces_test")
DEFAULT_EXPERIMENTS_DIR = Path("experiments/score2026_v2")
DEFAULT_FINAL_DIR = Path("Algorithm_score2026_v2")
DEFAULT_LBPH_PARAMS = {"radius": 2, "neighbors": 8, "grid_x": 7, "grid_y": 7}
DEFAULT_MAX_MODEL_MIB = 1024.0


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    stage: str
    preprocess_name: str
    preprocess_config: PreprocessConfig
    lbph_params: dict[str, int]


PREPROCESS_CANDIDATES: list[tuple[str, PreprocessConfig]] = [
    (
        "baseline",
        PreprocessConfig(
            size=(200, 200),
            detect_face=True,
            equalization="clahe",
            margin_ratio=0.15,
            min_face_size=(40, 40),
            scale_factor=1.1,
            min_neighbors=5,
            fallback_to_full_image=True,
        ),
    ),
    (
        "relaxed-1",
        PreprocessConfig(
            size=(200, 200),
            detect_face=True,
            equalization="clahe",
            margin_ratio=0.15,
            min_face_size=(40, 40),
            scale_factor=1.1,
            min_neighbors=3,
            fallback_to_full_image=True,
        ),
    ),
    (
        "relaxed-2",
        PreprocessConfig(
            size=(200, 200),
            detect_face=True,
            equalization="clahe",
            margin_ratio=0.15,
            min_face_size=(30, 30),
            scale_factor=1.1,
            min_neighbors=3,
            fallback_to_full_image=True,
        ),
    ),
    (
        "relaxed-3",
        PreprocessConfig(
            size=(200, 200),
            detect_face=True,
            equalization="clahe",
            margin_ratio=0.15,
            min_face_size=(30, 30),
            scale_factor=1.05,
            min_neighbors=3,
            fallback_to_full_image=True,
        ),
    ),
    (
        "relaxed-4",
        PreprocessConfig(
            size=(200, 200),
            detect_face=True,
            equalization="clahe",
            margin_ratio=0.15,
            min_face_size=(30, 30),
            scale_factor=1.1,
            min_neighbors=2,
            fallback_to_full_image=True,
        ),
    ),
    (
        "margin-low",
        PreprocessConfig(
            size=(200, 200),
            detect_face=True,
            equalization="clahe",
            margin_ratio=0.10,
            min_face_size=(40, 40),
            scale_factor=1.1,
            min_neighbors=5,
            fallback_to_full_image=True,
        ),
    ),
    (
        "margin-high",
        PreprocessConfig(
            size=(200, 200),
            detect_face=True,
            equalization="clahe",
            margin_ratio=0.20,
            min_face_size=(40, 40),
            scale_factor=1.1,
            min_neighbors=5,
            fallback_to_full_image=True,
        ),
    ),
    (
        "full-image",
        PreprocessConfig(
            size=(200, 200),
            detect_face=False,
            equalization="clahe",
            margin_ratio=0.15,
            min_face_size=(40, 40),
            scale_factor=1.1,
            min_neighbors=5,
            fallback_to_full_image=True,
        ),
    ),
]

LBPH_PARAM_CANDIDATES: list[dict[str, int]] = [
    {"radius": 1, "neighbors": 8, "grid_x": 8, "grid_y": 8},
    {"radius": 1, "neighbors": 12, "grid_x": 8, "grid_y": 8},
    {"radius": 1, "neighbors": 8, "grid_x": 10, "grid_y": 10},
    {"radius": 2, "neighbors": 8, "grid_x": 7, "grid_y": 7},
    {"radius": 2, "neighbors": 8, "grid_x": 8, "grid_y": 8},
    {"radius": 2, "neighbors": 12, "grid_x": 8, "grid_y": 8},
    {"radius": 2, "neighbors": 16, "grid_x": 8, "grid_y": 8},
    {"radius": 3, "neighbors": 8, "grid_x": 8, "grid_y": 8},
    {"radius": 3, "neighbors": 12, "grid_x": 8, "grid_y": 8},
]


def build_preprocess_run_configs() -> list[RunConfig]:
    runs = []
    for index, (name, config) in enumerate(PREPROCESS_CANDIDATES, start=1):
        runs.append(
            RunConfig(
                run_id=f"preprocess_{index:02d}_{name}",
                stage="preprocess",
                preprocess_name=name,
                preprocess_config=config,
                lbph_params=dict(DEFAULT_LBPH_PARAMS),
            )
        )
    return runs


def build_lbph_run_configs(preprocess_names: Iterable[str]) -> list[RunConfig]:
    preprocess_by_name = dict(PREPROCESS_CANDIDATES)
    runs = []
    run_index = 1
    for preprocess_name in preprocess_names:
        config = preprocess_by_name[preprocess_name]
        for params in LBPH_PARAM_CANDIDATES:
            params_label = (
                f"r{params['radius']}_n{params['neighbors']}_"
                f"g{params['grid_x']}x{params['grid_y']}"
            )
            runs.append(
                RunConfig(
                    run_id=f"lbph_{run_index:02d}_{preprocess_name}_{params_label}",
                    stage="lbph",
                    preprocess_name=preprocess_name,
                    preprocess_config=config,
                    lbph_params=dict(params),
                )
            )
            run_index += 1
    return runs


def sort_run_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            float(record.get("overall_accuracy", 0.0)),
            float(record.get("macro_f1", 0.0)),
            -int(record.get("num_failed_preprocess", 0)),
            -int(record.get("error_count", 0)),
        ),
        reverse=True,
    )


def remove_model_file(algorithm_dir: str | Path, experiments_root: str | Path) -> bool:
    experiments = Path(experiments_root).resolve()
    model_path = (Path(algorithm_dir) / "face_recognizer_model.xml").resolve()
    try:
        model_path.relative_to(experiments)
    except ValueError as exc:
        raise ValueError(f"refusing to remove model outside experiments: {model_path}") from exc
    if not model_path.exists():
        return False
    model_path.unlink()
    return True


def run_score2026_optimization(
    *,
    workspace: str | Path,
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    test_dir: str | Path = DEFAULT_TEST_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    final_dir: str | Path = DEFAULT_FINAL_DIR,
    baseline_accuracy: float = BASELINE_ACCURACY,
    baseline_macro_f1: float = BASELINE_MACRO_F1,
    min_accuracy_gain: float = 0.005,
    max_model_mib: float = DEFAULT_MAX_MODEL_MIB,
    keep_experiment_models: bool = False,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    experiments_root = _resolve_under(root, experiments_dir)
    final_root = _resolve_under(root, final_dir)
    train_root = _resolve_under(root, train_dir)
    test_root = _resolve_under(root, test_dir)
    _validate_inputs(train_root, test_root, experiments_root, final_root)

    experiments_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    preprocess_runs = build_preprocess_run_configs()
    for run_config in preprocess_runs:
        record = _execute_run(root, train_root, test_root, experiments_root, run_config, max_model_mib)
        records.append(record)
        best_record = _update_best_model(record, best_record, experiments_root, keep_experiment_models)
        _write_summary(experiments_root, records, best_record, None, baseline_accuracy, baseline_macro_f1)

    top_preprocess_names = _top_preprocess_names(records, count=2)
    for run_config in build_lbph_run_configs(top_preprocess_names):
        record = _execute_run(root, train_root, test_root, experiments_root, run_config, max_model_mib)
        records.append(record)
        best_record = _update_best_model(record, best_record, experiments_root, keep_experiment_models)
        _write_summary(experiments_root, records, best_record, None, baseline_accuracy, baseline_macro_f1)

    sorted_records = sort_run_records(records)
    best_record = sorted_records[0] if sorted_records else None
    promoted = False
    recommendation = "keep Algorithm_score2026"
    if best_record is not None and _is_promotable(best_record, baseline_accuracy, baseline_macro_f1, min_accuracy_gain):
        _promote_best_model(best_record, final_root)
        promoted = True
        recommendation = str(final_root)

    if not keep_experiment_models:
        for record in records:
            removed = remove_model_file(record["algorithm_dir"], experiments_root)
            record["model_removed"] = bool(removed or record.get("model_removed"))

    result = _write_summary(
        experiments_root,
        records,
        best_record,
        {
            "promoted": promoted,
            "recommendation": recommendation,
            "final_dir": str(final_root) if promoted else None,
        },
        baseline_accuracy,
        baseline_macro_f1,
    )
    return result


def _execute_run(
    root: Path,
    train_root: Path,
    test_root: Path,
    experiments_root: Path,
    run_config: RunConfig,
    max_model_mib: float,
) -> dict[str, Any]:
    run_root = experiments_root / "runs" / run_config.run_id
    algorithm_dir = run_root / "Algorithm"
    reports_dir = run_root / "reports"
    if run_root.exists():
        raise FileExistsError(f"experiment run already exists: {run_root}")
    print(f"[score2026-v2] training {run_config.run_id}", flush=True)
    params = run_config.lbph_params
    training_report = train_lbph(
        workspace=root,
        train_dir=train_root,
        algorithm_dir=algorithm_dir,
        preprocess_config=run_config.preprocess_config,
        radius=params["radius"],
        neighbors=params["neighbors"],
        grid_x=params["grid_x"],
        grid_y=params["grid_y"],
    )
    model_path = algorithm_dir / "face_recognizer_model.xml"
    model_bytes = model_path.stat().st_size if model_path.exists() else 0
    max_model_bytes = int(float(max_model_mib) * 1024 * 1024)
    if max_model_bytes > 0 and model_bytes > max_model_bytes:
        record = build_model_too_large_record(
            run_config=run_config,
            training_report=training_report,
            algorithm_dir=algorithm_dir,
            reports_dir=reports_dir,
            test_root=test_root,
            model_bytes=model_bytes,
            max_model_bytes=max_model_bytes,
        )
        remove_model_file(algorithm_dir, experiments_root)
        _write_json(run_root / "run_config.json", _run_config_payload(run_config))
        _write_json(run_root / "run_summary.json", record)
        print(
            "[score2026-v2] "
            f"{run_config.run_id} skipped model_too_large "
            f"model_bytes={model_bytes} max_model_bytes={max_model_bytes}",
            flush=True,
        )
        return record
    print(f"[score2026-v2] evaluating {run_config.run_id}", flush=True)
    metrics = evaluate_directory(test_dir=test_root, algorithm_dir=algorithm_dir, reports_dir=reports_dir)
    record = _record_from_result(run_config, training_report, metrics, algorithm_dir, reports_dir)
    _write_json(run_root / "run_config.json", _run_config_payload(run_config))
    _write_json(run_root / "run_summary.json", record)
    print(
        "[score2026-v2] "
        f"{run_config.run_id} accuracy={record['overall_accuracy']:.6f} "
        f"macro_f1={record['macro_f1']:.6f} errors={record['error_count']} "
        f"fallback={record['num_failed_preprocess']}",
        flush=True,
    )
    return record


def build_model_too_large_record(
    *,
    run_config: RunConfig,
    training_report: dict[str, Any],
    algorithm_dir: Path,
    reports_dir: Path,
    test_root: Path,
    model_bytes: int,
    max_model_bytes: int,
) -> dict[str, Any]:
    return {
        "run_id": run_config.run_id,
        "stage": run_config.stage,
        "preprocess_name": run_config.preprocess_name,
        **run_config.lbph_params,
        "overall_accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "num_test_images": _count_images(test_root),
        "num_failed_preprocess": 0,
        "num_non_ok_predictions": 0,
        "error_count": 0,
        "fallback_error_count": 0,
        "train_preprocess_statuses": training_report.get("preprocess_statuses", {}),
        "test_preprocess_statuses": {},
        "algorithm_dir": str(algorithm_dir),
        "reports_dir": str(reports_dir),
        "model_removed": True,
        "model_bytes": int(model_bytes),
        "max_model_bytes": int(max_model_bytes),
        "skipped_reason": "model_too_large",
    }


def _record_from_result(
    run_config: RunConfig,
    training_report: dict[str, Any],
    metrics: dict[str, Any],
    algorithm_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    error_cases = metrics.get("error_cases", [])
    fallback_error_count = sum(1 for row in error_cases if row.get("preprocess_status") == "face_not_found")
    return {
        "run_id": run_config.run_id,
        "stage": run_config.stage,
        "preprocess_name": run_config.preprocess_name,
        **run_config.lbph_params,
        "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
        "macro_precision": float(metrics.get("macro_precision", 0.0)),
        "macro_recall": float(metrics.get("macro_recall", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "num_test_images": int(metrics.get("num_test_images", 0)),
        "num_failed_preprocess": int(metrics.get("num_failed_preprocess", 0)),
        "num_non_ok_predictions": int(metrics.get("num_non_ok_predictions", 0)),
        "error_count": len(error_cases),
        "fallback_error_count": int(fallback_error_count),
        "model_bytes": int((algorithm_dir / "face_recognizer_model.xml").stat().st_size)
        if (algorithm_dir / "face_recognizer_model.xml").exists()
        else 0,
        "skipped_reason": "",
        "train_preprocess_statuses": training_report.get("preprocess_statuses", {}),
        "test_preprocess_statuses": metrics.get("preprocess_statuses", {}),
        "algorithm_dir": str(algorithm_dir),
        "reports_dir": str(reports_dir),
        "model_removed": False,
    }


def _update_best_model(
    record: dict[str, Any],
    best_record: dict[str, Any] | None,
    experiments_root: Path,
    keep_experiment_models: bool,
) -> dict[str, Any]:
    if keep_experiment_models:
        return sort_run_records([item for item in [best_record, record] if item is not None])[0]
    if best_record is None:
        return record
    current_best = sort_run_records([best_record, record])[0]
    if current_best is record:
        if remove_model_file(best_record["algorithm_dir"], experiments_root):
            best_record["model_removed"] = True
        return record
    if remove_model_file(record["algorithm_dir"], experiments_root):
        record["model_removed"] = True
    return best_record


def _top_preprocess_names(records: list[dict[str, Any]], *, count: int) -> list[str]:
    preprocess_records = [record for record in records if record.get("stage") == "preprocess"]
    return [record["preprocess_name"] for record in sort_run_records(preprocess_records)[:count]]


def _is_promotable(
    record: dict[str, Any],
    baseline_accuracy: float,
    baseline_macro_f1: float,
    min_accuracy_gain: float,
) -> bool:
    return (
        float(record.get("overall_accuracy", 0.0)) > baseline_accuracy + min_accuracy_gain
        and float(record.get("macro_f1", 0.0)) >= baseline_macro_f1
    )


def _promote_best_model(best_record: dict[str, Any], final_root: Path) -> None:
    algorithm_dir = Path(best_record["algorithm_dir"])
    if final_root.exists() and any(final_root.iterdir()):
        raise FileExistsError(f"final model directory is not empty: {final_root}")
    if final_root.exists():
        final_root.rmdir()
    shutil.copytree(algorithm_dir, final_root)


def _write_summary(
    experiments_root: Path,
    records: list[dict[str, Any]],
    best_record: dict[str, Any] | None,
    final_result: dict[str, Any] | None,
    baseline_accuracy: float,
    baseline_macro_f1: float,
) -> dict[str, Any]:
    sorted_records = sort_run_records(records)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "overall_accuracy": baseline_accuracy,
            "macro_f1": baseline_macro_f1,
        },
        "best": best_record,
        "final_result": final_result,
        "results": sorted_records,
    }
    _write_json(experiments_root / "summary.json", summary)
    _write_summary_csv(experiments_root / "summary.csv", sorted_records)
    return summary


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "stage",
        "preprocess_name",
        "radius",
        "neighbors",
        "grid_x",
        "grid_y",
        "overall_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "num_test_images",
        "num_failed_preprocess",
        "num_non_ok_predictions",
        "error_count",
        "fallback_error_count",
        "model_bytes",
        "skipped_reason",
        "model_removed",
        "algorithm_dir",
        "reports_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def _run_config_payload(run_config: RunConfig) -> dict[str, Any]:
    return {
        "run_id": run_config.run_id,
        "stage": run_config.stage,
        "preprocess_name": run_config.preprocess_name,
        "preprocess_config": run_config.preprocess_config.to_dict(),
        "lbph_params": run_config.lbph_params,
    }


def _validate_inputs(train_root: Path, test_root: Path, experiments_root: Path, final_root: Path) -> None:
    if not train_root.exists():
        raise FileNotFoundError(f"train directory not found: {train_root}")
    if not test_root.exists():
        raise FileNotFoundError(f"test directory not found: {test_root}")
    if experiments_root.exists() and any(experiments_root.iterdir()):
        raise FileExistsError(f"experiments directory is not empty: {experiments_root}")
    if final_root.exists() and any(final_root.iterdir()):
        raise FileExistsError(f"final model directory is not empty: {final_root}")


def _count_images(root: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _resolve_under(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run score2026 LBPH v2 accuracy optimization.")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--final-dir", default=str(DEFAULT_FINAL_DIR))
    parser.add_argument("--baseline-accuracy", type=float, default=BASELINE_ACCURACY)
    parser.add_argument("--baseline-macro-f1", type=float, default=BASELINE_MACRO_F1)
    parser.add_argument("--min-accuracy-gain", type=float, default=0.005)
    parser.add_argument("--max-model-mib", type=float, default=DEFAULT_MAX_MODEL_MIB)
    parser.add_argument("--keep-experiment-models", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_score2026_optimization(
        workspace=args.workspace,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        experiments_dir=args.experiments_dir,
        final_dir=args.final_dir,
        baseline_accuracy=args.baseline_accuracy,
        baseline_macro_f1=args.baseline_macro_f1,
        min_accuracy_gain=args.min_accuracy_gain,
        max_model_mib=args.max_model_mib,
        keep_experiment_models=args.keep_experiment_models,
    )
    print(json.dumps(result.get("final_result"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
