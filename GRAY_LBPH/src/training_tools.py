from __future__ import annotations

import csv
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .preprocess import PreprocessConfig


def build_quality_report(
    samples: Iterable[dict[str, Any]],
    *,
    low_sample_threshold: int = 3,
) -> dict[str, Any]:
    rows = [dict(sample) for sample in samples]
    identity_counts = Counter(str(row.get("identity", "")) for row in rows if row.get("identity"))
    quality_counts = Counter(str(row.get("quality_flag") or "unknown") for row in rows)
    face_status_counts = Counter(str(row.get("face_status") or "unprocessed") for row in rows)
    low_sample_identities = {
        identity: count for identity, count in sorted(identity_counts.items()) if count < low_sample_threshold
    }
    return {
        "num_samples": len(rows),
        "num_identities": len(identity_counts),
        "samples_per_identity": dict(sorted(identity_counts.items())),
        "low_sample_threshold": int(low_sample_threshold),
        "low_sample_identities": low_sample_identities,
        "quality_flag_distribution": dict(sorted(quality_counts.items())),
        "face_status_distribution": dict(sorted(face_status_counts.items())),
    }


def write_stage_comparison(stages: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage_index",
        "stage_name",
        "checkpoint_name",
        "training_mode",
        "stage_num_samples",
        "cumulative_num_samples",
        "num_identities",
        "overall_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "num_test_images",
        "num_non_ok_predictions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, stage in enumerate(stages, start=1):
            evaluation = stage.get("evaluation") or {}
            quality = stage.get("quality_report") or {}
            writer.writerow(
                {
                    "stage_index": stage.get("stage_index", index),
                    "stage_name": stage.get("stage_name", ""),
                    "checkpoint_name": stage.get("checkpoint_name", ""),
                    "training_mode": stage.get("training_mode", ""),
                    "stage_num_samples": stage.get("stage_num_samples", ""),
                    "cumulative_num_samples": stage.get("cumulative_num_samples", ""),
                    "num_identities": quality.get("num_identities", ""),
                    "overall_accuracy": evaluation.get("overall_accuracy", ""),
                    "macro_precision": evaluation.get("macro_precision", ""),
                    "macro_recall": evaluation.get("macro_recall", ""),
                    "macro_f1": evaluation.get("macro_f1", ""),
                    "num_test_images": evaluation.get("num_test_images", ""),
                    "num_non_ok_predictions": evaluation.get("num_non_ok_predictions", ""),
                }
            )
    return path


def run_threshold_sweep(
    *,
    test_dir: str | Path,
    algorithm_dir: str | Path,
    reports_dir: str | Path,
    thresholds: Iterable[float],
) -> dict[str, Any]:
    from .evaluate import evaluate_directory

    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for threshold in thresholds:
        threshold_dir = root / f"threshold_{float(threshold):.6g}".replace(".", "_")
        metrics = evaluate_directory(
            test_dir=test_dir,
            algorithm_dir=algorithm_dir,
            reports_dir=threshold_dir,
            threshold=float(threshold),
        )
        rows.append({"threshold": float(threshold), **_metric_summary(metrics)})
    best = max(rows, key=lambda item: (item.get("overall_accuracy", 0.0), item.get("macro_f1", 0.0))) if rows else None
    _write_dict_rows(root / "threshold_sweep.csv", rows)
    (root / "threshold_sweep.json").write_text(
        json.dumps({"best": best, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"best": best, "results": rows}


def grid_search_lbph(
    *,
    workspace: str | Path,
    manifest: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    param_grid: dict[str, Iterable[Any]],
    preprocess_config: PreprocessConfig | None = None,
) -> dict[str, Any]:
    from .evaluate import evaluate_directory
    from .train import train_lbph

    root = Path(workspace)
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    keys = ["radius", "neighbors", "grid_x", "grid_y", "threshold"]
    values = [list(param_grid.get(key, [None if key == "threshold" else _default_param(key)])) for key in keys]
    rows = []
    for index, combo in enumerate(itertools.product(*values), start=1):
        params = dict(zip(keys, combo))
        algorithm_dir = output_root / f"run_{index:03d}" / "Algorithm"
        reports_dir = output_root / f"run_{index:03d}" / "reports"
        train_lbph(
            workspace=root,
            manifest=manifest,
            algorithm_dir=algorithm_dir,
            preprocess_config=preprocess_config or PreprocessConfig(),
            radius=int(params["radius"]),
            neighbors=int(params["neighbors"]),
            grid_x=int(params["grid_x"]),
            grid_y=int(params["grid_y"]),
            threshold=params["threshold"],
        )
        metrics = evaluate_directory(
            test_dir=root / test_dir if not Path(test_dir).is_absolute() else test_dir,
            algorithm_dir=algorithm_dir,
            reports_dir=reports_dir,
            threshold=params["threshold"],
        )
        rows.append({"run": index, **params, **_metric_summary(metrics)})
    best = max(rows, key=lambda item: (item.get("overall_accuracy", 0.0), item.get("macro_f1", 0.0))) if rows else None
    _write_dict_rows(output_root / "grid_search_results.csv", rows)
    (output_root / "grid_search_results.json").write_text(
        json.dumps({"best": best, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"best": best, "results": rows}


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_accuracy": metrics.get("overall_accuracy", 0.0),
        "macro_precision": metrics.get("macro_precision", 0.0),
        "macro_recall": metrics.get("macro_recall", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "num_test_images": metrics.get("num_test_images", 0),
        "num_non_ok_predictions": metrics.get("num_non_ok_predictions", 0),
    }


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _default_param(key: str) -> int:
    defaults = {"radius": 2, "neighbors": 8, "grid_x": 7, "grid_y": 7}
    return defaults[key]
