from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .color_rerank import (
    ColorRerankConfig,
    ColorRerankLBPHModel,
    GrayPrediction,
    _collect_samples,
    _create_recognizer,
    _rows_for_params,
    collect_gray_candidates,
    compute_metrics,
    extract_color_feature,
    preprocess_gray_for_lbph,
)


Sample = tuple[str, Path]

FALLBACK_PARAMS = None


@dataclass(frozen=True, order=True)
class RerankParams:
    candidate_top_k: int
    confidence_gate: float
    margin_ratio: float

    @property
    def key(self) -> str:
        gate = f"{self.confidence_gate:g}".replace(".", "_")
        margin = f"{self.margin_ratio:g}".replace(".", "_")
        return f"topk_{self.candidate_top_k}_gate_{gate}_margin_{margin}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_top_k": int(self.candidate_top_k),
            "confidence_gate": float(self.confidence_gate),
            "margin_ratio": float(self.margin_ratio),
        }


FALLBACK_PARAMS = RerankParams(candidate_top_k=2, confidence_gate=60.0, margin_ratio=0.1)


def make_param_grid(
    *,
    top_ks: Iterable[int],
    confidence_gates: Iterable[float],
    margin_ratios: Iterable[float],
) -> list[RerankParams]:
    return [
        RerankParams(int(top_k), float(gate), float(margin))
        for top_k in top_ks
        for gate in confidence_gates
        for margin in margin_ratios
    ]


def make_stratified_folds(samples: list[Sample], *, n_splits: int, seed: int) -> list[tuple[list[Sample], list[Sample]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    by_label: dict[str, list[Sample]] = defaultdict(list)
    for label, image_path in samples:
        by_label[str(label)].append((str(label), Path(image_path)))
    rng = random.Random(seed)
    fold_validation: list[list[Sample]] = [[] for _ in range(n_splits)]
    for label_samples in by_label.values():
        shuffled = list(label_samples)
        rng.shuffle(shuffled)
        if len(shuffled) < n_splits:
            raise ValueError("every identity must have at least n_splits samples")
        for index, sample in enumerate(shuffled):
            fold_validation[index % n_splits].append(sample)
    folds = []
    all_samples = [(label, Path(path)) for label, path in samples]
    for validation_samples in fold_validation:
        validation_set = set(validation_samples)
        train_samples = [sample for sample in all_samples if sample not in validation_set]
        folds.append((train_samples, sorted(validation_samples, key=lambda item: (item[0], str(item[1])))))
    return folds


def count_rerank_effect(rows: list[dict[str, Any]]) -> dict[str, int]:
    help_count = 0
    harm_count = 0
    for row in rows:
        true_label = str(row["true_label"])
        gray_correct = str(row["gray_label"]) == true_label
        rerank_correct = str(row["predicted_label"]) == true_label
        if not gray_correct and rerank_correct:
            help_count += 1
        elif gray_correct and not rerank_correct:
            harm_count += 1
    return {
        "rerank_help": help_count,
        "rerank_harm": harm_count,
        "net_gain": help_count - harm_count,
    }


def select_best_params(rows: list[dict[str, Any]], *, min_folds_not_below: int = 4) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if float(row["mean_accuracy"]) > float(row["baseline_mean_accuracy"])
        and int(row["net_gain"]) > 0
        and int(row["folds_not_below_baseline"]) >= min_folds_not_below
        and int(row["rerank_harm"]) <= int(row["rerank_help"])
    ]
    if not eligible:
        return {
            "selection_status": "fallback",
            "params": FALLBACK_PARAMS,
            "reason": "no parameter set satisfied the safety rule",
        }
    selected = sorted(eligible, key=_selection_sort_key)[0]
    return {
        "selection_status": "selected",
        "params": selected["params"],
        "reason": "best safe cross-validation result",
        "metrics": selected,
    }


def run_cross_validation_search(
    *,
    raw_dir: str | Path,
    output_dir: str | Path,
    config: ColorRerankConfig,
    n_splits: int,
    seed: int,
    param_grid: list[RerankParams],
) -> dict[str, Any]:
    samples = _collect_samples(Path(raw_dir))
    if not samples:
        raise ValueError(f"no samples found: {raw_dir}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    folds = make_stratified_folds(samples, n_splits=n_splits, seed=seed)
    per_fold = []
    aggregate: dict[str, dict[str, Any]] = {}
    for fold_index, (train_samples, validation_samples) in enumerate(folds):
        model = train_model_from_samples(train_samples, config=config)
        base_items = build_base_items(validation_samples, model)
        baseline_rows = _baseline_rows(base_items)
        baseline_metrics = compute_metrics(baseline_rows)
        fold_payload = {
            "fold": fold_index,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "baseline": _compact_metrics(baseline_metrics),
            "params": {},
        }
        for params in param_grid:
            rows = _rows_for_params(
                base_items,
                model,
                confidence_gate=params.confidence_gate,
                margin_ratio=params.margin_ratio,
                top_k=params.candidate_top_k,
            )
            metrics = compute_metrics(rows)
            effect = count_rerank_effect(rows)
            compact = _compact_metrics(metrics)
            compact.update(effect)
            compact["params"] = params.to_dict()
            compact["baseline_accuracy"] = baseline_metrics["overall_accuracy"]
            compact["not_below_baseline"] = metrics["overall_accuracy"] >= baseline_metrics["overall_accuracy"]
            fold_payload["params"][params.key] = compact
            record = aggregate.setdefault(
                params.key,
                {
                    "params": params,
                    "fold_metrics": [],
                    "rerank_help": 0,
                    "rerank_harm": 0,
                    "net_gain": 0,
                    "num_reranked": 0,
                    "folds_not_below_baseline": 0,
                },
            )
            record["fold_metrics"].append(
                {
                    "accuracy": metrics["overall_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "baseline_accuracy": baseline_metrics["overall_accuracy"],
                }
            )
            record["rerank_help"] += effect["rerank_help"]
            record["rerank_harm"] += effect["rerank_harm"]
            record["net_gain"] += effect["net_gain"]
            record["num_reranked"] += metrics["num_reranked"]
            if metrics["overall_accuracy"] >= baseline_metrics["overall_accuracy"]:
                record["folds_not_below_baseline"] += 1
        per_fold.append(fold_payload)
        del model
    summary_rows = [_aggregate_record(record) for record in aggregate.values()]
    baseline_mean_accuracy = _mean(
        fold_payload["baseline"]["overall_accuracy"] for fold_payload in per_fold
    )
    for row in summary_rows:
        row["baseline_mean_accuracy"] = baseline_mean_accuracy
    selected = select_best_params(summary_rows, min_folds_not_below=max(1, n_splits - 1))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(Path(raw_dir)),
        "num_samples": len(samples),
        "num_identities": len({label for label, _path in samples}),
        "n_splits": n_splits,
        "seed": seed,
        "config": config.to_dict(),
        "fallback_params": FALLBACK_PARAMS.to_dict(),
        "selected": _jsonify_selection(selected),
        "baseline_mean_accuracy": baseline_mean_accuracy,
        "summary": [_jsonify_summary_row(row) for row in sorted(summary_rows, key=_selection_sort_key)],
        "folds": per_fold,
    }
    _write_summary_csv(output_root / "cv_param_summary.csv", summary_rows)
    (output_root / "cv_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def train_model_from_samples(samples: list[Sample], *, config: ColorRerankConfig) -> ColorRerankLBPHModel:
    labels = sorted({label for label, _path in samples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    faces = []
    numeric_labels = []
    sample_labels = []
    color_features = []
    for label, image_path in samples:
        faces.append(preprocess_gray_for_lbph(image_path, config))
        numeric_labels.append(label_to_id[label])
        sample_labels.append(label)
        color_features.append(extract_color_feature(image_path, config))
    recognizer = _create_recognizer(config)
    recognizer.train(faces, np.array(numeric_labels, dtype=np.int32))
    return ColorRerankLBPHModel(
        labels=labels,
        sample_labels=np.array(sample_labels),
        color_features=np.stack(color_features).astype(np.float32),
        config=config,
        recognizer=recognizer,
    )


def build_base_items(samples: list[Sample], model: ColorRerankLBPHModel) -> list[dict[str, Any]]:
    items = []
    for label, image_path in sorted(samples, key=lambda item: (item[0], str(item[1]))):
        gray = preprocess_gray_for_lbph(image_path, model.config)
        gray_candidates = collect_gray_candidates(model.recognizer, gray, model.labels)
        predicted_label, confidence = gray_candidates[0]
        items.append(
            {
                "image_path": str(image_path),
                "true_label": label,
                "gray_prediction": GrayPrediction(label=predicted_label, confidence=float(confidence)),
                "gray_candidates": gray_candidates,
                "color_feature": extract_color_feature(image_path, model.config),
            }
        )
    return items


def _baseline_rows(base_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in base_items:
        prediction = item["gray_prediction"]
        rows.append(
            {
                "image_path": item["image_path"],
                "true_label": item["true_label"],
                "predicted_label": prediction.label,
                "gray_label": prediction.label,
                "confidence": prediction.confidence,
                "reranked": False,
                "color_label": None,
                "color_distance": None,
                "gray_label_color_distance": None,
                "candidate_top_k": None,
                "confidence_gate": "inf",
                "margin_ratio": 0.0,
            }
        )
    return rows


def _aggregate_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["fold_metrics"]
    return {
        "params": record["params"],
        "mean_accuracy": _mean(item["accuracy"] for item in metrics),
        "mean_macro_f1": _mean(item["macro_f1"] for item in metrics),
        "baseline_mean_accuracy": _mean(item["baseline_accuracy"] for item in metrics),
        "folds_not_below_baseline": record["folds_not_below_baseline"],
        "rerank_help": record["rerank_help"],
        "rerank_harm": record["rerank_harm"],
        "net_gain": record["net_gain"],
        "num_reranked": record["num_reranked"],
    }


def _selection_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    params: RerankParams = row["params"]
    return (
        -float(row["mean_accuracy"]),
        -float(row["mean_macro_f1"]),
        params.candidate_top_k,
        -params.confidence_gate,
        -params.margin_ratio,
        int(row["num_reranked"]),
    )


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "key",
        "candidate_top_k",
        "confidence_gate",
        "margin_ratio",
        "mean_accuracy",
        "mean_macro_f1",
        "baseline_mean_accuracy",
        "folds_not_below_baseline",
        "rerank_help",
        "rerank_harm",
        "net_gain",
        "num_reranked",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=_selection_sort_key):
            params: RerankParams = row["params"]
            writer.writerow(
                {
                    "key": params.key,
                    **params.to_dict(),
                    "mean_accuracy": row["mean_accuracy"],
                    "mean_macro_f1": row["mean_macro_f1"],
                    "baseline_mean_accuracy": row["baseline_mean_accuracy"],
                    "folds_not_below_baseline": row["folds_not_below_baseline"],
                    "rerank_help": row["rerank_help"],
                    "rerank_harm": row["rerank_harm"],
                    "net_gain": row["net_gain"],
                    "num_reranked": row["num_reranked"],
                }
            )


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_accuracy": metrics["overall_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "correct": metrics["correct"],
        "wrong": metrics["wrong"],
        "num_test_images": metrics["num_test_images"],
        "num_reranked": metrics["num_reranked"],
    }


def _jsonify_selection(selection: dict[str, Any]) -> dict[str, Any]:
    payload = dict(selection)
    payload["params"] = selection["params"].to_dict()
    if "metrics" in payload:
        payload["metrics"] = _jsonify_summary_row(payload["metrics"])
    return payload


def _jsonify_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    params: RerankParams = row["params"]
    payload = dict(row)
    payload["key"] = params.key
    payload["params"] = params.to_dict()
    return payload


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else math.nan
