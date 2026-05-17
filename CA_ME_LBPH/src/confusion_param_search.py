from __future__ import annotations

import argparse
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

from .confusion_rerank import (
    ConfusionEvidenceFeature,
    ConfusionRerankConfig,
    ConfusionRerankLBPHModel,
    RerankRuntimeConfig,
    _collect_samples,
    _create_recognizer,
    compute_metrics,
    count_rerank_effect,
    extract_evidence_feature,
    preprocess_image,
)


Sample = tuple[str, Path]


@dataclass(frozen=True, order=True)
class ConfusionRerankParams:
    candidate_top_k: int
    confidence_gate: float
    gray_margin_gate: float
    switch_margin: float

    @property
    def key(self) -> str:
        return (
            f"topk_{int(self.candidate_top_k)}"
            f"_conf_{_key_float(self.confidence_gate)}"
            f"_graygap_{_key_float(self.gray_margin_gate)}"
            f"_switch_{_key_float(self.switch_margin)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_top_k": int(self.candidate_top_k),
            "confidence_gate": float(self.confidence_gate),
            "gray_margin_gate": float(self.gray_margin_gate),
            "switch_margin": float(self.switch_margin),
        }

    def to_runtime_config(self) -> RerankRuntimeConfig:
        return RerankRuntimeConfig(
            candidate_top_k=self.candidate_top_k,
            confidence_gate=self.confidence_gate,
            gray_margin_gate=self.gray_margin_gate,
            switch_margin=self.switch_margin,
        )


FALLBACK_PARAMS = ConfusionRerankParams(
    candidate_top_k=4,
    confidence_gate=60.0,
    gray_margin_gate=65.0,
    switch_margin=0.05,
)


def make_param_grid(
    *,
    top_ks: Iterable[int],
    confidence_gates: Iterable[float],
    gray_margin_gates: Iterable[float],
    switch_margins: Iterable[float],
) -> list[ConfusionRerankParams]:
    return [
        ConfusionRerankParams(int(top_k), float(confidence), float(gray_gap), float(switch))
        for top_k in top_ks
        for confidence in confidence_gates
        for gray_gap in gray_margin_gates
        for switch in switch_margins
    ]


def select_best_params(rows: list[dict[str, Any]], *, min_folds_not_below: int = 4) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if float(row["mean_accuracy"]) > float(row["baseline_mean_accuracy"])
        and int(row["net_gain"]) > 0
        and int(row["folds_not_below_baseline"]) >= int(min_folds_not_below)
        and int(row["rerank_harm"]) <= int(row["rerank_help"])
    ]
    if not eligible:
        return {
            "selection_status": "fallback",
            "params": FALLBACK_PARAMS,
            "reason": "no parameter set satisfied the aggressive safety rule",
        }
    selected = sorted(eligible, key=_selection_sort_key)[0]
    return {
        "selection_status": "selected",
        "params": selected["params"],
        "reason": "best aggressive cross-validation result",
        "metrics": selected,
    }


def run_cross_validation_search(
    *,
    raw_dir: str | Path,
    output_dir: str | Path,
    config: ConfusionRerankConfig,
    n_splits: int,
    seed: int,
    param_grid: list[ConfusionRerankParams],
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
            rows = rows_for_params(base_items, model, params=params)
            metrics = compute_metrics(rows)
            effect = count_rerank_effect(rows)
            compact = _compact_metrics(metrics)
            compact.update(effect)
            compact["baseline_accuracy"] = baseline_metrics["overall_accuracy"]
            compact["not_below_baseline"] = metrics["overall_accuracy"] >= baseline_metrics["overall_accuracy"]
            compact["params"] = params.to_dict()
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
    baseline_mean_accuracy = _mean(fold["baseline"]["overall_accuracy"] for fold in per_fold)
    for row in summary_rows:
        row["baseline_mean_accuracy"] = baseline_mean_accuracy
    selected = select_best_params(summary_rows, min_folds_not_below=max(1, n_splits - 1))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "CA-ME-LBPH",
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
    all_samples = [(label, Path(path)) for label, path in samples]
    folds = []
    for validation_samples in fold_validation:
        validation_set = set(validation_samples)
        train_samples = [sample for sample in all_samples if sample not in validation_set]
        folds.append((train_samples, sorted(validation_samples, key=lambda item: (item[0], str(item[1])))))
    return folds


def train_model_from_samples(samples: list[Sample], *, config: ConfusionRerankConfig) -> ConfusionRerankLBPHModel:
    labels = sorted({label for label, _path in samples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    primary_faces = []
    aux_faces = []
    numeric_labels = []
    sample_labels = []
    color_features = []
    texture_features = []
    quality_features = []
    for label, image_path in samples:
        primary = preprocess_image(image_path, config.primary_preprocess_config())
        aux = preprocess_image(image_path, config.aux_preprocess_config())
        evidence = extract_evidence_feature(image_path, config)
        primary_faces.append(primary.face)
        aux_faces.append(aux.face)
        numeric_labels.append(label_to_id[label])
        sample_labels.append(label)
        color_features.append(evidence.color)
        texture_features.append(evidence.texture)
        quality_features.append(_quality_vector(evidence))
    recognizer = _create_recognizer(config.radius, config.neighbors, config.grid_x, config.grid_y)
    recognizer.train(primary_faces, np.array(numeric_labels, dtype=np.int32))
    aux_recognizer = _create_recognizer(config.aux_radius, config.aux_neighbors, config.aux_grid_x, config.aux_grid_y)
    aux_recognizer.train(aux_faces, np.array(numeric_labels, dtype=np.int32))
    return ConfusionRerankLBPHModel(
        labels=labels,
        sample_labels=np.array(sample_labels),
        color_features=np.stack(color_features).astype(np.float32),
        texture_features=np.stack(texture_features).astype(np.float32),
        quality_features=np.stack(quality_features).astype(np.float32),
        config=config,
        recognizer=recognizer,
        aux_recognizer=aux_recognizer,
    )


def build_base_items(samples: list[Sample], model: ConfusionRerankLBPHModel) -> list[dict[str, Any]]:
    items = []
    for label, image_path in sorted(samples, key=lambda item: (item[0], str(item[1]))):
        primary = preprocess_image(image_path, model.config.primary_preprocess_config())
        aux = preprocess_image(image_path, model.config.aux_preprocess_config())
        if model.recognizer is None or model.aux_recognizer is None:
            raise RuntimeError("model recognizers are required")
        from .confusion_rerank import collect_gray_candidates

        gray_result = collect_gray_candidates(model.recognizer, primary.face, model.labels)
        aux_result = collect_gray_candidates(model.aux_recognizer, aux.face, model.labels)
        items.append(
            {
                "image_path": str(image_path),
                "true_label": label,
                "gray_candidates": gray_result,
                "aux_candidates": aux_result,
                "evidence_feature": extract_evidence_feature(image_path, model.config),
            }
        )
    return items


def rows_for_params(
    base_items: list[dict[str, Any]],
    model: ConfusionRerankLBPHModel,
    *,
    params: ConfusionRerankParams,
) -> list[dict[str, Any]]:
    rows = []
    runtime = params.to_runtime_config()
    for item in base_items:
        prediction = model.rerank(
            gray_candidates=item["gray_candidates"],
            aux_candidates=item["aux_candidates"],
            evidence_feature=item["evidence_feature"],
            runtime_config=runtime,
        )
        rows.append(
            {
                "image_path": item["image_path"],
                "true_label": item["true_label"],
                "predicted_label": prediction.label,
                "gray_label": prediction.original_label,
                "confidence": prediction.confidence,
                "reranked": prediction.reranked,
                "trigger_reason": prediction.trigger_reason,
                "secondary_label": prediction.secondary_label,
                **params.to_dict(),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ConfusionRerankConfig(
        size=_parse_size(args.resize),
        equalization=args.equalization,
        detect_face=bool(args.detect_face),
        input_adapter=args.input_adapter,
    )
    param_grid = make_param_grid(
        top_ks=args.top_ks,
        confidence_gates=args.confidence_gates,
        gray_margin_gates=args.gray_margin_gates,
        switch_margins=args.switch_margins,
    )
    payload = run_cross_validation_search(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        config=config,
        n_splits=args.folds,
        seed=args.seed,
        param_grid=param_grid,
    )
    print(json.dumps(payload["selected"], ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search CA-ME-LBPH rerank parameters.")
    parser.add_argument("--raw-dir", default="datasets/score2026/Faces_raw")
    parser.add_argument("--output-dir", default="experiments/confusion_rerank_param_search_5fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize", default="400x450")
    parser.add_argument("--equalization", default="clahe")
    parser.set_defaults(detect_face=False)
    parser.add_argument("--detect-face", dest="detect_face", action="store_true")
    parser.add_argument("--no-detect-face", dest="detect_face", action="store_false")
    parser.add_argument("--input-adapter", default="score2026_framework")
    parser.add_argument("--top-ks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--confidence-gates", type=float, nargs="+", default=[55.0, 60.0, 65.0, 70.0])
    parser.add_argument("--gray-margin-gates", type=float, nargs="+", default=[45.0, 65.0, 85.0])
    parser.add_argument("--switch-margins", type=float, nargs="+", default=[0.0, 0.03, 0.05, 0.08])
    return parser


def _baseline_rows(base_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in base_items:
        prediction = item["gray_candidates"][0]
        rows.append(
            {
                "image_path": item["image_path"],
                "true_label": item["true_label"],
                "predicted_label": prediction.label,
                "gray_label": prediction.label,
                "confidence": prediction.confidence,
                "reranked": False,
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
    params: ConfusionRerankParams = row["params"]
    return (
        -float(row["mean_accuracy"]),
        -float(row["mean_macro_f1"]),
        -int(row["net_gain"]),
        int(row["rerank_harm"]),
        params.candidate_top_k,
        -params.confidence_gate,
        -params.gray_margin_gate,
        float(params.switch_margin),
    )


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "key",
        "candidate_top_k",
        "confidence_gate",
        "gray_margin_gate",
        "switch_margin",
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
            params: ConfusionRerankParams = row["params"]
            writer.writerow({"key": params.key, **params.to_dict(), **{field: row[field] for field in fields if field in row}})


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
    params: ConfusionRerankParams = row["params"]
    payload = dict(row)
    payload["key"] = params.key
    payload["params"] = params.to_dict()
    return payload


def _quality_vector(evidence: ConfusionEvidenceFeature) -> np.ndarray:
    return np.array(
        [
            float(evidence.quality.get("brightness", 0.0)),
            float(evidence.quality.get("contrast", 0.0)),
            float(evidence.quality.get("color_reliability", 0.0)),
        ],
        dtype=np.float32,
    )


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else math.nan


def _key_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "_")


def _parse_size(value: str) -> tuple[int, int]:
    left, _, right = value.lower().partition("x")
    if not left or not right:
        raise ValueError("size must use WIDTHxHEIGHT")
    return int(left), int(right)


if __name__ == "__main__":
    raise SystemExit(main())
