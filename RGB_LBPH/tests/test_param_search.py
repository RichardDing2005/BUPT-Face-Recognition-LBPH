from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RGB_LBPH.src.param_search import (
    FALLBACK_PARAMS,
    RerankParams,
    count_rerank_effect,
    make_stratified_folds,
    select_best_params,
)


def test_make_stratified_folds_keeps_each_identity_in_every_validation_fold(tmp_path: Path) -> None:
    samples = []
    for label in ("alice", "bob"):
        for index in range(5):
            samples.append((label, tmp_path / label / f"{index}.jpg"))

    folds = make_stratified_folds(samples, n_splits=5, seed=42)

    assert len(folds) == 5
    for train_samples, validation_samples in folds:
        validation_labels = [label for label, _path in validation_samples]
        assert validation_labels.count("alice") == 1
        assert validation_labels.count("bob") == 1
        assert not set(train_samples).intersection(validation_samples)


def test_count_rerank_effect_records_help_and_harm() -> None:
    rows = [
        {"true_label": "a", "gray_label": "b", "predicted_label": "a", "reranked": True},
        {"true_label": "a", "gray_label": "a", "predicted_label": "b", "reranked": True},
        {"true_label": "a", "gray_label": "a", "predicted_label": "a", "reranked": False},
    ]

    effect = count_rerank_effect(rows)

    assert effect["rerank_help"] == 1
    assert effect["rerank_harm"] == 1
    assert effect["net_gain"] == 0


def test_select_best_params_requires_stable_gain_and_prefers_conservative_ties() -> None:
    unsafe = {
        "params": RerankParams(candidate_top_k=2, confidence_gate=55.0, margin_ratio=0.0),
        "mean_accuracy": 0.91,
        "mean_macro_f1": 0.91,
        "baseline_mean_accuracy": 0.90,
        "folds_not_below_baseline": 3,
        "rerank_help": 4,
        "rerank_harm": 1,
        "net_gain": 3,
        "num_reranked": 10,
    }
    less_conservative = {
        "params": RerankParams(candidate_top_k=3, confidence_gate=60.0, margin_ratio=0.1),
        "mean_accuracy": 0.92,
        "mean_macro_f1": 0.92,
        "baseline_mean_accuracy": 0.90,
        "folds_not_below_baseline": 5,
        "rerank_help": 4,
        "rerank_harm": 1,
        "net_gain": 3,
        "num_reranked": 6,
    }
    conservative = {
        "params": RerankParams(candidate_top_k=2, confidence_gate=62.0, margin_ratio=0.1),
        "mean_accuracy": 0.92,
        "mean_macro_f1": 0.92,
        "baseline_mean_accuracy": 0.90,
        "folds_not_below_baseline": 5,
        "rerank_help": 4,
        "rerank_harm": 1,
        "net_gain": 3,
        "num_reranked": 6,
    }

    selected = select_best_params([unsafe, less_conservative, conservative])

    assert selected["params"] == conservative["params"]
    assert selected["selection_status"] == "selected"


def test_select_best_params_falls_back_when_no_candidate_is_safe() -> None:
    selected = select_best_params(
        [
            {
                "params": RerankParams(candidate_top_k=3, confidence_gate=55.0, margin_ratio=0.0),
                "mean_accuracy": 0.90,
                "mean_macro_f1": 0.90,
                "baseline_mean_accuracy": 0.90,
                "folds_not_below_baseline": 5,
                "rerank_help": 1,
                "rerank_harm": 1,
                "net_gain": 0,
                "num_reranked": 12,
            }
        ]
    )

    assert selected["params"] == FALLBACK_PARAMS
    assert selected["selection_status"] == "fallback"
