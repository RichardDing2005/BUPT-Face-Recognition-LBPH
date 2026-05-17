from __future__ import annotations

import json
import importlib
import tarfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_confusion_config_defaults_match_score2026_aggressive_plan() -> None:
    from src.confusion_rerank import ConfusionRerankConfig, RerankRuntimeConfig

    config = ConfusionRerankConfig()
    runtime = RerankRuntimeConfig()

    assert config.size == (400, 450)
    assert config.detect_face is False
    assert config.input_adapter == "score2026_framework"
    assert config.grid_x == 10
    assert config.grid_y == 11
    assert config.aux_size == (200, 200)
    assert config.aux_grid_x == 7
    assert config.aux_grid_y == 7
    assert runtime.candidate_top_k == 4
    assert runtime.confidence_gate == 60.0
    assert runtime.gray_margin_gate == 65.0
    assert runtime.switch_margin == 0.05


def test_confusion_rerank_switches_only_within_gray_topk() -> None:
    from src.confusion_rerank import (
        ConfusionEvidenceFeature,
        ConfusionRerankConfig,
        ConfusionRerankLBPHModel,
        GrayCandidate,
        RerankRuntimeConfig,
    )

    model = ConfusionRerankLBPHModel(
        labels=["original", "target", "outside"],
        sample_labels=np.array(["original", "target", "outside"]),
        color_features=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        texture_features=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        quality_features=np.ones((3, 3), dtype=np.float32),
        config=ConfusionRerankConfig(),
    )
    query = ConfusionEvidenceFeature(
        color=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        texture=np.array([0.0, 1.0], dtype=np.float32),
        quality={"color_reliability": 1.0},
    )

    prediction = model.rerank(
        gray_candidates=[
            GrayCandidate("original", 80.0),
            GrayCandidate("target", 90.0),
            GrayCandidate("outside", 91.0),
        ],
        evidence_feature=query,
        aux_candidates=[GrayCandidate("target", 10.0), GrayCandidate("outside", 1.0)],
        runtime_config=RerankRuntimeConfig(candidate_top_k=2, switch_margin=0.05),
    )

    assert prediction.reranked is True
    assert prediction.original_label == "original"
    assert prediction.label == "target"
    assert "outside" not in prediction.candidate_scores


def test_confusion_rerank_does_not_trigger_on_confident_wide_margin_gray() -> None:
    from src.confusion_rerank import (
        ConfusionEvidenceFeature,
        ConfusionRerankConfig,
        ConfusionRerankLBPHModel,
        GrayCandidate,
        RerankRuntimeConfig,
    )

    model = ConfusionRerankLBPHModel(
        labels=["original", "target"],
        sample_labels=np.array(["original", "target"]),
        color_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        texture_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        quality_features=np.ones((2, 3), dtype=np.float32),
        config=ConfusionRerankConfig(),
    )
    query = ConfusionEvidenceFeature(
        color=np.array([0.0, 1.0], dtype=np.float32),
        texture=np.array([0.0, 1.0], dtype=np.float32),
        quality={"color_reliability": 1.0},
    )

    prediction = model.rerank(
        gray_candidates=[GrayCandidate("original", 20.0), GrayCandidate("target", 130.0)],
        evidence_feature=query,
        runtime_config=RerankRuntimeConfig(confidence_gate=60.0, gray_margin_gate=65.0),
    )

    assert prediction.reranked is False
    assert prediction.label == "original"
    assert prediction.trigger_reason == "not_triggered"


def test_count_rerank_effect_reports_help_harm_and_net_gain() -> None:
    from src.confusion_rerank import count_rerank_effect

    effect = count_rerank_effect(
        [
            {"true_label": "a", "gray_label": "b", "predicted_label": "a"},
            {"true_label": "b", "gray_label": "b", "predicted_label": "a"},
            {"true_label": "c", "gray_label": "c", "predicted_label": "c"},
        ]
    )

    assert effect == {"rerank_help": 1, "rerank_harm": 1, "net_gain": 0}


def test_aggressive_param_selection_allows_bounded_harm() -> None:
    from src.confusion_param_search import ConfusionRerankParams, select_best_params

    safe_aggressive = {
        "params": ConfusionRerankParams(candidate_top_k=4, confidence_gate=60.0, gray_margin_gate=65.0, switch_margin=0.05),
        "mean_accuracy": 0.999,
        "mean_macro_f1": 0.999,
        "baseline_mean_accuracy": 0.998,
        "folds_not_below_baseline": 4,
        "rerank_help": 3,
        "rerank_harm": 1,
        "net_gain": 2,
        "num_reranked": 7,
    }
    overfit = {
        "params": ConfusionRerankParams(candidate_top_k=4, confidence_gate=45.0, gray_margin_gate=120.0, switch_margin=0.0),
        "mean_accuracy": 0.997,
        "mean_macro_f1": 0.997,
        "baseline_mean_accuracy": 0.998,
        "folds_not_below_baseline": 2,
        "rerank_help": 4,
        "rerank_harm": 5,
        "net_gain": -1,
        "num_reranked": 40,
    }

    selected = select_best_params([overfit, safe_aggressive], min_folds_not_below=4)

    assert selected["selection_status"] == "selected"
    assert selected["params"].candidate_top_k == 4
    assert selected["metrics"]["rerank_harm"] == 1


def test_confusion_submission_package_maps_artifacts_and_skips_cache(tmp_path: Path) -> None:
    from src.confusion_rerank import ConfusionRerankConfig
    from src.confusion_submission_package import build_submission_package

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("gray_model.xml", "gray_aux_model.xml", "evidence_index.npz"):
        (model_dir / name).write_bytes(b"artifact")
    (model_dir / "label_mapping.json").write_text(
        json.dumps({"name_to_id": {"alice": 0}, "id_to_name": {"0": "alice"}}),
        encoding="utf-8",
    )
    (model_dir / "rerank_config.json").write_text(
        json.dumps(ConfusionRerankConfig().to_dict()),
        encoding="utf-8",
    )
    (model_dir / "training_report.json").write_text("{}", encoding="utf-8")

    template = tmp_path / "template" / "Algorithm"
    for relative in (
        "AlgorithmImplement.py",
        "requirements.txt",
        "Interface/AlgorithmInterface.py",
    ):
        path = template / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    cache = template / "__pycache__" / "ignored.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"x")

    output_dir = tmp_path / "submission"
    manifest = build_submission_package(
        model_dir=model_dir,
        output_dir=output_dir,
        template_dir=template,
        tar_path=output_dir / "Algorithm.tar.gz",
    )

    algorithm_dir = output_dir / "Algorithm"
    assert manifest["algorithm"] == "CA-ME-LBPH"
    assert (algorithm_dir / "face_recognizer_model.xml").read_bytes() == b"artifact"
    assert (algorithm_dir / "gray_aux_model.xml").is_file()
    assert (algorithm_dir / "evidence_index.npz").is_file()
    assert (algorithm_dir / "rerank_runtime_config.json").is_file()
    assert not (algorithm_dir / "__pycache__").exists()
    with tarfile.open(output_dir / "Algorithm.tar.gz", "r:gz") as archive:
        names = set(archive.getnames())
    assert "Algorithm/evidence_index.npz" in names
    assert all("__pycache__" not in name for name in names)


def test_confusion_submission_package_builds_tar_by_default(tmp_path: Path) -> None:
    from src.confusion_rerank import ConfusionRerankConfig
    from src.confusion_submission_package import build_submission_package

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in ("gray_model.xml", "gray_aux_model.xml", "evidence_index.npz"):
        (model_dir / name).write_bytes(b"artifact")
    (model_dir / "label_mapping.json").write_text(
        json.dumps({"name_to_id": {"alice": 0}, "id_to_name": {"0": "alice"}}),
        encoding="utf-8",
    )
    (model_dir / "rerank_config.json").write_text(
        json.dumps(ConfusionRerankConfig().to_dict()),
        encoding="utf-8",
    )

    output_dir = tmp_path / "submission"
    manifest = build_submission_package(model_dir=model_dir, output_dir=output_dir)

    assert manifest["archive"] == str(output_dir / "Algorithm.tar.gz")
    assert (output_dir / "Algorithm.tar.gz").is_file()


def test_runtime_gray_preprocess_matches_training_gray_before_resize() -> None:
    cv2 = __import__("pytest").importorskip("cv2")

    template_parent = ROOT / "submission_template"
    sys.path.insert(0, str(template_parent))
    try:
        for module_name in list(sys.modules):
            if module_name == "Algorithm" or module_name.startswith("Algorithm."):
                del sys.modules[module_name]
        module = importlib.import_module("Algorithm.AlgorithmImplement")
        algorithm = object.__new__(module.AlgorithmImplement)
        algorithm.preprocess_config = {
            "detect_face": False,
            "equalization": "none",
            "margin_ratio": 0.15,
            "min_face_size": [40, 40],
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "fallback_to_full_image": True,
            "input_adapter": "score2026_framework",
        }
        rgb = np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
                [[0, 0, 0], [120, 30, 200], [40, 220, 10], [200, 10, 80]],
            ],
            dtype=np.uint8,
        )

        actual = algorithm._prepare_gray_for_lbph(rgb, [2, 1])
        expected = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (2, 1), interpolation=cv2.INTER_AREA)

        assert np.array_equal(actual, expected)
    finally:
        if str(template_parent) in sys.path:
            sys.path.remove(str(template_parent))
        for module_name in list(sys.modules):
            if module_name == "Algorithm" or module_name.startswith("Algorithm."):
                del sys.modules[module_name]
