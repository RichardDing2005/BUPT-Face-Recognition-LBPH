from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RGB_LBPH.src.color_rerank import (
    ColorRerankConfig,
    ColorRerankLBPHModel,
    GrayPrediction,
    extract_color_feature,
    read_rgb_image,
    train_directory,
    evaluate_directory,
)


def write_rgb(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def write_channel_pattern(path: Path, channel: int, *, size: tuple[int, int] = (32, 32)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = size[1], size[0]
    yy, xx = np.indices((height, width))
    pattern = (((xx // 4 + yy // 4) % 2) * 180 + 40).astype(np.uint8)
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    image[:, :, channel] = pattern
    Image.fromarray(image).save(path)
    return path


def test_color_feature_length_and_rgb_channel_order(tmp_path: Path) -> None:
    image_path = write_rgb(tmp_path / "red.png", (255, 0, 0))
    config = ColorRerankConfig(size=(32, 32), grid_x=2, grid_y=2, color_bins=8, equalization="none")

    rgb = read_rgb_image(image_path, input_adapter="score2026_framework")
    feature = extract_color_feature(image_path, config)

    assert rgb is not None
    assert int(rgb[:, :, 0].mean()) > 240
    assert int(rgb[:, :, 1].mean()) < 10
    assert int(rgb[:, :, 2].mean()) < 10
    assert feature.shape == (2 * 2 * 3 * 8,)


def test_rerank_gate_keeps_confident_gray_prediction() -> None:
    model = ColorRerankLBPHModel(
        labels=["gray_label", "color_label"],
        sample_labels=np.array(["gray_label", "color_label"]),
        color_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        config=ColorRerankConfig(),
    )

    prediction = model.rerank(
        GrayPrediction(label="gray_label", confidence=30.0),
        np.array([0.0, 1.0], dtype=np.float32),
        confidence_gate=50.0,
        margin_ratio=0.0,
    )

    assert prediction.label == "gray_label"
    assert prediction.reranked is False


def test_rerank_switches_uncertain_gray_when_color_margin_is_strong() -> None:
    model = ColorRerankLBPHModel(
        labels=["gray_label", "color_label"],
        sample_labels=np.array(["gray_label", "color_label"]),
        color_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        config=ColorRerankConfig(),
    )

    prediction = model.rerank(
        GrayPrediction(label="gray_label", confidence=70.0),
        np.array([0.0, 1.0], dtype=np.float32),
        confidence_gate=50.0,
        margin_ratio=0.05,
    )

    assert prediction.label == "color_label"
    assert prediction.original_label == "gray_label"
    assert prediction.reranked is True


def test_rerank_respects_gray_candidate_limit() -> None:
    model = ColorRerankLBPHModel(
        labels=["gray_label", "candidate_label", "outside_label"],
        sample_labels=np.array(["gray_label", "candidate_label", "outside_label"]),
        color_features=np.array(
            [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        config=ColorRerankConfig(),
    )

    prediction = model.rerank(
        GrayPrediction(label="gray_label", confidence=70.0),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        confidence_gate=50.0,
        margin_ratio=0.0,
        candidate_labels=["gray_label", "candidate_label"],
    )

    assert prediction.label == "candidate_label"
    assert prediction.color_label == "candidate_label"
    assert prediction.reranked is True


def test_train_and_evaluate_small_rerank_dataset(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    model_dir = tmp_path / "model"
    reports_dir = tmp_path / "reports"
    for split_dir in (train_dir, test_dir):
        write_channel_pattern(split_dir / "alice" / "1.png", 0)
        write_channel_pattern(split_dir / "alice" / "2.png", 0)
        write_channel_pattern(split_dir / "bob" / "1.png", 2)
        write_channel_pattern(split_dir / "bob" / "2.png", 2)
    config = ColorRerankConfig(
        size=(32, 32),
        grid_x=2,
        grid_y=2,
        color_bins=8,
        equalization="none",
        input_adapter="score2026_framework",
    )

    report = train_directory(train_dir=train_dir, output_dir=model_dir, config=config)
    metrics = evaluate_directory(
        test_dir=test_dir,
        model_dir=model_dir,
        reports_dir=reports_dir,
        confidence_gates=[float("inf"), 0.0],
        margin_ratios=[0.0],
    )

    assert report["num_samples"] == 4
    assert report["num_identities"] == 2
    assert (model_dir / "gray_model.xml").is_file()
    assert (model_dir / "color_index.npz").is_file()
    assert metrics["best"]["num_test_images"] == 4
    assert metrics["best"]["overall_accuracy"] == 1.0
    assert (reports_dir / "metrics.json").is_file()
    assert json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))["best"]["wrong"] == 0
