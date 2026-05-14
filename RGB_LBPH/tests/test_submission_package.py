from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RGB_LBPH.src.submission_package import build_submission_package


def test_build_submission_package_maps_training_outputs_to_runtime_names(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "submission"
    model_dir.mkdir()
    (model_dir / "gray_model.xml").write_text("<opencv_storage/>", encoding="utf-8")
    (model_dir / "label_mapping.json").write_text(
        json.dumps({"id_to_name": {"0": "alice"}, "name_to_id": {"alice": 0}}),
        encoding="utf-8",
    )
    (model_dir / "rerank_config.json").write_text(
        json.dumps(
            {
                "size": [400, 450],
                "detect_face": False,
                "equalization": "clahe",
                "fallback_to_full_image": True,
                "input_adapter": "score2026_framework",
                "grid_x": 10,
                "grid_y": 11,
                "color_bins": 8,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "training_report.json").write_text(
        json.dumps({"algorithm": "RGB-LBPH"}),
        encoding="utf-8",
    )
    np.savez_compressed(
        model_dir / "color_index.npz",
        labels=np.array(["alice"]),
        sample_labels=np.array(["alice"]),
        color_features=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    manifest = build_submission_package(
        model_dir=model_dir,
        output_dir=output_dir,
        template_dir=ROOT / "submission_template" / "Algorithm",
        runtime_config={
            "candidate_top_k": 2,
            "confidence_gate": 70.0,
            "rerank_margin_ratio": 0.0,
        },
    )

    algorithm_dir = output_dir / "Algorithm"
    assert manifest["algorithm_dir"] == str(algorithm_dir)
    assert (algorithm_dir / "AlgorithmImplement.py").is_file()
    assert (algorithm_dir / "Interface" / "AlgorithmInterface.py").is_file()
    assert (algorithm_dir / "face_recognizer_model.xml").read_text(encoding="utf-8") == "<opencv_storage/>"
    assert (algorithm_dir / "color_index.npz").is_file()
    assert (algorithm_dir / "label_mapping.json").is_file()
    assert json.loads((algorithm_dir / "preprocess_config.json").read_text(encoding="utf-8"))["size"] == [400, 450]
    runtime = json.loads((algorithm_dir / "rerank_runtime_config.json").read_text(encoding="utf-8"))
    assert runtime["candidate_top_k"] == 2
    assert runtime["confidence_gate"] == 70.0
    assert runtime["rerank_margin_ratio"] == 0.0
    assert runtime["grid_x"] == 10
    assert runtime["grid_y"] == 11
    assert runtime["color_bins"] == 8
    assert not (algorithm_dir / "gray_model.xml").exists()
