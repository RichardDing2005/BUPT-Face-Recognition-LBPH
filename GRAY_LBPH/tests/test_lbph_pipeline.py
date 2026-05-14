from __future__ import annotations

import csv
import importlib.util
import json
import tarfile
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend_adapter import convert_algorithm_to_backend_artifacts
from src.benchmark_pack import create_benchmark_package
from src.dataset import (
    SUPPORTED_IMAGE_SUFFIXES,
    import_image_generate_manifest,
    read_manifest,
    scan_faces_raw,
    stratified_split,
)
from src.public_datasets import prepare_att_orl_dataset
from src.evaluate import compute_metrics, evaluate_directory
from src.predict import confidence_matches, load_label_mapping
from src.preprocess import BACKEND_COMPAT_PREPROCESS_CONFIG, PreprocessConfig, preprocess_image
from src.staged_train import run_training_stage
from src.training_tools import build_quality_report, write_stage_comparison
from src.train import build_label_mapping


SCORE2026_ROOT = ROOT.parent / "face_recognition_score2026" / "face_recognition_score2026"


def write_ppm(path: Path, *, value: int = 120, size: tuple[int, int] = (24, 20)) -> Path:
    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "P3\n"
        f"{width} {height}\n"
        "255\n"
        + "\n".join(
            " ".join(str(value) for _ in range(width * 3))
            for _ in range(height)
        ),
        encoding="ascii",
    )
    return path


def load_score_submission_module():
    module_path = SCORE2026_ROOT / "score_submission.py"
    if not module_path.is_file():
        pytest.skip("score2026 framework is not bundled with the public GRAY-LBPH source package")
    spec = importlib.util.spec_from_file_location("score_submission_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_pgm(path: Path, *, value: int = 120, size: tuple[int, int] = (92, 112)) -> Path:
    width, height = size
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"P5\n"
        + f"{width} {height}\n255\n".encode("ascii")
        + bytes([value]) * width * height
    )
    return path


def write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_scan_faces_raw_builds_stable_manifest_with_relative_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "LBPH"
    write_ppm(workspace / "TestData" / "Faces_raw" / "bob" / "b.ppm", value=80)
    write_ppm(workspace / "TestData" / "Faces_raw" / "alice" / "a.ppm", value=120)

    rows = scan_faces_raw(workspace)

    assert [row["identity"] for row in rows] == ["alice", "bob"]
    assert rows[0]["relative_path"] == "TestData/Faces_raw/alice/a.ppm"
    assert rows[0]["quality_flag"] == "normal"
    assert rows[0]["face_status"] == "unprocessed"
    assert (workspace / "metadata" / "manifest.csv").exists()


def test_pgm_is_supported_for_public_face_datasets() -> None:
    assert ".pgm" in SUPPORTED_IMAGE_SUFFIXES


def test_prepare_att_orl_dataset_from_existing_zip_builds_manifest_and_split(tmp_path: Path) -> None:
    workspace = tmp_path / "LBPH"
    source_zip = tmp_path / "att_faces.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        for subject in range(1, 3):
            for image_index in range(1, 11):
                image = tmp_path / "source" / f"s{subject}" / f"{image_index}.pgm"
                write_pgm(image, value=40 + subject * image_index)
                archive.write(image, f"s{subject}/{image_index}.pgm")

    report = prepare_att_orl_dataset(workspace=workspace, source_zip=source_zip, download=False)

    rows = read_manifest(workspace / "metadata" / "pretrain_att_orl_manifest.csv")
    assert report["dataset"] == "att_orl"
    assert report["num_identities"] == 2
    assert report["num_images"] == 20
    assert sorted({row["identity"] for row in rows}) == ["att_orl_s001", "att_orl_s002"]
    assert rows[0]["quality_flag"] == "normal"
    assert rows[0]["source_manifest"] == "att_faces.zip"
    assert len(list((workspace / "datasets" / "pretrain_att_orl" / "Faces_train").rglob("*.pgm"))) == 16
    assert len(list((workspace / "datasets" / "pretrain_att_orl" / "Faces_test").rglob("*.pgm"))) == 4
    assert (workspace / "datasets" / "pretrain_att_orl" / "dataset_report.json").exists()


def test_import_image_generate_manifest_keeps_provenance_without_runtime_dependency(tmp_path: Path) -> None:
    workspace = tmp_path / "LBPH"
    upstream = tmp_path / "image_generate"
    generated = write_ppm(upstream / "dynamic_range_faces" / "all" / "person_01_noise_mild.ppm")
    source = write_ppm(upstream / "original_faces" / "person_01.ppm")
    source_manifest = write_manifest(
        upstream / "metadata" / "dynamic_range_manifest.csv",
        [
            {
                "image_id": "person_01_noise_mild",
                "source_image": "person_01.ppm",
                "source_path": str(source),
                "effect_type": "noise",
                "severity_level": "mild",
                "effect_params_json": "{}",
                "output_path": str(generated),
                "width": "24",
                "height": "20",
            }
        ],
    )

    rows = import_image_generate_manifest(source_manifest, workspace=workspace)

    assert rows[0]["identity"] == "person_01"
    assert rows[0]["relative_path"] == "TestData/Faces_raw/person_01/person_01_noise_mild.ppm"
    assert rows[0]["effect_type"] == "noise"
    assert rows[0]["severity_level"] == "mild"
    assert (workspace / rows[0]["relative_path"]).exists()


def test_stratified_split_assigns_each_identity_to_train_and_test(tmp_path: Path) -> None:
    workspace = tmp_path / "LBPH"
    rows = []
    for identity in ("alice", "bob"):
        for index in range(4):
            image = write_ppm(
                workspace / "TestData" / "Faces_raw" / identity / f"{index}.ppm",
                value=80 + index,
            )
            rows.append(
                {
                    "relative_path": image.relative_to(workspace).as_posix(),
                    "identity": identity,
                    "split": "",
                    "quality_flag": "normal",
                    "face_status": "unprocessed",
                    "width": "24",
                    "height": "20",
                    "notes": "",
                }
            )
    write_manifest(workspace / "metadata" / "manifest.csv", rows)

    split_rows = stratified_split(workspace, train_ratio=0.5, val_ratio=0.0, test_ratio=0.5, seed=7)

    by_identity = {
        identity: {row["split"] for row in split_rows if row["identity"] == identity}
        for identity in ("alice", "bob")
    }
    assert by_identity == {"alice": {"train", "test"}, "bob": {"train", "test"}}
    assert list((workspace / "TestData" / "Faces_train" / "alice").glob("*.ppm"))
    assert list((workspace / "TestData" / "Faces_test" / "alice").glob("*.ppm"))


def test_stratified_split_cleans_old_split_outputs_and_preserves_nested_duplicates(tmp_path: Path) -> None:
    workspace = tmp_path / "LBPH"
    stale = workspace / "TestData" / "Faces_train" / "alice" / "stale.ppm"
    write_ppm(stale)
    rows = []
    for folder in ("session_a", "session_b", "session_c"):
        image = write_ppm(workspace / "TestData" / "Faces_raw" / "alice" / folder / "same.ppm")
        rows.append(
            {
                "relative_path": image.relative_to(workspace).as_posix(),
                "identity": "alice",
                "split": "",
                "quality_flag": "normal",
                "face_status": "unprocessed",
                "width": "24",
                "height": "20",
                "notes": "",
            }
        )
    write_manifest(workspace / "metadata" / "manifest.csv", rows)

    stratified_split(workspace, train_ratio=0.67, val_ratio=0.0, test_ratio=0.33, seed=1)

    assert stale.exists() is False
    copied = list((workspace / "TestData" / "Faces_train" / "alice").rglob("same.ppm"))
    assert len(copied) >= 1
    assert len({path.parent.name for path in copied}) == len(copied)


def test_preprocess_image_returns_uint8_gray_matrix_and_status(tmp_path: Path) -> None:
    image = write_ppm(tmp_path / "face.ppm", value=128, size=(18, 16))

    result = preprocess_image(image, PreprocessConfig(size=(10, 8), detect_face=False, equalization="none"))

    assert result.status == "ok"
    assert result.face.shape == (8, 10)
    assert str(result.face.dtype) == "uint8"
    assert result.metadata["size"] == [10, 8]


def test_preprocess_config_parses_string_booleans() -> None:
    config = PreprocessConfig.from_dict({"detect_face": "false", "fallback_to_full_image": "0"})

    assert config.detect_face is False
    assert config.fallback_to_full_image is False


def test_preprocess_config_serializes_score2026_input_adapter() -> None:
    config = PreprocessConfig.from_dict({"input_adapter": "score2026_framework"})

    assert config.input_adapter == "score2026_framework"
    assert config.to_dict()["input_adapter"] == "score2026_framework"


def test_score2026_framework_adapter_matches_framework_channel_pipeline(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    image_path = tmp_path / "color.jpg"
    bgr = np.zeros((24, 30, 3), dtype=np.uint8)
    bgr[:, :, 0] = 20
    bgr[:, :, 1] = np.arange(30, dtype=np.uint8)
    bgr[:, :, 2] = 220
    assert cv2.imwrite(str(image_path), bgr)

    disk_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(disk_bgr, cv2.COLOR_BGR2RGB)
    ok, buffer = cv2.imencode(".jpg", image_rgb)
    assert ok
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    expected = cv2.cvtColor(decoded, cv2.COLOR_RGB2GRAY)
    expected = cv2.resize(expected, (12, 10), interpolation=cv2.INTER_AREA)

    result = preprocess_image(
        image_path,
        PreprocessConfig(
            size=(12, 10),
            detect_face=False,
            equalization="none",
            input_adapter="score2026_framework",
        ),
    )

    assert result.status == "ok"
    assert result.metadata["input_adapter"] == "score2026_framework"
    assert np.array_equal(result.face, expected)


def test_backend_compatible_preprocess_profile_matches_offline_defaults() -> None:
    assert BACKEND_COMPAT_PREPROCESS_CONFIG.to_dict() == {
        "size": [200, 200],
        "detect_face": True,
        "equalization": "clahe",
        "margin_ratio": 0.15,
        "min_face_size": [40, 40],
        "scale_factor": 1.1,
        "min_neighbors": 5,
        "fallback_to_full_image": True,
        "input_adapter": "image_file",
    }


def test_train_cli_defaults_to_backend_compatible_preprocess() -> None:
    from src.train import _parser

    args = _parser().parse_args([])

    assert args.resize == "200x200"
    assert args.equalization == "clahe"
    assert args.no_detect_face is False
    assert args.input_adapter == "image_file"


def test_train_cli_accepts_score2026_framework_input_adapter() -> None:
    from src.train import _parser

    args = _parser().parse_args(["--input-adapter", "score2026_framework"])

    assert args.input_adapter == "score2026_framework"


def test_label_mapping_is_sorted_and_json_loadable(tmp_path: Path) -> None:
    mapping = build_label_mapping(["bob", "alice", "alice"])
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")

    loaded = load_label_mapping(mapping_path)

    assert mapping["name_to_id"] == {"alice": 0, "bob": 1}
    assert loaded.id_to_name == {0: "alice", 1: "bob"}


def test_compute_metrics_reports_accuracy_confusion_and_errors() -> None:
    metrics = compute_metrics(
        [
            {"true_label": "alice", "predicted_label": "alice", "status": "ok", "confidence": 12.0},
            {"true_label": "alice", "predicted_label": "bob", "status": "ok", "confidence": 90.0},
            {
                "true_label": "bob",
                "predicted_label": None,
                "status": "ok",
                "preprocess_status": "face_not_found",
                "confidence": None,
            },
        ]
    )

    assert metrics["overall_accuracy"] == pytest.approx(1 / 3)
    assert metrics["per_identity_accuracy"]["alice"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"]["alice"]["bob"] == 1
    assert metrics["num_failed_preprocess"] == 1
    assert metrics["preprocess_statuses"] == {"face_not_found": 1, "ok": 2}
    assert len(metrics["error_cases"]) == 2


def test_backend_adapter_writes_read_only_backend_artifacts(tmp_path: Path) -> None:
    algorithm_dir = tmp_path / "Algorithm"
    backend_dir = tmp_path / "lbph_backend"
    algorithm_dir.mkdir()
    (algorithm_dir / "face_recognizer_model.xml").write_text("model", encoding="utf-8")
    (algorithm_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "name_to_id": {"alice": 0, "bob": 1},
                "id_to_name": {"0": "alice", "1": "bob"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (algorithm_dir / "preprocess_config.json").write_text(
        json.dumps(BACKEND_COMPAT_PREPROCESS_CONFIG.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    result = convert_algorithm_to_backend_artifacts(algorithm_dir, backend_dir)

    assert (backend_dir / "lbph_model.yml").read_text(encoding="utf-8") == "model"
    labels = json.loads((backend_dir / "lbph_labels.json").read_text(encoding="utf-8"))
    profile = json.loads((backend_dir / "lbph_profile_config.json").read_text(encoding="utf-8"))
    assert labels["labels"][0]["student_id"] == "alice"
    assert profile["external_artifact"] is True
    assert profile["preprocess_profile_id"] == "lbph-backend-compatible-v1"
    assert profile["input_size"] == [200, 200]
    assert profile["generation_id"] == labels["generation_id"]
    assert profile["fingerprint"] == labels["fingerprint"]
    assert profile["labels_sha256"]
    assert profile["preprocess_config"]["size"] == [200, 200]
    assert result["model_path"].endswith("lbph_model.yml")


def test_backend_adapter_does_not_claim_canonical_profile_for_custom_preprocess_config(tmp_path: Path) -> None:
    algorithm_dir = tmp_path / "Algorithm"
    backend_dir = tmp_path / "lbph_backend"
    algorithm_dir.mkdir()
    (algorithm_dir / "face_recognizer_model.xml").write_text("model", encoding="utf-8")
    (algorithm_dir / "label_mapping.json").write_text(
        json.dumps({"name_to_id": {"alice": 0}, "id_to_name": {"0": "alice"}}),
        encoding="utf-8",
    )
    (algorithm_dir / "preprocess_config.json").write_text(
        json.dumps(
            PreprocessConfig(size=(92, 112), detect_face=False, equalization="none").to_dict(),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    convert_algorithm_to_backend_artifacts(algorithm_dir, backend_dir)

    profile = json.loads((backend_dir / "lbph_profile_config.json").read_text(encoding="utf-8"))
    assert profile["preprocess_profile_id"] != "lbph-backend-compatible-v1"
    assert profile["backend_compatible_preprocess"] is False
    assert profile["input_size"] == [92, 112]
    assert profile["preprocess_config"]["size"] == [92, 112]
    assert profile["preprocess_config"]["detect_face"] is False
    assert profile["preprocess_config"]["equalization"] == "none"


def test_backend_adapter_does_not_claim_canonical_profile_when_preprocess_config_missing(tmp_path: Path) -> None:
    algorithm_dir = tmp_path / "Algorithm"
    backend_dir = tmp_path / "lbph_backend"
    algorithm_dir.mkdir()
    (algorithm_dir / "face_recognizer_model.xml").write_text("model", encoding="utf-8")
    (algorithm_dir / "label_mapping.json").write_text(
        json.dumps({"name_to_id": {"alice": 0}, "id_to_name": {"0": "alice"}}),
        encoding="utf-8",
    )

    convert_algorithm_to_backend_artifacts(algorithm_dir, backend_dir)

    profile = json.loads((backend_dir / "lbph_profile_config.json").read_text(encoding="utf-8"))
    assert profile["preprocess_profile_id"] != "lbph-backend-compatible-v1"
    assert profile["backend_compatible_preprocess"] is False
    assert profile["preprocess_config_present"] is False
    assert profile["preprocess_config"] == {}


def test_predictor_resolves_relative_artifact_paths_under_algorithm_dir(tmp_path: Path) -> None:
    from src import predict

    algorithm_dir = tmp_path / "Algorithm"
    algorithm_dir.mkdir()
    (algorithm_dir / "face_recognizer_model.xml").write_text("model", encoding="utf-8")
    (algorithm_dir / "label_mapping.json").write_text(
        json.dumps({"name_to_id": {"alice": 0}, "id_to_name": {"0": "alice"}}),
        encoding="utf-8",
    )

    class FakeRecognizer:
        def read(self, path):
            self.path = path

    monkeypatch = pytest.MonkeyPatch()
    fake_cv2 = type(
        "FakeCv2",
        (),
        {"face": type("Face", (), {"LBPHFaceRecognizer_create": staticmethod(lambda: FakeRecognizer())})},
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    try:
        predictor = predict.LBPHPredictor(
            algorithm_dir=algorithm_dir,
            model_path="face_recognizer_model.xml",
            mapping_path="label_mapping.json",
        )
    finally:
        monkeypatch.undo()

    assert predictor.model_path == algorithm_dir / "face_recognizer_model.xml"
    assert predictor.mapping_path == algorithm_dir / "label_mapping.json"


def test_predictor_preserves_preprocess_status_for_successful_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.predict as predict

    algorithm_dir = tmp_path / "Algorithm"
    algorithm_dir.mkdir()
    (algorithm_dir / "face_recognizer_model.xml").write_text("model", encoding="utf-8")
    (algorithm_dir / "label_mapping.json").write_text(
        json.dumps({"name_to_id": {"alice": 0}, "id_to_name": {"0": "alice"}}),
        encoding="utf-8",
    )

    class FakeRecognizer:
        def read(self, _path):
            pass

        def predict(self, _face):
            return 0, 12.5

    class FakePreprocessResult:
        face = object()
        status = "face_not_found"
        metadata = {"face_detected": False, "face_rect": None}

    fake_cv2 = type(
        "FakeCv2",
        (),
        {"face": type("Face", (), {"LBPHFaceRecognizer_create": staticmethod(lambda: FakeRecognizer())})},
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(predict, "preprocess_image", lambda *_args, **_kwargs: FakePreprocessResult())

    result = predict.LBPHPredictor(algorithm_dir=algorithm_dir).predict_image(tmp_path / "face.jpg")

    assert result["status"] == "ok"
    assert result["label"] == "alice"
    assert result["preprocess_status"] == "face_not_found"
    assert result["face_detected"] is False
    assert result["face_rect"] is None


def test_confidence_threshold_inclusive_match() -> None:
    assert confidence_matches(80.0, 80.0) is True
    assert confidence_matches(80.1, 80.0) is False
    assert confidence_matches(80.1, None) is True


def test_train_face_model_accepts_legacy_relative_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import train_LBPH

    calls = []

    def fake_train_lbph(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(train_LBPH, "train_lbph", fake_train_lbph)

    result = train_LBPH.train_face_model("TestData/Faces_train", "Algorithm/face_recognizer_model.xml")

    assert result == {"ok": True}
    assert calls[0]["train_dir"] == Path("TestData/Faces_train")
    assert calls[0]["model_path"] == Path("Algorithm/face_recognizer_model.xml")


def test_train_face_model_default_output_keeps_algorithm_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    import train_LBPH

    calls = []

    def fake_train_lbph(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(train_LBPH, "train_lbph", fake_train_lbph)

    train_LBPH.train_face_model("TestData/Faces_train")

    assert calls[0]["algorithm_dir"] == train_LBPH.CURRENT_DIR / "Algorithm"
    assert calls[0]["model_path"] == Path("Algorithm/face_recognizer_model.xml")
    assert calls[0]["preprocess_config"].to_dict() == BACKEND_COMPAT_PREPROCESS_CONFIG.to_dict()


def test_training_samples_support_absolute_dataset_outside_workspace(tmp_path: Path) -> None:
    from src.train import _training_samples

    workspace = tmp_path / "LBPH"
    outside = tmp_path / "outside" / "alice" / "face.ppm"
    write_ppm(outside)

    samples = _training_samples(workspace, None, outside.parents[1])

    assert samples == [{"relative_path": str(outside), "identity": "alice"}]


def test_evaluate_directory_ignores_non_image_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.evaluate as evaluate

    test_dir = tmp_path / "Faces_test" / "alice"
    write_ppm(test_dir / "face.ppm")
    (test_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    class FakePredictor:
        def __init__(self, **_kwargs):
            pass

        def predict_image(self, path):
            return {
                "label": "alice",
                "confidence": 1.0,
                "status": "ok",
                "preprocess_status": "face_not_found",
                "face_detected": False,
                "face_rect": None,
                "path": str(path),
            }

    monkeypatch.setattr(evaluate, "LBPHPredictor", FakePredictor)

    metrics = evaluate.evaluate_directory(
        test_dir=test_dir.parent,
        algorithm_dir=tmp_path / "Algorithm",
        reports_dir=tmp_path / "reports",
    )

    assert metrics["num_test_images"] == 1
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["num_failed_preprocess"] == 1
    assert metrics["preprocess_statuses"] == {"face_not_found": 1}
    predictions_csv = tmp_path / "reports" / "prediction_results.csv"
    assert "preprocess_status" in predictions_csv.read_text(encoding="utf-8")
    assert "face_not_found" in predictions_csv.read_text(encoding="utf-8")


def test_benchmark_minimal_package_includes_src_modules(tmp_path: Path) -> None:
    workspace = tmp_path / "LBPH"
    for relative in (
        "train_LBPH.py",
        "test_face.py",
        "split_dataset.py",
        "staged_train.py",
        "prepare_public_dataset.py",
        "src/__init__.py",
        "src/train.py",
        "src/staged_train.py",
        "src/training_tools.py",
        "src/public_datasets.py",
        "Algorithm/face_recognizer_model.xml",
        "Algorithm/label_mapping.json",
        "Algorithm/preprocess_config.json",
        "requirements.txt",
    ):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    output = create_benchmark_package(workspace=workspace, output="benchmark_packages/lbph_minimal.zip")

    import zipfile

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "src/train.py" in names
    assert "src/staged_train.py" in names
    assert "src/training_tools.py" in names
    assert "src/public_datasets.py" in names
    assert "staged_train.py" in names
    assert "prepare_public_dataset.py" in names
    assert "src/__init__.py" in names


def test_score2026_submission_tar_contains_algorithm_without_pycache(tmp_path: Path) -> None:
    from src.benchmark_pack import create_score2026_submission_tar

    submission = tmp_path / "submission"
    for relative in (
        "Algorithm/AlgorithmImplement.py",
        "Algorithm/face_recognizer_model.xml",
        "Algorithm/label_mapping.json",
        "Algorithm/preprocess_config.json",
        "Algorithm/requirements.txt",
        "Algorithm/Interface/AlgorithmInterface.py",
    ):
        path = submission / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    pycache = submission / "Algorithm" / "__pycache__" / "ignored.pyc"
    pycache.parent.mkdir(parents=True, exist_ok=True)
    pycache.write_bytes(b"x")

    archive_path = create_score2026_submission_tar(submission)

    with tarfile.open(archive_path, "r:gz") as archive:
        name_list = archive.getnames()
        names = set(name_list)
    assert "Algorithm/AlgorithmImplement.py" in names
    assert "Algorithm/face_recognizer_model.xml" in names
    assert "Algorithm/Interface/AlgorithmInterface.py" in names
    assert len(name_list) == len(names)
    assert all("__pycache__" not in name for name in names)
    assert all(not name.endswith(".pyc") for name in names)


def test_notebook_workflow_exists_and_uses_src_modules() -> None:
    notebook_path = ROOT / "LBPH_training_testing_evaluation.ipynb"

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    assert notebook["nbformat"] == 4
    assert "scan_faces_raw" in sources
    assert "stratified_split" in sources
    assert "train_lbph" in sources
    assert "run_training_stage" in sources
    assert "evaluate_directory" in sources
    assert "LBPHPredictor" in sources
    assert "convert_algorithm_to_backend_artifacts" in sources
    assert "create_benchmark_package" in sources
    assert "D:\\" not in sources
    assert "C:\\" not in sources


def test_requirements_include_notebook_dependencies() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    for dependency in ("jupyter", "ipykernel", "pandas", "matplotlib"):
        assert dependency in requirements


def test_staged_rebuild_creates_state_checkpoint_and_promotes_top_level_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.staged_train as staged_train

    workspace = tmp_path / "LBPH"
    stage_dir = workspace / "datasets" / "pretrain"
    write_ppm(stage_dir / "alice" / "a.ppm")

    def fake_train_lbph(**kwargs):
        algorithm_dir = Path(kwargs["algorithm_dir"])
        algorithm_dir.mkdir(parents=True, exist_ok=True)
        (algorithm_dir / "face_recognizer_model.xml").write_text("model", encoding="utf-8")
        (algorithm_dir / "label_mapping.json").write_text(
            json.dumps({"name_to_id": {"alice": 0}, "id_to_name": {"0": "alice"}}),
            encoding="utf-8",
        )
        (algorithm_dir / "preprocess_config.json").write_text(
            json.dumps(kwargs["preprocess_config"].to_dict()),
            encoding="utf-8",
        )
        report = {"num_samples": 1, "num_valid_faces": 1, "num_identities": 1}
        (algorithm_dir / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(staged_train, "train_lbph", fake_train_lbph)

    result = run_training_stage(
        workspace=workspace,
        stage_name="pretrain",
        stage_train_dir=stage_dir,
        resume_mode="rebuild",
        run_id="run-a",
    )

    state = json.loads((workspace / "Algorithm" / "training_state.json").read_text(encoding="utf-8"))
    assert result["checkpoint_name"] == "001_pretrain"
    assert state["run_id"] == "run-a"
    assert state["latest_checkpoint"] == "checkpoints/001_pretrain"
    assert len(state["stages"]) == 1
    assert (workspace / "Algorithm" / "checkpoints" / "001_pretrain" / "face_recognizer_model.xml").exists()
    assert (workspace / "Algorithm" / "face_recognizer_model.xml").read_text(encoding="utf-8") == "model"


def test_staged_rebuild_accumulates_previous_stage_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.staged_train as staged_train

    workspace = tmp_path / "LBPH"
    pretrain_dir = workspace / "datasets" / "pretrain"
    main_dir = workspace / "datasets" / "main"
    write_ppm(pretrain_dir / "alice" / "a.ppm")
    write_ppm(main_dir / "bob" / "b.ppm")
    manifests = []

    def fake_train_lbph(**kwargs):
        rows = read_manifest(kwargs["manifest"])
        manifests.append(rows)
        identities = sorted({row["identity"] for row in rows})
        algorithm_dir = Path(kwargs["algorithm_dir"])
        algorithm_dir.mkdir(parents=True, exist_ok=True)
        mapping = build_label_mapping(identities)
        (algorithm_dir / "face_recognizer_model.xml").write_text("model", encoding="utf-8")
        (algorithm_dir / "label_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
        (algorithm_dir / "preprocess_config.json").write_text(
            json.dumps(kwargs["preprocess_config"].to_dict()),
            encoding="utf-8",
        )
        report = {"num_samples": len(rows), "num_valid_faces": len(rows), "num_identities": len(identities)}
        (algorithm_dir / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(staged_train, "train_lbph", fake_train_lbph)

    run_training_stage(workspace=workspace, stage_name="pretrain", stage_train_dir=pretrain_dir)
    run_training_stage(workspace=workspace, stage_name="main", stage_train_dir=main_dir)

    assert [row["identity"] for row in manifests[0]] == ["alice"]
    assert sorted(row["identity"] for row in manifests[1]) == ["alice", "bob"]
    state = json.loads((workspace / "Algorithm" / "training_state.json").read_text(encoding="utf-8"))
    assert state["stages"][1]["training_mode"] == "rebuild"
    assert state["stages"][1]["cumulative_num_samples"] == 2


def test_staged_training_removes_checkpoint_when_training_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.staged_train as staged_train

    workspace = tmp_path / "LBPH"
    stage_dir = workspace / "datasets" / "pretrain"
    write_ppm(stage_dir / "alice" / "a.ppm")

    def fake_train_lbph(**_kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(staged_train, "train_lbph", fake_train_lbph)

    with pytest.raises(RuntimeError, match="training failed"):
        run_training_stage(workspace=workspace, stage_name="pretrain", stage_train_dir=stage_dir)

    assert not (workspace / "Algorithm" / "checkpoints" / "001_pretrain").exists()
    assert not (workspace / "Algorithm" / "training_state.json").exists()


def test_staged_update_preserves_existing_label_ids_and_appends_new_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.staged_train as staged_train

    workspace = tmp_path / "LBPH"
    algorithm = workspace / "Algorithm"
    checkpoint = algorithm / "checkpoints" / "001_pretrain"
    checkpoint.mkdir(parents=True)
    (checkpoint / "face_recognizer_model.xml").write_text("old-model", encoding="utf-8")
    (checkpoint / "label_mapping.json").write_text(
        json.dumps({"name_to_id": {"alice": 0}, "id_to_name": {"0": "alice"}}),
        encoding="utf-8",
    )
    config = PreprocessConfig(size=(10, 8), detect_face=False, equalization="none").to_dict()
    (checkpoint / "preprocess_config.json").write_text(json.dumps(config), encoding="utf-8")
    (checkpoint / "training_report.json").write_text(json.dumps({"lbph_params": {}}), encoding="utf-8")
    (algorithm / "training_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "resume_mode": "rebuild",
                "stages": [
                    {
                        "stage_name": "pretrain",
                        "checkpoint_name": "001_pretrain",
                        "checkpoint": "checkpoints/001_pretrain",
                        "samples": [{"relative_path": "datasets/pretrain/alice/a.ppm", "identity": "alice"}],
                    }
                ],
                "latest_checkpoint": "checkpoints/001_pretrain",
                "preprocess_config": config,
                "lbph_params": {"radius": 2, "neighbors": 8, "grid_x": 7, "grid_y": 7, "threshold": None},
            }
        ),
        encoding="utf-8",
    )
    stage_dir = workspace / "datasets" / "main"
    write_ppm(stage_dir / "bob" / "b.ppm", size=(10, 8))
    update_calls = []

    def fake_update_stage(**kwargs):
        update_calls.append(kwargs)
        checkpoint_dir = Path(kwargs["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "face_recognizer_model.xml").write_text("updated-model", encoding="utf-8")
        (checkpoint_dir / "label_mapping.json").write_text(json.dumps(kwargs["mapping"]), encoding="utf-8")
        (checkpoint_dir / "preprocess_config.json").write_text(json.dumps(kwargs["preprocess_config"].to_dict()), encoding="utf-8")
        report = {"num_samples": 1, "num_valid_faces": 1, "num_identities": 2}
        (checkpoint_dir / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(staged_train, "_train_update_stage", fake_update_stage)

    result = run_training_stage(
        workspace=workspace,
        stage_name="main",
        stage_train_dir=stage_dir,
        resume_mode="update",
        preprocess_config=PreprocessConfig(size=(10, 8), detect_face=False, equalization="none"),
    )

    mapping = result["label_mapping"]
    assert mapping["name_to_id"] == {"alice": 0, "bob": 1}
    assert update_calls[0]["previous_checkpoint"] == checkpoint
    assert (algorithm / "face_recognizer_model.xml").read_text(encoding="utf-8") == "updated-model"


def test_staged_update_rejects_incompatible_preprocess_config(tmp_path: Path) -> None:
    workspace = tmp_path / "LBPH"
    algorithm = workspace / "Algorithm"
    checkpoint = algorithm / "checkpoints" / "001_pretrain"
    checkpoint.mkdir(parents=True)
    config = PreprocessConfig(size=(10, 8), detect_face=False, equalization="none").to_dict()
    (checkpoint / "preprocess_config.json").write_text(json.dumps(config), encoding="utf-8")
    (checkpoint / "label_mapping.json").write_text(
        json.dumps({"name_to_id": {"alice": 0}, "id_to_name": {"0": "alice"}}),
        encoding="utf-8",
    )
    (checkpoint / "face_recognizer_model.xml").write_text("old-model", encoding="utf-8")
    (algorithm / "training_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "stages": [{"checkpoint": "checkpoints/001_pretrain", "samples": []}],
                "latest_checkpoint": "checkpoints/001_pretrain",
                "preprocess_config": config,
                "lbph_params": {"radius": 2, "neighbors": 8, "grid_x": 7, "grid_y": 7, "threshold": None},
            }
        ),
        encoding="utf-8",
    )
    stage_dir = workspace / "datasets" / "main"
    write_ppm(stage_dir / "bob" / "b.ppm")

    with pytest.raises(ValueError, match="preprocess config"):
        run_training_stage(
            workspace=workspace,
            stage_name="main",
            stage_train_dir=stage_dir,
            resume_mode="update",
            preprocess_config=PreprocessConfig(size=(92, 112), detect_face=False, equalization="none"),
        )
    assert not (algorithm / "checkpoints" / "002_main").exists()


def test_training_tools_write_quality_and_stage_comparison_reports(tmp_path: Path) -> None:
    samples = [
        {"identity": "alice", "quality_flag": "normal", "face_status": "ok"},
        {"identity": "alice", "quality_flag": "blur", "face_status": "face_not_found"},
        {"identity": "bob", "quality_flag": "normal", "face_status": "ok"},
    ]

    quality = build_quality_report(samples, low_sample_threshold=2)
    comparison = write_stage_comparison(
        [
            {"stage_name": "pretrain", "evaluation": {"overall_accuracy": 0.5}, "quality_report": quality},
            {"stage_name": "main", "evaluation": {"overall_accuracy": 0.8}, "quality_report": quality},
        ],
        tmp_path / "reports" / "stage_comparison.csv",
    )

    assert quality["num_samples"] == 3
    assert quality["low_sample_identities"] == {"bob": 1}
    assert comparison.exists()
    assert "overall_accuracy" in comparison.read_text(encoding="utf-8")


def test_staged_rebuild_real_lbph_pipeline_promotes_latest_checkpoint(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "face"):
        pytest.skip("OpenCV LBPH requires opencv-contrib-python")
    workspace = tmp_path / "LBPH"
    config = PreprocessConfig(size=(92, 112), detect_face=False, equalization="none")
    write_ppm(workspace / "datasets" / "pretrain" / "alice" / "a.ppm", value=40)
    write_ppm(workspace / "datasets" / "main" / "bob" / "b.ppm", value=180)
    write_ppm(workspace / "datasets" / "finetune" / "alice" / "a2.ppm", value=60)
    write_ppm(workspace / "TestData" / "Faces_test" / "alice" / "a_test.ppm", value=40)
    write_ppm(workspace / "TestData" / "Faces_test" / "bob" / "b_test.ppm", value=180)

    run_training_stage(
        workspace=workspace,
        stage_name="pretrain",
        stage_train_dir=workspace / "datasets" / "pretrain",
        preprocess_config=config,
    )
    run_training_stage(
        workspace=workspace,
        stage_name="main",
        stage_train_dir=workspace / "datasets" / "main",
        preprocess_config=config,
    )
    result = run_training_stage(
        workspace=workspace,
        stage_name="finetune",
        stage_train_dir=workspace / "datasets" / "finetune",
        preprocess_config=config,
        evaluate_after_stage=True,
    )
    metrics = evaluate_directory(
        test_dir=workspace / "TestData" / "Faces_test",
        algorithm_dir=workspace / "Algorithm",
        reports_dir=workspace / "reports" / "compat_eval",
    )

    assert result["checkpoint_name"] == "003_finetune"
    assert (workspace / "Algorithm" / "face_recognizer_model.xml").exists()
    assert (workspace / "Algorithm" / "checkpoints" / "003_finetune" / "face_recognizer_model.xml").exists()
    assert metrics["num_test_images"] == 2


def test_score2026_optimizer_builds_fixed_experiment_grids() -> None:
    from optimize_score2026 import build_lbph_run_configs, build_preprocess_run_configs

    preprocess_runs = build_preprocess_run_configs()
    lbph_runs = build_lbph_run_configs(["baseline", "full-image"])

    assert len(preprocess_runs) == 8
    assert [run.preprocess_name for run in preprocess_runs] == [
        "baseline",
        "relaxed-1",
        "relaxed-2",
        "relaxed-3",
        "relaxed-4",
        "margin-low",
        "margin-high",
        "full-image",
    ]
    assert len(lbph_runs) == 18
    assert lbph_runs[0].stage == "lbph"
    assert lbph_runs[0].preprocess_name == "baseline"
    assert lbph_runs[0].lbph_params == {"radius": 1, "neighbors": 8, "grid_x": 8, "grid_y": 8}
    assert lbph_runs[-1].preprocess_name == "full-image"
    assert lbph_runs[-1].lbph_params == {"radius": 3, "neighbors": 12, "grid_x": 8, "grid_y": 8}


def test_score2026_optimizer_sorts_summary_by_accuracy_then_macro_f1() -> None:
    from optimize_score2026 import sort_run_records

    records = [
        {"run_id": "low", "overall_accuracy": 0.95, "macro_f1": 0.99, "num_failed_preprocess": 1},
        {"run_id": "best", "overall_accuracy": 0.97, "macro_f1": 0.94, "num_failed_preprocess": 5},
        {"run_id": "tie", "overall_accuracy": 0.97, "macro_f1": 0.93, "num_failed_preprocess": 0},
    ]

    assert [record["run_id"] for record in sort_run_records(records)] == ["best", "tie", "low"]


def test_score2026_optimizer_refuses_to_remove_model_outside_experiments(tmp_path: Path) -> None:
    from optimize_score2026 import remove_model_file

    experiments = tmp_path / "experiments" / "score2026_v2"
    safe_algorithm = experiments / "runs" / "run_001" / "Algorithm"
    safe_algorithm.mkdir(parents=True)
    safe_model = safe_algorithm / "face_recognizer_model.xml"
    safe_model.write_text("model", encoding="utf-8")
    outside_algorithm = tmp_path / "Algorithm_score2026"
    outside_algorithm.mkdir()
    outside_model = outside_algorithm / "face_recognizer_model.xml"
    outside_model.write_text("model", encoding="utf-8")

    assert remove_model_file(safe_algorithm, experiments) is True
    assert safe_model.exists() is False
    with pytest.raises(ValueError, match="outside experiments"):
        remove_model_file(outside_algorithm, experiments)
    assert outside_model.exists() is True


def test_score2026_optimizer_builds_skipped_record_for_oversized_model(tmp_path: Path) -> None:
    from optimize_score2026 import RunConfig, build_model_too_large_record

    test_root = tmp_path / "Faces_test" / "alice"
    write_ppm(test_root / "face.ppm")
    run_config = RunConfig(
        run_id="run_big",
        stage="lbph",
        preprocess_name="full-image",
        preprocess_config=PreprocessConfig(detect_face=False),
        lbph_params={"radius": 1, "neighbors": 12, "grid_x": 8, "grid_y": 8},
    )

    record = build_model_too_large_record(
        run_config=run_config,
        training_report={"preprocess_statuses": {"ok": 1}},
        algorithm_dir=tmp_path / "Algorithm",
        reports_dir=tmp_path / "reports",
        test_root=test_root.parent,
        model_bytes=2_000,
        max_model_bytes=1_000,
    )

    assert record["run_id"] == "run_big"
    assert record["overall_accuracy"] == 0.0
    assert record["num_test_images"] == 1
    assert record["model_removed"] is True
    assert record["skipped_reason"] == "model_too_large"


def test_score_submission_runner_requires_faces_test_by_default(tmp_path: Path) -> None:
    score_submission = load_score_submission_module()
    scoring_root = tmp_path / "score2026"
    (scoring_root / "TestData" / "Faces" / "alice").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Faces_test"):
        score_submission.resolve_dataset_dir(scoring_root, None)


def test_score_submission_runner_rejects_incomplete_algorithm(tmp_path: Path) -> None:
    score_submission = load_score_submission_module()
    algorithm_dir = tmp_path / "Algorithm"
    algorithm_dir.mkdir()
    (algorithm_dir / "AlgorithmImplement.py").write_text("class AlgorithmImplement: pass", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="face_recognizer_model.xml"):
        score_submission.validate_algorithm_dir(algorithm_dir)
