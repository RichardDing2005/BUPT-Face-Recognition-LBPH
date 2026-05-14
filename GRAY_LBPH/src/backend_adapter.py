from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .preprocess import BACKEND_COMPAT_PREPROCESS_CONFIG, BACKEND_COMPAT_PREPROCESS_PROFILE_ID, PreprocessConfig


def convert_algorithm_to_backend_artifacts(
    algorithm_dir: str | Path,
    backend_artifact_dir: str | Path,
    *,
    model_name: str = "face_recognizer_model.xml",
    mapping_name: str = "label_mapping.json",
    preprocess_config_name: str = "preprocess_config.json",
) -> dict[str, str]:
    algorithm_root = Path(algorithm_dir)
    backend_root = Path(backend_artifact_dir)
    model_source = algorithm_root / model_name
    mapping_source = algorithm_root / mapping_name
    preprocess_source = algorithm_root / preprocess_config_name
    if not model_source.exists():
        raise FileNotFoundError(f"LBPH model not found: {model_source}")
    if not mapping_source.exists():
        raise FileNotFoundError(f"LBPH label mapping not found: {mapping_source}")

    backend_root.mkdir(parents=True, exist_ok=True)
    model_target = backend_root / "lbph_model.yml"
    labels_target = backend_root / "lbph_labels.json"
    profile_target = backend_root / "lbph_profile_config.json"
    shutil.copy2(model_source, model_target)

    labels = _backend_labels(mapping_source)
    fingerprint = {
        "external_artifact": True,
        "source_algorithm_dir": str(algorithm_root.resolve()),
        "model_sha256": _sha256(model_target),
        "mapping_sha256": _sha256(mapping_source),
        "samples": [],
    }
    generation_id = _generation_id(model_target, mapping_source)
    labels_payload = {
        "generation_id": generation_id,
        "fingerprint": fingerprint,
        "labels": labels,
    }
    labels_target.write_text(json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    labels_sha256 = _sha256(labels_target)

    preprocess_config_present = preprocess_source.exists()
    normalized_preprocess_config: dict[str, Any] = {}
    if preprocess_config_present:
        preprocess_config = json.loads(preprocess_source.read_text(encoding="utf-8"))
        normalized_preprocess_config = PreprocessConfig.from_dict(preprocess_config).to_dict()
    backend_compatible = (
        preprocess_config_present
        and normalized_preprocess_config == BACKEND_COMPAT_PREPROCESS_CONFIG.to_dict()
    )
    input_size = normalized_preprocess_config.get("size") or [200, 200]
    profile = {
        "artifact_version": 1,
        "generation_id": generation_id,
        "fingerprint": fingerprint,
        "external_artifact": True,
        "requires_external_loader": True,
        "auto_retrain_safe": False,
        "backend_type": "lbph",
        "model_name": "LBPH",
        "model_version": "offline-lbph:1.0",
        "distance_metric": "lbph_confidence",
        "model_path": str(model_target),
        "labels_path": str(labels_target),
        "preprocess_profile_id": (
            BACKEND_COMPAT_PREPROCESS_PROFILE_ID if backend_compatible else "custom-lbph-preprocess-v1"
        ),
        "backend_compatible_preprocess": backend_compatible,
        "preprocess_config_present": preprocess_config_present,
        "input_size": [int(input_size[0]), int(input_size[1])],
        "preprocess_config": normalized_preprocess_config,
        "model_sha256": _sha256(model_target),
        "labels_sha256": labels_sha256,
    }
    profile_target.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "model_path": str(model_target),
        "labels_path": str(labels_target),
        "profile_path": str(profile_target),
    }


def _backend_labels(mapping_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    id_to_name = {int(label): str(name) for label, name in payload.get("id_to_name", {}).items()}
    if not id_to_name:
        id_to_name = {int(label): str(name) for name, label in payload.get("name_to_id", {}).items()}
    return [
        {
            "label": int(label),
            "student_id": name,
            "name": name,
            "class_name": "",
            "photo_id": None,
            "photo_ids": [],
        }
        for label, name in sorted(id_to_name.items())
    ]


def _generation_id(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(_sha256(path).encode("ascii"))
    return f"external-{digest.hexdigest()[:16]}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
