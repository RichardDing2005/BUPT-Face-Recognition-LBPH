from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
from zipfile import ZIP_DEFLATED, ZipFile


MINIMAL_PATHS = [
    "train_LBPH.py",
    "test_face.py",
    "split_dataset.py",
    "staged_train.py",
    "prepare_public_dataset.py",
    "src/__init__.py",
    "src/dataset.py",
    "src/preprocess.py",
    "src/public_datasets.py",
    "src/train.py",
    "src/staged_train.py",
    "src/training_tools.py",
    "src/predict.py",
    "src/evaluate.py",
    "src/backend_adapter.py",
    "src/benchmark_pack.py",
    "Algorithm/face_recognizer_model.xml",
    "Algorithm/label_mapping.json",
    "Algorithm/preprocess_config.json",
    "requirements.txt",
]

SCORE2026_REQUIRED_ALGORITHM_FILES = [
    "AlgorithmImplement.py",
    "face_recognizer_model.xml",
    "label_mapping.json",
    "preprocess_config.json",
    "requirements.txt",
]


def create_benchmark_package(
    *,
    workspace: str | Path,
    output: str | Path,
    mode: str = "minimal",
) -> Path:
    root = Path(workspace).resolve()
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = MINIMAL_PATHS if mode == "minimal" else _all_files(root, exclude=output_path)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative in paths:
            path = root / relative
            if path.exists() and path.is_file():
                archive.write(path, relative)
    return output_path


def create_score2026_submission_tar(
    submission_dir: str | Path,
    output: str | Path = "Algorithm.tar.gz",
) -> Path:
    root = Path(submission_dir).resolve()
    algorithm_root = root / "Algorithm"
    if not algorithm_root.is_dir():
        raise FileNotFoundError(f"Algorithm directory not found: {algorithm_root}")
    for relative in SCORE2026_REQUIRED_ALGORITHM_FILES:
        path = algorithm_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required score2026 file not found: {path}")

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        archive.add(algorithm_root, arcname="Algorithm", recursive=False)
        for path in sorted(algorithm_root.rglob("*")):
            if _skip_score2026_tar_path(path):
                continue
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create LBPH benchmark package.")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default="benchmark_packages/lbph_minimal.zip")
    parser.add_argument("--mode", choices=["minimal", "full"], default="minimal")
    args = parser.parse_args(argv)
    print(create_benchmark_package(workspace=args.workspace, output=args.output, mode=args.mode))
    return 0


def _all_files(root: Path, *, exclude: Path | None = None) -> list[str]:
    ignored = {"__pycache__", ".pytest_cache"}
    files = []
    for path in root.rglob("*"):
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        if path.is_file() and not any(part in ignored for part in path.parts):
            files.append(path.relative_to(root).as_posix())
    return sorted(files)


def _skip_score2026_tar_path(path: Path) -> bool:
    if any(part == "__pycache__" for part in path.parts):
        return True
    return path.is_file() and path.suffix.lower() in {".pyc", ".pyo"}


if __name__ == "__main__":
    raise SystemExit(main())
