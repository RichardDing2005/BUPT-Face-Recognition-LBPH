from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .dataset import image_size, workspace_root, write_manifest


ATT_ORL_URL = "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"


def prepare_att_orl_dataset(
    *,
    workspace: str | Path | None = None,
    source_zip: str | Path | None = None,
    download: bool = True,
    url: str = ATT_ORL_URL,
    seed: int = 42,
    train_per_identity: int = 8,
    test_per_identity: int = 2,
) -> dict[str, Any]:
    root = workspace_root(workspace)
    dataset_root = root / "datasets" / "pretrain_att_orl"
    source_dir = dataset_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(source_zip) if source_zip else source_dir / "att_faces.zip"
    if not zip_path.is_absolute():
        zip_path = root / zip_path
    if download and not zip_path.exists():
        _download_file(url, zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"AT&T/ORL source zip not found: {zip_path}")

    raw_root = dataset_root / "Faces_raw"
    train_root = dataset_root / "Faces_train"
    test_root = dataset_root / "Faces_test"
    for path in (raw_root, train_root, test_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    raw_rows = _extract_att_orl(zip_path, raw_root, root)
    split_rows = _split_rows(raw_rows, root, train_root, test_root, train_per_identity, test_per_identity, seed)
    manifest_path = root / "metadata" / "pretrain_att_orl_manifest.csv"
    write_manifest(manifest_path, split_rows)
    report = _build_report(
        dataset_root=dataset_root,
        zip_path=zip_path,
        manifest_path=manifest_path,
        rows=split_rows,
        train_per_identity=train_per_identity,
        test_per_identity=test_per_identity,
        seed=seed,
        url=url,
    )
    (dataset_root / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        output.write_bytes(response.read())


def _extract_att_orl(zip_path: Path, raw_root: Path, workspace: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(
            name for name in archive.namelist() if _is_att_orl_image(name)
        )
        for member in members:
            parts = Path(member).parts
            subject = next((part for part in parts if part.lower().startswith("s") and part[1:].isdigit()), None)
            if subject is None:
                continue
            subject_id = int(subject[1:])
            identity = f"att_orl_s{subject_id:03d}"
            image_name = f"{Path(member).stem}.pgm"
            target = raw_root / identity / image_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            width, height = image_size(target)
            rows.append(
                {
                    "relative_path": target.relative_to(workspace).as_posix(),
                    "identity": identity,
                    "split": "",
                    "quality_flag": "normal",
                    "face_status": "unprocessed",
                    "width": str(width),
                    "height": str(height),
                    "notes": "AT&T/ORL Database of Faces",
                    "source_manifest": "att_faces.zip",
                    "source_path": member,
                }
            )
    if not rows:
        raise ValueError(f"no AT&T/ORL .pgm images found in {zip_path}")
    return sorted(rows, key=lambda row: (row["identity"], row["relative_path"]))


def _split_rows(
    rows: list[dict[str, str]],
    workspace: Path,
    train_root: Path,
    test_root: Path,
    train_per_identity: int,
    test_per_identity: int,
    seed: int,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["identity"], []).append(row)
    rng = random.Random(seed)
    split_rows: list[dict[str, str]] = []
    for identity in sorted(grouped):
        items = sorted(grouped[identity], key=lambda row: row["relative_path"])
        rng.shuffle(items)
        if len(items) < train_per_identity + test_per_identity:
            raise ValueError(
                f"identity {identity} has {len(items)} images; need {train_per_identity + test_per_identity}"
            )
        for index, row in enumerate(items):
            split = "train" if index < train_per_identity else "test"
            if split == "test" and index >= train_per_identity + test_per_identity:
                continue
            source = workspace / row["relative_path"]
            destination_root = train_root if split == "train" else test_root
            destination = destination_root / identity / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            updated = dict(row)
            updated["relative_path"] = destination.relative_to(workspace).as_posix()
            updated["split"] = split
            split_rows.append(updated)
    return sorted(split_rows, key=lambda row: (row["identity"], row["split"], row["relative_path"]))


def _build_report(
    *,
    dataset_root: Path,
    zip_path: Path,
    manifest_path: Path,
    rows: list[dict[str, str]],
    train_per_identity: int,
    test_per_identity: int,
    seed: int,
    url: str,
) -> dict[str, Any]:
    identities = sorted({row["identity"] for row in rows})
    split_counts = {
        "train": sum(1 for row in rows if row["split"] == "train"),
        "test": sum(1 for row in rows if row["split"] == "test"),
    }
    return {
        "dataset": "att_orl",
        "source_url": url,
        "source_zip": str(zip_path),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "num_identities": len(identities),
        "num_images": len(rows),
        "split_counts": split_counts,
        "train_per_identity": train_per_identity,
        "test_per_identity": test_per_identity,
        "seed": seed,
        "identities": identities,
    }


def _is_att_orl_image(name: str) -> bool:
    path = Path(name)
    return path.suffix.lower() == ".pgm" and any(
        part.lower().startswith("s") and part[1:].isdigit()
        for part in path.parts
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dataset != "att_orl":
        raise ValueError("only --dataset att_orl is supported")
    report = prepare_att_orl_dataset(
        workspace=args.workspace,
        source_zip=args.source_zip,
        download=not args.no_download,
        url=args.url,
        seed=args.seed,
        train_per_identity=args.train_per_identity,
        test_per_identity=args.test_per_identity,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare public face datasets for LBPH training.")
    parser.add_argument("--dataset", choices=["att_orl"], default="att_orl")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-zip", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--url", default=ATT_ORL_URL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-identity", type=int, default=8)
    parser.add_argument("--test-per-identity", type=int, default=2)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
