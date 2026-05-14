from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.dataset import scan_faces_raw, stratified_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split LBPH face dataset into train/val/test folders.")
    parser.add_argument("--workspace", default=str(CURRENT_DIR))
    parser.add_argument("--manifest", default="metadata/manifest.csv")
    parser.add_argument("--raw-dir", default="TestData/Faces_raw")
    parser.add_argument("--train-dir", default="TestData/Faces_train")
    parser.add_argument("--val-dir", default="TestData/Faces_val")
    parser.add_argument("--test-dir", default="TestData/Faces_test")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-raw", action="store_true")
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args(argv)
    if args.scan_raw or not (Path(args.workspace) / args.manifest).exists():
        scan_faces_raw(args.workspace, raw_dir=args.raw_dir, out_manifest=args.manifest)
    rows = stratified_split(
        args.workspace,
        manifest=args.manifest,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_files=not args.no_copy,
    )
    print(json.dumps({"rows": len(rows), "manifest": str(Path(args.workspace) / args.manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

