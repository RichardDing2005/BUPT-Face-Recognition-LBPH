from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.evaluate import evaluate_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compatibility entrypoint for LBPH face recognition scoring.")
    parser.add_argument("--workspace", default=str(CURRENT_DIR))
    parser.add_argument("--test-dir", default="TestData/Faces_test")
    parser.add_argument("--algorithm-dir", default="Algorithm")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)
    metrics = evaluate_directory(
        test_dir=_resolve(workspace, args.test_dir),
        algorithm_dir=_resolve(workspace, args.algorithm_dir),
        reports_dir=_resolve(workspace, args.reports_dir),
        threshold=args.threshold,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
    return 0


def _resolve(workspace: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


if __name__ == "__main__":
    raise SystemExit(main())

