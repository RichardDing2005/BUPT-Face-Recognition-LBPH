from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_rerank import evaluate_directory


DEFAULT_GATES = [float("inf"), 55.0, 56.0, 60.0, 62.0, 65.0, 70.0]
DEFAULT_MARGINS = [0.0, 0.02, 0.05, 0.1]
DEFAULT_TOP_KS = [2, 3]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gates = [_parse_gate(value) for value in args.confidence_gate] if args.confidence_gate else DEFAULT_GATES
    margins = [float(value) for value in args.margin_ratio] if args.margin_ratio else DEFAULT_MARGINS
    top_ks = [int(value) for value in args.top_k] if args.top_k else DEFAULT_TOP_KS
    metrics = evaluate_directory(
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        confidence_gates=gates,
        margin_ratios=margins,
        candidate_top_ks=top_ks,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _parse_gate(value: str) -> float:
    if value.lower() in {"inf", "infinity", "none", "baseline"}:
        return float("inf")
    return float(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GRAY-LBPH with color rerank.")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--confidence-gate", action="append")
    parser.add_argument("--margin-ratio", action="append")
    parser.add_argument("--top-k", action="append", type=int, help="Gray candidate labels allowed for color rerank.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
