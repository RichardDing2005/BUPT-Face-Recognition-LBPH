from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_rerank import ColorRerankConfig
from src.param_search import make_param_grid, run_cross_validation_search


DEFAULT_TOP_KS = [2, 3]
DEFAULT_GATES = [55.0, 56.0, 58.0, 60.0, 62.0, 65.0, 70.0]
DEFAULT_MARGINS = [0.0, 0.02, 0.05, 0.1, 0.15]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ColorRerankConfig(
        size=_parse_size(args.resize),
        detect_face=not args.no_detect_face,
        equalization=args.equalization,
        input_adapter=args.input_adapter,
        radius=args.radius,
        neighbors=args.neighbors,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        color_bins=args.color_bins,
    )
    grid = make_param_grid(
        top_ks=args.top_k or DEFAULT_TOP_KS,
        confidence_gates=args.confidence_gate or DEFAULT_GATES,
        margin_ratios=args.margin_ratio or DEFAULT_MARGINS,
    )
    payload = run_cross_validation_search(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        config=config,
        n_splits=args.folds,
        seed=args.seed,
        param_grid=grid,
    )
    print(
        json.dumps(
            {
                "selected": payload["selected"],
                "baseline_mean_accuracy": payload["baseline_mean_accuracy"],
                "summary_path": str(Path(args.output_dir) / "cv_param_summary.csv"),
                "metrics_path": str(Path(args.output_dir) / "cv_metrics.json"),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def _parse_size(value: str) -> tuple[int, int]:
    left, _, right = value.lower().partition("x")
    if not left or not right:
        raise ValueError("resize must use WIDTHxHEIGHT")
    return int(left), int(right)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run safe K-fold search for GRAY-LBPH color rerank parameters.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize", default="400x450")
    parser.add_argument("--equalization", default="clahe", choices=["none", "hist", "equalizeHist", "clahe"])
    parser.add_argument("--no-detect-face", action="store_true")
    parser.add_argument("--input-adapter", default="score2026_framework")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--grid-x", type=int, default=10)
    parser.add_argument("--grid-y", type=int, default=11)
    parser.add_argument("--color-bins", type=int, default=8)
    parser.add_argument("--top-k", action="append", type=int)
    parser.add_argument("--confidence-gate", action="append", type=float)
    parser.add_argument("--margin-ratio", action="append", type=float)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
