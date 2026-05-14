from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_rerank import ColorRerankConfig, train_directory


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
    report = train_directory(train_dir=args.train_dir, output_dir=args.output_dir, config=config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parse_size(value: str) -> tuple[int, int]:
    left, _, right = value.lower().partition("x")
    if not left or not right:
        raise ValueError("resize must use WIDTHxHEIGHT")
    return int(left), int(right)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GRAY-LBPH with color rerank index.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resize", default="400x450")
    parser.add_argument("--equalization", default="clahe", choices=["none", "hist", "equalizeHist", "clahe"])
    parser.add_argument("--no-detect-face", action="store_true")
    parser.add_argument("--input-adapter", default="score2026_framework")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--grid-x", type=int, default=10)
    parser.add_argument("--grid-y", type=int, default=11)
    parser.add_argument("--color-bins", type=int, default=8)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
