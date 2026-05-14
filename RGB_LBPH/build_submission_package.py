from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.submission_package import build_submission_package


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime_config = {
        "candidate_top_k": args.candidate_top_k,
        "confidence_gate": args.confidence_gate,
        "rerank_margin_ratio": args.rerank_margin_ratio,
    }
    manifest = build_submission_package(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        template_dir=args.template_dir,
        runtime_config=runtime_config,
        tar_path=args.tar_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a score2026-style RGB-LBPH submission directory."
    )
    parser.add_argument("--model-dir", required=True, help="Directory produced by train_color_rerank_lbph.py.")
    parser.add_argument("--output-dir", required=True, help="Output root that will receive Algorithm/.")
    parser.add_argument(
        "--template-dir",
        default=str(ROOT / "submission_template" / "Algorithm"),
        help="Runtime Algorithm template directory.",
    )
    parser.add_argument("--candidate-top-k", type=int, default=2)
    parser.add_argument("--confidence-gate", type=float, default=70.0)
    parser.add_argument("--rerank-margin-ratio", type=float, default=0.0)
    parser.add_argument("--tar-path", default=None, help="Optional Algorithm.tar.gz output path.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
