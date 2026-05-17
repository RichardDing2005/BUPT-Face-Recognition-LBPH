from __future__ import annotations

import argparse
import json

from src.confusion_rerank import RerankRuntimeConfig, evaluate_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a CA-ME-LBPH model directory.")
    parser.add_argument("--test-dir", default="datasets/score2026_v4/Faces_test")
    parser.add_argument("--model-dir", default="Algorithm_score2026_confusion_rerank_full")
    parser.add_argument("--reports-dir", default="reports/confusion_rerank_eval")
    parser.add_argument("--candidate-top-k", type=int, default=4)
    parser.add_argument("--confidence-gate", type=float, default=60.0)
    parser.add_argument("--gray-margin-gate", type=float, default=65.0)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    args = parser.parse_args()
    runtime = RerankRuntimeConfig(
        candidate_top_k=args.candidate_top_k,
        confidence_gate=args.confidence_gate,
        gray_margin_gate=args.gray_margin_gate,
        switch_margin=args.switch_margin,
    )
    metrics = evaluate_directory(
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        runtime_config=runtime,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
