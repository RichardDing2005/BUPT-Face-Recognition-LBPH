from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.preprocess import BACKEND_COMPAT_PREPROCESS_CONFIG
from src.train import main as train_main
from src.train import train_lbph


def train_face_model(dataset_path: str | Path, output_model: str | Path = "face_recognizer_model.xml") -> dict:
    dataset = Path(dataset_path)
    output = Path(output_model)
    if output.is_absolute():
        algorithm_dir = output.parent
        model_path = output
    elif output.parent == Path("."):
        algorithm_dir = CURRENT_DIR / "Algorithm"
        model_path = Path("Algorithm") / output.name
    else:
        algorithm_dir = output.parent
        model_path = output
    return train_lbph(
        workspace=CURRENT_DIR,
        manifest=None,
        train_dir=dataset,
        algorithm_dir=algorithm_dir,
        model_path=model_path,
        preprocess_config=BACKEND_COMPAT_PREPROCESS_CONFIG,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compatibility entrypoint for LBPH model training.")
    parser.add_argument("--workspace", default=str(CURRENT_DIR))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--train-dir", default="TestData/Faces_train")
    parser.add_argument("--algorithm-dir", default="Algorithm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mapping", default=None)
    default_width, default_height = BACKEND_COMPAT_PREPROCESS_CONFIG.size
    parser.add_argument("--resize", default=f"{default_width}x{default_height}")
    parser.add_argument(
        "--equalization",
        default=BACKEND_COMPAT_PREPROCESS_CONFIG.equalization,
        choices=["none", "hist", "equalizeHist", "clahe"],
    )
    parser.add_argument("--no-detect-face", action="store_true")
    parser.add_argument("--input-adapter", default=BACKEND_COMPAT_PREPROCESS_CONFIG.input_adapter)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--grid-x", type=int, default=7)
    parser.add_argument("--grid-y", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)
    forwarded = [
        "--workspace",
        args.workspace,
        "--train-dir",
        args.train_dir,
        "--algorithm-dir",
        args.algorithm_dir,
        "--resize",
        args.resize,
        "--equalization",
        args.equalization,
        "--radius",
        str(args.radius),
        "--neighbors",
        str(args.neighbors),
        "--grid-x",
        str(args.grid_x),
        "--grid-y",
        str(args.grid_y),
    ]
    if args.manifest:
        forwarded.extend(["--manifest", args.manifest])
    if args.model:
        forwarded.extend(["--model", args.model])
    if args.mapping:
        forwarded.extend(["--mapping", args.mapping])
    if args.no_detect_face:
        forwarded.append("--no-detect-face")
    if args.input_adapter:
        forwarded.extend(["--input-adapter", args.input_adapter])
    if args.threshold is not None:
        forwarded.extend(["--threshold", str(args.threshold)])
    return train_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
