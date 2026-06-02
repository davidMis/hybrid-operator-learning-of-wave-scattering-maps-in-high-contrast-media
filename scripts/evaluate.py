#!/usr/bin/env python3
# Overview:
# Evaluate trained FNO, scOT, or hybrid checkpoints on validation/test splits.
# The script reports mean and median relative complex L2, includes model
# parameter counts for Figure 4 aggregation, and can optionally save prediction
# arrays for downstream visualization. All inference and metric logic lives in
# helmholtz_hybrid.evaluation so this file remains a thin CLI wrapper.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_TYPES = ("fno", "scot", "hybrid")
SINGLE_MODEL_TASKS = ("smooth2smooth", "contrast", "sharp2sharp")
EVALUATION_SPLITS = ("validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FNO, scOT, or hybrid checkpoints on a paper split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing processed dataset folders.",
    )
    parser.add_argument(
        "--dataset",
        default="const_back",
        help="Processed dataset name under --data-root.",
    )
    parser.add_argument(
        "--split",
        choices=EVALUATION_SPLITS,
        default="test",
        help="Prepared split used for evaluation metrics.",
    )
    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPES,
        required=True,
        help="Checkpoint family to evaluate; hybrid composes smooth FNO and contrast scOT checkpoints.",
    )
    parser.add_argument(
        "--task",
        choices=SINGLE_MODEL_TASKS,
        default=None,
        help="Task for single-model fno/scot evaluation; omitted for hybrid evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint directory for --model-type fno or scot.",
    )
    parser.add_argument(
        "--fno-smooth-checkpoint",
        type=Path,
        help="FNO smooth2smooth checkpoint directory used by --model-type hybrid.",
    )
    parser.add_argument(
        "--scot-contrast-checkpoint",
        type=Path,
        help="scOT contrast checkpoint directory used by --model-type hybrid.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device used for inference, for example cuda, cuda:0, or cpu; auto selects CUDA when available.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path where the metrics JSON should be written.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Optional compressed .npz path containing expected, actual, and per-sample rel_l2 arrays.",
    )
    return parser.parse_args()


def request_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Convert parsed CLI arguments into evaluation request keyword arguments."""

    return {
        "data_root": args.data_root,
        "dataset": args.dataset,
        "split": args.split,
        "model_type": args.model_type,
        "task": args.task,
        "checkpoint": args.checkpoint,
        "fno_smooth_checkpoint": args.fno_smooth_checkpoint,
        "scot_contrast_checkpoint": args.scot_contrast_checkpoint,
        "batch_size": args.batch_size,
        "device": None if args.device == "auto" else args.device,
    }


def main() -> int:
    args = parse_args()
    try:
        from helmholtz_hybrid.evaluation import EvaluationRequest, evaluate_checkpoint, write_evaluation_outputs

        result = evaluate_checkpoint(EvaluationRequest(**request_kwargs_from_args(args)))
        print(json.dumps(result.metrics, indent=2))
        write_evaluation_outputs(
            result,
            output_json=args.output_json,
            predictions_out=args.predictions_out,
        )
        return 0
    except ModuleNotFoundError as error:
        print(
            f"ERROR: Missing Python dependency '{error.name}'. Install the project environment with "
            "`python -m pip install -e .` before running evaluation.",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
