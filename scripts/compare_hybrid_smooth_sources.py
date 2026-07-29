#!/usr/bin/env python3
# Overview:
# Compare capacity-matched hybrid models when the residual scOT receives exact
# smooth pressure versus pressure predicted by the trained smooth-task FNO. The
# script evaluates both reconstructions against the same sharp-pressure target
# and writes a wide-form CSV plus a manuscript-ready LaTeX table.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helmholtz_hybrid.hybrid_comparison import (
    COMPARISON_SPLITS,
    DEFAULT_SWEEP_SIZES,
    compare_checkpoint_pair,
    fno_smooth_checkpoint,
    missing_comparison_inputs,
    render_markdown_table,
    scot_contrast_checkpoint,
    write_comparison_outputs,
)


def parse_args() -> argparse.Namespace:
    """Parse smooth-source comparison command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare hybrid sharp-pressure accuracy with exact versus FNO-predicted "
            "smooth pressure, using the same residual scOT and sharp target."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing prepared dataset folders.",
    )
    parser.add_argument(
        "--dataset",
        default="const_back",
        help="Prepared dataset name under --data-root.",
    )
    parser.add_argument(
        "--split",
        choices=COMPARISON_SPLITS,
        default="test",
        help="Prepared split used for both smooth-source evaluations.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Checkpoint root containing released fno/ and scot/ directories. "
            "Defaults to outputs/checkpoints/<dataset>/paper."
        ),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWEEP_SIZES),
        help="FNO layer counts and matching shared scOT stage depths to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of test samples evaluated per inference batch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Torch inference device, for example cuda, cuda:0, or cpu. "
            "The default selects CUDA and fails clearly when no GPU is visible."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap for smoke tests; omit for the publication table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Wide-form metrics CSV. Defaults to "
            "results/<dataset>/paper/hybrid_smooth_source_comparison.csv."
        ),
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help="Manuscript-ready LaTeX table; defaults next to --output with suffix _table.tex.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input paths and print the planned checkpoint pairs without inference.",
    )
    args = parser.parse_args()
    if args.checkpoint_root is None:
        args.checkpoint_root = Path("outputs/checkpoints") / args.dataset / "paper"
    if args.output is None:
        args.output = (
            Path("results")
            / args.dataset
            / "paper"
            / "hybrid_smooth_source_comparison.csv"
        )
    if args.latex_output is None:
        args.latex_output = args.output.with_name(
            f"{args.output.stem}_table.tex"
        )
    return args


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested inference device and validate CUDA availability."""

    if requested == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA device is visible. Run this command on a GPU node or pass "
                "--device cpu only for a small functional smoke test."
            )
        requested = "cuda"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device}, but PyTorch reports that CUDA is unavailable."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        torch.cuda.set_device(device)
    return device


def validate_args(args: argparse.Namespace) -> None:
    """Raise actionable errors for invalid options or missing input artifacts."""

    if not args.sizes:
        raise ValueError("At least one sweep size is required.")
    if any(size < 1 for size in args.sizes):
        raise ValueError(f"Sweep sizes must be positive; got {args.sizes}.")
    if len(set(args.sizes)) != len(args.sizes):
        raise ValueError(f"Sweep sizes must be unique; got {args.sizes}.")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be positive; got {args.batch_size}.")
    if args.workers < 0:
        raise ValueError(f"--workers must be non-negative; got {args.workers}.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError(f"--max-samples must be positive; got {args.max_samples}.")

    missing = missing_comparison_inputs(
        args.data_root,
        args.dataset,
        args.split,
        args.checkpoint_root,
        args.sizes,
    )
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing smooth-source comparison inputs:\n"
            f"{missing_list}\n"
            "Check --data-root, --checkpoint-root, --dataset, and --sizes."
        )


def print_plan(args: argparse.Namespace) -> None:
    """Print the exact dataset, checkpoint pairs, and outputs selected."""

    print(f"Dataset split: {args.data_root / args.dataset / args.split}")
    print(f"Checkpoint root: {args.checkpoint_root}")
    print(f"Metrics CSV: {args.output}")
    print(f"LaTeX table: {args.latex_output}")
    print("Checkpoint pairs:")
    for size in sorted(args.sizes):
        print(f"  n={size}")
        print(f"    FNO:  {fno_smooth_checkpoint(args.checkpoint_root, args.dataset, size)}")
        print(f"    scOT: {scot_contrast_checkpoint(args.checkpoint_root, args.dataset, size)}")


def run(args: argparse.Namespace) -> int:
    """Validate inputs, run the comparison sweep, and write both table formats."""

    validate_args(args)
    print_plan(args)
    if args.dry_run:
        print("Input validation passed; no models were loaded.")
        return 0

    device = resolve_device(args.device)
    sample_description = (
        "full split" if args.max_samples is None else f"first {args.max_samples} samples"
    )
    print(f"Running on {device} over {sample_description}.")

    results = []
    size_iterator = tqdm(
        sorted(args.sizes),
        desc="overall",
        unit="model pair",
        dynamic_ncols=True,
        position=0,
        disable=not sys.stderr.isatty(),
    )
    for size in size_iterator:
        result = compare_checkpoint_pair(
            data_root=args.data_root,
            dataset=args.dataset,
            split=args.split,
            checkpoint_root=args.checkpoint_root,
            size=size,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            max_samples=args.max_samples,
            show_progress=True,
            progress_position=1,
        )
        results.append(result)
        # Preserve completed capacities if a later checkpoint fails.
        write_comparison_outputs(
            results,
            csv_path=args.output,
            latex_path=args.latex_output,
        )
        size_iterator.set_postfix_str(
            f"n={size}, exact={result.ground_truth_mean_relative_l2:.4f}, "
            f"FNO={result.trained_fno_mean_relative_l2:.4f}"
        )

    print()
    print(render_markdown_table(results))
    print(f"\nWrote metrics to {args.output}")
    print(f"Wrote LaTeX table to {args.latex_output}")
    return 0


def main() -> int:
    """Run the CLI with concise, actionable error reporting."""

    args = parse_args()
    try:
        return run(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
