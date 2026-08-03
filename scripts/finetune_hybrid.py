#!/usr/bin/env python3
# Overview:
# Fine-tune a residual scOT for one epoch on pressure supplied by a frozen,
# capacity-matched smooth FNO. The script records the pretrained baseline, uses
# a constant learning rate, logs optionally to W&B, and saves the resulting
# matched pair in the native formats used by evaluate.py.
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from helmholtz_hybrid.cli_config import apply_yaml_defaults, config_path_from_argv
from helmholtz_hybrid.data import dataloader_performance_kwargs, load_task_dataset
from helmholtz_hybrid.evaluation import load_fno_checkpoint, load_scot_checkpoint
from helmholtz_hybrid.hybrid_comparison import (
    fno_smooth_checkpoint,
    missing_comparison_inputs,
    scot_contrast_checkpoint,
)
from helmholtz_hybrid.hybrid_finetuning import (
    FrozenFNOHybridOperator,
    EpochMetrics,
    append_epoch_metrics,
    evaluate_hybrid_epoch,
    save_hybrid_checkpoint,
    train_hybrid_epoch,
)
from helmholtz_hybrid.reproducibility import (
    configure_torch_reproducibility,
    count_parameters,
    write_run_manifest,
)
from helmholtz_hybrid.runtime import (
    resolve_torch_device,
    set_default_cache_dirs,
)


def parse_args() -> argparse.Namespace:
    """Parse frozen-FNO scOT fine-tuning options."""

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a residual-task scOT for one epoch using pressure from a "
            "frozen, capacity-matched smooth-task FNO."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional flat YAML file containing default scOT fine-tuning options.",
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
        "--checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Released checkpoint root containing fno/ and scot/ directories. "
            "Defaults to outputs/checkpoints/<dataset>/paper."
        ),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2,
        help="FNO layer count and shared scOT stage depth of the matched pair.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory that receives a new run subdirectory. Defaults to "
            "outputs/checkpoints/<dataset>/hybrid_finetune/seed<seed>."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name; otherwise a capacity/seed/timestamp name is used.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Constant AdamW learning rate used for the single scOT epoch.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW decoupled weight-decay coefficient.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum scOT gradient norm; set to 0 to disable clipping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Physical training and validation batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Microbatches averaged before each optimizer update.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes for each split.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional first-N training subset for smoke tests; omit for a full run.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Optional first-N validation subset for smoke tests; omit for a full run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for Python, NumPy, PyTorch, CUDA, and DataLoader shuffling.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Torch training device, for example cuda, cuda:0, or cpu. "
            "Auto selects CUDA and fails clearly when no GPU is visible."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request deterministic CUDA kernels where PyTorch supports them.",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow TF32 matrix math on supported NVIDIA GPUs.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Optional Weights & Biases project name; omit to disable W&B logging.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional Weights & Biases entity/team used with --wandb-project.",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Optional W&B group for collecting capacities or repeated seeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and options and print the run plan without loading models.",
    )
    apply_yaml_defaults(parser, config_path_from_argv())
    args = parser.parse_args()
    if args.checkpoint_root is None:
        args.checkpoint_root = Path("outputs/checkpoints") / args.dataset / "paper"
    else:
        args.checkpoint_root = Path(args.checkpoint_root)
    if args.output_root is None:
        args.output_root = (
            Path("outputs/checkpoints")
            / args.dataset
            / "hybrid_finetune"
            / f"seed{args.seed}"
        )
    else:
        args.output_root = Path(args.output_root)
    return args


def set_seed(seed: int) -> torch.Generator:
    """Seed all local random-number generators and the shuffled DataLoader."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_run_name(args: argparse.Namespace) -> str:
    """Return a self-describing unique run name unless one was supplied."""

    if args.run_name:
        return args.run_name
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return (
        f"hybrid_{args.dataset}_scot_finetune_layers{args.size}_"
        f"depths{args.size}-{args.size}-{args.size}-{args.size}_"
        f"seed{args.seed}_{stamp}"
    )


def validate_args(args: argparse.Namespace, output_dir: Path) -> None:
    """Raise actionable errors for invalid settings or missing inputs."""

    if args.size < 1:
        raise ValueError(f"--size must be positive; got {args.size}.")
    if args.learning_rate <= 0:
        raise ValueError(f"--learning-rate must be positive; got {args.learning_rate}.")
    if args.weight_decay < 0:
        raise ValueError(f"--weight-decay must be non-negative; got {args.weight_decay}.")
    if args.max_grad_norm < 0:
        raise ValueError(f"--max-grad-norm must be non-negative; got {args.max_grad_norm}.")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be positive; got {args.batch_size}.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "--gradient-accumulation-steps must be positive; "
            f"got {args.gradient_accumulation_steps}."
        )
    if args.workers < 0:
        raise ValueError(f"--workers must be non-negative; got {args.workers}.")
    for option in ("max_train_samples", "max_validation_samples"):
        value = getattr(args, option)
        if value is not None and value < 1:
            raise ValueError(f"--{option.replace('_', '-')} must be positive; got {value}.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {output_dir}. "
            "Choose a different --run-name so an earlier run is not overwritten."
        )

    missing: list[Path] = []
    for split in ("train", "validation"):
        missing.extend(
            missing_comparison_inputs(
                args.data_root,
                args.dataset,
                split,
                args.checkpoint_root,
                [args.size],
            )
        )
    # Deduplicate the checkpoint paths reported once per data split.
    unique_missing = list(dict.fromkeys(missing))
    if unique_missing:
        missing_list = "\n".join(f"  - {path}" for path in unique_missing)
        raise FileNotFoundError(
            "Missing frozen-FNO scOT fine-tuning inputs:\n"
            f"{missing_list}\n"
            "Check --data-root, --checkpoint-root, --dataset, and --size."
        )


def print_plan(args: argparse.Namespace, output_dir: Path) -> None:
    """Print the exact input pair, data splits, and output location."""

    fno_checkpoint = fno_smooth_checkpoint(args.checkpoint_root, args.dataset, args.size)
    scot_checkpoint = scot_contrast_checkpoint(args.checkpoint_root, args.dataset, args.size)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Training data: {args.data_root / args.dataset / 'train'}")
    print(f"Validation data: {args.data_root / args.dataset / 'validation'}")
    print(f"Initial FNO: {fno_checkpoint}")
    print(f"Initial scOT: {scot_checkpoint}")
    print(f"Run output: {output_dir}")
    print(
        f"Schedule: 1 epoch, constant LR {args.learning_rate:g}, "
        f"physical batch {args.batch_size}, effective batch {effective_batch_size}"
    )


def maybe_init_wandb(args: argparse.Namespace, run_name: str):
    """Start W&B only when the user supplied a project."""

    if args.wandb_project is None:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging was requested, but wandb is not installed. "
            "Install the logging extra with `pip install -e '.[logging]'`."
        ) from error

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config=config,
    )


def limited_dataset(dataset, maximum: int | None):
    """Return the full dataset or its deterministic first-N smoke-test subset."""

    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))


def run(args: argparse.Namespace) -> int:
    """Load the pretrained pair, fine-tune scOT once, and save the result."""

    run_name = make_run_name(args)
    output_dir = args.output_root / run_name
    validate_args(args, output_dir)
    print_plan(args, output_dir)
    if args.dry_run:
        print("Input validation passed; no models were loaded and no output was written.")
        return 0

    device = resolve_torch_device(args.device, require_cuda_for_auto=True)
    generator = set_seed(args.seed)
    configure_torch_reproducibility(args.deterministic, args.allow_tf32)
    set_default_cache_dirs()

    train_dataset = limited_dataset(
        load_task_dataset(args.data_root, args.dataset, "train", "hybrid"),
        args.max_train_samples,
    )
    validation_dataset = limited_dataset(
        load_task_dataset(args.data_root, args.dataset, "validation", "hybrid"),
        args.max_validation_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **dataloader_performance_kwargs(args.workers, device.type == "cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_performance_kwargs(args.workers, device.type == "cuda"),
    )

    initial_fno = fno_smooth_checkpoint(args.checkpoint_root, args.dataset, args.size)
    initial_scot = scot_contrast_checkpoint(args.checkpoint_root, args.dataset, args.size)
    model = FrozenFNOHybridOperator(
        load_fno_checkpoint(initial_fno, device),
        load_scot_checkpoint(initial_scot, device),
    ).to(device)
    fno_parameters = count_parameters(model.smooth_model)
    scot_parameters = count_parameters(model.contrast_model)
    total_parameters = fno_parameters + scot_parameters

    optimizer = torch.optim.AdamW(
        model.contrast_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = write_run_manifest(
        output_dir,
        args,
        run_name=run_name,
        model_type="hybrid_scot_finetune",
        parameters=total_parameters,
    )
    print(f"Training on {device}")
    print(
        f"Architecture parameters: {total_parameters:,} "
        f"(frozen FNO {fno_parameters:,}; trainable scOT {scot_parameters:,})"
    )
    print(f"Run manifest: {manifest_path}")

    wandb_run = maybe_init_wandb(args, run_name)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_metadata = {
        "initial_fno_checkpoint": str(initial_fno),
        "initial_scot_checkpoint": str(initial_scot),
        "fno_parameters": fno_parameters,
        "scot_parameters": scot_parameters,
        "parameters": total_parameters,
        "fno_frozen": True,
        "training_epochs": 1,
        "constant_learning_rate": args.learning_rate,
    }
    try:
        baseline_validation = evaluate_hybrid_epoch(
            model,
            validation_loader,
            device,
            epoch=0,
        )
        baseline_metrics = EpochMetrics(
            epoch=0,
            train_mean_relative_l2=None,
            validation_mean_relative_l2=baseline_validation,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        append_epoch_metrics(metrics_path, baseline_metrics)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": 0,
                    "validation/mean_relative_l2": baseline_validation,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=0,
            )
        print(f"Epoch 0 pretrained validation relative L2: {baseline_validation:.6f}")

        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_error = train_hybrid_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        validation_error = evaluate_hybrid_epoch(
            model,
            validation_loader,
            device,
            epoch=1,
        )
        metrics = EpochMetrics(
            epoch=1,
            train_mean_relative_l2=train_error,
            validation_mean_relative_l2=validation_error,
            learning_rate=learning_rate,
        )
        append_epoch_metrics(metrics_path, metrics)
        save_hybrid_checkpoint(
            model,
            output_dir / "checkpoint",
            epoch=1,
            validation_mean_relative_l2=validation_error,
            extra_metadata={
                **checkpoint_metadata,
                "baseline_validation_mean_relative_l2": baseline_validation,
                "improved_validation": validation_error < baseline_validation,
            },
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": 1,
                    "train/mean_relative_l2": train_error,
                    "validation/mean_relative_l2": validation_error,
                    "learning_rate": learning_rate,
                },
                step=1,
            )

        completion = {
            "status": "completed",
            "completed_epochs": 1,
            "requested_epochs": 1,
            "baseline_validation_mean_relative_l2": baseline_validation,
            "finetuned_validation_mean_relative_l2": validation_error,
            "improved_validation": validation_error < baseline_validation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (output_dir / "training_complete.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n"
        )
        if wandb_run is not None:
            wandb_run.summary["baseline_validation_mean_relative_l2"] = baseline_validation
            wandb_run.summary["finetuned_validation_mean_relative_l2"] = validation_error
            wandb_run.summary["improved_validation"] = validation_error < baseline_validation
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(
        f"Completed 1 scOT fine-tuning epoch at constant LR {args.learning_rate:g}. "
        f"Validation relative L2: {baseline_validation:.6f} -> {validation_error:.6f}."
    )
    print(f"Frozen FNO checkpoint: {output_dir / 'checkpoint' / 'fno'}")
    print(f"Fine-tuned scOT checkpoint: {output_dir / 'checkpoint' / 'scot'}")
    return 0


def main() -> int:
    """Run the CLI with concise errors suitable for unattended GPU jobs."""

    args = parse_args()
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, FloatingPointError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
