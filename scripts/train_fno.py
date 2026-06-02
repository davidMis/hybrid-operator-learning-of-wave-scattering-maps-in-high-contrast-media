#!/usr/bin/env python3
# Overview:
# Train a Fourier Neural Operator on one of the three paper tasks:
# smooth2smooth, contrast, or sharp2sharp. The script reads the processed data
# layout from prepare_data.py, optimizes relative complex L2 loss, saves the best
# validation checkpoint in NeuralOperator format, and records a run manifest for
# reproducibility.
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
from torch.utils.data import DataLoader

from helmholtz_hybrid.data import TASKS, load_task_dataset
from helmholtz_hybrid.cli_config import apply_yaml_defaults, config_path_from_argv
from helmholtz_hybrid.loss import ComplexL2Loss
from helmholtz_hybrid.reproducibility import (
    configure_torch_reproducibility,
    count_parameters,
    write_run_manifest,
)
from helmholtz_hybrid.runtime import set_default_cache_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an FNO checkpoint for a paper learning task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional flat YAML file containing default values for model and training options.",
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
        "--task",
        choices=["smooth2smooth", "contrast", "sharp2sharp"],
        default="contrast",
        help=(
            "Learning task: smooth2smooth maps v_smooth to p_smooth; contrast maps "
            "(v_delta,p_smooth) to p_delta; sharp2sharp maps v_sharp to p_sharp. "
            f"Available tasks: {TASKS}."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory where run subdirectories and checkpoints will be written. "
            "Defaults to outputs/checkpoints/<dataset>/seed<seed>/fno."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run directory name; defaults to a task/layer/seed timestamp.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training and validation batch size per process.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of FNO spectral convolution layers; paper sweep uses 2,4,6,8,10.",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=64,
        help="Number of retained Fourier modes in each spatial dimension.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Width of the FNO hidden representation.",
    )
    parser.add_argument(
        "--domain-padding",
        type=float,
        default=0.125,
        help="Fractional domain padding passed to neuralop.models.FNO.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs; paper protocol uses 100.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of linear learning-rate warmup epochs before cosine decay.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Peak AdamW learning rate reached after warmup.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW decoupled weight decay coefficient.",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=3e-6,
        help="Minimum learning rate for the cosine decay schedule.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes for each split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for Python, NumPy, PyTorch, CUDA, and DataLoader shuffling.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for training, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request deterministic CUDA kernels where PyTorch supports them.",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow TF32 matrix math on supported NVIDIA GPUs for faster exploratory runs.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Optional Weights & Biases project name; omit to disable W&B logging.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional Weights & Biases entity/team used when --wandb-project is set.",
    )
    apply_yaml_defaults(parser, config_path_from_argv())
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = Path("outputs/checkpoints") / args.dataset / f"seed{args.seed}" / "fno"
    else:
        args.output_root = Path(args.output_root)
    return args


def set_seed(seed: int) -> torch.Generator:
    """Seed local random number generators and return a DataLoader generator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"fno_{args.dataset}_{args.task}_layers{args.num_layers}_seed{args.seed}_{stamp}"


def maybe_init_wandb(args: argparse.Namespace, run_name: str):
    if args.wandb_project is None:
        return None
    import wandb

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        config=config,
    )


def main() -> None:
    args = parse_args()
    generator = set_seed(args.seed)
    configure_torch_reproducibility(args.deterministic, args.allow_tf32)
    device = torch.device(args.device)
    run_name = make_run_name(args)
    wandb_run = maybe_init_wandb(args, run_name)

    # Delay heavy imports until cache paths are configured. This keeps
    # third-party libraries from writing into user-level caches on shared nodes.
    set_default_cache_dirs()
    import neuralop
    from neuralop import Trainer
    from neuralop.models import FNO

    # The dataset object determines the channel count for each task, so the same
    # model construction path handles smooth, residual, and sharp mappings.
    train_dataset = load_task_dataset(args.data_root, args.dataset, "train", args.task)
    validation_dataset = load_task_dataset(args.data_root, args.dataset, "validation", args.task)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    # The paper sweep varies num_layers while keeping 64 modes and 64 hidden
    # channels by default.
    model = FNO(
        n_layers=args.num_layers,
        n_modes=(args.num_modes, args.num_modes),
        hidden_channels=args.hidden_channels,
        in_channels=train_dataset.input_channels,
        out_channels=train_dataset.output_channels,
        domain_padding=args.domain_padding,
    )
    parameters = count_parameters(model)
    print(f"Model parameters: {parameters}")

    loss = ComplexL2Loss()
    optimizer = neuralop.training.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Appendix C specifies AdamW, a 5-epoch linear warmup, and cosine decay.
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=args.eta_min,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs],
    )

    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # The manifest makes each checkpoint directory self-describing even when W&B
    # is disabled or unavailable.
    manifest_path = write_run_manifest(
        output_dir,
        args,
        run_name=run_name,
        model_type="fno",
        parameters=parameters,
    )
    print(f"Writing checkpoints to {output_dir}")
    print(f"Writing run manifest to {manifest_path}")

    # neuralop.Trainer writes best_model_state_dict.pt and model metadata when
    # save_best is set; helmholtz_hybrid.evaluation reloads that native checkpoint format.
    trainer = Trainer(
        model=model,
        n_epochs=args.epochs,
        device=device,
        wandb_log=wandb_run is not None,
        eval_interval=1,
        log_output=False,
        use_distributed=False,
        verbose=True,
    )
    trainer.train(
        train_loader=train_loader,
        test_loaders={"validation": validation_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=loss,
        eval_losses={"rel_complex_L2": loss},
        save_best="validation_rel_complex_L2",
        save_dir=str(output_dir),
        resume_from_dir=None,
    )
    (output_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_epochs": args.epochs,
                "requested_epochs": args.epochs,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
        + "\n"
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
