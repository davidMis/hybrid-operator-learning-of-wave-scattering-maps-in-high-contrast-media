#!/usr/bin/env python3
# Overview:
# Train a scOT transformer on one paper task. The script adapts the processed
# Helmholtz datasets to scOT's HuggingFace-style Trainer, saves the best model in
# scOT/transformers format, and records a run manifest with the exact training
# arguments and runtime metadata.
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

from helmholtz_hybrid.data import ScOTDatasetWrapper, load_task_dataset
from helmholtz_hybrid.cli_config import apply_yaml_defaults, config_path_from_argv
from helmholtz_hybrid.loss import complex_L2_norm
from helmholtz_hybrid.reproducibility import configure_torch_reproducibility, write_run_manifest
from helmholtz_hybrid.runtime import set_default_cache_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a scOT transformer checkpoint for a paper learning task.",
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
            "(v_delta,p_smooth) to p_delta; sharp2sharp maps v_sharp to p_sharp."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory where run subdirectories and scOT checkpoints will be written. "
            "Defaults to outputs/checkpoints/<dataset>/seed<seed>/scot."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run directory name; defaults to a task/depth/seed timestamp.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training and validation batch size per device.",
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
        help="Number of warmup epochs represented as a Trainer warmup ratio.",
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
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm passed to scOT's Trainer.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help="Spatial patch size used by the scOT image tokenizer.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=90,
        help="Base embedding dimension; paper runs use 90.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Shared depth for all four scOT stages when --depths is not provided.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs=4,
        default=None,
        help="Four per-stage depths; paper sweep uses n n n n for n in 2,4,6,8,10.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs=4,
        default=[3, 6, 12, 24],
        help="Attention-head counts for the four scOT stages.",
    )
    parser.add_argument(
        "--skip-connections",
        type=int,
        nargs=4,
        default=[2, 2, 2, 0],
        help="Skip-connection configuration for the four scOT stages.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Swin-style local attention window size.",
    )
    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
        help="Hidden expansion ratio in transformer MLP blocks.",
    )
    parser.add_argument(
        "--loss-p",
        type=int,
        default=2,
        help="p value passed into ScOTConfig for relative Lp-style training loss.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes used by the Trainer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for Python, NumPy, PyTorch, CUDA, and Trainer data shuffling.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for model initialization, for example cuda, cuda:0, or cpu.",
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
        args.output_root = Path("outputs/checkpoints") / args.dataset / f"seed{args.seed}" / "scot"
    else:
        args.output_root = Path(args.output_root)
    return args


def set_seed(seed: int) -> None:
    """Seed local random number generators before model creation."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    stamp = time.strftime("%Y%m%d-%H%M%S")
    depth_label = "-".join(str(d) for d in (args.depths or [args.depth] * 4))
    return f"scot_{args.dataset}_{args.task}_depths{depth_label}_seed{args.seed}_{stamp}"


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


def unpack_eval_preds(eval_preds):
    """Handle either tuple outputs or HuggingFace EvalPrediction objects."""

    if hasattr(eval_preds, "predictions"):
        return eval_preds.predictions, eval_preds.label_ids
    return eval_preds


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_torch_reproducibility(args.deterministic, args.allow_tf32)
    device = torch.device(args.device)
    run_name = make_run_name(args)
    wandb_run = maybe_init_wandb(args, run_name)

    # Configure cache paths before importing scOT/transformers. This avoids
    # user-home cache writes on shared compute nodes.
    set_default_cache_dirs()
    from scOT.model import ScOT, ScOTConfig
    from scOT.trainer import Trainer, TrainingArguments
    from scOT.utils import get_num_parameters

    # ScOT expects pixel_values/labels keys and dataset metadata such as
    # resolution/input_dim/output_dim; the wrapper supplies those from helmholtz_hybrid data.
    train_dataset = ScOTDatasetWrapper(
        load_task_dataset(args.data_root, args.dataset, "train", args.task),
        which="train",
    )
    validation_dataset = ScOTDatasetWrapper(
        load_task_dataset(args.data_root, args.dataset, "validation", args.task),
        which="val",
    )

    depths = args.depths if args.depths is not None else [args.depth] * 4
    # These defaults match the paper's Poseidon-B-style scOT configuration, with
    # depth as the main capacity sweep variable.
    model_config = ScOTConfig(
        image_size=train_dataset.resolution,
        patch_size=args.patch_size,
        num_channels=train_dataset.input_dim,
        num_out_channels=train_dataset.output_dim,
        embed_dim=args.embed_dim,
        depths=depths,
        num_heads=args.num_heads,
        skip_connections=args.skip_connections,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        p=args.loss_p,
        qkv_bias=True,
        drop_path_rate=0.0,
        residual_model="convnext",
        use_conditioning=False,
        learn_residual=False,
    )
    model = ScOT(model_config).to(device)
    parameters = int(get_num_parameters(model))
    print(f"Model parameters: {parameters}")

    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_run_manifest(
        output_dir,
        args,
        run_name=run_name,
        model_type="scot",
        parameters=parameters,
    )
    print(f"Writing checkpoints to {output_dir}")
    print(f"Writing run manifest to {manifest_path}")
    # scOT uses a HuggingFace-style Trainer. load_best_model_at_end selects the
    # best validation checkpoint by relative L2 error before save_model writes the
    # final reloadable config/weights in output_dir.
    train_config = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_accumulation_steps=16,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        optim="adamw_hf",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=float(args.warmup_epochs) / float(args.epochs),
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        dataloader_num_workers=args.workers,
        load_best_model_at_end=True,
        metric_for_best_model="rel_l2_err",
        seed=args.seed,
        data_seed=args.seed,
        report_to=["wandb"] if wandb_run is not None else [],
        run_name=run_name,
        greater_is_better=False,
    )

    def compute_metrics(eval_preds):
        # Validation metric mirrors the paper's relative complex L2 definition.
        predictions, labels = unpack_eval_preds(eval_preds)
        diff = labels - predictions
        rel_l2_err = (
            complex_L2_norm(torch.as_tensor(diff))
            / (complex_L2_norm(torch.as_tensor(labels)) + 1e-8)
        ).mean()
        return {"rel_l2_err": float(rel_l2_err)}

    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(str(output_dir))
    completed_epochs = getattr(trainer.state, "epoch", None) or args.epochs
    (output_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_epochs": int(float(completed_epochs)),
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
