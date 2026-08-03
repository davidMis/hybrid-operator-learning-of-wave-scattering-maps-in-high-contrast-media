# Overview:
# Compose a frozen smooth-pressure FNO with a trainable residual-pressure scOT.
# This module owns the reusable forward pass, epoch loops, metrics log, and
# paired native-format checkpoint writer used by scripts/finetune_hybrid.py.
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from helmholtz_hybrid.evaluation import move_to_device, scot_predictions
from helmholtz_hybrid.loss import relative_complex_l2


@dataclass(frozen=True)
class EpochMetrics:
    """Mean per-sample relative complex L2 errors for one completed epoch."""

    epoch: int
    train_mean_relative_l2: float | None
    validation_mean_relative_l2: float
    learning_rate: float


class FrozenFNOHybridOperator(nn.Module):
    """Hybrid operator that adapts scOT while keeping its FNO input fixed."""

    def __init__(self, smooth_model: nn.Module, contrast_model: nn.Module) -> None:
        super().__init__()
        self.smooth_model = smooth_model
        self.contrast_model = contrast_model
        self.smooth_model.requires_grad_(False)
        self.smooth_model.eval()

    def train(self, mode: bool = True) -> "FrozenFNOHybridOperator":
        """Set scOT's mode while keeping the frozen FNO in evaluation mode."""

        super().train(mode)
        self.smooth_model.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map ``[v_smooth, v_delta]`` to ``p_smooth + p_delta``."""

        if inputs.ndim != 4 or inputs.shape[1] != 2:
            raise ValueError(
                "End-to-end hybrid input must have shape [batch, 2, height, width] "
                f"with channels [v_smooth, v_delta]; got {tuple(inputs.shape)}."
            )
        velocity_smooth = inputs[:, 0:1]
        velocity_delta = inputs[:, 1:2]
        # The FNO output is an input feature for scOT, not part of the trainable
        # graph. no_grad both enforces that contract and avoids storing FNO
        # activations during the scOT update.
        with torch.no_grad():
            pressure_smooth = self.smooth_model(velocity_smooth)
        contrast_input = torch.cat([velocity_delta, pressure_smooth], dim=1)
        pressure_delta = scot_predictions(self.contrast_model(contrast_input))
        return pressure_smooth + pressure_delta


def train_hybrid_epoch(
    model: FrozenFNOHybridOperator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    progress_position: int = 1,
) -> float:
    """Train only the residual scOT for one epoch and return sample-mean error."""

    if gradient_accumulation_steps < 1:
        raise ValueError(
            "Gradient accumulation steps must be positive; "
            f"got {gradient_accumulation_steps}."
        )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    error_sum = 0.0
    sample_count = 0
    total_batches = len(loader)
    iterator = tqdm(
        loader,
        desc=f"train epoch {epoch}",
        unit="batch",
        dynamic_ncols=True,
        position=progress_position,
        leave=False,
        disable=not sys.stderr.isatty(),
    )

    for batch_index, batch in enumerate(iterator, start=1):
        inputs = move_to_device(batch["x"], device)
        targets = move_to_device(batch["y"], device)
        predictions = model(inputs)
        per_sample_error = relative_complex_l2(predictions, targets)
        loss = per_sample_error.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at epoch {epoch}, batch {batch_index}."
            )
        (loss / gradient_accumulation_steps).backward()

        should_step = (
            batch_index % gradient_accumulation_steps == 0 or batch_index == total_batches
        )
        if should_step:
            # Correct the scale of a short final accumulation group.
            accumulated_batches = (batch_index - 1) % gradient_accumulation_steps + 1
            if accumulated_batches < gradient_accumulation_steps:
                scale = gradient_accumulation_steps / accumulated_batches
                for parameter in optimizer_parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(optimizer_parameters, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(targets.shape[0])
        error_sum += float(per_sample_error.detach().sum())
        sample_count += batch_size
        iterator.set_postfix_str(f"relL2={error_sum / sample_count:.4f}")

    if sample_count == 0:
        raise RuntimeError("The training DataLoader produced no samples.")
    return error_sum / sample_count


@torch.no_grad()
def evaluate_hybrid_epoch(
    model: FrozenFNOHybridOperator,
    loader: DataLoader,
    device: torch.device,
    *,
    epoch: int,
    progress_position: int = 1,
) -> float:
    """Evaluate the composed sharp-pressure prediction without storing fields."""

    model.eval()
    error_sum = 0.0
    sample_count = 0
    iterator = tqdm(
        loader,
        desc=f"validation epoch {epoch}",
        unit="batch",
        dynamic_ncols=True,
        position=progress_position,
        leave=False,
        disable=not sys.stderr.isatty(),
    )
    for batch in iterator:
        inputs = move_to_device(batch["x"], device)
        targets = move_to_device(batch["y"], device)
        per_sample_error = relative_complex_l2(model(inputs), targets)
        batch_size = int(targets.shape[0])
        error_sum += float(per_sample_error.sum())
        sample_count += batch_size
        iterator.set_postfix_str(f"relL2={error_sum / sample_count:.4f}")

    if sample_count == 0:
        raise RuntimeError("The validation DataLoader produced no samples.")
    return error_sum / sample_count


def append_epoch_metrics(path: str | Path, metrics: EpochMetrics) -> None:
    """Append one JSON record so interrupted runs retain completed metrics."""

    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")


def save_hybrid_checkpoint(
    model: FrozenFNOHybridOperator,
    output_dir: str | Path,
    *,
    epoch: int,
    validation_mean_relative_l2: float,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Save a matched FNO/scOT pair in formats accepted by evaluation.py."""

    output_dir = Path(output_dir)
    fno_dir = output_dir / "fno"
    scot_dir = output_dir / "scot"
    fno_dir.mkdir(parents=True, exist_ok=True)
    scot_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model.smooth_model, "save_checkpoint"):
        raise TypeError("The smooth model does not provide neuralop's save_checkpoint method.")
    if not hasattr(model.contrast_model, "save_pretrained"):
        raise TypeError("The contrast model does not provide scOT's save_pretrained method.")

    model.smooth_model.save_checkpoint(fno_dir, "best_model")
    model.contrast_model.save_pretrained(scot_dir)
    metadata: dict[str, Any] = {
        "epoch": int(epoch),
        "validation_mean_relative_l2": float(validation_mean_relative_l2),
        "fno_checkpoint": "fno",
        "scot_checkpoint": "scot",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    (output_dir / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
