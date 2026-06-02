# Overview:
# Load trained FNO/scOT checkpoints and run batched inference for paper metrics
# and visualizations. The module exposes both low-level prediction loops and a
# reusable evaluate_checkpoint API used by CLI wrappers and run_all.sh.
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from helmholtz_hybrid.data import (
    HybridDataset,
    ScOTDatasetWrapper,
    dataloader_performance_kwargs,
    load_task_dataset,
)
from helmholtz_hybrid.loss import relative_complex_l2
from helmholtz_hybrid.reproducibility import PARAMETER_COUNT_METHOD, count_parameters
from helmholtz_hybrid.runtime import set_default_cache_dirs


MODEL_TYPES = ("fno", "scot", "hybrid")
SINGLE_MODEL_TASKS = ("smooth2smooth", "contrast", "sharp2sharp")
EVALUATION_SPLITS = ("validation", "test")
ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True)
class EvaluationRequest:
    """Inputs required to evaluate one checkpoint family on one paper split."""

    data_root: str | Path
    dataset: str
    model_type: str
    split: str = "test"
    task: str | None = None
    checkpoint: str | Path | None = None
    fno_smooth_checkpoint: str | Path | None = None
    scot_contrast_checkpoint: str | Path | None = None
    batch_size: int = 64
    workers: int = 4
    store_predictions: bool = False
    device: str | torch.device | None = None
    progress_prefix: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and optional arrays produced by one evaluation request."""

    metrics: dict[str, Any]
    expected: np.ndarray | None
    actual: np.ndarray | None
    per_sample_relative_l2: np.ndarray


@dataclass(frozen=True)
class PredictionResult:
    """Per-sample metrics and optional full-field arrays from batched inference."""

    per_sample_relative_l2: np.ndarray
    expected: np.ndarray | None = None
    actual: np.ndarray | None = None


def default_device() -> str:
    """Return the preferred inference device for this machine."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_evaluation_request(request: EvaluationRequest) -> None:
    """Raise helpful errors for incomplete or inconsistent evaluation inputs."""

    if request.model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type '{request.model_type}'. Expected one of {MODEL_TYPES}.")
    if request.split not in EVALUATION_SPLITS:
        raise ValueError(f"Unknown split '{request.split}'. Expected one of {EVALUATION_SPLITS}.")
    if request.batch_size < 1:
        raise ValueError(f"Evaluation batch size must be positive; got {request.batch_size}.")
    if request.workers < 0:
        raise ValueError(f"Evaluation worker count must be non-negative; got {request.workers}.")

    if request.model_type in {"fno", "scot"}:
        if request.task not in SINGLE_MODEL_TASKS:
            raise ValueError(f"{request.model_type} evaluation requires --task from {SINGLE_MODEL_TASKS}.")
        if request.checkpoint is None:
            raise ValueError(f"{request.model_type} evaluation requires a checkpoint directory.")
    elif request.fno_smooth_checkpoint is None or request.scot_contrast_checkpoint is None:
        raise ValueError("Hybrid evaluation requires FNO smooth and scOT contrast checkpoint directories.")


def evaluate_checkpoint(
    request: EvaluationRequest,
    *,
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
    progress_position: int = 0,
) -> EvaluationResult:
    """Evaluate one FNO, scOT, or hybrid checkpoint specification."""

    validate_evaluation_request(request)
    device = torch.device(request.device or default_device())
    data_root = Path(request.data_root)

    # Single-model evaluation scores the native task target. For contrast, that
    # means p_delta, not reconstructed full pressure.
    if request.model_type == "fno":
        assert request.task is not None and request.checkpoint is not None
        dataset = load_task_dataset(data_root, request.dataset, request.split, request.task)
        model = load_fno_checkpoint(request.checkpoint, device)
        prediction = evaluate_fno(
            model,
            dataset,
            device,
            batch_size=request.batch_size,
            workers=request.workers,
            store_predictions=request.store_predictions,
            progress_label=progress_label(request, f"FNO {request.dataset} {request.task}"),
            progress_callback=progress_callback,
            show_progress=show_progress,
            progress_position=progress_position,
        )
        parameters = model_parameter_count(model)
    elif request.model_type == "scot":
        assert request.task is not None and request.checkpoint is not None
        dataset = load_task_dataset(data_root, request.dataset, request.split, request.task)
        model = load_scot_checkpoint(request.checkpoint, device)
        prediction = evaluate_scot(
            model,
            dataset,
            device,
            batch_size=request.batch_size,
            workers=request.workers,
            store_predictions=request.store_predictions,
            progress_label=progress_label(request, f"scOT {request.dataset} {request.task}"),
            progress_callback=progress_callback,
            show_progress=show_progress,
            progress_position=progress_position,
        )
        parameters = model_parameter_count(model)
    else:
        assert request.fno_smooth_checkpoint is not None and request.scot_contrast_checkpoint is not None
        dataset = load_task_dataset(data_root, request.dataset, request.split, "hybrid")
        smooth_model = load_fno_checkpoint(request.fno_smooth_checkpoint, device)
        contrast_model = load_scot_checkpoint(request.scot_contrast_checkpoint, device)
        prediction = evaluate_hybrid(
            smooth_model=smooth_model,
            contrast_model=contrast_model,
            dataset=dataset,
            device=device,
            batch_size=request.batch_size,
            workers=request.workers,
            store_predictions=request.store_predictions,
            progress_label=progress_label(request, f"hybrid {request.dataset} sharp"),
            progress_callback=progress_callback,
            show_progress=show_progress,
            progress_position=progress_position,
        )
        parameters = model_parameter_count(smooth_model) + model_parameter_count(contrast_model)

    per_sample = prediction.per_sample_relative_l2
    metrics = {
        "dataset": request.dataset,
        "split": request.split,
        "model_type": request.model_type,
        "task": request.task or "hybrid",
        "num_samples": int(per_sample.shape[0]),
        "parameters": int(parameters),
        "parameter_count_method": PARAMETER_COUNT_METHOD,
        "mean_relative_l2": float(np.mean(per_sample)),
        "median_relative_l2": float(np.median(per_sample)),
    }
    return EvaluationResult(
        metrics=metrics,
        expected=prediction.expected,
        actual=prediction.actual,
        per_sample_relative_l2=per_sample,
    )


def progress_label(request: EvaluationRequest, label: str) -> str:
    """Prefix tqdm labels with run_all.sh context when provided."""

    if request.progress_prefix:
        return f"{request.progress_prefix} {label}"
    return label


def write_evaluation_outputs(
    result: EvaluationResult,
    *,
    output_json: str | Path | None = None,
    predictions_out: str | Path | None = None,
) -> None:
    """Persist metrics and optional prediction arrays from an evaluation run."""

    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.metrics, indent=2) + "\n")
    if predictions_out is not None:
        if result.expected is None or result.actual is None:
            raise ValueError("Prediction arrays were not retained; rerun evaluation with prediction storage enabled.")
        predictions_path = Path(predictions_out)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_path,
            expected=result.expected,
            actual=result.actual,
            rel_l2=result.per_sample_relative_l2,
        )


def relative_l2_per_sample(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return relative_complex_l2(torch.as_tensor(actual), torch.as_tensor(expected)).numpy()


def mean_relative_l2(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(relative_l2_per_sample(expected, actual)))


def model_parameter_count(model) -> int:
    return count_parameters(model)


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a batch tensor with asynchronous copies when CUDA pinning is active."""

    return tensor.to(device, non_blocking=device.type == "cuda")


def make_evaluation_loader(dataset, batch_size: int, device: torch.device, workers: int) -> DataLoader:
    """Construct a DataLoader tuned for inference throughput."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_performance_kwargs(workers, device.type == "cuda"),
    )


def iter_with_progress(
    loader: DataLoader,
    *,
    progress_label: str,
    progress_callback: ProgressCallback | None,
    show_progress: bool,
    progress_position: int,
):
    """Yield batches while reporting progress either locally or to a caller."""

    total_batches = len(loader)
    if progress_callback is not None:
        progress_callback(0, total_batches)
    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            desc=progress_label,
            unit="batch",
            dynamic_ncols=True,
            position=progress_position,
            disable=not sys.stderr.isatty(),
        )
    for batch in iterator:
        yield batch
        if progress_callback is not None:
            progress_callback(1, total_batches)


def collect_prediction_batch(
    expected_chunks: list[np.ndarray],
    actual_chunks: list[np.ndarray],
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    """Append CPU prediction arrays only for explicit prediction export runs."""

    expected_chunks.append(expected.detach().cpu().numpy())
    actual_chunks.append(actual.detach().cpu().numpy())


def prediction_result(
    per_sample_chunks: list[np.ndarray],
    expected_chunks: list[np.ndarray],
    actual_chunks: list[np.ndarray],
    store_predictions: bool,
) -> PredictionResult:
    """Assemble streamed scalar metrics and optional full-field prediction arrays."""

    expected = np.concatenate(expected_chunks, axis=0) if store_predictions else None
    actual = np.concatenate(actual_chunks, axis=0) if store_predictions else None
    return PredictionResult(
        per_sample_relative_l2=np.concatenate(per_sample_chunks, axis=0),
        expected=expected,
        actual=actual,
    )


def load_fno_checkpoint(checkpoint: str | Path, device: torch.device):
    set_default_cache_dirs()
    from neuralop.models import FNO

    return FNO.from_checkpoint(
        save_folder=str(checkpoint),
        save_name="best_model",
        map_location=device,
    ).to(device)


def resolve_scot_checkpoint(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    if (checkpoint / "config.json").is_file():
        return checkpoint

    checkpoint_re = re.compile(r"^checkpoint-(\d+)$")
    candidates: list[tuple[int, Path]] = []
    for child in checkpoint.iterdir():
        if child.is_dir():
            match = checkpoint_re.match(child.name)
            if match:
                candidates.append((int(match.group(1)), child))

    if not candidates:
        raise FileNotFoundError(f"No scOT config.json or checkpoint-N directory found in {checkpoint}")
    return max(candidates, key=lambda item: item[0])[1]


def load_scot_checkpoint(checkpoint: str | Path, device: torch.device):
    set_default_cache_dirs()
    from scOT.model import ScOT

    model = ScOT.from_pretrained(resolve_scot_checkpoint(checkpoint))
    model.to(device)
    model.eval()
    return model


def scot_predictions(output) -> torch.Tensor:
    if hasattr(output, "predictions"):
        return output.predictions
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "output"):
        return output.output
    return output[0]


def evaluate_fno(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    workers: int = 4,
    store_predictions: bool = False,
    progress_label: str = "FNO evaluation",
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
    progress_position: int = 0,
) -> PredictionResult:
    model.eval()
    loader = make_evaluation_loader(dataset, batch_size, device, workers)
    per_sample_chunks: list[np.ndarray] = []
    expected_chunks: list[np.ndarray] = []
    actual_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in iter_with_progress(
            loader,
            progress_label=progress_label,
            progress_callback=progress_callback,
            show_progress=show_progress,
            progress_position=progress_position,
        ):
            expected = move_to_device(batch["y"], device)
            actual = model(move_to_device(batch["x"], device))
            per_sample_chunks.append(relative_complex_l2(actual, expected).cpu().numpy())
            if store_predictions:
                collect_prediction_batch(expected_chunks, actual_chunks, expected, actual)

    return prediction_result(per_sample_chunks, expected_chunks, actual_chunks, store_predictions)


def evaluate_scot(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    workers: int = 4,
    store_predictions: bool = False,
    progress_label: str = "scOT evaluation",
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
    progress_position: int = 0,
) -> PredictionResult:
    model.eval()
    wrapped = ScOTDatasetWrapper(dataset, which="test")
    loader = make_evaluation_loader(wrapped, batch_size, device, workers)
    per_sample_chunks: list[np.ndarray] = []
    expected_chunks: list[np.ndarray] = []
    actual_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in iter_with_progress(
            loader,
            progress_label=progress_label,
            progress_callback=progress_callback,
            show_progress=show_progress,
            progress_position=progress_position,
        ):
            expected = move_to_device(batch["labels"], device)
            output = model(move_to_device(batch["pixel_values"], device))
            actual = scot_predictions(output)
            per_sample_chunks.append(relative_complex_l2(actual, expected).cpu().numpy())
            if store_predictions:
                collect_prediction_batch(expected_chunks, actual_chunks, expected, actual)

    return prediction_result(per_sample_chunks, expected_chunks, actual_chunks, store_predictions)


def evaluate_hybrid(
    smooth_model,
    contrast_model,
    dataset: HybridDataset,
    device: torch.device,
    batch_size: int = 64,
    workers: int = 4,
    store_predictions: bool = False,
    progress_label: str = "hybrid evaluation",
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
    progress_position: int = 0,
) -> PredictionResult:
    smooth_model.eval()
    contrast_model.eval()
    loader = make_evaluation_loader(dataset, batch_size, device, workers)
    per_sample_chunks: list[np.ndarray] = []
    expected_chunks: list[np.ndarray] = []
    actual_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in iter_with_progress(
            loader,
            progress_label=progress_label,
            progress_callback=progress_callback,
            show_progress=show_progress,
            progress_position=progress_position,
        ):
            x = move_to_device(batch["x"], device)
            expected = move_to_device(batch["y"], device)
            velocity_smooth = x[:, 0:1]
            velocity_delta = x[:, 1:2]
            pressure_smooth = smooth_model(velocity_smooth)
            contrast_input = torch.cat([velocity_delta, pressure_smooth], dim=1)
            pressure_delta = scot_predictions(contrast_model(contrast_input))
            actual = pressure_smooth + pressure_delta
            per_sample_chunks.append(relative_complex_l2(actual, expected).cpu().numpy())
            if store_predictions:
                collect_prediction_batch(expected_chunks, actual_chunks, expected, actual)

    return prediction_result(per_sample_chunks, expected_chunks, actual_chunks, store_predictions)
