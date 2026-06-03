# Overview:
# Provide reusable utilities for timing short training runs from the paper model
# sweep. The CLI script schedules these timing jobs across GPUs, while this
# module builds the FNO/scOT models, runs warmup and timed training epochs, and
# returns structured records that can be rendered into the publication table.
from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from helmholtz_hybrid.data import (
    TASKS,
    ScOTDatasetWrapper,
    dataloader_performance_kwargs,
    load_task_dataset,
)
from helmholtz_hybrid.loss import ComplexL2Loss
from helmholtz_hybrid.reproducibility import (
    configure_torch_reproducibility,
    count_parameters,
)
from helmholtz_hybrid.runtime import set_default_cache_dirs


SINGLE_MODEL_TASKS = tuple(task for task in TASKS if task != "hybrid")
MODEL_TYPES = ("fno", "scot")
DEFAULT_SWEEP_SIZES = (2, 4, 6, 8, 10)
TASK_TO_PANEL = {
    "smooth2smooth": "Smooth",
    "contrast": "Residual",
    "sharp2sharp": "Sharp",
}
MODEL_LABEL = {
    "fno": "FNO",
    "scot": "scOT",
}
TASK_REQUIRED_TRAIN_FILES = {
    "smooth2smooth": ("velocity_smooth.npy", "pressure_smooth.npy"),
    "contrast": ("velocity_delta.npy", "pressure_smooth.npy", "pressure_delta.npy"),
    "sharp2sharp": ("velocity_sharp.npy", "pressure_sharp.npy"),
}

ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True)
class TimingJob:
    """One standalone Figure 4 model timing job."""

    model_type: str
    task: str
    size: int

    @property
    def label(self) -> str:
        if self.model_type == "fno":
            capacity = f"layers={self.size}"
        else:
            capacity = f"depths={self.size}-{self.size}-{self.size}-{self.size}"
        return f"{self.model_type} {self.task} {capacity}"


@dataclass(frozen=True)
class TimingSettings:
    """Shared benchmark settings used by every timing worker."""

    data_root: Path
    dataset: str
    fno_config: Mapping[str, object]
    scot_config: Mapping[str, object]
    warmup_epochs: int = 1
    timed_epochs: int = 3
    seed: int = 123
    max_batches: int | None = None
    batch_size_override: int | None = None
    workers_override: int | None = None
    deterministic_override: bool | None = None
    allow_tf32_override: bool | None = None

    def config_for(self, model_type: str) -> Mapping[str, object]:
        if model_type == "fno":
            return self.fno_config
        if model_type == "scot":
            return self.scot_config
        raise ValueError(f"Unknown model type '{model_type}'. Expected one of {MODEL_TYPES}.")

    def batch_size_for(self, model_type: str) -> int:
        if self.batch_size_override is not None:
            return self.batch_size_override
        return int(self.config_for(model_type).get("batch_size", 32))

    def workers_for(self, model_type: str) -> int:
        if self.workers_override is not None:
            return self.workers_override
        return int(self.config_for(model_type).get("workers", 4))

    def deterministic_for(self, model_type: str) -> bool:
        if self.deterministic_override is not None:
            return self.deterministic_override
        return bool(self.config_for(model_type).get("deterministic", False))

    def allow_tf32_for(self, model_type: str) -> bool:
        if self.allow_tf32_override is not None:
            return self.allow_tf32_override
        return bool(self.config_for(model_type).get("allow_tf32", False))


@dataclass(frozen=True)
class TrainingTimingResult:
    """Structured timing result for one standalone model."""

    panel: str
    task: str
    model_type: str
    model: str
    size: int
    parameters: int
    train_samples: int
    batch_size: int
    batches_per_epoch: int
    warmup_epoch_seconds: tuple[float, ...]
    timed_epoch_seconds: tuple[float, ...]
    seconds_per_epoch: float
    minutes_per_epoch: float
    device: str
    device_name: str

    def to_csv_row(self) -> dict[str, str]:
        """Return a stable long-form CSV representation."""

        return {
            "panel": self.panel,
            "task": self.task,
            "model": self.model,
            "model_type": self.model_type,
            "size": str(self.size),
            "parameters": str(self.parameters),
            "train_samples": str(self.train_samples),
            "batch_size": str(self.batch_size),
            "batches_per_epoch": str(self.batches_per_epoch),
            "warmup_epoch_seconds": json.dumps(_rounded_list(self.warmup_epoch_seconds)),
            "timed_epoch_seconds": json.dumps(_rounded_list(self.timed_epoch_seconds)),
            "seconds_per_epoch": f"{self.seconds_per_epoch:.6f}",
            "minutes_per_epoch": f"{self.minutes_per_epoch:.6f}",
            "device": self.device,
            "device_name": self.device_name,
        }


def load_yaml_config(path: str | Path) -> dict[str, object]:
    """Load a flat YAML config file with helpful validation errors."""

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Configuration file does not exist: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Could not parse YAML configuration {config_path}: {error}") from error
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")
    return {str(key): value for key, value in payload.items()}


def build_timing_jobs(
    tasks: list[str] | tuple[str, ...],
    model_types: list[str] | tuple[str, ...],
    sizes: list[int] | tuple[int, ...],
) -> list[TimingJob]:
    """Return the ordered standalone Figure 4 timing jobs."""

    unknown_tasks = sorted(set(tasks) - set(SINGLE_MODEL_TASKS))
    if unknown_tasks:
        raise ValueError(
            f"Unknown task(s): {', '.join(unknown_tasks)}. Expected one of {SINGLE_MODEL_TASKS}."
        )
    unknown_models = sorted(set(model_types) - set(MODEL_TYPES))
    if unknown_models:
        raise ValueError(
            f"Unknown model type(s): {', '.join(unknown_models)}. Expected one of {MODEL_TYPES}."
        )
    if any(size < 1 for size in sizes):
        raise ValueError(f"Timing sizes must be positive integers; got {list(sizes)}.")

    jobs: list[TimingJob] = []
    for task in tasks:
        for model_type in model_types:
            for size in sizes:
                jobs.append(TimingJob(model_type=model_type, task=task, size=int(size)))
    return jobs


def missing_training_files(
    data_root: str | Path,
    dataset: str,
    tasks: list[str] | tuple[str, ...],
) -> list[Path]:
    """Return required train-split arrays that are absent from disk."""

    missing: list[Path] = []
    train_dir = Path(data_root) / dataset / "train"
    for task in tasks:
        for filename in TASK_REQUIRED_TRAIN_FILES[task]:
            path = train_dir / filename
            if not path.is_file():
                missing.append(path)
    return missing


def time_training_job(
    job: TimingJob,
    settings: TimingSettings,
    device: str,
    progress_callback: ProgressCallback | None = None,
) -> TrainingTimingResult:
    """Run one warmup/timed training benchmark for a standalone model."""

    _validate_timing_request(job, settings)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_default_cache_dirs()

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this process.")
        torch.cuda.set_device(torch_device)

    _set_seed(settings.seed + _stable_job_offset(job))
    configure_torch_reproducibility(
        deterministic=settings.deterministic_for(job.model_type),
        allow_tf32=settings.allow_tf32_for(job.model_type),
    )

    train_dataset = _load_training_dataset(job, settings)
    batch_size = settings.batch_size_for(job.model_type)
    workers = settings.workers_for(job.model_type)
    generator = torch.Generator()
    generator.manual_seed(settings.seed + _stable_job_offset(job))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **dataloader_performance_kwargs(workers, torch_device.type == "cuda"),
    )
    model, optimizer, parameters = _build_model_and_optimizer(
        job,
        settings,
        train_dataset,
        torch_device,
    )

    total_epochs = settings.warmup_epochs + settings.timed_epochs
    warmup_seconds: list[float] = []
    timed_seconds: list[float] = []
    batches_per_epoch = 0
    try:
        for epoch_index in range(total_epochs):
            elapsed_seconds, batches = _run_training_epoch(
                model=model,
                optimizer=optimizer,
                loader=train_loader,
                device=torch_device,
                model_type=job.model_type,
                max_batches=settings.max_batches,
            )
            batches_per_epoch = batches
            if epoch_index < settings.warmup_epochs:
                warmup_seconds.append(elapsed_seconds)
            else:
                timed_seconds.append(elapsed_seconds)
            if progress_callback is not None:
                progress_callback(1, total_epochs)
    finally:
        del model
        del optimizer
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    seconds_per_epoch = float(mean(timed_seconds))
    return TrainingTimingResult(
        panel=TASK_TO_PANEL[job.task],
        task=job.task,
        model_type=job.model_type,
        model=MODEL_LABEL[job.model_type],
        size=job.size,
        parameters=int(parameters),
        train_samples=len(train_dataset),
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        warmup_epoch_seconds=tuple(warmup_seconds),
        timed_epoch_seconds=tuple(timed_seconds),
        seconds_per_epoch=seconds_per_epoch,
        minutes_per_epoch=seconds_per_epoch / 60.0,
        device=device,
        device_name=_device_name(torch_device),
    )


def _validate_timing_request(job: TimingJob, settings: TimingSettings) -> None:
    if job.model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type '{job.model_type}'. Expected one of {MODEL_TYPES}.")
    if job.task not in SINGLE_MODEL_TASKS:
        raise ValueError(f"Unknown task '{job.task}'. Expected one of {SINGLE_MODEL_TASKS}.")
    if settings.warmup_epochs < 0:
        raise ValueError(f"Warmup epoch count must be non-negative; got {settings.warmup_epochs}.")
    if settings.timed_epochs < 1:
        raise ValueError(f"Timed epoch count must be positive; got {settings.timed_epochs}.")
    if settings.max_batches is not None and settings.max_batches < 1:
        raise ValueError(f"--max-batches must be positive when provided; got {settings.max_batches}.")
    if settings.batch_size_for(job.model_type) < 1:
        raise ValueError(f"Batch size must be positive; got {settings.batch_size_for(job.model_type)}.")


def _load_training_dataset(job: TimingJob, settings: TimingSettings):
    dataset = load_task_dataset(settings.data_root, settings.dataset, "train", job.task)
    if job.model_type == "scot":
        return ScOTDatasetWrapper(dataset, which="train")
    return dataset


def _build_model_and_optimizer(
    job: TimingJob,
    settings: TimingSettings,
    train_dataset,
    device: torch.device,
):
    if job.model_type == "fno":
        return _build_fno(job, settings, train_dataset, device)
    if job.model_type == "scot":
        return _build_scot(job, settings, train_dataset, device)
    raise ValueError(f"Unknown model type '{job.model_type}'. Expected one of {MODEL_TYPES}.")


def _build_fno(job: TimingJob, settings: TimingSettings, train_dataset, device: torch.device):
    import neuralop
    from neuralop.models import FNO

    config = settings.fno_config
    num_modes = int(config.get("num_modes", 64))
    model = FNO(
        n_layers=job.size,
        n_modes=(num_modes, num_modes),
        hidden_channels=int(config.get("hidden_channels", 64)),
        in_channels=train_dataset.input_channels,
        out_channels=train_dataset.output_channels,
        domain_padding=float(config.get("domain_padding", 0.125)),
    ).to(device)
    optimizer = neuralop.training.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1.0e-3)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
    )
    return model, optimizer, count_parameters(model)


def _build_scot(job: TimingJob, settings: TimingSettings, train_dataset, device: torch.device):
    from scOT.model import ScOT, ScOTConfig
    from scOT.utils import get_num_parameters

    config = settings.scot_config
    depths = [job.size] * 4
    model_config = ScOTConfig(
        image_size=train_dataset.resolution,
        patch_size=int(config.get("patch_size", 4)),
        num_channels=train_dataset.input_dim,
        num_out_channels=train_dataset.output_dim,
        embed_dim=int(config.get("embed_dim", 90)),
        depths=depths,
        num_heads=list(config.get("num_heads", [3, 6, 12, 24])),
        skip_connections=list(config.get("skip_connections", [2, 2, 2, 0])),
        window_size=int(config.get("window_size", 8)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        p=int(config.get("loss_p", 2)),
        qkv_bias=True,
        drop_path_rate=0.0,
        residual_model="convnext",
        use_conditioning=False,
        learn_residual=False,
    )
    model = ScOT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1.0e-3)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
    )
    return model, optimizer, int(get_num_parameters(model))


def _run_training_epoch(
    *,
    model,
    optimizer,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
    max_batches: int | None,
) -> tuple[float, int]:
    loss_fn = ComplexL2Loss()
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = _training_loss(model, batch, device, model_type, loss_fn)
        loss.backward()
        optimizer.step()
        batches += 1
    if batches == 0:
        raise RuntimeError(
            "Training loader produced no batches; check the dataset and --max-batches."
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start, batches


def _training_loss(
    model,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    model_type: str,
    loss_fn,
):
    if model_type == "fno":
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        prediction = model(x)
        return loss_fn(prediction, y)
    if model_type == "scot":
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        outputs = model(pixel_values=pixel_values, labels=labels)
        return outputs.loss.mean()
    raise ValueError(f"Unknown model type '{model_type}'. Expected one of {MODEL_TYPES}.")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_job_offset(job: TimingJob) -> int:
    model_offset = MODEL_TYPES.index(job.model_type) * 10_000
    task_offset = SINGLE_MODEL_TASKS.index(job.task) * 100
    return model_offset + task_offset + job.size


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_name(index)
    return str(device)


def _rounded_list(values: tuple[float, ...]) -> list[float]:
    return [round(float(value), 6) for value in values]
