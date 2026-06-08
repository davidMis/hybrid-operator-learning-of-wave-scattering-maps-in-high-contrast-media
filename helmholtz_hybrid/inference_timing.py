# Overview:
# Provide reusable utilities for timing checkpoint inference throughput for the
# paper model sweep. The CLI scheduler builds jobs from the released checkpoint
# layout, while this module loads FNO/scOT/hybrid models, performs forward-only
# warmup passes, times forward-only inference, and returns structured records
# for the manuscript table.
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import torch
from torch.utils.data import DataLoader

from helmholtz_hybrid.data import (
    TASKS,
    ScOTDatasetWrapper,
    dataloader_performance_kwargs,
    load_task_dataset,
)
from helmholtz_hybrid.evaluation import (
    load_fno_checkpoint,
    load_scot_checkpoint,
    model_parameter_count,
    scot_predictions,
)
from helmholtz_hybrid.runtime import set_default_cache_dirs


SINGLE_MODEL_TASKS = tuple(task for task in TASKS if task != "hybrid")
MODEL_TYPES = ("fno", "scot", "hybrid")
DEFAULT_SWEEP_SIZES = (2, 4, 6, 8, 10)
EVALUATION_SPLITS = ("validation", "test")
TASK_TO_PANEL = {
    "smooth2smooth": "Smooth",
    "contrast": "Residual",
    "sharp2sharp": "Sharp",
    "hybrid": "Sharp",
}
MODEL_LABEL = {
    "fno": "FNO",
    "scot": "scOT",
    "hybrid": "Hybrid",
}
TASK_REQUIRED_SPLIT_FILES = {
    "smooth2smooth": ("velocity_smooth.npy", "pressure_smooth.npy"),
    "contrast": ("velocity_delta.npy", "pressure_smooth.npy", "pressure_delta.npy"),
    "sharp2sharp": ("velocity_sharp.npy", "pressure_sharp.npy"),
    "hybrid": ("velocity_smooth.npy", "velocity_delta.npy", "pressure_sharp.npy"),
}

ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True)
class InferenceTimingJob:
    """One Figure 4 checkpoint inference timing job."""

    model_type: str
    task: str
    size: int
    checkpoint: Path | None = None
    fno_smooth_checkpoint: Path | None = None
    scot_contrast_checkpoint: Path | None = None

    @property
    def label(self) -> str:
        if self.model_type == "fno":
            capacity = f"layers={self.size}"
        elif self.model_type == "scot":
            capacity = f"depths={self.size}-{self.size}-{self.size}-{self.size}"
        else:
            capacity = f"size={self.size}"
        return f"{self.model_type} {self.task} {capacity}"

    @property
    def source_checkpoints(self) -> list[Path]:
        if self.model_type == "hybrid":
            return [
                path
                for path in (self.fno_smooth_checkpoint, self.scot_contrast_checkpoint)
                if path is not None
            ]
        return [self.checkpoint] if self.checkpoint is not None else []


@dataclass(frozen=True)
class InferenceTimingSettings:
    """Shared benchmark settings used by every inference timing worker."""

    data_root: Path
    dataset: str
    split: str = "test"
    batch_size: int = 64
    workers: int = 4
    warmup_passes: int = 1
    timed_passes: int = 1
    max_batches: int | None = None
    preload_device_batches: bool = True


@dataclass(frozen=True)
class InferenceTimingResult:
    """Structured timing result for one checkpoint inference job."""

    panel: str
    task: str
    model_type: str
    model: str
    size: int
    parameters: int
    num_samples: int
    batch_size: int
    batches: int
    warmup_passes: int
    timed_passes: int
    timed_pass_seconds: tuple[float, ...]
    seconds_per_sample: float
    milliseconds_per_sample: float
    preload_device_batches: bool
    device: str
    device_name: str
    checkpoint: str

    def to_csv_row(self) -> dict[str, str]:
        """Return a stable long-form CSV representation."""

        return {
            "panel": self.panel,
            "task": self.task,
            "model": self.model,
            "model_type": self.model_type,
            "size": str(self.size),
            "parameters": str(self.parameters),
            "num_samples": str(self.num_samples),
            "batch_size": str(self.batch_size),
            "batches": str(self.batches),
            "warmup_passes": str(self.warmup_passes),
            "timed_passes": str(self.timed_passes),
            "timed_pass_seconds": json.dumps(_rounded_list(self.timed_pass_seconds)),
            "seconds_per_sample": f"{self.seconds_per_sample:.9f}",
            "milliseconds_per_sample": f"{self.milliseconds_per_sample:.6f}",
            "preload_device_batches": str(self.preload_device_batches),
            "device": self.device,
            "device_name": self.device_name,
            "checkpoint": self.checkpoint,
        }


@dataclass
class _LoadedInferenceJob:
    model_type: str
    dataset: object
    model: object | None = None
    smooth_model: object | None = None
    contrast_model: object | None = None
    parameters: int = 0


def fno_checkpoint_name(dataset: str, task: str, size: int) -> str:
    """Return the FNO run directory name used by run_all.sh and the paper release."""

    return f"fno_{dataset}_{task}_layers{size}"


def scot_checkpoint_name(dataset: str, task: str, size: int) -> str:
    """Return the scOT run directory name used by run_all.sh and the paper release."""

    return f"scot_{dataset}_{task}_depths{size}-{size}-{size}-{size}"


def build_inference_timing_jobs(
    checkpoint_root: str | Path,
    dataset: str,
    tasks: list[str] | tuple[str, ...],
    sizes: list[int] | tuple[int, ...],
    *,
    include_hybrid: bool = True,
) -> list[InferenceTimingJob]:
    """Return the ordered standalone and hybrid Figure 4 inference timing jobs."""

    unknown_tasks = sorted(set(tasks) - set(SINGLE_MODEL_TASKS))
    if unknown_tasks:
        raise ValueError(
            f"Unknown task(s): {', '.join(unknown_tasks)}. Expected one of {SINGLE_MODEL_TASKS}."
        )
    if any(size < 1 for size in sizes):
        raise ValueError(f"Timing sizes must be positive integers; got {list(sizes)}.")

    root = Path(checkpoint_root)
    jobs: list[InferenceTimingJob] = []
    for task in tasks:
        for size in sizes:
            jobs.append(
                InferenceTimingJob(
                    model_type="fno",
                    task=task,
                    size=int(size),
                    checkpoint=root / "fno" / fno_checkpoint_name(dataset, task, size),
                )
            )
            jobs.append(
                InferenceTimingJob(
                    model_type="scot",
                    task=task,
                    size=int(size),
                    checkpoint=root / "scot" / scot_checkpoint_name(dataset, task, size),
                )
            )

    if include_hybrid:
        for size in sizes:
            jobs.append(
                InferenceTimingJob(
                    model_type="hybrid",
                    task="hybrid",
                    size=int(size),
                    fno_smooth_checkpoint=root
                    / "fno"
                    / fno_checkpoint_name(dataset, "smooth2smooth", size),
                    scot_contrast_checkpoint=root
                    / "scot"
                    / scot_checkpoint_name(dataset, "contrast", size),
                )
            )
    return jobs


def missing_inference_inputs(
    data_root: str | Path,
    dataset: str,
    split: str,
    jobs: list[InferenceTimingJob],
) -> list[Path]:
    """Return split arrays or checkpoint directories that are absent from disk."""

    missing: list[Path] = []
    split_dir = Path(data_root) / dataset / split
    required_tasks = sorted({job.task for job in jobs})
    for task in required_tasks:
        for filename in TASK_REQUIRED_SPLIT_FILES[task]:
            path = split_dir / filename
            if not path.is_file():
                missing.append(path)

    for job in jobs:
        for checkpoint in job.source_checkpoints:
            if not checkpoint.is_dir():
                missing.append(checkpoint)
    return missing


def time_inference_job(
    job: InferenceTimingJob,
    settings: InferenceTimingSettings,
    device: str,
    progress_callback: ProgressCallback | None = None,
) -> InferenceTimingResult:
    """Run one forward-only inference benchmark for a checkpoint job."""

    _validate_timing_request(job, settings)
    set_default_cache_dirs()
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this process.")
        torch.cuda.set_device(torch_device)

    loaded = _load_inference_job(job, settings, torch_device)
    parameters = loaded.parameters
    loader = _make_loader(loaded, settings, torch_device)
    batches, sample_count = _device_batches_or_count(
        loaded,
        loader,
        settings,
        torch_device,
    )
    if sample_count == 0:
        raise RuntimeError("Inference loader produced no samples; check the dataset and --max-batches.")
    batch_count = _batch_count_from_batches(batches, loader, settings)

    total_passes = settings.warmup_passes + settings.timed_passes
    try:
        for _ in range(settings.warmup_passes):
            _run_inference_pass(
                loaded,
                batches,
                loader,
                settings,
                torch_device,
                progress_callback,
                total_passes,
                timed=False,
            )

        timed_seconds: list[float] = []
        for _ in range(settings.timed_passes):
            timed_seconds.append(
                _run_inference_pass(
                    loaded,
                    batches,
                    loader,
                    settings,
                    torch_device,
                    progress_callback,
                    total_passes,
                    timed=True,
                )
            )
    finally:
        del loaded
        del batches
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    seconds_per_sample = float(mean(timed_seconds)) / float(sample_count)
    return InferenceTimingResult(
        panel=TASK_TO_PANEL[job.task],
        task=job.task,
        model_type=job.model_type,
        model=MODEL_LABEL[job.model_type],
        size=job.size,
        parameters=parameters,
        num_samples=sample_count,
        batch_size=settings.batch_size,
        batches=batch_count,
        warmup_passes=settings.warmup_passes,
        timed_passes=settings.timed_passes,
        timed_pass_seconds=tuple(timed_seconds),
        seconds_per_sample=seconds_per_sample,
        milliseconds_per_sample=seconds_per_sample * 1_000.0,
        preload_device_batches=settings.preload_device_batches,
        device=device,
        device_name=_device_name(torch_device),
        checkpoint=";".join(str(path) for path in job.source_checkpoints),
    )


def _validate_timing_request(job: InferenceTimingJob, settings: InferenceTimingSettings) -> None:
    if job.model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type '{job.model_type}'. Expected one of {MODEL_TYPES}.")
    if job.task not in TASK_REQUIRED_SPLIT_FILES:
        raise ValueError(f"Unknown task '{job.task}'. Expected one of {tuple(TASK_REQUIRED_SPLIT_FILES)}.")
    if settings.split not in EVALUATION_SPLITS:
        raise ValueError(f"Unknown split '{settings.split}'. Expected one of {EVALUATION_SPLITS}.")
    if settings.batch_size < 1:
        raise ValueError(f"Inference batch size must be positive; got {settings.batch_size}.")
    if settings.workers < 0:
        raise ValueError(f"DataLoader worker count must be non-negative; got {settings.workers}.")
    if settings.warmup_passes < 0:
        raise ValueError(f"Warmup pass count must be non-negative; got {settings.warmup_passes}.")
    if settings.timed_passes < 1:
        raise ValueError(f"Timed pass count must be positive; got {settings.timed_passes}.")
    if settings.max_batches is not None and settings.max_batches < 1:
        raise ValueError(f"--max-batches must be positive when provided; got {settings.max_batches}.")
    if job.model_type in {"fno", "scot"} and job.checkpoint is None:
        raise ValueError(f"{job.model_type} timing requires a checkpoint directory.")
    if job.model_type == "hybrid" and (
        job.fno_smooth_checkpoint is None or job.scot_contrast_checkpoint is None
    ):
        raise ValueError("Hybrid timing requires FNO smooth and scOT contrast checkpoints.")


def _load_inference_job(
    job: InferenceTimingJob,
    settings: InferenceTimingSettings,
    device: torch.device,
) -> _LoadedInferenceJob:
    data_root = Path(settings.data_root)
    if job.model_type == "fno":
        assert job.checkpoint is not None
        dataset = load_task_dataset(data_root, settings.dataset, settings.split, job.task)
        model = load_fno_checkpoint(job.checkpoint, device)
        model.eval()
        return _LoadedInferenceJob(
            model_type=job.model_type,
            dataset=dataset,
            model=model,
            parameters=model_parameter_count(model),
        )
    if job.model_type == "scot":
        assert job.checkpoint is not None
        dataset = load_task_dataset(data_root, settings.dataset, settings.split, job.task)
        model = load_scot_checkpoint(job.checkpoint, device)
        model.eval()
        return _LoadedInferenceJob(
            model_type=job.model_type,
            dataset=dataset,
            model=model,
            parameters=model_parameter_count(model),
        )

    assert job.fno_smooth_checkpoint is not None and job.scot_contrast_checkpoint is not None
    dataset = load_task_dataset(data_root, settings.dataset, settings.split, "hybrid")
    smooth_model = load_fno_checkpoint(job.fno_smooth_checkpoint, device)
    contrast_model = load_scot_checkpoint(job.scot_contrast_checkpoint, device)
    smooth_model.eval()
    contrast_model.eval()
    return _LoadedInferenceJob(
        model_type=job.model_type,
        dataset=dataset,
        smooth_model=smooth_model,
        contrast_model=contrast_model,
        parameters=model_parameter_count(smooth_model) + model_parameter_count(contrast_model),
    )


def _make_loader(
    loaded: _LoadedInferenceJob,
    settings: InferenceTimingSettings,
    device: torch.device,
) -> DataLoader:
    dataset = loaded.dataset
    if loaded.model_type == "scot":
        dataset = ScOTDatasetWrapper(dataset, which=settings.split)
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        **dataloader_performance_kwargs(settings.workers, device.type == "cuda"),
    )


def _device_batches_or_count(
    loaded: _LoadedInferenceJob,
    loader: DataLoader,
    settings: InferenceTimingSettings,
    device: torch.device,
) -> tuple[list[torch.Tensor] | None, int]:
    if not settings.preload_device_batches:
        sample_count = 0
        for batch_index, batch in enumerate(loader):
            if settings.max_batches is not None and batch_index >= settings.max_batches:
                break
            sample_count += _batch_sample_count(batch, loaded.model_type)
        return None, sample_count

    batches: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if settings.max_batches is not None and batch_index >= settings.max_batches:
            break
        batches.append(_batch_input(batch, loaded.model_type).to(device, non_blocking=device.type == "cuda"))
    sample_count = sum(int(batch.shape[0]) for batch in batches)
    return batches, sample_count


def _run_inference_pass(
    loaded: _LoadedInferenceJob,
    device_batches: list[torch.Tensor] | None,
    loader: DataLoader,
    settings: InferenceTimingSettings,
    device: torch.device,
    progress_callback: ProgressCallback | None,
    total_passes: int,
    *,
    timed: bool,
) -> float:
    if device_batches is not None:
        return _run_preloaded_pass(
            loaded,
            device_batches,
            progress_callback,
            total_passes,
            timed=timed,
        )
    return _run_streamed_pass(
        loaded,
        loader,
        settings,
        device,
        progress_callback,
        total_passes,
        timed=timed,
    )


def _run_preloaded_pass(
    loaded: _LoadedInferenceJob,
    device_batches: list[torch.Tensor],
    progress_callback: ProgressCallback | None,
    total_passes: int,
    *,
    timed: bool,
) -> float:
    device = device_batches[0].device
    _synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in device_batches:
            _forward_batch(loaded, batch)
            if progress_callback is not None:
                progress_callback(1, total_passes * len(device_batches))
    _synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed if timed else 0.0


def _run_streamed_pass(
    loaded: _LoadedInferenceJob,
    loader: DataLoader,
    settings: InferenceTimingSettings,
    device: torch.device,
    progress_callback: ProgressCallback | None,
    total_passes: int,
    *,
    timed: bool,
) -> float:
    elapsed = 0.0
    total_batches = _limited_batch_count(loader, settings)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if settings.max_batches is not None and batch_index >= settings.max_batches:
                break
            x = _batch_input(batch, loaded.model_type).to(device, non_blocking=device.type == "cuda")
            _synchronize(device)
            start = time.perf_counter()
            _forward_batch(loaded, x)
            _synchronize(device)
            elapsed += time.perf_counter() - start
            if progress_callback is not None:
                progress_callback(1, total_passes * total_batches)
    return elapsed if timed else 0.0


def _forward_batch(loaded: _LoadedInferenceJob, x: torch.Tensor) -> None:
    if loaded.model_type == "fno":
        assert loaded.model is not None
        loaded.model(x)
    elif loaded.model_type == "scot":
        assert loaded.model is not None
        scot_predictions(loaded.model(x))
    else:
        assert loaded.smooth_model is not None and loaded.contrast_model is not None
        velocity_smooth = x[:, 0:1]
        velocity_delta = x[:, 1:2]
        pressure_smooth = loaded.smooth_model(velocity_smooth)
        contrast_input = torch.cat([velocity_delta, pressure_smooth], dim=1)
        pressure_delta = scot_predictions(loaded.contrast_model(contrast_input))
        pressure_smooth + pressure_delta


def _batch_input(batch: dict[str, torch.Tensor], model_type: str) -> torch.Tensor:
    if model_type == "scot":
        return batch["pixel_values"]
    return batch["x"]


def _batch_sample_count(batch: dict[str, torch.Tensor], model_type: str) -> int:
    return int(_batch_input(batch, model_type).shape[0])


def _batch_count_from_batches(
    batches: list[torch.Tensor] | None,
    loader: DataLoader,
    settings: InferenceTimingSettings,
) -> int:
    if batches is not None:
        return len(batches)
    return _limited_batch_count(loader, settings)


def _limited_batch_count(loader: DataLoader, settings: InferenceTimingSettings) -> int:
    if settings.max_batches is None:
        return len(loader)
    return min(len(loader), settings.max_batches)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_name(index)
    return str(device)


def _rounded_list(values: tuple[float, ...]) -> list[float]:
    return [round(float(value), 6) for value in values]
