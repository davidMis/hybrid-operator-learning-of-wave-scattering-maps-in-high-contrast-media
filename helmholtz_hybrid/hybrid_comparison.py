# Overview:
# Evaluate the hybrid Helmholtz model with either exact or FNO-predicted smooth
# pressure while holding the residual scOT and sharp-pressure target fixed. The
# module also writes the machine-readable and LaTeX tables used for the smooth-
# source ablation, leaving the CLI wrapper responsible only for orchestration.
from __future__ import annotations

import csv
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import Subset

from helmholtz_hybrid.data import HybridSmoothSourceDataset, split_dir
from helmholtz_hybrid.evaluation import (
    iter_with_progress,
    load_fno_checkpoint,
    load_scot_checkpoint,
    make_evaluation_loader,
    model_parameter_count,
    move_to_device,
    resolve_scot_checkpoint,
    scot_predictions,
)
from helmholtz_hybrid.loss import relative_complex_l2


DEFAULT_SWEEP_SIZES = (2, 4, 6, 8, 10)
COMPARISON_SPLITS = ("validation", "test")
REQUIRED_SPLIT_FILES = (
    "velocity_smooth.npy",
    "velocity_delta.npy",
    "pressure_smooth.npy",
    "pressure_sharp.npy",
)
CSV_FIELDNAMES = (
    "sweep_size",
    "fno_parameters",
    "scot_parameters",
    "hybrid_parameters",
    "num_samples",
    "split",
    "ground_truth_mean_relative_l2",
    "trained_fno_mean_relative_l2",
    "absolute_increase",
    "percent_increase",
    "ground_truth_median_relative_l2",
    "trained_fno_median_relative_l2",
    "fno_smooth_checkpoint",
    "scot_contrast_checkpoint",
)
ProgressCallback = Callable[[int, int | None], None]


@dataclass(frozen=True)
class HybridSmoothSourceResult:
    """Metrics for one capacity-matched FNO/scOT hybrid pair."""

    sweep_size: int
    fno_parameters: int
    scot_parameters: int
    num_samples: int
    split: str
    ground_truth_mean_relative_l2: float
    trained_fno_mean_relative_l2: float
    ground_truth_median_relative_l2: float
    trained_fno_median_relative_l2: float
    fno_smooth_checkpoint: str
    scot_contrast_checkpoint: str

    @property
    def hybrid_parameters(self) -> int:
        """Return the total trainable parameters in the composed hybrid."""

        return self.fno_parameters + self.scot_parameters

    @property
    def absolute_increase(self) -> float:
        """Return the trained-FNO error minus the ground-truth-smooth error."""

        return self.trained_fno_mean_relative_l2 - self.ground_truth_mean_relative_l2

    @property
    def percent_increase(self) -> float:
        """Return the relative error increase caused by using the trained FNO."""

        baseline = self.ground_truth_mean_relative_l2
        if baseline == 0.0:
            return float("inf") if self.trained_fno_mean_relative_l2 > 0.0 else 0.0
        return 100.0 * self.absolute_increase / baseline

    def to_csv_row(self) -> dict[str, str]:
        """Return a stable wide-form row for downstream analysis."""

        return {
            "sweep_size": str(self.sweep_size),
            "fno_parameters": str(self.fno_parameters),
            "scot_parameters": str(self.scot_parameters),
            "hybrid_parameters": str(self.hybrid_parameters),
            "num_samples": str(self.num_samples),
            "split": self.split,
            "ground_truth_mean_relative_l2": f"{self.ground_truth_mean_relative_l2:.12g}",
            "trained_fno_mean_relative_l2": f"{self.trained_fno_mean_relative_l2:.12g}",
            "absolute_increase": f"{self.absolute_increase:.12g}",
            "percent_increase": f"{self.percent_increase:.12g}",
            "ground_truth_median_relative_l2": f"{self.ground_truth_median_relative_l2:.12g}",
            "trained_fno_median_relative_l2": f"{self.trained_fno_median_relative_l2:.12g}",
            "fno_smooth_checkpoint": self.fno_smooth_checkpoint,
            "scot_contrast_checkpoint": self.scot_contrast_checkpoint,
        }


def fno_smooth_checkpoint(checkpoint_root: str | Path, dataset: str, size: int) -> Path:
    """Return the released smooth-task FNO checkpoint directory for one size."""

    return (
        Path(checkpoint_root)
        / "fno"
        / f"fno_{dataset}_smooth2smooth_layers{size}"
    )


def scot_contrast_checkpoint(checkpoint_root: str | Path, dataset: str, size: int) -> Path:
    """Return the released residual-task scOT checkpoint directory for one size."""

    return (
        Path(checkpoint_root)
        / "scot"
        / f"scot_{dataset}_contrast_depths{size}-{size}-{size}-{size}"
    )


def missing_comparison_inputs(
    data_root: str | Path,
    dataset: str,
    split: str,
    checkpoint_root: str | Path,
    sizes: Iterable[int],
) -> list[Path]:
    """Return required dataset files or checkpoint directories that are absent."""

    missing: list[Path] = []
    input_directory = split_dir(data_root, dataset, split)
    for filename in REQUIRED_SPLIT_FILES:
        path = input_directory / filename
        if not path.is_file():
            missing.append(path)
    for size in sizes:
        fno_path = fno_smooth_checkpoint(checkpoint_root, dataset, size)
        if not fno_path.is_dir():
            missing.append(fno_path)
        else:
            for filename in ("best_model_metadata.pkl", "best_model_state_dict.pt"):
                path = fno_path / filename
                if not path.is_file():
                    missing.append(path)

        scot_path = scot_contrast_checkpoint(checkpoint_root, dataset, size)
        if not scot_path.is_dir():
            missing.append(scot_path)
            continue
        try:
            resolved_scot_path = resolve_scot_checkpoint(scot_path)
        except FileNotFoundError:
            missing.append(scot_path / "config.json")
            continue
        weight_candidates = (
            resolved_scot_path / "pytorch_model.bin",
            resolved_scot_path / "model.safetensors",
        )
        if not any(path.is_file() for path in weight_candidates):
            missing.append(weight_candidates[0])
    return missing


def compare_checkpoint_pair(
    *,
    data_root: str | Path,
    dataset: str,
    split: str,
    checkpoint_root: str | Path,
    size: int,
    device: str | torch.device,
    batch_size: int = 64,
    workers: int = 4,
    max_samples: int | None = None,
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
    progress_position: int = 0,
) -> HybridSmoothSourceResult:
    """Load one capacity-matched checkpoint pair and evaluate both smooth sources."""

    if split not in COMPARISON_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {COMPARISON_SPLITS}.")
    if size < 1:
        raise ValueError(f"Sweep size must be positive; got {size}.")
    if batch_size < 1:
        raise ValueError(f"Evaluation batch size must be positive; got {batch_size}.")
    if workers < 0:
        raise ValueError(f"DataLoader worker count must be non-negative; got {workers}.")
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"--max-samples must be positive; got {max_samples}.")

    torch_device = torch.device(device)
    smooth_checkpoint = fno_smooth_checkpoint(checkpoint_root, dataset, size)
    contrast_checkpoint = scot_contrast_checkpoint(checkpoint_root, dataset, size)
    input_directory = split_dir(data_root, dataset, split)
    source_dataset = HybridSmoothSourceDataset.load(input_directory)
    evaluation_dataset = source_dataset
    if max_samples is not None and max_samples < len(source_dataset):
        evaluation_dataset = Subset(source_dataset, range(max_samples))

    smooth_model = load_fno_checkpoint(smooth_checkpoint, torch_device)
    contrast_model = load_scot_checkpoint(contrast_checkpoint, torch_device)
    try:
        ground_truth_errors, trained_fno_errors = evaluate_smooth_sources(
            smooth_model=smooth_model,
            contrast_model=contrast_model,
            dataset=evaluation_dataset,
            device=torch_device,
            batch_size=batch_size,
            workers=workers,
            progress_label=f"n={size} smooth-source comparison",
            progress_callback=progress_callback,
            show_progress=show_progress,
            progress_position=progress_position,
        )
        return HybridSmoothSourceResult(
            sweep_size=size,
            fno_parameters=model_parameter_count(smooth_model),
            scot_parameters=model_parameter_count(contrast_model),
            num_samples=int(ground_truth_errors.shape[0]),
            split=split,
            ground_truth_mean_relative_l2=float(np.mean(ground_truth_errors)),
            trained_fno_mean_relative_l2=float(np.mean(trained_fno_errors)),
            ground_truth_median_relative_l2=float(np.median(ground_truth_errors)),
            trained_fno_median_relative_l2=float(np.median(trained_fno_errors)),
            fno_smooth_checkpoint=str(smooth_checkpoint),
            scot_contrast_checkpoint=str(contrast_checkpoint),
        )
    finally:
        del smooth_model
        del contrast_model
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()


def evaluate_smooth_sources(
    *,
    smooth_model,
    contrast_model,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    workers: int = 4,
    progress_label: str = "smooth-source comparison",
    progress_callback: ProgressCallback | None = None,
    show_progress: bool = True,
    progress_position: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Score exact and FNO smooth-pressure inputs against the same sharp target."""

    smooth_model.eval()
    contrast_model.eval()
    loader = make_evaluation_loader(dataset, batch_size, device, workers)
    ground_truth_chunks: list[np.ndarray] = []
    trained_fno_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in iter_with_progress(
            loader,
            progress_label=progress_label,
            progress_callback=progress_callback,
            show_progress=show_progress,
            progress_position=progress_position,
        ):
            x = move_to_device(batch["x"], device)
            expected_sharp = move_to_device(batch["y"], device)
            exact_smooth = move_to_device(batch["pressure_smooth"], device)
            velocity_smooth = x[:, 0:1]
            velocity_delta = x[:, 1:2]
            trained_fno_smooth = smooth_model(velocity_smooth)

            exact_delta = scot_predictions(
                contrast_model(torch.cat([velocity_delta, exact_smooth], dim=1))
            )
            trained_fno_delta = scot_predictions(
                contrast_model(torch.cat([velocity_delta, trained_fno_smooth], dim=1))
            )
            exact_reconstruction = exact_smooth + exact_delta
            trained_fno_reconstruction = trained_fno_smooth + trained_fno_delta

            ground_truth_chunks.append(
                relative_complex_l2(exact_reconstruction, expected_sharp).cpu().numpy()
            )
            trained_fno_chunks.append(
                relative_complex_l2(trained_fno_reconstruction, expected_sharp).cpu().numpy()
            )

    if not ground_truth_chunks:
        raise RuntimeError("Evaluation produced no batches; check the selected split and batch size.")
    return np.concatenate(ground_truth_chunks), np.concatenate(trained_fno_chunks)


def write_comparison_outputs(
    results: Iterable[HybridSmoothSourceResult],
    *,
    csv_path: str | Path,
    latex_path: str | Path,
) -> None:
    """Write sorted wide-form CSV and manuscript-ready LaTeX table outputs."""

    sorted_results = sorted(results, key=lambda result: result.sweep_size)
    if not sorted_results:
        raise ValueError("Cannot write an empty smooth-source comparison.")
    sizes = [result.sweep_size for result in sorted_results]
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"Duplicate sweep sizes are not allowed: {sizes}.")

    output_csv = Path(csv_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(result.to_csv_row() for result in sorted_results)

    output_latex = Path(latex_path)
    output_latex.parent.mkdir(parents=True, exist_ok=True)
    output_latex.write_text(render_latex_table(sorted_results))


def render_latex_table(results: list[HybridSmoothSourceResult]) -> str:
    """Render the compact n-column table used in the manuscript."""

    splits = {result.split for result in results}
    sample_counts = {result.num_samples for result in results}
    if len(splits) != 1 or len(sample_counts) != 1:
        raise ValueError(
            "LaTeX table rows must use one common split and sample count; got "
            f"splits={sorted(splits)}, sample_counts={sorted(sample_counts)}."
        )
    split = next(iter(splits))
    num_samples = next(iter(sample_counts))
    column_spec = "l" + ("r" * len(results))
    end_column = len(results) + 1
    header = " & ".join(str(result.sweep_size) for result in results)
    ground_truth = " & ".join(
        f"{result.ground_truth_mean_relative_l2:.4f}" for result in results
    )
    trained_fno = " & ".join(
        f"{result.trained_fno_mean_relative_l2:.4f}" for result in results
    )
    increases = " & ".join(f"{result.percent_increase:.1f}\\%" for result in results)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Effect of the smooth-pressure source on hybrid-model test accuracy. "
        "The column variable $n$ denotes both the number of FNO layers and the shared "
        "scOT depth in each of its four stages. Values are mean relative complex "
        f"$L^2$ errors on {num_samples:,} samples from the {split} split. Both rows "
        "reconstruct the full sharp pressure and are evaluated against the same "
        "numerical target.}\n"
        "\\label{tab:hybrid-smooth-source}\n"
        f"\\begin{{tabular}}{{{column_spec}}}\n"
        "\\toprule\n"
        f"Smooth-pressure source & \\multicolumn{{{len(results)}}}{{c}}{{Sweep variable $n$}} \\\\\n"
        f"\\cmidrule(lr){{2-{end_column}}}\n"
        f"& {header} \\\\\n"
        "\\midrule\n"
        f"Ground truth & {ground_truth} \\\\\n"
        f"Trained FNO & {trained_fno} \\\\\n"
        "\\addlinespace\n"
        f"Relative increase & {increases} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def render_markdown_table(results: Iterable[HybridSmoothSourceResult]) -> str:
    """Render a concise terminal summary of paired comparison results."""

    lines = [
        "| n | Ground-truth smooth | Trained FNO smooth | Absolute increase | Relative increase |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item.sweep_size):
        lines.append(
            f"| {result.sweep_size} "
            f"| {result.ground_truth_mean_relative_l2:.4f} "
            f"| {result.trained_fno_mean_relative_l2:.4f} "
            f"| {result.absolute_increase:.4f} "
            f"| {result.percent_increase:.1f}% |"
        )
    return "\n".join(lines)
