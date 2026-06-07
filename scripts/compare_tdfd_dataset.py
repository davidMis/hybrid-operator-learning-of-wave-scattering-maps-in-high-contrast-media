#!/usr/bin/env python3
# Overview:
# Sweep Devito TDFD geometry candidates against published Helmholtz pressure
# samples. This is intended for iterative recovery of inherited dataset
# parameters such as Lx and Ly: run candidate solves on a GPU node, copy the CSV
# and optional pressure/plot outputs back, then refine the candidate grid.
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helmholtz_hybrid.tdfd import (
    DEFAULT_BACKGROUND_KM_S,
    DEFAULT_CHUNK_STEPS,
    DEFAULT_DT_MS,
    DEFAULT_END_TIME_MS,
    DEFAULT_EXTRA_ABSORBING_M,
    DEFAULT_FREQUENCY_HZ,
    DEFAULT_MINIMUM_POINTS_PER_WAVELENGTH,
    DEFAULT_N_WAVELENGTHS,
    DEFAULT_SPACE_ORDER,
    DEFAULT_START_TIME_MS,
    DevitoRuntimeSettings,
    TDFDConfigurationError,
    TDFDHelmholtzSettings,
    TDFDHelmholtzSolver,
    load_velocity_sample,
    save_pressure_sample,
)
from helmholtz_hybrid.tdfd_comparison import (
    DEFAULT_TRANSFORMS,
    apply_pressure_transform,
    compare_complex_fields,
    downsample_complex_mean,
    load_complex_pressure_sample,
    parse_transforms,
    upsample_velocity_nearest,
)


LONG_FIELDNAMES = [
    "status",
    "sample_index",
    "domain_size_x_m",
    "domain_size_y_m",
    "requested_source_x_m",
    "requested_source_y_m",
    "actual_source_x_m",
    "actual_source_y_m",
    "solver_upsample_factor",
    "solver_shape_y",
    "solver_shape_x",
    "target_shape_y",
    "target_shape_x",
    "frequency_hz",
    "dt_ms",
    "start_time_ms",
    "end_time_ms",
    "nbl",
    "nt",
    "critical_dt_ms",
    "points_per_min_wavelength",
    "build_seconds",
    "sample_generation_seconds",
    "transform",
    "aligned_relative_l2",
    "raw_relative_l2",
    "correlation_abs",
    "scale_real",
    "scale_imag",
    "scale_abs",
    "scale_phase_deg",
    "target_norm",
    "predicted_norm",
    "max_abs_target",
    "max_abs_predicted",
    "pressure_path",
    "error",
]

SUMMARY_FIELDNAMES = [
    "rank",
    "status",
    "domain_size_x_m",
    "domain_size_y_m",
    "requested_source_x_m",
    "requested_source_y_m",
    "solver_upsample_factor",
    "samples",
    "mean_aligned_relative_l2",
    "median_aligned_relative_l2",
    "max_aligned_relative_l2",
    "mean_raw_relative_l2",
    "mean_correlation_abs",
    "mean_points_per_min_wavelength",
    "mean_sample_generation_seconds",
    "transforms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Lx/Ly and related Devito TDFD settings against published "
            "sharp-to-sharp Helmholtz pressure samples."
        ),
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
        help="Dataset folder under --data-root.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Prepared split used for comparison.",
    )
    parser.add_argument(
        "--velocity",
        type=Path,
        default=None,
        help="Velocity .npy path; defaults to <data-root>/<dataset>/<split>/velocity_sharp.npy.",
    )
    parser.add_argument(
        "--pressure",
        type=Path,
        default=None,
        help="Published pressure .npy path; defaults to <data-root>/<dataset>/<split>/pressure_sharp.npy.",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=[0],
        help="Sample indices to compare. Start with one sample for coarse geometry sweeps.",
    )
    parser.add_argument(
        "--velocity-units",
        choices=("auto", "km/s", "m/s"),
        default="auto",
        help="Units of the velocity input; auto treats large median values as m/s.",
    )
    parser.add_argument(
        "--lx-values",
        type=float,
        nargs="+",
        required=True,
        help="Candidate physical x-domain sizes in meters.",
    )
    parser.add_argument(
        "--ly-values",
        type=float,
        nargs="+",
        default=None,
        help="Candidate physical y-domain sizes in meters. Defaults to --lx-values.",
    )
    parser.add_argument(
        "--source-x-values-m",
        type=float,
        nargs="+",
        default=None,
        help="Optional source x-coordinates to sweep. Omit to use the domain center.",
    )
    parser.add_argument(
        "--source-y-values-m",
        type=float,
        nargs="+",
        default=None,
        help="Optional source y-coordinates to sweep. Omit to use one solver grid cell below the free surface.",
    )
    parser.add_argument(
        "--solver-upsample-factor",
        type=int,
        default=1,
        help=(
            "Integer nearest-neighbor velocity upsample factor for the Devito solve. "
            "Generated pressure is mean-pooled back before comparison."
        ),
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=DEFAULT_FREQUENCY_HZ,
        help="Target Helmholtz frequency accumulated from the time-domain wavefield.",
    )
    parser.add_argument(
        "--start-time-ms",
        type=float,
        default=DEFAULT_START_TIME_MS,
        help="Start time for the Ricker-source simulation.",
    )
    parser.add_argument(
        "--end-time-ms",
        type=float,
        default=DEFAULT_END_TIME_MS,
        help="End time for the Ricker-source simulation.",
    )
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=DEFAULT_DT_MS,
        help="Time step in milliseconds.",
    )
    parser.add_argument(
        "--space-order",
        type=int,
        default=DEFAULT_SPACE_ORDER,
        help="Finite-difference spatial order used by Devito.",
    )
    parser.add_argument(
        "--nbl",
        type=int,
        default=None,
        help="Absorbing boundary thickness in grid cells; omit to derive it from wavelength.",
    )
    parser.add_argument(
        "--absorbing-wavelengths",
        type=float,
        default=DEFAULT_N_WAVELENGTHS,
        help="Number of reference wavelengths used for automatic absorbing-boundary thickness.",
    )
    parser.add_argument(
        "--absorbing-reference-velocity-km-s",
        type=float,
        default=DEFAULT_BACKGROUND_KM_S,
        help="Reference velocity used to convert --absorbing-wavelengths to meters.",
    )
    parser.add_argument(
        "--absorbing-extra-m",
        type=float,
        default=DEFAULT_EXTRA_ABSORBING_M,
        help="Additional meters added to automatic absorbing-boundary thickness.",
    )
    parser.add_argument(
        "--free-surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Devito's pressure free-surface stencil on the top boundary.",
    )
    parser.add_argument(
        "--dft-sign",
        choices=("positive", "negative"),
        default="positive",
        help="Sign convention used in exp(+/- i omega t) Fourier accumulation.",
    )
    parser.add_argument(
        "--normalize-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Divide generated pressure by the source wavelet Fourier coefficient.",
    )
    parser.add_argument(
        "--minimum-points-per-wavelength",
        type=float,
        default=DEFAULT_MINIMUM_POINTS_PER_WAVELENGTH,
        help="Minimum PPW used when --enforce-ppw-guard is set.",
    )
    parser.add_argument(
        "--enforce-ppw-guard",
        action="store_true",
        help="Reject candidates below --minimum-points-per-wavelength. By default, low-PPW candidates are allowed and recorded.",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=DEFAULT_CHUNK_STEPS,
        help="Number of time steps per Devito apply call.",
    )
    parser.add_argument(
        "--backend",
        choices=("env", "cpu", "gpu"),
        default="env",
        help="Devito backend preset. gpu sets nvc/openacc/nvidiaX defaults.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value set before Devito is imported.",
    )
    parser.add_argument(
        "--devito-arch",
        default=None,
        help="Optional DEVITO_ARCH override.",
    )
    parser.add_argument(
        "--devito-language",
        default=None,
        help="Optional DEVITO_LANGUAGE override.",
    )
    parser.add_argument(
        "--devito-platform",
        default=None,
        help="Optional DEVITO_PLATFORM override.",
    )
    parser.add_argument(
        "--gpu-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask Devito to keep solver arrays resident on the GPU.",
    )
    parser.add_argument(
        "--transforms",
        default=",".join(DEFAULT_TRANSFORMS),
        help="Comma-separated pressure transforms tested before selecting the best aligned metric.",
    )
    parser.add_argument(
        "--compare-crop-cells",
        type=int,
        nargs=4,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        default=(0, 0, 0, 0),
        help="Crop cells from target and prediction before computing metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tdfd/dataset_match"),
        help="Directory for optional saved pressures, plots, and metadata.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Long-form per-candidate CSV. Defaults to <output-dir>/tdfd_dataset_comparison.csv.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Aggregate candidate ranking CSV. Defaults to <output-dir>/tdfd_dataset_comparison_summary.csv.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="JSON file describing the sweep settings. Defaults to <output-dir>/tdfd_dataset_comparison_metadata.json.",
    )
    parser.add_argument(
        "--save-pressures",
        action="store_true",
        help="Save each downsampled generated pressure field as [2,H,W] float32 channels.",
    )
    parser.add_argument(
        "--plot-best",
        type=int,
        default=0,
        help="Write diagnostic PNGs for the top N rows. Requires --save-pressures.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing results CSV instead of overwriting it.",
    )
    parser.add_argument(
        "--solver-progress",
        action="store_true",
        help="Show a per-candidate Devito time-step progress bar.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = resolve_paths(args)
        transforms = parse_transforms(args.transforms)
        candidates = build_candidates(args)
        write_metadata(args, paths, transforms, candidates)
        rows = run_sweep(args, paths, transforms, candidates)
        write_summary(paths["summary_csv"], rows)
        if not any(row.get("status") == "ok" for row in rows):
            raise RuntimeError(f"No candidates completed successfully; inspect {paths['results_csv']}.")
        if args.plot_best > 0:
            plot_best_rows(args, paths, rows)
        return 0
    except (OSError, ValueError, RuntimeError, TDFDConfigurationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Return input and output paths with defaults expanded."""

    split_root = args.data_root / args.dataset / args.split
    output_dir = args.output_dir
    paths = {
        "velocity": args.velocity or split_root / "velocity_sharp.npy",
        "pressure": args.pressure or split_root / "pressure_sharp.npy",
        "output_dir": output_dir,
        "results_csv": args.results_csv or output_dir / "tdfd_dataset_comparison.csv",
        "summary_csv": args.summary_csv or output_dir / "tdfd_dataset_comparison_summary.csv",
        "metadata_output": args.metadata_output or output_dir / "tdfd_dataset_comparison_metadata.json",
        "pressure_dir": output_dir / "pressures",
        "plot_dir": output_dir / "plots",
    }
    return paths


def build_candidates(args: argparse.Namespace) -> list[dict[str, float | None]]:
    """Return all geometry/source candidates requested by the CLI."""

    ly_values = args.ly_values if args.ly_values is not None else args.lx_values
    source_x_values = args.source_x_values_m if args.source_x_values_m is not None else [None]
    source_y_values = args.source_y_values_m if args.source_y_values_m is not None else [None]
    candidates = []
    for lx in args.lx_values:
        for ly in ly_values:
            for source_x in source_x_values:
                for source_y in source_y_values:
                    candidates.append(
                        {
                            "domain_size_x_m": float(lx),
                            "domain_size_y_m": float(ly),
                            "source_x_m": None if source_x is None else float(source_x),
                            "source_y_m": None if source_y is None else float(source_y),
                        }
                    )
    if not candidates:
        raise ValueError("No candidates were requested.")
    return candidates


def write_metadata(
    args: argparse.Namespace,
    paths: dict[str, Path],
    transforms: tuple[str, ...],
    candidates: list[dict[str, float | None]],
) -> None:
    """Write JSON metadata before starting the expensive sweep."""

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "velocity": str(paths["velocity"]),
        "pressure": str(paths["pressure"]),
        "sample_indices": args.sample_indices,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "solver_upsample_factor": args.solver_upsample_factor,
        "frequency_hz": args.frequency_hz,
        "start_time_ms": args.start_time_ms,
        "end_time_ms": args.end_time_ms,
        "dt_ms": args.dt_ms,
        "space_order": args.space_order,
        "transforms": transforms,
        "compare_crop_cells": list(args.compare_crop_cells),
    }
    paths["metadata_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["metadata_output"].write_text(json.dumps(payload, indent=2) + "\n")


def run_sweep(
    args: argparse.Namespace,
    paths: dict[str, Path],
    transforms: tuple[str, ...],
    candidates: list[dict[str, float | None]],
) -> list[dict[str, Any]]:
    """Run all requested candidates and append per-candidate rows to CSV."""

    if args.solver_upsample_factor < 1:
        raise ValueError(f"--solver-upsample-factor must be positive; got {args.solver_upsample_factor}.")
    mode = "a" if args.append and paths["results_csv"].is_file() else "w"
    paths["results_csv"].parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total = len(args.sample_indices) * len(candidates)

    with paths["results_csv"].open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_FIELDNAMES)
        if mode == "w":
            writer.writeheader()

        with tqdm(
            total=total,
            desc="TDFD candidates",
            unit="candidate",
            disable=args.no_progress,
            dynamic_ncols=True,
        ) as overall:
            for sample_index in args.sample_indices:
                velocity = load_velocity_sample(
                    paths["velocity"],
                    sample_index=sample_index,
                    units=args.velocity_units,
                )
                target = load_complex_pressure_sample(paths["pressure"], sample_index=sample_index)
                for candidate in candidates:
                    row = run_one_candidate(
                        args,
                        paths,
                        transforms,
                        velocity,
                        target,
                        sample_index,
                        candidate,
                    )
                    writer.writerow(format_long_row(row))
                    handle.flush()
                    rows.append(row)
                    overall.update(1)
                    if row["status"] == "ok":
                        overall.set_postfix_str(
                            f"best={float(row['aligned_relative_l2']):.4f} "
                            f"L=({row['domain_size_x_m']:.0f},{row['domain_size_y_m']:.0f})"
                        )
                    else:
                        overall.set_postfix_str("failed")
    return rows


def run_one_candidate(
    args: argparse.Namespace,
    paths: dict[str, Path],
    transforms: tuple[str, ...],
    velocity: np.ndarray,
    target: np.ndarray,
    sample_index: int,
    candidate: dict[str, float | None],
) -> dict[str, Any]:
    """Run and compare one geometry/source candidate."""

    base_row: dict[str, Any] = {
        "status": "failed",
        "sample_index": sample_index,
        "domain_size_x_m": float(candidate["domain_size_x_m"]),
        "domain_size_y_m": float(candidate["domain_size_y_m"]),
        "requested_source_x_m": candidate["source_x_m"],
        "requested_source_y_m": candidate["source_y_m"],
        "solver_upsample_factor": args.solver_upsample_factor,
        "target_shape_y": target.shape[0],
        "target_shape_x": target.shape[1],
        "frequency_hz": args.frequency_hz,
        "dt_ms": args.dt_ms,
        "start_time_ms": args.start_time_ms,
        "end_time_ms": args.end_time_ms,
        "pressure_path": "",
        "error": "",
    }
    try:
        solver_velocity = upsample_velocity_nearest(velocity, args.solver_upsample_factor)
        settings = make_settings(args, candidate)
        build_start = time.perf_counter()
        solver = TDFDHelmholtzSolver(solver_velocity, settings)
        build_seconds = time.perf_counter() - build_start

        if args.solver_progress and not args.no_progress:
            with tqdm(desc="time steps", unit="step", leave=False, dynamic_ncols=True) as solver_bar:

                def progress(increment: int, total_steps: int) -> None:
                    if solver_bar.total != total_steps:
                        solver_bar.reset(total=total_steps)
                    solver_bar.update(increment)

                sample = solver.solve(progress_callback=progress)
        else:
            sample = solver.solve()

        predicted = downsample_complex_mean(sample.pressure, args.solver_upsample_factor)
        comparison = compare_complex_fields(
            predicted,
            target,
            transforms=transforms,
            crop=tuple(args.compare_crop_cells),
        )
        pressure_path = ""
        if args.save_pressures:
            pressure_path = str(save_candidate_pressure(paths, predicted, sample_index, candidate, args))

        diagnostics = sample.diagnostics
        return {
            **base_row,
            "status": "ok",
            "actual_source_x_m": diagnostics.source_coordinates_m[0],
            "actual_source_y_m": diagnostics.source_coordinates_m[1],
            "solver_shape_y": solver_velocity.shape[0],
            "solver_shape_x": solver_velocity.shape[1],
            "nbl": diagnostics.nbl,
            "nt": diagnostics.nt,
            "critical_dt_ms": diagnostics.critical_dt_ms,
            "points_per_min_wavelength": diagnostics.points_per_min_wavelength,
            "build_seconds": build_seconds,
            "sample_generation_seconds": diagnostics.devito_apply_seconds,
            **comparison.to_row(),
            "pressure_path": pressure_path,
        }
    except (OSError, ValueError, RuntimeError, TDFDConfigurationError) as error:
        return {
            **base_row,
            "error": f"{type(error).__name__}: {error}",
        }


def make_settings(args: argparse.Namespace, candidate: dict[str, float | None]) -> TDFDHelmholtzSettings:
    """Build solver settings for one candidate."""

    runtime = DevitoRuntimeSettings(
        backend=args.backend,
        devito_arch=args.devito_arch,
        devito_language=args.devito_language,
        devito_platform=args.devito_platform,
        cuda_visible_devices=args.cuda_visible_devices,
        gpu_fit=args.gpu_fit,
    )
    return TDFDHelmholtzSettings(
        frequency_hz=args.frequency_hz,
        domain_size_x_m=float(candidate["domain_size_x_m"]),
        domain_size_y_m=float(candidate["domain_size_y_m"]),
        source_x_m=candidate["source_x_m"],
        source_y_m=candidate["source_y_m"],
        start_time_ms=args.start_time_ms,
        end_time_ms=args.end_time_ms,
        dt_ms=args.dt_ms,
        space_order=args.space_order,
        nbl=args.nbl,
        absorbing_reference_velocity_km_s=args.absorbing_reference_velocity_km_s,
        absorbing_wavelengths=args.absorbing_wavelengths,
        absorbing_extra_m=args.absorbing_extra_m,
        free_surface=args.free_surface,
        dft_sign=args.dft_sign,
        normalize_by_source_spectrum=args.normalize_source,
        minimum_points_per_wavelength=(
            args.minimum_points_per_wavelength if args.enforce_ppw_guard else 0.0
        ),
        chunk_steps=args.chunk_steps,
        runtime=runtime,
    )


def save_candidate_pressure(
    paths: dict[str, Path],
    predicted: np.ndarray,
    sample_index: int,
    candidate: dict[str, float | None],
    args: argparse.Namespace,
) -> Path:
    """Save one comparable generated pressure field and return its path."""

    paths["pressure_dir"].mkdir(parents=True, exist_ok=True)
    tag = candidate_tag(sample_index, candidate, args.solver_upsample_factor)
    path = paths["pressure_dir"] / f"{tag}.npy"
    save_pressure_sample(predicted, path, channels=True)
    return path


def candidate_tag(sample_index: int, candidate: dict[str, float | None], upsample_factor: int) -> str:
    """Return a filesystem-friendly candidate identifier."""

    source_x = candidate.get("source_x_m", candidate.get("requested_source_x_m"))
    source_y = candidate.get("source_y_m", candidate.get("requested_source_y_m"))
    sx = "center" if source_x in (None, "") else f"{float(source_x):.3f}"
    sy = "default" if source_y in (None, "") else f"{float(source_y):.3f}"
    return (
        f"sample{sample_index:06d}"
        f"_Lx{float(candidate['domain_size_x_m']):.3f}"
        f"_Ly{float(candidate['domain_size_y_m']):.3f}"
        f"_sx{sx}_sy{sy}_up{upsample_factor}"
    ).replace(".", "p")


def format_long_row(row: dict[str, Any]) -> dict[str, str]:
    """Format one long CSV row, filling absent failure fields with blanks."""

    formatted: dict[str, str] = {}
    for field in LONG_FIELDNAMES:
        value = row.get(field, "")
        if value is None:
            formatted[field] = ""
        elif isinstance(value, float):
            formatted[field] = f"{value:.9g}" if math.isfinite(value) else str(value)
        else:
            formatted[field] = str(value)
    return formatted


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write aggregate candidate rankings sorted by mean aligned error."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row["domain_size_x_m"],
            row["domain_size_y_m"],
            row["requested_source_x_m"],
            row["requested_source_y_m"],
            row["solver_upsample_factor"],
        )
        groups[key].append(row)

    summaries = []
    for key, group_rows in groups.items():
        aligned = np.asarray([float(row["aligned_relative_l2"]) for row in group_rows], dtype=np.float64)
        raw = np.asarray([float(row["raw_relative_l2"]) for row in group_rows], dtype=np.float64)
        corr = np.asarray([float(row["correlation_abs"]) for row in group_rows], dtype=np.float64)
        ppw = np.asarray([float(row["points_per_min_wavelength"]) for row in group_rows], dtype=np.float64)
        seconds = np.asarray([float(row["sample_generation_seconds"]) for row in group_rows], dtype=np.float64)
        transforms = sorted({str(row["transform"]) for row in group_rows})
        summaries.append(
            {
                "status": "ok",
                "domain_size_x_m": key[0],
                "domain_size_y_m": key[1],
                "requested_source_x_m": key[2],
                "requested_source_y_m": key[3],
                "solver_upsample_factor": key[4],
                "samples": len(group_rows),
                "mean_aligned_relative_l2": float(np.mean(aligned)),
                "median_aligned_relative_l2": float(np.median(aligned)),
                "max_aligned_relative_l2": float(np.max(aligned)),
                "mean_raw_relative_l2": float(np.mean(raw)),
                "mean_correlation_abs": float(np.mean(corr)),
                "mean_points_per_min_wavelength": float(np.mean(ppw)),
                "mean_sample_generation_seconds": float(np.mean(seconds)),
                "transforms": ";".join(transforms),
            }
        )
    summaries.sort(key=lambda item: item["mean_aligned_relative_l2"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for rank, summary in enumerate(summaries, start=1):
            writer.writerow(format_summary_row(rank, summary))


def format_summary_row(rank: int, summary: dict[str, Any]) -> dict[str, str]:
    """Format one aggregate summary CSV row."""

    row = {"rank": str(rank)}
    for field in SUMMARY_FIELDNAMES:
        if field == "rank":
            continue
        value = summary.get(field, "")
        if value is None:
            row[field] = ""
        elif isinstance(value, float):
            row[field] = f"{value:.9g}" if math.isfinite(value) else str(value)
        else:
            row[field] = str(value)
    return row


def plot_best_rows(args: argparse.Namespace, paths: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    """Plot the best saved pressure comparisons for quick visual inspection."""

    if not args.save_pressures:
        print("Skipping --plot-best because --save-pressures was not set.", file=sys.stderr)
        return

    from helmholtz_hybrid.runtime import set_default_cache_dirs

    set_default_cache_dirs()
    import matplotlib.pyplot as plt

    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("pressure_path")]
    ok_rows.sort(key=lambda item: float(item["aligned_relative_l2"]))
    for rank, row in enumerate(ok_rows[: args.plot_best], start=1):
        sample_index = int(row["sample_index"])
        velocity = load_velocity_sample(paths["velocity"], sample_index=sample_index, units=args.velocity_units)
        target = load_complex_pressure_sample(paths["pressure"], sample_index=sample_index)
        predicted = load_complex_pressure_sample(row["pressure_path"], sample_index=None)
        transformed = apply_pressure_transform(predicted, str(row["transform"]))
        scale = complex(float(row["scale_real"]), float(row["scale_imag"]))
        aligned = scale * transformed
        error = np.abs(aligned - target)

        pressure_abs = float(np.max(np.abs([target.real, aligned.real, target.imag, aligned.imag])))
        error_abs = float(np.max(error))

        fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.0), constrained_layout=True)
        axes[0, 0].imshow(velocity, cmap="bwr")
        axes[0, 0].set_title("Velocity")
        axes[0, 1].imshow(target.real, cmap="seismic", vmin=-pressure_abs, vmax=pressure_abs)
        axes[0, 1].set_title("Target real")
        axes[0, 2].imshow(aligned.real, cmap="seismic", vmin=-pressure_abs, vmax=pressure_abs)
        axes[0, 2].set_title("TDFD real")
        axes[1, 0].imshow(error, cmap="inferno", vmin=0.0, vmax=error_abs)
        axes[1, 0].set_title("|error|")
        axes[1, 1].imshow(target.imag, cmap="seismic", vmin=-pressure_abs, vmax=pressure_abs)
        axes[1, 1].set_title("Target imag")
        axes[1, 2].imshow(aligned.imag, cmap="seismic", vmin=-pressure_abs, vmax=pressure_abs)
        axes[1, 2].set_title("TDFD imag")

        for ax in axes.ravel():
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

        fig.suptitle(
            f"rank {rank}: L=({float(row['domain_size_x_m']):.0f},"
            f"{float(row['domain_size_y_m']):.0f}) rel={float(row['aligned_relative_l2']):.4f} "
            f"transform={row['transform']}"
        )
        paths["plot_dir"].mkdir(parents=True, exist_ok=True)
        output = paths["plot_dir"] / f"rank{rank:02d}_{candidate_tag(sample_index, row, int(row['solver_upsample_factor']))}.png"
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
