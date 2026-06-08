#!/usr/bin/env python3
# Overview:
# Compare the paper-scaled FEM Helmholtz solver against published
# pressure samples. This script loads existing velocity/pressure arrays, runs one
# FE solve per requested sample, aligns the generated field by the best complex
# scalar, and writes metrics for reproducibility checks.
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helmholtz_hybrid.fem_helmholtz import (
    DEFAULT_FEM_ABC_SCALE,
    DEFAULT_FEM_ABC_VELOCITY_M_S,
    DEFAULT_FEM_BICGSTAB_MAXITER,
    DEFAULT_FEM_BICGSTAB_RTOL,
    DEFAULT_FEM_DOMAIN_SIZE_M,
    DEFAULT_FEM_FREQUENCY_HZ,
    DEFAULT_FEM_SOURCE_AMPLITUDE,
    DEFAULT_FEM_SOURCE_ROWS_BELOW_TOP,
    DEFAULT_FEM_SOURCE_SPREAD_GRID_CELLS,
    DEFAULT_FEM_SPILU_DROP_TOL,
    DEFAULT_FEM_SPILU_FILL_FACTOR,
    DEFAULT_FEM_VELOCITY_SAMPLING,
    FEMConfigurationError,
    FEMHelmholtzSettings,
    FEMHelmholtzSolver,
)
from helmholtz_hybrid.field_comparison import (
    DEFAULT_TRANSFORMS,
    apply_pressure_transform,
    compare_complex_fields,
    complex_to_pressure_channels,
    load_complex_pressure_sample,
    parse_transforms,
)
from helmholtz_hybrid.sample_io import load_velocity_sample


FIELDNAMES = [
    "status",
    "sample_index",
    "domain_size_x_m",
    "domain_size_y_m",
    "frequency_hz",
    "source_x_m",
    "source_y_m",
    "source_rows_below_top",
    "source_spread_grid_cells",
    "source_sigma_m",
    "source_amplitude",
    "abc_velocity_m_s",
    "abc_scale",
    "velocity_sampling",
    "legacy_mask_shape",
    "linear_solver",
    "solver_shape_y",
    "solver_shape_x",
    "target_shape_y",
    "target_shape_x",
    "spacing_x_m",
    "spacing_y_m",
    "matrix_nnz",
    "solve_method",
    "bicgstab_info",
    "assemble_seconds",
    "solve_seconds",
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


def parse_args() -> argparse.Namespace:
    """Parse CLI options for one FEM comparison run."""

    parser = argparse.ArgumentParser(
        description="Compare the paper-scaled FEM Helmholtz solver against published pressure samples.",
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
        help="Sample indices to compare.",
    )
    parser.add_argument(
        "--velocity-units",
        choices=("auto", "km/s", "m/s"),
        default="auto",
        help="Units of the velocity input; auto treats large median values as m/s.",
    )
    parser.add_argument(
        "--domain-size-x-m",
        type=float,
        default=DEFAULT_FEM_DOMAIN_SIZE_M,
        help="Physical x-domain size used by the FEM solver.",
    )
    parser.add_argument(
        "--domain-size-y-m",
        type=float,
        default=DEFAULT_FEM_DOMAIN_SIZE_M,
        help="Physical y-domain size used by the FEM solver.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=DEFAULT_FEM_FREQUENCY_HZ,
        help="Frequency in the frequency-domain Helmholtz equation.",
    )
    parser.add_argument(
        "--source-x-m",
        type=float,
        default=None,
        help="Absolute source x-coordinate in meters; omit to use the domain center.",
    )
    parser.add_argument(
        "--source-rows-below-top",
        type=float,
        default=DEFAULT_FEM_SOURCE_ROWS_BELOW_TOP,
        help="Source y-coordinate expressed as solver-grid rows below the top boundary.",
    )
    parser.add_argument(
        "--source-spread-grid-cells",
        type=float,
        default=DEFAULT_FEM_SOURCE_SPREAD_GRID_CELLS,
        help="Gaussian source standard deviation in solver-grid spacings.",
    )
    parser.add_argument(
        "--source-amplitude",
        type=float,
        default=DEFAULT_FEM_SOURCE_AMPLITUDE,
        help="Gaussian source amplitude. Comparisons align by a complex scalar, so this mostly affects raw amplitude.",
    )
    parser.add_argument(
        "--abc-velocity-m-s",
        type=float,
        default=DEFAULT_FEM_ABC_VELOCITY_M_S,
        help="Reference velocity used in the Robin absorbing boundary condition.",
    )
    parser.add_argument(
        "--abc-scale",
        type=float,
        default=DEFAULT_FEM_ABC_SCALE,
        help="Multiplier applied to omega / --abc-velocity-m-s in the Robin boundary condition.",
    )
    parser.add_argument(
        "--velocity-sampling",
        choices=("nearest", "bilinear", "legacy-mask128"),
        default=DEFAULT_FEM_VELOCITY_SAMPLING,
        help="How element-centroid velocities are sampled from the stored velocity grid.",
    )
    parser.add_argument(
        "--legacy-mask-shape",
        type=int,
        nargs=2,
        metavar=("NY", "NX"),
        default=(128, 128),
        help="Low-resolution salt mask shape used when --velocity-sampling legacy-mask128 is selected.",
    )
    parser.add_argument(
        "--linear-solver",
        choices=("bicgstab", "direct"),
        default="bicgstab",
        help="Sparse linear solver. bicgstab uses SPILU preconditioning and falls back to spsolve.",
    )
    parser.add_argument(
        "--spilu-drop-tol",
        type=float,
        default=DEFAULT_FEM_SPILU_DROP_TOL,
        help="Drop tolerance for scipy.sparse.linalg.spilu.",
    )
    parser.add_argument(
        "--spilu-fill-factor",
        type=float,
        default=DEFAULT_FEM_SPILU_FILL_FACTOR,
        help="Fill factor for scipy.sparse.linalg.spilu.",
    )
    parser.add_argument(
        "--bicgstab-rtol",
        type=float,
        default=DEFAULT_FEM_BICGSTAB_RTOL,
        help="Relative tolerance passed to scipy.sparse.linalg.bicgstab.",
    )
    parser.add_argument(
        "--bicgstab-maxiter",
        type=int,
        default=DEFAULT_FEM_BICGSTAB_MAXITER,
        help="Maximum BiCGSTAB iterations before falling back to a direct sparse solve.",
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
        default=Path("outputs/fem/dataset_match"),
        help="Directory for metrics, optional saved pressures, plots, and metadata.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Per-sample CSV path. Defaults to <output-dir>/fem_dataset_comparison.csv.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="JSON metadata path. Defaults to <output-dir>/fem_dataset_comparison_metadata.json.",
    )
    parser.add_argument(
        "--save-pressures",
        action="store_true",
        help="Save each generated pressure field as [2,H,W] float32 channels.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write diagnostic PNGs for generated versus target fields. Requires matplotlib.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing results CSV instead of overwriting it.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested FEM comparison and report actionable errors."""

    args = parse_args()
    try:
        paths = resolve_paths(args)
        transforms = parse_transforms(args.transforms)
        settings = make_settings(args)
        write_metadata(args, paths, transforms, settings)
        rows = run_comparisons(args, paths, transforms, settings)
        if not any(row.get("status") == "ok" for row in rows):
            raise RuntimeError(f"No samples completed successfully; inspect {paths['results_csv']}.")
        return 0
    except (OSError, ValueError, RuntimeError, FEMConfigurationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Return input and output paths with defaults expanded."""

    split_root = args.data_root / args.dataset / args.split
    output_dir = args.output_dir
    return {
        "velocity": args.velocity or split_root / "velocity_sharp.npy",
        "pressure": args.pressure or split_root / "pressure_sharp.npy",
        "output_dir": output_dir,
        "results_csv": args.results_csv or output_dir / "fem_dataset_comparison.csv",
        "metadata_output": args.metadata_output or output_dir / "fem_dataset_comparison_metadata.json",
        "pressure_dir": output_dir / "pressures",
        "plot_dir": output_dir / "plots",
    }


def make_settings(args: argparse.Namespace) -> FEMHelmholtzSettings:
    """Build solver settings from CLI arguments."""

    return FEMHelmholtzSettings(
        frequency_hz=args.frequency_hz,
        domain_size_x_m=args.domain_size_x_m,
        domain_size_y_m=args.domain_size_y_m,
        source_x_m=args.source_x_m,
        source_rows_below_top=args.source_rows_below_top,
        source_spread_grid_cells=args.source_spread_grid_cells,
        source_amplitude=args.source_amplitude,
        abc_velocity_m_s=args.abc_velocity_m_s,
        abc_scale=args.abc_scale,
        velocity_sampling=args.velocity_sampling,
        legacy_mask_shape=tuple(args.legacy_mask_shape),
        linear_solver=args.linear_solver,
        spilu_drop_tol=args.spilu_drop_tol,
        spilu_fill_factor=args.spilu_fill_factor,
        bicgstab_rtol=args.bicgstab_rtol,
        bicgstab_maxiter=args.bicgstab_maxiter,
    )


def write_metadata(
    args: argparse.Namespace,
    paths: dict[str, Path],
    transforms: tuple[str, ...],
    settings: FEMHelmholtzSettings,
) -> None:
    """Write JSON metadata describing the comparison before solving."""

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "velocity": str(paths["velocity"]),
        "pressure": str(paths["pressure"]),
        "sample_indices": args.sample_indices,
        "settings": {
            "frequency_hz": settings.frequency_hz,
            "domain_size_x_m": settings.domain_size_x_m,
            "domain_size_y_m": settings.domain_size_y_m,
            "source_x_m": settings.source_x_m,
            "source_rows_below_top": settings.source_rows_below_top,
            "source_spread_grid_cells": settings.source_spread_grid_cells,
            "source_amplitude": settings.source_amplitude,
            "abc_velocity_m_s": settings.abc_velocity_m_s,
            "abc_scale": settings.abc_scale,
            "velocity_sampling": settings.velocity_sampling,
            "legacy_mask_shape": list(settings.legacy_mask_shape),
            "linear_solver": settings.linear_solver,
        },
        "transforms": transforms,
        "compare_crop_cells": list(args.compare_crop_cells),
    }
    paths["metadata_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["metadata_output"].write_text(json.dumps(payload, indent=2) + "\n")


def run_comparisons(
    args: argparse.Namespace,
    paths: dict[str, Path],
    transforms: tuple[str, ...],
    settings: FEMHelmholtzSettings,
) -> list[dict[str, Any]]:
    """Run one FEM solve and comparison per requested sample."""

    mode = "a" if args.append and paths["results_csv"].is_file() else "w"
    paths["results_csv"].parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    solvers: dict[tuple[int, int], FEMHelmholtzSolver] = {}

    with paths["results_csv"].open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()

        with tqdm(
            total=len(args.sample_indices),
            desc="FEM samples",
            unit="sample",
            disable=args.no_progress,
            dynamic_ncols=True,
        ) as progress:
            for sample_index in args.sample_indices:
                row = run_one_sample(args, paths, transforms, settings, solvers, sample_index)
                writer.writerow(format_row(row))
                handle.flush()
                rows.append(row)
                progress.update(1)
                if row["status"] == "ok":
                    progress.set_postfix_str(
                        f"rel={float(row['aligned_relative_l2']):.4f} "
                        f"corr={float(row['correlation_abs']):.4f}"
                    )
                else:
                    progress.set_postfix_str("failed")
    return rows


def run_one_sample(
    args: argparse.Namespace,
    paths: dict[str, Path],
    transforms: tuple[str, ...],
    settings: FEMHelmholtzSettings,
    solvers: dict[tuple[int, int], FEMHelmholtzSolver],
    sample_index: int,
) -> dict[str, Any]:
    """Run and compare one sample, converting exceptions into a CSV row."""

    base_row: dict[str, Any] = {
        "status": "failed",
        "sample_index": sample_index,
        "domain_size_x_m": settings.domain_size_x_m,
        "domain_size_y_m": settings.domain_size_y_m,
        "frequency_hz": settings.frequency_hz,
        "source_x_m": settings.source_x_m if settings.source_x_m is not None else "",
        "source_rows_below_top": settings.source_rows_below_top,
        "source_spread_grid_cells": settings.source_spread_grid_cells,
        "source_amplitude": settings.source_amplitude,
        "abc_velocity_m_s": settings.abc_velocity_m_s,
        "abc_scale": settings.abc_scale,
        "velocity_sampling": settings.velocity_sampling,
        "legacy_mask_shape": "x".join(str(value) for value in settings.legacy_mask_shape),
        "linear_solver": settings.linear_solver,
        "pressure_path": "",
        "error": "",
    }
    try:
        velocity_km_s = load_velocity_sample(paths["velocity"], sample_index=sample_index, units=args.velocity_units)
        velocity_m_s = velocity_km_s * 1000.0
        target = load_complex_pressure_sample(paths["pressure"], sample_index=sample_index)
        if velocity_m_s.shape != target.shape:
            raise ValueError(
                f"Velocity shape {velocity_m_s.shape} and target pressure shape {target.shape} must match."
            )
        solver = solvers.get(velocity_m_s.shape)
        if solver is None:
            solver = FEMHelmholtzSolver(velocity_m_s.shape, settings)
            solvers[velocity_m_s.shape] = solver
        sample = solver.solve(velocity_m_s)
        comparison = compare_complex_fields(
            sample.pressure,
            target,
            transforms=transforms,
            crop=tuple(args.compare_crop_cells),
        )
        pressure_path = ""
        if args.save_pressures:
            pressure_path = str(save_pressure(paths, sample.pressure, sample_index))
        if args.plot:
            plot_sample(args, paths, velocity_km_s, target, sample.pressure, comparison, sample_index)

        diagnostics = sample.diagnostics
        return {
            **base_row,
            "status": "ok",
            "source_x_m": diagnostics.source_coordinates_m[0],
            "source_y_m": diagnostics.source_coordinates_m[1],
            "source_sigma_m": diagnostics.source_sigma_m,
            "solver_shape_y": diagnostics.velocity_shape[0],
            "solver_shape_x": diagnostics.velocity_shape[1],
            "target_shape_y": target.shape[0],
            "target_shape_x": target.shape[1],
            "spacing_x_m": diagnostics.spacing_m[0],
            "spacing_y_m": diagnostics.spacing_m[1],
            "matrix_nnz": diagnostics.matrix_nnz,
            "solve_method": diagnostics.solve_method,
            "bicgstab_info": "" if diagnostics.bicgstab_info is None else diagnostics.bicgstab_info,
            "assemble_seconds": diagnostics.assemble_seconds,
            "solve_seconds": diagnostics.solve_seconds,
            "sample_generation_seconds": diagnostics.sample_generation_seconds,
            **comparison.to_row(),
            "pressure_path": pressure_path,
        }
    except (OSError, ValueError, RuntimeError, FEMConfigurationError) as error:
        return {
            **base_row,
            "error": f"{type(error).__name__}: {error}",
        }


def save_pressure(paths: dict[str, Path], pressure: np.ndarray, sample_index: int) -> Path:
    """Save one generated pressure sample and return the output path."""

    paths["pressure_dir"].mkdir(parents=True, exist_ok=True)
    path = paths["pressure_dir"] / f"sample{sample_index:06d}_fem_pressure.npy"
    np.save(path, complex_to_pressure_channels(pressure))
    return path


def plot_sample(
    args: argparse.Namespace,
    paths: dict[str, Path],
    velocity_km_s: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    comparison,
    sample_index: int,
) -> None:
    """Write a diagnostic comparison figure for one sample."""

    from helmholtz_hybrid.runtime import set_default_cache_dirs

    set_default_cache_dirs()
    import matplotlib.pyplot as plt

    transformed = apply_pressure_transform(predicted, comparison.transform)
    scale = complex(comparison.scale_real, comparison.scale_imag)
    aligned = scale * transformed
    error = np.abs(aligned - target)
    pressure_abs = float(np.max(np.abs([target.real, aligned.real, target.imag, aligned.imag])))
    error_abs = float(np.max(error))

    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.0), constrained_layout=True)
    axes[0, 0].imshow(velocity_km_s, cmap="bwr")
    axes[0, 0].set_title("Velocity")
    axes[0, 1].imshow(target.real, cmap="seismic", vmin=-pressure_abs, vmax=pressure_abs)
    axes[0, 1].set_title("Target real")
    axes[0, 2].imshow(aligned.real, cmap="seismic", vmin=-pressure_abs, vmax=pressure_abs)
    axes[0, 2].set_title("FEM real")
    axes[1, 0].imshow(error, cmap="inferno", vmin=0.0, vmax=error_abs)
    axes[1, 0].set_title("|error|")
    axes[1, 1].imshow(target.imag, cmap="seismic", vmin=-pressure_abs, vmax=pressure_abs)
    axes[1, 1].set_title("Target imag")
    axes[1, 2].imshow(aligned.imag, cmap="seismic", vmin=-pressure_abs, vmax=pressure_abs)
    axes[1, 2].set_title("FEM imag")

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
        f"sample {sample_index}: rel={comparison.aligned_relative_l2:.4f} "
        f"corr={comparison.correlation_abs:.4f} transform={comparison.transform}"
    )
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    output = paths["plot_dir"] / f"sample{sample_index:06d}_fem_comparison.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_row(row: dict[str, Any]) -> dict[str, str]:
    """Format one CSV row, filling absent failure fields with blanks."""

    formatted: dict[str, str] = {}
    for field in FIELDNAMES:
        value = row.get(field, "")
        if value is None:
            formatted[field] = ""
        elif isinstance(value, float):
            formatted[field] = f"{value:.9g}" if math.isfinite(value) else str(value)
        else:
            formatted[field] = str(value)
    return formatted


if __name__ == "__main__":
    raise SystemExit(main())
