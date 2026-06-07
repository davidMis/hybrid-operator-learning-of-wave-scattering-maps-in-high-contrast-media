#!/usr/bin/env python3
# Overview:
# Time the recovered paper-scaled FEM Helmholtz solver on one or more velocity
# samples. The script loads all requested velocity models and constructs the
# reusable solver before timing, then measures only per-sample finite-element
# matrix assembly and sparse linear solve work.
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
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
from helmholtz_hybrid.tdfd import VelocityUnits, normalize_velocity_units


FIELDNAMES = [
    "sample_index",
    "outer_wall_seconds",
    "sample_generation_seconds",
    "assemble_seconds",
    "linear_solve_seconds",
    "solve_method",
    "bicgstab_info",
    "matrix_nnz",
    "max_abs_pressure",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the FEM timing run."""

    parser = argparse.ArgumentParser(
        description=(
            "Time paper-scaled FEM Helmholtz sample generation. Inputs are loaded "
            "and the solver mesh is constructed before timing."
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
        help="Prepared split used for timing.",
    )
    parser.add_argument(
        "--velocity",
        type=Path,
        default=None,
        help="Velocity .npy path; defaults to <data-root>/<dataset>/<split>/velocity_sharp.npy.",
    )
    parser.add_argument(
        "--sample-start-index",
        type=int,
        default=0,
        help="First sample index used when --sample-indices is omitted.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of consecutive samples to time when --sample-indices is omitted.",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit sample indices to time; overrides --sample-start-index and --sample-count.",
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
        help="Gaussian source amplitude.",
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
        "--warmup-runs",
        type=int,
        default=0,
        help="Untimed solves run after data loading and solver construction.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the timing summary JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for per-sample timing rows.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def main() -> int:
    """Load inputs, run the timed FEM solves, and write timing summaries."""

    args = parse_args()
    try:
        sample_indices = resolve_sample_indices(args)
        velocity_path = resolve_velocity_path(args)
        velocities_m_s = load_velocity_samples(
            velocity_path,
            sample_indices=sample_indices,
            units=args.velocity_units,
        )
        settings = make_settings(args)
        solver = FEMHelmholtzSolver(velocities_m_s[0].shape, settings)
        validate_shapes(velocities_m_s)

        run_warmups(args, solver, velocities_m_s)
        rows = run_timed_samples(args, solver, sample_indices, velocities_m_s)
        summary = build_summary(args, velocity_path, sample_indices, rows, solver)

        if args.output_csv is not None:
            write_csv(args.output_csv, rows)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summary, indent=2) + "\n")

        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, FEMConfigurationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def resolve_sample_indices(args: argparse.Namespace) -> list[int]:
    """Return the sample indices requested by the CLI."""

    if args.sample_indices is not None:
        indices = list(args.sample_indices)
    else:
        if args.sample_count <= 0:
            raise ValueError(f"--sample-count must be positive; got {args.sample_count}.")
        indices = list(range(args.sample_start_index, args.sample_start_index + args.sample_count))
    if not indices:
        raise ValueError("At least one sample index is required.")
    if min(indices) < 0:
        raise ValueError(f"Sample indices must be non-negative; got {indices}.")
    return indices


def resolve_velocity_path(args: argparse.Namespace) -> Path:
    """Return the velocity path with dataset defaults expanded."""

    if args.velocity is not None:
        return args.velocity
    return args.data_root / args.dataset / args.split / "velocity_sharp.npy"


def load_velocity_samples(
    path: Path,
    *,
    sample_indices: list[int],
    units: VelocityUnits,
) -> list[np.ndarray]:
    """Load all requested velocity samples into memory before timing."""

    array = np.load(path, mmap_mode="r")
    velocities: list[np.ndarray] = []
    if array.ndim == 2:
        if any(index != 0 for index in sample_indices):
            raise ValueError(f"{path} contains one sample; only sample index 0 is valid.")
        sample = normalize_velocity_units(np.asarray(array, dtype=np.float32), units=units) * 1000.0
        return [np.ascontiguousarray(sample, dtype=np.float32)]
    if array.ndim != 3:
        raise ValueError(f"Expected velocity shape [ny,nx] or [N,ny,nx], got {array.shape} from {path}.")
    sample_count = int(array.shape[0])
    for index in sample_indices:
        if index >= sample_count:
            raise ValueError(f"Sample index {index} is out of bounds for {path}, which has {sample_count} samples.")
        sample = normalize_velocity_units(np.asarray(array[index], dtype=np.float32), units=units) * 1000.0
        velocities.append(np.ascontiguousarray(sample, dtype=np.float32))
    return velocities


def make_settings(args: argparse.Namespace) -> FEMHelmholtzSettings:
    """Build paper-scaled FEM settings from CLI arguments."""

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


def validate_shapes(velocities_m_s: list[np.ndarray]) -> None:
    """Ensure every timed sample has the same shape as the solver."""

    expected = velocities_m_s[0].shape
    for offset, velocity in enumerate(velocities_m_s):
        if velocity.shape != expected:
            raise ValueError(f"Velocity sample {offset} has shape {velocity.shape}; expected {expected}.")


def run_warmups(args: argparse.Namespace, solver: FEMHelmholtzSolver, velocities_m_s: list[np.ndarray]) -> None:
    """Run untimed warmup solves after setup and before timing."""

    if args.warmup_runs < 0:
        raise ValueError(f"--warmup-runs must be non-negative; got {args.warmup_runs}.")
    for run_index in tqdm(
        range(args.warmup_runs),
        desc="FEM warmup",
        unit="sample",
        disable=args.no_progress or args.warmup_runs == 0,
        dynamic_ncols=True,
    ):
        solver.solve(velocities_m_s[run_index % len(velocities_m_s)])


def run_timed_samples(
    args: argparse.Namespace,
    solver: FEMHelmholtzSolver,
    sample_indices: list[int],
    velocities_m_s: list[np.ndarray],
) -> list[dict[str, Any]]:
    """Run the timed samples and return per-sample timing rows."""

    rows: list[dict[str, Any]] = []
    for sample_index, velocity in tqdm(
        list(zip(sample_indices, velocities_m_s)),
        desc="FEM timed samples",
        unit="sample",
        disable=args.no_progress,
        dynamic_ncols=True,
    ):
        start = time.perf_counter()
        sample = solver.solve(velocity)
        outer_wall_seconds = time.perf_counter() - start
        diagnostics = sample.diagnostics
        rows.append(
            {
                "sample_index": sample_index,
                "outer_wall_seconds": outer_wall_seconds,
                "sample_generation_seconds": diagnostics.sample_generation_seconds,
                "assemble_seconds": diagnostics.assemble_seconds,
                "linear_solve_seconds": diagnostics.solve_seconds,
                "solve_method": diagnostics.solve_method,
                "bicgstab_info": diagnostics.bicgstab_info,
                "matrix_nnz": diagnostics.matrix_nnz,
                "max_abs_pressure": diagnostics.max_abs_pressure,
            }
        )
    return rows


def build_summary(
    args: argparse.Namespace,
    velocity_path: Path,
    sample_indices: list[int],
    rows: list[dict[str, Any]],
    solver: FEMHelmholtzSolver,
) -> dict[str, Any]:
    """Return a stable JSON-serializable timing summary."""

    outer = [float(row["outer_wall_seconds"]) for row in rows]
    sample_generation = [float(row["sample_generation_seconds"]) for row in rows]
    assembly = [float(row["assemble_seconds"]) for row in rows]
    linear_solve = [float(row["linear_solve_seconds"]) for row in rows]
    return {
        "velocity": str(velocity_path),
        "sample_indices": sample_indices,
        "timed_sample_count": len(rows),
        "warmup_runs": args.warmup_runs,
        "timed_region": (
            "Per-sample FEMHelmholtzSolver.solve call. Velocity samples are loaded "
            "and the solver mesh is constructed before timing; no pressure I/O is performed."
        ),
        "settings": {
            "frequency_hz": args.frequency_hz,
            "domain_size_x_m": args.domain_size_x_m,
            "domain_size_y_m": args.domain_size_y_m,
            "source_x_m": args.source_x_m,
            "source_rows_below_top": args.source_rows_below_top,
            "source_spread_grid_cells": args.source_spread_grid_cells,
            "source_amplitude": args.source_amplitude,
            "abc_velocity_m_s": args.abc_velocity_m_s,
            "abc_scale": args.abc_scale,
            "velocity_sampling": args.velocity_sampling,
            "legacy_mask_shape": list(args.legacy_mask_shape),
            "linear_solver": args.linear_solver,
        },
        "velocity_shape": [solver.ny, solver.nx],
        "spacing_m": [solver.dx, solver.dy],
        "mean_outer_wall_seconds": mean(outer),
        "std_outer_wall_seconds": stdev(outer),
        "mean_sample_generation_seconds": mean(sample_generation),
        "std_sample_generation_seconds": stdev(sample_generation),
        "mean_assemble_seconds": mean(assembly),
        "std_assemble_seconds": stdev(assembly),
        "mean_linear_solve_seconds": mean(linear_solve),
        "std_linear_solve_seconds": stdev(linear_solve),
        "min_sample_generation_seconds": min(sample_generation),
        "max_sample_generation_seconds": max(sample_generation),
        "per_sample": [format_json_row(row) for row in rows],
    }


def mean(values: list[float]) -> float:
    """Return a regular float mean for JSON output."""

    return float(statistics.fmean(values))


def stdev(values: list[float]) -> float:
    """Return sample standard deviation, or zero for one timed value."""

    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def format_json_row(row: dict[str, Any]) -> dict[str, Any]:
    """Format a per-sample row for JSON output without losing useful precision."""

    formatted = dict(row)
    for key in (
        "outer_wall_seconds",
        "sample_generation_seconds",
        "assemble_seconds",
        "linear_solve_seconds",
        "max_abs_pressure",
    ):
        formatted[key] = float(formatted[key])
    return formatted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-sample timing rows to a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_csv_row(row))


def format_csv_row(row: dict[str, Any]) -> dict[str, str]:
    """Format one timing row for CSV output."""

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
