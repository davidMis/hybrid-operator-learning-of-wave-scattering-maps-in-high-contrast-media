# Overview:
# Regenerate unsplit raw Helmholtz pressure arrays from published velocity
# arrays using the recovered finite-element solver. The generated folder matches
# the raw data layout consumed by scripts/prepare_data.py: velocity_sharp.npy,
# velocity_smooth.npy, pressure_sharp.npy, and pressure_smooth.npy.
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from tqdm.auto import tqdm

from helmholtz_hybrid.fem_helmholtz import FEMConfigurationError, FEMHelmholtzSettings, FEMHelmholtzSolver
from helmholtz_hybrid.sample_io import VelocityUnits, normalize_velocity_units


VelocityKind = Literal["sharp", "smooth"]
VELOCITY_KINDS: tuple[VelocityKind, ...] = ("sharp", "smooth")
RAW_VELOCITY_FILES = {
    "sharp": "velocity_sharp.npy",
    "smooth": "velocity_smooth.npy",
}
RAW_PRESSURE_FILES = {
    "sharp": "pressure_sharp.npy",
    "smooth": "pressure_smooth.npy",
}
GENERATION_FIELDNAMES = [
    "velocity_kind",
    "output_index",
    "source_index",
    "outer_wall_seconds",
    "sample_generation_seconds",
    "assemble_seconds",
    "linear_solve_seconds",
    "solve_method",
    "bicgstab_info",
    "matrix_nnz",
    "max_abs_pressure",
]


def generate_raw_fem_dataset(
    *,
    velocity_root: Path,
    output_root: Path,
    dataset: str,
    sample_start_index: int = 0,
    sample_count: int | None = None,
    velocity_units: VelocityUnits = "auto",
    settings: FEMHelmholtzSettings | None = None,
    sharp_velocity_sampling: str = "legacy-mask128",
    smooth_velocity_sampling: str = "bilinear",
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Generate one raw-layout FEM dataset folder from sharp and smooth velocities."""

    settings = settings or FEMHelmholtzSettings()
    input_dir = velocity_root / dataset
    output_dir = output_root / dataset
    velocity_paths = {kind: input_dir / filename for kind, filename in RAW_VELOCITY_FILES.items()}
    pressure_paths = {kind: output_dir / filename for kind, filename in RAW_PRESSURE_FILES.items()}
    output_velocity_paths = {kind: output_dir / filename for kind, filename in RAW_VELOCITY_FILES.items()}
    metadata_path = output_dir / "fem_generation_metadata.json"
    timings_path = output_dir / "fem_generation_times.csv"

    if velocity_root.resolve() == output_root.resolve():
        raise ValueError(
            "--output-root must differ from --velocity-root so raw input velocities are not overwritten."
        )
    validate_inputs(velocity_paths)
    source_arrays = {kind: np.load(path, mmap_mode="r") for kind, path in velocity_paths.items()}
    sample_indices = resolve_sample_indices(source_arrays["sharp"], sample_start_index, sample_count)
    validate_velocity_stacks(source_arrays, sample_indices)
    ensure_outputs_available(
        [*output_velocity_paths.values(), *pressure_paths.values(), metadata_path, timings_path],
        overwrite=overwrite,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_shape = velocity_sample_shape(source_arrays["sharp"])
    output_arrays = create_output_arrays(output_dir, len(sample_indices), sample_shape)
    solvers: dict[tuple[VelocityKind, tuple[int, int]], FEMHelmholtzSolver] = {}
    rows: list[dict[str, Any]] = []
    sampling_by_kind = {
        "sharp": sharp_velocity_sampling,
        "smooth": smooth_velocity_sampling,
    }

    progress = tqdm(
        total=len(sample_indices) * len(VELOCITY_KINDS),
        desc="FEM raw samples",
        unit="solve",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    try:
        for output_index, source_index in enumerate(sample_indices):
            sharp_km_s = load_velocity_from_stack(source_arrays["sharp"], source_index, velocity_units)
            smooth_km_s = load_velocity_from_stack(source_arrays["smooth"], source_index, velocity_units)
            output_arrays["velocity_sharp"][output_index] = sharp_km_s
            output_arrays["velocity_smooth"][output_index] = smooth_km_s

            for kind, velocity_km_s in (("sharp", sharp_km_s), ("smooth", smooth_km_s)):
                solver_key = (kind, velocity_km_s.shape)
                solver = solvers.get(solver_key)
                if solver is None:
                    solver_settings = replace(settings, velocity_sampling=sampling_by_kind[kind])
                    solver = FEMHelmholtzSolver(velocity_km_s.shape, solver_settings)
                    solvers[solver_key] = solver

                row, pressure = solve_one_pressure(kind, output_index, source_index, solver, velocity_km_s)
                output_arrays[f"pressure_{kind}"][output_index] = pressure
                rows.append(row)
                progress.update(1)
                progress.set_postfix_str(
                    f"{kind} idx={source_index} t={float(row['sample_generation_seconds']):.3g}s"
                )
    finally:
        progress.close()
        for output in output_arrays.values():
            output.flush()

    write_generation_csv(timings_path, rows)
    metadata = build_metadata(
        velocity_root=velocity_root,
        output_root=output_root,
        dataset=dataset,
        sample_indices=sample_indices,
        settings=settings,
        sharp_velocity_sampling=sharp_velocity_sampling,
        smooth_velocity_sampling=smooth_velocity_sampling,
        sample_shape=sample_shape,
        timings_path=timings_path,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def validate_inputs(velocity_paths: dict[VelocityKind, Path]) -> None:
    """Fail early with clear messages when required velocity arrays are absent."""

    missing = [str(path) for path in velocity_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required raw velocity array(s):\n" + "\n".join(missing))


def resolve_sample_indices(array: np.ndarray, sample_start_index: int, sample_count: int | None) -> list[int]:
    """Return source sample indices to regenerate from one velocity stack."""

    if sample_start_index < 0:
        raise ValueError(f"--sample-start-index must be non-negative; got {sample_start_index}.")
    total = velocity_stack_length(array)
    if sample_start_index >= total:
        raise ValueError(f"--sample-start-index {sample_start_index} is outside the {total}-sample velocity array.")
    if sample_count is None:
        sample_count = total - sample_start_index
    if sample_count <= 0:
        raise ValueError(f"--sample-count must be positive when provided; got {sample_count}.")
    stop = sample_start_index + sample_count
    if stop > total:
        raise ValueError(
            f"Requested samples [{sample_start_index}, {stop}) but the velocity array has only {total} samples."
        )
    return list(range(sample_start_index, stop))


def validate_velocity_stacks(arrays: dict[VelocityKind, np.ndarray], sample_indices: list[int]) -> None:
    """Ensure sharp and smooth velocity arrays can produce matching samples."""

    sharp_shape = velocity_sample_shape(arrays["sharp"])
    smooth_shape = velocity_sample_shape(arrays["smooth"])
    if sharp_shape != smooth_shape:
        raise ValueError(f"Sharp and smooth velocity sample shapes differ: {sharp_shape} versus {smooth_shape}.")
    for kind, array in arrays.items():
        if velocity_stack_length(array) <= max(sample_indices):
            raise ValueError(f"{kind} velocity array has too few samples for requested index {max(sample_indices)}.")


def velocity_stack_length(array: np.ndarray) -> int:
    """Return the number of samples in a [H,W] or [N,H,W] velocity array."""

    if array.ndim == 2:
        return 1
    if array.ndim == 3:
        return int(array.shape[0])
    raise ValueError(f"Expected velocity array with shape [H,W] or [N,H,W], got {array.shape}.")


def velocity_sample_shape(array: np.ndarray) -> tuple[int, int]:
    """Return the spatial shape of a velocity stack."""

    if array.ndim == 2:
        return int(array.shape[0]), int(array.shape[1])
    if array.ndim == 3:
        return int(array.shape[1]), int(array.shape[2])
    raise ValueError(f"Expected velocity array with shape [H,W] or [N,H,W], got {array.shape}.")


def ensure_outputs_available(paths: list[Path], *, overwrite: bool) -> None:
    """Protect existing generated arrays unless overwrite was explicitly requested."""

    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing FEM data-generation output(s). "
            "Pass --overwrite to replace them:\n" + "\n".join(existing)
        )


def create_output_arrays(output_dir: Path, sample_count: int, sample_shape: tuple[int, int]) -> dict[str, np.memmap]:
    """Create raw-layout output arrays as writable .npy memmaps."""

    ny, nx = sample_shape
    return {
        "velocity_sharp": np.lib.format.open_memmap(
            output_dir / "velocity_sharp.npy",
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, ny, nx),
        ),
        "velocity_smooth": np.lib.format.open_memmap(
            output_dir / "velocity_smooth.npy",
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, ny, nx),
        ),
        "pressure_sharp": np.lib.format.open_memmap(
            output_dir / "pressure_sharp.npy",
            mode="w+",
            dtype=np.complex64,
            shape=(sample_count, ny, nx),
        ),
        "pressure_smooth": np.lib.format.open_memmap(
            output_dir / "pressure_smooth.npy",
            mode="w+",
            dtype=np.complex64,
            shape=(sample_count, ny, nx),
        ),
    }


def load_velocity_from_stack(array: np.ndarray, sample_index: int, units: VelocityUnits) -> np.ndarray:
    """Load one velocity sample, normalize to km/s, and return a contiguous array."""

    if array.ndim == 2:
        if sample_index != 0:
            raise ValueError(f"Single-sample velocity array only supports sample index 0, got {sample_index}.")
        sample = np.asarray(array, dtype=np.float32)
    else:
        sample = np.asarray(array[sample_index], dtype=np.float32)
    return np.ascontiguousarray(normalize_velocity_units(sample, units=units), dtype=np.float32)


def solve_one_pressure(
    kind: VelocityKind,
    output_index: int,
    source_index: int,
    solver: FEMHelmholtzSolver,
    velocity_km_s: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Solve one FEM pressure field and return timing metadata plus pressure."""

    start = time.perf_counter()
    sample = solver.solve(velocity_km_s * 1000.0)
    outer_seconds = time.perf_counter() - start
    diagnostics = sample.diagnostics
    row = {
        "velocity_kind": kind,
        "output_index": output_index,
        "source_index": source_index,
        "sample_generation_seconds": diagnostics.sample_generation_seconds,
        "outer_wall_seconds": outer_seconds,
        "assemble_seconds": diagnostics.assemble_seconds,
        "linear_solve_seconds": diagnostics.solve_seconds,
        "solve_method": diagnostics.solve_method,
        "bicgstab_info": diagnostics.bicgstab_info,
        "matrix_nnz": diagnostics.matrix_nnz,
        "max_abs_pressure": diagnostics.max_abs_pressure,
    }
    return row, np.asarray(sample.pressure, dtype=np.complex64)


def write_generation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-solve FEM generation timings in a stable CSV schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_generation_row(row))


def format_generation_row(row: dict[str, Any]) -> dict[str, str]:
    """Format one FEM generation timing row for CSV output."""

    formatted: dict[str, str] = {}
    for field in GENERATION_FIELDNAMES:
        value = row.get(field, "")
        if value is None:
            formatted[field] = ""
        elif isinstance(value, float):
            formatted[field] = f"{value:.9g}" if math.isfinite(value) else str(value)
        else:
            formatted[field] = str(value)
    return formatted


def build_metadata(
    *,
    velocity_root: Path,
    output_root: Path,
    dataset: str,
    sample_indices: list[int],
    settings: FEMHelmholtzSettings,
    sharp_velocity_sampling: str,
    smooth_velocity_sampling: str,
    sample_shape: tuple[int, int],
    timings_path: Path,
) -> dict[str, Any]:
    """Return JSON metadata describing the regenerated raw dataset."""

    settings_payload = asdict(settings)
    return {
        "dataset": dataset,
        "velocity_root": str(velocity_root),
        "output_root": str(output_root),
        "sample_count": len(sample_indices),
        "sample_start_index": sample_indices[0],
        "sample_stop_index": sample_indices[-1] + 1,
        "sample_shape": list(sample_shape),
        "raw_layout": {
            "velocity_sharp": "float32 [N,H,W] in km/s",
            "velocity_smooth": "float32 [N,H,W] in km/s",
            "pressure_sharp": "complex64 [N,H,W]",
            "pressure_smooth": "complex64 [N,H,W]",
        },
        "sharp_velocity_sampling": sharp_velocity_sampling,
        "smooth_velocity_sampling": smooth_velocity_sampling,
        "settings": settings_payload,
        "timings_csv": str(timings_path),
    }
