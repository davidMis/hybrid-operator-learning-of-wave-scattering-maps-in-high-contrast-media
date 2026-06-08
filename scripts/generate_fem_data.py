#!/usr/bin/env python3
# Overview:
# Regenerate the unsplit raw published Helmholtz data from sharp and smooth
# velocity arrays using the FEM solver. The output folder matches the raw layout
# consumed by scripts/prepare_data.py; splitting remains a separate step.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helmholtz_hybrid.fem_data_generation import generate_raw_fem_dataset
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
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for raw FEM data regeneration."""

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the unsplit raw published Helmholtz arrays with the FEM solver. "
            "The output can be passed to scripts/prepare_data.py for train/validation/test splitting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--velocity-root",
        type=Path,
        default=Path("data/raw"),
        help="Root containing <dataset>/velocity_sharp.npy and <dataset>/velocity_smooth.npy.",
    )
    parser.add_argument(
        "--dataset",
        default="const_back",
        help="Raw dataset folder under --velocity-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/generated/fem/raw"),
        help="Root where the regenerated raw-layout dataset folder will be written.",
    )
    parser.add_argument(
        "--sample-start-index",
        type=int,
        default=0,
        help="First raw velocity sample index to regenerate.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Number of raw samples to regenerate; omit to process all samples through the end of the array.",
    )
    parser.add_argument(
        "--velocity-units",
        choices=("auto", "km/s", "m/s"),
        default="auto",
        help="Units of the input velocity arrays; auto treats large median values as m/s.",
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
        "--sharp-velocity-sampling",
        choices=("nearest", "bilinear", "legacy-mask128"),
        default=DEFAULT_FEM_VELOCITY_SAMPLING,
        help="Element-centroid velocity sampling used for velocity_sharp.npy.",
    )
    parser.add_argument(
        "--smooth-velocity-sampling",
        choices=("nearest", "bilinear", "legacy-mask128"),
        default="bilinear",
        help="Element-centroid velocity sampling used for velocity_smooth.npy.",
    )
    parser.add_argument(
        "--legacy-mask-shape",
        type=int,
        nargs=2,
        metavar=("NY", "NX"),
        default=(128, 128),
        help="Low-resolution salt mask shape used when a velocity sampling mode is legacy-mask128.",
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
        "--overwrite",
        action="store_true",
        help="Replace an existing regenerated raw dataset at --output-root/--dataset.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def make_settings(args: argparse.Namespace) -> FEMHelmholtzSettings:
    """Build solver settings shared by sharp and smooth pressure generation."""

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
        legacy_mask_shape=tuple(args.legacy_mask_shape),
        linear_solver=args.linear_solver,
        spilu_drop_tol=args.spilu_drop_tol,
        spilu_fill_factor=args.spilu_fill_factor,
        bicgstab_rtol=args.bicgstab_rtol,
        bicgstab_maxiter=args.bicgstab_maxiter,
    )


def main() -> int:
    """Run raw FEM data regeneration and print output metadata."""

    args = parse_args()
    try:
        metadata = generate_raw_fem_dataset(
            velocity_root=args.velocity_root,
            output_root=args.output_root,
            dataset=args.dataset,
            sample_start_index=args.sample_start_index,
            sample_count=args.sample_count,
            velocity_units=args.velocity_units,
            settings=make_settings(args),
            sharp_velocity_sampling=args.sharp_velocity_sampling,
            smooth_velocity_sampling=args.smooth_velocity_sampling,
            overwrite=args.overwrite,
            show_progress=not args.no_progress,
        )
        print(json.dumps(metadata, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, FEMConfigurationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
