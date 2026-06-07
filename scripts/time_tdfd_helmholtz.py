#!/usr/bin/env python3
# Overview:
# Time one Devito time-domain finite-difference sample generation for the
# 40 Hz Helmholtz sharp-to-sharp task. The script loads one velocity model
# outside the timed region, compiles the Devito operator once, then reports the
# wall-clock time spent generating the complex pressure field.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helmholtz_hybrid.tdfd import (
    DEFAULT_BACKGROUND_KM_S,
    DEFAULT_CHUNK_STEPS,
    DEFAULT_DOMAIN_SIZE_M,
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
    benchmark_tdfd_helmholtz_sample,
    load_velocity_sample,
    save_pressure_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and time one 40 Hz Helmholtz pressure sample using a Devito "
            "time-domain finite-difference solve with on-the-fly Fourier accumulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--velocity",
        type=Path,
        required=True,
        help="Path to a velocity .npy file with shape [ny,nx] or [N,ny,nx].",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Sample index to load when --velocity points to a batched [N,ny,nx] array.",
    )
    parser.add_argument(
        "--velocity-units",
        choices=("auto", "km/s", "m/s"),
        default="auto",
        help="Units of the velocity input; auto treats large median values as m/s.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=DEFAULT_FREQUENCY_HZ,
        help="Target Helmholtz frequency accumulated from the time-domain wavefield.",
    )
    parser.add_argument(
        "--domain-size-m",
        type=float,
        nargs=2,
        metavar=("LX", "LY"),
        default=(DEFAULT_DOMAIN_SIZE_M, DEFAULT_DOMAIN_SIZE_M),
        help="Physical domain side lengths in meters for the x and y axes.",
    )
    parser.add_argument(
        "--source-x-m",
        type=float,
        default=None,
        help="Source x-coordinate in meters; omit to use the horizontal center.",
    )
    parser.add_argument(
        "--source-y-m",
        type=float,
        default=None,
        help=(
            "Source y-coordinate in meters; omit to place the source one grid cell "
            "below the free surface."
        ),
    )
    parser.add_argument(
        "--start-time-ms",
        type=float,
        default=DEFAULT_START_TIME_MS,
        help="Start time for the Ricker-source simulation in milliseconds.",
    )
    parser.add_argument(
        "--end-time-ms",
        type=float,
        default=DEFAULT_END_TIME_MS,
        help="End time for the Ricker-source simulation in milliseconds.",
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
        help="Additional meters added to the automatic absorbing-boundary thickness.",
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
        help="Divide the accumulated pressure by the source wavelet Fourier coefficient.",
    )
    parser.add_argument(
        "--minimum-points-per-wavelength",
        type=float,
        default=DEFAULT_MINIMUM_POINTS_PER_WAVELENGTH,
        help="Minimum grid points per minimum-velocity wavelength required before running.",
    )
    parser.add_argument(
        "--allow-low-ppw",
        action="store_true",
        help="Disable the points-per-wavelength guard for diagnostic sweeps.",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=DEFAULT_CHUNK_STEPS,
        help="Number of time steps per Devito apply call; smaller chunks update the progress bar more often.",
    )
    parser.add_argument(
        "--backend",
        choices=("env", "cpu", "gpu"),
        default="env",
        help=(
            "Devito backend preset. gpu sets nvc/openacc/nvidiaX defaults; env preserves "
            "the caller's DEVITO_* variables."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value set before Devito is imported.",
    )
    parser.add_argument(
        "--devito-arch",
        default=None,
        help="Optional DEVITO_ARCH override, for example nvc on NVIDIA OpenACC builds.",
    )
    parser.add_argument(
        "--devito-language",
        default=None,
        help="Optional DEVITO_LANGUAGE override, for example openacc or openmp.",
    )
    parser.add_argument(
        "--devito-platform",
        default=None,
        help="Optional DEVITO_PLATFORM override, for example nvidiaX.",
    )
    parser.add_argument(
        "--gpu-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask Devito to keep wavefield, source, and Fourier accumulators resident on the GPU.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Untimed sample-generation runs after compilation and before timing.",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=1,
        help="Number of timed sample-generation runs to average.",
    )
    parser.add_argument(
        "--output-pressure",
        type=Path,
        default=None,
        help="Optional .npy path for the generated pressure sample; saving is excluded from timing.",
    )
    parser.add_argument(
        "--complex-output",
        action="store_true",
        help="Save --output-pressure as complex [ny,nx] instead of processed float32 [2,ny,nx].",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional JSON path for timing settings and diagnostics.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during warmup and timed solves.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        velocity = load_velocity_sample(
            args.velocity,
            sample_index=args.sample_index,
            units=args.velocity_units,
        )
        runtime = DevitoRuntimeSettings(
            backend=args.backend,
            devito_arch=args.devito_arch,
            devito_language=args.devito_language,
            devito_platform=args.devito_platform,
            cuda_visible_devices=args.cuda_visible_devices,
            gpu_fit=args.gpu_fit,
        )
        settings = TDFDHelmholtzSettings(
            frequency_hz=args.frequency_hz,
            domain_size_x_m=args.domain_size_m[0],
            domain_size_y_m=args.domain_size_m[1],
            source_x_m=args.source_x_m,
            source_y_m=args.source_y_m,
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
                0.0 if args.allow_low_ppw else args.minimum_points_per_wavelength
            ),
            chunk_steps=args.chunk_steps,
            runtime=runtime,
        )

        total_runs = args.warmup_runs + args.timed_runs
        with tqdm(
            total=None,
            disable=args.no_progress,
            desc="TDFD sample",
            unit="step",
            dynamic_ncols=True,
        ) as bar:

            def progress(increment: int, total_steps: int) -> None:
                if bar.total != total_runs * total_steps:
                    bar.reset(total=total_runs * total_steps)
                bar.update(increment)

            result = benchmark_tdfd_helmholtz_sample(
                velocity,
                settings,
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
                progress_callback=None if args.no_progress else progress,
            )

        if args.output_pressure is not None:
            save_pressure_sample(
                result.last_sample.pressure,
                args.output_pressure,
                channels=not args.complex_output,
            )

        metadata = build_metadata(args, result)
        if args.metadata_output is not None:
            args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
            args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")

        print(json.dumps(metadata, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, TDFDConfigurationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def build_metadata(args: argparse.Namespace, result) -> dict[str, object]:
    """Return a stable JSON-serializable timing summary."""

    diagnostics = result.last_sample.diagnostics
    return {
        "velocity": str(args.velocity),
        "sample_index": args.sample_index,
        "warmup_runs": result.warmup_runs,
        "timed_runs": result.timed_runs,
        "timed_seconds": [round(value, 6) for value in result.timed_seconds],
        "seconds_per_sample": result.seconds_per_sample,
        "milliseconds_per_sample": result.milliseconds_per_sample,
        "velocity_shape": list(diagnostics.velocity_shape),
        "padded_shape": list(diagnostics.padded_shape),
        "spacing_m": list(diagnostics.spacing_m),
        "nbl": diagnostics.nbl,
        "nt": diagnostics.nt,
        "critical_dt_ms": diagnostics.critical_dt_ms,
        "points_per_min_wavelength": diagnostics.points_per_min_wavelength,
        "source_coordinates_m": list(diagnostics.source_coordinates_m),
        "source_spectrum": {
            "real": diagnostics.source_spectrum.real,
            "imag": diagnostics.source_spectrum.imag,
        },
        "max_abs_pressure": diagnostics.max_abs_pressure,
        "devito_apply_seconds_last_run": diagnostics.devito_apply_seconds,
        "output_pressure": str(args.output_pressure) if args.output_pressure is not None else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
