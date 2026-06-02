#!/usr/bin/env python3
# Overview:
# Convert raw 50k Helmholtz arrays into the processed split layout used by the
# paper training scripts. This script keeps velocity fields real-valued, stacks
# complex pressures as [real, imag], and writes train/validation/test arrays with
# deterministic contiguous splits.
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DATASETS = ("const_back", "grf_back", "varied_salt")
RAW_FILENAMES = {
    "velocity_sharp": "velocity_sharp.npy",
    "velocity_smooth": "velocity_smooth.npy",
    "pressure_sharp_complex": "pressure_sharp.npy",
    "pressure_smooth_complex": "pressure_smooth.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare raw Helmholtz arrays into the processed data layout used by the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Directory containing raw dataset subfolders such as const_back/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed train/validation/test folders will be written.",
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="const_back",
        help=(
            "Dataset folder to prepare. Each raw dataset folder must contain "
            "velocity_sharp.npy, velocity_smooth.npy, pressure_sharp.npy, and pressure_smooth.npy. "
            "Use 'all' to process const_back, grf_back, and varied_salt."
        ),
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=40_000,
        help="Number of leading samples assigned to the training split.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=5_000,
        help="Number of samples assigned to the validation split after the training slice.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=5_000,
        help="Number of samples assigned to the test split after the validation slice.",
    )
    parser.add_argument(
        "--downsample-to",
        type=int,
        default=None,
        help="Optional square resolution produced by area/mean pooling; omit to keep raw resolution.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of samples transformed and written per chunk to limit memory use.",
    )
    return parser.parse_args()


def stack_complex_channels(array: np.ndarray) -> np.ndarray:
    """Convert complex arrays from [N,H,W] to float32 [N,2,H,W]."""

    return np.stack([array.real, array.imag], axis=1).astype(np.float32, copy=False)


def downsample_mean(array: np.ndarray, target_size: int) -> np.ndarray:
    """Mean-pool square image arrays to a lower square resolution."""

    if array.shape[-1] == target_size and array.shape[-2] == target_size:
        return array
    if array.shape[-1] % target_size != 0 or array.shape[-2] % target_size != 0:
        raise ValueError(f"Cannot mean-pool shape {array.shape} to {target_size}x{target_size}.")
    factor_y = array.shape[-2] // target_size
    factor_x = array.shape[-1] // target_size
    return array.reshape(array.shape[0], target_size, factor_y, target_size, factor_x).mean(axis=(2, 4))


def load_raw_dataset(raw_root: Path, dataset: str) -> dict[str, np.ndarray]:
    root = raw_root / dataset
    # Memory mapping lets this script process the full 50k sample arrays without
    # loading all velocities and pressures into RAM at once.
    return {key: np.load(root / filename, mmap_mode="r") for key, filename in RAW_FILENAMES.items()}


def split_slices(train_size: int, validation_size: int, test_size: int) -> dict[str, slice]:
    """Return deterministic contiguous slices matching the paper split."""

    validation_start = train_size
    test_start = train_size + validation_size
    end = test_start + test_size
    return {
        "train": slice(0, train_size),
        "validation": slice(validation_start, test_start),
        "test": slice(test_start, end),
    }


def output_resolution(raw: dict[str, np.ndarray], downsample_to: int | None) -> int:
    if downsample_to is not None:
        return downsample_to
    return int(raw["velocity_sharp"].shape[-1])


def create_output_arrays(split_root: Path, n_samples: int, resolution: int) -> dict[str, np.memmap]:
    split_root.mkdir(parents=True, exist_ok=True)
    # open_memmap writes valid .npy files incrementally, so interrupted writes do
    # not require materializing full processed arrays in memory.
    return {
        "velocity_sharp": np.lib.format.open_memmap(
            split_root / "velocity_sharp.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, resolution, resolution),
        ),
        "velocity_smooth": np.lib.format.open_memmap(
            split_root / "velocity_smooth.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, resolution, resolution),
        ),
        "velocity_delta": np.lib.format.open_memmap(
            split_root / "velocity_delta.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, resolution, resolution),
        ),
        "pressure_sharp": np.lib.format.open_memmap(
            split_root / "pressure_sharp.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, 2, resolution, resolution),
        ),
        "pressure_smooth": np.lib.format.open_memmap(
            split_root / "pressure_smooth.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, 2, resolution, resolution),
        ),
        "pressure_delta": np.lib.format.open_memmap(
            split_root / "pressure_delta.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, 2, resolution, resolution),
        ),
    }


def transform_velocity(array: np.ndarray, downsample_to: int | None) -> np.ndarray:
    if downsample_to is not None:
        array = downsample_mean(array, downsample_to)
    return np.asarray(array, dtype=np.float32)


def transform_pressure(array: np.ndarray, downsample_to: int | None) -> np.ndarray:
    if downsample_to is not None:
        array = downsample_mean(array, downsample_to)
    return stack_complex_channels(array)


def save_processed_dataset(
    raw: dict[str, np.ndarray],
    output_root: Path,
    dataset: str,
    splits: dict[str, slice],
    downsample_to: int | None,
    chunk_size: int,
) -> None:
    from tqdm.auto import tqdm

    required = max(split.stop for split in splits.values())
    n_samples = raw["velocity_sharp"].shape[0]
    if n_samples < required:
        raise ValueError(f"{dataset} has {n_samples} samples, but {required} are required by the split.")

    resolution = output_resolution(raw, downsample_to)

    for split_name, split_slice in splits.items():
        split_root = output_root / dataset / split_name
        print(f"Writing {split_root}")
        split_length = split_slice.stop - split_slice.start
        outputs = create_output_arrays(split_root, split_length, resolution)

        # Transform and persist one chunk at a time. This is the only expensive
        # loop in the preprocessing step for the paper dataset.
        for offset in tqdm(
            range(0, split_length, chunk_size),
            desc=f"{dataset} {split_name}",
            unit="chunk",
            dynamic_ncols=True,
        ):
            count = min(chunk_size, split_length - offset)
            source_slice = slice(split_slice.start + offset, split_slice.start + offset + count)
            dest_slice = slice(offset, offset + count)

            velocity_sharp = transform_velocity(raw["velocity_sharp"][source_slice], downsample_to)
            velocity_smooth = transform_velocity(raw["velocity_smooth"][source_slice], downsample_to)
            pressure_sharp = transform_pressure(raw["pressure_sharp_complex"][source_slice], downsample_to)
            pressure_smooth = transform_pressure(raw["pressure_smooth_complex"][source_slice], downsample_to)

            outputs["velocity_sharp"][dest_slice] = velocity_sharp
            outputs["velocity_smooth"][dest_slice] = velocity_smooth
            outputs["velocity_delta"][dest_slice] = velocity_sharp - velocity_smooth
            outputs["pressure_sharp"][dest_slice] = pressure_sharp
            outputs["pressure_smooth"][dest_slice] = pressure_smooth
            outputs["pressure_delta"][dest_slice] = pressure_sharp - pressure_smooth

        for output in outputs.values():
            output.flush()


def prepare_dataset(args: argparse.Namespace, dataset: str) -> None:
    raw = load_raw_dataset(
        raw_root=args.raw_root,
        dataset=dataset,
    )
    save_processed_dataset(
        raw=raw,
        output_root=args.output_root,
        dataset=dataset,
        splits=split_slices(args.train_size, args.validation_size, args.test_size),
        downsample_to=args.downsample_to,
        chunk_size=args.chunk_size,
    )


def main() -> None:
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        prepare_dataset(args, dataset)


if __name__ == "__main__":
    main()
