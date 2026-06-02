#!/usr/bin/env python3
# Overview:
# Create appendix-style grids of randomly selected sharp velocity models and the
# real part of their target pressure fields. The script reads prepared .npy
# arrays and writes publication-ready PNGs without requiring trained checkpoints.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helmholtz_hybrid.runtime import set_default_cache_dirs

set_default_cache_dirs()

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot appendix-style grids of input wavespeeds and target pressures.",
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
        help="Processed dataset name under --data-root.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Prepared split to sample examples from.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of rows in each output image grid.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in each output image grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used to select sample indices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory where velocity and pressure example figures will be saved.",
    )
    return parser.parse_args()


def plot_grid(images: np.ndarray, indices: np.ndarray, cmap: str, output: Path, vmin=None, vmax=None) -> None:
    """Plot a fixed index grid from a memory-mapped image array."""

    rows, cols = indices.shape
    fig, axes = plt.subplots(rows, cols, figsize=(5, 7.6))
    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    for row in range(rows):
        for col in range(cols):
            ax = axes[row, col]
            ax.imshow(images[indices[row, col]], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    split_root = args.data_root / args.dataset / args.split
    # Arrays are memory mapped so plotting works without loading the full 50k
    # sample dataset into RAM.
    velocity = np.load(split_root / "velocity_sharp.npy", mmap_mode="r")
    pressure = np.load(split_root / "pressure_sharp.npy", mmap_mode="r")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(velocity), size=args.rows * args.cols, replace=False).reshape(args.rows, args.cols)

    plot_grid(
        images=velocity,
        indices=indices,
        cmap="bwr",
        output=args.output_dir / f"{args.dataset}_{args.split}_velocity_examples.png",
    )

    # Channel 0 is the real component by convention in prepare_data.py.
    real_pressure = pressure[:, 0]
    pressure_abs = float(np.max(np.abs(real_pressure[indices])))
    plot_grid(
        images=real_pressure,
        indices=indices,
        cmap="seismic",
        output=args.output_dir / f"{args.dataset}_{args.split}_pressure_real_examples.png",
        vmin=-pressure_abs,
        vmax=pressure_abs,
    )


if __name__ == "__main__":
    main()
