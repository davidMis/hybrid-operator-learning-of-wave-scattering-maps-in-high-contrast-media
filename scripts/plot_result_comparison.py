#!/usr/bin/env python3
# Overview:
# Recreate a Figure 3-style qualitative comparison for one validation/test
# sample. The script loads FNO, scOT, and hybrid checkpoints, predicts the sharp
# pressure field, and plots real-part pressure predictions plus absolute errors.
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
import torch

from helmholtz_hybrid.evaluation import (
    load_fno_checkpoint,
    load_scot_checkpoint,
    scot_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a Figure 3-style model comparison for one prepared sample.",
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
        default="test",
        choices=["validation", "test"],
        help="Prepared split containing the sample to plot.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index within --split to visualize.",
    )
    parser.add_argument(
        "--fno-sharp-checkpoint",
        type=Path,
        required=True,
        help="FNO sharp2sharp checkpoint directory used for the FNO baseline column.",
    )
    parser.add_argument(
        "--scot-sharp-checkpoint",
        type=Path,
        required=True,
        help="scOT sharp2sharp checkpoint directory used for the scOT baseline column.",
    )
    parser.add_argument(
        "--fno-smooth-checkpoint",
        type=Path,
        required=True,
        help="FNO smooth2smooth checkpoint directory used as the hybrid smooth model.",
    )
    parser.add_argument(
        "--scot-contrast-checkpoint",
        type=Path,
        required=True,
        help="scOT contrast checkpoint directory used as the hybrid residual model.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/result_comparison.png"),
        help="Output PNG path for the comparison figure.",
    )
    return parser.parse_args()


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copy one NumPy sample into a float32 tensor on the requested device."""

    return torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(device)


def predict_fno(model, velocity: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = to_tensor(velocity, device)[None, None]
        return model(x).cpu().numpy()[0]


def predict_scot(model, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return scot_predictions(model(x)).cpu().numpy()[0]


def load_sample(split_root: Path, index: int) -> dict[str, np.ndarray]:
    """Load only the arrays needed for one qualitative comparison sample."""

    return {
        "velocity_sharp": np.load(split_root / "velocity_sharp.npy", mmap_mode="r")[index],
        "velocity_smooth": np.load(split_root / "velocity_smooth.npy", mmap_mode="r")[index],
        "velocity_delta": np.load(split_root / "velocity_delta.npy", mmap_mode="r")[index],
        "pressure_sharp": np.load(split_root / "pressure_sharp.npy", mmap_mode="r")[index],
    }


def plot_result(
    velocity: np.ndarray,
    expected: np.ndarray,
    fno: np.ndarray,
    scot: np.ndarray,
    hybrid: np.ndarray,
    output: Path,
) -> None:
    """Render expected pressure, model predictions, and real-part errors."""

    pressure_abs = float(np.max(np.abs([expected, fno, scot, hybrid])))
    errors = [np.abs(fno[0] - expected[0]), np.abs(scot[0] - expected[0]), np.abs(hybrid[0] - expected[0])]
    error_max = float(np.max(errors))

    fig = plt.figure(constrained_layout=True, figsize=(8.5, 4))
    grid = fig.add_gridspec(2, 5, width_ratios=[1.0, 0.12, 1.0, 1.0, 1.0], wspace=0.05)
    axes = [[fig.add_subplot(grid[row, col]) for col in [0, 2, 3, 4]] for row in range(2)]

    axes[0][0].set_title("Expected")
    axes[0][0].imshow(expected[0], vmin=-pressure_abs, vmax=pressure_abs, cmap="seismic")
    axes[0][0].set_ylabel("pressure (real part)")

    axes[1][0].imshow(velocity, cmap="bwr")
    axes[1][0].set_ylabel("wavespeed")

    for col, title, prediction, error in zip(
        [1, 2, 3],
        ["FNO", "scOT", "Hybrid"],
        [fno, scot, hybrid],
        errors,
    ):
        axes[0][col].set_title(title)
        pressure_image = axes[0][col].imshow(
            prediction[0],
            vmin=-pressure_abs,
            vmax=pressure_abs,
            cmap="seismic",
        )
        error_image = axes[1][col].imshow(error, vmin=0, vmax=error_max, cmap="inferno")
        if col == 1:
            axes[1][col].set_ylabel("absolute error")

    fig.colorbar(pressure_image, ax=axes[0][1:], location="right", pad=0.01)
    fig.colorbar(error_image, ax=axes[1][1:], location="right", pad=0.01)

    for ax in fig.axes:
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

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sample = load_sample(args.data_root / args.dataset / args.split, args.index)

    fno_sharp = load_fno_checkpoint(args.fno_sharp_checkpoint, device)
    fno_smooth = load_fno_checkpoint(args.fno_smooth_checkpoint, device)
    scot_sharp = load_scot_checkpoint(args.scot_sharp_checkpoint, device)
    scot_contrast = load_scot_checkpoint(args.scot_contrast_checkpoint, device)

    fno_prediction = predict_fno(fno_sharp, sample["velocity_sharp"], device)

    scot_input = to_tensor(sample["velocity_sharp"], device)[None, None]
    scot_prediction = predict_scot(scot_sharp, scot_input)

    # Hybrid composition mirrors the paper algorithm: predict smooth pressure
    # from v_smooth, predict residual pressure from (v_delta, p_smooth), then add.
    with torch.no_grad():
        velocity_smooth = to_tensor(sample["velocity_smooth"], device)[None, None]
        velocity_delta = to_tensor(sample["velocity_delta"], device)[None, None]
        pressure_smooth = fno_smooth(velocity_smooth)
        contrast_input = torch.cat([velocity_delta, pressure_smooth], dim=1)
        pressure_delta = scot_predictions(scot_contrast(contrast_input))
        hybrid_prediction = (pressure_smooth + pressure_delta).cpu().numpy()[0]

    plot_result(
        velocity=sample["velocity_sharp"],
        expected=sample["pressure_sharp"],
        fno=fno_prediction,
        scot=scot_prediction,
        hybrid=hybrid_prediction,
        output=args.output,
    )


if __name__ == "__main__":
    main()
