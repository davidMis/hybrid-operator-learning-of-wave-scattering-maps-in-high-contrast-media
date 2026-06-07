# Overview:
# Compare generated TDFD Helmholtz fields against the published complex pressure
# arrays. The primary metric aligns each generated field by the best complex
# scalar first, which separates solver geometry errors from unknown source
# amplitude and phase conventions in the inherited dataset.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_TRANSFORMS = (
    "identity",
    "conjugate",
    "flip_y",
    "flip_y_conjugate",
    "flip_x",
    "flip_x_conjugate",
    "flip_xy",
    "flip_xy_conjugate",
)


@dataclass(frozen=True)
class ComplexFieldComparison:
    """Metrics for one generated field after the best requested transform."""

    transform: str
    aligned_relative_l2: float
    raw_relative_l2: float
    correlation_abs: float
    scale_real: float
    scale_imag: float
    scale_abs: float
    scale_phase_deg: float
    target_norm: float
    predicted_norm: float
    max_abs_target: float
    max_abs_predicted: float

    def to_row(self) -> dict[str, str]:
        """Return a stable CSV representation of comparison metrics."""

        return {
            "transform": self.transform,
            "aligned_relative_l2": f"{self.aligned_relative_l2:.9f}",
            "raw_relative_l2": f"{self.raw_relative_l2:.9f}",
            "correlation_abs": f"{self.correlation_abs:.9f}",
            "scale_real": f"{self.scale_real:.9e}",
            "scale_imag": f"{self.scale_imag:.9e}",
            "scale_abs": f"{self.scale_abs:.9e}",
            "scale_phase_deg": f"{self.scale_phase_deg:.6f}",
            "target_norm": f"{self.target_norm:.9e}",
            "predicted_norm": f"{self.predicted_norm:.9e}",
            "max_abs_target": f"{self.max_abs_target:.9e}",
            "max_abs_predicted": f"{self.max_abs_predicted:.9e}",
        }


def load_complex_pressure_sample(path: str | Path, sample_index: int | None = None) -> np.ndarray:
    """Load one complex pressure field from raw or processed paper arrays."""

    array = np.load(Path(path), mmap_mode="r")
    if array.ndim == 2:
        if sample_index not in (None, 0):
            raise ValueError(
                f"{path} is a single complex pressure field, but sample index {sample_index} was requested."
            )
        return np.asarray(array, dtype=np.complex64)
    if array.ndim == 3:
        if np.iscomplexobj(array):
            if sample_index is None:
                raise ValueError(f"{path} contains {array.shape[0]} samples; pass --sample-indices.")
            return np.asarray(array[sample_index], dtype=np.complex64)
        if array.shape[0] == 2 and sample_index in (None, 0):
            return pressure_channels_to_complex(np.asarray(array, dtype=np.float32))
    if array.ndim == 4 and array.shape[1] == 2:
        if sample_index is None:
            raise ValueError(f"{path} contains {array.shape[0]} samples; pass --sample-indices.")
        return pressure_channels_to_complex(np.asarray(array[sample_index], dtype=np.float32))

    raise ValueError(
        "Expected pressure shape [H,W] complex, [N,H,W] complex, [2,H,W], or [N,2,H,W]; "
        f"got {array.shape} from {path}."
    )


def pressure_channels_to_complex(array: np.ndarray) -> np.ndarray:
    """Convert real-valued [2,H,W] pressure channels to one complex [H,W] field."""

    if array.ndim != 3 or array.shape[0] != 2:
        raise ValueError(f"Expected pressure channels with shape [2,H,W], got {array.shape}.")
    return np.asarray(array[0], dtype=np.float32) + 1j * np.asarray(array[1], dtype=np.float32)


def complex_to_pressure_channels(field: np.ndarray) -> np.ndarray:
    """Convert one complex [H,W] field to float32 [2,H,W] channels."""

    if field.ndim != 2:
        raise ValueError(f"Expected a 2D complex pressure field, got {field.shape}.")
    return np.stack([field.real, field.imag], axis=0).astype(np.float32, copy=False)


def upsample_velocity_nearest(velocity: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a velocity model by integer nearest-neighbor replication."""

    if factor < 1:
        raise ValueError(f"Upsample factor must be positive; got {factor}.")
    if factor == 1:
        return np.asarray(velocity, dtype=np.float32)
    return np.repeat(np.repeat(np.asarray(velocity, dtype=np.float32), factor, axis=0), factor, axis=1)


def downsample_complex_mean(field: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool a complex field by an integer factor in both spatial axes."""

    if factor < 1:
        raise ValueError(f"Downsample factor must be positive; got {factor}.")
    if factor == 1:
        return np.asarray(field, dtype=np.complex64)
    height, width = field.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(f"Cannot mean-pool field shape {field.shape} by factor {factor}.")
    reshaped = np.asarray(field, dtype=np.complex64).reshape(
        height // factor,
        factor,
        width // factor,
        factor,
    )
    return reshaped.mean(axis=(1, 3)).astype(np.complex64, copy=False)


def parse_transforms(value: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize a comma-separated transform list and validate the names."""

    if isinstance(value, str):
        transforms = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        transforms = tuple(value)
    if not transforms:
        raise ValueError("At least one transform must be requested.")
    unknown = sorted(set(transforms) - set(DEFAULT_TRANSFORMS))
    if unknown:
        raise ValueError(
            f"Unknown pressure transform(s): {', '.join(unknown)}. "
            f"Expected any of {', '.join(DEFAULT_TRANSFORMS)}."
        )
    return transforms


def apply_pressure_transform(field: np.ndarray, transform: str) -> np.ndarray:
    """Apply a spatial/conjugation transform used to diagnose convention mismatches."""

    if transform not in DEFAULT_TRANSFORMS:
        raise ValueError(f"Unknown pressure transform '{transform}'.")
    transformed = np.asarray(field)
    if "flip_y" in transform or "flip_xy" in transform:
        transformed = np.flip(transformed, axis=0)
    if "flip_x" in transform or "flip_xy" in transform:
        transformed = np.flip(transformed, axis=1)
    if "conjugate" in transform:
        transformed = np.conj(transformed)
    return np.asarray(transformed, dtype=np.complex64)


def crop_field(field: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Crop top, bottom, left, and right cells before computing metrics."""

    top, bottom, left, right = crop
    if min(crop) < 0:
        raise ValueError(f"Crop values must be non-negative; got {crop}.")
    height, width = field.shape
    if top + bottom >= height or left + right >= width:
        raise ValueError(f"Crop {crop} removes all data from field shape {field.shape}.")
    y_slice = slice(top, height - bottom if bottom else height)
    x_slice = slice(left, width - right if right else width)
    return field[y_slice, x_slice]


def compare_complex_fields(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    transforms: Iterable[str] = DEFAULT_TRANSFORMS,
    crop: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> ComplexFieldComparison:
    """Return the best complex-aligned comparison over the requested transforms."""

    if predicted.shape != target.shape:
        raise ValueError(f"Predicted and target field shapes must match; got {predicted.shape} and {target.shape}.")
    target_cropped = crop_field(np.asarray(target, dtype=np.complex64), crop)
    comparisons = [
        compare_one_transform(
            crop_field(apply_pressure_transform(predicted, transform), crop),
            target_cropped,
            transform=transform,
        )
        for transform in transforms
    ]
    return min(comparisons, key=lambda item: item.aligned_relative_l2)


def compare_one_transform(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    transform: str,
) -> ComplexFieldComparison:
    """Compare one transformed field to target with optimal complex scaling."""

    predicted = np.asarray(predicted, dtype=np.complex64)
    target = np.asarray(target, dtype=np.complex64)
    target_norm = field_norm(target)
    predicted_norm = field_norm(predicted)
    if target_norm == 0.0:
        raise ValueError("Target pressure norm is zero; cannot compute relative errors.")
    if predicted_norm == 0.0:
        scale = 0.0 + 0.0j
        aligned_relative_l2 = float("inf")
        correlation_abs = 0.0
    else:
        scale = np.vdot(predicted, target) / np.vdot(predicted, predicted)
        aligned_relative_l2 = relative_l2(scale * predicted, target)
        correlation_abs = abs(np.vdot(predicted, target)) / (predicted_norm * target_norm)

    raw_relative_l2 = relative_l2(predicted, target)
    return ComplexFieldComparison(
        transform=transform,
        aligned_relative_l2=float(aligned_relative_l2),
        raw_relative_l2=float(raw_relative_l2),
        correlation_abs=float(correlation_abs),
        scale_real=float(np.real(scale)),
        scale_imag=float(np.imag(scale)),
        scale_abs=float(abs(scale)),
        scale_phase_deg=float(np.angle(scale, deg=True)),
        target_norm=target_norm,
        predicted_norm=predicted_norm,
        max_abs_target=float(np.max(np.abs(target))),
        max_abs_predicted=float(np.max(np.abs(predicted))),
    )


def relative_l2(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return the complex relative L2 error over a uniform grid."""

    return field_norm(np.asarray(predicted) - np.asarray(target)) / field_norm(np.asarray(target))


def field_norm(field: np.ndarray) -> float:
    """Return the Euclidean norm of one complex field."""

    return float(np.sqrt(np.sum(np.abs(field) ** 2)))
