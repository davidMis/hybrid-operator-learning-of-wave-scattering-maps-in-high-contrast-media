# Overview:
# Shared NumPy sample-loading helpers for numerical Helmholtz tools. The
# functions validate the raw or processed paper array layouts, normalize velocity
# units to km/s where needed, and save generated complex pressure fields in the
# same two-channel format used by the prepared datasets.
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


VelocityUnits = Literal["auto", "km/s", "m/s"]


def load_velocity_sample(
    path: str | Path,
    *,
    sample_index: int | None = None,
    units: VelocityUnits = "auto",
) -> np.ndarray:
    """Load one 2D velocity sample from a paper raw or processed NumPy array."""

    array = np.load(Path(path), mmap_mode="r")
    if array.ndim == 2:
        if sample_index not in (None, 0):
            raise ValueError(
                f"{path} is a single 2D velocity field, but sample index {sample_index} was requested."
            )
        velocity = np.asarray(array, dtype=np.float32)
    elif array.ndim == 3:
        if sample_index is None:
            raise ValueError(f"{path} contains {array.shape[0]} samples; pass --sample-index.")
        if not 0 <= sample_index < array.shape[0]:
            raise ValueError(
                f"Sample index {sample_index} is out of bounds for {path}, which has {array.shape[0]} samples."
            )
        velocity = np.asarray(array[sample_index], dtype=np.float32)
    else:
        raise ValueError(
            f"Expected a velocity array with shape [ny,nx] or [N,ny,nx], got {array.shape} from {path}."
        )

    return normalize_velocity_units(velocity, units=units)


def normalize_velocity_units(velocity: np.ndarray, *, units: VelocityUnits = "auto") -> np.ndarray:
    """Return velocity in km/s with helpful validation for common data mistakes."""

    if velocity.ndim != 2:
        raise ValueError(f"Expected a 2D velocity field, got shape {velocity.shape}.")
    if not np.isfinite(velocity).all():
        raise ValueError("Velocity field contains NaN or Inf values.")
    if np.any(velocity <= 0.0):
        raise ValueError("Velocity field must be strictly positive everywhere.")

    velocity = np.asarray(velocity, dtype=np.float32)
    if units == "m/s" or (units == "auto" and float(np.nanmedian(velocity)) > 100.0):
        velocity = velocity / 1000.0
    elif units not in ("auto", "km/s"):
        raise ValueError(f"Unknown velocity unit mode '{units}'. Expected auto, km/s, or m/s.")

    vmin = float(np.min(velocity))
    vmax = float(np.max(velocity))
    if vmin < 0.1 or vmax > 10.0:
        raise ValueError(
            "Velocity values should be in km/s after unit conversion; "
            f"got min={vmin:.3f}, max={vmax:.3f}. Pass --velocity-units if the input units are different."
        )
    return velocity


def save_pressure_sample(
    pressure: np.ndarray,
    path: str | Path,
    *,
    channels: bool = True,
) -> None:
    """Save a generated pressure sample in processed-channel or complex layout."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if channels:
        data = np.stack([pressure.real, pressure.imag], axis=0).astype(np.float32, copy=False)
    else:
        data = np.asarray(pressure, dtype=np.complex64)
    np.save(output, data)
