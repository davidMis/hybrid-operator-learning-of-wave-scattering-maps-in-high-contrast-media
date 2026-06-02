# Overview:
# Define the complex-valued relative L2 objective used by FNO and scOT training.
# Pressures are represented as two real channels [real, imag], so the norm sums
# channel energy before averaging over the spatial grid.
from __future__ import annotations

import warnings

import torch

warnings.filterwarnings("once", category=UserWarning)


def complex_L2_norm(x: torch.Tensor) -> torch.Tensor:
    """Return per-sample complex L2 norms for [N, 2, H, W] tensors."""

    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"Expected tensor with shape [N, 2, H, W], got {tuple(x.shape)}.")
    # Assumes uniform grid, quadrature is simple Riemann sum.
    return torch.square(x).sum(dim=1).mean(dim=(1, 2)).sqrt()


def relative_complex_l2(y_pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return per-sample relative complex L2 errors for pressure predictions."""

    if y_pred.shape != y.shape:
        raise ValueError(f"Prediction and target shapes must match; got {y_pred.shape} and {y.shape}.")
    return complex_L2_norm(y_pred - y) / (complex_L2_norm(y) + eps)


class ComplexL2Loss:
    """Callable relative/absolute complex L2 loss compatible with Trainer APIs."""

    def __init__(self, relative: bool = True, eps: float = 1e-8) -> None:
        self.relative = relative
        self.eps = eps

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs:
            warnings.warn(
                f"ComplexL2Loss.__call__() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if y_pred.shape != y.shape:
            raise ValueError(f"Prediction and target shapes must match; got {y_pred.shape} and {y.shape}.")
        if y_pred.ndim != 4 or y_pred.shape[1] != 2:
            raise ValueError(f"Expected tensors with shape [N, 2, H, W], got {tuple(y_pred.shape)}.")

        if self.relative:
            result = relative_complex_l2(y_pred, y, self.eps)
        else:
            result = complex_L2_norm(y_pred - y)

        return result.sum()
