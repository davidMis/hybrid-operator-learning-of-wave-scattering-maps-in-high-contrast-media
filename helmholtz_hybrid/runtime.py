# Overview:
# Configure local cache directories and resolve concrete PyTorch devices before
# importing libraries that otherwise write into user-level caches or require an
# indexed CUDA device.
from __future__ import annotations

import os
from pathlib import Path

import torch


def set_default_cache_dirs(root: str | Path = "outputs/cache") -> None:
    cache_root = Path(root)
    huggingface_root = cache_root / "huggingface"
    transformers_root = huggingface_root / "transformers"
    matplotlib_root = cache_root / "matplotlib"
    huggingface_root.mkdir(parents=True, exist_ok=True)
    transformers_root.mkdir(parents=True, exist_ok=True)
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(huggingface_root.resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_root.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_root.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))


def resolve_torch_device(requested: str, *, require_cuda_for_auto: bool = False) -> torch.device:
    """Resolve a device string and turn generic CUDA requests into ``cuda:0``.

    Some third-party operations reject ``torch.device("cuda")`` because it has
    no explicit index. Returning a concrete device here keeps model loading and
    training behavior consistent across neuralop, scOT, and PyTorch.
    """

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda:0"
        elif require_cuda_for_auto:
            raise RuntimeError(
                "No CUDA device is visible. Run this command on a GPU node or pass "
                "--device cpu only for a small functional smoke test."
            )
        else:
            requested = "cpu"

    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but PyTorch reports that CUDA is unavailable.")

    device_index = 0 if device.index is None else device.index
    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise RuntimeError(
            f"Requested cuda:{device_index}, but only {device_count} CUDA device(s) are visible."
        )
    torch.cuda.set_device(device_index)
    return torch.device("cuda", device_index)
