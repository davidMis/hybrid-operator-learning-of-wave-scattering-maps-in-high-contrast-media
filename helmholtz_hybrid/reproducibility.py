# Overview:
# Provide reproducibility helpers shared by training scripts. The module counts
# trainable parameters, configures deterministic/TF32 PyTorch behavior, and
# writes run manifests that capture command, environment, and git metadata.
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch


PARAMETER_COUNT_METHOD = "real_scalar_trainable_v1"


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of real scalar trainable parameters in a model.

    PyTorch reports each complex-valued tensor entry as one element, but an
    optimizer stores independent real and imaginary values. Count those as two
    scalar parameters so FNO spectral weights are comparable to real-valued
    architectures such as scOT.
    """

    total = 0
    for parameter in model.parameters():
        multiplier = 2 if parameter.is_complex() else 1
        total += multiplier * parameter.numel()
    return int(total)


def configure_torch_reproducibility(deterministic: bool, allow_tf32: bool) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)


def write_run_manifest(
    output_dir: str | Path,
    args: Namespace,
    *,
    run_name: str,
    model_type: str,
    parameters: int,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_name": run_name,
        "model_type": model_type,
        "parameters": int(parameters),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "args": _jsonable(vars(args)),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        },
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "dirty": _git(["status", "--porcelain"]) != "",
        },
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
