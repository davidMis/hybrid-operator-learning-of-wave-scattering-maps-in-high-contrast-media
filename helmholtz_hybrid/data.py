# Overview:
# Define dataset adapters for the processed Helmholtz arrays used in the paper.
# Each task-specific Dataset exposes the channel layout expected by FNO or scOT,
# while load_task_dataset centralizes split and task validation.
from __future__ import annotations

from pathlib import Path
from typing import Type

import numpy as np
import torch
from torch.utils.data import Dataset


PRESSURE_CHANNELS = ("real", "imag")
TASKS = ("smooth2smooth", "contrast", "sharp2sharp", "hybrid")
SPLITS = ("train", "validation", "test")


def _load_array(directory: str | Path, filename: str) -> np.ndarray:
    # Copy-on-write mmap keeps samples tensor-compatible without copying each
    # read-only slice in __getitem__; any accidental writes stay private.
    return np.load(Path(directory) / filename, mmap_mode="c")


def dataloader_performance_kwargs(num_workers: int, use_cuda: bool) -> dict[str, object]:
    """Return DataLoader options that reduce host/device input stalls."""

    if num_workers < 0:
        raise ValueError(f"DataLoader worker count must be non-negative; got {num_workers}.")
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


class _BasePaperDataset(Dataset):
    input_channels: int
    output_channels: int = 2
    resolution: int

    def __len__(self) -> int:
        return self.length


class ContrastDataset(_BasePaperDataset):
    """Residual task: (delta v, smooth pressure) -> pressure residual."""

    input_channels = 3

    def __init__(
        self,
        velocity_delta: np.ndarray,
        pressure_smooth: np.ndarray,
        pressure_delta: np.ndarray,
    ) -> None:
        self.velocity_delta = velocity_delta
        self.pressure_smooth = pressure_smooth
        self.pressure_delta = pressure_delta
        self.length = int(velocity_delta.shape[0])
        self.resolution = int(velocity_delta.shape[-1])

    @staticmethod
    def load(input_directory: str | Path) -> "ContrastDataset":
        return ContrastDataset(
            velocity_delta=_load_array(input_directory, "velocity_delta.npy"),
            pressure_smooth=_load_array(input_directory, "pressure_smooth.npy"),
            pressure_delta=_load_array(input_directory, "pressure_delta.npy"),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        velocity = torch.from_numpy(self.velocity_delta[index])[None]
        pressure = torch.from_numpy(self.pressure_smooth[index])
        target = torch.from_numpy(self.pressure_delta[index])
        return {"x": torch.cat([velocity, pressure], dim=0), "y": target}


class Smooth2SmoothDataset(_BasePaperDataset):
    """Smooth task: smoothed velocity -> smooth pressure."""

    input_channels = 1

    def __init__(self, velocity_smooth: np.ndarray, pressure_smooth: np.ndarray) -> None:
        self.velocity_smooth = velocity_smooth
        self.pressure_smooth = pressure_smooth
        self.length = int(velocity_smooth.shape[0])
        self.resolution = int(velocity_smooth.shape[-1])

    @staticmethod
    def load(input_directory: str | Path) -> "Smooth2SmoothDataset":
        return Smooth2SmoothDataset(
            velocity_smooth=_load_array(input_directory, "velocity_smooth.npy"),
            pressure_smooth=_load_array(input_directory, "pressure_smooth.npy"),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(self.velocity_smooth[index])[None]
        y = torch.from_numpy(self.pressure_smooth[index])
        return {"x": x, "y": y}


class Sharp2SharpDataset(_BasePaperDataset):
    """Sharp task: high-contrast velocity -> full pressure."""

    input_channels = 1

    def __init__(self, velocity_sharp: np.ndarray, pressure_sharp: np.ndarray) -> None:
        self.velocity_sharp = velocity_sharp
        self.pressure_sharp = pressure_sharp
        self.length = int(velocity_sharp.shape[0])
        self.resolution = int(velocity_sharp.shape[-1])

    @staticmethod
    def load(input_directory: str | Path) -> "Sharp2SharpDataset":
        return Sharp2SharpDataset(
            velocity_sharp=_load_array(input_directory, "velocity_sharp.npy"),
            pressure_sharp=_load_array(input_directory, "pressure_sharp.npy"),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(self.velocity_sharp[index])[None]
        y = torch.from_numpy(self.pressure_sharp[index])
        return {"x": x, "y": y}


class HybridDataset(_BasePaperDataset):
    """Hybrid evaluation input: (smooth velocity, delta velocity) -> full pressure."""

    input_channels = 2

    def __init__(
        self,
        velocity_smooth: np.ndarray,
        velocity_delta: np.ndarray,
        pressure_sharp: np.ndarray,
    ) -> None:
        self.velocity_smooth = velocity_smooth
        self.velocity_delta = velocity_delta
        self.pressure_sharp = pressure_sharp
        self.length = int(velocity_smooth.shape[0])
        self.resolution = int(velocity_smooth.shape[-1])

    @staticmethod
    def load(input_directory: str | Path) -> "HybridDataset":
        return HybridDataset(
            velocity_smooth=_load_array(input_directory, "velocity_smooth.npy"),
            velocity_delta=_load_array(input_directory, "velocity_delta.npy"),
            pressure_sharp=_load_array(input_directory, "pressure_sharp.npy"),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        velocity_smooth = torch.from_numpy(self.velocity_smooth[index])[None]
        velocity_delta = torch.from_numpy(self.velocity_delta[index])[None]
        target = torch.from_numpy(self.pressure_sharp[index])
        return {"x": torch.cat([velocity_smooth, velocity_delta], dim=0), "y": target}


class HybridSmoothSourceDataset(_BasePaperDataset):
    """Hybrid ablation input with exact smooth pressure and full-pressure target."""

    input_channels = 2

    def __init__(
        self,
        velocity_smooth: np.ndarray,
        velocity_delta: np.ndarray,
        pressure_smooth: np.ndarray,
        pressure_sharp: np.ndarray,
    ) -> None:
        self.velocity_smooth = velocity_smooth
        self.velocity_delta = velocity_delta
        self.pressure_smooth = pressure_smooth
        self.pressure_sharp = pressure_sharp
        self.length = int(velocity_smooth.shape[0])
        self.resolution = int(velocity_smooth.shape[-1])

    @staticmethod
    def load(input_directory: str | Path) -> "HybridSmoothSourceDataset":
        """Load arrays needed to compare exact and FNO-predicted smooth pressure."""

        return HybridSmoothSourceDataset(
            velocity_smooth=_load_array(input_directory, "velocity_smooth.npy"),
            velocity_delta=_load_array(input_directory, "velocity_delta.npy"),
            pressure_smooth=_load_array(input_directory, "pressure_smooth.npy"),
            pressure_sharp=_load_array(input_directory, "pressure_sharp.npy"),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        velocity_smooth = torch.from_numpy(self.velocity_smooth[index])[None]
        velocity_delta = torch.from_numpy(self.velocity_delta[index])[None]
        pressure_smooth = torch.from_numpy(self.pressure_smooth[index])
        pressure_sharp = torch.from_numpy(self.pressure_sharp[index])
        return {
            "x": torch.cat([velocity_smooth, velocity_delta], dim=0),
            "pressure_smooth": pressure_smooth,
            "y": pressure_sharp,
        }


DATASET_BY_TASK: dict[str, Type[_BasePaperDataset]] = {
    "smooth2smooth": Smooth2SmoothDataset,
    "contrast": ContrastDataset,
    "sharp2sharp": Sharp2SharpDataset,
    "hybrid": HybridDataset,
}


def split_dir(data_root: str | Path, dataset: str, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Expected one of {SPLITS}.")
    return Path(data_root) / dataset / split


def load_task_dataset(
    data_root: str | Path,
    dataset: str,
    split: str,
    task: str,
) -> _BasePaperDataset:
    if task not in DATASET_BY_TASK:
        raise ValueError(f"Unknown task '{task}'. Expected one of {TASKS}.")
    return DATASET_BY_TASK[task].load(split_dir(data_root, dataset, split))


class ScOTDatasetWrapper(Dataset):
    """Adapter from the paper dataset dictionaries to scOT's Trainer format."""

    def __init__(self, dataset: _BasePaperDataset, which: str = "train", **kwargs) -> None:
        self.dataset = dataset
        self.which = which
        self.num_trajectories = len(dataset)
        self.input_dim = dataset.input_channels
        self.output_dim = dataset.output_channels
        self.resolution = dataset.resolution
        self.label_description = "[p_real,p_imag]"
        self.N_max = 50_000
        self.N_val = 5_000
        self.N_test = 5_000
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        return {"pixel_values": sample["x"], "labels": sample["y"]}
