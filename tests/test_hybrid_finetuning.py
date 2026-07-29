# Overview:
# Exercise the differentiable hybrid composition, joint epoch update, streamed
# validation metric, and paired checkpoint layout without loading neuralop or
# scOT checkpoints.
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from helmholtz_hybrid.hybrid_finetuning import (
    EndToEndHybridOperator,
    evaluate_hybrid_epoch,
    save_hybrid_checkpoint,
    train_hybrid_epoch,
)


class TinyHybridDataset(Dataset):
    """Deterministic fields with an exactly representable hybrid target."""

    def __init__(self, samples: int = 8, resolution: int = 4) -> None:
        generator = torch.Generator().manual_seed(7)
        self.inputs = torch.randn(samples, 2, resolution, resolution, generator=generator)
        smooth = self.inputs[:, 0:1].repeat(1, 2, 1, 1)
        residual = self.inputs[:, 1:2].repeat(1, 2, 1, 1)
        self.targets = smooth + residual

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.inputs[index], "y": self.targets[index]}


class TinySmooth(nn.Module):
    """One-by-one convolution standing in for a smooth-task FNO."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 2, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)

    def save_checkpoint(self, directory: str | Path, name: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / f"{name}_state_dict.pt")
        torch.save({"toy": True}, directory / f"{name}_metadata.pkl")


class TinyContrast(nn.Module):
    """One-by-one convolution with a tuple output matching scOT semantics."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 2, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor):
        return (self.projection(inputs),)

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}\n")
        torch.save(self.state_dict(), directory / "pytorch_model.bin")


def make_tiny_model() -> EndToEndHybridOperator:
    torch.manual_seed(11)
    return EndToEndHybridOperator(TinySmooth(), TinyContrast())


def test_end_to_end_forward_backpropagates_through_both_components() -> None:
    model = make_tiny_model()
    sample = TinyHybridDataset(samples=2)[0]
    prediction = model(sample["x"].unsqueeze(0))
    prediction.square().mean().backward()

    assert prediction.shape == (1, 2, 4, 4)
    assert model.smooth_model.projection.weight.grad is not None
    assert model.contrast_model.projection.weight.grad is not None


def test_training_and_validation_epoch_return_finite_sample_means() -> None:
    model = make_tiny_model()
    loader = DataLoader(TinyHybridDataset(), batch_size=2, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    initial_smooth = model.smooth_model.projection.weight.detach().clone()
    initial_contrast = model.contrast_model.projection.weight.detach().clone()

    train_error = train_hybrid_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        epoch=1,
        gradient_accumulation_steps=2,
    )
    validation_error = evaluate_hybrid_epoch(
        model,
        loader,
        torch.device("cpu"),
        epoch=1,
    )

    assert torch.isfinite(torch.tensor(train_error))
    assert torch.isfinite(torch.tensor(validation_error))
    assert not torch.equal(initial_smooth, model.smooth_model.projection.weight)
    assert not torch.equal(initial_contrast, model.contrast_model.projection.weight)


def test_checkpoint_writer_saves_reloadable_pair_and_metadata(tmp_path: Path) -> None:
    model = make_tiny_model()
    save_hybrid_checkpoint(
        model,
        tmp_path,
        epoch=3,
        validation_mean_relative_l2=0.125,
        extra_metadata={"source": "test"},
    )

    assert (tmp_path / "fno" / "best_model_state_dict.pt").is_file()
    assert (tmp_path / "fno" / "best_model_metadata.pkl").is_file()
    assert (tmp_path / "scot" / "config.json").is_file()
    assert (tmp_path / "scot" / "pytorch_model.bin").is_file()
    metadata = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert metadata["best_epoch"] == 3
    assert metadata["validation_mean_relative_l2"] == 0.125
    assert metadata["source"] == "test"
