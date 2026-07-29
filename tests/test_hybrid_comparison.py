# Overview:
# Verify that the hybrid smooth-source ablation holds the residual model and
# sharp-pressure target fixed, computes the expected per-sample errors, and
# writes consistent CSV and LaTeX table artifacts.
from __future__ import annotations

import csv

import numpy as np
import torch

from helmholtz_hybrid.data import HybridSmoothSourceDataset
from helmholtz_hybrid.hybrid_comparison import (
    REQUIRED_SPLIT_FILES,
    HybridSmoothSourceResult,
    evaluate_smooth_sources,
    fno_smooth_checkpoint,
    missing_comparison_inputs,
    render_latex_table,
    scot_contrast_checkpoint,
    write_comparison_outputs,
)
from helmholtz_hybrid.loss import relative_complex_l2


class _BiasedSmoothModel(torch.nn.Module):
    """Return a known smooth field with a fixed real-channel bias."""

    def forward(self, velocity_smooth: torch.Tensor) -> torch.Tensor:
        real = 2.0 * velocity_smooth + 0.05
        return torch.cat([real, torch.zeros_like(real)], dim=1)


class _ImperfectContrastModel(torch.nn.Module):
    """Return 80% of the known residual encoded in velocity_delta."""

    def forward(self, contrast_input: torch.Tensor) -> tuple[torch.Tensor]:
        real = 0.8 * contrast_input[:, 0:1]
        return (torch.cat([real, torch.zeros_like(real)], dim=1),)


def _comparison_dataset() -> HybridSmoothSourceDataset:
    velocity_smooth = np.array(
        [
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.3, 0.4], [0.5, 0.6]],
            [[0.4, 0.5], [0.6, 0.7]],
        ],
        dtype=np.float32,
    )
    velocity_delta = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.3, 0.4], [0.5, 0.6]],
        ],
        dtype=np.float32,
    )
    pressure_smooth = np.stack(
        [2.0 * velocity_smooth, np.zeros_like(velocity_smooth)],
        axis=1,
    )
    pressure_delta = np.stack(
        [velocity_delta, np.zeros_like(velocity_delta)],
        axis=1,
    )
    pressure_sharp = pressure_smooth + pressure_delta
    return HybridSmoothSourceDataset(
        velocity_smooth=velocity_smooth,
        velocity_delta=velocity_delta,
        pressure_smooth=pressure_smooth,
        pressure_sharp=pressure_sharp,
    )


def test_evaluate_smooth_sources_uses_common_sharp_target() -> None:
    dataset = _comparison_dataset()
    ground_truth_errors, trained_fno_errors = evaluate_smooth_sources(
        smooth_model=_BiasedSmoothModel(),
        contrast_model=_ImperfectContrastModel(),
        dataset=dataset,
        device=torch.device("cpu"),
        batch_size=2,
        workers=0,
        show_progress=False,
    )

    exact_smooth = torch.from_numpy(dataset.pressure_smooth)
    sharp_target = torch.from_numpy(dataset.pressure_sharp)
    velocity_smooth = torch.from_numpy(dataset.velocity_smooth)[:, None]
    velocity_delta = torch.from_numpy(dataset.velocity_delta)[:, None]
    predicted_smooth = _BiasedSmoothModel()(velocity_smooth)
    predicted_delta = _ImperfectContrastModel()(
        torch.cat([velocity_delta, exact_smooth], dim=1)
    )[0]
    expected_ground_truth = relative_complex_l2(
        exact_smooth + predicted_delta,
        sharp_target,
    ).numpy()
    expected_trained_fno = relative_complex_l2(
        predicted_smooth + predicted_delta,
        sharp_target,
    ).numpy()

    np.testing.assert_allclose(ground_truth_errors, expected_ground_truth)
    np.testing.assert_allclose(trained_fno_errors, expected_trained_fno)


def test_write_comparison_outputs(tmp_path) -> None:
    results = [
        HybridSmoothSourceResult(
            sweep_size=4,
            fno_parameters=20,
            scot_parameters=21,
            num_samples=5_000,
            split="test",
            ground_truth_mean_relative_l2=0.10,
            trained_fno_mean_relative_l2=0.11,
            ground_truth_median_relative_l2=0.09,
            trained_fno_median_relative_l2=0.10,
            fno_smooth_checkpoint="fno-n4",
            scot_contrast_checkpoint="scot-n4",
        ),
        HybridSmoothSourceResult(
            sweep_size=2,
            fno_parameters=10,
            scot_parameters=11,
            num_samples=5_000,
            split="test",
            ground_truth_mean_relative_l2=0.20,
            trained_fno_mean_relative_l2=0.22,
            ground_truth_median_relative_l2=0.19,
            trained_fno_median_relative_l2=0.21,
            fno_smooth_checkpoint="fno-n2",
            scot_contrast_checkpoint="scot-n2",
        ),
    ]
    csv_path = tmp_path / "comparison.csv"
    latex_path = tmp_path / "comparison.tex"

    write_comparison_outputs(
        results,
        csv_path=csv_path,
        latex_path=latex_path,
    )

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sweep_size"] for row in rows] == ["2", "4"]
    assert rows[0]["hybrid_parameters"] == "21"
    assert float(rows[0]["percent_increase"]) == 10.0

    latex = latex_path.read_text()
    assert latex == render_latex_table(sorted(results, key=lambda result: result.sweep_size))
    assert "& 2 & 4 \\\\" in latex
    assert "Ground truth & 0.2000 & 0.1000" in latex


def test_missing_comparison_inputs_checks_checkpoint_files(tmp_path) -> None:
    data_root = tmp_path / "data"
    checkpoint_root = tmp_path / "checkpoints"
    split_directory = data_root / "const_back" / "test"
    split_directory.mkdir(parents=True)
    for filename in REQUIRED_SPLIT_FILES:
        (split_directory / filename).touch()

    fno_path = fno_smooth_checkpoint(checkpoint_root, "const_back", 2)
    fno_path.mkdir(parents=True)
    (fno_path / "best_model_metadata.pkl").touch()
    (fno_path / "best_model_state_dict.pt").touch()

    scot_path = scot_contrast_checkpoint(checkpoint_root, "const_back", 2)
    scot_path.mkdir(parents=True)
    (scot_path / "config.json").touch()
    (scot_path / "pytorch_model.bin").touch()

    assert missing_comparison_inputs(
        data_root,
        "const_back",
        "test",
        checkpoint_root,
        [2],
    ) == []

    (scot_path / "pytorch_model.bin").unlink()
    assert missing_comparison_inputs(
        data_root,
        "const_back",
        "test",
        checkpoint_root,
        [2],
    ) == [scot_path / "pytorch_model.bin"]
