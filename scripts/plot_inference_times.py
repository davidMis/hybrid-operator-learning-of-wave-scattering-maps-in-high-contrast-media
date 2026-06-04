#!/usr/bin/env python3
# Overview:
# Plot inference time per sample for the paper model sweep. The script mirrors
# the Figure 4 panel layout and model colors while plotting each model against
# its actual trainable-parameter count, including summed component counts for
# hybrid models.
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helmholtz_hybrid.runtime import set_default_cache_dirs

set_default_cache_dirs()

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


PANELS = ("Smooth", "Residual", "Sharp")
MODEL_ORDER_BY_PANEL = {
    "Smooth": ("FNO", "scOT"),
    "Residual": ("FNO", "scOT"),
    "Sharp": ("FNO", "scOT", "Hybrid"),
}
MODEL_STYLES = {
    "FNO": {"color": "C0", "marker": "o"},
    "scOT": {"color": "C1", "marker": "o"},
    "Hybrid": {"color": "C2", "marker": "o"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 4-style inference-time scaling from a timing CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timing-csv",
        type=Path,
        required=True,
        help="CSV with panel, model, parameters, and milliseconds_per_sample columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/inference_times.png"),
        help="Output image path for the inference-time scaling figure.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Optional PDF output path for paper workflows that prefer vector figures.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Validate and group CSV rows by figure panel and model label."""

    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    valid_models = set().union(*MODEL_ORDER_BY_PANEL.values())
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"panel", "model", "parameters", "milliseconds_per_sample"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            panel = row["panel"]
            model = row["model"]
            if panel not in PANELS:
                raise ValueError(f"Unknown panel '{panel}'. Expected one of {PANELS}.")
            if model not in valid_models:
                raise ValueError(f"Unknown model '{model}'. Expected one of {sorted(valid_models)}.")
            if model not in MODEL_ORDER_BY_PANEL[panel]:
                raise ValueError(f"Model '{model}' is not expected in panel '{panel}'.")
            grouped[(panel, model)].append(
                (float(row["parameters"]), float(row["milliseconds_per_sample"]))
            )

    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    return grouped


def main() -> None:
    args = parse_args()
    grouped = load_rows(args.timing_csv)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 2]},
        tight_layout=True,
    )

    for ax, panel in zip(axes, PANELS):
        for model in MODEL_ORDER_BY_PANEL[panel]:
            values = grouped.get((panel, model), [])
            if not values:
                continue
            params, milliseconds = zip(*values)
            ax.plot(params, milliseconds, "-", label=model, **MODEL_STYLES[model])
        ax.set_title(panel)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x / 1e6:g}"))

    axes[0].set_ylabel("Inference time per sample (ms)")
    fig.supxlabel("Millions of parameters")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="best")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.output_pdf is not None:
        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_pdf, bbox_inches="tight")


if __name__ == "__main__":
    main()
