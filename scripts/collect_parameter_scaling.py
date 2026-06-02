#!/usr/bin/env python3
# Overview:
# Collect evaluation metrics JSON files into the CSV schema expected by
# plot_parameter_scaling.py. This keeps Figure 4 data traceable back to the
# individual checkpoint evaluations while standardizing panel and model labels.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASK_TO_PANEL = {
    "smooth2smooth": "Smooth",
    "contrast": "Residual",
    "sharp2sharp": "Sharp",
    "hybrid": "Sharp",
}
MODEL_LABEL = {
    "fno": "FNO",
    "scot": "scOT",
    "hybrid": "Hybrid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect evaluation JSON files into the Figure 4 metrics CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "metrics_json",
        type=Path,
        nargs="+",
        help="One or more metrics JSON files produced by scripts/evaluate.py or run_all.sh.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/parameter_scaling.csv"),
        help="Output CSV path consumed by scripts/plot_parameter_scaling.py.",
    )
    return parser.parse_args()


def row_from_metrics(path: Path) -> dict[str, str]:
    """Map one evaluation JSON payload to the Figure 4 CSV row schema."""

    metrics = json.loads(path.read_text())
    task = metrics["task"]
    model_type = metrics["model_type"]
    return {
        "panel": TASK_TO_PANEL[task],
        "model": MODEL_LABEL[model_type],
        "parameters": str(int(metrics["parameters"])),
        "rel_l2": f"{float(metrics['mean_relative_l2']):.12g}",
        "source_json": str(path),
    }


def main() -> None:
    args = parse_args()
    rows = [row_from_metrics(path) for path in args.metrics_json]
    # Sorting keeps the CSV deterministic even when shell glob order differs.
    rows.sort(key=lambda row: (row["panel"], row["model"], int(row["parameters"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "model", "parameters", "rel_l2", "source_json"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
