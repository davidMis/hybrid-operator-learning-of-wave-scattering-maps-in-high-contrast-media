#!/usr/bin/env python3
# Overview:
# Evaluate trained FNO, scOT, or hybrid checkpoints on validation/test splits.
# The script reports mean and median relative complex L2, includes model
# parameter counts for Figure 4 aggregation, and can optionally save prediction
# arrays for downstream visualization. It can evaluate one checkpoint or schedule
# the full paper sweep across available GPUs with coordinated progress bars.
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_TYPES = ("fno", "scot", "hybrid")
SINGLE_MODEL_TASKS = ("smooth2smooth", "contrast", "sharp2sharp")
EVALUATION_SPLITS = ("validation", "test")
DEFAULT_SWEEP_SIZES = (2, 4, 6, 8, 10)
OVERALL_BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]"
TASK_BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}"


@dataclass(frozen=True)
class SweepJob:
    """One checkpoint evaluation in the paper sweep."""

    label: str
    request_kwargs: dict[str, Any]
    output_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one FNO/scOT/hybrid checkpoint or the full paper checkpoint sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate the full paper sweep under --checkpoint-root and write one JSON per checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing processed dataset folders.",
    )
    parser.add_argument(
        "--dataset",
        default="const_back",
        help="Processed dataset name under --data-root.",
    )
    parser.add_argument(
        "--split",
        choices=EVALUATION_SPLITS,
        default="test",
        help="Prepared split used for evaluation metrics.",
    )
    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPES,
        help="Checkpoint family to evaluate; hybrid composes smooth FNO and contrast scOT checkpoints.",
    )
    parser.add_argument(
        "--task",
        choices=SINGLE_MODEL_TASKS,
        default=None,
        help="Task for single-model fno/scot evaluation; omitted for hybrid evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint directory for --model-type fno or scot.",
    )
    parser.add_argument(
        "--fno-smooth-checkpoint",
        type=Path,
        help="FNO smooth2smooth checkpoint directory used by --model-type hybrid.",
    )
    parser.add_argument(
        "--scot-contrast-checkpoint",
        type=Path,
        help="scOT contrast checkpoint directory used by --model-type hybrid.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes used during evaluation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device used for inference, for example cuda, cuda:0, or cpu; auto selects CUDA when available.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path where the metrics JSON should be written.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Optional compressed .npz path containing expected, actual, and per-sample rel_l2 arrays.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Sweep mode: checkpoint root containing fno/ and scot/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sweep mode: directory where per-checkpoint metrics JSON files will be written.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWEEP_SIZES),
        help="Sweep mode: FNO layer counts and scOT per-stage depth counts to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        choices=SINGLE_MODEL_TASKS,
        nargs="+",
        default=list(SINGLE_MODEL_TASKS),
        help="Sweep mode: single-model tasks to evaluate.",
    )
    parser.add_argument(
        "--include-hybrid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sweep mode: also evaluate the hybrid sharp reconstruction for every size.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Sweep mode: comma-separated torch devices, for example cuda:0,cuda:1. "
            "Numeric tokens such as 0,1 are interpreted as CUDA device indices. "
            "Defaults to all visible CUDA devices, or cpu when CUDA is unavailable."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Sweep mode: cap the number of concurrent device workers.",
    )
    return parser.parse_args()


def request_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Convert parsed CLI arguments into evaluation request keyword arguments."""

    if args.model_type is None:
        raise ValueError("Single-checkpoint evaluation requires --model-type. Use --sweep for the paper sweep.")
    return {
        "data_root": args.data_root,
        "dataset": args.dataset,
        "split": args.split,
        "model_type": args.model_type,
        "task": args.task,
        "checkpoint": args.checkpoint,
        "fno_smooth_checkpoint": args.fno_smooth_checkpoint,
        "scot_contrast_checkpoint": args.scot_contrast_checkpoint,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "store_predictions": args.predictions_out is not None,
        "device": None if args.device == "auto" else args.device,
    }


def fno_checkpoint_name(dataset: str, task: str, size: int) -> str:
    """Return the FNO run directory name used by run_all.sh and the paper release."""

    return f"fno_{dataset}_{task}_layers{size}"


def scot_checkpoint_name(dataset: str, task: str, size: int) -> str:
    """Return the scOT run directory name used by run_all.sh and the paper release."""

    return f"scot_{dataset}_{task}_depths{size}-{size}-{size}-{size}"


def build_sweep_jobs(args: argparse.Namespace) -> list[SweepJob]:
    """Build the ordered paper-sweep evaluation jobs and output paths."""

    if args.checkpoint_root is None:
        raise ValueError("Sweep evaluation requires --checkpoint-root.")
    if args.output_dir is None:
        raise ValueError("Sweep evaluation requires --output-dir.")
    if args.output_json is not None:
        raise ValueError("Use --output-dir instead of --output-json with --sweep.")
    if args.predictions_out is not None:
        raise ValueError("--predictions-out is only supported for single-checkpoint evaluation.")
    if any(size < 1 for size in args.sizes):
        raise ValueError(f"Sweep sizes must be positive; got {args.sizes}.")

    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    jobs: list[SweepJob] = []
    base_kwargs: dict[str, Any] = {
        "data_root": args.data_root,
        "dataset": args.dataset,
        "split": args.split,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "store_predictions": False,
    }

    for task in args.tasks:
        for size in args.sizes:
            jobs.append(
                SweepJob(
                    label=f"fno {task} layers={size}",
                    request_kwargs={
                        **base_kwargs,
                        "model_type": "fno",
                        "task": task,
                        "checkpoint": checkpoint_root / "fno" / fno_checkpoint_name(args.dataset, task, size),
                    },
                    output_json=output_dir / f"fno_{args.dataset}_{task}_layers{size}.json",
                )
            )
            jobs.append(
                SweepJob(
                    label=f"scot {task} depths={size}",
                    request_kwargs={
                        **base_kwargs,
                        "model_type": "scot",
                        "task": task,
                        "checkpoint": checkpoint_root / "scot" / scot_checkpoint_name(args.dataset, task, size),
                    },
                    output_json=output_dir / f"scot_{args.dataset}_{task}_depths{size}.json",
                )
            )

    if args.include_hybrid:
        for size in args.sizes:
            jobs.append(
                SweepJob(
                    label=f"hybrid sharp size={size}",
                    request_kwargs={
                        **base_kwargs,
                        "model_type": "hybrid",
                        "task": None,
                        "fno_smooth_checkpoint": checkpoint_root
                        / "fno"
                        / fno_checkpoint_name(args.dataset, "smooth2smooth", size),
                        "scot_contrast_checkpoint": checkpoint_root
                        / "scot"
                        / scot_checkpoint_name(args.dataset, "contrast", size),
                    },
                    output_json=output_dir / f"hybrid_{args.dataset}_sharp_layers{size}.json",
                )
            )

    if not jobs:
        raise ValueError("No evaluation jobs were requested.")
    return jobs


def normalize_device_token(token: str) -> str:
    """Interpret bare numeric device tokens as CUDA device indices."""

    token = token.strip()
    if token.isdigit():
        return f"cuda:{token}"
    return token


def selected_sweep_devices(args: argparse.Namespace) -> list[str]:
    """Return the torch devices used by sweep workers."""

    if args.devices is not None:
        devices = [normalize_device_token(token) for token in args.devices.split(",") if token.strip()]
    elif args.device != "auto":
        devices = [args.device]
    else:
        import torch

        if torch.cuda.is_available():
            devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]

    if args.max_parallel is not None:
        if args.max_parallel < 1:
            raise ValueError(f"--max-parallel must be positive; got {args.max_parallel}.")
        devices = devices[: args.max_parallel]
    if not devices:
        raise ValueError("No evaluation devices selected.")
    return devices


def sweep_worker(slot: int, device: str, job_queue, event_queue) -> None:
    """Evaluate jobs assigned by the parent process on one device."""

    from helmholtz_hybrid.evaluation import EvaluationRequest, evaluate_checkpoint, write_evaluation_outputs

    try:
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                break

            event_queue.put(
                {
                    "type": "started",
                    "slot": slot,
                    "device": device,
                    "label": job.label,
                }
            )

            def progress_callback(increment: int, total: int | None) -> None:
                event_queue.put(
                    {
                        "type": "progress",
                        "slot": slot,
                        "increment": increment,
                        "total": total,
                    }
                )

            try:
                request_kwargs = dict(job.request_kwargs)
                request_kwargs["device"] = device
                result = evaluate_checkpoint(
                    EvaluationRequest(**request_kwargs),
                    progress_callback=progress_callback,
                    show_progress=False,
                )
                write_evaluation_outputs(result, output_json=job.output_json)
            except BaseException as error:
                event_queue.put(
                    {
                        "type": "failed",
                        "slot": slot,
                        "device": device,
                        "label": job.label,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                return

            event_queue.put(
                {
                    "type": "completed",
                    "slot": slot,
                    "device": device,
                    "label": job.label,
                    "mean_relative_l2": result.metrics["mean_relative_l2"],
                    "output_json": str(job.output_json),
                }
            )
    finally:
        event_queue.put({"type": "worker_done", "slot": slot, "device": device})


def reset_task_bar(bar: tqdm, description: str, total: int = 1) -> None:
    """Reset a per-worker bar for a new checkpoint task."""

    bar.reset(total=total)
    bar.set_description_str(description)
    bar.set_postfix_str("")
    bar.refresh()


def run_sweep_jobs(jobs: list[SweepJob], devices: list[str]) -> int:
    """Run sweep jobs across devices while rendering coordinated progress bars."""

    devices = devices[: min(len(devices), len(jobs))]
    disable_progress = not sys.stderr.isatty()
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    event_queue = context.Queue()
    for job in jobs:
        job_queue.put(job)

    processes = [
        context.Process(target=sweep_worker, args=(slot, device, job_queue, event_queue))
        for slot, device in enumerate(devices)
    ]
    for process in processes:
        process.start()

    overall_bar = tqdm(
        total=len(jobs),
        desc="overall",
        unit="model",
        position=0,
        dynamic_ncols=True,
        bar_format=OVERALL_BAR_FORMAT,
        disable=disable_progress,
    )
    task_bars = [
        tqdm(
            total=1,
            desc=f"{device} idle",
            unit="batch",
            position=slot + 1,
            leave=False,
            dynamic_ncols=True,
            bar_format=TASK_BAR_FORMAT,
            disable=disable_progress,
        )
        for slot, device in enumerate(devices)
    ]

    failed_event: dict[str, Any] | None = None
    finished_slots: set[int] = set()
    try:
        while len(finished_slots) < len(processes):
            try:
                event = event_queue.get(timeout=0.2)
            except queue.Empty:
                for slot, process in enumerate(processes):
                    if slot in finished_slots or process.exitcode is None:
                        continue
                    if process.exitcode != 0:
                        failed_event = {
                            "label": f"worker {slot}",
                            "device": devices[slot],
                            "error": f"worker exited with status {process.exitcode}",
                            "traceback": "",
                        }
                        finished_slots.add(slot)
                        break
                if failed_event is not None:
                    break
                continue

            event_type = event["type"]
            slot = event["slot"]
            if event_type == "started":
                reset_task_bar(task_bars[slot], f"{event['device']} {event['label']}")
            elif event_type == "progress":
                bar = task_bars[slot]
                total = event["total"]
                if total is not None and total != bar.total:
                    description = bar.desc
                    reset_task_bar(bar, description, total=max(int(total), 1))
                increment = int(event["increment"])
                if increment > 0:
                    bar.update(increment)
            elif event_type == "completed":
                bar = task_bars[slot]
                if bar.total is not None and bar.n < bar.total:
                    bar.update(bar.total - bar.n)
                bar.set_postfix_str(f"mean={event['mean_relative_l2']:.4g}")
                overall_bar.update(1)
                overall_bar.set_postfix_str(event["label"])
            elif event_type == "failed":
                failed_event = event
                break
            elif event_type == "worker_done":
                finished_slots.add(slot)
    finally:
        if failed_event is not None:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join()
        for bar in task_bars:
            bar.close()
        overall_bar.close()

    if failed_event is not None:
        print(
            f"ERROR: {failed_event['label']} failed on {failed_event['device']}: {failed_event['error']}",
            file=sys.stderr,
        )
        if failed_event.get("traceback"):
            print(failed_event["traceback"], file=sys.stderr)
        return 1

    return 0


def run_sweep(args: argparse.Namespace) -> int:
    """Evaluate the paper sweep across selected devices."""

    jobs = build_sweep_jobs(args)
    devices = selected_sweep_devices(args)
    print(f"Evaluating {len(jobs)} checkpoints across {len(devices)} device(s): {', '.join(devices)}")
    status = run_sweep_jobs(jobs, devices)
    if status == 0:
        print(f"Wrote evaluation metrics to {args.output_dir}")
    return status


def run_single(args: argparse.Namespace) -> int:
    """Evaluate one checkpoint request."""

    from helmholtz_hybrid.evaluation import EvaluationRequest, evaluate_checkpoint, write_evaluation_outputs

    result = evaluate_checkpoint(EvaluationRequest(**request_kwargs_from_args(args)))
    print(json.dumps(result.metrics, indent=2))
    write_evaluation_outputs(
        result,
        output_json=args.output_json,
        predictions_out=args.predictions_out,
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.sweep:
            return run_sweep(args)
        return run_single(args)
    except ModuleNotFoundError as error:
        print(
            f"ERROR: Missing Python dependency '{error.name}'. Install the project environment with "
            "`python -m pip install -e .` before running evaluation.",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
