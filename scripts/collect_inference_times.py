#!/usr/bin/env python3
# Overview:
# Measure forward-only inference time for the FNO, scOT, and hybrid checkpoints
# used in Figure 4. The script schedules one checkpoint job per CUDA device,
# writes a long-form CSV with exact timings and parameter counts, and writes the
# compact n-column CSV used for the manuscript inference-time table.
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import queue
import sys
import traceback
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helmholtz_hybrid.inference_timing import (
    DEFAULT_SWEEP_SIZES,
    SINGLE_MODEL_TASKS,
    InferenceTimingJob,
    InferenceTimingResult,
    InferenceTimingSettings,
    build_inference_timing_jobs,
    missing_inference_inputs,
    time_inference_job,
)


PANEL_ORDER = ("Smooth", "Residual", "Sharp")
MODEL_ORDER_BY_PANEL = {
    "Smooth": ("FNO", "scOT"),
    "Residual": ("FNO", "scOT"),
    "Sharp": ("FNO", "scOT", "Hybrid"),
}
LONG_FIELDNAMES = [
    "panel",
    "task",
    "model",
    "model_type",
    "size",
    "parameters",
    "num_samples",
    "batch_size",
    "batches",
    "warmup_passes",
    "timed_passes",
    "timed_pass_seconds",
    "seconds_per_sample",
    "milliseconds_per_sample",
    "preload_device_batches",
    "device",
    "device_name",
    "checkpoint",
]
OVERALL_BAR_FORMAT = (
    "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} "
    "[{elapsed}<{remaining}]"
)
TASK_BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time forward-only inference for the Figure 4 checkpoint sweep. "
            "By default, all visible CUDA devices are used with one model per device."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        choices=("validation", "test"),
        default="test",
        help="Prepared split used for inference timing.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Checkpoint root containing fno/ and scot/ subdirectories. Defaults to "
            "outputs/checkpoints/<dataset>/paper."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Long-form CSV output path. Defaults to results/<dataset>/paper/inference_times.csv.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=None,
        help="Compact manuscript-table CSV path. Defaults next to --output as inference_times_table.csv.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWEEP_SIZES),
        help="FNO layer counts and shared scOT depth counts to time.",
    )
    parser.add_argument(
        "--tasks",
        choices=SINGLE_MODEL_TASKS,
        nargs="+",
        default=list(SINGLE_MODEL_TASKS),
        help="Standalone learning tasks to include in the timing sweep.",
    )
    parser.add_argument(
        "--include-hybrid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also time hybrid sharp reconstruction for every requested size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size used for every model family.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes used while preparing inference batches.",
    )
    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=1,
        help="Number of forward-only passes to run before timing.",
    )
    parser.add_argument(
        "--timed-passes",
        type=int,
        default=1,
        help="Number of forward-only passes averaged for each table entry.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on batches per pass for smoke tests; omit for publication timings.",
    )
    parser.add_argument(
        "--preload-device-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Move inference inputs to the target device before timing so reported "
            "values exclude disk, DataLoader, and host-to-device transfer time."
        ),
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated torch devices, for example cuda:0,cuda:1. Numeric tokens "
            "such as 0,1 are interpreted as CUDA device indices. Defaults to all visible CUDA devices."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Cap the number of concurrent device workers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs and output paths without loading data or checkpoints.",
    )
    args = parser.parse_args()
    if args.checkpoint_root is None:
        args.checkpoint_root = Path("outputs/checkpoints") / args.dataset / "paper"
    if args.output is None:
        args.output = Path("results") / args.dataset / "paper" / "inference_times.csv"
    if args.table_output is None:
        args.table_output = args.output.with_name("inference_times_table.csv")
    return args


def normalize_device_token(token: str) -> str:
    """Interpret bare numeric device tokens as CUDA device indices."""

    token = token.strip()
    if token.isdigit():
        return f"cuda:{token}"
    return token


def selected_devices(args: argparse.Namespace) -> list[str]:
    """Return one worker device per visible CUDA device unless overridden."""

    if args.devices is not None:
        devices = [
            normalize_device_token(token)
            for token in args.devices.split(",")
            if token.strip()
        ]
    else:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise ValueError(
                "No CUDA devices are visible. Run this script on a GPU node, or pass --devices cpu "
                "for a functional dry run or smoke test."
            )
        devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]

    if args.max_parallel is not None:
        if args.max_parallel < 1:
            raise ValueError(f"--max-parallel must be positive; got {args.max_parallel}.")
        devices = devices[: args.max_parallel]
    if not devices:
        raise ValueError("No timing devices selected.")
    if len(set(devices)) != len(devices):
        raise ValueError(
            f"Duplicate devices are not allowed because each worker owns one device: {devices}."
        )
    return devices


def timing_worker(
    slot: int,
    device: str,
    settings: InferenceTimingSettings,
    job_queue,
    event_queue,
) -> None:
    """Run inference timing jobs assigned by the parent process on one device."""

    try:
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                break

            total_passes = settings.warmup_passes + settings.timed_passes
            event_queue.put(
                {
                    "type": "started",
                    "slot": slot,
                    "device": device,
                    "label": job.label,
                    "total": total_passes,
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
                result = time_inference_job(job, settings, device, progress_callback=progress_callback)
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
                    "result": result,
                }
            )
    finally:
        event_queue.put({"type": "worker_done", "slot": slot, "device": device})


def reset_task_bar(bar: tqdm, description: str, total: int = 1) -> None:
    """Reset a per-worker progress bar for a new inference timing job."""

    bar.reset(total=total)
    bar.set_description_str(description)
    bar.set_postfix_str("")
    bar.refresh()


def run_timing_jobs(
    jobs: list[InferenceTimingJob],
    devices: list[str],
    settings: InferenceTimingSettings,
    output: Path,
    table_output: Path,
    table_sizes: list[int],
) -> int:
    """Run timing jobs across devices while incrementally writing CSV outputs."""

    devices = devices[: min(len(devices), len(jobs))]
    disable_progress = not sys.stderr.isatty()
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    event_queue = context.Queue()
    for job in jobs:
        job_queue.put(job)

    processes = [
        context.Process(target=timing_worker, args=(slot, device, settings, job_queue, event_queue))
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

    results: list[InferenceTimingResult] = []
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
                reset_task_bar(task_bars[slot], f"{event['device']} {event['label']}", total=event["total"])
            elif event_type == "progress":
                bar = task_bars[slot]
                total = event["total"]
                if total is not None and total != bar.total:
                    reset_task_bar(bar, bar.desc, total=max(int(total), 1))
                increment = int(event["increment"])
                if increment > 0:
                    bar.update(increment)
            elif event_type == "completed":
                result = event["result"]
                results.append(result)
                write_outputs(output, table_output, results, table_sizes)
                bar = task_bars[slot]
                if bar.total is not None and bar.n < bar.total:
                    bar.update(bar.total - bar.n)
                bar.set_postfix_str(f"{result.milliseconds_per_sample:.3f} ms/sample")
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


def write_outputs(
    output: Path,
    table_output: Path,
    results: list[InferenceTimingResult],
    table_sizes: list[int],
) -> None:
    """Write both long-form and n-column manuscript-table CSV outputs."""

    sorted_results = sorted(results, key=result_sort_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_FIELDNAMES)
        writer.writeheader()
        writer.writerows(result.to_csv_row() for result in sorted_results)

    table_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Panel", "Model", *[f"n={size}" for size in table_sizes]]
    values = {
        (result.panel, result.model, result.size): result.milliseconds_per_sample
        for result in sorted_results
    }
    with table_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel in PANEL_ORDER:
            for model in MODEL_ORDER_BY_PANEL[panel]:
                row = {"Panel": panel, "Model": model}
                for size in table_sizes:
                    value = values.get((panel, model, size))
                    row[f"n={size}"] = "" if value is None else f"{value:.3f}"
                writer.writerow(row)


def result_sort_key(result: InferenceTimingResult) -> tuple[int, int, int]:
    return (
        PANEL_ORDER.index(result.panel),
        MODEL_ORDER_BY_PANEL[result.panel].index(result.model),
        result.size,
    )


def run(args: argparse.Namespace) -> int:
    """Validate inputs, schedule timing jobs, and write results."""

    jobs = build_inference_timing_jobs(
        args.checkpoint_root,
        args.dataset,
        args.tasks,
        args.sizes,
        include_hybrid=args.include_hybrid,
    )
    if not jobs:
        raise ValueError("No inference timing jobs were requested.")

    devices = selected_devices(args)
    if args.dry_run:
        print(
            f"Would time {len(jobs)} model(s) across {len(devices)} device worker(s): "
            f"{', '.join(devices)}"
        )
        print(f"Checkpoint root: {args.checkpoint_root}")
        print(f"Long-form CSV: {args.output}")
        print(f"Table CSV: {args.table_output}")
        for job in jobs:
            print(f"  - {job.label}")
        return 0

    missing = missing_inference_inputs(args.data_root, args.dataset, args.split, jobs)
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise ValueError(f"Missing required inference inputs:\n{missing_list}")

    settings = InferenceTimingSettings(
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        warmup_passes=args.warmup_passes,
        timed_passes=args.timed_passes,
        max_batches=args.max_batches,
        preload_device_batches=args.preload_device_batches,
    )
    print(
        f"Timing {len(jobs)} model(s) across {len(devices)} device worker(s): "
        f"{', '.join(devices)}"
    )
    print(
        f"Each job runs {args.warmup_passes} warmup pass(es) and "
        f"{args.timed_passes} timed pass(es)."
    )
    if args.preload_device_batches:
        print("Inference inputs are preloaded onto the target device before timing.")

    status = run_timing_jobs(
        jobs,
        devices,
        settings,
        args.output,
        args.table_output,
        list(args.sizes),
    )
    if status == 0:
        print(f"Wrote long-form inference timing data to {args.output}")
        print(f"Wrote manuscript table data to {args.table_output}")
    return status


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
