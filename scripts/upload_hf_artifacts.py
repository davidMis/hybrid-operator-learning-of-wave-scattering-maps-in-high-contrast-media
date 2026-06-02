#!/usr/bin/env python3
# Overview:
# Build clean Hugging Face staging directories for the paper raw data and
# trained checkpoint sweep, then optionally upload those directories to Hub
# repos. The staging step hard-links, symlinks, or copies only reloadable
# artifacts so optimizer state, trainer checkpoints, logs, and caches are not
# published by accident.
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


DEFAULT_RAW_SOURCE = Path("data/raw/const_back")
DEFAULT_MODEL_SOURCE = Path("outputs/checkpoints/const_back/paper")
RAW_FILENAMES = (
    "velocity_sharp.npy",
    "velocity_smooth.npy",
    "pressure_sharp.npy",
    "pressure_smooth.npy",
)
FNO_REQUIRED_FILES = ("best_model_metadata.pkl", "best_model_state_dict.pt")
OPTIONAL_RUN_METADATA = ("run_manifest.json", "training_complete.json")
IGNORED_PATH_PARTS = {".cache", ".git", "__pycache__", "logs", "runs", "wandb"}
SCOT_SHARD_PATTERNS = (
    ("model.safetensors.index.json", "model-*-of-*.safetensors"),
    ("pytorch_model.bin.index.json", "pytorch_model-*-of-*.bin"),
)
SCOT_SINGLE_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
RECOMMENDED_MAX_FILE_SIZE = 200 * 1024**3
HARD_MAX_FILE_SIZE = 500 * 1024**3
PAPER_TITLE = "Hybrid operator learning of wave scattering maps in high-contrast media"
PAPER_URL = "https://arxiv.org/abs/2602.11197"


class ReleaseError(RuntimeError):
    """Raised for user-facing release preparation errors."""


@dataclass(frozen=True)
class StagedFile:
    """A staged file record with local audit data and public manifest metadata."""

    source: Path
    path: str
    size_bytes: int
    kind: str
    dtype: str | None = None
    shape: list[int] | None = None
    sha256: str | None = None

    def public_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "size": format_bytes(self.size_bytes),
        }
        if self.dtype is not None:
            data["dtype"] = self.dtype
        if self.shape is not None:
            data["shape"] = self.shape
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        return data


@dataclass(frozen=True)
class StageJob:
    """One Hub upload target prepared under a local staging directory."""

    label: str
    repo_id: str | None
    repo_type: str
    folder: Path
    files: list[StagedFile]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage and optionally upload the minimal raw data and trained checkpoint "
            "artifacts needed to reproduce the paper experiments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        choices=("all", "data", "models"),
        default="all",
        help="Which artifact group to stage and optionally upload.",
    )
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=DEFAULT_RAW_SOURCE,
        help="Directory containing the four raw const_back NumPy arrays.",
    )
    parser.add_argument(
        "--model-source",
        type=Path,
        default=DEFAULT_MODEL_SOURCE,
        help="Directory containing the trained paper checkpoint sweep.",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path(".hf_staging"),
        help=(
            "Local directory where clean Hub payloads are assembled before upload."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="const_back",
        help="Dataset name used in staged paths and generated Hub cards.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Hugging Face dataset repo id, for example owner/hybrid-helmholtz-const-back-raw.",
    )
    parser.add_argument(
        "--model-repo-id",
        default=None,
        help="Hugging Face model repo id, for example owner/hybrid-helmholtz-const-back-models.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=None,
        help="Create Hub repos as private. Omit when the release is ready to be public.",
    )
    parser.add_argument(
        "--license",
        default="mit",
        help="SPDX-style Hugging Face license id written into generated dataset and model cards.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help=(
            "How large source files are placed in staging. Hardlinks avoid copying "
            "but require source and staging directories to share a filesystem."
        ),
    )
    parser.add_argument(
        "--replace-staging",
        action="store_true",
        help="Remove existing dataset/models staging subdirectories before rebuilding them.",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute SHA-256 checksums for staged files. This can take a long time for raw arrays.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Create/update the Hub repos and upload staged contents. Without this flag, only stage and validate.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Worker count passed to huggingface_hub.upload_large_folder during upload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        staging_root = args.staging_root.expanduser().resolve()
        jobs: list[StageJob] = []

        if args.artifact in {"all", "data"}:
            jobs.append(stage_raw_data(args, staging_root / "dataset"))
        if args.artifact in {"all", "models"}:
            jobs.append(stage_models(args, staging_root / "models"))

        for job in jobs:
            total = sum(record.size_bytes for record in job.files)
            print(f"Staged {job.label}: {len(job.files)} files, {format_bytes(total)} at {job.folder}")

        if args.upload:
            upload_jobs(jobs, args.private, args.num_workers)
        else:
            print("Staging complete. Re-run with --upload after inspecting the generated payloads.")
        return 0
    except ReleaseError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except ModuleNotFoundError as error:
        print(
            f"ERROR: Missing Python dependency '{error.name}'. Activate .venv and run "
            "`python -m pip install -e .` before using this release utility.",
            file=sys.stderr,
        )
        return 1


def validate_args(args: argparse.Namespace) -> None:
    if args.upload:
        if args.artifact in {"all", "data"} and not args.dataset_repo_id:
            raise ReleaseError("--dataset-repo-id is required when uploading data artifacts.")
        if args.artifact in {"all", "models"} and not args.model_repo_id:
            raise ReleaseError("--model-repo-id is required when uploading model artifacts.")
    for repo_id, label in ((args.dataset_repo_id, "dataset"), (args.model_repo_id, "model")):
        if repo_id is not None and not valid_repo_id(repo_id):
            raise ReleaseError(
                f"Invalid {label} repo id '{repo_id}'. Use the form owner/repository-name."
            )
    if args.num_workers < 1:
        raise ReleaseError("--num-workers must be at least 1.")


def valid_repo_id(repo_id: str) -> bool:
    return re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$", repo_id) is not None


def stage_raw_data(args: argparse.Namespace, stage_dir: Path) -> StageJob:
    source_dir = args.raw_source.expanduser()
    if not source_dir.is_dir():
        raise ReleaseError(f"Raw data source does not exist or is not a directory: {source_dir}")
    prepare_stage_dir(stage_dir, args.replace_staging)

    records: list[StagedFile] = []
    staged_paths: set[str] = set()
    data_dir = stage_dir / args.dataset
    for filename in RAW_FILENAMES:
        source = source_dir / filename
        if not source.is_file():
            raise ReleaseError(f"Missing required raw array: {source}")
        dtype, shape = read_npy_metadata(source)
        record = stage_file(
            source=source,
            destination=data_dir / filename,
            stage_dir=stage_dir,
            kind="raw_numpy_array",
            link_mode=args.link_mode,
            staged_paths=staged_paths,
            compute_hash=args.hash,
            dtype=dtype,
            shape=shape,
        )
        records.append(record)

    write_dataset_card(stage_dir, args, records)
    write_manifest(
        stage_dir / "hf_release_manifest.json",
        artifact_type="raw_data",
        dataset=args.dataset,
        records=records,
        extra={
            "layout": f"{args.dataset}/<raw-array>.npy",
            "selection_policy": "Only the four raw arrays consumed by scripts/prepare_data.py are staged.",
        },
    )
    return StageJob(
        label="raw data",
        repo_id=args.dataset_repo_id,
        repo_type="dataset",
        folder=stage_dir,
        files=records,
    )


def stage_models(args: argparse.Namespace, stage_dir: Path) -> StageJob:
    source_dir = args.model_source.expanduser()
    if not source_dir.is_dir():
        raise ReleaseError(f"Model source does not exist or is not a directory: {source_dir}")
    prepare_stage_dir(stage_dir, args.replace_staging)

    records: list[StagedFile] = []
    staged_paths: set[str] = set()
    run_summaries: list[dict[str, Any]] = []

    for run_dir in discover_fno_runs(source_dir):
        run_files = [run_dir / name for name in FNO_REQUIRED_FILES]
        run_files.extend(run_dir / name for name in OPTIONAL_RUN_METADATA if (run_dir / name).is_file())
        staged = stage_model_file_group(
            files=run_files,
            source_root=source_dir,
            stage_dir=stage_dir,
            kind="fno_checkpoint",
            link_mode=args.link_mode,
            staged_paths=staged_paths,
            compute_hash=args.hash,
        )
        records.extend(staged)
        run_summaries.append(
            {
                "model_type": "fno",
                "path": run_dir.relative_to(source_dir).as_posix(),
                "files": [record.path for record in staged],
            }
        )

    for model_dir in discover_scot_model_dirs(source_dir):
        weight_files = scot_weight_files(model_dir)
        run_dir = model_dir.parent if is_checkpoint_dir(model_dir) else model_dir
        run_files = [model_dir / "config.json", *weight_files]
        run_files.extend(run_dir / name for name in OPTIONAL_RUN_METADATA if (run_dir / name).is_file())
        staged = stage_model_file_group(
            files=run_files,
            source_root=source_dir,
            stage_dir=stage_dir,
            kind="scot_checkpoint",
            link_mode=args.link_mode,
            staged_paths=staged_paths,
            compute_hash=args.hash,
        )
        records.extend(staged)
        run_summaries.append(
            {
                "model_type": "scot",
                "path": run_dir.relative_to(source_dir).as_posix(),
                "loader_path": model_dir.relative_to(source_dir).as_posix(),
                "files": [record.path for record in staged],
            }
        )

    if not records:
        raise ReleaseError(
            f"No reloadable FNO or scOT checkpoint artifacts were found under {source_dir}."
        )

    write_model_card(stage_dir, args, records, run_summaries)
    write_manifest(
        stage_dir / "hf_release_manifest.json",
        artifact_type="trained_models",
        dataset=args.dataset,
        records=records,
        extra={
            "layout": "Matches the local checkpoint root expected by helmholtz_hybrid.evaluation.",
            "selection_policy": (
                "FNO runs include best_model_metadata.pkl and best_model_state_dict.pt. "
                "scOT runs include config.json and the selected model weights. "
                "run_manifest.json and training_complete.json are included when present."
            ),
            "excluded_artifacts": [
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
                "training_args.bin",
                "checkpoint optimizer state",
                "logs",
                "wandb",
                "cache directories",
            ],
            "runs": run_summaries,
        },
    )
    return StageJob(
        label="trained models",
        repo_id=args.model_repo_id,
        repo_type="model",
        folder=stage_dir,
        files=records,
    )


def prepare_stage_dir(stage_dir: Path, replace: bool) -> None:
    if stage_dir.exists():
        if not replace:
            raise ReleaseError(
                f"Staging directory already exists: {stage_dir}. "
                "Inspect it, remove it manually, or pass --replace-staging."
            )
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=False)


def stage_model_file_group(
    *,
    files: list[Path],
    source_root: Path,
    stage_dir: Path,
    kind: str,
    link_mode: str,
    staged_paths: set[str],
    compute_hash: bool,
) -> list[StagedFile]:
    records: list[StagedFile] = []
    for source in files:
        if not source.is_file():
            raise ReleaseError(f"Expected checkpoint file does not exist: {source}")
        destination = stage_dir / source.relative_to(source_root)
        record = stage_file(
            source=source,
            destination=destination,
            stage_dir=stage_dir,
            kind=kind,
            link_mode=link_mode,
            staged_paths=staged_paths,
            compute_hash=compute_hash,
        )
        if record is not None:
            records.append(record)
    return records


def stage_file(
    *,
    source: Path,
    destination: Path,
    stage_dir: Path,
    kind: str,
    link_mode: str,
    staged_paths: set[str],
    compute_hash: bool,
    dtype: str | None = None,
    shape: list[int] | None = None,
) -> StagedFile | None:
    relative_path = destination.relative_to(stage_dir).as_posix()
    if relative_path in staged_paths:
        return None
    staged_paths.add(relative_path)
    size = source.stat().st_size
    validate_file_size(source, size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    place_file(source, destination, link_mode, size)
    file_hash = sha256_file(source) if compute_hash else None
    return StagedFile(
        source=source,
        path=relative_path,
        size_bytes=size,
        kind=kind,
        dtype=dtype,
        shape=shape,
        sha256=file_hash,
    )


def place_file(source: Path, destination: Path, link_mode: str, size: int) -> None:
    if destination.exists() or destination.is_symlink():
        raise ReleaseError(f"Refusing to overwrite staged file: {destination}")
    if link_mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            raise ReleaseError(
                f"Could not hardlink {source} to {destination}: {error}. "
                "Use --staging-root on the same filesystem, or choose --link-mode symlink or copy."
            ) from error
    elif link_mode == "symlink":
        destination.symlink_to(source.resolve())
    elif link_mode == "copy":
        copy_with_progress(source, destination, size)
    else:
        raise ReleaseError(f"Unsupported link mode: {link_mode}")


def copy_with_progress(source: Path, destination: Path, size: int) -> None:
    with source.open("rb") as read_handle, destination.open("wb") as write_handle:
        with tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            desc=f"copy {source.name}",
            dynamic_ncols=True,
        ) as progress:
            for chunk in iter(lambda: read_handle.read(1024 * 1024), b""):
                write_handle.write(chunk)
                progress.update(len(chunk))
    shutil.copystat(source, destination)


def validate_file_size(source: Path, size: int) -> None:
    if size > HARD_MAX_FILE_SIZE:
        raise ReleaseError(
            f"{source} is {format_bytes(size)}, which exceeds Hugging Face's 500 GB single-file hard limit."
        )
    if size > RECOMMENDED_MAX_FILE_SIZE:
        print(
            f"WARNING: {source} is {format_bytes(size)}. Hugging Face recommends keeping "
            "individual files below 200 GB when possible.",
            file=sys.stderr,
        )


def discover_fno_runs(source_root: Path) -> list[Path]:
    runs: list[Path] = []
    for metadata in source_root.rglob("best_model_metadata.pkl"):
        run_dir = metadata.parent
        if is_ignored_path(run_dir) or not (run_dir / "best_model_state_dict.pt").is_file():
            continue
        runs.append(run_dir)
    return sorted(set(runs))


def discover_scot_model_dirs(source_root: Path) -> list[Path]:
    candidates = {
        config.parent
        for config in source_root.rglob("config.json")
        if not is_ignored_path(config.parent) and scot_has_weights(config.parent)
    }
    root_candidates = {candidate for candidate in candidates if not is_checkpoint_dir(candidate)}
    checkpoint_candidates: dict[Path, list[Path]] = {}
    for candidate in candidates:
        if not is_checkpoint_dir(candidate):
            continue
        run_dir = candidate.parent
        if run_dir in root_candidates:
            continue
        checkpoint_candidates.setdefault(run_dir, []).append(candidate)

    selected = list(root_candidates)
    for checkpoints in checkpoint_candidates.values():
        selected.append(max(checkpoints, key=checkpoint_number))
    return sorted(selected)


def scot_has_weights(model_dir: Path) -> bool:
    if any((model_dir / name).is_file() for name in SCOT_SINGLE_WEIGHTS):
        return True
    return any((model_dir / index_name).is_file() for index_name, _ in SCOT_SHARD_PATTERNS)


def scot_weight_files(model_dir: Path) -> list[Path]:
    for index_name, shard_pattern in SCOT_SHARD_PATTERNS:
        index_file = model_dir / index_name
        if index_file.is_file():
            shards = sorted(model_dir.glob(shard_pattern))
            if not shards:
                raise ReleaseError(f"{index_file} exists but no matching shards were found.")
            return [index_file, *shards]
    for name in SCOT_SINGLE_WEIGHTS:
        weight_file = model_dir / name
        if weight_file.is_file():
            return [weight_file]
    raise ReleaseError(f"No scOT model weights found in {model_dir}")


def is_checkpoint_dir(path: Path) -> bool:
    return re.match(r"^checkpoint-\d+$", path.name) is not None


def checkpoint_number(path: Path) -> int:
    if not is_checkpoint_dir(path):
        return -1
    return int(path.name.split("-", 1)[1])


def is_ignored_path(path: Path) -> bool:
    return any(part in IGNORED_PATH_PARTS for part in path.parts)


def read_npy_metadata(path: Path) -> tuple[str, list[int]]:
    import numpy as np

    try:
        array = np.load(path, mmap_mode="r")
    except Exception as error:
        raise ReleaseError(f"Could not read NumPy metadata from {path}: {error}") from error
    return str(array.dtype), [int(dim) for dim in array.shape]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        with tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            desc=f"hash {path.name}",
            dynamic_ncols=True,
        ) as progress:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                progress.update(len(chunk))
    return digest.hexdigest()


def write_dataset_card(stage_dir: Path, args: argparse.Namespace, records: list[StagedFile]) -> None:
    repo_id = args.dataset_repo_id or "<dataset-repo-id>"
    rows = "\n".join(
        f"| `{record.path}` | `{record.dtype}` | `{record.shape}` | {format_bytes(record.size_bytes)} |"
        for record in records
    )
    text = f"""---
license: {args.license}
pretty_name: Hybrid Helmholtz {args.dataset} raw data
tags:
- geophysics
- helmholtz
- wave-scattering
- operator-learning
---

# Hybrid Helmholtz {args.dataset} Raw Data

This dataset repository contains the raw NumPy arrays used by the paper
[{PAPER_TITLE}]({PAPER_URL}).

The repository is intentionally minimal. It contains only the four raw arrays
consumed by `scripts/prepare_data.py`; processed splits can be regenerated from
these files.

## Files

| Path | Dtype | Shape | Size |
| --- | --- | --- | --- |
{rows}

## Usage

```bash
hf download {repo_id} --repo-type dataset --local-dir data/raw
python scripts/prepare_data.py --raw-root data/raw --output-root data/processed --dataset {args.dataset}
```
"""
    (stage_dir / "README.md").write_text(text, encoding="utf-8")


def write_model_card(
    stage_dir: Path,
    args: argparse.Namespace,
    records: list[StagedFile],
    run_summaries: list[dict[str, Any]],
) -> None:
    repo_id = args.model_repo_id or "<model-repo-id>"
    dataset_repo = args.dataset_repo_id or "<dataset-repo-id>"
    n_fno = sum(1 for item in run_summaries if item["model_type"] == "fno")
    n_scot = sum(1 for item in run_summaries if item["model_type"] == "scot")
    total_size = format_bytes(sum(record.size_bytes for record in records))
    text = f"""---
license: {args.license}
library_name: pytorch
datasets:
- {dataset_repo}
tags:
- geophysics
- helmholtz
- neural-operator
- operator-learning
- fno
- transformer
---

# Hybrid Helmholtz {args.dataset} Paper Checkpoints

This model repository contains the trained checkpoint artifacts used by the paper
[{PAPER_TITLE}]({PAPER_URL}).

The release is filtered for inference and paper reproduction. FNO directories
include `best_model_metadata.pkl` and `best_model_state_dict.pt`; scOT directories
include `config.json` and model weights. Optimizer state, scheduler state,
trainer state, intermediate checkpoint state, logs, W&B files, and caches are not
included.

## Contents

- FNO runs: {n_fno}
- scOT runs: {n_scot}
- Staged files: {len(records)}
- Total payload size: {total_size}

## Usage

```bash
hf download {repo_id} --repo-type model --local-dir outputs/checkpoints/{args.dataset}/paper
```

The downloaded directory preserves the checkpoint-root layout expected by
`helmholtz_hybrid.evaluation` and `run_all.sh`.
"""
    (stage_dir / "README.md").write_text(text, encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    artifact_type: str,
    dataset: str,
    records: list[StagedFile],
    extra: dict[str, Any],
) -> None:
    manifest = {
        "artifact_type": artifact_type,
        "dataset": dataset,
        "paper": {
            "title": PAPER_TITLE,
            "url": PAPER_URL,
        },
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": [record.public_dict() for record in sorted(records, key=lambda item: item.path)],
        **extra,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def upload_jobs(jobs: list[StageJob], private: bool, num_workers: int) -> None:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as error:
        raise ReleaseError(
            "huggingface_hub is required for uploads. Install the project environment with "
            "`python -m pip install -e .`."
        ) from error

    api = HfApi()
    for job in jobs:
        if job.repo_id is None:
            raise ReleaseError(f"Missing repo id for {job.label}.")
        print(f"Creating or updating {job.repo_type} repo: {job.repo_id}")
        api.create_repo(
            repo_id=job.repo_id,
            repo_type=job.repo_type,
            private=private,
            exist_ok=True,
        )
        api.upload_large_folder(
            repo_id=job.repo_id,
            repo_type=job.repo_type,
            folder_path=job.folder,
            num_workers=num_workers,
            ignore_patterns=[".cache/huggingface/**"],
        )


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


if __name__ == "__main__":
    raise SystemExit(main())
