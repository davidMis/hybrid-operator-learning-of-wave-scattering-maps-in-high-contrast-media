# Overview:
# Load flat YAML configuration files into argparse parsers. Training scripts use
# this helper so fixed paper hyperparameters can live in config files while CLI
# arguments remain available for dataset/task/run-specific overrides.
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml


def config_path_from_argv(argv: Sequence[str] | None = None) -> Path | None:
    """Return --config from argv without consuming the full training CLI."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return args.config


def apply_yaml_defaults(parser: argparse.ArgumentParser, config_path: Path | None) -> None:
    """Set parser defaults from a flat YAML mapping after validating keys."""

    if config_path is None:
        return
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"Configuration file does not exist: {config_path}")
        return
    except yaml.YAMLError as error:
        parser.error(f"Could not parse YAML configuration {config_path}: {error}")
        return

    if raw_config is None:
        return
    if not isinstance(raw_config, dict):
        parser.error(f"Configuration file must contain a YAML mapping: {config_path}")

    valid_keys = {action.dest for action in parser._actions if action.dest != "help"}
    unknown_keys = sorted(str(key) for key in raw_config if str(key) not in valid_keys)
    if unknown_keys:
        parser.error(f"Unknown configuration keys in {config_path}: {', '.join(unknown_keys)}")

    parser.set_defaults(**{str(key): value for key, value in raw_config.items()})
