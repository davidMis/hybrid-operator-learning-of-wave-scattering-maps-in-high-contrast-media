# Overview:
# Configure local cache directories before importing libraries that otherwise
# write into user-level cache locations. Scripts call set_default_cache_dirs()
# before matplotlib, HuggingFace, neuralop, or scOT imports.
from __future__ import annotations

import os
from pathlib import Path


def set_default_cache_dirs(root: str | Path = "outputs/cache") -> None:
    cache_root = Path(root)
    huggingface_root = cache_root / "huggingface"
    transformers_root = huggingface_root / "transformers"
    matplotlib_root = cache_root / "matplotlib"
    huggingface_root.mkdir(parents=True, exist_ok=True)
    transformers_root.mkdir(parents=True, exist_ok=True)
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(huggingface_root.resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_root.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_root.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))
