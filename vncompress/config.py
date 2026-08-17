"""
Unified experiment configuration for vncompress scripts.
=========================================================
Resolves P1 issue "repro: tao experiment configuration thong nhat": the
repo previously had one independent argparse surface per script (
run_benchmark.py, run_ablation.py, run_training.py, ...), each with its
own defaults, no seed control, and no record of what was actually run.
That makes results hard to compare or reproduce.

This module provides:
  - ExperimentConfig: a single dataclass covering model/tokenizer, seed,
    data/split, compression ratios, generation/eval settings, and
    hardware (device/dtype) -- the fields every script needs.
  - load_experiment_config(): loads a JSON (or YAML, if PyYAML is
    installed) config file and merges CLI overrides on top of it.
    Precedence (lowest to highest): dataclass defaults < config file <
    explicit CLI flags. A CLI flag counts as "explicit" only if the
    caller passed a non-None override (see merge_cli_overrides).
  - set_seed(): seeds python `random`, numpy, and torch (CPU + all CUDA
    devices) so stochastic components (RandomCompressor, class-budget
    sampling in NoModelMorphCompressor, model sampling, ...) are
    reproducible run-to-run.
  - snapshot_environment(): captures the git commit and key package
    versions so a result directory can be traced back to the exact code
    and dependency versions that produced it.
  - save_run_metadata(): writes config.json + environment.json into a
    run's output directory -- the "snapshot config alongside results"
    requirement from the same issue.

Not YAML-only by design: PyYAML isn't a declared dependency of this
project (see vncompress/requirements.txt), so JSON is the format that
always works; YAML is accepted opportunistically if PyYAML happens to
be installed, so `.yaml`/`.yml` config files also work in that case.
"""
import dataclasses
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Single source of truth for the settings every evaluation/training
    script needs. Individual scripts may have additional CLI-only flags
    (e.g. --output-dir naming), but anything that affects *what is being
    measured* (model, seed, data, ratios, generation params) belongs
    here so runs can be compared and reproduced."""

    # Reproducibility
    seed: int = 42

    # Model / tokenizer
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer: Optional[str] = None  # defaults to `model` if unset

    # Hardware
    device: str = "cuda"          # 'cuda' | 'cpu' | 'mps'
    dtype: str = "float16"        # 'float16' | 'bfloat16' | 'float32'

    # Data
    data_path: Optional[str] = None   # defaults to vcc_bench_data/vcc_bench_v1.json
    split: str = "full"               # VCC-Bench v1 is a single fixed eval split
                                       # (see docs/benchmark.md) -- 'full' is the
                                       # only value scripts currently support;
                                       # kept as a field so a future train/val/test
                                       # split doesn't require a schema change.

    # Compression
    methods: List[str] = field(default_factory=lambda: [
        "none", "random", "llmlingua", "tone_aware", "morphology_aware", "combined",
    ])
    compression_ratios: List[float] = field(default_factory=lambda: [2.0, 4.0, 8.0])

    # Generation / evaluation
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False

    # Output
    output_dir: str = "./results"

    def resolved_tokenizer(self) -> str:
        return self.tokenizer or self.model

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_CONFIG_FIELDS = {f.name for f in dataclasses.fields(ExperimentConfig)}


def _read_config_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    f"Config file {path} is YAML but PyYAML is not installed. "
                    "Install pyyaml, or use a .json config instead."
                ) from e
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)
    unknown = set(data) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown ExperimentConfig field(s) in {path}: {sorted(unknown)}. "
            f"Valid fields: {sorted(_CONFIG_FIELDS)}"
        )
    return data


def load_experiment_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Build an ExperimentConfig from (in increasing precedence order):
    dataclass defaults -> config_path file (if given) -> cli_overrides
    (only keys with a non-None value override anything before them, so
    argparse defaults of None mean "no explicit CLI choice").
    """
    values: Dict[str, Any] = {}
    if config_path:
        values.update(_read_config_file(config_path))
    if cli_overrides:
        for key, val in cli_overrides.items():
            if val is not None:
                if key not in _CONFIG_FIELDS:
                    raise ValueError(f"Unknown ExperimentConfig field in CLI overrides: {key!r}")
                values[key] = val
    return ExperimentConfig(**values)


def set_seed(seed: int) -> None:
    """Seed every stochastic component vncompress scripts touch."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


_TRACKED_PACKAGES = [
    "torch", "transformers", "accelerate", "peft", "numpy", "scipy",
    "datasets", "evaluate", "sentencepiece", "rouge_score", "sacrebleu",
    "bert-score", "underthesea",
]


def _package_versions() -> Dict[str, Optional[str]]:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - py<3.8
        import importlib_metadata as metadata  # type: ignore
    versions: Dict[str, Optional[str]] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def snapshot_environment() -> Dict[str, Any]:
    """Capture what code + dependencies produced a run's results."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_version": sys.version,
        "package_versions": _package_versions(),
    }


def save_run_metadata(output_dir: str, config: ExperimentConfig) -> None:
    """Write config.json + environment.json into `output_dir`, so any
    result directory can be traced back to exactly what produced it."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot_environment(), f, indent=2, ensure_ascii=False)
