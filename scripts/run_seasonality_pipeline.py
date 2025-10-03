#!/usr/bin/env python3
"""End-to-end helper for building seasonality artifacts and launching training.

The script performs the following workflow:

1. Ensure a historical trade/latency dataset is present.
2. Run :mod:`scripts.build_hourly_seasonality` to compute liquidity/latency
   multipliers.
3. (Optional) Plot or validate the produced multipliers.
4. Launch :mod:`train_model_multi_patch` with the generated JSON.

It is designed to mirror the manual shell commands that are typically executed
when preparing the seasonality bundle.  All sub-steps can be toggled on/off via
command-line flags, and additional arguments may be forwarded to the underlying
CLIs.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

DEFAULT_DATA_PATH = Path("data/seasonality_source/latest.parquet")
DEFAULT_SEASONALITY_OUT = Path("data/latency/liquidity_latency_seasonality.json")
DEFAULT_PLOTS_DIR = Path("reports/seasonality/plots")


@dataclass
class PipelineOptions:
    """Configuration for the orchestrated workflow."""

    data: Path
    seasonality_out: Path
    run_plot: bool
    plots_dir: Path
    run_validate: bool
    validation_threshold: float
    run_train: bool
    train_config: str
    regime_config: str
    dataset_split: str | None
    symbol: str | None
    build_args: Sequence[str]
    validate_args: Sequence[str]
    train_args: Sequence[str]


def _expand_existing(path: Path) -> Path:
    """Return *path* with ``~`` expanded and ensure the file exists."""

    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Required dataset not found: {resolved}")
    return resolved


def _ensure_parent(path: Path) -> None:
    """Create parent directories for *path* if they do not exist."""

    path.expanduser().parent.mkdir(parents=True, exist_ok=True)


def _parse_extra_args(raw: str | None) -> List[str]:
    if not raw:
        return []
    return shlex.split(raw)


def _command(title: str, parts: Iterable[str]) -> Tuple[str, List[str]]:
    return title, list(parts)


def build_pipeline_steps(options: PipelineOptions) -> List[Tuple[str, List[str]]]:
    """Assemble subprocess commands for the requested pipeline steps."""

    steps: List[Tuple[str, List[str]]] = []
    build_cmd = [
        sys.executable,
        "scripts/build_hourly_seasonality.py",
        "--data",
        str(options.data),
        "--out",
        str(options.seasonality_out),
    ]
    if options.symbol:
        build_cmd.extend(["--symbol", options.symbol])
    build_cmd.extend(options.build_args)
    steps.append(_command("build seasonality", build_cmd))

    if options.run_plot:
        plot_cmd = [
            sys.executable,
            "scripts/plot_seasonality.py",
            "--multipliers",
            str(options.seasonality_out),
            "--out-dir",
            str(options.plots_dir),
        ]
        steps.append(_command("plot seasonality", plot_cmd))

    if options.run_validate:
        validate_cmd = [
            sys.executable,
            "scripts/validate_seasonality.py",
            "--historical",
            str(options.data),
            "--multipliers",
            str(options.seasonality_out),
            "--threshold",
            str(options.validation_threshold),
        ]
        if options.symbol:
            validate_cmd.extend(["--symbol", options.symbol])
        validate_cmd.extend(options.validate_args)
        steps.append(_command("validate seasonality", validate_cmd))

    if options.run_train:
        train_cmd = [
            sys.executable,
            "train_model_multi_patch.py",
            "--config",
            options.train_config,
            "--regime-config",
            options.regime_config,
            "--liquidity-seasonality",
            str(options.seasonality_out),
        ]
        if options.dataset_split:
            train_cmd.extend(["--dataset-split", options.dataset_split])
        train_cmd.extend(options.train_args)
        steps.append(_command("train model", train_cmd))

    return steps


def run_pipeline(options: PipelineOptions) -> None:
    """Execute the configured pipeline in sequence."""

    _expand_existing(options.data)
    _ensure_parent(options.seasonality_out)
    if options.run_plot:
        options.plots_dir.expanduser().mkdir(parents=True, exist_ok=True)

    steps = build_pipeline_steps(options)
    for title, cmd in steps:
        pretty = " ".join(shlex.quote(part) for part in cmd)
        print(f"==> {title}: {pretty}")
        subprocess.run(cmd, check=True)


def parse_args(argv: Sequence[str] | None = None) -> PipelineOptions:
    parser = argparse.ArgumentParser(
        description="Build liquidity/latency seasonality and run training."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Historical trades/latency dataset (CSV or Parquet).",
    )
    parser.add_argument(
        "--seasonality-out",
        default=str(DEFAULT_SEASONALITY_OUT),
        help="Destination for liquidity_latency_seasonality.json.",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Optional instrument symbol to forward to the sub-commands.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate QA plots via scripts/plot_seasonality.py.",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(DEFAULT_PLOTS_DIR),
        help="Directory for generated plots (when --plot is enabled).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run scripts/validate_seasonality.py after building multipliers.",
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.1,
        help="Maximum relative difference used by the validation step.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip launching train_model_multi_patch.py.",
    )
    parser.add_argument(
        "--train-config",
        default="configs/config_train.yaml",
        help="Training configuration YAML file.",
    )
    parser.add_argument(
        "--regime-config",
        default="configs/market_regimes.json",
        help="Market regime configuration JSON file.",
    )
    parser.add_argument(
        "--dataset-split",
        default="none",
        help="Dataset split to use for training (pass 'none' to disable).",
    )
    parser.add_argument(
        "--build-args",
        default=None,
        help="Extra arguments forwarded to build_hourly_seasonality.",
    )
    parser.add_argument(
        "--validate-args",
        default=None,
        help="Extra arguments forwarded to validate_seasonality.",
    )
    parser.add_argument(
        "--train-args",
        default=None,
        help="Extra arguments forwarded to train_model_multi_patch.py.",
    )

    args = parser.parse_args(argv)

    dataset_split = args.dataset_split
    if dataset_split == "none":
        dataset_split = "none"
    elif dataset_split in {"", "null", "None"}:
        dataset_split = None

    return PipelineOptions(
        data=Path(args.data),
        seasonality_out=Path(args.seasonality_out),
        run_plot=args.plot,
        plots_dir=Path(args.plots_dir),
        run_validate=args.validate,
        validation_threshold=args.validation_threshold,
        run_train=not args.skip_train,
        train_config=args.train_config,
        regime_config=args.regime_config,
        dataset_split=dataset_split,
        symbol=args.symbol,
        build_args=_parse_extra_args(args.build_args),
        validate_args=_parse_extra_args(args.validate_args),
        train_args=_parse_extra_args(args.train_args),
    )


def main(argv: Sequence[str] | None = None) -> None:
    options = parse_args(argv)
    run_pipeline(options)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
