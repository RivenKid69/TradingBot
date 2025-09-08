#!/usr/bin/env python3
"""Aggregate metrics from multiple runs.

This script loads ``metrics.json`` files and aggregates key
performance indicators into a table. Each path supplied on the command
line may point directly to a ``metrics.json`` file or to a directory
containing one. By default the table is written to ``compare_runs.csv``
in the current working directory, but the ``--stdout`` flag allows
printing the table to standard output instead.
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, Tuple

# Known aliases for canonical metric names.
KEY_ALIASES = {
    "sharpe": ["sharpe", "sharpe_ratio"],
    "sortino": ["sortino", "sortino_ratio"],
    "mdd": ["max_drawdown", "mdd"],
    "pnl": ["pnl", "pnl_total", "profit"],
    "hit_rate": ["hit_rate", "hitratio", "hit_ratio"],
    "cvar": ["cvar", "conditional_value_at_risk"],
}


def _find_metrics_file(path: str) -> str | None:
    """Return path to metrics.json for the given input path."""
    if os.path.isdir(path):
        candidate = os.path.join(path, "metrics.json")
        if os.path.isfile(candidate):
            return candidate
        for root, _, files in os.walk(path):
            if "metrics.json" in files:
                return os.path.join(root, "metrics.json")
        return None
    if os.path.isfile(path):
        return path
    return None


def _flatten(prefix: str, obj, out: Dict[str, float]) -> None:
    """Recursively flattens dictionaries of metrics."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}{key}.", value, out)
    else:
        out[prefix[:-1]] = obj


def _load_metrics(path: str) -> Tuple[str, Dict[str, float]]:
    """Load metrics from the given run path."""
    metrics_file = _find_metrics_file(path)
    if not metrics_file:
        raise FileNotFoundError(f"metrics.json not found for {path}")
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat: Dict[str, float] = {}
    _flatten("", data, flat)

    result: Dict[str, float] = {}
    remaining = dict(flat)
    for canon, aliases in KEY_ALIASES.items():
        for alias in aliases:
            if alias in remaining:
                result[canon] = remaining.pop(alias)
                break
    result.update(remaining)

    run_id = os.path.basename(os.path.dirname(metrics_file))
    return run_id, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare metrics from multiple runs.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to metrics.json files or run directories.",
    )
    parser.add_argument(
        "--csv",
        default="compare_runs.csv",
        help="Path to save CSV output (default: compare_runs.csv).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print table to stdout instead of saving CSV.",
    )
    args = parser.parse_args()

    rows = []
    all_keys = set()
    for path in args.paths:
        try:
            run_id, metrics = _load_metrics(path)
        except FileNotFoundError as exc:  # pragma: no cover - user feedback
            print(exc, file=sys.stderr)
            continue
        rows.append({"run_id": run_id, **metrics})
        all_keys.update(metrics.keys())

    preferred = ["sharpe", "sortino", "mdd", "pnl", "hit_rate", "cvar"]
    ordered_keys = [k for k in preferred if k in all_keys]
    ordered_keys += sorted(k for k in all_keys if k not in ordered_keys)
    headers = ["run_id"] + ordered_keys

    def write_rows(writer: csv.writer) -> None:
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(h, "") for h in headers])

    if args.stdout:
        write_rows(csv.writer(sys.stdout))
    else:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            write_rows(csv.writer(f))


if __name__ == "__main__":
    main()
