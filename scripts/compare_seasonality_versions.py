#!/usr/bin/env python3
"""Compare archived seasonality JSON files.

The script walks the archive directory, sorts files by name and prints
maximum absolute differences for liquidity, spread and latency arrays
between consecutive versions.
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("array lengths differ")
    return float(np.max(np.abs(arr_a - arr_b)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare archived seasonality JSON files"
    )
    parser.add_argument(
        "--archive-dir",
        default="configs/seasonality/archive",
        help="Directory containing archived JSON files",
    )
    args = parser.parse_args()

    archive_path = Path(args.archive_dir)
    files = sorted(archive_path.glob("*.json"))
    if len(files) < 2:
        print("Need at least two archived JSON files to compare")
        return

    prev_data = None
    prev_name = None
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        if prev_data is not None:
            print(f"=== {prev_name} -> {file.name} ===")
            for key in ("liquidity", "spread", "latency"):
                if key in prev_data and key in data:
                    diff = _max_abs_diff(prev_data[key], data[key])
                    print(f"{key}: max abs diff {diff:.6f}")
        prev_data = data
        prev_name = file.name


if __name__ == "__main__":
    main()

