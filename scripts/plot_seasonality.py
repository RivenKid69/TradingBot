#!/usr/bin/env python3
"""Plot liquidity and latency seasonality multipliers.

This helper reads a `liquidity_latency_seasonality.json` file and outputs
line charts and heatmaps for quick inspection. Generated PNGs are stored in
`reports/seasonality/plots` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HOURS_IN_WEEK = 168


def _load_array(data: dict, key: str) -> np.ndarray:
    arr = np.asarray(data.get(key, [1.0] * HOURS_IN_WEEK), dtype=float)
    if arr.size != HOURS_IN_WEEK:
        raise ValueError(f"{key} array must contain {HOURS_IN_WEEK} values")
    return arr


def _plot_line(arr: np.ndarray, key: str, out_dir: Path) -> None:
    plt.figure(figsize=(10, 3))
    plt.plot(arr)
    plt.title(f"{key.capitalize()} multipliers")
    plt.xlabel("Hour of week")
    plt.ylabel("Multiplier")
    plt.tight_layout()
    plt.savefig(out_dir / f"{key}_line.png")
    plt.close()


def _plot_heatmap(arr: np.ndarray, key: str, out_dir: Path) -> None:
    data = arr.reshape(7, 24)
    plt.figure(figsize=(8, 3))
    im = plt.imshow(data, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="Multiplier")
    plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.xlabel("Hour of day")
    plt.title(f"{key.capitalize()} multipliers")
    plt.tight_layout()
    plt.savefig(out_dir / f"{key}_heatmap.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot seasonality multipliers")
    parser.add_argument(
        "--multipliers",
        default="data/latency/liquidity_latency_seasonality.json",
        help="Path to liquidity_latency_seasonality.json",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/seasonality/plots",
        help="Directory to store generated plots",
    )
    args = parser.parse_args()

    with open(args.multipliers, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in ("liquidity", "latency"):
        arr = _load_array(data, key)
        _plot_line(arr, key, out_dir)
        _plot_heatmap(arr, key, out_dir)


if __name__ == "__main__":
    main()
