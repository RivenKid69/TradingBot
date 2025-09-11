#!/usr/bin/env python3
"""Plot hour-of-week multiplier metric coverage.

Reads Prometheus metrics either from a local file or an HTTP endpoint and
produces a simple chart visualising how often each hour-of-week multiplier
was used in the simulator and latency model.
"""
from __future__ import annotations

import argparse
import urllib.request
from typing import List

import matplotlib.pyplot as plt
from prometheus_client.parser import text_string_to_metric_families


def _fetch(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as resp:  # nosec - optional usage
            return resp.read().decode("utf-8")
    with open(source, "r", encoding="utf-8") as fh:
        return fh.read()


def _extract(metrics_text: str, name: str) -> List[float]:
    values = [0.0] * 168
    for fam in text_string_to_metric_families(metrics_text):
        if fam.name != name:
            continue
        for sample in fam.samples:
            try:
                hour = int(sample.labels.get("hour", 0))
                if 0 <= hour < len(values):
                    values[hour] = float(sample.value)
            except Exception:
                continue
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="Metrics URL or file path")
    ap.add_argument("--out", default="hourly_multiplier_coverage.png", help="Output image path")
    args = ap.parse_args()

    text = _fetch(args.source)
    sim = _extract(text, "sim_hour_of_week_multiplier_total")
    lat = _extract(text, "latency_hour_of_week_multiplier_total")

    plt.figure(figsize=(12, 5))
    plt.plot(sim, label="simulator")
    plt.plot(lat, label="latency")
    plt.xlabel("hour of week")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
