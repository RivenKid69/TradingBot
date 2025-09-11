"""Microbenchmark for ExecutionSimulator seasonality lookup."""
import time
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from execution_sim import ExecutionSimulator


def _run(use_seasonality: bool, interpolate: bool = False, n: int = 100_000) -> float:
    sim = ExecutionSimulator(
        liquidity_seasonality=[1.0] * 168,
        spread_seasonality=[1.0] * 168,
        use_seasonality=use_seasonality,
        seasonality_interpolate=interpolate,
    )
    ts = np.arange(n, dtype=np.int64) * 60_000  # 1-minute steps
    start = time.perf_counter()
    for t in ts:
        sim.set_market_snapshot(bid=100.0, ask=100.1, ts_ms=int(t))
    return time.perf_counter() - start


def main() -> None:
    cases = [
        (False, False),
        (True, False),
        (True, True),
    ]
    for use_seasonality, interp in cases:
        dt = _run(use_seasonality, interpolate=interp)
        print(
            f"use_seasonality={use_seasonality} interpolate={interp}: {dt:.3f}s"
        )


if __name__ == "__main__":
    main()
