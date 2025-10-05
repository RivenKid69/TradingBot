"""Smoke-test for :class:`DeterministicPortfolioAllocator`.

The original file validated Gym wrappers.  After the transition to
deterministic allocation we keep a lightweight smoke check that verifies the
allocator honours the configured constraints when fed with dummy scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_allocator import DeterministicPortfolioAllocator


def run_smoke_test() -> None:
    rng = np.random.default_rng(42)
    symbols = [f"SYM{i}" for i in range(8)]
    scores = rng.normal(loc=0.5, scale=0.2, size=len(symbols))
    df = pd.DataFrame([scores], columns=symbols)

    allocator = DeterministicPortfolioAllocator()
    weights = allocator.compute_weights(
        df,
        top_n=5,
        threshold=0.1,
        max_weight_per_symbol=0.3,
        max_gross_exposure=0.9,
        realloc_threshold=0.0,
    )

    assert not weights.empty, "Allocator returned empty weights for synthetic scores"
    assert (weights <= 0.3 + 1e-9).all(), "Per-symbol cap violated"
    assert float(weights.sum()) <= 0.9 + 1e-9, "Gross exposure exceeds max_gross_exposure"

    print("✅ DeterministicPortfolioAllocator smoke-test passed")


if __name__ == "__main__":  # pragma: no cover
    run_smoke_test()
