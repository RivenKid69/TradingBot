from __future__ import annotations

from typing import Sequence
import numpy as np


def simple_moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """Simple moving average.

    Скользящее среднее.
    """
    arr = np.asarray(values, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")
    if arr.size < window:
        raise ValueError("window larger than data")
    # Use convolution for efficiency / Используем свёртку для эффективности
    weights = np.ones(window) / window
    return np.convolve(arr, weights, mode="valid")
