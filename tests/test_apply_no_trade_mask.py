import numpy as np
import pytest

from apply_no_trade_mask import _blocked_durations


def test_blocked_durations_respect_timeframe_for_final_bar():
    tf_ms = 60_000
    ts = np.array([0, 60_000, 120_000, 180_000, 240_000], dtype=np.int64)
    mask = np.array([False, True, True, False, True])

    durations = _blocked_durations(ts, mask, tf_ms=tf_ms)

    assert durations.tolist() == pytest.approx([2.0, 1.0])
    assert durations.sum() == pytest.approx(3.0)


def test_blocked_durations_single_bar_uses_timeframe():
    tf_ms = 60_000
    ts = np.array([1_000_000], dtype=np.int64)
    mask = np.array([True])

    durations = _blocked_durations(ts, mask, tf_ms=tf_ms)

    assert durations.tolist() == [pytest.approx(tf_ms / 60_000.0)]
