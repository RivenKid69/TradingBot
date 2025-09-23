import numpy as np

from apply_no_trade_mask import _blocked_durations


def test_blocked_durations_single_bar_counts_full_minute():
    ts = np.array([0, 60_000, 120_000], dtype=np.int64)
    mask = [False, True, False]

    durations = _blocked_durations(ts, mask, bar_length_ms=60_000)

    np.testing.assert_allclose(durations, [1.0])


def test_blocked_durations_two_bars_counts_full_minutes():
    ts = np.array([0, 60_000, 120_000, 180_000], dtype=np.int64)
    mask = [False, True, True, False]

    durations = _blocked_durations(ts, mask)

    np.testing.assert_allclose(durations, [2.0])
