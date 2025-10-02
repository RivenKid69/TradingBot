from __future__ import annotations

import pytest

from pipeline import check_ttl, compute_expires_at


def test_compute_expires_at_normalizes_inputs() -> None:
    close_ms = "1700000000000"
    timeframe_ms = "60000"

    expires_at = compute_expires_at(close_ms, timeframe_ms)

    assert expires_at == 1_700_000_000_000 + 60_000


def test_compute_expires_at_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        compute_expires_at(None, 60_000)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_expires_at(1_700_000_000_000, 0)


@pytest.mark.parametrize(
    "offset, expected_valid",
    [
        (0, True),
        (59_999, True),
        (60_000, True),
        (60_001, False),
    ],
)
def test_check_ttl_window(offset: int, expected_valid: bool) -> None:
    close_ms = 1_700_000_000_000
    timeframe_ms = 60_000
    now_ms = close_ms + offset

    expires_at_ms = compute_expires_at(close_ms, timeframe_ms)
    ok, observed_expires_at_ms, reason = check_ttl(
        close_ms, now_ms, timeframe_ms
    )

    assert observed_expires_at_ms == expires_at_ms
    assert ok is expected_valid
    if expected_valid:
        assert reason == ""
    else:
        assert str(timeframe_ms) in reason
