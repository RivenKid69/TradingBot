from datetime import datetime, timedelta, timezone

import pytest

from utils_time import get_hourly_multiplier


def _ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def test_daily_interpolation_progresses_over_entire_day():
    multipliers = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    monday = datetime(2024, 1, 1, tzinfo=timezone.utc)

    early = monday + timedelta(hours=1, minutes=30)
    early_frac = (1.5) / 24.0
    assert get_hourly_multiplier(_ts_ms(early), multipliers, interpolate=True) == pytest.approx(
        multipliers[0] + (multipliers[1] - multipliers[0]) * early_frac
    )

    late = monday + timedelta(hours=23, minutes=30)
    late_frac = (23.5) / 24.0
    assert get_hourly_multiplier(_ts_ms(late), multipliers, interpolate=True) == pytest.approx(
        multipliers[0] + (multipliers[1] - multipliers[0]) * late_frac
    )


def test_daily_interpolation_matches_base_without_interpolation():
    multipliers = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    monday = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = _ts_ms(monday + timedelta(hours=12))

    assert get_hourly_multiplier(ts, multipliers, interpolate=False) == multipliers[0]
