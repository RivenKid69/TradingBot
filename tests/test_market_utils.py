import datetime

from market_data_port import to_ms
from utils_time import parse_time_to_ms


def test_numeric_seconds_to_ms():
    assert to_ms(1) == 1000
    assert to_ms(1234567890) == 1234567890 * 1000
    assert to_ms(1.5) == 1500


def test_numeric_ms_passthrough():
    ms = 1_600_000_000_000
    assert to_ms(ms) == ms


def test_string_delegation():
    s = "2023-01-02 03:04:05"
    assert to_ms(s) == parse_time_to_ms(s)


def test_datetime_to_ms():
    dt = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
    assert to_ms(dt) == 1672531200000
