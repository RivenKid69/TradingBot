import datetime

from utils_time import parse_time_to_ms
from impl_offline_data import to_ms


def test_numeric_seconds_to_ms():
    assert parse_time_to_ms(1) == 1000
    assert parse_time_to_ms(1234567890) == 1234567890 * 1000


def test_numeric_ms_passthrough():
    ms = 1_600_000_000_000
    assert parse_time_to_ms(ms) == ms


def test_datetime_string_to_ms():
    s = "2023-01-02 03:04:05"
    assert parse_time_to_ms(s) == 1672628645000


def test_iso_string_to_ms():
    dt = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
    assert parse_time_to_ms(dt.isoformat()) == 1672531200000


def test_to_ms_naive_iso_treated_as_utc():
    dt_str = "2023-01-01T00:00:00"
    assert to_ms(dt_str) == 1672531200000
