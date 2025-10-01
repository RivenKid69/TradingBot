import math
import random
from typing import Any, Optional

import logging
import pytest

import execution_sim
from execution_sim import ExecutionSimulator


@pytest.fixture
def bare_sim() -> ExecutionSimulator:
    sim = ExecutionSimulator.__new__(ExecutionSimulator)
    sim.symbol = "TESTUSDT"
    sim.seed = 1234
    sim.step_ms = 1000
    sim._intrabar_timeframe_ms = 1000
    sim._intrabar_config_timeframe_ms = None
    sim._timing_timeframe_ms = None
    sim._intrabar_latency_source = "latency"
    sim._intrabar_latency_constant_ms = None
    sim._intrabar_log_warnings = False
    sim._intrabar_warn_next_log_ms = 0
    sim._intrabar_price_model = None
    sim._intrabar_seed_mode = "stable"
    sim._order_seq_counter = 0
    sim._intrabar_path = []
    sim._intrabar_volume_profile = []
    sim._intrabar_path_bar_ts = None
    sim._intrabar_path_start_ts = None
    sim._intrabar_path_timeframe_ms = None
    sim._intrabar_path_total_volume = None
    sim._intrabar_volume_used = 0.0
    sim._intrabar_reference_debug_logged = 0
    sim._intrabar_debug_max_logs = 0
    sim._last_bar_open = 100.0
    sim._last_bar_high = 115.0
    sim._last_bar_low = 95.0
    sim._last_bar_close = 110.0
    sim._last_bar_close_ts = 1_000_000
    sim._last_bid = 99.0
    sim._last_ask = 101.0
    sim._last_vol_raw = {}
    sim._last_vol_factor = None
    sim._last_ref_price = None
    return sim


@pytest.mark.parametrize(
    "source,sample,child_offset,const_override,expected",
    [
        ("latency", {"total_ms": 123.4}, None, None, 123),
        ("latency", -5, None, None, 0),
        ("constant", 999.9, None, 321, 321),
        ("constant", 999.9, 456, None, 456),
        ("child_offset", 80, 12, None, 12),
        ("latency+child", 80, 20, None, 100),
        ("child", 80, None, None, 80),
    ],
)
def test_intrabar_latency_ms_modes(
    bare_sim: ExecutionSimulator,
    source: str,
    sample: Any,
    child_offset: Optional[int],
    const_override: Optional[int],
    expected: int,
) -> None:
    bare_sim._intrabar_latency_source = source
    bare_sim._intrabar_latency_constant_ms = const_override

    result = bare_sim._intrabar_latency_ms(sample, child_offset)
    assert result == expected


@pytest.mark.parametrize(
    "latency,timeframe,expected",
    [
        (500, 1000, 0.5),
        (-20, 1000, 0.0),
        (1500, 1000, 1.0),
        (200, 0, 0.2),
        (None, None, 0.0),
    ],
)
def test_intrabar_time_fraction_clamping(
    bare_sim: ExecutionSimulator, latency: Optional[float], timeframe: Optional[float], expected: float
) -> None:
    fraction = bare_sim._intrabar_time_fraction(latency, timeframe)
    assert fraction == pytest.approx(expected)


def test_intrabar_latency_warning_throttle(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bare_sim: ExecutionSimulator) -> None:
    bare_sim._intrabar_log_warnings = True
    bare_sim._intrabar_price_model = "linear"

    events = []

    def fake_now() -> int:
        return events[-1]

    monkeypatch.setattr("execution_sim.now_ms", fake_now)

    caplog.set_level("WARNING")
    events.append(1_000)
    first = bare_sim._intrabar_time_fraction(2_000, 1_000)
    events.append(1_200)
    second = bare_sim._intrabar_time_fraction(2_000, 1_000)

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(1.0)
    assert bare_sim._intrabar_warn_next_log_ms == 2_000
    assert sum(1 for record in caplog.records if "exceeds timeframe" in record.msg) == 1


def test_set_intrabar_reference_and_reset(bare_sim: ExecutionSimulator) -> None:
    bar_ts = 1_001_000
    path = [
        {"fraction": 0.0, "price": 100.0, "volume": 1.0},
        {"offset_ms": 400, "price": 105.0, "volume": 2.0},
        {"fraction": 1.0, "price": 110.0},
    ]

    bare_sim.set_intrabar_reference(bar_ts, path)

    assert bare_sim._intrabar_path_bar_ts == bar_ts
    assert bare_sim._intrabar_path_timeframe_ms == 1_000
    assert bare_sim._intrabar_path_start_ts == bar_ts - 1_000
    assert bare_sim._intrabar_path_total_volume == pytest.approx(3.0)
    assert bare_sim._intrabar_volume_profile == [
        (bar_ts - 1_000, 1.0),
        (bar_ts - 600, 2.0),
    ]

    before_reset = list(bare_sim._intrabar_path)
    bare_sim._maybe_reset_intrabar_reference(bar_ts - 1)
    assert bare_sim._intrabar_path == before_reset

    bare_sim._maybe_reset_intrabar_reference(bar_ts + 1_000)
    assert bare_sim._intrabar_path == []
    assert bare_sim._intrabar_path_total_volume is None


def test_intrabar_price_from_path_interpolation_and_clipping(bare_sim: ExecutionSimulator) -> None:
    bar_ts = 1_001_000
    bare_sim.set_intrabar_reference(
        bar_ts,
        [
            {"fraction": 0.0, "price": 100.0},
            {"offset_ms": 400, "price": 105.0},
            {"fraction": 1.0, "price": 120.0},
        ],
    )

    price_start, clipped_start = bare_sim._intrabar_price_from_path(
        side="BUY", time_fraction=0.0, fallback=None, bar_ts=bar_ts
    )
    assert price_start == pytest.approx(100.0)
    assert clipped_start is True

    price_mid, clipped_mid = bare_sim._intrabar_price_from_path(
        side="BUY", time_fraction=0.4, fallback=None, bar_ts=bar_ts
    )
    assert price_mid == pytest.approx(105.0)
    assert clipped_mid is False

    price_end, clipped_end = bare_sim._intrabar_price_from_path(
        side="SELL", time_fraction=1.0, fallback=None, bar_ts=bar_ts
    )
    assert price_end == pytest.approx(115.0)
    assert clipped_end is True

    bare_sim._intrabar_path[0] = (bare_sim._intrabar_path_start_ts + 200, bare_sim._intrabar_path[0][1])
    assert (
        bare_sim._intrabar_price_from_path(
            side="BUY", time_fraction=0.5, fallback=None, bar_ts=bar_ts
        )
        is None
    )


def test_intrabar_atr_hint_sources(bare_sim: ExecutionSimulator) -> None:
    bare_sim._last_vol_raw = {"atr": 1.5, "atr_usd": 2.5}
    assert bare_sim._intrabar_atr_hint() == pytest.approx(2.5)

    bare_sim._last_vol_raw = {}
    bare_sim._last_vol_factor = 2.0
    bare_sim._last_ref_price = 100.0
    assert bare_sim._intrabar_atr_hint() == pytest.approx(2.0)

    bare_sim._last_vol_factor = None
    assert bare_sim._intrabar_atr_hint() is None


@pytest.mark.parametrize(
    "mode,fraction,expected_price,expected_clipped",
    [
        ("linear", 0.25, 102.5, False),
        ("mid", 0.5, 100.0, False),
        ("bridge", 0.75, None, None),
    ],
)
def test_compute_intrabar_price_modes(
    bare_sim: ExecutionSimulator,
    mode: str,
    fraction: float,
    expected_price: Optional[float],
    expected_clipped: Optional[bool],
) -> None:
    bar_ts = bare_sim._last_bar_close_ts
    bare_sim._intrabar_price_model = mode
    bare_sim._last_bid = 99.0
    bare_sim._last_ask = 101.0
    if mode == "bridge":
        bare_sim._last_bar_high = 120.0
        bare_sim._last_bar_low = 90.0
        atr_hint = 40.0
        bare_sim._last_vol_raw = {"atr": atr_hint}
        sigma = max(abs(bare_sim._last_bar_high - bare_sim._last_bar_low), atr_hint)
        rng_seed = bare_sim._intrabar_rng_seed(bar_ts=bar_ts, side="BUY", order_seq=7)
        rng = random.Random(rng_seed)
        std = float(sigma) * math.sqrt(fraction * (1.0 - fraction))
        noise = rng.gauss(0.0, std)
        linear = bare_sim._last_bar_open + (bare_sim._last_bar_close - bare_sim._last_bar_open) * fraction
        expected_price = max(min(linear + noise, bare_sim._last_bar_high), bare_sim._last_bar_low)
        expected_clipped = not (bare_sim._last_bar_low <= linear + noise <= bare_sim._last_bar_high)

    price, clipped, returned_fraction = bare_sim._compute_intrabar_price(
        side="BUY",
        time_fraction=fraction,
        fallback_price=100.0,
        bar_ts=bar_ts,
        order_seq=7,
    )
    assert returned_fraction == pytest.approx(max(0.0, min(1.0, fraction)))
    assert price == pytest.approx(expected_price)
    assert clipped is expected_clipped


def test_compute_intrabar_price_reference_and_fallback(bare_sim: ExecutionSimulator) -> None:
    bar_ts = bare_sim._last_bar_close_ts
    bare_sim._intrabar_price_model = "reference"
    bare_sim.set_intrabar_reference(
        bar_ts,
        [
            (bar_ts - 1_000, 100.0),
            (bar_ts - 500, 105.0),
            (bar_ts, 110.0),
        ],
    )

    price, clipped, _ = bare_sim._compute_intrabar_price(
        side="BUY", time_fraction=0.0, fallback_price=90.0, bar_ts=bar_ts
    )
    assert price == pytest.approx(100.0)
    assert clipped is True

    bare_sim._intrabar_path = [(bar_ts - 400, 104.0)]
    bare_sim._last_bar_open = None
    bare_sim._last_bar_close = None
    price_fallback, clipped_fallback, _ = bare_sim._compute_intrabar_price(
        side="BUY", time_fraction=0.5, fallback_price=99.0, bar_ts=bar_ts
    )
    assert price_fallback == pytest.approx(99.0)
    assert clipped_fallback is False


def test_current_bar_window_cache_and_recompute(bare_sim: ExecutionSimulator) -> None:
    ts = 1_005_500

    bare_sim._intrabar_path_timeframe_ms = None
    bare_sim._intrabar_path_start_ts = None
    bare_sim._intrabar_path_bar_ts = None

    timeframe, start_ts, end_ts = bare_sim._current_bar_window(ts)
    assert timeframe == 1_000
    assert start_ts == 1_005_000
    assert end_ts == 1_006_000

    bare_sim._intrabar_path_timeframe_ms = 2_000
    bare_sim._intrabar_path_start_ts = 1_004_000
    bare_sim._intrabar_path_bar_ts = None

    timeframe_cached, start_cached, end_cached = bare_sim._current_bar_window(ts)
    assert timeframe_cached == 2_000
    assert start_cached == 1_004_000
    assert end_cached == 1_006_000

    bare_sim._intrabar_path_start_ts = None
    bare_sim._intrabar_path_bar_ts = None
    ts_jump = ts + 3_000

    timeframe_jump, start_jump, end_jump = bare_sim._current_bar_window(ts_jump)
    assert timeframe_jump == 2_000
    assert start_jump == 1_008_000
    assert end_jump == 1_010_000

    bare_sim._intrabar_path_timeframe_ms = -50
    bare_sim._intrabar_path_start_ts = 1_000_000
    bare_sim._intrabar_path_bar_ts = 1_000_900

    timeframe_invalid, start_invalid, end_invalid = bare_sim._current_bar_window(1_000_123)
    assert timeframe_invalid is None
    assert start_invalid == 1_000_000
    assert end_invalid == 1_000_900


def test_intrabar_debug_counter_reset_and_logging_gate(
    bare_sim: ExecutionSimulator, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG", logger="execution_sim")

    bare_sim._intrabar_debug_max_logs = 1
    bare_sim._intrabar_debug_logged = 7
    bare_sim._intrabar_reference_debug_logged = 3

    bare_sim._reset_intrabar_debug_counter()
    assert bare_sim._intrabar_debug_logged == 0
    assert bare_sim._intrabar_reference_debug_logged == 0

    bar_ts = bare_sim._last_bar_close_ts + bare_sim._intrabar_timeframe_ms
    path = [
        {"fraction": 0.0, "price": 100.0},
        {"fraction": 0.5, "price": 105.0},
        {"fraction": 1.0, "price": 110.0},
    ]

    caplog.clear()
    bare_sim.set_intrabar_reference(bar_ts, path)
    assert bare_sim._intrabar_reference_debug_logged == 1
    assert sum("set intrabar reference" in record.message for record in caplog.records) == 1

    caplog.clear()
    bare_sim.set_intrabar_reference(bar_ts, path)
    assert bare_sim._intrabar_reference_debug_logged == 1
    assert not any("set intrabar reference" in record.message for record in caplog.records)

    caplog.clear()

    def emit_intrabar_debug() -> None:
        if execution_sim.logger.isEnabledFor(logging.DEBUG):
            limit = int(bare_sim._intrabar_debug_max_logs)
            if limit <= 0 or bare_sim._intrabar_debug_logged < limit:
                execution_sim.logger.debug("generic intrabar debug gate test")
                bare_sim._intrabar_debug_logged += 1

    emit_intrabar_debug()
    emit_intrabar_debug()

    assert bare_sim._intrabar_debug_logged == 1
    assert sum("generic intrabar debug gate test" in record.message for record in caplog.records) == 1

    bare_sim._reset_intrabar_debug_counter()
    assert bare_sim._intrabar_debug_logged == 0
    assert bare_sim._intrabar_reference_debug_logged == 0

    caplog.clear()
    emit_intrabar_debug()
    assert bare_sim._intrabar_debug_logged == 1
    assert sum("generic intrabar debug gate test" in record.message for record in caplog.records) == 1

    bare_sim._reset_intrabar_debug_counter()
    caplog.clear()

    if bare_sim._intrabar_path_start_ts is not None:
        bare_sim._intrabar_path_start_ts += 123
    next_bar_ts = bar_ts + bare_sim._intrabar_timeframe_ms
    bare_sim._maybe_reset_intrabar_reference(next_bar_ts)

    assert bare_sim._intrabar_reference_debug_logged == 1
    assert any("reset intrabar reference path" in record.message for record in caplog.records)
    assert bare_sim._intrabar_path == []
    assert bare_sim._intrabar_path_start_ts is None
    assert bare_sim._intrabar_path_bar_ts is None
    assert bare_sim._intrabar_path_timeframe_ms is None

