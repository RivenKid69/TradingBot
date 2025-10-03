"""Focused tests for intrabar helpers on :class:`TradingEnv`."""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest


_lob_state_module = types.ModuleType("lob_state_cython")
_lob_state_module.N_FEATURES = 8
sys.modules.setdefault("lob_state_cython", _lob_state_module)

import trading_patchnew


class _StubTimeProvider:
    def time_ms(self) -> int:
        return 0


_TOPICS_STUB = types.SimpleNamespace(RISK="risk")


def _make_intrabar_dataframe() -> pd.DataFrame:
    """Construct a compact dataframe with intrabar payload variations."""

    base_ts = np.arange(4, dtype=np.int64) * 60_000 + 1_000
    return pd.DataFrame(
        {
            "ts_ms": base_ts.copy(),
            "decision_ts": (base_ts // 1000).astype(np.int64),
            "open": np.linspace(100.0, 103.0, base_ts.size),
            "high": np.linspace(100.5, 103.5, base_ts.size),
            "low": np.linspace(99.5, 102.5, base_ts.size),
            "close": np.linspace(100.2, 103.2, base_ts.size),
            "price": np.linspace(100.1, 103.1, base_ts.size),
            "quote_asset_volume": np.linspace(1_000, 1_300, base_ts.size),
            "intrabar_path": [
                [100.0, 100.5, None, 101.0],
                json.dumps({"path": [None, None]}),
                np.array([102.0, np.nan, 102.5]),
                None,
            ],
            "intrabar_points": [
                None,
                [201.0, None, 202.0],
                json.dumps({"points": [203.0, 204.0]}),
                pd.Series([205.0, np.nan, 206.0]),
            ],
        }
    )


@pytest.fixture(autouse=True)
def _stub_external_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``TradingEnv`` deterministic and lightweight for unit tests."""

    monkeypatch.setattr(trading_patchnew, "_HAVE_FAST_MARKET", False, raising=False)
    monkeypatch.setattr(trading_patchnew, "Topics", _TOPICS_STUB, raising=False)
    monkeypatch.setattr(trading_patchnew, "TimeProvider", _StubTimeProvider, raising=False)
    monkeypatch.setattr(trading_patchnew, "RealTimeProvider", _StubTimeProvider, raising=False)
    monkeypatch.setattr(
        trading_patchnew,
        "load_hourly_seasonality",
        lambda *_, **__: np.ones(trading_patchnew.HOURS_IN_WEEK, dtype=float),
    )
    monkeypatch.setattr(
        trading_patchnew,
        "get_no_trade_config",
        lambda *_args, **_kwargs: trading_patchnew.NoTradeConfig(),
    )


@pytest.fixture
def env_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[pd.DataFrame | None, Any | None], trading_patchnew.TradingEnv]:
    """Return a helper that instantiates :class:`TradingEnv` with a stub mediator."""

    def _factory(
        df: pd.DataFrame | None = None,
        exec_sim: Any | None = None,
    ) -> trading_patchnew.TradingEnv:
        class _DummyMediator:
            def __init__(self, env: trading_patchnew.TradingEnv) -> None:
                self.env = env
                self.exec = exec_sim
                self.calls: list[Any] = []

            def reset(self) -> None:
                self.calls.append("reset")

            def step(self, action: Any) -> Any:
                self.calls.append(action)
                return action

        monkeypatch.setattr(trading_patchnew, "Mediator", _DummyMediator)
        frame = df.copy() if df is not None else _make_intrabar_dataframe()
        return trading_patchnew.TradingEnv(frame, seed=13)

    return _factory


# ---------------------------------------------------------------------------
# ``_update_bar_interval``
# ---------------------------------------------------------------------------

def test_update_bar_interval_uses_row_values(env_factory: Callable[..., trading_patchnew.TradingEnv]) -> None:
    """Row-level interval columns should take precedence over previous values."""

    df = _make_intrabar_dataframe().assign(
        bar_interval_ms=[45_000, 30_000, 30_000, 30_000]
    )
    env = env_factory(df=df)

    # initial inference picks the first valid entry (45 seconds)
    assert env.bar_interval_ms == 45_000

    env._update_bar_interval(env.df.iloc[1], 1)
    assert env.bar_interval_ms == 30_000


def test_update_bar_interval_falls_back_to_index_diffs(
    env_factory: Callable[..., trading_patchnew.TradingEnv],
) -> None:
    """Interval should be inferred from neighbouring timestamp deltas when needed."""

    df = _make_intrabar_dataframe().drop(columns=["intrabar_points"]).drop(
        columns=["intrabar_path"], errors="ignore"
    )
    df = df.assign(bar_interval_ms=np.nan, bar_timeframe_ms=np.nan)
    env = env_factory(df=df)

    env.bar_interval_ms = None
    env._update_bar_interval(env.df.iloc[1], 1)
    assert env.bar_interval_ms == 60_000


# ---------------------------------------------------------------------------
# ``_normalize_intrabar_path_payload``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, expected",
    [
        pytest.param(b"[1, 2, null]", [1, 2], id="bytes_json_array"),
        pytest.param("{\"points\": [3, 4, null]}", [3, 4], id="json_string_dict"),
        pytest.param(pd.Series([5.0, np.nan, 6.0]), [5.0, 6.0], id="pandas_series"),
        pytest.param(np.array([7.0, np.nan, 8.0]), [7.0, 8.0], id="numpy_array"),
        pytest.param(b"not-json", None, id="malformed_bytes"),
    ],
)
def test_normalize_intrabar_payload_variants(
    env_factory: Callable[..., trading_patchnew.TradingEnv],
    payload: Any,
    expected: list[Any] | None,
) -> None:
    """The helper should sanitize multiple container types consistently."""

    env = env_factory()
    assert env._normalize_intrabar_path_payload(payload) == expected


# ---------------------------------------------------------------------------
# ``_maybe_configure_exec_timeframe``
# ---------------------------------------------------------------------------


class _TimeframeRecorder:
    def __init__(self) -> None:
        self.received: list[int] = []
        self._intrabar_timeframe_ms: int | None = 0

    def set_intrabar_timeframe_ms(self, value: int) -> None:
        self.received.append(int(value))
        self._intrabar_timeframe_ms = int(value)


def test_configure_exec_timeframe_idempotent_and_guarded(
    env_factory: Callable[..., trading_patchnew.TradingEnv],
) -> None:
    """Ensure configuration executes once and honours guard conditions."""

    df = _make_intrabar_dataframe().assign(bar_interval_ms=[np.nan, 15_000, 15_000, 15_000])
    exec_sim = _TimeframeRecorder()
    env = env_factory(df=df, exec_sim=exec_sim)

    assert exec_sim.received == [15_000]
    env._maybe_configure_exec_timeframe()
    assert exec_sim.received == [15_000]

    env._exec_intrabar_timeframe_configured = False
    exec_sim._intrabar_timeframe_ms = 5_000
    env._maybe_configure_exec_timeframe()
    assert exec_sim.received == [15_000]

    env._exec_intrabar_timeframe_configured = False
    env.bar_interval_ms = None
    env._maybe_configure_exec_timeframe()
    assert env._exec_intrabar_timeframe_configured is False
    assert exec_sim.received == [15_000]


# ---------------------------------------------------------------------------
# ``_maybe_forward_intrabar_path``
# ---------------------------------------------------------------------------


class _RecordingExecSim:
    def __init__(self) -> None:
        self.calls: list[list[Any]] = []
        self.lookup_counts: dict[str, int] = {}

    def set_intrabar_path(self, payload: list[Any]) -> None:
        self.calls.append(payload)

    def __getattr__(self, name: str) -> Any:
        self.lookup_counts[name] = self.lookup_counts.get(name, 0) + 1
        raise AttributeError(name)


class _MissingExecSim:
    def __init__(self) -> None:
        self.lookups: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.lookups.append(name)
        raise AttributeError(name)


def test_forward_intrabar_path_caching_and_error_handling(
    env_factory: Callable[..., trading_patchnew.TradingEnv],
) -> None:
    """Exercise caching behaviour and graceful degradation for missing hooks."""

    df = _make_intrabar_dataframe()
    env = env_factory(df=df)

    exec_sim = _RecordingExecSim()

    first_row = env.df.iloc[0]
    env._maybe_forward_intrabar_path(exec_sim, first_row)
    assert env._exec_intrabar_path_method == "set_intrabar_path"
    assert exec_sim.calls[0] == [100.0, 100.5, 101.0]

    second_row = env.df.iloc[1]
    env._maybe_forward_intrabar_path(exec_sim, second_row)
    assert exec_sim.calls[1] == [201.0, 202.0]

    assert exec_sim.lookup_counts.get("set_intrabar_reference_path", 0) == 1
    assert exec_sim.lookup_counts.get("set_intrabar_reference_points", 0) == 1

    missing_exec = _MissingExecSim()
    env._exec_intrabar_path_method = None
    env._maybe_forward_intrabar_path(missing_exec, env.df.iloc[2])
    assert env._exec_intrabar_path_method is False

    env._maybe_forward_intrabar_path(missing_exec, env.df.iloc[3])
    assert missing_exec.lookups  # ensured the first attempt tried to resolve hooks
