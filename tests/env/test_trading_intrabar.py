"""Unit tests for intrabar utilities on :class:`TradingEnv`."""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest


lob_state_module = types.ModuleType("lob_state_cython")
lob_state_module.N_FEATURES = 8
sys.modules.setdefault("lob_state_cython", lob_state_module)

import trading_patchnew


class _StubTimeProvider:
    def time_ms(self) -> int:
        return 0


_TOPICS_STUB = types.SimpleNamespace(RISK="risk")


def _make_base_dataframe() -> pd.DataFrame:
    """Return a synthetic bar dataframe with minimal trading columns."""

    base_ts = np.arange(5, dtype=np.int64) * 60_000 + 1_000
    return pd.DataFrame(
        {
            "ts_ms": base_ts.copy(),
            "decision_ts": (base_ts // 1000).astype(np.int64),
            "open": np.linspace(100.0, 102.0, base_ts.size),
            "high": np.linspace(100.5, 102.5, base_ts.size),
            "low": np.linspace(99.5, 101.5, base_ts.size),
            "close": np.linspace(100.2, 102.2, base_ts.size),
            "price": np.linspace(100.1, 102.1, base_ts.size),
            "quote_asset_volume": np.linspace(1_000, 1_400, base_ts.size),
            "intrabar_path": [
                json.dumps({"path": [100.0, 100.5, 101.0]}),
                [101.0, 101.5, 102.0],
                np.array([102.0, np.nan, 103.0]),
                {"points": [103.0, 103.5]},
                None,
            ],
        }
    )


@pytest.fixture(autouse=True)
def _stub_external_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``TradingEnv`` deterministic for unit tests."""

    def _fake_seasonality(*_: Any, **__: Any) -> np.ndarray:
        return np.ones(trading_patchnew.HOURS_IN_WEEK, dtype=float)

    monkeypatch.setattr(trading_patchnew, "load_hourly_seasonality", _fake_seasonality)
    monkeypatch.setattr(trading_patchnew, "Topics", _TOPICS_STUB, raising=False)
    monkeypatch.setattr(trading_patchnew, "TimeProvider", _StubTimeProvider, raising=False)
    monkeypatch.setattr(trading_patchnew, "RealTimeProvider", _StubTimeProvider, raising=False)
    monkeypatch.setattr(
        trading_patchnew,
        "get_no_trade_config",
        lambda *_args, **_kwargs: trading_patchnew.NoTradeConfig(),
    )


@pytest.fixture
def env_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[[pd.DataFrame | None, Any | None], trading_patchnew.TradingEnv]:
    """Return a factory producing :class:`TradingEnv` instances with a dummy mediator."""

    def _factory(df: pd.DataFrame | None = None, exec_sim: Any | None = None) -> trading_patchnew.TradingEnv:
        class _DummyMediator:
            def __init__(self, env: trading_patchnew.TradingEnv):
                self.env = env
                self.exec = exec_sim
                self.calls: list[str] = []

            def reset(self) -> None:
                self.calls.append("reset")

        monkeypatch.setattr(trading_patchnew, "Mediator", _DummyMediator)
        dataframe = df.copy() if df is not None else _make_base_dataframe()
        return trading_patchnew.TradingEnv(dataframe, seed=7)

    return _factory


@pytest.mark.parametrize(
    "mutator, expected",
    [
        pytest.param(
            lambda df: df.assign(bar_interval_ms=[np.nan, 61_000, 61_000, 61_000, 61_000]),
            61_000,
            id="bar_interval_column",
        ),
        pytest.param(
            lambda df: df.assign(bar_timeframe_ms=[0, 45_000, 45_000, 45_000, 45_000]),
            45_000,
            id="bar_timeframe_column",
        ),
        pytest.param(lambda df: df, 60_000, id="timestamp_ms_diff"),
        pytest.param(
            lambda df: df.drop(columns=["ts_ms", "decision_ts"]).assign(open_ts=[0, 60, 120, 180, 240]),
            60_000,
            id="open_ts_seconds",
        ),
    ],
)
def test_infer_bar_interval_from_dataframe(env_factory: Callable[..., trading_patchnew.TradingEnv], mutator: Callable[[pd.DataFrame], pd.DataFrame], expected: int) -> None:
    """``TradingEnv._infer_bar_interval_from_dataframe`` honours explicit columns and diffs."""

    df = mutator(_make_base_dataframe())
    env = env_factory(df=df)
    assert env.bar_interval_ms == expected


@pytest.mark.parametrize(
    "payload, expected",
    [
        (b"[1, 2, 3]", [1, 2, 3]),
        (pd.Series([1.0, np.nan, 2.0]), [1.0, 2.0]),
        (np.array([1.0, np.nan, 3.0]), [1.0, 3.0]),
        ("{\"path\": [4, 5, null]}", [4, 5]),
        ({"data": [None, 7.0]}, [7.0]),
        ("not json", None),
        ([None, float("nan")], None),
    ],
)
def test_normalize_intrabar_payload(env_factory: Callable[..., trading_patchnew.TradingEnv], payload: Any, expected: list[Any] | None) -> None:
    """Ensure ``_normalize_intrabar_path_payload`` accepts multiple payload types."""

    env = env_factory()
    result = env._normalize_intrabar_path_payload(payload)
    assert result == expected


class _RecordingExecSim:
    def __init__(self) -> None:
        self.calls: list[list[Any]] = []
        self.lookup_counts: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        self.lookup_counts[name] = self.lookup_counts.get(name, 0) + 1
        if name == "set_intrabar_path":
            def _setter(payload: list[Any]) -> None:
                self.calls.append(payload)

            return _setter
        raise AttributeError(name)


def test_forward_intrabar_path_caches_method(env_factory: Callable[..., trading_patchnew.TradingEnv]) -> None:
    """``_maybe_forward_intrabar_path`` normalizes payloads and caches the resolved method."""

    env = env_factory()
    exec_sim = _RecordingExecSim()

    first_row = env.df.iloc[0]
    env._maybe_forward_intrabar_path(exec_sim, first_row)
    assert env._exec_intrabar_path_method == "set_intrabar_path"
    assert exec_sim.calls[0] == [100.0, 100.5, 101.0]

    second_row = env.df.iloc[2]
    env._maybe_forward_intrabar_path(exec_sim, second_row)
    assert len(exec_sim.calls) == 2
    assert exec_sim.calls[1] == [102.0, 103.0]

    assert exec_sim.lookup_counts["set_intrabar_path"] == 2
    assert exec_sim.lookup_counts.get("set_intrabar_reference_path", 0) == 1


class _TimeframeExecSim:
    def __init__(self) -> None:
        self.received: list[int] = []
        self._intrabar_timeframe_ms: int = 0

    def set_intrabar_timeframe_ms(self, value: int) -> None:
        self.received.append(value)
        self._intrabar_timeframe_ms = value


def test_configure_exec_timeframe_respects_existing_value(env_factory: Callable[..., trading_patchnew.TradingEnv]) -> None:
    """``_maybe_configure_exec_timeframe`` sets the timeframe once and respects cached state."""

    exec_sim = _TimeframeExecSim()
    df = _make_base_dataframe().assign(bar_interval_ms=[np.nan, 15_000, 15_000, 15_000, 15_000])
    env = env_factory(df=df, exec_sim=exec_sim)

    assert exec_sim.received == [15_000]
    assert env._exec_intrabar_timeframe_configured is True

    env._maybe_configure_exec_timeframe()
    assert exec_sim.received == [15_000]

    env._exec_intrabar_timeframe_configured = False
    exec_sim._intrabar_timeframe_ms = 10_000
    env._maybe_configure_exec_timeframe()
    assert exec_sim.received == [15_000]

