from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

import pipeline
from core_models import Bar
from pipeline import Reason, Stage, closed_bar_guard


def _make_bar(ts: int, *, is_final: bool) -> Bar:
    return Bar(
        ts=ts,
        symbol="BTCUSDT",
        open=Decimal("0"),
        high=Decimal("0"),
        low=Decimal("0"),
        close=Decimal("0"),
        is_final=is_final,
    )


@pytest.fixture()
def monitoring_mocks(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    stage_mock = Mock()
    reason_mock = Mock()
    monkeypatch.setattr(pipeline, "inc_stage", stage_mock)
    monkeypatch.setattr(pipeline, "inc_reason", reason_mock)
    return stage_mock, reason_mock


def test_closed_bar_guard_enforcement_disabled(monitoring_mocks: tuple[Mock, Mock]) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=False)

    result = closed_bar_guard(bar, now_ms=2_000, enforce=False, lag_ms=0)

    assert result.action == "pass"
    assert result.stage is Stage.CLOSED_BAR
    assert result.reason is None
    stage_mock.assert_called_once_with(Stage.CLOSED_BAR)
    reason_mock.assert_not_called()


def test_closed_bar_guard_requires_final_flag(monitoring_mocks: tuple[Mock, Mock]) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=False)

    result = closed_bar_guard(bar, now_ms=2_000, enforce=True, lag_ms=0)

    assert result.action == "drop"
    assert result.stage is Stage.CLOSED_BAR
    assert result.reason is Reason.INCOMPLETE_BAR
    stage_mock.assert_called_once_with(Stage.CLOSED_BAR)
    reason_mock.assert_called_once_with(Reason.INCOMPLETE_BAR)


def test_closed_bar_guard_final_websocket_bar_passes(
    monitoring_mocks: tuple[Mock, Mock]
) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=True)

    result = closed_bar_guard(bar, now_ms=1_050, enforce=True, lag_ms=0)

    assert result.action == "pass"
    assert result.stage is Stage.CLOSED_BAR
    assert result.reason is None
    stage_mock.assert_called_once_with(Stage.CLOSED_BAR)
    reason_mock.assert_not_called()


def test_closed_bar_guard_lag_rejects_recent_bar(monitoring_mocks: tuple[Mock, Mock]) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=False)

    result = closed_bar_guard(bar, now_ms=1_150, enforce=True, lag_ms=200)

    assert result.action == "drop"
    assert result.stage is Stage.CLOSED_BAR
    assert result.reason is Reason.INCOMPLETE_BAR
    stage_mock.assert_called_once_with(Stage.CLOSED_BAR)
    reason_mock.assert_called_once_with(Reason.INCOMPLETE_BAR)


def test_closed_bar_guard_lag_allows_stale_bar(monitoring_mocks: tuple[Mock, Mock]) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=False)

    result = closed_bar_guard(bar, now_ms=1_300, enforce=True, lag_ms=200)

    assert result.action == "pass"
    assert result.stage is Stage.CLOSED_BAR
    assert result.reason is None
    stage_mock.assert_called_once_with(Stage.CLOSED_BAR)
    reason_mock.assert_not_called()
