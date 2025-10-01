from __future__ import annotations

from decimal import Decimal
from typing import cast
from unittest.mock import Mock

import pytest

import pipeline
from core_models import Bar
from pipeline import Reason, Stage, open_bar_guard


def _make_bar(ts: int | None, *, is_final: bool) -> Bar:
    return Bar(
        ts=cast(int, ts),
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


def test_open_bar_guard_enforcement_disabled(
    monitoring_mocks: tuple[Mock, Mock]
) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=False)

    result = open_bar_guard(bar, now_ms=2_000, enforce=False, lag_ms=0)

    assert result.action == "pass"
    assert result.stage is Stage.OPEN_BAR
    assert result.reason is None
    stage_mock.assert_called_once_with(Stage.OPEN_BAR)
    reason_mock.assert_not_called()


def test_open_bar_guard_requires_final_flag_with_zero_lag(
    monitoring_mocks: tuple[Mock, Mock]
) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=False)

    result = open_bar_guard(bar, now_ms=2_000, enforce=True, lag_ms=0)

    assert result.action == "drop"
    assert result.stage is Stage.OPEN_BAR
    assert result.reason is Reason.INCOMPLETE_BAR
    stage_mock.assert_called_once_with(Stage.OPEN_BAR)
    reason_mock.assert_called_once_with(Reason.INCOMPLETE_BAR)


def test_open_bar_guard_allows_final_bar_with_zero_lag(
    monitoring_mocks: tuple[Mock, Mock]
) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=True)

    result = open_bar_guard(bar, now_ms=1_050, enforce=True, lag_ms=0)

    assert result.action == "pass"
    assert result.stage is Stage.OPEN_BAR
    assert result.reason is None
    stage_mock.assert_called_once_with(Stage.OPEN_BAR)
    reason_mock.assert_not_called()


def test_open_bar_guard_rejects_missing_timestamp(
    monitoring_mocks: tuple[Mock, Mock]
) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(None, is_final=True)

    result = open_bar_guard(bar, now_ms=2_000, enforce=True, lag_ms=100)

    assert result.action == "drop"
    assert result.stage is Stage.OPEN_BAR
    assert result.reason is Reason.INCOMPLETE_BAR
    stage_mock.assert_called_once_with(Stage.OPEN_BAR)
    reason_mock.assert_called_once_with(Reason.INCOMPLETE_BAR)


def test_open_bar_guard_positive_lag_rejects_recent_bar(
    monitoring_mocks: tuple[Mock, Mock]
) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=True)

    result = open_bar_guard(bar, now_ms=1_050, enforce=True, lag_ms=200)

    assert result.action == "drop"
    assert result.stage is Stage.OPEN_BAR
    assert result.reason is Reason.INCOMPLETE_BAR
    stage_mock.assert_called_once_with(Stage.OPEN_BAR)
    reason_mock.assert_called_once_with(Reason.INCOMPLETE_BAR)


def test_open_bar_guard_positive_lag_allows_sufficient_delay(
    monitoring_mocks: tuple[Mock, Mock]
) -> None:
    stage_mock, reason_mock = monitoring_mocks
    bar = _make_bar(1_000, is_final=True)

    result = open_bar_guard(bar, now_ms=1_300, enforce=True, lag_ms=200)

    assert result.action == "pass"
    assert result.stage is Stage.OPEN_BAR
    assert result.reason is None
    stage_mock.assert_called_once_with(Stage.OPEN_BAR)
    reason_mock.assert_not_called()
