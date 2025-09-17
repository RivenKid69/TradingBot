# -*- coding: utf-8 -*-
"""
core_contracts.py
Единые интерфейсы (контракты) для ключевых компонентов системы.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Iterable, Iterator, Optional, Sequence, Mapping, Any, Dict, List, runtime_checkable

import pandas as pd

from core_models import Instrument, Bar, Tick, Order, ExecReport, Position, PortfolioLimits


RunId = str


@runtime_checkable
class MarketDataSource(Protocol):
    """
    Источник рыночных данных.
    Реализации: OfflineBarSource (parquet/csv), BinancePublicDataSource (REST/WS).
    """

    def stream_bars(self, symbols: Sequence[str], interval_ms: int) -> Iterator[Bar]:
        ...

    def stream_ticks(self, symbols: Sequence[str]) -> Iterator[Tick]:
        ...


@runtime_checkable
class TradeExecutor(Protocol):
    """
    Исполнитель торговых приказов.
    В симуляторе — синхронное исполнение.
    В live — может быть асинхронной интеграцией, но метод execute возвращает фактический ExecReport при завершении сделки.
    """

    def execute(self, order: Order) -> ExecReport:
        ...

    def cancel(self, client_order_id: str) -> None:
        ...

    def get_open_positions(self, symbols: Optional[Sequence[str]] = None) -> Mapping[str, Position]:
        ...


@runtime_checkable
class FeaturePipe(Protocol):
    """
    Преобразование входящих баров/тиков в вектор признаков.

    Стриминговое использование:
        последовательно вызывайте :meth:`update` для каждого нового бара и
        получайте словарь ``feature_name -> value``.

    Офлайн-использование:
        реализации могут поддерживать методы :meth:`fit`,
        :meth:`transform_df` и :meth:`make_targets` для работы с
        :class:`pandas.DataFrame`.
        Эти методы опциональны и могут отсутствовать у конкретной реализации.
    """

    def reset(self) -> None:
        ...

    def warmup(self) -> None:
        ...

    def update(self, bar: Bar) -> Mapping[str, Any]:
        ...

    # ------------------------------------------------------------------
    # Optional offline helpers
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> None:
        ...

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def make_targets(self, df: pd.DataFrame) -> Optional[pd.Series]:
        ...


@runtime_checkable
class RiskGuards(Protocol):
    """
    Пред- и пост-торговые проверки и обновления состояний риска.
    pre_trade возвращает None, если всё ок, либо строковый код/сообщение причины блокировки.
    """

    def pre_trade(self, order: Order, position: Optional[Position] = None) -> Optional[str]:
        ...

    def post_trade(self, report: ExecReport) -> None:
        ...


@dataclass(frozen=True)
class PolicyCtx:
    """
    Контекст принятия решения, который может предоставлять BacktestEngine/сервис.
    Обязательные поля: ``ts`` (мс) и ``symbol``.
    Дополнительно могут передаваться текущая позиция и лимиты портфеля.
    """

    ts: int
    symbol: str
    position: Optional[Position] = None
    limits: Optional[PortfolioLimits] = None
    extra: Optional[Dict[str, Any]] = None
    signal_quality_cfg: Optional[Any] = None


@runtime_checkable
class SignalPolicy(Protocol):
    """
    Политика принятия решений. На вход подаются признаки и контекст,
    возвращает список заявок (:class:`Order`) для исполнения.
    """

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        ...


@runtime_checkable
class BacktestEngine(Protocol):
    """
    Движок бэктеста: перебор данных, вызов политики, исполнение и агрегирование отчётов.
    """

    def run(self, *, run_id: RunId) -> Mapping[str, Any]:
        """
        Возвращает словарь с итоговыми артефактами: trades: List[ExecReport], equity: List[Dict], metrics: Dict
        """
        ...
