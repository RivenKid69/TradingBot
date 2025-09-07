# -*- coding: utf-8 -*-
"""
core_contracts.py
Единые интерфейсы (контракты) для ключевых компонентов системы.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Iterable, Iterator, Optional, Sequence, Mapping, Any, Dict, List, runtime_checkable

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
    Возвращает словарь feature_name->value или None, если окно/прогрев ещё не готов.
    """

    def reset(self) -> None:
        ...

    def warmup(self) -> None:
        ...

    def on_bar(self, bar: Bar) -> Optional[Mapping[str, Any]]:
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


@runtime_checkable
class SignalPolicy(Protocol):
    """
    Политика принятия решений. На вход подаются признаки и контекст.
    Возвращает список заявок (Order) для исполнения.
    """

    def decide(self, features: Mapping[str, Any], ctx: Mapping[str, Any]) -> List[Order]:
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


@dataclass(frozen=True)
class DecisionContext:
    """
    Контекст принятия решения, который может предоставлять BacktestEngine/сервис.
    Обязательные ключи: ts (int ms), symbol (str).
    Дополнительно: position (Position), limits (PortfolioLimits), extra (Dict[str, Any]).
    """
    ts: int
    symbol: str
    position: Optional[Position] = None
    limits: Optional[PortfolioLimits] = None
    extra: Optional[Dict[str, Any]] = None
