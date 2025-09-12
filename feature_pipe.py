# realtime/feature_pipe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

from transformers import FeatureSpec, OnlineFeatureTransformer

if TYPE_CHECKING:
    from core_contracts import FeaturePipe as FeaturePipeProtocol
    from core_models import Bar

@dataclass
class FeatureConfig:
    """
    Обёртка-конфиг для обратной совместимости.
    Реально используется FeatureSpec из features.transformers.
    """
    lookbacks_prices: List[int] = None
    rsi_period: int = 14

    def __post_init__(self) -> None:
        if self.lookbacks_prices is None:
            self.lookbacks_prices = [5, 15, 60]


class FeaturePipe:
    """Потоковый адаптер для онлайнового трансформера.

    На вход принимает :class:`core_models.Bar` и возвращает словарь фичей,
    совместимый с :class:`service_signal_runner.FeaturePipe`.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        spec = FeatureSpec(
            lookbacks_prices=list(cfg.lookbacks_prices or [5, 15, 60]),
            rsi_period=int(cfg.rsi_period),
        )
        self._tr = OnlineFeatureTransformer(spec)

    def warmup(self) -> None:  # pragma: no cover - нет инициализации
        """Нет тёплого старта для онлайновых фичей."""

    def on_bar(self, bar: "Bar") -> Dict[str, Any]:
        """Обрабатывает закрытую свечу и возвращает фичи."""
        feats = self._tr.update(
            symbol=str(bar.symbol).upper(),
            ts_ms=int(bar.ts),
            close=float(bar.close),
        )
        return feats or {}
