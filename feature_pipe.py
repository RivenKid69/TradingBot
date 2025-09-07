# realtime/feature_pipe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from transformers import FeatureSpec, OnlineFeatureTransformer
from core_models import Bar
from core_contracts import FeaturePipe as FeaturePipeProtocol  # noqa: F401

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
    """Потоковый адаптер для :class:`OnlineFeatureTransformer`.

    На вход принимает объекты :class:`Bar` и возвращает словарь признаков,
    вычисленных без утечек во времени.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self._spec = FeatureSpec(
            lookbacks_prices=list(cfg.lookbacks_prices or [5, 15, 60]),
            rsi_period=int(cfg.rsi_period),
        )
        self._tr = OnlineFeatureTransformer(self._spec)

    def reset(self) -> None:
        """Сброс внутреннего состояния трансформера."""
        self._tr = OnlineFeatureTransformer(self._spec)

    def warmup(self, bars: Iterable[Bar] = ()) -> None:
        """Прогрев трансформера историческими барами."""
        self.reset()
        for b in bars:
            self._tr.update(symbol=b.symbol.upper(), ts_ms=int(b.ts), close=float(b.close))

    def on_bar(self, bar: Bar) -> Dict[str, Any]:
        """Обработка очередного бара."""
        feats = self._tr.update(symbol=bar.symbol.upper(), ts_ms=int(bar.ts), close=float(bar.close))
        return feats
