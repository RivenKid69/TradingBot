# realtime/feature_pipe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from transformers import FeatureSpec, OnlineFeatureTransformer

if TYPE_CHECKING:
    from core_contracts import FeaturePipe as FeaturePipeProtocol

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
    """
    Потоковый адаптер, делегирующий в общий онлайновый трансформер.
    На вход: закрытая 1m kline (dict из binance_ws).
    На выход: features dict, без утечек во время.
    """
    def __init__(self, cfg: FeatureConfig) -> None:
        spec = FeatureSpec(lookbacks_prices=list(cfg.lookbacks_prices or [5, 15, 60]), rsi_period=int(cfg.rsi_period))
        self._tr = OnlineFeatureTransformer(spec)

    def on_kline(self, kline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sym = str(kline["symbol"]).upper()
        close = float(kline["close"])
        ts_ms = int(kline["close_time"])
        feats = self._tr.update(symbol=sym, ts_ms=ts_ms, close=close)
        return feats
