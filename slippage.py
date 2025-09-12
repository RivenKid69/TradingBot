# sim/slippage.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence


@dataclass
class SlippageConfig:
    """
    Конфиг слиппеджа «среднего уровня реализма» для среднечастотного бота.
    Формула (в bps):
        slippage_bps = half_spread_bps + k * vol_factor * sqrt( max(size, eps) / max(liquidity, eps) )
    где:
      - half_spread_bps = max(spread_bps * 0.5, min_half_spread_bps)
      - vol_factor: ATR/σ/др. масштаб волатильности, нормированный (например, ATR% за бар)
      - size: абсолютное торгуемое количество (в базовой валюте, штуках)
      - liquidity: прокси ликвидности (например, rolling_volume_shares или ADV в штуках)
    """
    k: float = 0.8
    min_half_spread_bps: float = 0.0
    default_spread_bps: float = 2.0
    eps: float = 1e-12

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SlippageConfig":
        return cls(
            k=float(d.get("k", 0.8)),
            min_half_spread_bps=float(d.get("min_half_spread_bps", 0.0)),
            default_spread_bps=float(d.get("default_spread_bps", 2.0)),
            eps=float(d.get("eps", 1e-12)),
        )

    @classmethod
    def from_file(cls, path: str) -> "SlippageConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):  # pragma: no cover - sanity check
            raise ValueError("slippage config file must contain a JSON object")
        return cls.from_dict(data)


def estimate_slippage_bps(
    *,
    spread_bps: Optional[float],
    size: float,
    liquidity: Optional[float],
    vol_factor: Optional[float],
    cfg: SlippageConfig,
) -> float:
    """
    Оценка слиппеджа в bps по простой калибруемой формуле.
    Если spread_bps/liquidity/vol_factor отсутствуют — используются дефолты/единицы.
    """
    sbps = float(spread_bps) if (spread_bps is not None and math.isfinite(float(spread_bps))) else float(cfg.default_spread_bps)
    half_spread_bps = max(0.5 * sbps, float(cfg.min_half_spread_bps))

    vf = float(vol_factor) if (vol_factor is not None and math.isfinite(float(vol_factor))) else 1.0
    liq = float(liquidity) if (liquidity is not None and float(liquidity) > 0.0 and math.isfinite(float(liquidity))) else 1.0
    sz = abs(float(size))

    impact_term = float(cfg.k) * vf * math.sqrt(max(sz, cfg.eps) / max(liq, cfg.eps))
    return float(half_spread_bps + impact_term)


def apply_slippage_price(*, side: str, quote_price: float, slippage_bps: float) -> float:
    """
    Применить слиппедж к котировке:
      - для BUY цена ухудшается (увеличивается)
      - для SELL цена ухудшается (уменьшается)
    """
    q = float(quote_price)
    bps = float(slippage_bps) / 1e4
    if str(side).upper() == "BUY":
        return float(q * (1.0 + bps))
    else:
        return float(q * (1.0 - bps))


def compute_spread_bps_from_quotes(*, bid: Optional[float], ask: Optional[float], cfg: SlippageConfig) -> float:
    """
    Рассчитать spread_bps из котировок. Если данных нет — вернуть cfg.default_spread_bps.
    """
    if bid is None or ask is None:
        return float(cfg.default_spread_bps)
    b = float(bid)
    a = float(ask)
    if not (math.isfinite(b) and math.isfinite(a)) or a <= 0.0:
        return float(cfg.default_spread_bps)
    return float((a - b) / a * 1e4)


def mid_from_quotes(*, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    b = float(bid)
    a = float(ask)
    if not (math.isfinite(b) and math.isfinite(a)):
        return None
    return float((a + b) / 2.0)


def model_curve(
    participations: Sequence[float],
    *,
    cfg: SlippageConfig,
    spread_bps: float,
    vol_factor: float = 1.0,
) -> list[float]:
    """Return expected slippage for a range of participation rates.

    ``participations`` are interpreted as size/liquidity ratios.  The model is
    evaluated with ``liquidity=1`` and ``size=participation`` for each value.
    """

    out = []
    for p in participations:
        s = estimate_slippage_bps(
            spread_bps=spread_bps,
            size=float(p),
            liquidity=1.0,
            vol_factor=vol_factor,
            cfg=cfg,
        )
        out.append(float(s))
    return out
