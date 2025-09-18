# sim/slippage.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Sequence


@dataclass
class DynamicSpreadConfig:
    enabled: bool = False
    profile_kind: Optional[str] = None
    multipliers: Optional[tuple[float, ...]] = None
    path: Optional[str] = None
    override_path: Optional[str] = None
    hash: Optional[str] = None
    alpha_bps: Optional[float] = None
    beta_coef: Optional[float] = None
    min_spread_bps: Optional[float] = None
    max_spread_bps: Optional[float] = None
    smoothing_alpha: Optional[float] = None
    vol_metric: Optional[str] = None
    vol_window: Optional[int] = None
    use_volatility: bool = False
    gamma: Optional[float] = None
    zscore_clip: Optional[float] = None
    refresh_warn_days: Optional[int] = None
    refresh_fail_days: Optional[int] = None
    refresh_on_start: bool = False
    last_refresh_ts: Optional[int] = None
    fallback_spread_bps: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynamicSpreadConfig":
        if not isinstance(d, dict):
            raise TypeError("dynamic spread config must be a dict")

        multipliers_raw = d.get("multipliers")
        multipliers: Optional[tuple[float, ...]] = None
        if multipliers_raw is not None:
            if isinstance(multipliers_raw, Sequence) and not isinstance(multipliers_raw, (str, bytes, bytearray)):
                multipliers = tuple(float(x) for x in multipliers_raw)
            else:
                try:
                    multipliers = (float(multipliers_raw),)
                except (TypeError, ValueError):
                    multipliers = None

        known_keys = {
            "enabled",
            "profile_kind",
            "multipliers",
            "path",
            "override_path",
            "hash",
            "alpha_bps",
            "beta_coef",
            "min_spread_bps",
            "max_spread_bps",
            "smoothing_alpha",
            "vol_metric",
            "vol_window",
            "use_volatility",
            "gamma",
            "zscore_clip",
            "refresh_warn_days",
            "refresh_fail_days",
            "refresh_on_start",
            "last_refresh_ts",
            "fallback_spread_bps",
            # legacy aliases kept for backwards compatibility
            "alpha",
            "beta",
            "volatility_metric",
            "volatility_window",
        }

        extra = {k: v for k, v in d.items() if k not in known_keys}

        alpha_bps_val = d.get("alpha_bps")
        if alpha_bps_val is None:
            alpha_bps_val = d.get("alpha")
        beta_coef_val = d.get("beta_coef")
        if beta_coef_val is None:
            beta_coef_val = d.get("beta")
        vol_metric_val = d.get("vol_metric")
        if vol_metric_val is None:
            vol_metric_val = d.get("volatility_metric")
        vol_window_val = d.get("vol_window")
        if vol_window_val is None:
            vol_window_val = d.get("volatility_window")

        return cls(
            enabled=bool(d.get("enabled", False)),
            profile_kind=str(d["profile_kind"]) if d.get("profile_kind") is not None else None,
            multipliers=multipliers,
            path=str(d["path"]) if d.get("path") is not None else None,
            override_path=str(d["override_path"]) if d.get("override_path") is not None else None,
            hash=str(d["hash"]) if d.get("hash") is not None else None,
            alpha_bps=float(alpha_bps_val) if alpha_bps_val is not None else None,
            beta_coef=float(beta_coef_val) if beta_coef_val is not None else None,
            min_spread_bps=float(d["min_spread_bps"]) if d.get("min_spread_bps") is not None else None,
            max_spread_bps=float(d["max_spread_bps"]) if d.get("max_spread_bps") is not None else None,
            smoothing_alpha=float(d["smoothing_alpha"]) if d.get("smoothing_alpha") is not None else None,
            vol_metric=str(vol_metric_val) if vol_metric_val is not None else None,
            vol_window=int(vol_window_val) if vol_window_val is not None else None,
            use_volatility=bool(d.get("use_volatility", False)),
            gamma=float(d["gamma"]) if d.get("gamma") is not None else None,
            zscore_clip=float(d["zscore_clip"]) if d.get("zscore_clip") is not None else None,
            refresh_warn_days=int(d["refresh_warn_days"]) if d.get("refresh_warn_days") is not None else None,
            refresh_fail_days=int(d["refresh_fail_days"]) if d.get("refresh_fail_days") is not None else None,
            refresh_on_start=bool(d.get("refresh_on_start", False)),
            last_refresh_ts=int(d["last_refresh_ts"]) if d.get("last_refresh_ts") is not None else None,
            fallback_spread_bps=float(d["fallback_spread_bps"]) if d.get("fallback_spread_bps") is not None else None,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(self.extra)
        data["enabled"] = bool(self.enabled)
        if self.profile_kind is not None:
            data["profile_kind"] = str(self.profile_kind)
        if self.multipliers is not None:
            data["multipliers"] = [float(x) for x in self.multipliers]
        if self.path is not None:
            data["path"] = str(self.path)
        if self.override_path is not None:
            data["override_path"] = str(self.override_path)
        if self.hash is not None:
            data["hash"] = str(self.hash)
        if self.alpha_bps is not None:
            data["alpha_bps"] = float(self.alpha_bps)
        if self.beta_coef is not None:
            data["beta_coef"] = float(self.beta_coef)
        if self.min_spread_bps is not None:
            data["min_spread_bps"] = float(self.min_spread_bps)
        if self.max_spread_bps is not None:
            data["max_spread_bps"] = float(self.max_spread_bps)
        if self.smoothing_alpha is not None:
            data["smoothing_alpha"] = float(self.smoothing_alpha)
        if self.vol_metric is not None:
            data["vol_metric"] = str(self.vol_metric)
        if self.vol_window is not None:
            data["vol_window"] = int(self.vol_window)
        data["use_volatility"] = bool(self.use_volatility)
        if self.gamma is not None:
            data["gamma"] = float(self.gamma)
        if self.zscore_clip is not None:
            data["zscore_clip"] = float(self.zscore_clip)
        if self.refresh_warn_days is not None:
            data["refresh_warn_days"] = int(self.refresh_warn_days)
        if self.refresh_fail_days is not None:
            data["refresh_fail_days"] = int(self.refresh_fail_days)
        data["refresh_on_start"] = bool(self.refresh_on_start)
        if self.last_refresh_ts is not None:
            data["last_refresh_ts"] = int(self.last_refresh_ts)
        if self.fallback_spread_bps is not None:
            data["fallback_spread_bps"] = float(self.fallback_spread_bps)
        return data


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
    dynamic_spread: Optional[DynamicSpreadConfig] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SlippageConfig":
        dynamic_cfg: Optional[DynamicSpreadConfig] = None
        for key in ("dynamic", "dynamic_spread"):
            block = d.get(key)
            if isinstance(block, dict):
                dynamic_cfg = DynamicSpreadConfig.from_dict(block)
                break
            if isinstance(block, DynamicSpreadConfig):
                dynamic_cfg = block
                break

        return cls(
            k=float(d.get("k", 0.8)),
            min_half_spread_bps=float(d.get("min_half_spread_bps", 0.0)),
            default_spread_bps=float(d.get("default_spread_bps", 2.0)),
            eps=float(d.get("eps", 1e-12)),
            dynamic_spread=dynamic_cfg,
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
