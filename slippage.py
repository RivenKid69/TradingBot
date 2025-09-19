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
            if isinstance(multipliers_raw, Sequence) and not isinstance(
                multipliers_raw, (str, bytes, bytearray)
            ):
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

        def _first_non_null(*keys: str) -> Any:
            for key in keys:
                if key in d and d[key] is not None:
                    return d[key]
            return None

        alpha_bps_val = _first_non_null("alpha_bps", "alpha")
        beta_coef_val = _first_non_null("beta_coef", "beta")
        vol_metric_val = _first_non_null("vol_metric", "volatility_metric")
        vol_window_val = _first_non_null("vol_window", "volatility_window")

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
class DynamicImpactConfig:
    enabled: bool = False
    beta_vol: float = 0.0
    beta_participation: float = 0.0
    min_k: Optional[float] = None
    max_k: Optional[float] = None
    fallback_k: Optional[float] = None
    vol_metric: Optional[str] = None
    vol_window: Optional[int] = None
    participation_metric: Optional[str] = None
    participation_window: Optional[int] = None
    smoothing_alpha: Optional[float] = None
    zscore_clip: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynamicImpactConfig":
        if not isinstance(d, dict):
            raise TypeError("dynamic impact config must be a dict")

        known_keys = {
            "enabled",
            "beta_vol",
            "beta_participation",
            "min_k",
            "max_k",
            "fallback_k",
            "vol_metric",
            "vol_window",
            "participation_metric",
            "participation_window",
            "smoothing_alpha",
            "zscore_clip",
        }

        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            enabled=bool(d.get("enabled", False)),
            beta_vol=float(d.get("beta_vol", 0.0)),
            beta_participation=float(d.get("beta_participation", 0.0)),
            min_k=float(d["min_k"]) if d.get("min_k") is not None else None,
            max_k=float(d["max_k"]) if d.get("max_k") is not None else None,
            fallback_k=float(d["fallback_k"]) if d.get("fallback_k") is not None else None,
            vol_metric=str(d["vol_metric"]) if d.get("vol_metric") is not None else None,
            vol_window=int(d["vol_window"]) if d.get("vol_window") is not None else None,
            participation_metric=str(d["participation_metric"]) if d.get("participation_metric") is not None else None,
            participation_window=int(d["participation_window"]) if d.get("participation_window") is not None else None,
            smoothing_alpha=float(d["smoothing_alpha"]) if d.get("smoothing_alpha") is not None else None,
            zscore_clip=float(d["zscore_clip"]) if d.get("zscore_clip") is not None else None,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(self.extra)
        data["enabled"] = bool(self.enabled)
        data["beta_vol"] = float(self.beta_vol)
        data["beta_participation"] = float(self.beta_participation)
        if self.min_k is not None:
            data["min_k"] = float(self.min_k)
        if self.max_k is not None:
            data["max_k"] = float(self.max_k)
        if self.fallback_k is not None:
            data["fallback_k"] = float(self.fallback_k)
        if self.vol_metric is not None:
            data["vol_metric"] = str(self.vol_metric)
        if self.vol_window is not None:
            data["vol_window"] = int(self.vol_window)
        if self.participation_metric is not None:
            data["participation_metric"] = str(self.participation_metric)
        if self.participation_window is not None:
            data["participation_window"] = int(self.participation_window)
        if self.smoothing_alpha is not None:
            data["smoothing_alpha"] = float(self.smoothing_alpha)
        if self.zscore_clip is not None:
            data["zscore_clip"] = float(self.zscore_clip)
        return data


@dataclass
class TailShockConfig:
    enabled: bool = False
    probability: float = 0.0
    shock_bps: float = 0.0
    shock_multiplier: float = 1.0
    decay_halflife_bars: Optional[int] = None
    min_multiplier: Optional[float] = None
    max_multiplier: Optional[float] = None
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TailShockConfig":
        if not isinstance(d, dict):
            raise TypeError("tail shock config must be a dict")

        known_keys = {
            "enabled",
            "probability",
            "shock_bps",
            "shock_multiplier",
            "decay_halflife_bars",
            "min_multiplier",
            "max_multiplier",
            "seed",
        }

        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            enabled=bool(d.get("enabled", False)),
            probability=float(d.get("probability", 0.0)),
            shock_bps=float(d.get("shock_bps", 0.0)),
            shock_multiplier=float(d.get("shock_multiplier", 1.0)),
            decay_halflife_bars=int(d["decay_halflife_bars"]) if d.get("decay_halflife_bars") is not None else None,
            min_multiplier=float(d["min_multiplier"]) if d.get("min_multiplier") is not None else None,
            max_multiplier=float(d["max_multiplier"]) if d.get("max_multiplier") is not None else None,
            seed=int(d["seed"]) if d.get("seed") is not None else None,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(self.extra)
        data["enabled"] = bool(self.enabled)
        data["probability"] = float(self.probability)
        data["shock_bps"] = float(self.shock_bps)
        data["shock_multiplier"] = float(self.shock_multiplier)
        if self.decay_halflife_bars is not None:
            data["decay_halflife_bars"] = int(self.decay_halflife_bars)
        if self.min_multiplier is not None:
            data["min_multiplier"] = float(self.min_multiplier)
        if self.max_multiplier is not None:
            data["max_multiplier"] = float(self.max_multiplier)
        if self.seed is not None:
            data["seed"] = int(self.seed)
        return data


@dataclass
class AdvConfig:
    enabled: bool = False
    window_days: int = 30
    smoothing_alpha: Optional[float] = None
    fallback_adv: Optional[float] = None
    min_adv: Optional[float] = None
    max_adv: Optional[float] = None
    seasonality_path: Optional[str] = None
    override_path: Optional[str] = None
    hash: Optional[str] = None
    profile_kind: Optional[str] = None
    multipliers: Optional[tuple[float, ...]] = None
    zscore_clip: Optional[float] = None
    liquidity_buffer: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdvConfig":
        if not isinstance(d, dict):
            raise TypeError("adv config must be a dict")

        multipliers_raw = d.get("multipliers")
        multipliers: Optional[tuple[float, ...]] = None
        if multipliers_raw is not None:
            if isinstance(multipliers_raw, Sequence) and not isinstance(
                multipliers_raw, (str, bytes, bytearray)
            ):
                multipliers = tuple(float(x) for x in multipliers_raw)
            else:
                try:
                    multipliers = (float(multipliers_raw),)
                except (TypeError, ValueError):
                    multipliers = None

        known_keys = {
            "enabled",
            "window_days",
            "smoothing_alpha",
            "fallback_adv",
            "min_adv",
            "max_adv",
            "seasonality_path",
            "override_path",
            "hash",
            "profile_kind",
            "multipliers",
            "zscore_clip",
            "liquidity_buffer",
        }

        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            enabled=bool(d.get("enabled", False)),
            window_days=int(d.get("window_days", 30)),
            smoothing_alpha=float(d["smoothing_alpha"]) if d.get("smoothing_alpha") is not None else None,
            fallback_adv=float(d["fallback_adv"]) if d.get("fallback_adv") is not None else None,
            min_adv=float(d["min_adv"]) if d.get("min_adv") is not None else None,
            max_adv=float(d["max_adv"]) if d.get("max_adv") is not None else None,
            seasonality_path=str(d["seasonality_path"]) if d.get("seasonality_path") is not None else None,
            override_path=str(d["override_path"]) if d.get("override_path") is not None else None,
            hash=str(d["hash"]) if d.get("hash") is not None else None,
            profile_kind=str(d["profile_kind"]) if d.get("profile_kind") is not None else None,
            multipliers=multipliers,
            zscore_clip=float(d["zscore_clip"]) if d.get("zscore_clip") is not None else None,
            liquidity_buffer=float(d.get("liquidity_buffer", 1.0)),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(self.extra)
        data["enabled"] = bool(self.enabled)
        data["window_days"] = int(self.window_days)
        if self.smoothing_alpha is not None:
            data["smoothing_alpha"] = float(self.smoothing_alpha)
        if self.fallback_adv is not None:
            data["fallback_adv"] = float(self.fallback_adv)
        if self.min_adv is not None:
            data["min_adv"] = float(self.min_adv)
        if self.max_adv is not None:
            data["max_adv"] = float(self.max_adv)
        if self.seasonality_path is not None:
            data["seasonality_path"] = str(self.seasonality_path)
        if self.override_path is not None:
            data["override_path"] = str(self.override_path)
        if self.hash is not None:
            data["hash"] = str(self.hash)
        if self.profile_kind is not None:
            data["profile_kind"] = str(self.profile_kind)
        if self.multipliers is not None:
            data["multipliers"] = [float(x) for x in self.multipliers]
        if self.zscore_clip is not None:
            data["zscore_clip"] = float(self.zscore_clip)
        data["liquidity_buffer"] = float(self.liquidity_buffer)
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
    dynamic_impact: DynamicImpactConfig = field(default_factory=DynamicImpactConfig)
    tail_shock: TailShockConfig = field(default_factory=TailShockConfig)
    adv: AdvConfig = field(default_factory=AdvConfig)

    def get_dynamic_block(self) -> Optional[Any]:
        dyn = getattr(self, "dynamic_spread", None)
        if dyn is not None:
            return dyn
        return getattr(self, "dynamic", None)

    def dynamic_trade_cost_enabled(self) -> bool:
        block = self.get_dynamic_block()
        if block is None:
            return False
        if isinstance(block, DynamicSpreadConfig):
            return bool(block.enabled)
        if isinstance(block, dict):
            return bool(block.get("enabled"))
        return bool(getattr(block, "enabled", False))

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

        dynamic_impact_cfg: DynamicImpactConfig
        impact_block = d.get("dynamic_impact")
        if isinstance(impact_block, DynamicImpactConfig):
            dynamic_impact_cfg = impact_block
        elif isinstance(impact_block, dict):
            dynamic_impact_cfg = DynamicImpactConfig.from_dict(impact_block)
        else:
            dynamic_impact_cfg = DynamicImpactConfig()

        tail_shock_cfg: TailShockConfig
        tail_block = d.get("tail_shock")
        if isinstance(tail_block, TailShockConfig):
            tail_shock_cfg = tail_block
        elif isinstance(tail_block, dict):
            tail_shock_cfg = TailShockConfig.from_dict(tail_block)
        else:
            tail_shock_cfg = TailShockConfig()

        adv_cfg: AdvConfig
        adv_block = d.get("adv")
        if isinstance(adv_block, AdvConfig):
            adv_cfg = adv_block
        elif isinstance(adv_block, dict):
            adv_cfg = AdvConfig.from_dict(adv_block)
        else:
            adv_cfg = AdvConfig()

        return cls(
            k=float(d.get("k", 0.8)),
            min_half_spread_bps=float(d.get("min_half_spread_bps", 0.0)),
            default_spread_bps=float(d.get("default_spread_bps", 2.0)),
            eps=float(d.get("eps", 1e-12)),
            dynamic_spread=dynamic_cfg,
            dynamic_impact=dynamic_impact_cfg,
            tail_shock=tail_shock_cfg,
            adv=adv_cfg,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "k": float(self.k),
            "min_half_spread_bps": float(self.min_half_spread_bps),
            "default_spread_bps": float(self.default_spread_bps),
            "eps": float(self.eps),
        }
        if self.dynamic_spread is not None:
            dyn_dict = self.dynamic_spread.to_dict()
            data["dynamic"] = dict(dyn_dict)
            data.setdefault("dynamic_spread", dict(dyn_dict))
        if self.dynamic_impact is not None:
            data["dynamic_impact"] = self.dynamic_impact.to_dict()
        if self.tail_shock is not None:
            data["tail_shock"] = self.tail_shock.to_dict()
        if self.adv is not None:
            data["adv"] = self.adv.to_dict()
        return data

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
    try:
        size_val = float(size)
    except (TypeError, ValueError):
        size_val = 0.0

    delegate = getattr(cfg, "get_trade_cost_bps", None)
    dynamic_enabled = False
    if callable(delegate):
        detector = getattr(cfg, "dynamic_trade_cost_enabled", None)
        if callable(detector):
            try:
                dynamic_enabled = bool(detector())
            except Exception:
                dynamic_enabled = False
        if not dynamic_enabled:
            block: Any = None
            getter = getattr(cfg, "get_dynamic_block", None)
            if callable(getter):
                try:
                    block = getter()
                except Exception:
                    block = None
            if block is None:
                block = getattr(cfg, "dynamic_spread", None)
            if block is None:
                block = getattr(cfg, "dynamic", None)
            if block is not None:
                if isinstance(block, dict):
                    dynamic_enabled = bool(block.get("enabled"))
                else:
                    dynamic_enabled = bool(getattr(block, "enabled", False))
        if dynamic_enabled:
            side = "BUY" if size_val >= 0.0 else "SELL"
            qty = abs(size_val)
            vol_metrics_payload: Optional[Dict[str, float]] = None
            vol_payload: Dict[str, float] = {}
            if vol_factor is not None:
                try:
                    vf_val = float(vol_factor)
                except (TypeError, ValueError):
                    vf_val = None
                else:
                    if math.isfinite(vf_val):
                        vol_payload["vol_factor"] = vf_val
            if liquidity is not None:
                try:
                    liq_val = float(liquidity)
                except (TypeError, ValueError):
                    liq_val = None
                else:
                    if math.isfinite(liq_val):
                        vol_payload["liquidity"] = liq_val
            if vol_payload:
                vol_metrics_payload = vol_payload
            kwargs: Dict[str, Any] = {
                "side": side,
                "qty": qty,
                "mid": None,
                "spread_bps": spread_bps,
                "bar_close_ts": None,
                "order_seq": 0,
            }
            if vol_metrics_payload is not None:
                kwargs["vol_metrics"] = vol_metrics_payload
            try:
                result = delegate(**kwargs)
            except TypeError:
                kwargs.pop("order_seq", None)
                try:
                    result = delegate(**kwargs)
                except TypeError:
                    kwargs.pop("vol_metrics", None)
                    result = delegate(**kwargs)
                except Exception:
                    result = None
            except Exception:
                result = None
            if result is not None:
                try:
                    candidate = float(result)
                except (TypeError, ValueError):
                    candidate = None
                else:
                    if math.isfinite(candidate):
                        return float(candidate)

    sbps = (
        float(spread_bps)
        if (spread_bps is not None and math.isfinite(float(spread_bps)))
        else float(cfg.default_spread_bps)
    )
    half_spread_bps = max(0.5 * sbps, float(cfg.min_half_spread_bps))

    vf = float(vol_factor) if (vol_factor is not None and math.isfinite(float(vol_factor))) else 1.0
    liq = float(liquidity) if (liquidity is not None and float(liquidity) > 0.0 and math.isfinite(float(liquidity))) else 1.0
    sz = abs(size_val)

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
