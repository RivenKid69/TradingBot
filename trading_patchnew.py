from __future__ import annotations
try:
    import infra_shim  # noqa: F401
except Exception:
    pass
"""
TradingEnv – Phase 11
Fully modern pipeline (Dict action‑space). Legacy box/array paths removed.
"""
import os
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Sequence, Tuple
import types
from core_constants import PRICE_SCALE

try:  # Prefer gymnasium but gracefully fall back to classic gym.
    import gymnasium as gym
    from gymnasium import spaces as spaces
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
    try:
        import gym  # type: ignore[no-redef]
        from gym import spaces as spaces  # type: ignore[assignment]
    except ModuleNotFoundError:  # pragma: no cover - fallback stub for tests
        class _BaseSpace:
            def sample(self) -> Any:
                raise NotImplementedError

        class _Discrete(_BaseSpace):
            def __init__(self, n: int):
                self.n = int(n)

        class _Box(_BaseSpace):
            def __init__(self, low: Any, high: Any, shape: Tuple[int, ...] | None = None, dtype: Any | None = None):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _Dict(_BaseSpace, dict):
            def __init__(self, mapping: dict[str, _BaseSpace]):
                dict.__init__(self, mapping)
                self.mapping = mapping

        class _Env:
            metadata: dict[str, Any] = {}
            observation_space: _BaseSpace | None = None
            action_space: _BaseSpace | None = None

            def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
                return None, {}

            def step(self, action: Any):
                raise NotImplementedError

        spaces = types.SimpleNamespace(Box=_Box, Discrete=_Discrete, Dict=_Dict)
        gym = types.SimpleNamespace(Env=_Env, spaces=spaces)

import numpy as np
import pandas as pd
from event_bus import Topics
from utils.time_provider import TimeProvider, RealTimeProvider
from leakguard import LeakGuard, LeakConfig
from no_trade import (
    NoTradeConfig,
    _parse_daily_windows_min,
    _in_daily_window,
    _in_funding_buffer,
    _in_custom_window,
)

logger = logging.getLogger(__name__)
from no_trade_config import get_no_trade_config
from utils.time import hour_of_week as _hour_of_week, HOURS_IN_WEEK
from utils_time import load_hourly_seasonality

try:  # existing dynamic-spread config (pydantic model)
    from trainingtcost import DynSpreadCfg
except Exception:  # pragma: no cover - fallback to simple dataclass
    @dataclass
    class DynSpreadCfg:
        base_bps: float = 3.0
        alpha_vol: float = 0.5
        beta_illiquidity: float = 1.0
        liq_ref: float = 1000.0
        min_bps: float = 1.0
        max_bps: float = 25.0

# --- auto‑import compiled C++ simulator ---
try:
    from fast_market import MarketSimulatorWrapper, CyMicrostructureGenerator
    _HAVE_FAST_MARKET = True
except ImportError:
    _HAVE_FAST_MARKET = False
# --- runtime switch for mini/full core ---
try:
    from runtime_flags import USE_MINI_CORE  # project-local
except Exception:
    import os as _os
    def _to_bool(v: object) -> bool:
        return str(v).strip().lower() in {"1", "true", "yes", "on"}
    USE_MINI_CORE = _to_bool(_os.environ.get("USE_MINI_CORE", "0"))

if USE_MINI_CORE:
    _HAVE_FAST_MARKET = False  # принудительно mini-режим


# --- unified MarketRegime enum (try C++ version first) ---
try:
    from core_constants import MarketRegime  # C++ enum
except ImportError:
    # -----------------------------------------------------------------
    # MarketRegime enum – single source of truth
    # -----------------------------------------------------------------
    try:
        from cy_constants import MarketRegime  # из C++ хедера через Cython
    except Exception:
        try:
            from core_constants import MarketRegime  # старый путь, если cy_constants не собран
        except Exception:
            class MarketRegime(int):
                NORMAL        = 0
                CHOPPY_FLAT   = 1
                STRONG_TREND  = 2
                ILLIQUID      = 3
                # aliases for backward compatibility
                FLAT          = CHOPPY_FLAT
                TREND         = STRONG_TREND
                _COUNT        = 4

                def __new__(cls, value: int):
                    if not 0 <= value < cls._COUNT:
                        raise ValueError("invalid MarketRegime")
                    return int.__new__(cls, int(value))
    # PATCH‑ID:P15_TENV_enum
from action_proto import ActionProto, ActionType
from mediator import Mediator


class DecisionTiming(IntEnum):
    CLOSE_TO_OPEN = 0
    INTRA_HOUR_WITH_LATENCY = 1


def _dynamic_spread_bps(vol_factor: float, liquidity: float, cfg: DynSpreadCfg) -> float:
    """Compute dynamic spread in basis points.

    Parameters
    ----------
    vol_factor : float
        Volatility factor, dimensionless (e.g. ATR percentage).
    liquidity : float
        Rolling liquidity measure.
    cfg : DynSpreadCfg
        Configuration for dynamic spread parameters.

    Returns
    -------
    float
        Clamped spread in basis points.
    """
    ratio = 0.0
    if getattr(cfg, "liq_ref", 0) > 0 and liquidity == liquidity:  # NaN check
        ratio = max(0.0, (float(cfg.liq_ref) - float(liquidity)) / float(cfg.liq_ref))
    spread_bps = (
        float(cfg.base_bps)
        + float(cfg.alpha_vol) * float(vol_factor) * 10000.0
        + float(cfg.beta_illiquidity) * ratio * float(cfg.base_bps)
    )
    return float(max(float(cfg.min_bps), min(float(cfg.max_bps), spread_bps)))


class _AgentOrders(set):
    def count(self) -> int:  # noqa: D401
        return len(self)


@dataclass(slots=True)
class _EnvState:
    cash: float
    units: float
    net_worth: float
    step_idx: int
    peak_value: float
    agent_orders: _AgentOrders
    max_position: float
    max_position_risk_on: float
    is_bankrupt: bool = False

# ------------------------------- Environment -------------------------------
class TradingEnv(gym.Env):
    # -----------------------------------------------------------------
    # External control: force regime for adversarial evaluation
    # -----------------------------------------------------------------
    def set_market_regime(self, regime: str = "normal", duration: int = 0):
        """Proxy to MarketSimulator.force_market_regime()."""
        if not _HAVE_FAST_MARKET:
            print("⚠️  fast_market not available – regime control ignored")
            return
        mapping = {
            "normal":       MarketRegime.NORMAL,
            "choppy_flat":  MarketRegime.FLAT,
            "flat":         MarketRegime.FLAT,
            "strong_trend": MarketRegime.TREND,
            "trend":        MarketRegime.TREND,
            "liquidity_shock": MarketRegime.ILLIQUID,
            "illiquid":     MarketRegime.ILLIQUID,
        }
        # duration=0 → действовать до конца эпизода
        self.market_sim.force_market_regime(
            mapping.get(regime, MarketRegime.NORMAL),
            self.state.step_idx if self.state else 0,
            duration,
        8)
    metadata = {"render.modes": []}


    # ------------------------------------------------ ctor
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        seed: int | None = None,
        initial_cash: float = 1_000.0,
        latency_steps: int | None = None,
        slip_k: float | None = None,
        # risk parameters
        max_abs_position: float | None = None,
        max_drawdown_pct: float | None = None,
        bankruptcy_cash_th: float | None = None,
        # regime / shock
        regime_dist: Sequence[float] | None = None,
        enable_shocks: bool = False,
        flash_prob: float = 0.01,
        rng: np.random.Generator | None = None,
        validate_data: bool = False,
        event_bus: Any | None = None,
        leak_guard: LeakGuard | None = None,
        no_trade_cfg: NoTradeConfig | None = None,
        time_provider: TimeProvider | None = None,
        decision_mode: DecisionTiming = DecisionTiming.CLOSE_TO_OPEN,
        decision_delay_ms: int = 0,
        liquidity_seasonality_path: str | None = None,
        liquidity_seasonality_hash: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # store seed and initialize per-instance RNG
        self.seed_value = seed
        rank_offset = getattr(self, "rank", 0)
        base_seed = seed or 0
        self._rng: np.random.Generator = rng or np.random.default_rng(base_seed + rank_offset)

        # event bus / publisher
        self._bus = event_bus
        self._publish = getattr(self._bus, "publish", lambda *a, **k: None)
        self.time = time_provider or RealTimeProvider()
        self.decision_mode = decision_mode
        # action scheduled for next bar when using delayed decisions
        self._pending_action: ActionProto | None = None
        self._action_queue: deque[ActionProto] = deque()
        self._leak_guard = leak_guard or LeakGuard(LeakConfig(decision_delay_ms=int(decision_delay_ms)))
        # price data
        self.df = df.reset_index(drop=True).copy()
        if "ts_ms" in self.df.columns and "decision_ts" not in self.df.columns:
            self.df = self._leak_guard.attach_decision_time(self.df, ts_col="ts_ms")
        if "close_orig" in self.df.columns:
            self._close_actual = self.df["close_orig"].copy()
        elif "close" in self.df.columns:
            self._close_actual = self.df["close"].copy()
            self.df["close"] = self.df["close"].shift(1)
        else:
            self._close_actual = pd.Series(dtype="float64")

        # --- bar interval & intrabar path metadata ---
        self._bar_interval_columns = [
            col
            for col in (
                "bar_interval_ms",
                "bar_timeframe_ms",
                "interval_ms",
                "bar_duration_ms",
                "timeframe_ms",
                "step_ms",
            )
            if col in self.df.columns
        ]
        self.bar_interval_ms: int | None = self._infer_bar_interval_from_dataframe()
        self._exec_intrabar_timeframe_configured = False
        self._intrabar_path_columns = self._detect_intrabar_path_columns()
        self._exec_intrabar_path_method: str | bool | None = None

        # --- load liquidity seasonality coefficients ---
        liq_path = liquidity_seasonality_path or kwargs.get(
            "liquidity_seasonality_path", "configs/liquidity_seasonality.json"
        )
        self._liq_seasonality = load_hourly_seasonality(
            liq_path,
            "liquidity",
            expected_hash=liquidity_seasonality_hash,
        )
        if self._liq_seasonality is None:
            logger.warning(
                "Liquidity seasonality config %s not found or invalid; using default multipliers of 1.0; "
                "run scripts/build_hourly_seasonality.py to generate them.",
                liq_path,
            )
            self._liq_seasonality = np.ones(HOURS_IN_WEEK, dtype=float)

        # --- precompute ATR-based volatility and rolling liquidity ---
        close_col = "close" if "close" in self.df.columns else "price"
        high = self.df.get("high")
        low = self.df.get("low")
        if high is not None and low is not None and close_col in self.df.columns:
            prev_close = self.df[close_col]
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            self.df["tr"] = tr
            atr_window = int(kwargs.get("atr_window", 14))
            self.df["atr"] = tr.ewm(alpha=1 / atr_window, adjust=False).mean()
            denom = prev_close.replace(0, np.nan)
            self.df["atr_pct"] = (self.df["atr"] / denom).ffill().fillna(0.0)
        else:
            self.df["tr"] = 0.0
            self.df["atr"] = 0.0
            self.df["atr_pct"] = 0.0

        # dynamic spread config and rolling liquidity
        dyn_cfg_dict = dict(kwargs.get("dynamic_spread", {}) or {})
        self._dyn_cfg = DynSpreadCfg(**dyn_cfg_dict)

        self._liq_col = next(
            (c for c in ["quote_asset_volume", "quote_volume", "volume"] if c in self.df.columns),
            None,
        )
        self._liq_window_h = float(kwargs.get("liquidity_window_hours", 1.0))
        if self._liq_col is not None:
            if "ts_ms" in self.df.columns and len(self.df) > 1:
                step_ms = float(self.df["ts_ms"].iloc[1] - self.df["ts_ms"].iloc[0])
                bars_per_hour = int(3600000 / step_ms) if step_ms > 0 else 1
            else:
                bars_per_hour = 60
            win = max(1, int(self._liq_window_h * bars_per_hour))
            self.df["liq_roll"] = (
                self.df[self._liq_col]
                .rolling(win, min_periods=1)
                .sum()
                .ffill()
                .fillna(0.0)
            )
        else:
            self.df["liq_roll"] = 0.0

        if "ts_ms" in self.df.columns:
            how = _hour_of_week(self.df["ts_ms"].to_numpy())
            self.df["hour_of_week"] = how
            self.df["liq_roll"] = self.df["liq_roll"] * self._liq_seasonality[how]

        self._rolling_liquidity = self.df["liq_roll"].to_numpy()

        # --- precompute no-trade mask ---
        override = kwargs.get("no_trade")
        if no_trade_cfg is not None:
            cfg_nt = no_trade_cfg
        elif override:
            cfg_nt = NoTradeConfig(**override)
        else:
            sandbox_path = kwargs.get("sandbox_config", "configs/legacy_sandbox.yaml")
            try:
                cfg_nt = get_no_trade_config(sandbox_path)
            except FileNotFoundError:
                logger.warning(
                    "Sandbox config %s not found; using default no-trade settings", sandbox_path
                )
                cfg_nt = NoTradeConfig()
        self._no_trade_cfg = cfg_nt
        if "ts_ms" in self.df.columns:
            ts = (
                pd.to_numeric(self.df["ts_ms"], errors="coerce")
                .astype("Int64")
                .to_numpy(dtype="int64")
            )
            daily_min = _parse_daily_windows_min(cfg_nt.daily_utc or [])
            m_daily = _in_daily_window(ts, daily_min)
            m_funding = _in_funding_buffer(ts, int(cfg_nt.funding_buffer_min or 0))
            m_custom = _in_custom_window(ts, cfg_nt.custom_ms or [])
            self._no_trade_mask = m_daily | m_funding | m_custom
        else:
            self._no_trade_mask = np.zeros(len(self.df), dtype=bool)
        self.no_trade_blocks = 0
        self.total_steps = 0
        self.no_trade_block_ratio = float(self._no_trade_mask.mean()) if len(self._no_trade_mask) else 0.0

        self.last_bid: float | None = None
        self.last_ask: float | None = None
        self.last_mid: float | None = None
        self.last_mtm_price: float | None = None

        # reward / cost toggles (keep backward-compatible defaults)
        self.turnover_penalty_coef = float(kwargs.get("turnover_penalty_coef", 0.0) or 0.0)
        self._turnover_total = 0.0

        # optional strict data validation
        if validate_data or os.getenv("DATA_VALIDATE") == "1":
            try:
                from data_validation import DataValidator
                DataValidator().validate(self.df)
                import time as _time
                self._publish(Topics.RISK, {
                    "step": 0,
                    "ts": int(_time.time()),
                    "reason": "data_validation_ok",
                    "details": {"rows": int(len(self.df))},
                })
            except Exception as e:
                # лог + немедленный fail: некондиционные данные нам не нужны
                import time as _time
                self._publish(Topics.RISK, {
                    "step": 0,
                    "ts": int(_time.time()),
                    "reason": "data_validation_fail",
                    "details": {"error": str(e)},
                })
                raise

        self.initial_cash = float(initial_cash)
        self._max_steps = len(self.df)

        # store config for Mediator
        self.latency_steps = int(latency_steps or 0)
        if self.latency_steps < 0 or self.latency_steps > self._max_steps:
            raise ValueError("latency_steps out of range")
        self.slip_k = slip_k or 0.0
        self.max_abs_position = max_abs_position or 1e12
        self.max_drawdown_pct = max_drawdown_pct or 1.0
        self.bankruptcy_cash_th = bankruptcy_cash_th or -1e12

        # spaces
        self.action_space = spaces.Dict(
            {
                "type": spaces.Discrete(4),
                "price_offset_ticks": spaces.Discrete(201),
                "volume_frac": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "ttl_steps": spaces.Discrete(33),
            }
        )
        # --- patched: dynamic N_FEATURES shim ---
        try:
            import lob_state_cython as _lob_module  # type: ignore[import-not-found]
        except Exception:
            _lob_module = None  # type: ignore[assignment]

        _lob_nf = None
        if _lob_module is not None:
            _lob_nf = getattr(_lob_module, "_compute_n_features", None)

        if callable(_lob_nf):
            N_FEATURES = int(_lob_nf())
        elif _lob_module is not None and hasattr(_lob_module, "N_FEATURES"):
            try:
                N_FEATURES = int(getattr(_lob_module, "N_FEATURES"))
            except Exception as exc:
                raise ImportError(
                    "Cannot determine N_FEATURES from lob_state_cython.N_FEATURES"
                ) from exc
        else:
            # Фолбэк: посчитать через obs_builder при наличии layout
            try:
                import obs_builder as _ob
                from feature_config import FEATURES_LAYOUT as _OBS_LAYOUT  # если в проекте иначе — подставь актуальное имя
                N_FEATURES = int(_ob.compute_n_features(_OBS_LAYOUT))
            except Exception as _e:
                raise ImportError(
                    "Cannot determine N_FEATURES: build Cython (lob_state_cython) "
                    "or expose OBS_LAYOUT for obs_builder.compute_n_features(layout)."
                ) from _e
        # --- end patch ---

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES+2,), dtype=np.float32
        )

        # Phase 09 regime machinery
        self._init_regime_dist = regime_dist
        self._init_enable_shocks = bool(enable_shocks)
        self._init_flash_prob = float(flash_prob)

        # attach minimal market simulator stub
        if _HAVE_FAST_MARKET:
            # --- allocate contiguous arrays (C‑contiguous float32) ---
            price_arr   = self.df["price"].to_numpy(dtype="float32", copy=True)
            open_arr    = self.df["open" ].to_numpy(dtype="float32", copy=True)
            high_arr    = self.df["high" ].to_numpy(dtype="float32", copy=True)
            low_arr     = self.df["low"  ].to_numpy(dtype="float32", copy=True)
            vol_usd_arr = self.df["quote_asset_volume"].to_numpy(dtype="float32", copy=True)

            pid = 0
            try:
                pid = int(os.getpid())
            except Exception:
                pid = 0
            base_seed = int(self.seed_value or 0)
            sim_seed = base_seed ^ pid if pid else base_seed

            self.market_sim = MarketSimulatorWrapper(
                price_arr,
                open_arr,
                high_arr,
                low_arr,
                vol_usd_arr,
                seed=int(sim_seed & 0xFFFFFFFFFFFFFFFF),
            )
            # создаём генератор микроструктурных событий
            self.flow_gen = CyMicrostructureGenerator(
                momentum_factor=0.3, mean_reversion_factor=0.5,
                adversarial_factor=0.6
            )
            try:
                # уникальный seed: исходный seed XOR PID
                self.flow_gen.set_seed(int(sim_seed))
            except Exception:
                pass  # fallback, если метода нет
        else:
            # fallback: используем Python‑stub для совместимости
            from trading_patchnew import _SimpleMarketSim as SMS  # self‑import safe
            self.market_sim = SMS(self._rng)
            self.flow_gen = None
        from trading_patchnew import MarketRegime  # self‑import safe


        # runtime state / orchestrator
        self.state: _EnvState | None = None
        self._mediator = Mediator(self)
        if not hasattr(self._mediator, "calls"):
            self._mediator.calls = []
        self._maybe_configure_exec_timeframe()

    # ------------------------------------------------ helpers
    def _init_state(self) -> Tuple[np.ndarray, dict]:
        self.total_steps = 0
        self.no_trade_blocks = 0
        self.state = _EnvState(
            cash=self.initial_cash,
            units=0.0,
            net_worth=self.initial_cash,
            step_idx=0,
            peak_value=self.initial_cash,
            agent_orders=_AgentOrders(),
            max_position=1.0,
            max_position_risk_on=1.0,
        )
        self._turnover_total = 0.0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def _to_proto(self, action) -> ActionProto:
        if isinstance(action, ActionProto):
            return action
        if isinstance(action, dict):
            from domain.adapters import gym_to_action_v1, action_v1_to_proto
            v1 = gym_to_action_v1(action)
            return action_v1_to_proto(v1)
        if isinstance(action, np.ndarray):
            # старые массивы не поддерживаем (используй legacy_bridge.from_legacy_box вручную)
            raise TypeError("NumPy array actions are no longer supported")
        raise TypeError("Unsupported action type")

    def _assert_feature_timestamps(self, row: pd.Series) -> None:
        decision_ts = row.get("decision_ts")
        if pd.isna(decision_ts):
            return
        dec_ts = int(decision_ts)
        for col in row.index:
            if col.endswith("_ts"):
                val = row[col]
                if pd.notna(val) and int(val) > dec_ts:
                    raise AssertionError(f"{col}={int(val)} > decision_ts={dec_ts}")


    def _infer_bar_interval_from_dataframe(self) -> int | None:
        """Infer bar interval in milliseconds using explicit columns or timestamp diffs."""

        for col in self._bar_interval_columns:
            series = pd.to_numeric(self.df[col], errors="coerce")
            for value in series:
                if pd.isna(value):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric) or numeric <= 0:
                    continue
                return int(round(numeric))

        ts_candidates = [
            "decision_ts",
            "ts_ms",
            "close_ts",
            "open_ts",
        ]
        for col in ts_candidates:
            if col not in self.df.columns:
                continue
            series = pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype="float64")
            if series.size < 2:
                continue
            mask = np.isfinite(series)
            if mask.sum() < 2:
                continue
            finite_vals = series[mask]
            diffs = np.diff(finite_vals)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if not diffs.size:
                continue
            median = float(np.median(diffs))
            if median <= 0:
                continue
            is_ms = "ms" in col.lower()
            if not is_ms:
                median *= 1000.0
            return int(round(median))
        return None

    def _detect_intrabar_path_columns(self) -> list[str]:
        cols: list[str] = []
        for col in self.df.columns:
            if not isinstance(col, str):
                continue
            lower = col.lower()
            if any(key in lower for key in ("intrabar_path", "intrabar_points")):
                cols.append(col)
                continue
            if "m1" in lower and any(token in lower for token in ("path", "points", "series")):
                cols.append(col)
        # Preserve order while removing duplicates
        seen: set[str] = set()
        unique: list[str] = []
        for name in cols:
            if name in seen:
                continue
            seen.add(name)
            unique.append(name)
        return unique

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple, dict, set)):
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    def _diff_to_ms(self, start: Any, end: Any, col_name: str) -> int | None:
        try:
            if pd.isna(start) or pd.isna(end):
                return None
        except Exception:
            pass
        try:
            diff = float(end) - float(start)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(diff) or diff <= 0:
            return None
        if "ms" not in col_name.lower():
            diff *= 1000.0
        return int(round(diff))

    def _bar_interval_from_row(self, row: pd.Series) -> int | None:
        for col in self._bar_interval_columns:
            if col not in row.index:
                continue
            candidate = self._safe_float(row.get(col))
            if candidate is None or candidate <= 0:
                continue
            return int(round(candidate))
        return None

    def _bar_interval_from_index(self, idx: int) -> int | None:
        ts_cols = ["decision_ts", "ts_ms", "close_ts", "open_ts"]
        for col in ts_cols:
            if col not in self.df.columns:
                continue
            series = pd.to_numeric(self.df[col], errors="coerce")
            if idx > 0:
                diff = self._diff_to_ms(series.iloc[idx - 1], series.iloc[idx], col)
                if diff:
                    return diff
            if idx + 1 < len(series):
                diff = self._diff_to_ms(series.iloc[idx], series.iloc[idx + 1], col)
                if diff:
                    return diff
        return None

    def _update_bar_interval(self, row: pd.Series, idx: int) -> None:
        candidate = self._bar_interval_from_row(row)
        if candidate is None:
            candidate = self._bar_interval_from_index(idx)
        if candidate is not None and candidate > 0:
            self.bar_interval_ms = int(candidate)

    def _coerce_timestamp(self, value: Any, is_ms: bool | None, column_name: str) -> int | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        if is_ms is None:
            is_ms = "ms" in column_name.lower()
        if not is_ms:
            numeric *= 1000.0
        return int(round(numeric))

    def _resolve_snapshot_timestamp(self, row: pd.Series) -> int:
        for col, is_ms in (
            ("decision_ts", True),
            ("ts_ms", True),
            ("close_ts", None),
            ("open_ts", None),
            ("timestamp_ms", True),
            ("timestamp", False),
            ("ts", False),
        ):
            if col not in row.index:
                continue
            ts = self._coerce_timestamp(row.get(col), is_ms, col)
            if ts is not None:
                return ts
        time_fn = getattr(self.time, "time_ms", None)
        if callable(time_fn):
            try:
                return int(time_fn())
            except Exception:
                pass
        return int(time.time() * 1000)

    def _normalize_intrabar_path_payload(self, payload: Any) -> list[Any] | None:
        if payload is None:
            return None
        if isinstance(payload, bytes):
            try:
                payload = payload.decode("utf-8")
            except Exception:
                return None
        if isinstance(payload, str):
            data = payload.strip()
            if not data:
                return None
            try:
                payload = json.loads(data)
            except Exception:
                return None
        if isinstance(payload, pd.Series):
            payload = payload.tolist()
        elif isinstance(payload, np.ndarray):
            payload = payload.tolist()
        elif hasattr(payload, "tolist") and not isinstance(payload, (str, bytes)):
            try:
                payload = payload.tolist()
            except Exception:
                pass
        if isinstance(payload, dict):
            for key in ("points", "path", "prices", "data"):
                if key in payload:
                    return self._normalize_intrabar_path_payload(payload.get(key))
            return None
        if isinstance(payload, (list, tuple)):
            normalized: list[Any] = []
            for item in payload:
                if item is None:
                    continue
                if isinstance(item, float) and not math.isfinite(item):
                    continue
                normalized.append(item)
            return normalized if normalized else None
        return None

    def _maybe_forward_intrabar_path(self, exec_sim: Any, row: pd.Series) -> None:
        if not exec_sim or not self._intrabar_path_columns:
            return
        path_payload: list[Any] | None = None
        for col in self._intrabar_path_columns:
            if col not in row.index:
                continue
            value = row.get(col)
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            path_payload = self._normalize_intrabar_path_payload(value)
            if path_payload:
                break
        if not path_payload:
            return

        method_name = getattr(self, "_exec_intrabar_path_method", None)
        method = None
        if isinstance(method_name, str):
            method = getattr(exec_sim, method_name, None)
            if not callable(method):
                method = None
                self._exec_intrabar_path_method = None
        elif method_name is False:
            return
        if method is None:
            for cand in (
                "set_intrabar_reference_path",
                "set_intrabar_reference_points",
                "set_intrabar_price_path",
                "set_intrabar_m1_points",
                "set_intrabar_path_points",
                "set_intrabar_path",
            ):
                fn = getattr(exec_sim, cand, None)
                if callable(fn):
                    self._exec_intrabar_path_method = cand
                    method = fn
                    break
            else:
                self._exec_intrabar_path_method = False
                return
        if not callable(method):
            return
        try:
            method(path_payload)
        except Exception:
            logger.debug(
                "Failed to forward intrabar path via %s", self._exec_intrabar_path_method, exc_info=True
            )
            self._exec_intrabar_path_method = False

    def _maybe_configure_exec_timeframe(self) -> None:
        if self._exec_intrabar_timeframe_configured:
            return
        if self.bar_interval_ms is None:
            return
        exec_sim = getattr(self._mediator, "exec", None)
        if exec_sim is None or not hasattr(exec_sim, "set_intrabar_timeframe_ms"):
            self._exec_intrabar_timeframe_configured = True
            return
        try:
            current = getattr(exec_sim, "_intrabar_timeframe_ms", None)
        except Exception:
            current = None
        should_set = True
        if current is not None:
            try:
                should_set = int(current) <= 0
            except (TypeError, ValueError):
                should_set = True
        if should_set:
            try:
                exec_sim.set_intrabar_timeframe_ms(int(self.bar_interval_ms))
            except Exception:
                pass
        self._exec_intrabar_timeframe_configured = True

    def _extract_trade_price(self, row: pd.Series) -> float | None:
        for key in (
            "trade_price",
            "last_price",
            "agg_trade_price",
            "trade_px",
        ):
            if key not in row.index:
                continue
            val = self._safe_float(row.get(key))
            if val is not None:
                return val
        return None

    def _extract_trade_qty(self, row: pd.Series) -> float | None:
        for key in (
            "trade_qty",
            "trade_volume",
            "last_trade_qty",
            "agg_trade_volume",
            "agg_trade_qty",
        ):
            if key not in row.index:
                continue
            val = self._safe_float(row.get(key))
            if val is not None:
                return val
        return None



    # ------------------------------------------------ Gym API
    def reset(self, *args, **kwargs):
        obs, info = self._init_state()

        # prepare regime & shocks
        p_vec = (
            np.asarray(self._init_regime_dist, dtype=float)
            if self._init_regime_dist is not None
            else np.asarray([0.8, 0.05, 0.10, 0.05], dtype=float)
        )
        self.market_sim.set_regime_distribution(p_vec)
        self.market_sim.enable_random_shocks(self._init_enable_shocks, self._init_flash_prob)
        regime_idx = self._rng.choice(4, p=self.market_sim.regime_distribution)
        self.market_sim.force_market_regime(MarketRegime(regime_idx))

        # mediator internal clear
        self._mediator.reset()

        # queue default action for delayed execution
        if self.decision_mode == DecisionTiming.CLOSE_TO_OPEN:
            # first action is deferred to the next bar, so execute HOLD on bar 0
            self._pending_action = ActionProto(ActionType.HOLD, 0.0)
            self._action_queue.clear()
        elif self.decision_mode == DecisionTiming.INTRA_HOUR_WITH_LATENCY:
            self._pending_action = None
            self._action_queue = deque(
                ActionProto(ActionType.HOLD, 0.0) for _ in range(self.latency_steps)
            )
        else:
            self._pending_action = None
            self._action_queue.clear()

        return obs, info

    def step(self, action):
        self.total_steps += 1
        row_idx = self.state.step_idx
        if self.decision_mode == DecisionTiming.INTRA_HOUR_WITH_LATENCY:
            row_idx = max(0, row_idx - self.latency_steps)
        row = self.df.iloc[row_idx]
        self._assert_feature_timestamps(row)
        self._update_bar_interval(row, row_idx)

        bid_col = next((c for c in ("bid", "best_bid", "bid_price") if c in row.index), None)
        ask_col = next((c for c in ("ask", "best_ask", "ask_price") if c in row.index), None)
        bid = ask = None

        price_key = "open" if self.decision_mode == DecisionTiming.CLOSE_TO_OPEN else "close"

        if bid_col and ask_col:
            bid = float(row[bid_col])
            ask = float(row[ask_col])
            mid = (bid + ask) / 2.0
        else:
            mid = float(row.get(price_key, row.get("price", 0.0)))
            if price_key == "close" and hasattr(self, "_close_actual") and len(self._close_actual) > row_idx:
                mid = float(self._close_actual.iloc[row_idx])

        vol_factor = float(row.get("atr_pct", 0.0))
        liquidity = float(row.get("liq_roll", 0.0))

        if bid_col and ask_col:
            spread_bps = (ask - bid) / mid * 10000 if mid else 0.0
        else:
            spread_bps = _dynamic_spread_bps(vol_factor=vol_factor, liquidity=liquidity, cfg=self._dyn_cfg)
            half = mid * spread_bps / 20000.0
            bid = mid - half
            ask = mid + half

        exec_sim = getattr(self._mediator, "exec", None)
        if exec_sim is not None and hasattr(exec_sim, "set_market_snapshot"):
            if not self._exec_intrabar_timeframe_configured:
                self._maybe_configure_exec_timeframe()
            ts_ms = self._resolve_snapshot_timestamp(row)
            trade_price = self._extract_trade_price(row)
            trade_qty = self._extract_trade_qty(row)
            exec_sim.set_market_snapshot(
                bid=bid,
                ask=ask,
                spread_bps=spread_bps,
                vol_factor=vol_factor,
                liquidity=liquidity,
                ts_ms=ts_ms,
                trade_price=trade_price,
                trade_qty=trade_qty,
                bar_open=self._safe_float(row.get("open")),
                bar_high=self._safe_float(row.get("high")),
                bar_low=self._safe_float(row.get("low")),
                bar_close=self._safe_float(row.get("close")),
            )
            self._maybe_forward_intrabar_path(exec_sim, row)

        self.last_bid = bid
        self.last_ask = ask
        self.last_mid = mid
        self.last_mtm_price = mid
        blocked = bool(self._no_trade_mask[row_idx]) if row_idx < len(self._no_trade_mask) else False
        if blocked:
            self.no_trade_blocks += 1
            proto = ActionProto(ActionType.HOLD, 0.0)
            self._pending_action = None
            self._action_queue.clear()
        else:
            if self.decision_mode == DecisionTiming.CLOSE_TO_OPEN:
                proto = self._pending_action or ActionProto(ActionType.HOLD, 0.0)
                self._pending_action = self._to_proto(action)
            elif self.decision_mode == DecisionTiming.INTRA_HOUR_WITH_LATENCY:
                proto = (
                    self._action_queue.popleft()
                    if self._action_queue
                    else ActionProto(ActionType.HOLD, 0.0)
                )
                self._action_queue.append(self._to_proto(action))
            else:
                proto = self._to_proto(action)

        prev_net_worth = self._safe_float(getattr(self.state, "net_worth", 0.0))
        if prev_net_worth is None:
            prev_net_worth = 0.0
        prev_turnover_total = self._safe_float(getattr(self, "_turnover_total", 0.0))
        if prev_turnover_total is None:
            prev_turnover_total = 0.0
        pre_len = len(getattr(self._mediator, "calls", []))

        context_ts = self._resolve_snapshot_timestamp(row)
        context_setter = getattr(self._mediator, "set_market_context", None)
        if callable(context_setter):
            try:
                context_setter(row=row, row_idx=row_idx, timestamp=context_ts)
            except Exception:
                pass
        else:
            try:
                setattr(self._mediator, "_context_row", row)
                setattr(self._mediator, "_context_row_idx", int(row_idx))
                setattr(self._mediator, "_context_timestamp", int(context_ts))
            except Exception:
                pass

        result = self._mediator.step(proto)
        if hasattr(self._mediator, "calls") and len(self._mediator.calls) == pre_len:
            self._mediator.calls.append(proto)
        obs, reward, terminated, truncated, info = result
        info = dict(info or {})

        # --- recompute ΔPnL / turnover adjusted reward -----------------
        mark_price = self._safe_float(self.last_mtm_price)
        if mark_price is None:
            mark_price = self._safe_float(self.last_mid)
        if mark_price is None:
            mark_price = self._safe_float(info.get("mark_price"))
        if mark_price is None:
            mark_price = 0.0

        cash = self._safe_float(getattr(self.state, "cash", 0.0))
        if cash is None:
            cash = 0.0
        units = self._safe_float(getattr(self.state, "units", 0.0))
        if units is None:
            units = 0.0
        new_net_worth = self._safe_float(getattr(self.state, "net_worth", prev_net_worth))
        if new_net_worth is None:
            new_net_worth = prev_net_worth
        if not math.isfinite(new_net_worth):
            new_net_worth = prev_net_worth
        if abs(new_net_worth - prev_net_worth) < 1e-12:
            new_net_worth = cash + units * mark_price
            try:
                self.state.net_worth = float(new_net_worth)
            except Exception:
                pass

        delta_pnl = new_net_worth - prev_net_worth
        if not math.isfinite(delta_pnl):
            delta_pnl = 0.0

        step_turnover = info.get("turnover")
        if step_turnover is None:
            step_turnover = info.get("executed_notional", 0.0)
        try:
            step_turnover = float(step_turnover)
        except (TypeError, ValueError):
            step_turnover = 0.0
        if not math.isfinite(step_turnover):
            step_turnover = 0.0
        step_turnover = abs(step_turnover)
        self._turnover_total = prev_turnover_total + step_turnover

        if not math.isfinite(self._turnover_total):
            self._turnover_total = prev_turnover_total

        turnover_penalty = info.get("turnover_penalty")
        if turnover_penalty is None:
            turnover_penalty = self.turnover_penalty_coef * step_turnover
        try:
            turnover_penalty = float(turnover_penalty)
        except (TypeError, ValueError):
            turnover_penalty = 0.0
        if not math.isfinite(turnover_penalty):
            turnover_penalty = 0.0

        reward = float(delta_pnl - turnover_penalty)
        if not math.isfinite(reward):
            reward = 0.0

        info["delta_pnl"] = float(delta_pnl)
        info["equity"] = float(new_net_worth if math.isfinite(new_net_worth) else prev_net_worth)
        info["turnover"] = float(step_turnover)
        info["cum_turnover"] = float(self._turnover_total)
        info["turnover_penalty"] = float(turnover_penalty)

        if terminated or truncated:
            info["no_trade_stats"] = self.get_no_trade_stats()
        return obs, reward, terminated, truncated, info

    def get_no_trade_stats(self) -> dict:
        """Return total and blocked step counts."""
        return {
            "total_steps": int(self.total_steps),
            "blocked_steps": int(self.no_trade_blocks),
        }

    def close(self) -> None:
        """Close all external resources."""
        ms = getattr(self, "market_sim", None)
        if ms is not None:
            try:
                close_fn = getattr(ms, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
        bus = getattr(self, "_bus", None)
        if bus is not None:
            try:
                getattr(bus, "close", lambda: None)()
            except Exception:
                pass
        lg = getattr(self, "_leak_guard", None)
        if lg is not None:
            try:
                getattr(lg, "close", lambda: None)()
            except Exception:
                pass
        try:
            super().close()
        except Exception:
            pass


    # ------------------------------------------------ util
    def seed(self, seed: int) -> None:  # noqa: D401
        """Seed the environment's RNG and propagate to sub-components."""
        self.seed_value = int(seed)
        rank_offset = getattr(self, "rank", 0)
        self._rng = np.random.default_rng(self.seed_value + rank_offset)

        # propagate to market simulator if possible
        ms = getattr(self, "market_sim", None)
        if ms is not None:
            if hasattr(ms, "set_seed"):
                try:
                    ms.set_seed(int(self.seed_value + rank_offset))
                except Exception:
                    pass
            elif hasattr(ms, "_rng"):
                try:
                    ms._rng = self._rng
                except Exception:
                    pass
# ----------------------- Simple market-sim stub (unchanged) -----------------------
class _SimpleMarketSim:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        import numpy as _np
        import json as _json
        import os as _os

        self._rng = rng or _np.random.default_rng()
        self._regime_distribution = _np.array([0.8, 0.05, 0.10, 0.05], dtype=float)
        self._current_regime = MarketRegime(MarketRegime.NORMAL)
        self._shocks_enabled = False
        self._flash_prob = 0.01
        self._fired_steps: set[int] = set()

        cfg_path = _os.getenv("MARKET_REGIMES_JSON", "configs/market_regimes.json")
        try:
            with open(cfg_path, "r") as f:
                cfg = _json.load(f)
            self._regime_distribution = _np.array(cfg.get("regime_probs", self._regime_distribution), dtype=float)
            flash_cfg = cfg.get("flash_shock", {})
            self._flash_prob = float(flash_cfg.get("probability", self._flash_prob))
        except Exception:
            pass

    def set_regime_distribution(self, p_vec: Sequence[float]) -> None:
        p = np.asarray(p_vec, dtype=float)
        if p.shape != (4,):
            raise ValueError("regime_dist must have length 4")
        s = float(p.sum())
        if s <= 0.0:
            raise ValueError("regime_dist must sum to > 0")
        self._regime_distribution = p / s

    def enable_random_shocks(self, enabled: bool = True, flash_prob: float = 0.01) -> None:
        self._shocks_enabled = bool(enabled)
        self._flash_prob = float(np.clip(flash_prob, 0.0, 1.0))

    def force_market_regime(self, regime: MarketRegime, *_, **__) -> None:
        self._current_regime = MarketRegime(regime)

    def shock_triggered(self, step_idx: int) -> float:
        if not self._shocks_enabled or step_idx in self._fired_steps:
            return 0.0
        if self._rng.random() < self._flash_prob:
            self._fired_steps.add(step_idx)
            return 1.0 if self._rng.random() < 0.5 else -1.0
        return 0.0

    @property
    def regime_distribution(self) -> np.ndarray:
        return self._regime_distribution.copy()

all = ["TradingEnv", "MarketRegime"]
