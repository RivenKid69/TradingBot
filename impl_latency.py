# -*- coding: utf-8 -*-
"""
impl_latency.py
Обёртка над latency.LatencyModel. Подключает задержки к симулятору.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import importlib.util
import sysconfig
from pathlib import Path
import threading
import warnings
import math
import weakref

import numpy as np
try:
    from runtime_flags import seasonality_enabled
except Exception:  # pragma: no cover - fallback when module not found
    def seasonality_enabled(default: bool = True) -> bool:
        return default

from utils_time import hour_of_week
from utils.prometheus import Counter

_logging_spec = importlib.util.spec_from_file_location(
    "py_logging", Path(sysconfig.get_path("stdlib")) / "logging/__init__.py"
)
logging = importlib.util.module_from_spec(_logging_spec)
_logging_spec.loader.exec_module(logging)

try:
    from utils_time import (
        load_hourly_seasonality,
        get_latency_multiplier,
        watch_seasonality_file,
    )
except Exception:  # pragma: no cover - fallback
    try:
        import pathlib, sys
        sys.path.append(str(pathlib.Path(__file__).resolve().parent))
        from utils_time import (
            load_hourly_seasonality,
            get_latency_multiplier,
            watch_seasonality_file,
        )
    except Exception:  # pragma: no cover
        def load_hourly_seasonality(*a, **k):
            return None  # type: ignore

logger = logging.getLogger(__name__)
seasonality_logger = logging.getLogger("seasonality").getChild(__name__)

_LATENCY_MULT_COUNTER = Counter(
    "latency_hour_of_week_multiplier_total",
    "Latency multiplier applications per hour of week",
    ["hour"],
)

try:
    from latency import LatencyModel, validate_multipliers
except Exception:  # pragma: no cover
    LatencyModel = None  # type: ignore
    def validate_multipliers(multipliers, *, expected_len=168, cap=10.0):  # type: ignore
        return [float(x) for x in multipliers]


@dataclass
class LatencyCfg:
    base_ms: int = 250
    jitter_ms: int = 50
    spike_p: float = 0.01
    spike_mult: float = 5.0
    timeout_ms: int = 2500
    retries: int = 1
    seed: int = 0
    symbol: str | None = None
    seasonality_path: str | None = None
    use_seasonality: bool = True
    seasonality_override: Sequence[float] | None = None
    seasonality_override_path: str | None = None
    seasonality_hash: str | None = None
    seasonality_interpolate: bool = False
    seasonality_day_only: bool = False
    seasonality_auto_reload: bool = False
    vol_metric: str = "sigma"
    vol_window: int = 120
    volatility_gamma: float = 0.0
    zscore_clip: float = 3.0
    min_ms: int = 0
    max_ms: int = 10000
    debug_log: bool = False


class _LatencyWithSeasonality:
    """Wraps LatencyModel applying hourly multipliers and collecting stats."""

    def __init__(
        self,
        model: LatencyModel,
        multipliers: Sequence[float],
        *,
        interpolate: bool = False,
        symbol: str | None = None,
        volatility_callback: Optional[
            Callable[[Optional[str], int], Tuple[float, Dict[str, Any]]]
        ] = None,
        volatility_update: Optional[Callable[[Optional[str], int, float], None]] = None,
        min_ms: int | None = None,
        max_ms: int | None = None,
        debug_log: bool = False,
    ):  # type: ignore[name-defined]
        self._model = model
        n = len(multipliers)
        if n not in (7, 168):
            raise ValueError("multipliers must have length 7 or 168")
        arr = np.asarray(validate_multipliers(multipliers, expected_len=n), dtype=float)
        self._mult = arr
        self._interpolate = bool(interpolate)
        self._mult_sum: List[float] = [0.0] * n
        self._lat_sum: List[float] = [0.0] * n
        self._count: List[int] = [0] * n
        self._lock = threading.Lock()
        self._symbol: Optional[str] = str(symbol).upper() if symbol else None
        self._vol_cb = volatility_callback
        self._vol_update = volatility_update
        self._debug_log = bool(debug_log)
        self._min_ms = int(round(float(min_ms))) if min_ms is not None else 0
        if max_ms is None:
            self._max_ms: Optional[int] = None
        else:
            self._max_ms = int(round(float(max_ms)))
        if self._max_ms is not None and self._max_ms < self._min_ms:
            raise ValueError("max_ms must be >= min_ms")

    def set_symbol(self, symbol: str | None) -> None:
        with self._lock:
            self._symbol = str(symbol).upper() if symbol else None

    def set_volatility_callback(
        self,
        callback: Optional[Callable[[Optional[str], int], Tuple[float, Dict[str, Any]]]],
    ) -> None:
        with self._lock:
            self._vol_cb = callback

    def set_volatility_update(
        self, callback: Optional[Callable[[Optional[str], int, float], None]]
    ) -> None:
        with self._lock:
            self._vol_update = callback

    def update_volatility(
        self, symbol: str | None, ts_ms: int, value: float | None
    ) -> None:
        if value is None:
            return
        with self._lock:
            cb = self._vol_update
            sym = symbol or self._symbol
        if cb is None:
            return
        try:
            ts = int(ts_ms)
        except (TypeError, ValueError):
            return
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(val):
            return
        try:
            cb(sym, ts, val)
        except TypeError:
            try:
                if sym is not None:
                    cb(sym, ts, value)  # type: ignore[arg-type]
                else:
                    cb(None, ts, val)
            except Exception:
                return
        except Exception:
            return

    def sample(self, ts_ms: int | None = None):
        if ts_ms is None:
            return self._model.sample()
        hour = hour_of_week(int(ts_ms)) % len(self._mult)
        m = get_latency_multiplier(int(ts_ms), self._mult, interpolate=self._interpolate)

        vol_mult = 1.0
        vol_debug: Dict[str, Any] = {}
        cb = self._vol_cb
        symbol = self._symbol
        if cb is not None:
            try:
                result = cb(symbol, int(ts_ms))
            except TypeError:
                try:
                    result = cb(symbol=symbol, ts_ms=int(ts_ms))  # type: ignore[misc]
                except Exception as exc:  # pragma: no cover - defensive fallback
                    result = (1.0, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive fallback
                result = (1.0, {"error": str(exc)})
            if isinstance(result, tuple):
                vol_mult = result[0]
                if len(result) > 1 and isinstance(result[1], dict):
                    vol_debug = dict(result[1])
                elif len(result) > 1:
                    vol_debug = {"payload": result[1]}
            else:
                vol_mult = result
            try:
                vol_mult = float(vol_mult)
            except (TypeError, ValueError):
                vol_debug.setdefault("fallback", "non_numeric")
                vol_mult = 1.0
            if not math.isfinite(vol_mult) or vol_mult <= 0.0:
                vol_debug.setdefault("fallback", "invalid")
                vol_mult = 1.0

        with self._lock:
            base, jitter, timeout = (
                self._model.base_ms,
                self._model.jitter_ms,
                self._model.timeout_ms,
            )
            seed = getattr(self._model, "seed", None)
            state_after = None
            try:
                eff_base = float(base) * float(m) * float(vol_mult)
                eff_jitter = float(jitter) * float(m) * float(vol_mult)
                scaled_base = int(round(eff_base))
                if scaled_base > timeout:
                    seasonality_logger.warning(
                        "scaled base_ms %s exceeds timeout_ms %s; capping",
                        scaled_base,
                        timeout,
                    )
                    scaled_base = timeout
                    eff_base = float(scaled_base)
                self._model.base_ms = scaled_base
                scaled_jitter = int(round(eff_jitter))
                if scaled_jitter < 0:
                    scaled_jitter = 0
                self._model.jitter_ms = scaled_jitter
                res = self._model.sample()
                if hasattr(self._model, "_rng"):
                    state_after = self._model._rng.getstate()
            finally:
                self._model.base_ms, self._model.jitter_ms, self._model.timeout_ms = (
                    base,
                    jitter,
                    timeout,
                )
                if seed is not None:
                    self._model.seed = seed
                if state_after is not None and hasattr(self._model, "_rng"):
                    self._model._rng.setstate(state_after)

            attempts = int(res.get("attempts", 1) or 1)
            if attempts < 1:
                attempts = 1
            raw_total = float(res.get("total_ms", 0.0))
            base_adjust = eff_base * attempts - float(scaled_base) * attempts
            lat_ms = raw_total + base_adjust
            lat_ms = max(float(self._min_ms), lat_ms)
            if self._max_ms is not None:
                lat_ms = min(float(self._max_ms), lat_ms)
            lat_ms_int = int(round(lat_ms))
            lat_ms_int = max(self._min_ms, lat_ms_int)
            if self._max_ms is not None and lat_ms_int > self._max_ms:
                lat_ms_int = self._max_ms
            res["total_ms"] = lat_ms_int
            res["timeout"] = bool(lat_ms_int > timeout)

            if self._debug_log:
                debug_entry = {
                    "hour": hour,
                    "seasonality_multiplier": float(m),
                    "volatility_multiplier": float(vol_mult),
                    "volatility_debug": vol_debug,
                    "raw_total_ms": raw_total,
                    "adjusted_total_ms": lat_ms_int,
                    "attempts": attempts,
                    "min_ms": self._min_ms,
                    "max_ms": self._max_ms,
                }
                try:
                    debug_dict = res.setdefault("debug", {})
                    if isinstance(debug_dict, dict):
                        debug_dict["latency"] = debug_entry
                except Exception:  # pragma: no cover - defensive fallback
                    res["debug"] = {"latency": debug_entry}

            if self._debug_log or seasonality_logger.isEnabledFor(logging.DEBUG):
                seasonality_logger.debug(
                    "latency sample h%03d season=%.3f vol=%.3f raw=%.3f final=%s attempts=%s payload=%s",
                    hour,
                    float(m),
                    float(vol_mult),
                    raw_total,
                    lat_ms_int,
                    attempts,
                    vol_debug,
                )

            self._mult_sum[hour] += m
            self._lat_sum[hour] += float(lat_ms_int)
            self._count[hour] += 1
            _LATENCY_MULT_COUNTER.labels(hour=hour).inc()
            return res

    def stats(self):  # pragma: no cover - simple delegation
        return self._model.stats()

    def reset_stats(self) -> None:  # pragma: no cover - simple delegation
        self._model.reset_stats()
        self._mult_sum = [0.0] * 168
        self._lat_sum = [0.0] * 168
        self._count = [0] * 168

    def hourly_stats(self) -> Dict[str, List[float]]:
        avg_mult = [self._mult_sum[i] / self._count[i] if self._count[i] else 0.0 for i in range(168)]
        avg_lat = [self._lat_sum[i] / self._count[i] if self._count[i] else 0.0 for i in range(168)]
        return {"multiplier": avg_mult, "latency_ms": avg_lat, "count": list(self._count)}

class LatencyImpl:
    def __init__(self, cfg: LatencyCfg) -> None:
        self.cfg = cfg
        self._model = LatencyModel(
            base_ms=int(cfg.base_ms),
            jitter_ms=int(cfg.jitter_ms),
            spike_p=float(cfg.spike_p),
            spike_mult=float(cfg.spike_mult),
            timeout_ms=int(cfg.timeout_ms),
            retries=int(cfg.retries),
            seed=int(cfg.seed),
        ) if LatencyModel is not None else None
        self.latency: List[float] = [1.0] * (7 if cfg.seasonality_day_only else 168)
        self._mult_lock = threading.Lock()
        path = cfg.seasonality_path or "configs/liquidity_latency_seasonality.json"
        self._seasonality_path = path
        self._has_seasonality = bool(cfg.use_seasonality and seasonality_enabled())
        if self._has_seasonality:
            arr = load_hourly_seasonality(
                path,
                "latency",
                symbol=cfg.symbol,
                expected_hash=cfg.seasonality_hash,
            )
            from utils_time import interpolate_daily_multipliers, daily_from_hourly
            if arr is None:
                logger.warning(
                    "Seasonality helper returned no multipliers for %s; using default multipliers.",
                    path,
                )
                self._has_seasonality = False
            else:
                if cfg.seasonality_day_only and arr.size == 168:
                    arr = daily_from_hourly(arr)
                elif not cfg.seasonality_day_only and arr.size == 7:
                    arr = interpolate_daily_multipliers(arr)
                self.latency = [float(x) for x in arr]
            if not self._has_seasonality:
                logger.warning(
                    "Using default latency seasonality multipliers of 1.0; "
                    "run scripts/build_hourly_seasonality.py to generate them.",
                )
            override = cfg.seasonality_override
            o_path = cfg.seasonality_override_path
            if override is None and o_path:
                override = load_hourly_seasonality(o_path, "latency", symbol=cfg.symbol)
                if override is None:
                    logger.warning(
                        "Seasonality helper returned no multipliers for override %s; ignoring.",
                        o_path,
                    )
            if override is not None:
                arr = np.asarray(override, dtype=float)
                if cfg.seasonality_day_only and arr.size == 168:
                    arr = daily_from_hourly(arr)
                elif not cfg.seasonality_day_only and arr.size == 7:
                    arr = interpolate_daily_multipliers(arr)
                if arr.size != len(self.latency):
                    logger.warning(
                        "Latency override array length %s does not match expected %s; ignoring.",
                        arr.size,
                        len(self.latency),
                    )
                else:
                    self.latency = (
                        np.asarray(self.latency, dtype=float) * arr
                    ).tolist()
        self.latency = validate_multipliers(self.latency, expected_len=len(self.latency))
        self.attached_sim = None
        self._wrapper: _LatencyWithSeasonality | None = None
        if self._has_seasonality and cfg.seasonality_auto_reload and path:
            def _reload(data: Dict[str, np.ndarray]) -> None:
                arr = data.get("latency")
                if arr is not None:
                    try:
                        self.load_multipliers(arr)
                        seasonality_logger.info("Reloaded latency multipliers from %s", path)
                    except Exception:
                        seasonality_logger.exception(
                            "Failed to reload latency multipliers from %s", path
                        )

            watch_seasonality_file(path, _reload)

    @property
    def model(self):
        return self._model

    def attach_to(self, sim) -> None:
        if self._model is not None:
            mult = self.latency if self._has_seasonality else [1.0] * len(self.latency)
            vol_cb = self._build_volatility_callback(sim)
            vol_update = self._build_volatility_updater(sim)
            symbol = self.cfg.symbol or getattr(sim, "symbol", None)
            self._wrapper = _LatencyWithSeasonality(
                self._model,
                mult,
                interpolate=self.cfg.seasonality_interpolate,
                symbol=symbol,
                volatility_callback=vol_cb,
                volatility_update=vol_update,
                min_ms=self.cfg.min_ms,
                max_ms=self.cfg.max_ms,
                debug_log=self.cfg.debug_log,
            )
            sim_symbol = getattr(sim, "symbol", None)
            if sim_symbol is not None:
                self._wrapper.set_symbol(sim_symbol)
            elif symbol is not None:
                self._wrapper.set_symbol(symbol)
            if vol_cb is not None:
                self._wrapper.set_volatility_callback(vol_cb)
            if vol_update is not None:
                self._wrapper.set_volatility_update(vol_update)
            setattr(sim, "latency", self._wrapper)
            final_symbol = getattr(self._wrapper, "_symbol", None)
            if final_symbol:
                try:
                    setattr(sim, "_latency_symbol", str(final_symbol).upper())
                except Exception:
                    pass
        self.attached_sim = sim

    def _build_volatility_callback(
        self, sim
    ) -> Optional[Callable[[Optional[str], int], Tuple[float, Dict[str, Any]]]]:
        gamma = float(self.cfg.volatility_gamma)
        if gamma == 0.0:
            return None

        metric = str(self.cfg.vol_metric or "sigma")
        window = int(self.cfg.vol_window or 1)
        clip = float(self.cfg.zscore_clip)
        sim_ref = weakref.ref(sim)

        def _resolve(symbol: Optional[str], ts_ms: int) -> Tuple[float, Dict[str, Any]]:
            debug: Dict[str, Any] = {}
            if gamma == 0.0:
                debug["reason"] = "gamma_zero"
                return 1.0, debug

            sim_obj = sim_ref()
            if sim_obj is None:
                debug["reason"] = "sim_released"
                return 1.0, debug

            cache = getattr(sim_obj, "volatility_cache", None)
            if cache is None:
                debug["reason"] = "cache_missing"
                return 1.0, debug

            ready = True
            ready_attr = getattr(cache, "ready", None)
            if isinstance(ready_attr, bool):
                ready = ready_attr
            else:
                ready_fn = getattr(cache, "is_ready", None)
                if callable(ready_fn):
                    try:
                        ready = bool(ready_fn())
                    except Exception:
                        ready = True
            if not ready:
                debug["reason"] = "cache_not_ready"
                return 1.0, debug

            sym = symbol or getattr(sim_obj, "symbol", None)
            if not sym:
                debug["reason"] = "symbol_missing"
                return 1.0, debug

            method = None
            for name in (
                "latency_multiplier",
                "get_latency_multiplier",
                "latency_factor",
                "get_latency_factor",
            ):
                cand = getattr(cache, name, None)
                if callable(cand):
                    method = cand
                    break

            if method is None:
                debug["reason"] = "no_method"
                return 1.0, debug

            payload: Dict[str, Any] = {}
            try:
                try:
                    result = method(
                        symbol=sym,
                        ts_ms=int(ts_ms),
                        metric=metric,
                        window=window,
                        gamma=gamma,
                        clip=clip,
                    )
                except TypeError:
                    result = method(sym, int(ts_ms), metric, window, gamma, clip)
            except Exception as exc:
                debug["error"] = str(exc)
                return 1.0, debug

            if isinstance(result, tuple):
                vol_mult = result[0]
                if len(result) > 1 and isinstance(result[1], dict):
                    payload.update(result[1])
                elif len(result) > 1:
                    payload["payload"] = result[1]
            else:
                vol_mult = result

            try:
                value = float(vol_mult)
            except (TypeError, ValueError):
                payload.setdefault("reason", "non_numeric")
                return 1.0, payload or debug

            if not math.isfinite(value) or value <= 0.0:
                payload.setdefault("reason", "invalid_multiplier")
                payload.setdefault("vol_mult", value)
                return 1.0, payload

            payload.setdefault("metric", metric)
            payload.setdefault("window", window)
            payload.setdefault("gamma", gamma)
            payload.setdefault("clip", clip)
            return value, payload

        return _resolve

    def _build_volatility_updater(
        self, sim
    ) -> Optional[Callable[[Optional[str], int, float], None]]:
        gamma = float(self.cfg.volatility_gamma)
        if gamma == 0.0:
            return None

        sim_ref = weakref.ref(sim)

        def _update(symbol: Optional[str], ts_ms: int, value: float) -> None:
            sim_obj = sim_ref()
            if sim_obj is None:
                return
            cache = getattr(sim_obj, "volatility_cache", None)
            if cache is None:
                return
            sym = symbol or getattr(sim_obj, "_latency_symbol", None) or getattr(sim_obj, "symbol", None)
            sym_norm = str(sym).upper() if sym else None
            try:
                ts = int(ts_ms)
            except (TypeError, ValueError):
                return
            try:
                val = float(value)
            except (TypeError, ValueError):
                return
            if not math.isfinite(val):
                return
            method = None
            for name in (
                "update_latency_factor",
                "update_latency",
                "update",
                "append",
                "observe",
            ):
                cand = getattr(cache, name, None)
                if callable(cand):
                    method = cand
                    break
            if method is None:
                return
            try:
                if sym_norm is not None:
                    method(symbol=sym_norm, ts_ms=ts, value=val)
                else:
                    method(ts_ms=ts, value=val)
                return
            except TypeError:
                pass
            try:
                if sym_norm is not None:
                    method(sym_norm, ts, val)
                else:
                    method(ts, val)
                return
            except TypeError:
                try:
                    method(ts, val)
                    return
                except TypeError:
                    try:
                        if sym_norm is not None:
                            method(sym_norm, val)
                            return
                    except Exception:
                        pass
                    try:
                        method(val)
                        return
                    except Exception:
                        return
            except Exception:
                return

        return _update

    def get_stats(self):
        if self._wrapper is not None:
            return self._wrapper.stats()
        if self._model is None:
            return None
        return self._model.stats()

    def reset_stats(self) -> None:
        if self._wrapper is not None:
            self._wrapper.reset_stats()
        elif self._model is not None:
            self._model.reset_stats()

    def update_volatility(
        self, symbol: Optional[str], ts_ms: int, value: float | None
    ) -> None:
        if self._wrapper is None:
            return
        updater = getattr(self._wrapper, "update_volatility", None)
        if not callable(updater):
            return
        try:
            updater(symbol, ts_ms, value)
        except Exception:
            return

    def get_hourly_stats(self):
        if self._wrapper is None:
            return None
        return self._wrapper.hourly_stats()

    def dump_multipliers(self) -> List[float]:
        """Return current latency seasonality multipliers as a list."""

        return list(self.latency)

    def load_multipliers(self, arr: Sequence[float]) -> None:
        """Load latency seasonality multipliers from ``arr``.

        ``arr`` must contain 168 float values (or 7 when
        ``seasonality_day_only`` is enabled). Raises ``ValueError`` if the
        length is incorrect. If the implementation is already attached to a
        simulator, the underlying wrapper is updated as well.
        """

        expected = 7 if self.cfg.seasonality_day_only else 168
        arr_list = validate_multipliers(arr, expected_len=expected)
        with self._mult_lock:
            self.latency = arr_list
            if self._wrapper is not None:
                with self._wrapper._lock:
                    self._wrapper._mult = np.asarray(self.latency, dtype=float)
                    n = len(self.latency)
                    self._wrapper._mult_sum = [0.0] * n
                    self._wrapper._lat_sum = [0.0] * n
                    self._wrapper._count = [0] * n

    def dump_latency_multipliers(self) -> List[float]:
        warnings.warn(
            "dump_latency_multipliers() is deprecated; use dump_multipliers() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dump_multipliers()

    def load_latency_multipliers(self, arr: Sequence[float]) -> None:
        warnings.warn(
            "load_latency_multipliers() is deprecated; use load_multipliers() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.load_multipliers(arr)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LatencyImpl":
        vol_metric = d.get("vol_metric")
        vol_window = d.get("vol_window")
        volatility_gamma = d.get("volatility_gamma")
        zscore_clip = d.get("zscore_clip")
        min_ms = d.get("min_ms")
        max_ms = d.get("max_ms")
        debug_log = d.get("debug_log", False)
        return LatencyImpl(LatencyCfg(
            base_ms=int(d.get("base_ms", 250)),
            jitter_ms=int(d.get("jitter_ms", 50)),
            spike_p=float(d.get("spike_p", 0.01)),
            spike_mult=float(d.get("spike_mult", 5.0)),
            timeout_ms=int(d.get("timeout_ms", 2500)),
            retries=int(d.get("retries", 1)),
            seed=int(d.get("seed", 0)),
            symbol=(d.get("symbol") if d.get("symbol") is not None else None),
            seasonality_path=d.get("seasonality_path"),
            use_seasonality=bool(d.get("use_seasonality", True)),
            seasonality_override=d.get("seasonality_override"),
            seasonality_override_path=d.get("seasonality_override_path"),
            seasonality_hash=d.get("seasonality_hash"),
            seasonality_interpolate=bool(d.get("seasonality_interpolate", False)),
            seasonality_day_only=bool(d.get("seasonality_day_only", False)),
            seasonality_auto_reload=bool(d.get("seasonality_auto_reload", False)),
            vol_metric=str(vol_metric) if vol_metric is not None else "sigma",
            vol_window=int(vol_window) if vol_window is not None else 120,
            volatility_gamma=(
                float(volatility_gamma) if volatility_gamma is not None else 0.0
            ),
            zscore_clip=float(zscore_clip) if zscore_clip is not None else 3.0,
            min_ms=int(min_ms) if min_ms is not None else 0,
            max_ms=int(max_ms) if max_ms is not None else 10000,
            debug_log=bool(debug_log),
        ))
