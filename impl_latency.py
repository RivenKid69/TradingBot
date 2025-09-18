# -*- coding: utf-8 -*-
"""
impl_latency.py
Обёртка над latency.LatencyModel. Подключает задержки к симулятору.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Mapping
import importlib.util
import sysconfig
from pathlib import Path
import threading
import warnings
import math
import weakref
import subprocess
import sys

try:
    from latency_volatility_cache import LatencyVolatilityCache
except ModuleNotFoundError as exc:  # pragma: no cover - explicit guidance for legacy setups
    raise ModuleNotFoundError(
        "latency_volatility_cache module is required; missing latency_volatility_cache.py"
    ) from exc
except Exception:  # pragma: no cover - optional dependency for older deployments
    LatencyVolatilityCache = None  # type: ignore

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
    latency_seasonality_path: str | None = None
    refresh_period_days: int | None = 30
    seasonality_default: Sequence[float] | float | None = 1.0
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
    vol_debug_log: bool = False


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
        vol_debug_log: bool = False,
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
        self._vol_debug_log = bool(vol_debug_log)
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
        idx = hour_of_week(int(ts_ms))
        length = len(self._mult)
        if length == 7:
            hour = (idx // 24) % 7
        else:
            hour = idx % length
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
                    result = (1.0, {"error": str(exc), "reason": "callback_error"})
            except Exception as exc:  # pragma: no cover - defensive fallback
                result = (1.0, {"error": str(exc), "reason": "callback_error"})
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
                vol_debug.setdefault("reason", "non_numeric")
                vol_mult = 1.0
            if not math.isfinite(vol_mult) or vol_mult <= 0.0:
                vol_debug.setdefault("reason", "invalid_multiplier")
                vol_debug.setdefault("vol_mult", vol_mult)
                vol_mult = 1.0

        if symbol is not None:
            try:
                vol_debug.setdefault("symbol", str(symbol).upper())
            except Exception:
                vol_debug.setdefault("symbol", symbol)
        try:
            vol_debug.setdefault("ts", int(ts_ms))
        except Exception:
            pass
        if vol_debug.get("reason"):
            vol_debug.setdefault("vol_mult", float(vol_mult) if math.isfinite(vol_mult) else vol_mult)
            vol_mult = 1.0
        for key in ("value", "mean", "std", "zscore", "clip", "gamma", "window"):
            vol_debug.setdefault(key, vol_debug.get(key))

        with self._lock:
            base, jitter, timeout = (
                self._model.base_ms,
                self._model.jitter_ms,
                self._model.timeout_ms,
            )
            seed = getattr(self._model, "seed", None)
            state_after = None
            try:
                season_mult = float(m)
                eff_base = float(base) * season_mult
                eff_jitter = float(jitter) * season_mult
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
            scaled_base_total = float(scaled_base) * attempts
            jitter_component = raw_total - scaled_base_total
            base_adjust = eff_base * attempts - scaled_base_total
            vol_adjust = eff_base * (float(vol_mult) - 1.0) * attempts
            lat_ms = raw_total + base_adjust + vol_adjust
            min_limit = float(self._min_ms)
            lat_ms = max(min_limit, lat_ms)
            if self._max_ms is not None:
                lat_ms = min(float(self._max_ms), lat_ms)
            lat_ms_int = int(round(lat_ms))
            if lat_ms_int < self._min_ms:
                lat_ms_int = self._min_ms
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
                    "jitter_component": jitter_component,
                    "base_adjust": base_adjust,
                    "vol_adjust": vol_adjust,
                }
                try:
                    debug_dict = res.setdefault("debug", {})
                    if isinstance(debug_dict, dict):
                        debug_dict["latency"] = debug_entry
                except Exception:  # pragma: no cover - defensive fallback
                    res["debug"] = {"latency": debug_entry}

            log_enabled = seasonality_logger.isEnabledFor(logging.DEBUG)
            should_emit_sample_log = (self._debug_log or log_enabled) and log_enabled
            if should_emit_sample_log:
                vol_value = vol_debug.get("value")
                vol_zscore = vol_debug.get("zscore")
                seasonality_logger.debug(
                    "latency sample h%03d season=%.3f vol=%.3f vol_value=%s zscore=%s raw=%.3f final=%s attempts=%s payload=%s",
                    hour,
                    float(m),
                    float(vol_mult),
                    vol_value,
                    vol_zscore,
                    raw_total,
                    lat_ms_int,
                    attempts,
                    vol_debug,
                )

            if self._vol_debug_log and logger.isEnabledFor(logging.DEBUG):
                log_payload = {
                    "symbol": symbol,
                    "hour": hour,
                    "seasonality_multiplier": float(m),
                    "volatility_multiplier": float(vol_mult),
                    "raw_total_ms": raw_total,
                    "adjusted_total_ms": lat_ms_int,
                    "attempts": attempts,
                    "volatility_debug": vol_debug,
                    "vol_value": vol_debug.get("value"),
                    "zscore": vol_debug.get("zscore"),
                }
                logger.debug("latency volatility sample: %s", log_payload)

            self._mult_sum[hour] += m
            self._lat_sum[hour] += float(lat_ms_int)
            self._count[hour] += 1
            _LATENCY_MULT_COUNTER.labels(hour=hour).inc()
            return res

    def stats(self):  # pragma: no cover - simple delegation
        return self._model.stats()

    def reset_stats(self) -> None:  # pragma: no cover - simple delegation
        self._model.reset_stats()
        with self._lock:
            n = len(self._mult)
            self._mult_sum = [0.0] * n
            self._lat_sum = [0.0] * n
            self._count = [0] * n

    def hourly_stats(self) -> Dict[str, List[float]]:
        with self._lock:
            mult_sum = list(self._mult_sum)
            lat_sum = list(self._lat_sum)
            count = list(self._count)
        n = len(count)
        avg_mult = [mult_sum[i] / count[i] if count[i] else 0.0 for i in range(n)]
        avg_lat = [lat_sum[i] / count[i] if count[i] else 0.0 for i in range(n)]
        return {"multiplier": avg_mult, "latency_ms": avg_lat, "count": count}

class LatencyImpl:
    @staticmethod
    def _normalize_default(
        default: Sequence[float] | float | None,
        *,
        length: int,
    ) -> List[float]:
        base = [1.0] * length
        if default is None:
            return base
        if isinstance(default, (int, float)):
            try:
                val = float(default)
            except (TypeError, ValueError):
                return base
            if not math.isfinite(val) or val <= 0.0:
                return base
            return [float(val)] * length
        if isinstance(default, Sequence) and not isinstance(
            default, (str, bytes, bytearray)
        ):
            try:
                arr = [float(x) for x in list(default)]
            except (TypeError, ValueError):
                return base
            if len(arr) != length:
                return base
            try:
                arr = validate_multipliers(arr, expected_len=length)
            except Exception:
                return base
            return list(arr)
        return base

    @staticmethod
    def _coerce_array(data: Any) -> Optional[np.ndarray]:
        if isinstance(data, Mapping):
            entries: Dict[int, float] = {}
            try:
                for key, raw in data.items():
                    idx = int(key)
                    if idx < 0:
                        return None
                    entries[idx] = float(raw)
            except (TypeError, ValueError):
                return None
            if not entries:
                return None
            max_idx = max(entries)
            length = max_idx + 1
            arr = np.full(length, np.nan, dtype=float)
            for idx, value in entries.items():
                if idx >= length:
                    return None
                arr[idx] = value
            if np.isnan(arr).any():
                return None
            return arr
        if isinstance(data, np.ndarray):
            arr = np.asarray(data, dtype=float)
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            arr = np.asarray(list(data), dtype=float)
        else:
            return None
        if arr.ndim != 1:
            return None
        return arr

    def _prepare_multipliers(
        self,
        data: Any,
        *,
        allow_convert: bool = True,
    ) -> Optional[List[float]]:
        arr = self._coerce_array(data)
        if arr is None:
            return None
        expected = 7 if self.cfg.seasonality_day_only else 168
        size = arr.size
        if size != expected and allow_convert:
            from utils_time import interpolate_daily_multipliers, daily_from_hourly

            if self.cfg.seasonality_day_only and size == 168:
                arr = daily_from_hourly(arr)
            elif not self.cfg.seasonality_day_only and size == 7:
                arr = interpolate_daily_multipliers(arr)
            else:
                return None
            size = arr.size
        if size != expected:
            return None
        try:
            arr_list = validate_multipliers(arr.tolist(), expected_len=expected)
        except Exception:
            return None
        return list(arr_list)

    def _log_seasonality_enabled(self, path: str) -> None:
        arr = np.asarray(self.latency, dtype=float)
        try:
            mtime = os.path.getmtime(path)
            mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except Exception:
            mtime_iso = "unknown"
        mean = float(arr.mean()) if arr.size else float("nan")
        seasonality_logger.info(
            "seasonality on: path=%s mtime=%s mean=%.6f len=%d",
            path,
            mtime_iso,
            mean,
            int(arr.size),
        )

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
        expected = 7 if cfg.seasonality_day_only else 168
        self.latency = self._normalize_default(cfg.seasonality_default, length=expected)
        self._latency_cache: List[float] = list(self.latency)
        self._mult_lock = threading.Lock()
        path = (
            cfg.seasonality_path
            or cfg.latency_seasonality_path
            or "data/latency/liquidity_latency_seasonality.json"
        )
        self._seasonality_path = path
        refresh_raw = cfg.refresh_period_days
        try:
            refresh_days = int(refresh_raw) if refresh_raw is not None else 0
        except (TypeError, ValueError):
            refresh_days = 0
        if refresh_days < 0:
            refresh_days = 0
        self._refresh_period_days = refresh_days
        builder_path = Path(__file__).resolve().parent / "scripts" / "build_hourly_seasonality.py"
        use_requested = bool(cfg.use_seasonality)
        cli_hint = None
        if path and refresh_days > 0 and use_requested:
            cli_hint = (
                f"python scripts/build_hourly_seasonality.py --out {path} --window-days {refresh_days}"
            )
            needs_refresh = False
            now = datetime.now(timezone.utc)
            mtime_dt: Optional[datetime] = None
            refresh_reason = ""
            try:
                mtime = os.path.getmtime(path)
            except FileNotFoundError:
                needs_refresh = True
                refresh_reason = "missing"
            except OSError as exc:
                needs_refresh = True
                refresh_reason = f"stat_failed: {exc}"
            else:
                mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
                if mtime_dt < now - timedelta(days=refresh_days):
                    needs_refresh = True
                    refresh_reason = "stale"
            if needs_refresh:
                if refresh_reason == "missing":
                    detail = f"Latency seasonality file {path} is missing"
                elif refresh_reason.startswith("stat_failed"):
                    detail = (
                        f"Latency seasonality file {path} could not be inspected "
                        f"({refresh_reason.split(':', 1)[1].strip()})"
                    )
                else:
                    if mtime_dt is not None:
                        age = now - mtime_dt
                        age_days = age.days + age.seconds / 86400.0
                        detail = (
                            f"Latency seasonality file {path} is older than {refresh_days} days"
                            f" (mtime={mtime_dt.isoformat()}, age≈{age_days:.2f}d)"
                        )
                    else:
                        detail = (
                            f"Latency seasonality file {path} is older than {refresh_days} days"
                        )
                if builder_path.exists():
                    python_bin = sys.executable or "python"
                    cmd = [
                        python_bin,
                        str(builder_path),
                        "--out",
                        str(path),
                        "--window-days",
                        str(refresh_days),
                    ]
                    try:
                        proc = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                    except Exception as exc:
                        seasonality_logger.warning(
                            "%s; auto-refresh command failed (%s). Run `%s` manually.",
                            detail,
                            exc,
                            cli_hint,
                        )
                    else:
                        if proc.returncode == 0:
                            snippet = proc.stdout.strip() if proc.stdout else ""
                            if len(snippet) > 200:
                                snippet = snippet[:200] + "..."
                            seasonality_logger.info(
                                "%s; auto-refresh succeeded via `%s`%s",
                                detail,
                                cli_hint,
                                f" (stdout: {snippet})" if snippet else "",
                            )
                        else:
                            output = proc.stderr or proc.stdout or ""
                            output = output.strip()
                            if len(output) > 200:
                                output = output[:200] + "..."
                            seasonality_logger.warning(
                                "%s; auto-refresh command exited with code %d. Run `%s` manually.%s",
                                detail,
                                proc.returncode,
                                cli_hint,
                                f" Output: {output}" if output else "",
                            )
                else:
                    seasonality_logger.warning(
                        "%s; auto-refresh script not found. Run `%s` manually.",
                        detail,
                        cli_hint,
                    )
        runtime_allowed = bool(seasonality_enabled())
        self._has_seasonality = bool(use_requested and runtime_allowed)
        loaded_from_file = False
        if self._has_seasonality and path:
            try:
                arr = load_hourly_seasonality(
                    path,
                    "latency",
                    symbol=cfg.symbol,
                    expected_hash=cfg.seasonality_hash,
                )
            except ValueError as exc:
                seasonality_logger.warning(
                    "Failed to parse latency seasonality file %s: %s",
                    path,
                    exc,
                )
                arr = None
            if arr is None:
                seasonality_logger.warning(
                    "Seasonality helper returned no latency multipliers for %s; using defaults.",
                    path,
                )
                self._has_seasonality = False
            else:
                prepared = self._prepare_multipliers(arr)
                if prepared is None:
                    seasonality_logger.warning(
                        "Latency seasonality array from %s has unexpected length; using defaults.",
                        path,
                    )
                    self._has_seasonality = False
                else:
                    self.latency = prepared
                    self._latency_cache = list(self.latency)
                    loaded_from_file = True
        if not self._has_seasonality:
            if use_requested and runtime_allowed:
                seasonality_logger.warning(
                    "Using default latency seasonality multipliers of 1.0; "
                    "run scripts/build_hourly_seasonality.py to generate them.",
                )
            self.latency = self._normalize_default(cfg.seasonality_default, length=expected)
            self._latency_cache = list(self.latency)
        else:
            override = cfg.seasonality_override
            o_path = cfg.seasonality_override_path
            if override is None and o_path:
                try:
                    override = load_hourly_seasonality(o_path, "latency", symbol=cfg.symbol)
                except ValueError as exc:
                    seasonality_logger.warning(
                        "Failed to parse latency override file %s: %s",
                        o_path,
                        exc,
                    )
                    override = None
                if override is None:
                    seasonality_logger.warning(
                        "Seasonality helper returned no multipliers for override %s; ignoring.",
                        o_path,
                    )
            if override is not None:
                override_prepared = self._prepare_multipliers(override)
                if override_prepared is None:
                    seasonality_logger.warning(
                        "Latency override payload could not be parsed; ignoring.",
                    )
                else:
                    base_arr = np.asarray(self.latency, dtype=float)
                    override_arr = np.asarray(override_prepared, dtype=float)
                    try:
                        combined = validate_multipliers(
                            (base_arr * override_arr).tolist(),
                            expected_len=len(self.latency),
                        )
                    except Exception:
                        seasonality_logger.warning(
                            "Latency override produced invalid multipliers; ignoring.",
                        )
                    else:
                        self.latency = list(combined)
                        self._latency_cache = list(self.latency)
        self.latency = list(
            validate_multipliers(self.latency, expected_len=len(self.latency))
        )
        self._latency_cache = list(self.latency)
        if self._has_seasonality and loaded_from_file and path:
            self._log_seasonality_enabled(path)
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
                vol_debug_log=self.cfg.vol_debug_log,
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
            sym_norm: Optional[str] = None

            def _finalize(data: Dict[str, Any], ts_value: Optional[int]) -> Dict[str, Any]:
                if sym_norm:
                    data.setdefault("symbol", sym_norm)
                if ts_value is not None:
                    data.setdefault("ts", ts_value)
                if "ts_ms" in data and "ts" not in data:
                    try:
                        data.setdefault("ts", int(data["ts_ms"]))
                    except Exception:
                        data.setdefault("ts", ts_value)
                data.setdefault("gamma", gamma)
                data.setdefault("window", window)
                data.setdefault("clip", clip)
                if data.get("value") is None and "last_value" in data:
                    data["value"] = data.get("last_value")
                data.setdefault("value", data.get("value"))
                data.setdefault("mean", data.get("mean"))
                data.setdefault("std", data.get("std"))
                data.setdefault("zscore", data.get("zscore"))
                return data

            if gamma == 0.0:
                debug["reason"] = "gamma_zero"
                return 1.0, _finalize(debug, None)

            sim_obj = sim_ref()
            if sim_obj is None:
                debug["reason"] = "sim_released"
                return 1.0, _finalize(debug, None)

            sym = (
                symbol
                or getattr(sim_obj, "_latency_symbol", None)
                or getattr(sim_obj, "symbol", None)
            )
            if not sym:
                debug["reason"] = "symbol_missing"
                return 1.0, _finalize(debug, None)

            sym_norm = str(sym).upper()

            cache = getattr(sim_obj, "volatility_cache", None)
            if cache is None:
                debug["reason"] = "cache_missing"
                return 1.0, _finalize(debug, None)

            ready = True
            ready_attr = getattr(cache, "ready", None)
            if isinstance(ready_attr, bool):
                ready = ready_attr
            else:
                ready_fn = getattr(cache, "is_ready", None)
                if callable(ready_fn):
                    try:
                        ready = bool(ready_fn(sym_norm))
                    except TypeError:
                        try:
                            ready = bool(ready_fn(symbol=sym_norm))
                        except TypeError:
                            ready = bool(ready_fn())
                    except Exception:
                        ready = True
            if not ready:
                debug["reason"] = "cache_not_ready"
                return 1.0, _finalize(debug, None)

            resolver = getattr(cache, "latency_multiplier", None)
            if not callable(resolver):
                debug["reason"] = "no_method"
                return 1.0, _finalize(debug, None)

            try:
                ts_val = int(ts_ms)
            except (TypeError, ValueError):
                debug["reason"] = "invalid_ts"
                return 1.0, _finalize(debug, None)

            payload: Dict[str, Any] = {}
            try:
                try:
                    result = resolver(
                        symbol=sym_norm,
                        ts_ms=ts_val,
                        metric=metric,
                        window=window,
                        gamma=gamma,
                        clip=clip,
                    )
                except TypeError:
                    result = resolver(sym_norm, ts_val, metric, window, gamma, clip)
            except Exception as exc:
                return 1.0, _finalize(
                    {
                        "reason": "exception",
                        "error": str(exc),
                    },
                    ts_val,
                )

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
                return 1.0, _finalize(payload or debug, ts_val)

            if not math.isfinite(value) or value <= 0.0:
                payload.setdefault("reason", "invalid_multiplier")
                payload.setdefault("vol_mult", value)
                return 1.0, _finalize(payload, ts_val)

            payload.setdefault("metric", metric)
            payload.setdefault("window", window)
            payload.setdefault("gamma", gamma)
            payload.setdefault("clip", clip)
            payload.setdefault("vol_mult", value)
            finalized = _finalize(payload, ts_val)
            return value, finalized

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
            updater = getattr(cache, "update_latency_factor", None)
            if not callable(updater):
                return
            sym = symbol or getattr(sim_obj, "_latency_symbol", None) or getattr(sim_obj, "symbol", None)
            if sym is None:
                return
            sym_norm = str(sym).upper()
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
                updater(symbol=sym_norm, ts_ms=ts, value=val)
                return
            except TypeError:
                try:
                    updater(sym_norm, ts, val)
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

        return list(self._latency_cache)

    def load_multipliers(self, arr: Sequence[float] | Mapping[int, float] | np.ndarray) -> None:
        """Load latency seasonality multipliers from ``arr``.

        ``arr`` must contain 168 float values (or 7 when
        ``seasonality_day_only`` is enabled). Raises ``ValueError`` if the
        length is incorrect. If the implementation is already attached to a
        simulator, the underlying wrapper is updated as well. ``arr`` may be a
        sequence or a mapping keyed by hour-of-week.
        """

        expected = 7 if self.cfg.seasonality_day_only else 168
        prepared = self._prepare_multipliers(arr)
        if prepared is None:
            raise ValueError(f"multipliers must have length {expected}")
        with self._mult_lock:
            self.latency = list(prepared)
            self._latency_cache = list(self.latency)
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
        vol_debug_log = d.get("vol_debug_log", False)
        seasonality_path = d.get("seasonality_path")
        latency_seasonality_path = d.get("latency_seasonality_path")
        if seasonality_path is None and latency_seasonality_path:
            seasonality_path = latency_seasonality_path
        refresh_raw = d.get("refresh_period_days", d.get("seasonality_refresh_period_days"))
        if refresh_raw is None:
            refresh_period_days = 30
        else:
            try:
                refresh_period_days = int(refresh_raw)
            except (TypeError, ValueError):
                refresh_period_days = 30
        if "seasonality_default" in d:
            seasonality_default = d.get("seasonality_default")
        else:
            seasonality_default = 1.0
        return LatencyImpl(LatencyCfg(
            base_ms=int(d.get("base_ms", 250)),
            jitter_ms=int(d.get("jitter_ms", 50)),
            spike_p=float(d.get("spike_p", 0.01)),
            spike_mult=float(d.get("spike_mult", 5.0)),
            timeout_ms=int(d.get("timeout_ms", 2500)),
            retries=int(d.get("retries", 1)),
            seed=int(d.get("seed", 0)),
            symbol=(d.get("symbol") if d.get("symbol") is not None else None),
            seasonality_path=seasonality_path,
            latency_seasonality_path=latency_seasonality_path,
            refresh_period_days=refresh_period_days,
            seasonality_default=seasonality_default,
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
            vol_debug_log=bool(vol_debug_log),
        ))
