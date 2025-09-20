# -*- coding: utf-8 -*-
"""
impl_quantizer.py
Обёртка над quantizer. Строит Quantizer из JSON-фильтров Binance и умеет подключаться к ExecutionSimulator.

Ключевые возможности:
- :attr:`QuantizerImpl.quantizer` — объект :class:`quantizer.Quantizer` с загруженными
  фильтрами биржи.
- :attr:`QuantizerImpl.symbol_filters` — read-only отображение «символ → фильтры» в виде
  подготовленных :class:`quantizer.SymbolFilters`.
- :meth:`QuantizerImpl.validate_order` — helper, повторяющий последовательность
  ``Quantizer.quantize_order`` и пригодный для использования исполнителями вроде
  :class:`impl_sim_executor.SimExecutor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, Mapping
from types import MappingProxyType
import hashlib
import logging
import os
import subprocess
import time
import warnings

from services import monitoring

try:
    from quantizer import Quantizer, OrderCheckResult, SymbolFilters
except Exception as e:  # pragma: no cover
    Quantizer = None  # type: ignore
    OrderCheckResult = None  # type: ignore
    SymbolFilters = Any  # type: ignore


logger = logging.getLogger(__name__)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _filters_age_days(meta: Dict[str, Any], path: str) -> Optional[float]:
    ts: Optional[datetime] = None
    for key in ("generated_at", "built_at"):
        ts = _parse_timestamp(meta.get(key))
        if ts is not None:
            break
    if ts is None and path:
        try:
            ts = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        except OSError:
            ts = None
    if ts is None:
        return None
    delta = datetime.now(timezone.utc) - ts
    return max(delta.total_seconds() / 86400.0, 0.0)


def _is_stale(age_days: Optional[float], max_age_days: int) -> bool:
    if age_days is None:
        return False
    return age_days > float(max_age_days)


def _file_size_bytes(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _file_mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _file_sha256(path: str) -> Optional[str]:
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


@dataclass
class QuantizerConfig:
    path: str
    strict_filters: bool = True
    quantize_mode: str = "inward"
    enforce_percent_price_by_side: bool = True  # передаётся в симулятор как enforce_ppbs
    filters_path: str = ""
    auto_refresh_days: int = 30
    refresh_on_start: bool = False

    def resolved_filters_path(self) -> str:
        return self.filters_path or self.path

    @property
    def strict(self) -> bool:
        """Backward compatibility accessor for legacy strict flag."""

        return self.strict_filters

    @strict.setter
    def strict(self, value: bool) -> None:
        self.strict_filters = bool(value)


@dataclass
class _SimpleOrderCheckResult:
    price: float
    qty: float
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    @property
    def accepted(self) -> bool:
        return self.reason_code is None


def _make_order_check_result(**kwargs: Any):
    cls = OrderCheckResult
    if cls is None:
        return _SimpleOrderCheckResult(**kwargs)
    return cls(**kwargs)


class QuantizerImpl:
    _REFRESH_GUARD: Dict[str, Tuple[float, Optional[float]]] = {}
    _REFRESH_COOLDOWN_SEC: float = 30.0

    @classmethod
    def _should_refresh(cls, path: str, current_mtime: Optional[float]) -> bool:
        entry = cls._REFRESH_GUARD.get(path)
        if entry is None:
            return True
        last_ts, last_mtime = entry
        if current_mtime is not None and last_mtime is not None and current_mtime > last_mtime:
            return True
        if current_mtime is None and last_mtime is not None:
            return True
        if (time.monotonic() - last_ts) > cls._REFRESH_COOLDOWN_SEC:
            return True
        return False

    @classmethod
    def _record_refresh(cls, path: str, current_mtime: Optional[float]) -> None:
        cls._REFRESH_GUARD[path] = (time.monotonic(), current_mtime)

    @staticmethod
    def _load_filters(path: str, max_age_days: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        if Quantizer is None:
            return {}, {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filters, meta = Quantizer.load_filters(
                path,
                max_age_days=max(0, int(max_age_days)),
                fatal=False,
            )
        for warn in caught:
            try:
                formatted = warnings.formatwarning(
                    warn.message, warn.category, warn.filename, warn.lineno, warn.line
                )
            except Exception:
                formatted = f"{getattr(warn.category, '__name__', 'Warning')}: {warn.message}"
            logger.warning("Quantizer warning (%s): %s", path, str(formatted).strip())
        return filters, meta

    def __init__(self, cfg: QuantizerConfig) -> None:
        self.cfg = cfg
        self._quantizer = None
        self._filters_raw: Dict[str, Dict[str, Any]] = {}
        self._symbol_filters_map: Dict[str, SymbolFilters] = {}
        self._symbol_filters_view: Mapping[str, SymbolFilters] = MappingProxyType(self._symbol_filters_map)
        self._filters_metadata: Dict[str, Any] = {}
        self._filters_metadata_view: Mapping[str, Any] = MappingProxyType(self._filters_metadata)
        self._validation_fallback_warned = False
        filters_path = cfg.resolved_filters_path()
        if not cfg.filters_path:
            cfg.filters_path = filters_path
        if Quantizer is None or not filters_path:
            return

        max_age_days = max(_as_int(cfg.auto_refresh_days, 30), 0)
        filters, meta = self._load_filters(filters_path, max_age_days)
        meta_dict: Dict[str, Any] = dict(meta or {}) if isinstance(meta, dict) else {}
        age_days = _filters_age_days(meta_dict, filters_path)
        stale = _is_stale(age_days, max_age_days)
        missing = not filters

        if cfg.refresh_on_start and (missing or stale):
            refresh_key = os.path.abspath(filters_path)
            current_mtime = _file_mtime(filters_path)
            if self._should_refresh(refresh_key, current_mtime):
                logger.info(
                    "Refreshing Binance filters: path=%s missing=%s stale=%s auto_refresh_days=%s",
                    filters_path,
                    missing,
                    stale,
                    max_age_days,
                )
                if self._refresh_filters(filters_path):
                    refreshed_mtime = _file_mtime(filters_path)
                    self._record_refresh(refresh_key, refreshed_mtime)
                    filters, meta = self._load_filters(filters_path, max_age_days)
                    meta_dict = dict(meta or {}) if isinstance(meta, dict) else {}
                    age_days = _filters_age_days(meta_dict, filters_path)
                    stale = _is_stale(age_days, max_age_days)
                    missing = not filters
                else:
                    self._record_refresh(refresh_key, current_mtime)
                    filters = {}
                    missing = True
            else:
                logger.debug(
                    "Skipping Binance filters refresh for %s; recent attempt detected",
                    filters_path,
                )

        size_bytes = _file_size_bytes(filters_path)
        sha256 = _file_sha256(filters_path) if size_bytes is not None else None
        logger.info(
            "Quantizer filters file %s: age_days=%s size_bytes=%s sha256=%s stale=%s missing=%s",
            filters_path,
            f"{age_days:.2f}" if age_days is not None else "n/a",
            size_bytes if size_bytes is not None else "n/a",
            sha256 if sha256 is not None else "n/a",
            stale,
            missing,
        )

        symbol_count = len(filters or {})
        filters_mtime = _file_mtime(filters_path)
        metadata_payload: Dict[str, Any] = {
            "path": filters_path,
            "symbol_count": symbol_count,
            "missing": bool(missing),
            "stale": bool(stale),
        }
        if age_days is not None:
            try:
                metadata_payload["age_days"] = float(age_days)
            except (TypeError, ValueError):
                pass
        if filters_mtime is not None:
            try:
                metadata_payload["mtime"] = float(filters_mtime)
                metadata_payload["mtime_iso"] = datetime.fromtimestamp(
                    float(filters_mtime), tz=timezone.utc
                ).isoformat()
            except Exception:
                metadata_payload["mtime"] = float(filters_mtime)
        if size_bytes is not None:
            try:
                metadata_payload["size_bytes"] = int(size_bytes)
            except (TypeError, ValueError):
                pass
        if sha256 is not None:
            metadata_payload["sha256"] = sha256
        if meta_dict:
            try:
                metadata_payload["source"] = dict(meta_dict)
            except Exception:
                metadata_payload["source"] = meta_dict

        self._filters_metadata.clear()
        self._filters_metadata.update(metadata_payload)
        strict_active = bool(self.cfg.strict_filters and symbol_count > 0)
        enforce_active = bool(
            self.cfg.enforce_percent_price_by_side and symbol_count > 0
        )
        enriched_metadata = self._refresh_runtime_metadata(
            strict_active=strict_active,
            enforce_active=enforce_active,
        )
        mtime_repr = (
            enriched_metadata.get("mtime_iso")
            or enriched_metadata.get("mtime")
            or "n/a"
        )
        logger.info(
            "Quantizer filters metadata: path=%s mtime=%s symbols=%s strict_filters_active=%s",
            filters_path,
            mtime_repr,
            symbol_count,
            enriched_metadata.get("strict_filters_active", False),
        )

        age = float(age_days) if age_days is not None else float("nan")
        try:
            monitoring.filters_age_days.set(age)
        except Exception:
            pass

        if not filters:
            if cfg.refresh_on_start and (missing or stale):
                logger.warning(
                    "Quantizer filters unavailable after refresh attempt; quantizer disabled (path=%s)",
                    filters_path,
                )
            else:
                logger.warning(
                    "Quantizer filters unavailable at %s; quantizer disabled",
                    filters_path,
                )
            return

        self._filters_raw = dict(filters)
        self._quantizer = Quantizer(filters, strict=bool(cfg.strict_filters))
        filters_map = getattr(self._quantizer, "_filters", None)
        if isinstance(filters_map, dict):
            self._symbol_filters_map.clear()
            self._symbol_filters_map.update(filters_map)

    @property
    def quantizer(self):
        return self._quantizer

    @property
    def symbol_filters(self) -> Mapping[str, SymbolFilters]:
        return self._symbol_filters_view

    @property
    def filters_metadata(self) -> Mapping[str, Any]:
        return self._filters_metadata_view

    def validate_order(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        ref_price: Optional[float] = None,
        enforce_ppbs: Optional[bool] = None,
    ):
        """Validate and quantize order parameters using :class:`quantizer.Quantizer`.

        When the underlying quantizer or symbol filters are unavailable the method
        returns the original values and logs a warning once, emulating permissive
        behaviour.
        """

        quantizer = self._quantizer
        enforce = self.cfg.enforce_percent_price_by_side if enforce_ppbs is None else bool(enforce_ppbs)
        ref_value = price if ref_price is None else ref_price

        if quantizer is None or not self._filters_raw:
            if not self._validation_fallback_warned:
                logger.warning(
                    "Quantizer or filters unavailable; validate_order falling back to permissive behaviour"
                )
                self._validation_fallback_warned = True
            return _make_order_check_result(
                price=float(price),
                qty=float(qty),
            )

        try:
            return quantizer.quantize_order(
                symbol,
                side,
                price,
                qty,
                ref_value,
                enforce_ppbs=bool(enforce),
            )
        except Exception as exc:
            if not self._validation_fallback_warned:
                logger.warning(
                    "Quantizer validation failed (%s); falling back to permissive behaviour",
                    exc,
                )
                self._validation_fallback_warned = True
            return _make_order_check_result(
                price=float(price),
                qty=float(qty),
                reason=None,
                reason_code=None,
            )

    def attach_to(
        self,
        sim,
        *,
        strict: Optional[bool] = None,
        enforce_percent_price_by_side: Optional[bool] = None,
    ) -> None:
        """Подключает квантайзер к симулятору."""
        if strict is not None:
            self.cfg.strict = bool(strict)
        if enforce_percent_price_by_side is not None:
            self.cfg.enforce_percent_price_by_side = bool(enforce_percent_price_by_side)

        try:
            setattr(sim, "validate_order", self.validate_order)
        except Exception:
            pass

        try:
            setattr(sim, "symbol_filters", self.symbol_filters)
        except Exception:
            pass

        quantizer = self._quantizer
        if quantizer is not None:
            try:
                setattr(sim, "quantizer", quantizer)
            except Exception:
                pass

        filters_payload: Optional[Dict[str, Dict[str, Any]]] = None
        if self._filters_raw:
            filters_payload = dict(self._filters_raw)

        strict_active = bool(self.cfg.strict_filters and filters_payload is not None)
        enforce_active = bool(
            self.cfg.enforce_percent_price_by_side and filters_payload is not None
        )
        metadata_view = self._refresh_runtime_metadata(
            strict_active=strict_active,
            enforce_active=enforce_active,
        )
        metadata_for_sim = dict(metadata_view) if metadata_view else {}

        try:
            setattr(sim, "quantize_mode", str(self.cfg.quantize_mode))
        except Exception:
            pass
        if metadata_for_sim:
            try:
                setattr(sim, "quantizer_metadata", dict(metadata_for_sim))
            except Exception:
                pass

        attach_api = getattr(sim, "attach_quantizer", None)
        if callable(attach_api):
            try:
                attach_api(
                    quantizer=quantizer,
                    filters=filters_payload,
                    strict_filters=strict_active,
                    enforce_ppbs=enforce_active,
                    metadata=dict(metadata_for_sim) if metadata_for_sim else None,
                    quantize_mode=str(self.cfg.quantize_mode),
                )
            except TypeError:
                logger.debug(
                    "Simulator %s.attach_quantizer signature mismatch; falling back to legacy attachment",
                    type(sim).__name__,
                )
            except Exception as exc:
                logger.warning(
                    "Simulator %s.attach_quantizer failed: %s; falling back to legacy attachment",
                    type(sim).__name__,
                    exc,
                )
            else:
                return

        filters_attached = False
        warn_message: Optional[str] = None
        if filters_payload is not None:
            if hasattr(sim, "filters"):
                try:
                    setattr(sim, "filters", dict(filters_payload))
                    filters_attached = True
                except Exception as exc:
                    warn_message = (
                        f"Failed to attach quantizer filters to {type(sim).__name__}: {exc}"
                    )
            else:
                warn_message = (
                    f"Simulator {type(sim).__name__} has no 'filters' attribute; strict filter enforcement disabled"
                )
        else:
            warn_message = (
                f"Quantizer filters are unavailable; permissive mode enabled for {type(sim).__name__}"
            )

        if warn_message:
            logger.warning(warn_message)

        try:
            setattr(sim, "enforce_ppbs", enforce_active if filters_attached else False)
        except Exception:
            pass
        try:
            setattr(sim, "strict_filters", strict_active if filters_attached else False)
        except Exception:
            pass

        if metadata_for_sim:
            try:
                metadata_for_sim = dict(metadata_for_sim)
                metadata_for_sim["strict_filters_active"] = bool(
                    strict_active if filters_attached else False
                )
                metadata_for_sim["enforce_percent_price_by_side_active"] = bool(
                    enforce_active if filters_attached else False
                )
                setattr(sim, "quantizer_metadata", metadata_for_sim)
            except Exception:
                pass

        self._refresh_runtime_metadata(
            strict_active=bool(strict_active if filters_attached else False),
            enforce_active=bool(enforce_active if filters_attached else False),
        )

    def _refresh_runtime_metadata(
        self,
        *,
        strict_active: bool,
        enforce_active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        metadata = dict(self._filters_metadata)
        if enforce_active is None:
            enforce_active = bool(
                self.cfg.enforce_percent_price_by_side and strict_active
            )
        metadata.update(
            {
                "strict_filters": bool(self.cfg.strict_filters),
                "strict_filters_active": bool(strict_active),
                "enforce_percent_price_by_side": bool(
                    self.cfg.enforce_percent_price_by_side
                ),
                "enforce_percent_price_by_side_active": bool(enforce_active),
                "quantize_mode": str(self.cfg.quantize_mode),
            }
        )
        self._filters_metadata.clear()
        self._filters_metadata.update(metadata)
        return metadata

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QuantizerImpl":
        path = str(d.get("path", ""))
        filters_path_raw = d.get("filters_path")
        if filters_path_raw is None:
            filters_path = path
        else:
            filters_path_str = str(filters_path_raw).strip()
            filters_path = filters_path_str or path
        auto_refresh = max(_as_int(d.get("auto_refresh_days", 30), 30), 0)
        refresh_on_start = _as_bool(d.get("refresh_on_start"), False)

        strict_filters_raw = d.get("strict_filters")
        if strict_filters_raw is None and "strict" in d:
            strict_filters_raw = d.get("strict")
        strict_filters = _as_bool(strict_filters_raw, True)

        quantize_mode_raw = d.get("quantize_mode", "inward")
        quantize_mode = str(quantize_mode_raw).strip() or "inward"

        enforce_ppbs = _as_bool(d.get("enforce_percent_price_by_side"), True)

        return QuantizerImpl(QuantizerConfig(
            path=path,
            strict_filters=bool(strict_filters),
            quantize_mode=quantize_mode,
            enforce_percent_price_by_side=bool(enforce_ppbs),
            filters_path=filters_path,
            auto_refresh_days=auto_refresh,
            refresh_on_start=refresh_on_start,
        ))

    @staticmethod
    def _refresh_filters(out_path: str) -> bool:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "scripts", "fetch_binance_filters.py")
        universe_path = os.path.join(script_dir, "data", "universe", "symbols.json")
        cmd = [
            "python",
            script_path,
            "--universe",
            universe_path,
            "--out",
            out_path,
        ]

        try:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        except OSError:
            pass

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            message = exc.stderr or exc.stdout or str(exc)
            logger.warning(
                "Failed to refresh Binance filters via '%s' (code=%s): %s",
                " ".join(cmd),
                exc.returncode,
                message.strip(),
            )
            return False
        except Exception as exc:
            logger.warning(
                "Failed to execute Binance filters refresh '%s': %s",
                " ".join(cmd),
                exc,
            )
            return False

        if result.stdout:
            logger.debug("fetch_binance_filters stdout: %s", result.stdout.strip())
        if result.stderr:
            logger.debug("fetch_binance_filters stderr: %s", result.stderr.strip())
        return True
