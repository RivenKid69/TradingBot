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
import json
import logging
import os
import subprocess
import sys
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


def _filters_age_days(meta: Mapping[str, Any] | None, path: str) -> Optional[float]:
    ts: Optional[datetime] = None
    metadata_sources: Tuple[Mapping[str, Any], ...] = ()
    if isinstance(meta, Mapping):
        base_meta: Mapping[str, Any] = meta
        nested_meta = base_meta.get("metadata") if isinstance(base_meta, Mapping) else None
        if isinstance(nested_meta, Mapping):
            metadata_sources = (nested_meta, base_meta)
        else:
            metadata_sources = (base_meta,)
    for source in metadata_sources:
        for key in ("built_at", "builtAt", "generated_at", "generatedAt"):
            ts = _parse_timestamp(source.get(key))
            if ts is not None:
                break
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
    filters_path: Optional[str] = ""
    auto_refresh_days: Optional[int] = 30
    refresh_on_start: Optional[bool] = False

    def resolved_filters_path(self) -> str:
        candidate = self.filters_path
        if isinstance(candidate, str):
            candidate = candidate.strip()
        else:
            candidate = ""
        return candidate or self.path

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
    def _load_filters(
        path: str, max_age_days: Optional[int]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        if not path:
            return {}, {}

        def _normalize_payload(
            filters_raw: Any, metadata_raw: Any
        ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
            filters_map: Dict[str, Dict[str, Any]] = {}
            if isinstance(filters_raw, Mapping):
                try:
                    filters_map = {str(sym): dict(f or {}) for sym, f in filters_raw.items()}
                except Exception:
                    filters_map = {str(sym): f for sym, f in filters_raw.items()}  # type: ignore[assignment]
            metadata_map: Dict[str, Any] = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
            return filters_map, metadata_map

        if Quantizer is not None:
            max_age = 30
            if max_age_days is not None:
                try:
                    max_age = max(0, int(max_age_days))
                except Exception:
                    max_age = 30
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    loaded = Quantizer.load_filters(
                        path,
                        max_age_days=max_age,
                        fatal=False,
                    )
                except TypeError:
                    loaded = Quantizer.load_filters(path)  # type: ignore[misc]
            for warn in caught:
                try:
                    formatted = warnings.formatwarning(
                        warn.message, warn.category, warn.filename, warn.lineno, warn.line
                    )
                except Exception:
                    formatted = f"{getattr(warn.category, '__name__', 'Warning')}: {warn.message}"
                logger.warning("Quantizer warning (%s): %s", path, str(formatted).strip())

            if isinstance(loaded, tuple):
                filters_raw, meta_raw = loaded
            else:
                filters_raw, meta_raw = loaded, {}
            return _normalize_payload(filters_raw, meta_raw)

        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh) or {}
        except FileNotFoundError:
            return {}, {}
        except Exception as exc:
            logger.warning("Failed to load quantizer filters from %s: %s", path, exc)
            return {}, {}

        if isinstance(payload, Mapping):
            filters_block = payload.get("filters") if "filters" in payload else payload
            metadata_block = payload.get("metadata")
        else:
            filters_block = {}
            metadata_block = {}

        return _normalize_payload(filters_block, metadata_block)

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

        auto_refresh_days = cfg.auto_refresh_days
        if auto_refresh_days is None:
            auto_refresh_days = 30
        auto_refresh_days = max(_as_int(auto_refresh_days, 30), 0)
        cfg.auto_refresh_days = auto_refresh_days

        refresh_on_start = _as_bool(cfg.refresh_on_start, False)
        cfg.refresh_on_start = refresh_on_start

        filters: Dict[str, Dict[str, Any]] = {}
        meta_dict: Dict[str, Any] = {}
        age_days: Optional[float] = None
        stale = False
        missing = True
        refresh_requested = False
        refresh_executed = False
        refresh_succeeded = False
        refresh_returncode: Optional[int] = None
        refresh_message: Optional[str] = None
        status: str = "ready"
        status_reason: Optional[str] = None

        metadata_payload: Dict[str, Any] = {
            "config_path": str(cfg.path or ""),
            "path": filters_path,
            "resolved_path": filters_path,
            "auto_refresh_days": auto_refresh_days,
            "refresh_on_start": bool(refresh_on_start),
        }

        if not filters_path:
            status = "missing"
            status_reason = "Filters path is not configured"
        else:
            filters, meta = self._load_filters(filters_path, auto_refresh_days)
            meta_dict = dict(meta or {}) if isinstance(meta, dict) else {}
            age_days = _filters_age_days(meta_dict, filters_path)
            stale = _is_stale(age_days, auto_refresh_days)
            missing = not filters

            if Quantizer is None:
                status = "error"
                status_reason = "Quantizer module is unavailable"
            else:
                if refresh_on_start and (missing or stale):
                    refresh_requested = True
                    logger.info(
                        "Refreshing Binance filters: path=%s missing=%s stale=%s auto_refresh_days=%s",
                        filters_path,
                        missing,
                        stale,
                        auto_refresh_days,
                    )
                    executed, refresh_succeeded, refresh_returncode, refresh_msg = self._refresh_filters(
                        filters_path
                    )
                    refresh_executed = executed
                    if refresh_msg:
                        refresh_message = refresh_msg
                    if refresh_succeeded:
                        filters, meta = self._load_filters(filters_path, auto_refresh_days)
                        meta_dict = dict(meta or {}) if isinstance(meta, dict) else {}
                        age_days = _filters_age_days(meta_dict, filters_path)
                        stale = _is_stale(age_days, auto_refresh_days)
                        missing = not filters

                if missing:
                    status = "missing"
                    reason_parts = [
                        f"Filters unavailable at {filters_path}" if filters_path else "Filters unavailable"
                    ]
                    if refresh_message and not refresh_succeeded:
                        reason_parts.append(str(refresh_message))
                    status_reason = "; ".join(reason_parts)
                elif stale:
                    status = "stale"
                    if auto_refresh_days > 0:
                        if age_days is not None:
                            status_reason = (
                                f"Filters age {age_days:.2f}d exceeds {auto_refresh_days}d threshold"
                            )
                        else:
                            status_reason = (
                                f"Filters staleness exceeds {auto_refresh_days}d threshold"
                            )
                        if refresh_message and not refresh_succeeded:
                            status_reason = f"{status_reason}; {refresh_message}"

        size_bytes = _file_size_bytes(filters_path) if filters_path else None
        sha256 = _file_sha256(filters_path) if size_bytes is not None else None
        logger.info(
            "Quantizer filters file %s: age_days=%s size_bytes=%s sha256=%s stale=%s missing=%s",
            filters_path or "<unset>",
            f"{age_days:.2f}" if age_days is not None else "n/a",
            size_bytes if size_bytes is not None else "n/a",
            sha256 if sha256 is not None else "n/a",
            stale,
            missing,
        )

        symbol_count = len(filters or {})
        filters_mtime = _file_mtime(filters_path) if filters_path else None
        metadata_payload.update(
            {
                "symbol_count": symbol_count,
                "missing": bool(missing),
                "stale": bool(stale),
                "refresh_requested": bool(refresh_requested),
                "refresh_executed": bool(refresh_executed),
                "refresh_succeeded": bool(refresh_succeeded),
                "status": status,
            }
        )
        if status_reason:
            metadata_payload["status_reason"] = status_reason
        if refresh_returncode is not None:
            try:
                metadata_payload["refresh_returncode"] = int(refresh_returncode)
            except (TypeError, ValueError):
                metadata_payload["refresh_returncode"] = refresh_returncode
        if refresh_message:
            metadata_payload["refresh_message"] = str(refresh_message)
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
            filters_path or "<unset>",
            mtime_repr,
            symbol_count,
            enriched_metadata.get("strict_filters_active", False),
        )

        age = float(age_days) if age_days is not None else float("nan")
        try:
            monitoring.filters_age_days.set(age)
        except Exception:
            pass

        if not filters or Quantizer is None:
            warn_reason = (
                enriched_metadata.get("status_reason")
                if isinstance(enriched_metadata, Mapping)
                else status_reason
            )
            status_label = (
                enriched_metadata.get("status")
                if isinstance(enriched_metadata, Mapping)
                else status
            )
            if warn_reason:
                logger.warning(
                    "Quantizer filters unavailable (status=%s reason=%s path=%s); quantizer disabled",
                    status_label or "unknown",
                    warn_reason,
                    filters_path or "<unset>",
                )
            elif cfg.refresh_on_start and (missing or stale):
                logger.warning(
                    "Quantizer filters unavailable after refresh attempt; quantizer disabled (path=%s)",
                    filters_path or "<unset>",
                )
            else:
                logger.warning(
                    "Quantizer filters unavailable at %s; quantizer disabled",
                    filters_path or "<unset>",
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

        quantizer = self._quantizer
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

        attach_api = getattr(sim, "attach_quantizer", None)
        if callable(attach_api):
            try:
                attach_api(
                    impl=self,
                    metadata=dict(metadata_for_sim) if metadata_for_sim else None,
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

        try:
            setattr(sim, "validate_order", self.validate_order)
        except Exception:
            pass

        try:
            setattr(sim, "symbol_filters", self.symbol_filters)
        except Exception:
            pass

        if quantizer is not None:
            try:
                setattr(sim, "quantizer", quantizer)
            except Exception:
                pass

        try:
            setattr(sim, "quantize_mode", str(self.cfg.quantize_mode))
        except Exception:
            pass
        if metadata_for_sim:
            try:
                setattr(sim, "quantizer_metadata", dict(metadata_for_sim))
            except Exception:
                pass

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
        status = metadata.get("status")
        if strict_active:
            metadata["permissive_mode"] = False
            if status in (None, "permissive"):
                metadata["status"] = "ready"
                reason = metadata.get("status_reason")
                if reason == "Strict filter enforcement disabled":
                    metadata.pop("status_reason", None)
        else:
            metadata["permissive_mode"] = True
            if status in (None, "ready", "permissive"):
                metadata["status"] = "permissive"
                metadata.setdefault("status_reason", "Strict filter enforcement disabled")
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
        data = dict(d or {})
        path = str(data.get("path", "")).strip()

        filters_cfg = data.get("filters")
        filters_map = filters_cfg if isinstance(filters_cfg, Mapping) else {}

        filters_path: Optional[str] = None
        for candidate in (
            data.get("filters_path"),
            data.get("filtersPath"),
            filters_map.get("path") if isinstance(filters_map, Mapping) else None,
        ):
            if candidate is None:
                continue
            candidate_str = str(candidate).strip()
            if candidate_str:
                filters_path = candidate_str
                break

        auto_refresh_days_value: Optional[int] = None
        for candidate in (
            data.get("auto_refresh_days"),
            data.get("autoRefreshDays"),
            filters_map.get("auto_refresh_days") if isinstance(filters_map, Mapping) else None,
            filters_map.get("autoRefreshDays") if isinstance(filters_map, Mapping) else None,
        ):
            if candidate is None:
                continue
            auto_refresh_days_value = max(_as_int(candidate, 30), 0)
            break

        refresh_on_start_value: Optional[bool] = None
        for candidate in (
            data.get("refresh_on_start"),
            data.get("refreshOnStart"),
            filters_map.get("refresh_on_start") if isinstance(filters_map, Mapping) else None,
            filters_map.get("refreshOnStart") if isinstance(filters_map, Mapping) else None,
        ):
            if candidate is None:
                continue
            refresh_on_start_value = _as_bool(candidate, False)
            break

        strict_filters_raw = data.get("strict_filters")
        if strict_filters_raw is None:
            strict_filters_raw = data.get("strict")
        if strict_filters_raw is None and isinstance(filters_map, Mapping):
            strict_filters_raw = filters_map.get("strict_filters", filters_map.get("strict"))
        strict_filters = _as_bool(strict_filters_raw, True)

        quantize_mode_raw = data.get("quantize_mode", "inward")
        quantize_mode = str(quantize_mode_raw).strip() or "inward"

        enforce_ppbs_candidate = data.get("enforce_percent_price_by_side")
        if enforce_ppbs_candidate is None and isinstance(filters_map, Mapping):
            enforce_ppbs_candidate = filters_map.get("enforce_percent_price_by_side")
        enforce_ppbs = _as_bool(enforce_ppbs_candidate, True)

        cfg_kwargs: Dict[str, Any] = {
            "path": path,
            "strict_filters": bool(strict_filters),
            "quantize_mode": quantize_mode,
            "enforce_percent_price_by_side": bool(enforce_ppbs),
        }
        if filters_path is not None:
            cfg_kwargs["filters_path"] = filters_path
        if auto_refresh_days_value is not None:
            cfg_kwargs["auto_refresh_days"] = auto_refresh_days_value
        if refresh_on_start_value is not None:
            cfg_kwargs["refresh_on_start"] = refresh_on_start_value

        return QuantizerImpl(QuantizerConfig(**cfg_kwargs))

    @classmethod
    def _refresh_filters(
        cls, out_path: str
    ) -> Tuple[bool, bool, Optional[int], Optional[str]]:
        resolved_out = os.path.abspath(out_path)
        refresh_key = resolved_out
        current_mtime = _file_mtime(resolved_out)

        if not cls._should_refresh(refresh_key, current_mtime):
            logger.debug(
                "Skipping Binance filters refresh for %s; cooldown active",
                resolved_out,
            )
            return False, False, None, "Refresh skipped due to cooldown"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        universe_path = os.path.join(script_dir, "data", "universe", "symbols.json")
        python_exe = sys.executable or "python"
        cmd = [
            python_exe,
            "-m",
            "scripts.fetch_binance_filters",
            "--universe",
            universe_path,
            "--out",
            resolved_out,
        ]

        try:
            out_dir = os.path.dirname(resolved_out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            logger.debug("Failed to ensure directory for %s: %s", resolved_out, exc)

        cmd_display = " ".join(cmd)
        logger.info("Executing Binance filters refresh: %s", cmd_display)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            logger.warning(
                "Failed to execute Binance filters refresh '%s': %s",
                cmd_display,
                exc,
            )
            cls._record_refresh(refresh_key, current_mtime)
            return True, False, None, f"Refresh command failed: {exc}"

        if result.stdout:
            logger.debug("fetch_binance_filters stdout: %s", result.stdout.strip())
        if result.stderr:
            logger.debug("fetch_binance_filters stderr: %s", result.stderr.strip())

        returncode = result.returncode
        if returncode != 0:
            message = result.stderr or result.stdout or "<no output>"
            logger.warning(
                "Binance filters refresh failed (code=%s): %s",
                returncode,
                message.strip(),
            )
            cls._record_refresh(refresh_key, current_mtime)
            return True, False, returncode, f"Refresh failed with code {returncode}"

        refreshed_mtime = _file_mtime(resolved_out)
        cls._record_refresh(refresh_key, refreshed_mtime)
        logger.info(
            "Binance filters refresh completed (code=%s, path=%s)",
            returncode,
            resolved_out,
        )
        return True, True, returncode, None
