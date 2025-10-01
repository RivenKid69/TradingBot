"""Runtime helpers for accessing ADV/turnover datasets."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


def _cfg_attr(block: Any, key: str, default: Any = None) -> Any:
    if block is None:
        return default
    if isinstance(block, Mapping):
        return block.get(key, default)
    return getattr(block, key, default)


def _safe_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _safe_positive_int(value: Any) -> Optional[int]:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    return num


def _normalise_policy(value: Any) -> str:
    if value is None:
        return "warn"
    try:
        policy = str(value).strip().lower()
    except Exception:
        return "warn"
    if policy not in {"warn", "skip", "error"}:
        return "warn"
    return policy


class ADVStore:
    """Load, cache and serve ADV quotes for runtime consumers."""

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self._path = self._resolve_path(cfg)
        self._refresh_days = _safe_positive_int(_cfg_attr(cfg, "refresh_days"))
        self._default_quote = _safe_float(_cfg_attr(cfg, "default_quote"))
        self._floor_quote = _safe_float(_cfg_attr(cfg, "floor_quote"))
        self._missing_policy = _normalise_policy(
            _cfg_attr(cfg, "missing_symbol_policy", "warn")
        )
        self._cache: Dict[str, float] = {}
        self._meta: Dict[str, Any] = {}
        self._mtime: float | None = None
        self._stale = False
        self._lock = threading.Lock()
        self._missing_logged: set[str] = set()
        self._bar_cache: dict[str, dict[str, Any]] = {}
        self._bar_meta_cache: dict[str, dict[str, Any]] = {}
        self._bar_cache_mtime: dict[str, float] = {}
        self._bar_cache_order: list[str] = []
        self._bar_cache_limit = _safe_positive_int(_cfg_attr(cfg, "bar_cache_limit"))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def default_quote(self) -> Optional[float]:
        return self._default_quote if self._default_quote and self._default_quote > 0.0 else None

    @property
    def floor_quote(self) -> Optional[float]:
        return self._floor_quote if self._floor_quote and self._floor_quote > 0.0 else None

    @property
    def missing_symbol_policy(self) -> str:
        return self._missing_policy

    @property
    def metadata(self) -> Mapping[str, Any]:
        with self._lock:
            return dict(self._meta)

    @property
    def is_dataset_stale(self) -> bool:
        with self._lock:
            return bool(self._stale)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, cfg: Any) -> Optional[str]:
        candidates: list[str] = []
        raw_path = _cfg_attr(cfg, "path")
        if raw_path:
            try:
                text = str(raw_path).strip()
            except Exception:
                text = ""
            if text:
                candidates.append(text)
        extra = _cfg_attr(cfg, "extra")
        if isinstance(extra, Mapping):
            for key in ("quote_path", "adv_path", "data_path", "path"):
                candidate = extra.get(key)
                if candidate:
                    try:
                        text = str(candidate).strip()
                    except Exception:
                        text = ""
                    if text:
                        candidates.append(text)
        dataset_name = None
        raw_dataset = _cfg_attr(cfg, "dataset")
        if raw_dataset is not None:
            try:
                dataset_name = str(raw_dataset).strip()
            except Exception:
                dataset_name = None
            if dataset_name == "":
                dataset_name = None
        for base in candidates:
            if dataset_name and os.path.isdir(base):
                candidate = os.path.join(base, dataset_name)
                if os.path.exists(candidate):
                    return candidate
            if dataset_name and dataset_name not in os.path.basename(base):
                candidate = os.path.join(base, dataset_name)
                if os.path.exists(candidate):
                    return candidate
            if os.path.exists(base) and (not dataset_name or not os.path.isdir(base)):
                return base
        return None

    def _ensure_loaded_locked(self) -> None:
        path = self._path
        if not path:
            self._cache.clear()
            self._meta = {}
            self._stale = False
            return
        try:
            mtime = os.path.getmtime(path)
        except (OSError, TypeError, ValueError):
            if not self._cache:
                logger.warning("ADV dataset file %s is not accessible", path)
            self._stale = False
            return
        if self._mtime is not None and self._cache and mtime <= self._mtime:
            self._check_refresh_locked()
            return
        payload = self._read_payload(path)
        if payload is None:
            self._check_refresh_locked()
            return
        data, meta = payload
        self._cache = data
        self._meta = meta
        self._mtime = mtime
        self._check_refresh_locked()

    def _read_payload(self, path: str) -> tuple[dict[str, float], dict[str, Any]] | None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            logger.warning("ADV dataset file %s not found", path)
            return {}, {}
        except Exception:
            logger.exception("Failed to load ADV dataset from %s", path)
            return None
        if not isinstance(payload, Mapping):
            logger.warning("ADV dataset %s must be a JSON object", path)
            return {}, {}
        meta_raw = payload.get("meta")
        data_raw = payload.get("data")
        if not isinstance(data_raw, Mapping):
            logger.warning("ADV dataset %s is missing 'data' mapping", path)
            data_raw = {}
        dataset: dict[str, float] = {}
        for key, value in data_raw.items():
            try:
                symbol = str(key).strip().upper()
            except Exception:
                continue
            if not symbol:
                continue
            if isinstance(value, Mapping):
                candidate = value.get("adv_quote")
            else:
                candidate = value
            adv_val = _safe_float(candidate)
            if adv_val is None or adv_val <= 0.0:
                continue
            dataset[symbol] = float(adv_val)
        meta: dict[str, Any] = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}
        meta.setdefault("path", path)
        meta.setdefault("symbol_count", len(dataset))
        return dataset, meta

    def _extract_timestamp_locked(self) -> Optional[int]:
        meta = self._meta
        candidates = [
            meta.get("generated_at_ms"),
            meta.get("generated_ms"),
            meta.get("timestamp_ms"),
            meta.get("end_ms"),
        ]
        for candidate in candidates:
            ts_val = _safe_positive_int(candidate)
            if ts_val is not None:
                return ts_val
        for key in ("generated_at", "end_at"):
            value = meta.get(key)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text:
                continue
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        if self._mtime is not None:
            return int(self._mtime * 1000)
        return None

    def _check_refresh_locked(self) -> None:
        refresh_days = self._refresh_days
        if not refresh_days:
            self._stale = False
            return
        ts_ms = self._extract_timestamp_locked()
        if ts_ms is None:
            self._stale = True
            logger.warning("ADV dataset %s lacks timestamp metadata", self._path)
            return
        age_ms = int(time.time() * 1000) - ts_ms
        if age_ms <= 0:
            self._stale = False
            return
        age_days = age_ms / 86_400_000
        if age_days > float(refresh_days):
            self._stale = True
            logger.warning(
                "ADV dataset %s is older than %.1f days (threshold=%s)",
                self._path,
                age_days,
                refresh_days,
            )
        else:
            self._stale = False

    def _handle_missing_symbol(self, symbol: str) -> None:
        if symbol in self._missing_logged:
            return
        policy = self._missing_policy
        message = "ADV quote missing for %s (policy=%s)"
        if policy == "error":
            logger.error(message, symbol, policy)
        elif policy == "warn":
            logger.warning(message, symbol, policy)
        else:
            logger.debug(message, symbol, policy)
        self._missing_logged.add(symbol)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset_runtime_state(self) -> None:
        with self._lock:
            self._missing_logged.clear()

    def get_adv_quote(self, symbol: str) -> Optional[float]:
        if not symbol:
            return None
        try:
            sym_key = str(symbol).strip().upper()
        except Exception:
            return None
        if not sym_key:
            return None
        with self._lock:
            self._ensure_loaded_locked()
            if self._stale:
                return None
            if not self._cache:
                return None
            value = self._cache.get(sym_key)
        if value is None:
            self._handle_missing_symbol(sym_key)
            return None
        return float(value)

    def get_bar_capacity_quote(self, symbol: str) -> Optional[float]:
        # For now this mirrors ``get_adv_quote``. Kept separate to allow
        # future extensions (e.g. per-bar aggregation) without changing call sites.
        quote = self.get_adv_quote(symbol)
        if quote is None:
            default_quote = self.default_quote
            if default_quote is None:
                return None
            quote = float(default_quote)
        floor_quote = self.floor_quote
        if floor_quote is not None and quote < floor_quote:
            return float(floor_quote)
        return float(quote)

    # ------------------------------------------------------------------
    # Bar dataset helpers
    # ------------------------------------------------------------------
    def _bar_cache_key(self, symbol: Any) -> str:
        try:
            text = str(symbol).strip().upper()
        except Exception:
            return ""
        return text

    def _extract_bar_meta(
        self, payload: Mapping[str, Any] | None, path: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(payload, Mapping):
            return {}, {"path": path, "symbol_count": 0}
        data_raw = payload.get("bars")
        if not isinstance(data_raw, Mapping):
            data_raw = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
        dataset: dict[str, Any] = {}
        for key, value in data_raw.items():
            symbol = self._bar_cache_key(key)
            if not symbol:
                continue
            dataset[symbol] = value
        meta_raw = payload.get("meta")
        meta: dict[str, Any] = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}
        meta.setdefault("path", path)
        meta["symbol_count"] = len(dataset)
        return dataset, meta

    def _drop_bar_cache_entry(self, cache_key: str) -> None:
        self._bar_cache.pop(cache_key, None)
        self._bar_meta_cache.pop(cache_key, None)
        self._bar_cache_mtime.pop(cache_key, None)
        try:
            self._bar_cache_order.remove(cache_key)
        except ValueError:
            pass

    def _trim_bar_cache(self) -> None:
        limit = self._bar_cache_limit
        if not limit:
            return
        while len(self._bar_cache_order) > limit:
            oldest = self._bar_cache_order.pop(0)
            self._drop_bar_cache_entry(oldest)

    def _maybe_reset_bar_cache(self, path: str) -> None:
        cache_key = os.path.abspath(path)
        if cache_key not in self._bar_cache:
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            self._drop_bar_cache_entry(cache_key)
            return
        cached_mtime = self._bar_cache_mtime.get(cache_key)
        if cached_mtime is None or mtime != cached_mtime:
            self._drop_bar_cache_entry(cache_key)

    def _read_bar_payload(self, path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            logger.warning("Bar dataset %s not found", path)
            return {}, {"path": path, "symbol_count": 0}
        except Exception:
            logger.exception("Failed to read bar dataset from %s", path)
            return {}, {"path": path, "symbol_count": 0}
        return self._extract_bar_meta(payload, path)

    def _load_bar_dataset_locked(
        self, path: str
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if not path:
            return {}, {}
        cache_key = os.path.abspath(path)
        self._maybe_reset_bar_cache(path)
        dataset = self._bar_cache.get(cache_key)
        if dataset is not None:
            # Refresh order to behave like LRU.
            try:
                self._bar_cache_order.remove(cache_key)
            except ValueError:
                pass
            self._bar_cache_order.append(cache_key)
            meta = self._bar_meta_cache.get(cache_key, {})
            return dataset, dict(meta)
        dataset, meta = self._read_bar_payload(path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = time.time()
        cache_copy = dict(dataset)
        meta_copy = dict(meta)
        self._bar_cache[cache_key] = cache_copy
        self._bar_meta_cache[cache_key] = meta_copy
        self._bar_cache_mtime[cache_key] = mtime
        self._bar_cache_order.append(cache_key)
        self._trim_bar_cache()
        return dict(cache_copy), dict(meta_copy)

    def load_bar_dataset(self, path: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        with self._lock:
            return self._load_bar_dataset_locked(path)

    def get_bar_entry(self, path: str, symbol: Any) -> Any:
        dataset, _ = self.load_bar_dataset(path)
        key = self._bar_cache_key(symbol)
        if not key:
            return None
        return dataset.get(key)


__all__ = ["ADVStore"]
