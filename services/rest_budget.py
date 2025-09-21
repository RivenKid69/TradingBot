"""Helpers for budgeting REST API requests."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import math
import random
import tempfile
import threading
import time
import urllib.parse
import weakref
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, TypeVar

from pathlib import Path

import requests

from core_config import RetryConfig, TokenBucketConfig

from . import monitoring
from .retry import retry_sync


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Return ``getattr``/``get`` result for ``name`` from ``obj``."""

    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class TokenBucket:
    """Simple token bucket limiter operating on :func:`time.monotonic`."""

    rps: float
    burst: float
    tokens: float | None = None
    last_ts: float = field(default_factory=time.monotonic)
    cooldown_until: float = 0.0
    configured_rps: float = field(init=False)
    configured_burst: float = field(init=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.rps = float(self.rps)
        self.burst = float(self.burst)
        self.configured_rps = self.rps
        self.configured_burst = self.burst
        tokens = float(self.burst if self.tokens is None else self.tokens)
        enabled = self.rps > 0.0 and self.burst > 0.0
        with self._lock:
            self.enabled = enabled
            self.tokens = float(self.burst if not enabled else tokens)
            self.last_ts = float(self.last_ts)
            self.cooldown_until = float(self.cooldown_until)

    def _refill(self, now: float) -> None:
        if not self.enabled:
            self.last_ts = now
            return
        elapsed = max(0.0, now - self.last_ts)
        if elapsed > 0.0:
            self.tokens = min(self.burst, self.tokens + elapsed * self.rps)
            self.last_ts = now

    def wait_time(self, tokens: float = 1.0, now: float | None = None) -> float:
        """Return seconds to wait until ``tokens`` can be consumed."""

        if not self.enabled:
            return 0.0
        now = float(time.monotonic() if now is None else now)
        with self._lock:
            self._refill(now)
            if now < self.cooldown_until:
                return self.cooldown_until - now
            tokens = float(tokens)
            if self.tokens >= tokens:
                return 0.0
            if self.rps <= 0.0:
                return float("inf")
            deficit = tokens - self.tokens
            return max(deficit / self.rps, 0.0)

    def consume(self, tokens: float = 1.0, now: float | None = None) -> None:
        """Consume ``tokens`` if available, assuming :meth:`wait_time` was 0."""

        if not self.enabled:
            return
        now = float(time.monotonic() if now is None else now)
        with self._lock:
            self._refill(now)
            if now < self.cooldown_until:
                raise RuntimeError("cooldown in effect")
            tokens = float(tokens)
            if self.tokens < tokens:
                raise RuntimeError("insufficient tokens")
            self.tokens -= tokens

    def start_cooldown(self, seconds: float, now: float | None = None) -> None:
        """Start (or extend) cooldown for ``seconds`` seconds."""

        seconds = float(seconds)
        if seconds <= 0.0:
            return
        now = float(time.monotonic() if now is None else now)
        with self._lock:
            self.cooldown_until = max(self.cooldown_until, now + seconds)

    def adjust_rate(
        self,
        *,
        rps: float | None = None,
        burst: float | None = None,
        update_configured: bool = False,
    ) -> None:
        """Adjust the bucket limits while keeping state consistent."""

        new_rps = self.rps if rps is None else max(float(rps), 0.0)
        new_burst = self.burst if burst is None else max(float(burst), 0.0)
        with self._lock:
            self.rps = new_rps
            self.burst = new_burst
            if update_configured:
                self.configured_rps = new_rps
                self.configured_burst = new_burst
            self.enabled = self.rps > 0.0 and self.burst > 0.0
            current_tokens = float(self.tokens if self.tokens is not None else 0.0)
            if not self.enabled:
                self.tokens = float(self.burst)
            else:
                self.tokens = min(current_tokens, self.burst)


class RestBudgetSession:
    """Requests session with token-bucket budgeting and retry logic."""

    def __init__(
        self,
        cfg: Any,
        *,
        session: requests.Session | None = None,
        rng: random.Random | None = None,
        sleep: Callable[[float], None] = time.sleep,
        cache_dir: str | os.PathLike[str] | None = None,
        ttl_days: float | int | None = None,
        mode: str | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        checkpoint_enabled: bool | None = None,
        resume_from_checkpoint: bool | None = None,
    ) -> None:
        self._cfg = cfg
        self._base_session = session or requests.Session()
        self._owns_session = session is None
        self._thread_local = threading.local()
        self._thread_local.last_response_metadata = None
        self._session_lock = threading.Lock()
        self._child_sessions: "weakref.WeakSet[requests.Session]" = weakref.WeakSet()

        enabled_flag = self._interpret_bool(_get_attr(cfg, "enabled", None))
        self._enabled = True if enabled_flag is None else bool(enabled_flag)

        concurrency_cfg = _get_attr(cfg, "concurrency", None)
        workers_val = self._coerce_positive_int(
            _get_attr(concurrency_cfg, "workers", None)
        )
        self._max_workers = workers_val
        batch_val = self._coerce_positive_int(_get_attr(cfg, "batch_size", None))
        self.batch_size = batch_val

        self._executor: ThreadPoolExecutor | None = None
        self._task_semaphore: threading.BoundedSemaphore | None = None
        self._queue_capacity = 0
        if workers_val > 0:
            self._executor = ThreadPoolExecutor(
                max_workers=workers_val, thread_name_prefix="rest-budget"
            )
            queue_capacity = batch_val if batch_val > 0 else workers_val
            self._task_semaphore = threading.BoundedSemaphore(queue_capacity)
            self._queue_capacity = queue_capacity

        self._rng = rng or random.Random()
        self._rng_lock = threading.Lock()
        self._sleep = sleep
        self._stats_lock = threading.Lock()

        cache_cfg = _get_attr(cfg, "cache", None)

        cache_dir_value = cache_dir
        if cache_dir_value is None and cache_cfg is not None:
            cache_dir_value = (
                _get_attr(cache_cfg, "dir", None)
                or _get_attr(cache_cfg, "path", None)
                or _get_attr(cache_cfg, "cache_dir", None)
            )
        if cache_dir_value is None:
            cache_dir_value = _get_attr(cfg, "cache_dir", None)
        self._cache_dir = Path(cache_dir_value).expanduser() if cache_dir_value else None

        ttl_value = ttl_days
        if ttl_value is None and cache_cfg is not None:
            ttl_value = _get_attr(cache_cfg, "ttl_days", None)
            if ttl_value is None:
                ttl_value = _get_attr(cache_cfg, "ttl", None)
        if ttl_value is None:
            ttl_value = _get_attr(cfg, "cache_ttl_days", None)
            if ttl_value is None:
                ttl_value = _get_attr(cfg, "ttl_days", None)
        self._cache_ttl_days = self._coerce_positive_float(ttl_value)

        mode_value = mode
        if mode_value is None and cache_cfg is not None:
            mode_value = _get_attr(cache_cfg, "mode", None)
        if mode_value is None:
            mode_value = _get_attr(cfg, "cache_mode", None)
        self._cache_mode = self._normalize_cache_mode(mode_value)

        checkpoint_cfg = _get_attr(cfg, "checkpoint", None)

        checkpoint_path_value = checkpoint_path
        if checkpoint_path_value is None and checkpoint_cfg is not None:
            checkpoint_path_value = _get_attr(checkpoint_cfg, "path", None)
        self._checkpoint_path = (
            Path(checkpoint_path_value).expanduser() if checkpoint_path_value else None
        )

        if checkpoint_enabled is None and checkpoint_cfg is not None:
            checkpoint_enabled = _get_attr(checkpoint_cfg, "enabled", None)
        if checkpoint_enabled is None:
            checkpoint_enabled = bool(checkpoint_path_value)
        self._checkpoint_enabled = bool(checkpoint_enabled) and self._checkpoint_path is not None

        resume_value = resume_from_checkpoint
        if resume_value is None and checkpoint_cfg is not None:
            resume_value = _get_attr(checkpoint_cfg, "resume_from_checkpoint", None)
        if resume_value is None:
            resume_value = self._checkpoint_enabled
        self._resume_from_checkpoint = bool(resume_value) and self._checkpoint_enabled

        self._endpoint_cache_settings: MutableMapping[str, dict[str, Any]] = {}

        global_cfg = _get_attr(cfg, "global_", _get_attr(cfg, "global", None))

        dynamic_headers_value = _get_attr(cfg, "dynamic_from_headers", None)
        if dynamic_headers_value is None and global_cfg is not None:
            dynamic_headers_value = _get_attr(global_cfg, "dynamic_from_headers", None)
        dynamic_flag = self._interpret_bool(dynamic_headers_value)
        self._dynamic_from_headers = bool(dynamic_flag) if dynamic_flag is not None else False

        jitter_value = _get_attr(cfg, "jitter_ms", None)
        if jitter_value is None:
            jitter_value = _get_attr(cfg, "jitter", None)
        if jitter_value is None and global_cfg is not None:
            jitter_value = _get_attr(
                global_cfg,
                "jitter_ms",
                _get_attr(global_cfg, "jitter", None),
            )
        if jitter_value is None:
            jitter_value = 0.0
        self._jitter_min_ms, self._jitter_max_ms = self._parse_jitter(jitter_value)

        cooldown_value = _get_attr(
            cfg, "cooldown_s", _get_attr(cfg, "cooldown_sec", None)
        )
        if cooldown_value is None and global_cfg is not None:
            cooldown_value = _get_attr(
                global_cfg,
                "cooldown_s",
                _get_attr(global_cfg, "cooldown_sec", None),
            )
        try:
            cooldown_float = float(cooldown_value) if cooldown_value is not None else 0.0
        except (TypeError, ValueError):
            cooldown_float = 0.0
        self._cooldown_s = max(cooldown_float, 0.0)

        timeout_default = _get_attr(cfg, "timeout", _get_attr(cfg, "timeout_s", 0.0))
        if timeout_default is None and global_cfg is not None:
            timeout_default = _get_attr(
                global_cfg,
                "timeout",
                _get_attr(global_cfg, "timeout_s", None),
            )
        self._timeout = float(timeout_default) if timeout_default else None

        retry_cfg = _get_attr(cfg, "retry", None)
        self._retry_cfg = self._parse_retry_cfg(retry_cfg)

        baseline_global = self._make_bucket(global_cfg) if self._enabled else None
        self._baseline_global_bucket: TokenBucket | None = baseline_global
        self._global_bucket = (
            TokenBucket(
                rps=baseline_global.configured_rps,
                burst=baseline_global.configured_burst,
            )
            if baseline_global is not None
            else None
        )

        self._endpoint_buckets: MutableMapping[str, TokenBucket] = {}
        self._baseline_endpoint_buckets: MutableMapping[str, TokenBucket] = {}
        endpoints_cfg = _get_attr(cfg, "endpoints", {}) or {}
        if isinstance(endpoints_cfg, Mapping):
            for key, spec in endpoints_cfg.items():
                baseline_bucket = self._make_bucket(spec) if self._enabled else None
                if baseline_bucket is not None:
                    runtime_bucket = TokenBucket(
                        rps=baseline_bucket.configured_rps,
                        burst=baseline_bucket.configured_burst,
                    )
                    self._register_endpoint_bucket(str(key), runtime_bucket)
                    self._register_baseline_endpoint_bucket(str(key), baseline_bucket)
                cache_meta = self._parse_endpoint_cache_spec(spec)
                if cache_meta:
                    self._register_endpoint_cache_cfg(str(key), cache_meta)

        self.wait_counts: Counter[str] = Counter()
        self.cooldown_counts: Counter[str] = Counter()
        self.cooldown_reasons: Counter[str] = Counter()
        self.error_counts: Counter[str] = Counter()
        self.retry_counts: Counter[str] = Counter()
        self._request_counts: Counter[str] = Counter()
        self._request_tokens: defaultdict[str, float] = defaultdict(float)
        self._planned_counts: Counter[str] = Counter()
        self._planned_tokens: defaultdict[str, float] = defaultdict(float)
        self._cache_hits: Counter[str] = Counter()
        self._cache_misses: Counter[str] = Counter()
        self._cache_stores: Counter[str] = Counter()
        self._checkpoint_loads = 0
        self._checkpoint_saves = 0
        self._wait_seconds: defaultdict[str, float] = defaultdict(float)
        self._stats_started = time.monotonic()
        self._stats_started_wall = time.time()
        self._last_checkpoint_snapshot: dict[str, Any] | None = None

    @staticmethod
    def _parse_retry_cfg(cfg: Any) -> RetryConfig:
        if isinstance(cfg, RetryConfig):
            return cfg
        if isinstance(cfg, Mapping):
            return RetryConfig(**cfg)  # type: ignore[arg-type]
        return RetryConfig()

    @staticmethod
    def _interpret_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"", "unset"}:
                return None
            if text in {"true", "yes", "1", "on", "enabled"}:
                return True
            if text in {"false", "no", "0", "off", "disabled"}:
                return False
            return None
        return bool(value)

    @staticmethod
    def _parse_jitter(jitter: Any) -> tuple[float, float]:
        if isinstance(jitter, (tuple, list)) and len(jitter) == 2:
            lo, hi = jitter
        else:
            lo, hi = 0.0, jitter
        try:
            lo_v = max(float(lo), 0.0)
        except (TypeError, ValueError):
            lo_v = 0.0
        try:
            hi_v = max(float(hi), 0.0)
        except (TypeError, ValueError):
            hi_v = 0.0
        if hi_v < lo_v:
            lo_v, hi_v = hi_v, lo_v
        return lo_v, hi_v

    @staticmethod
    def _make_bucket(spec: Any) -> TokenBucket | None:
        def _is_disabled(candidate: Any) -> bool:
            enabled_flag = RestBudgetSession._interpret_bool(
                _get_attr(candidate, "enabled", None)
            )
            disabled_flag = RestBudgetSession._interpret_bool(
                _get_attr(candidate, "disabled", None)
            )
            if disabled_flag is True:
                return True
            if enabled_flag is False:
                return True
            return False

        if _is_disabled(spec):
            return None
        if isinstance(spec, TokenBucket):
            return spec
        if isinstance(spec, TokenBucketConfig):
            rps = spec.rps
            burst = spec.burst
        elif isinstance(spec, Mapping):
            rps = (
                spec.get("rps")
                or spec.get("rate")
                or spec.get("qps")
                or spec.get("requests_per_second")
                or spec.get("queries_per_second")
                or spec.get("per_second")
            )
            burst = (
                spec.get("burst")
                or spec.get("capacity")
                or spec.get("tokens")
                or spec.get("max_tokens")
                or spec.get("burst_tokens")
            )
        else:
            rps = (
                getattr(spec, "rps", None)
                or getattr(spec, "rate", None)
                or getattr(spec, "qps", None)
                or getattr(spec, "requests_per_second", None)
                or getattr(spec, "queries_per_second", None)
                or getattr(spec, "per_second", 0.0)
            )
            burst = (
                getattr(spec, "burst", None)
                or getattr(spec, "capacity", None)
                or getattr(spec, "tokens", 0.0)
                or getattr(spec, "max_tokens", None)
                or getattr(spec, "burst_tokens", None)
            )
        try:
            rps_f = float(rps)
        except (TypeError, ValueError):
            rps_f = 0.0
        try:
            burst_f = float(burst)
        except (TypeError, ValueError):
            burst_f = 0.0
        if rps_f <= 0.0 or burst_f <= 0.0:
            return None
        return TokenBucket(rps=rps_f, burst=burst_f)

    @staticmethod
    def _coerce_positive_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if val <= 0.0:
            return None
        return val

    @staticmethod
    def _coerce_positive_int(value: Any) -> int:
        if value is None:
            return 0
        try:
            val = int(value)
        except (TypeError, ValueError):
            return 0
        return val if val > 0 else 0

    @staticmethod
    def _normalize_cache_mode(mode: Any) -> str:
        if mode is None:
            return "off"
        text = str(mode).strip().lower().replace("-", "_")
        if not text:
            return "off"
        if text in {"rw", "readwrite", "read_write", "write"}:
            return "read_write"
        if text in {"ro", "read", "read_only", "readonly"}:
            return "read"
        if text in {"off", "disable", "disabled", "none"}:
            return "off"
        return text if text in {"off", "read", "read_write"} else "off"

    @staticmethod
    def _sanitize_cache_token(token: str) -> str:
        return "".join(
            ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in token
        )

    def _endpoint_variants(self, key: str) -> set[str]:
        variants = {key.strip()}
        norm = self._normalize_endpoint_key(key)
        if norm:
            variants.add(norm)
            if " " in norm:
                _, path = norm.split(" ", 1)
                variants.add(path)
        return {v for v in variants if v}

    def _register_endpoint_bucket(self, key: str, bucket: TokenBucket) -> None:
        for v in self._endpoint_variants(key):
            self._endpoint_buckets[v] = bucket

    def _register_baseline_endpoint_bucket(self, key: str, bucket: TokenBucket) -> None:
        for v in self._endpoint_variants(key):
            self._baseline_endpoint_buckets[v] = bucket

    def _register_endpoint_cache_cfg(self, key: str, cfg: Mapping[str, Any]) -> None:
        for v in self._endpoint_variants(key):
            current = self._endpoint_cache_settings.setdefault(v, {})
            current.update(cfg)

    def _clone_session(self) -> requests.Session:
        base = self._base_session
        clone = requests.Session()
        clone.headers.update(base.headers)
        if getattr(base, "params", None):
            clone.params = dict(base.params)  # type: ignore[arg-type]
        clone.auth = base.auth
        clone.cookies.update(base.cookies)
        clone.hooks = {k: list(v) for k, v in base.hooks.items()}
        clone.proxies = dict(base.proxies)
        clone.verify = base.verify
        clone.cert = base.cert
        clone.trust_env = base.trust_env
        clone.stream = base.stream
        clone.max_redirects = base.max_redirects
        for prefix, adapter in base.adapters.items():
            clone.mount(prefix, adapter)
        return clone

    def _get_requests_session(self) -> requests.Session:
        if self._executor is None:
            return self._base_session
        sess = getattr(self._thread_local, "session", None)
        if sess is None:
            sess = self._clone_session()
            self._thread_local.session = sess
            with self._session_lock:
                self._child_sessions.add(sess)
        return sess

    @staticmethod
    def _parse_endpoint_cache_spec(spec: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        candidates = []
        cache_candidate = _get_attr(spec, "cache", None)
        if cache_candidate is not None:
            candidates.append(cache_candidate)
        candidates.append(spec)
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, Mapping):
                min_refresh = candidate.get("min_refresh_days")
            else:
                min_refresh = getattr(candidate, "min_refresh_days", None)
            if min_refresh is not None:
                try:
                    val = float(min_refresh)
                except (TypeError, ValueError):
                    continue
                if val > 0.0:
                    meta["min_refresh_days"] = val
        return meta

    def _get_endpoint_cache_meta(self, key: str) -> Mapping[str, Any] | None:
        for variant in self._endpoint_variants(key):
            cfg = self._endpoint_cache_settings.get(variant)
            if cfg:
                return cfg
        return None

    @staticmethod
    def _normalize_endpoint_key(key: str) -> str:
        key = key.strip()
        if not key:
            return ""
        if " " not in key:
            method = "GET"
            path = key
        else:
            method, path = key.split(" ", 1)
        parsed = urllib.parse.urlsplit(path)
        norm_path = parsed.path or "/"
        return f"{method.upper()} {norm_path}"

    def _resolve_endpoint_key(self, method: str, url: str, override: str | None) -> str:
        if override:
            norm = self._normalize_endpoint_key(override)
            for cand in (override, norm, norm.split(" ", 1)[-1]):
                if cand and cand in self._endpoint_buckets:
                    return cand
            return override
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path or "/"
        key = f"{method.upper()} {path}"
        if key in self._endpoint_buckets:
            return key
        if path in self._endpoint_buckets:
            return path
        return key

    _CACHE_PARAM_ALIASES: Mapping[str, str] = {
        "symbol": "symbol",
        "symbols": "symbols",
        "pair": "symbol",
        "trading_pair": "symbol",
        "base": "symbol",
        "quote": "symbol",
        "interval": "interval",
        "timeframe": "interval",
        "tick_interval": "interval",
        "granularity": "interval",
        "start": "start",
        "start_time": "startTime",
        "starttime": "startTime",
        "start_ts": "startTime",
        "starttimestamp": "startTime",
        "from": "startTime",
        "end": "end",
        "end_time": "endTime",
        "endtime": "endTime",
        "end_ts": "endTime",
        "endtimestamp": "endTime",
        "to": "endTime",
        "from_id": "fromId",
        "fromid": "fromId",
        "page": "page",
        "offset": "offset",
        "limit": "limit",
        "window": "window",
        "date": "date",
        "day": "date",
    }

    _CACHE_PARAM_ORDER: tuple[str, ...] = (
        "symbol",
        "symbols",
        "interval",
        "startTime",
        "endTime",
        "start",
        "end",
        "fromId",
        "limit",
        "page",
        "offset",
        "window",
        "date",
    )

    def _make_cache_key(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        endpoint_key: str,
    ) -> str:
        prepared = requests.Request(method.upper(), url, params=params).prepare()
        canonical_url = prepared.url or url
        parsed = urllib.parse.urlsplit(canonical_url)

        base = self._normalize_endpoint_key(endpoint_key) or endpoint_key or method.upper()
        base = base.replace(" ", "_")
        safe_base = self._sanitize_cache_token(base) or method.upper()

        # Collect parameters from the explicit ``params`` mapping and the canonical URL.
        values: dict[str, list[str]] = defaultdict(list)

        def _normalise_key(key: str) -> str:
            canon = self._CACHE_PARAM_ALIASES.get(key)
            if canon is not None:
                return canon
            lower = key.lower()
            return self._CACHE_PARAM_ALIASES.get(lower, key)

        def _append_value(target: dict[str, list[str]], key: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _append_value(target, key, item)
                return
            text = str(value).strip()
            if not text:
                return
            canon_key = _normalise_key(str(key))
            bucket = target.setdefault(canon_key, [])
            if text not in bucket:
                bucket.append(text)

        if isinstance(params, Mapping):
            for key, value in params.items():
                _append_value(values, str(key), value)

        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
            _append_value(values, key, value)

        parts = [safe_base]
        used_keys: set[str] = set()
        for name in self._CACHE_PARAM_ORDER:
            bucket = values.get(name)
            if not bucket:
                continue
            used_keys.add(name)
            cleaned = "-".join(
                filter(
                    None,
                    (
                        self._sanitize_cache_token(val.replace(" ", "_"))
                        for val in bucket
                    ),
                )
            )
            if cleaned:
                parts.append(f"{name}_{cleaned}")

        remaining_keys = {
            key
            for key in values.keys()
            if key not in used_keys and key not in self._CACHE_PARAM_ALIASES.values()
        }

        if len(parts) == 1:
            # No human-friendly tokens detected; fall back to digesting the canonical URL.
            digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
            parts.append(digest)
        elif remaining_keys:
            digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
            parts.append(digest[:12])

        return "__".join(parts)

    def _cache_path(self, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        safe_key = self._sanitize_cache_token(key)
        if not safe_key:
            safe_key = "cache_entry"
        return self._cache_dir / f"{safe_key}.json"

    def _cache_lookup(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        endpoint_key: str,
        *,
        load_payload: bool = True,
    ) -> tuple[str | None, Any | None, bool]:
        if self._cache_mode == "off" or self._cache_dir is None:
            return None, None, False

        cache_key = self._make_cache_key(method, url, params, endpoint_key)
        path = self._cache_path(cache_key)
        if path is None:
            return cache_key, None, False

        meta = self._get_endpoint_cache_meta(endpoint_key)
        ttl_days = None
        if meta is not None:
            ttl_candidate = meta.get("min_refresh_days")
            if ttl_candidate is not None:
                ttl_days = self._coerce_positive_float(ttl_candidate)
        if ttl_days is None:
            ttl_days = self._cache_ttl_days

        try:
            stat = path.stat()
        except FileNotFoundError:
            return cache_key, None, False

        if ttl_days is not None:
            ttl_seconds = ttl_days * 86_400.0
            if ttl_seconds <= 0.0:
                return cache_key, None, False
            age = time.time() - stat.st_mtime
            if age > ttl_seconds:
                return cache_key, None, False

        if not load_payload:
            return cache_key, None, True

        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read cache file %s: %s", path, exc)
            try:
                path.unlink()
            except OSError:
                pass
            return cache_key, None, False
        return cache_key, payload, True

    def _cache_store(self, key: str, payload: Any) -> bool:
        if self._cache_mode != "read_write":
            return False
        path = self._cache_path(key)
        if path is None:
            return False
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            logger.debug("Skipping cache store for %s: payload not JSON-serializable", key)
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = f".{path.stem}."
        try:
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
        except OSError as exc:
            logger.warning("Failed to create cache temp file in %s: %s", path.parent, exc)
            return False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(encoded)
            os.replace(tmp_name, path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write cache file %s: %s", path, exc)
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            return False
        return True

    def is_cached(
        self,
        url: str,
        *,
        method: str = "GET",
        params: Mapping[str, Any] | None = None,
        endpoint: str | None = None,
        budget: str | None = None,
    ) -> bool:
        """Return ``True`` when a fresh cache entry exists for the request."""

        method_upper = method.upper()
        override = endpoint or budget
        key = self._resolve_endpoint_key(method_upper, url, override)
        _, _, hit = self._cache_lookup(
            method_upper, url, params, key, load_payload=False
        )
        return hit
    def _next_jitter(self) -> float:
        if self._jitter_max_ms <= 0.0:
            return 0.0
        with self._rng_lock:
            return self._rng.uniform(self._jitter_min_ms, self._jitter_max_ms) / 1000.0

    def _acquire_tokens(self, key: str, tokens: float = 1.0) -> None:
        while True:
            waits: list[tuple[str, float, TokenBucket]] = []
            now = time.monotonic()
            if self._global_bucket:
                w = self._global_bucket.wait_time(tokens=tokens, now=now)
                if w > 0.0:
                    waits.append(("global", w, self._global_bucket))
            bucket = self._endpoint_buckets.get(key)
            if bucket is not None:
                w = bucket.wait_time(tokens=tokens, now=now)
                if w > 0.0:
                    waits.append((key, w, bucket))
            if not waits:
                if self._global_bucket:
                    self._global_bucket.consume(tokens=tokens, now=now)
                if bucket is not None:
                    bucket.consume(tokens=tokens, now=now)
                return
            wait_for = max(w for _, w, _ in waits)
            if wait_for > 0.0:
                with self._stats_lock:
                    for name, wait_amount, _ in waits:
                        self.wait_counts[name] += 1
                        self._wait_seconds[name] += float(wait_amount)
                self._sleep(wait_for)

    def _record_request(self, key: str, tokens: float) -> None:
        with self._stats_lock:
            self._request_counts[key] += 1
            self._request_tokens[key] += float(tokens)

    def _record_cache_hit(self, key: str) -> None:
        with self._stats_lock:
            self._cache_hits[key] += 1

    def _record_cache_miss(self, key: str) -> None:
        with self._stats_lock:
            self._cache_misses[key] += 1

    def _record_cache_store(self, key: str) -> None:
        with self._stats_lock:
            self._cache_stores[key] += 1

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        value = value.strip()
        try:
            seconds = float(value)
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime

                dt = parsedate_to_datetime(value)
                if dt is None:
                    return None
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_dt.timezone.utc)
                now = _dt.datetime.now(tz=_dt.timezone.utc)
                seconds = (dt - now).total_seconds()
            except Exception:
                return None
        return max(seconds, 0.0)

    def _start_cooldown(
        self,
        key: str,
        seconds: float | None = None,
        *,
        reason: str | None = None,
    ) -> None:
        sec = self._cooldown_s if seconds is None else max(float(seconds), self._cooldown_s)
        if sec <= 0.0:
            return
        now = time.monotonic()
        cooled: list[str] = []
        if self._global_bucket:
            self._global_bucket.start_cooldown(sec, now=now)
            cooled.append("global")
        bucket = self._endpoint_buckets.get(key)
        if bucket is not None:
            bucket.start_cooldown(sec, now=now)
            cooled.append(key)
        if not cooled:
            return
        label = str(reason) if reason else "unspecified"
        with self._stats_lock:
            for name in cooled:
                self.cooldown_counts[name] += 1
            self.cooldown_reasons[label] += 1

    @staticmethod
    def _try_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _parse_weight_pair(cls, value: str | None) -> tuple[float | None, float | None]:
        if not value:
            return None, None
        text = value.strip()
        if not text:
            return None, None
        if "/" in text:
            used_text, limit_text = text.split("/", 1)
            used = cls._try_float(used_text)
            limit = cls._try_float(limit_text)
            return used, limit
        return cls._try_float(text), None

    @classmethod
    def _extract_binance_weights(
        cls, headers: Mapping[str, str]
    ) -> tuple[float | None, float | None]:
        if not headers:
            return None, None
        try:
            items = headers.items()
        except AttributeError:
            return None, None
        lower = {str(k).lower(): v for k, v in items}
        used: float | None = None
        limit: float | None = None
        for name in ("x-mbx-used-weight-1m", "x-mbx-used-weight"):
            candidate = lower.get(name)
            if candidate is None:
                continue
            used_candidate, limit_candidate = cls._parse_weight_pair(candidate)
            if used_candidate is not None:
                used = used_candidate
            if limit_candidate is not None:
                limit = limit_candidate
            if used is not None and limit is not None:
                break
        if limit is None:
            for name in ("x-mbx-weight-limit-1m", "x-mbx-weight-limit"):
                limit_candidate = cls._try_float(lower.get(name))
                if limit_candidate is not None:
                    limit = limit_candidate
                    break
        return used, limit

    def _adjust_bucket_from_usage(
        self,
        bucket: TokenBucket | None,
        baseline: TokenBucket | None,
        used: float,
        limit: float | None,
    ) -> None:
        if bucket is None or used < 0.0:
            return
        baseline_bucket = baseline or bucket
        base_limit = limit
        if base_limit is None or base_limit <= 0.0:
            base_limit = baseline_bucket.configured_burst
        if base_limit is None or base_limit <= 0.0:
            return
        baseline_rps = baseline_bucket.configured_rps
        baseline_burst = baseline_bucket.configured_burst
        if baseline_rps <= 0.0 or baseline_burst <= 0.0:
            return
        ratio = used / base_limit
        if ratio < 0.0:
            return
        current_scale = bucket.rps / baseline_rps if baseline_rps > 0.0 else 1.0
        target_scale = current_scale
        if ratio >= 0.98:
            target_scale = min(current_scale, 0.2)
        elif ratio >= 0.95:
            target_scale = min(current_scale, 0.4)
        elif ratio >= 0.9:
            target_scale = min(current_scale, 0.7)
        elif ratio <= 0.5 and current_scale < 1.0:
            target_scale = min(1.0, current_scale + 0.2)
        elif ratio <= 0.7 and current_scale < 1.0:
            target_scale = min(1.0, current_scale + 0.1)
        else:
            return
        target_scale = max(target_scale, 0.05)
        new_rps = baseline_rps * target_scale
        new_burst = baseline_burst * target_scale
        if (
            abs(new_rps - bucket.rps) < 1e-9
            and abs(new_burst - bucket.burst) < 1e-9
        ):
            return
        bucket.adjust_rate(rps=new_rps, burst=new_burst)

    def _handle_dynamic_headers(self, key: str, headers: Mapping[str, str]) -> None:
        if not self._dynamic_from_headers:
            return
        used, limit = self._extract_binance_weights(headers)
        if used is None:
            return
        self._adjust_bucket_from_usage(
            self._global_bucket,
            self._baseline_global_bucket,
            used,
            limit,
        )
        endpoint_bucket: TokenBucket | None = None
        baseline_bucket: TokenBucket | None = None
        for variant in self._endpoint_variants(key):
            if endpoint_bucket is None:
                endpoint_bucket = self._endpoint_buckets.get(variant)
            if baseline_bucket is None:
                baseline_bucket = self._baseline_endpoint_buckets.get(variant)
            if endpoint_bucket is not None and baseline_bucket is not None:
                break
        self._adjust_bucket_from_usage(endpoint_bucket, baseline_bucket, used, limit)

    @staticmethod
    def _classify_for_retry(exc: Exception) -> str | None:
        if isinstance(exc, requests.exceptions.HTTPError):
            resp = exc.response
            status = resp.status_code if resp is not None else None
            if status == 429 or (status is not None and 500 <= status < 600):
                return "rest"
        elif isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return "rest"
        return None

    @staticmethod
    def _error_label(exc: Exception) -> str:
        if isinstance(exc, requests.exceptions.HTTPError):
            resp = exc.response
            if resp is not None:
                return str(resp.status_code)
        if isinstance(exc, requests.exceptions.Timeout):
            return "timeout"
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "connection"
        return exc.__class__.__name__

    def _store_last_response_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        if metadata is None:
            self._thread_local.last_response_metadata = None
        else:
            self._thread_local.last_response_metadata = dict(metadata)

    def get_last_response_metadata(self) -> dict[str, Any] | None:
        metadata = getattr(self._thread_local, "last_response_metadata", None)
        if isinstance(metadata, Mapping):
            return dict(metadata)
        return None

    @staticmethod
    def _extract_binance_weights(headers: Mapping[str, Any]) -> dict[str, Any]:
        weights: dict[str, Any] = {}
        for raw_key, raw_value in headers.items():
            key = str(raw_key).lower()
            if key.startswith("x-mbx-used-weight") or key.startswith("x-mbx-order-count"):
                value: Any
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    value = str(raw_value)
                weights[key] = value
        return weights

    @staticmethod
    def _extract_body(resp: requests.Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            return resp.text

    @staticmethod
    def _normalize_checkpoint_symbol(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text.upper() if text else None

    @staticmethod
    def _normalize_checkpoint_range(value: Any) -> list[int] | None:
        if value is None:
            return None
        start: Any
        end: Any
        if isinstance(value, Mapping):
            if "start" in value:
                start = value.get("start")
            else:
                start = value.get("start_ms", value.get("from"))
            if "end" in value:
                end = value.get("end")
            else:
                end = value.get("end_ms", value.get("to"))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            start, end = value
        else:
            return None
        try:
            start_int = int(start)
            end_int = int(end)
        except (TypeError, ValueError):
            return None
        return [start_int, end_int]

    @staticmethod
    def _coerce_progress_pct(value: Any) -> float | None:
        if value is None:
            return None
        try:
            pct = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(pct) or math.isinf(pct):
            return None
        return max(0.0, min(pct, 100.0))

    def load_checkpoint(self) -> Any | None:
        """Return checkpoint payload when resume is enabled."""

        if not self._resume_from_checkpoint or not self._checkpoint_path:
            return None
        try:
            with self._checkpoint_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read checkpoint %s: %s", self._checkpoint_path, exc)
            return None

        if isinstance(payload, Mapping):
            data_part = payload.get("data", payload)
            if isinstance(data_part, Mapping):
                base: dict[str, Any] = dict(data_part)
                base["data"] = dict(data_part)
            else:
                base = {"data": data_part}
            last_symbol = self._normalize_checkpoint_symbol(payload.get("last_symbol"))
            last_range = self._normalize_checkpoint_range(payload.get("last_range"))
            progress_pct = self._coerce_progress_pct(payload.get("progress_pct"))
            saved_at = payload.get("saved_at")
            version = payload.get("version")
            meta = {
                "last_symbol": last_symbol,
                "last_range": last_range,
                "progress_pct": progress_pct,
                "saved_at": saved_at,
                "version": version,
            }
            base.setdefault("last_symbol", last_symbol)
            base.setdefault("last_range", last_range)
            base.setdefault("progress_pct", progress_pct)
            base.setdefault("checkpoint_saved_at", saved_at)
            base.setdefault("checkpoint_version", version)
            base["_checkpoint"] = meta
            payload = base

        with self._stats_lock:
            self._checkpoint_loads += 1
        return payload

    def save_checkpoint(
        self,
        data: Any,
        *,
        last_symbol: Any | None = None,
        last_range: Any | None = None,
        progress_pct: Any | None = None,
    ) -> None:
        """Persist *data* atomically when checkpointing is enabled."""

        if not self._checkpoint_enabled or not self._checkpoint_path:
            return

        payload = {
            "version": 2,
            "saved_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "data": data,
            "last_symbol": self._normalize_checkpoint_symbol(last_symbol),
            "last_range": self._normalize_checkpoint_range(last_range),
            "progress_pct": self._coerce_progress_pct(progress_pct),
        }

        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            logger.warning("Checkpoint payload is not JSON serialisable: %r", data)
            return
        path = self._checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = f".{path.stem}.ckpt."
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=prefix, suffix=".tmp"
            )
        except OSError as exc:
            logger.warning("Failed to create checkpoint temp file in %s: %s", path.parent, exc)
            return
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(encoded)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_name, path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write checkpoint %s: %s", path, exc)
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        else:
            with self._stats_lock:
                self._checkpoint_saves += 1
                snapshot: dict[str, Any] = {
                    "saved_at": payload.get("saved_at"),
                    "progress_pct": payload.get("progress_pct"),
                    "last_symbol": payload.get("last_symbol"),
                    "last_range": payload.get("last_range"),
                }
                data_summary: dict[str, Any] = {}
                if isinstance(data, Mapping):
                    for key, caster in (
                        ("tasks_total", int),
                        ("tasks_completed", int),
                        ("pending_tasks", int),
                        ("pending_tasks_remaining", int),
                        ("pending_bars", int),
                        ("pending_bars_remaining", int),
                        ("processed", int),
                        ("total", int),
                    ):
                        if key not in data:
                            continue
                        value = data.get(key)
                        try:
                            data_summary[key] = caster(value)
                        except (TypeError, ValueError):
                            data_summary[key] = value
                    if "completed" in data:
                        data_summary["completed"] = bool(data.get("completed"))
                if data_summary:
                    snapshot["data_summary"] = data_summary
                self._last_checkpoint_snapshot = snapshot

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        if self._executor is None:
            return self._submit_sync(fn, *args, **kwargs)
        sem = self._task_semaphore
        if sem is not None:
            sem.acquire()
        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except BaseException:
            if sem is not None:
                sem.release()
            raise
        if sem is not None:
            future.add_done_callback(lambda _: sem.release())
        return future

    @staticmethod
    def _submit_sync(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        future: Future[T] = Future()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - passthrough
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future

    def map(
        self,
        fn: Callable[..., R],
        *iterables: Iterable[Any],
        prefetch: int | None = None,
    ) -> Iterator[R]:
        if not iterables:
            raise ValueError("map() requires at least one iterable")
        if self._executor is None:
            for args in zip(*iterables):
                yield fn(*args)
            return

        iterator = zip(*iterables)
        max_pending = prefetch if prefetch is not None and prefetch > 0 else 0
        if max_pending <= 0:
            default_prefetch = self.batch_size or self._queue_capacity or self._max_workers or 1
            max_pending = max(int(default_prefetch), 1)

        pending: deque[Future[R]] = deque()
        while True:
            while len(pending) < max_pending:
                try:
                    args = next(iterator)
                except StopIteration:
                    break
                pending.append(self.submit(fn, *args))
            if not pending:
                break
            yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()

    def shutdown(self, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
            self._task_semaphore = None
            self._queue_capacity = 0

    def close(self) -> None:
        self.shutdown(wait=True)
        sessions: list[requests.Session] = []
        with self._session_lock:
            sessions.extend(list(self._child_sessions))
            self._child_sessions.clear()
        for sess in sessions:
            try:
                sess.close()
            except Exception:  # pragma: no cover - best effort cleanup
                continue
        try:
            del self._thread_local.session
        except AttributeError:
            pass
        if self._owns_session:
            try:
                self._base_session.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    def __enter__(self) -> "RestBudgetSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        endpoint: str | None = None,
        budget: str | None = None,
        tokens: float = 1.0,
    ) -> Any:
        """Perform GET request obeying configured budgets."""

        override = endpoint or budget
        key = self._resolve_endpoint_key("GET", url, override)
        stats_key = self._normalize_endpoint_key(key) or key

        self._store_last_response_metadata(None)

        cache_key: str | None = None
        cache_hit = False
        if self._cache_mode != "off" and self._cache_dir is not None:
            cache_key, cached_payload, cache_hit = self._cache_lookup(
                "GET", url, params, key
            )
            if cache_hit:
                self._record_cache_hit(stats_key)
                self._store_last_response_metadata(
                    {
                        "method": "GET",
                        "url": url,
                        "params": dict(params) if isinstance(params, Mapping) else None,
                        "headers": {},
                        "binance_weights": {},
                        "cache_hit": True,
                        "endpoint": key,
                        "budget": override,
                        "tokens": float(tokens),
                    }
                )
                return cached_payload
            if cache_key is not None:
                self._record_cache_miss(stats_key)

        attempt = 0

        def _do_request() -> Any:
            nonlocal attempt
            attempt += 1
            if attempt > 1:
                with self._stats_lock:
                    self.retry_counts[stats_key] += 1
            self._record_request(stats_key, tokens)
            self._acquire_tokens(key, tokens=tokens)
            jitter = self._next_jitter()
            if jitter > 0.0:
                self._sleep(jitter)

            monitoring.record_http_request()
            http_session = self._get_requests_session()
            try:
                resp = http_session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout if timeout is not None else self._timeout,
                )
            except requests.exceptions.RequestException as exc:
                label = self._error_label(exc)
                with self._stats_lock:
                    self.error_counts[label] += 1
                monitoring.record_http_error(
                    label, timed_out=isinstance(exc, requests.exceptions.Timeout)
                )
                raise

            status = resp.status_code
            retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
            cooldown_reason: str | None = None
            if status == 429:
                with self._stats_lock:
                    self.error_counts[str(status)] += 1
                cooldown_reason = "http_429"
                monitoring.record_http_error(status)
            elif 500 <= status < 600:
                with self._stats_lock:
                    self.error_counts[str(status)] += 1
                cooldown_reason = f"http_{status}"
                monitoring.record_http_error(status)
            elif retry_after is not None:
                cooldown_reason = "retry_after"

            if cooldown_reason:
                self._start_cooldown(key, seconds=retry_after, reason=cooldown_reason)

            self._handle_dynamic_headers(key, resp.headers)

            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError:
                raise

            monitoring.record_http_success(status)
            payload = self._extract_body(resp)
            response_headers = dict(resp.headers)
            metadata = {
                "method": "GET",
                "url": resp.url or url,
                "params": dict(params) if isinstance(params, Mapping) else None,
                "headers": response_headers,
                "binance_weights": self._extract_binance_weights(response_headers),
                "cache_hit": False,
                "endpoint": key,
                "budget": override,
                "tokens": float(tokens),
                "status": status,
                "retry_after": retry_after,
            }
            self._store_last_response_metadata(metadata)
            if cache_key is not None and self._cache_store(cache_key, payload):
                self._record_cache_store(stats_key)
            return payload

        wrapped = retry_sync(self._retry_cfg, self._classify_for_retry)(_do_request)
        return wrapped()

    def plan_request(
        self,
        endpoint: str,
        *,
        count: int = 1,
        tokens: float = 1.0,
    ) -> None:
        """Record a planned request without performing HTTP calls."""

        key = self._normalize_endpoint_key(str(endpoint))
        if not key:
            return
        try:
            cnt = int(count)
        except (TypeError, ValueError):
            return
        if cnt <= 0:
            return
        try:
            token_val = float(tokens)
        except (TypeError, ValueError):
            token_val = 0.0
        total_tokens = token_val * cnt
        with self._stats_lock:
            self._planned_counts[key] += cnt
            self._planned_tokens[key] += total_tokens

    def stats(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of collected statistics."""

        with self._stats_lock:
            wait_seconds = {k: float(v) for k, v in self._wait_seconds.items()}
            total_wait = float(sum(wait_seconds.values()))
            total_requests = int(sum(self._request_counts.values()))
            total_retries = int(sum(self.retry_counts.values()))
            cache_hits_total = int(sum(self._cache_hits.values()))
            cache_misses_total = int(sum(self._cache_misses.values()))
            cache_stores_total = int(sum(self._cache_stores.values()))
            elapsed = max(time.monotonic() - self._stats_started, 1e-9)
            avg_qps = float(total_requests / elapsed) if total_requests else 0.0
            return {
                "requests": dict(self._request_counts),
                "request_tokens": {k: float(v) for k, v in self._request_tokens.items()},
                "planned_requests": dict(self._planned_counts),
                "planned_tokens": {k: float(v) for k, v in self._planned_tokens.items()},
                "cache_hits": dict(self._cache_hits),
                "cache_misses": dict(self._cache_misses),
                "cache_stores": dict(self._cache_stores),
                "cache_totals": {
                    "hits": cache_hits_total,
                    "misses": cache_misses_total,
                    "stores": cache_stores_total,
                },
                "waits": dict(self.wait_counts),
                "wait_seconds": wait_seconds,
                "cooldowns": dict(self.cooldown_counts),
                "cooldown_reasons": dict(self.cooldown_reasons),
                "errors": dict(self.error_counts),
                "retry_counts": dict(self.retry_counts),
                "requests_total": total_requests,
                "total_retries": total_retries,
                "total_wait_seconds": total_wait,
                "avg_qps": avg_qps,
                "checkpoint": {
                    "loads": self._checkpoint_loads,
                    "saves": self._checkpoint_saves,
                },
            }

    @staticmethod
    def _format_ts(ts: float) -> str | None:
        if not math.isfinite(ts) or ts <= 0.0:
            return None
        try:
            dt = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
        return dt.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _bucket_snapshot(
        bucket: TokenBucket,
        *,
        remaining: float,
        now_wall: float,
    ) -> dict[str, Any]:
        with bucket._lock:
            tokens = float(bucket.tokens)
            burst = float(bucket.burst)
            rps = float(bucket.rps)
            configured_rps = float(bucket.configured_rps)
            configured_burst = float(bucket.configured_burst)
        until_wall = now_wall + remaining if math.isfinite(remaining) else float("inf")
        snapshot: dict[str, Any] = {
            "active": remaining > 0.0 and math.isfinite(remaining),
            "remaining_s": float(remaining) if math.isfinite(remaining) else float("inf"),
            "until_ts": until_wall,
            "until_iso": RestBudgetSession._format_ts(until_wall),
            "tokens": tokens,
            "burst": burst,
            "rps": rps,
            "configured_burst": configured_burst,
            "configured_rps": configured_rps,
        }
        return snapshot

    def _cooldown_snapshot(self, now_monotonic: float, now_wall: float) -> dict[str, Any]:
        endpoints: dict[str, Any] = {}
        seen_ids: set[int] = set()
        for key, bucket in list(self._endpoint_buckets.items()):
            ident = id(bucket)
            if ident in seen_ids:
                continue
            seen_ids.add(ident)
            remaining = bucket.wait_time(tokens=0.0, now=now_monotonic)
            if remaining <= 0.0:
                continue
            canonical = self._normalize_endpoint_key(key) or key
            endpoints[canonical] = self._bucket_snapshot(
                bucket,
                remaining=remaining,
                now_wall=now_wall,
            )

        global_snapshot: dict[str, Any] | None = None
        if self._global_bucket is not None:
            remaining = self._global_bucket.wait_time(tokens=0.0, now=now_monotonic)
            if remaining > 0.0:
                global_snapshot = self._bucket_snapshot(
                    self._global_bucket,
                    remaining=remaining,
                    now_wall=now_wall,
                )

        count = len(endpoints) + (1 if global_snapshot else 0)
        payload: dict[str, Any] = {
            "count": int(count),
            "ts": now_wall,
            "ts_iso": self._format_ts(now_wall),
            "endpoints": endpoints,
        }
        if global_snapshot:
            payload["global"] = global_snapshot
        else:
            payload["global"] = {"active": False}
        return payload

    def write_stats(self, path: str | os.PathLike[str]) -> None:
        """Persist the current :meth:`stats` snapshot to ``path`` atomically."""

        stats_payload = self.stats()
        now_wall = time.time()
        now_monotonic = time.monotonic()
        cooldowns = self._cooldown_snapshot(now_monotonic, now_wall)

        with self._stats_lock:
            checkpoint_meta = (
                dict(self._last_checkpoint_snapshot)
                if isinstance(self._last_checkpoint_snapshot, Mapping)
                else None
            )

        session_meta: dict[str, Any] = {
            "enabled": bool(self._enabled),
            "max_workers": int(self._max_workers),
            "batch_size": int(self.batch_size),
            "queue_capacity": int(self._queue_capacity),
            "cooldown_s": float(self._cooldown_s),
            "jitter_ms": [float(self._jitter_min_ms), float(self._jitter_max_ms)],
            "cache_mode": self._cache_mode,
        }
        if self._cache_dir is not None:
            session_meta["cache_dir"] = str(self._cache_dir)
        if self._checkpoint_path is not None:
            session_meta["checkpoint_path"] = str(self._checkpoint_path)

        payload = dict(stats_payload)
        payload["ts"] = now_wall
        payload["ts_iso"] = self._format_ts(now_wall)
        payload["runtime_seconds"] = float(max(now_wall - self._stats_started_wall, 0.0))
        payload["session"] = session_meta
        payload["cooldowns_active"] = cooldowns
        if checkpoint_meta:
            payload["checkpoint_meta"] = checkpoint_meta

        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        except (TypeError, ValueError):
            logger.warning("Stats payload is not JSON serialisable")
            return

        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        prefix = f".{target.stem}.stats."
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target.parent), prefix=prefix, suffix=".tmp"
            )
        except OSError as exc:
            logger.warning("Failed to create stats temp file in %s: %s", target.parent, exc)
            return
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(encoded)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_name, target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write stats file %s: %s", target, exc)
            try:
                os.remove(tmp_name)
            except OSError:
                pass



DAY_MS = 86_400_000


def iter_time_chunks(
    start_ms: int, end_ms: int, *, chunk_days: int = 30
) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` windows covering ``[start_ms, end_ms)``.

    Each chunk spans at most ``chunk_days`` days (default 30).  The ``end``
    value is never less than ``start`` and the final chunk always ends at
    ``end_ms``.  Empty ranges yield no chunks.
    """

    start = int(start_ms)
    end = int(end_ms)
    if end <= start:
        return
    days = max(int(chunk_days), 1)
    span_ms = days * DAY_MS
    current = start
    while current < end:
        stop = min(current + span_ms, end)
        yield current, stop
        if stop >= end:
            break
        current = stop


def split_time_range(
    start_ms: int, end_ms: int, *, chunk_days: int = 30
) -> list[tuple[int, int]]:
    """Return a list of ``(start, end)`` chunks covering the range."""

    return list(iter_time_chunks(start_ms, end_ms, chunk_days=chunk_days))


__all__ = ["TokenBucket", "RestBudgetSession", "iter_time_chunks", "split_time_range"]
