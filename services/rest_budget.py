"""Helpers for budgeting REST API requests."""

from __future__ import annotations

import datetime as _dt
import logging
import random
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping

import requests

from core_config import RetryConfig, TokenBucketConfig

from . import monitoring
from .retry import retry_sync


logger = logging.getLogger(__name__)


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

    def __post_init__(self) -> None:
        self.rps = float(self.rps)
        self.burst = float(self.burst)
        self.enabled = self.rps > 0.0 and self.burst > 0.0
        if self.tokens is None:
            self.tokens = float(self.burst)
        else:
            self.tokens = float(self.tokens)
        if not self.enabled:
            self.tokens = float(self.burst)

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
        self.cooldown_until = max(self.cooldown_until, now + seconds)


class RestBudgetSession:
    """Requests session with token-bucket budgeting and retry logic."""

    def __init__(
        self,
        cfg: Any,
        *,
        session: requests.Session | None = None,
        rng: random.Random | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cfg = cfg
        self._session = session or requests.Session()
        self._rng = rng or random.Random()
        self._sleep = sleep

        jitter_ms = _get_attr(cfg, "jitter_ms", 0.0)
        self._jitter_min_ms, self._jitter_max_ms = self._parse_jitter(jitter_ms)
        self._cooldown_s = float(
            _get_attr(cfg, "cooldown_s", _get_attr(cfg, "cooldown_sec", 0.0))
        )
        timeout_default = _get_attr(cfg, "timeout", _get_attr(cfg, "timeout_s", 0.0))
        self._timeout = float(timeout_default) if timeout_default else None

        retry_cfg = _get_attr(cfg, "retry", None)
        self._retry_cfg = self._parse_retry_cfg(retry_cfg)

        global_cfg = _get_attr(cfg, "global_", _get_attr(cfg, "global", None))
        self._global_bucket = self._make_bucket(global_cfg)

        self._endpoint_buckets: MutableMapping[str, TokenBucket] = {}
        endpoints_cfg = _get_attr(cfg, "endpoints", {}) or {}
        if isinstance(endpoints_cfg, Mapping):
            for key, spec in endpoints_cfg.items():
                bucket = self._make_bucket(spec)
                if bucket:
                    self._register_endpoint_bucket(str(key), bucket)

        self.wait_counts: Counter[str] = Counter()
        self.cooldown_counts: Counter[str] = Counter()
        self.error_counts: Counter[str] = Counter()

    @staticmethod
    def _parse_retry_cfg(cfg: Any) -> RetryConfig:
        if isinstance(cfg, RetryConfig):
            return cfg
        if isinstance(cfg, Mapping):
            return RetryConfig(**cfg)  # type: ignore[arg-type]
        return RetryConfig()

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
        if isinstance(spec, TokenBucket):
            return spec
        if isinstance(spec, TokenBucketConfig):
            rps = spec.rps
            burst = spec.burst
        elif isinstance(spec, Mapping):
            rps = spec.get("rps") or spec.get("rate") or 0.0
            burst = spec.get("burst") or spec.get("capacity") or spec.get("tokens") or 0.0
        else:
            rps = getattr(spec, "rps", None) or getattr(spec, "rate", 0.0)
            burst = (
                getattr(spec, "burst", None)
                or getattr(spec, "capacity", None)
                or getattr(spec, "tokens", 0.0)
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

    def _register_endpoint_bucket(self, key: str, bucket: TokenBucket) -> None:
        variants = {key.strip()}
        norm = self._normalize_endpoint_key(key)
        if norm:
            variants.add(norm)
            if " " in norm:
                _, path = norm.split(" ", 1)
                variants.add(path)
        variants = {v for v in variants if v}
        for v in variants:
            self._endpoint_buckets[v] = bucket

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

    def _next_jitter(self) -> float:
        if self._jitter_max_ms <= 0.0:
            return 0.0
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
            for name, _, b in waits:
                self.wait_counts[name] += 1
                if self._cooldown_s > 0.0:
                    b.start_cooldown(self._cooldown_s, now=now)
                    self.cooldown_counts[name] += 1
            self._sleep(wait_for)

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

    def _start_cooldown(self, key: str, seconds: float | None = None) -> None:
        sec = self._cooldown_s if seconds is None else max(float(seconds), self._cooldown_s)
        if sec <= 0.0:
            return
        now = time.monotonic()
        if self._global_bucket:
            self._global_bucket.start_cooldown(sec, now=now)
            self.cooldown_counts["global"] += 1
        bucket = self._endpoint_buckets.get(key)
        if bucket is not None:
            bucket.start_cooldown(sec, now=now)
            self.cooldown_counts[key] += 1

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

    @staticmethod
    def _extract_body(resp: requests.Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        endpoint: str | None = None,
        tokens: float = 1.0,
    ) -> Any:
        """Perform GET request obeying configured budgets."""

        key = self._resolve_endpoint_key("GET", url, endpoint)

        def _do_request() -> Any:
            self._acquire_tokens(key, tokens=tokens)
            jitter = self._next_jitter()
            if jitter > 0.0:
                self._sleep(jitter)

            monitoring.record_http_request()
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout if timeout is not None else self._timeout,
                )
            except requests.exceptions.RequestException as exc:
                label = self._error_label(exc)
                self.error_counts[label] += 1
                monitoring.record_http_error(label)
                raise

            status = resp.status_code
            if status == 429:
                self.error_counts[str(status)] += 1
                retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
                self._start_cooldown(key, seconds=retry_after)
                monitoring.record_http_error(status)
            elif 500 <= status < 600:
                self.error_counts[str(status)] += 1
                monitoring.record_http_error(status)

            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError:
                raise

            monitoring.record_http_success(status)
            return self._extract_body(resp)

        wrapped = retry_sync(self._retry_cfg, self._classify_for_retry)(_do_request)
        return wrapped()


__all__ = ["TokenBucket", "RestBudgetSession"]
