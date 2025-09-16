"""Helpers for budgeting REST API requests."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import random
import tempfile
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping

from pathlib import Path

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
        cache_dir: str | os.PathLike[str] | None = None,
        ttl_days: float | int | None = None,
        mode: str | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        checkpoint_enabled: bool | None = None,
        resume_from_checkpoint: bool | None = None,
    ) -> None:
        self._cfg = cfg
        self._session = session or requests.Session()
        self._rng = rng or random.Random()
        self._sleep = sleep

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
                cache_meta = self._parse_endpoint_cache_spec(spec)
                if cache_meta:
                    self._register_endpoint_cache_cfg(str(key), cache_meta)

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

    def _register_endpoint_cache_cfg(self, key: str, cfg: Mapping[str, Any]) -> None:
        for v in self._endpoint_variants(key):
            current = self._endpoint_cache_settings.setdefault(v, {})
            current.update(cfg)

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

    def _make_cache_key(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        endpoint_key: str,
    ) -> str:
        prepared = requests.Request(method.upper(), url, params=params).prepare()
        canonical_url = prepared.url or url
        digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
        base = self._normalize_endpoint_key(endpoint_key) or endpoint_key or method.upper()
        safe_base = self._sanitize_cache_token(base.replace(" ", "_"))
        if not safe_base:
            safe_base = method.upper()
        return f"{safe_base}_{digest}"

    def _cache_path(self, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        safe_key = self._sanitize_cache_token(key)
        if not safe_key:
            safe_key = "cache_entry"
        return self._cache_dir / f"{safe_key}.json"

    def _cache_lookup(self, key: str, ttl_days: float | None = None) -> Any | None:
        path = self._cache_path(key)
        if path is None:
            return None
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        ttl = self._coerce_positive_float(ttl_days) if ttl_days is not None else self._cache_ttl_days
        if ttl is not None:
            ttl_seconds = ttl * 86_400.0
            if ttl_seconds <= 0.0:
                return None
            age = time.time() - stat.st_mtime
            if age > ttl_seconds:
                return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read cache file %s: %s", path, exc)
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def _cache_store(self, key: str, payload: Any) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            logger.debug("Skipping cache store for %s: payload not JSON-serializable", key)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = f".{path.stem}."
        try:
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
        except OSError as exc:
            logger.warning("Failed to create cache temp file in %s: %s", path.parent, exc)
            return
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

    def load_checkpoint(self) -> Any | None:
        """Return checkpoint payload when resume is enabled."""

        if not self._resume_from_checkpoint or not self._checkpoint_path:
            return None
        try:
            with self._checkpoint_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read checkpoint %s: %s", self._checkpoint_path, exc)
            return None

    def save_checkpoint(self, data: Any) -> None:
        """Persist *data* atomically when checkpointing is enabled."""

        if not self._checkpoint_enabled or not self._checkpoint_path:
            return
        try:
            encoded = json.dumps(data, ensure_ascii=False)
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

        cache_key: str | None = None
        if self._cache_mode != "off" and self._cache_dir is not None:
            cache_key = self._make_cache_key("GET", url, params, key)
            cache_meta = self._get_endpoint_cache_meta(key)
            min_refresh_days = None
            if cache_meta is not None:
                min_refresh_days = cache_meta.get("min_refresh_days")
            if min_refresh_days is not None:
                cached_payload = self._cache_lookup(cache_key, ttl_days=min_refresh_days)
                if cached_payload is not None:
                    return cached_payload
            cached_payload = self._cache_lookup(cache_key)
            if cached_payload is not None:
                return cached_payload

        should_store_cache = (
            cache_key is not None and self._cache_mode == "read_write"
        )

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
            payload = self._extract_body(resp)
            if should_store_cache and cache_key is not None:
                self._cache_store(cache_key, payload)
            return payload

        wrapped = retry_sync(self._retry_cfg, self._classify_for_retry)(_do_request)
        return wrapped()


__all__ = ["TokenBucket", "RestBudgetSession"]
