from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from typing import Any, Awaitable, Callable, TypeVar

from core_config import RetryConfig
from . import ops_kill_switch

logger = logging.getLogger(__name__)

T = TypeVar("T")


def compute_backoff(
    cfg: RetryConfig,
    attempt: int,
    *,
    rng: random.Random | None = None,
) -> float:
    """Compute full-jitter exponential backoff for *attempt*.

    Parameters
    ----------
    cfg:
        Retry configuration.
    attempt:
        1-based attempt number.
    rng:
        Optional random number generator.
    """
    if attempt <= 0:
        return 0.0
    rng = rng or random
    base = max(float(cfg.backoff_base_s), 0.0)
    cap = max(float(cfg.max_backoff_s), 0.0)
    exp = min(base * (2 ** (attempt - 1)), cap)
    return rng.random() * exp


def retry_sync(
    cfg: RetryConfig,
    classify: Callable[[Exception], str | None],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator for synchronous functions."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        consecutive_failures = 0

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal consecutive_failures
            attempts = cfg.max_attempts if cfg.max_attempts > 0 else 1
            last_exc: Exception | None = None
            had_failure = False
            for attempt in range(1, attempts + 1):
                try:
                    logger.info("attempt %d/%d", attempt, attempts)
                    result = func(*args, **kwargs)
                    if had_failure or consecutive_failures:
                        try:
                            ops_kill_switch.manual_reset()
                        except Exception:
                            pass
                        consecutive_failures = 0
                    return result
                except Exception as e:  # pragma: no cover - log and retry
                    last_exc = e
                    kind = None
                    try:
                        kind = classify(e)
                    except Exception:
                        kind = None
                    if kind:
                        try:
                            ops_kill_switch.record_error(kind)
                        except Exception:
                            pass
                    consecutive_failures += 1
                    had_failure = True
                    logger.warning("attempt %d/%d failed: %s", attempt, attempts, e)
                    if attempt >= attempts:
                        break
                    time.sleep(compute_backoff(cfg, attempt))
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def retry_async(
    cfg: RetryConfig,
    classify: Callable[[Exception], str | None],
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Retry decorator for asynchronous functions."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        consecutive_failures = 0

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal consecutive_failures
            attempts = cfg.max_attempts if cfg.max_attempts > 0 else 1
            last_exc: Exception | None = None
            had_failure = False
            for attempt in range(1, attempts + 1):
                try:
                    logger.info("attempt %d/%d", attempt, attempts)
                    result = await func(*args, **kwargs)
                    if had_failure or consecutive_failures:
                        try:
                            ops_kill_switch.manual_reset()
                        except Exception:
                            pass
                        consecutive_failures = 0
                    return result
                except Exception as e:  # pragma: no cover - log and retry
                    last_exc = e
                    kind = None
                    try:
                        kind = classify(e)
                    except Exception:
                        kind = None
                    if kind:
                        try:
                            ops_kill_switch.record_error(kind)
                        except Exception:
                            pass
                    consecutive_failures += 1
                    had_failure = True
                    logger.warning("attempt %d/%d failed: %s", attempt, attempts, e)
                    if attempt >= attempts:
                        break
                    await asyncio.sleep(compute_backoff(cfg, attempt))
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
