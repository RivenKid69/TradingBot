# -*- coding: utf-8 -*-
"""
impl_quantizer.py
Обёртка над quantizer. Строит Quantizer из JSON-фильтров Binance и умеет подключаться к ExecutionSimulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import hashlib
import logging
import os
import subprocess

from services import monitoring

try:
    from quantizer import Quantizer
except Exception as e:  # pragma: no cover
    Quantizer = None  # type: ignore


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
    strict: bool = True
    enforce_percent_price_by_side: bool = True  # передаётся в симулятор как enforce_ppbs
    filters_path: str = ""
    auto_refresh_days: int = 30
    refresh_on_start: bool = False

    def resolved_filters_path(self) -> str:
        return self.filters_path or self.path


class QuantizerImpl:
    def __init__(self, cfg: QuantizerConfig) -> None:
        self.cfg = cfg
        self._quantizer = None
        filters_path = cfg.resolved_filters_path()
        if not cfg.filters_path:
            cfg.filters_path = filters_path
        if Quantizer is None or not filters_path:
            return

        max_age_days = max(_as_int(cfg.auto_refresh_days, 30), 0)
        filters, meta = Quantizer.load_filters(
            filters_path,
            max_age_days=max_age_days,
            fatal=False,
        )
        meta_dict: Dict[str, Any] = dict(meta or {}) if isinstance(meta, dict) else {}
        age_days = _filters_age_days(meta_dict, filters_path)
        stale = _is_stale(age_days, max_age_days)
        missing = not filters

        if cfg.refresh_on_start and (missing or stale):
            logger.info(
                "Refreshing Binance filters: path=%s missing=%s stale=%s auto_refresh_days=%s",
                filters_path,
                missing,
                stale,
                max_age_days,
            )
            if self._refresh_filters(filters_path):
                filters, meta = Quantizer.load_filters(
                    filters_path,
                    max_age_days=max_age_days,
                    fatal=False,
                )
                meta_dict = dict(meta or {}) if isinstance(meta, dict) else {}
                age_days = _filters_age_days(meta_dict, filters_path)
                stale = _is_stale(age_days, max_age_days)
                missing = not filters
            else:
                filters = {}
                missing = True

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

        try:
            monitoring.filters_age_days.set(
                float(age_days) if age_days is not None else float("nan")
            )
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

        self._quantizer = Quantizer(filters, strict=bool(cfg.strict))

    @property
    def quantizer(self):
        return self._quantizer

    def attach_to(self, sim, *, strict: Optional[bool] = None, enforce_percent_price_by_side: Optional[bool] = None) -> None:
        """Подключает квантайзер к симулятору."""
        if strict is not None:
            self.cfg.strict = bool(strict)
        if enforce_percent_price_by_side is not None:
            self.cfg.enforce_percent_price_by_side = bool(enforce_percent_price_by_side)
        if self._quantizer is not None:
            setattr(sim, "quantizer", self._quantizer)
        setattr(sim, "enforce_ppbs", bool(self.cfg.enforce_percent_price_by_side))
        setattr(sim, "strict_filters", bool(self.cfg.strict))

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

        return QuantizerImpl(QuantizerConfig(
            path=path,
            strict=bool(d.get("strict", True)),
            enforce_percent_price_by_side=bool(d.get("enforce_percent_price_by_side", True)),
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
