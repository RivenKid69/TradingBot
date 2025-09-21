from __future__ import annotations

import csv
import os
import time
from datetime import datetime, date
from typing import Dict, Sequence, Any

from .utils_app import atomic_write_with_retry


class SignalCSVWriter:
    """Write signal rows to a CSV file with daily rotation.

    The writer maintains a file at ``path`` and automatically rotates it when
    UTC day changes.  Previous day files are renamed to ``<name>-YYYY-MM-DD.csv``.
    """

    DEFAULT_HEADER = [
        "ts_ms",
        "symbol",
        "side",
        "volume_frac",
        "score",
        "features_hash",
    ]

    def __init__(
        self,
        path: str,
        header: Sequence[str] | None = None,
        *,
        fsync_mode: str = "batch",
        rotate_daily: bool = True,
        flush_interval_s: float | None = 5.0,
    ) -> None:
        self.path = str(path)
        self.header = list(header) if header is not None else list(self.DEFAULT_HEADER)
        self._file: Any | None = None
        self._writer: csv.DictWriter | None = None
        self._day: date | None = None
        self._fsync_mode = self._normalize_fsync_mode(fsync_mode)
        self._rotate_daily = bool(rotate_daily)
        self._flush_interval_s = self._normalize_flush_interval(flush_interval_s)
        self._last_flush_ts = time.monotonic()
        self._written = 0
        self._retries = 0
        self._errors = 0
        self._dropped = 0
        self._ensure_dir()
        self._rotate_existing()
        self._open_file(initial=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_fsync_mode(mode: str) -> str:
        value = (mode or "off").lower()
        if value not in {"always", "batch", "off"}:
            return "off"
        return value

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_flush_interval(interval: float | None) -> float:
        if interval is None:
            return 0.0
        try:
            value = float(interval)
        except (TypeError, ValueError):
            return 0.0
        if value < 0:
            return 0.0
        return value

    # ------------------------------------------------------------------
    def _ensure_dir(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # ------------------------------------------------------------------
    def _rotate_name(self, d: date) -> str:
        base, ext = os.path.splitext(self.path)
        return f"{base}-{d.isoformat()}{ext}"

    # ------------------------------------------------------------------
    def _rotate_existing(self) -> None:
        if not os.path.exists(self.path):
            return
        mtime = os.path.getmtime(self.path)
        mday = datetime.utcfromtimestamp(mtime).date()
        today = datetime.utcnow().date()
        if mday != today:
            try:
                os.replace(self.path, self._rotate_name(mday))
            except Exception:
                pass
        else:
            self._day = mday

    # ------------------------------------------------------------------
    def _open_file(self, *, initial: bool = False) -> None:
        need_header = True
        if os.path.exists(self.path):
            if os.path.getsize(self.path) > 0:
                need_header = False
            if initial and self._day is None:
                mtime = os.path.getmtime(self.path)
                self._day = datetime.utcfromtimestamp(mtime).date()
        if self._day is None:
            self._day = datetime.utcnow().date()
        self._file = open(self.path, "a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.header)
        if need_header:
            self._writer.writeheader()
            self._file.flush()
            self._last_flush_ts = time.monotonic()
        else:
            self._last_flush_ts = time.monotonic()

    # ------------------------------------------------------------------
    def _maybe_rotate(self, ts_ms: int) -> None:
        if not self._rotate_daily:
            if self._day is None:
                self._day = datetime.utcfromtimestamp(int(ts_ms) / 1000).date()
            return
        day = datetime.utcfromtimestamp(int(ts_ms) / 1000).date()
        if self._day is None:
            self._day = day
        if day == self._day:
            return
        self.flush_fsync()
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
        try:
            os.replace(self.path, self._rotate_name(self._day))
        except Exception:
            pass
        self._day = day
        self._open_file()

    # ------------------------------------------------------------------
    def _write_row(self, row: Dict[str, Any], ts_ms: int) -> None:
        if self._writer is None:
            raise RuntimeError("writer is closed")
        self._maybe_rotate(ts_ms)
        self._writer.writerow(row)
        self._maybe_flush()

    # ------------------------------------------------------------------
    def _maybe_flush(self) -> None:
        if not self._file:
            return
        mode = self._fsync_mode
        if mode == "always":
            self.flush_fsync()
            return
        interval = self._flush_interval_s
        if interval <= 0:
            if mode != "off":
                self.flush_fsync()
            elif self._file:
                try:
                    self._file.flush()
                    self._last_flush_ts = time.monotonic()
                except Exception as exc:
                    self._errors += 1
                    raise exc
            return
        now = time.monotonic()
        if now - self._last_flush_ts < interval:
            return
        if mode == "off":
            try:
                self._file.flush()
                self._last_flush_ts = now
            except Exception as exc:
                self._errors += 1
                raise exc
            return
        self.flush_fsync()

    # ------------------------------------------------------------------
    def write(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._dropped += 1
            return
        ts_ms = int(row.get("ts_ms", 0))
        payload = {k: row.get(k, "") for k in self.header}
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                self._write_row(payload, ts_ms)
            except Exception as exc:
                last_exc = exc
                self._errors += 1
                if attempt == 0:
                    try:
                        self.reopen()
                    except Exception:
                        break
                    else:
                        self._retries += 1
                        continue
                break
            else:
                self._written += 1
                return
        self._dropped += 1
        if last_exc is not None:
            raise last_exc

    # ------------------------------------------------------------------
    def flush_fsync(self, *, force: bool = False) -> None:
        if not self._file:
            return
        try:
            self._file.flush()
            if force or self._fsync_mode != "off":
                atomic_write_with_retry(self.path, None, retries=3, backoff=0.1)
            self._last_flush_ts = time.monotonic()
        except Exception as exc:
            self._errors += 1
            raise exc

    # ------------------------------------------------------------------
    def reopen(self) -> None:
        self.flush_fsync(force=True)
        self.close()
        self._open_file()

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "fsync_mode": self._fsync_mode,
            "rotate_daily": self._rotate_daily,
            "flush_interval_s": self._flush_interval_s,
            "written": self._written,
            "retries": self._retries,
            "errors": self._errors,
            "dropped": self._dropped,
            "open": self._file is not None,
        }

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._writer = None


__all__ = ["SignalCSVWriter"]
