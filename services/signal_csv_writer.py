from __future__ import annotations

import csv
import os
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

    def __init__(self, path: str, header: Sequence[str] | None = None) -> None:
        self.path = str(path)
        self.header = list(header) if header is not None else list(self.DEFAULT_HEADER)
        self._file: Any | None = None
        self._writer: csv.DictWriter | None = None
        self._day: date | None = None
        self._ensure_dir()
        self._rotate_existing()
        self._open_file(initial=True)

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

    # ------------------------------------------------------------------
    def _maybe_rotate(self, ts_ms: int) -> None:
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
    def write(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            return
        ts_ms = int(row.get("ts_ms", 0))
        self._maybe_rotate(ts_ms)
        self._writer.writerow({k: row.get(k, "") for k in self.header})

    # ------------------------------------------------------------------
    def flush_fsync(self) -> None:
        if not self._file:
            return
        try:
            self._file.flush()
            atomic_write_with_retry(self.path, None, retries=3, backoff=0.1)
        except Exception:
            pass

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
