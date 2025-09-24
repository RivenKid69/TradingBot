from __future__ import annotations

import os
import platform
import signal
import subprocess
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.fspath(path)) or "."
    os.makedirs(directory, exist_ok=True)


# ------------------------------------------------------------------
# Atomic file writing with retries
ALERT_LEVEL = logging.CRITICAL + 10
logging.addLevelName(ALERT_LEVEL, "ALERT")


def atomic_write_with_retry(
    path: str | Path,
    data: str | bytes | None,
    retries: int = 3,
    backoff: float = 0.1,
    mode: str = "w",
) -> None:
    """Atomically write *data* to *path* with retry logic.

    If ``data`` is ``None`` the function will simply ``fsync`` the existing
    file at ``path``.  When ``data`` is provided it is written to a temporary
    file which is then ``os.replace``'d into place.  In both cases the
    directory containing ``path`` is created if necessary and ``fsync`` is
    attempted for durability.  After all retry attempts are exhausted the
    failure is logged at ALERT level and the exception is re-raised.
    """

    p = Path(path)
    for attempt in range(retries + 1):
        try:
            if data is None:
                fd = os.open(str(p), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                if mode == "a":
                    if isinstance(data, (bytes, bytearray)):
                        with open(p, "ab") as f:
                            f.write(data)
                            f.flush()
                            os.fsync(f.fileno())
                    else:
                        with open(p, "a", encoding="utf-8", newline="") as f:
                            f.write(str(data))
                            f.flush()
                            os.fsync(f.fileno())
                    try:
                        dir_fd = os.open(str(p.parent), os.O_DIRECTORY)
                        os.fsync(dir_fd)
                        os.close(dir_fd)
                    except Exception:
                        pass
                else:
                    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent))
                    try:
                        if isinstance(data, (bytes, bytearray)):
                            with os.fdopen(fd, "wb") as f:
                                f.write(data)
                                f.flush()
                                os.fsync(f.fileno())
                        else:
                            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                                f.write(str(data))
                                f.flush()
                                os.fsync(f.fileno())
                        os.replace(tmp_path, str(p))
                        try:
                            dir_fd = os.open(str(p.parent), os.O_DIRECTORY)
                            os.fsync(dir_fd)
                            os.close(dir_fd)
                        except Exception:
                            pass
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
            return
        except Exception:
            if attempt >= retries:
                logging.getLogger(__name__).log(ALERT_LEVEL, "Failed to write %s", path, exc_info=True)
                raise
            time.sleep(backoff)


def run_cmd(cmd: List[str], cwd: Optional[str] = None, log_path: Optional[str] = None) -> int:
    """Blocking command execution with optional logging."""
    if log_path:
        ensure_dir(log_path)
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=f, text=True)
            return int(proc.returncode)
    else:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if proc.stderr:
            print(proc.stderr)
        return int(proc.returncode)


def start_background(cmd: List[str], pid_file: str, log_file: str) -> int:
    """Start background process and save PID."""
    ensure_dir(pid_file)
    ensure_dir(log_file)
    if os.path.exists(pid_file):
        raise RuntimeError("Process already running (PID file exists). Stop it first.")
    logf = open(log_file, "a", encoding="utf-8", newline="")
    if platform.system() == "Windows":
        creationflags = 0x00000200
        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, creationflags=creationflags)
    else:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, preexec_fn=os.setsid)
    with open(pid_file, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))
    return int(proc.pid)


def stop_background(pid_file: str) -> bool:
    """Stop background process using stored PID."""
    if not os.path.exists(pid_file):
        return False
    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
    except Exception:
        os.remove(pid_file)
        return False
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
    try:
        os.remove(pid_file)
    except Exception:
        pass
    return True


def background_running(pid_file: str) -> bool:
    if not os.path.exists(pid_file):
        return False
    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        if platform.system() == "Windows":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
            return str(pid) in (out.stdout or "")
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        try:
            os.remove(pid_file)
        except Exception:
            pass
        return False


def tail_file(path: str, n: int = 200) -> str:
    if n <= 0 or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            if end <= 0:
                return ""
            block_size = 8192
            chunks: list[bytes] = []
            lines_found = 0
            while end > 0 and lines_found <= n:
                read_size = block_size if end >= block_size else end
                fh.seek(end - read_size)
                chunk = fh.read(read_size)
                chunks.append(chunk)
                lines_found += chunk.count(b"\n")
                end -= read_size
            data = b"".join(reversed(chunks))
            text = data.decode("utf-8", errors="ignore")
            lines = text.splitlines()
            if len(lines) > n:
                lines = lines[-n:]
            return "\n".join(lines)
    except Exception:
        return ""


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_csv(path: str, n: int = 200) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if len(df) > n:
            return df.tail(n).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def read_signals_csv(path: str, n: int = 200) -> pd.DataFrame:
    df = read_csv(path, n=n)
    if df.empty:
        return df
    try:
        if "uid" not in df.columns:
            df["uid"] = df.apply(lambda r: signal_uid(r.to_dict()), axis=1)
    except Exception:
        pass
    return df


def signal_uid(row: Dict[str, Any]) -> str:
    ts = str(int(row.get("ts_ms", 0)))
    sym = str(row.get("symbol", "")).upper()
    fh = str(row.get("features_hash", ""))
    side = str(row.get("side", ""))
    vol = str(row.get("volume_frac", ""))
    return f"{ts}_{sym}_{fh}_{side}_{vol}"


def append_row_csv(path: str, header: List[str], row: List[Any]) -> None:
    ensure_dir(path)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def append_jsonl(path: str, data: Dict[str, Any]) -> None:
    atomic_write_with_retry(
        path,
        json.dumps(data, separators=(",", ":")) + "\n",
        mode="a",
    )


def load_signals_full(path: str, max_rows: int = 500) -> pd.DataFrame:
    return read_signals_csv(path, n=max_rows)


__all__ = [
    "ensure_dir",
    "atomic_write_with_retry",
    "run_cmd",
    "start_background",
    "stop_background",
    "background_running",
    "tail_file",
    "read_json",
    "read_csv",
    "read_signals_csv",
    "signal_uid",
    "append_row_csv",
    "append_jsonl",
    "load_signals_full",
]
