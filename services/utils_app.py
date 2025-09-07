from __future__ import annotations

import os
import platform
import signal
import subprocess
import json
from typing import Any, Dict, List, Optional

import pandas as pd


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d:
        os.makedirs(d, exist_ok=True)


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
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-n:]
        return "".join(lines)
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


def load_signals_full(path: str, max_rows: int = 500) -> pd.DataFrame:
    df = read_csv(path, n=max_rows)
    if df.empty:
        return df
    try:
        if "uid" not in df.columns:
            df["uid"] = df.apply(lambda r: signal_uid(r.to_dict()), axis=1)
    except Exception:
        pass
    return df
