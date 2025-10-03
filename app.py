# app.py
from __future__ import annotations

import io
import os
import sys
import json
import time
import subprocess
import copy
import difflib
import shlex
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Header
import pandas as pd
import streamlit as st
import yaml

from utils_time import load_seasonality

from core_config import ClockSyncConfig, load_config, load_config_from_str
from ingest_config import (
    load_config as load_ingest_config,
    load_config_from_str as parse_ingest_config,
)
from legacy_sandbox_config import (
    load_config as load_sandbox_config,
    load_config_from_str as parse_sandbox_config,
    SandboxConfig,
)

import clock
from services import monitoring
from services.rest_budget import RestBudgetSession
from services.utils_app import (
    ensure_dir as _ensure_dir,
    run_cmd,
    start_background,
    stop_background,
    background_running,
    tail_file,
    read_json,
    read_csv,
    append_row_csv,
    load_signals_full,
    atomic_write_with_retry,
)
from service_backtest import BacktestConfig, from_config as backtest_from_config
from service_calibrate_slippage import (
    from_config as calibrate_slippage_from_config,
)
from service_calibrate_tcost import TCostCalibrateConfig, run as calibrate_tcost_run
from service_signal_runner import (
    ServiceSignalRunner,
    RunnerConfig,
    clear_dirty_restart,
)
from service_eval import ServiceEval, EvalConfig
from runtime_trade_defaults import (
    DEFAULT_RUNTIME_TRADE_PATH,
    load_runtime_trade_defaults,
)


_ROOT_DIR = Path(__file__).resolve().parent
_INGEST_SCRIPT = str(_ROOT_DIR / "ingest_orchestrator.py")
_MAKE_FEATURES_SCRIPT = str(_ROOT_DIR / "make_features.py")
_BUILD_TRAINING_TABLE_SCRIPT = str(_ROOT_DIR / "build_training_table.py")


# --------------------------- Seasonality API ---------------------------

API_TOKEN = os.environ.get("SEASONALITY_API_TOKEN", "changeme")

api = FastAPI()


def _check_auth(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """Simple header-based authentication."""
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@api.get("/seasonality")
def fetch_seasonality(
    path: str = "data/latency/liquidity_latency_seasonality.json",
    _: None = Depends(_check_auth),
) -> Dict[str, Any]:
    """Return seasonality multipliers from JSON file."""
    try:
        data = load_seasonality(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Seasonality file not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {k: v.tolist() for k, v in data.items()}


@api.post("/seasonality/refresh")
def refresh_seasonality(
    data: str = "data/seasonality_source/latest.parquet",
    out: str = "data/latency/liquidity_latency_seasonality.json",
    _: None = Depends(_check_auth),
) -> Dict[str, Any]:
    """Rebuild seasonality JSON from historical data and return it."""
    cmd = [
        sys.executable,
        "scripts/build_hourly_seasonality.py",
        "--data",
        data,
        "--out",
        out,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=res.stderr)
    try:
        sdata = load_seasonality(out)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Seasonality JSON not generated")
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {k: v.tolist() for k, v in sdata.items()}


@api.get("/monitoring/snapshot")
def monitoring_snapshot(
    path: str = "logs/snapshot_metrics.json",
    _: None = Depends(_check_auth),
) -> Dict[str, Any]:
    """Return monitoring metrics snapshot from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Snapshot file not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# --------------------------- Utility ---------------------------


def build_all_pipeline(
    *,
    py: str,
    cfg_ingest: str,
    prices_in: str,
    features_out: str,
    lookbacks: str,
    rsi_period: int,
    bt_base: str,
    bt_prices: str,
    bt_price_col: str,
    bt_decision_delay: int,
    bt_horizon: int,
    bt_out: str,
    cfg_sandbox: str,
    trades_path: str,
    reports_path: str,
    metrics_json: str,
    out_md: str,
    equity_png: str,
    cfg_realtime: str,
    start_realtime: bool,
    realtime_pid: str,
    realtime_log: str,
    logs_dir: str,
) -> None:
    rc = run_cmd(
        [py, _INGEST_SCRIPT, "--config", cfg_ingest],
        log_path=os.path.join(logs_dir, "ingest.log"),
    )
    if rc != 0:
        st.error(f"Ingest завершился с кодом {rc}")
        return

    rc = run_cmd(
        [
            py,
            _MAKE_FEATURES_SCRIPT,
            "--in",
            prices_in,
            "--out",
            features_out,
            "--lookbacks",
            lookbacks,
            "--rsi-period",
            str(int(rsi_period)),
        ],
        log_path=os.path.join(logs_dir, "features.log"),
    )
    if rc != 0:
        st.error(f"make_features завершился с кодом {rc}")
        return

    args = [
        py,
        _BUILD_TRAINING_TABLE_SCRIPT,
        "--base",
        bt_base,
        "--prices",
        bt_prices,
        "--price-col",
        bt_price_col,
        "--decision-delay-ms",
        str(int(bt_decision_delay)),
        "--label-horizon-ms",
        str(int(bt_horizon)),
        "--out",
        bt_out,
    ]
    rc = run_cmd(args, log_path=os.path.join(logs_dir, "train_table.log"))
    if rc != 0:
        st.error(f"build_training_table завершился с кодом {rc}")
        return

    try:
        run_backtest_from_yaml(cfg_sandbox, reports_path, logs_dir)
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        return

    eval_cfg = EvalConfig(
        trades_path=trades_path,
        reports_path=reports_path,
        out_json=metrics_json,
        out_md=out_md,
        equity_png=equity_png,
        capital_base=10000.0,
        rf_annual=0.0,
    )
    try:
        ServiceEval(eval_cfg).run()
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return

    st.success("Полный прогон: метрики готовы")

    if start_realtime:
        if background_running(realtime_pid):
            st.info("Realtime сигналер уже запущен")
        else:
            try:
                pid = start_background(
                    [py, "script_live.py", "--config", cfg_realtime],
                    pid_file=realtime_pid,
                    log_file=realtime_log,
                )
                st.success(f"Realtime сигналер запущен, PID={pid}")
            except Exception as e:
                st.error(str(e))


# --------------------------- Service wrappers ---------------------------


def run_backtest_from_yaml(
    cfg_path: str,
    default_out: str,
    logs_dir: str,
    *,
    bar_report_path: str | None = None,
) -> str:
    cfg: SandboxConfig = load_sandbox_config(cfg_path)
    sim_cfg = load_config(cfg.sim_config_path)

    sb_cfg = BacktestConfig(
        symbol=(
            sim_cfg.data.symbols[0]
            if getattr(getattr(sim_cfg, "data", None), "symbols", [])
            else "BTCUSDT"
        ),
        timeframe=getattr(sim_cfg.data, "timeframe", "1m"),
        dynamic_spread_config=cfg.dynamic_spread,
        exchange_specs_path=cfg.exchange_specs_path,
        guards_config=cfg.sim_guards,
        signal_cooldown_s=int(cfg.min_signal_gap_s),
        no_trade_config=cfg.no_trade,
        logs_dir=logs_dir,
        bar_report_path=bar_report_path or cfg.bar_report_path,
    )

    data_cfg = cfg.data
    path = data_cfg.path
    df = (
        pd.read_parquet(path)
        if path.lower().endswith(".parquet")
        else pd.read_csv(path)
    )
    reports = backtest_from_config(
        sim_cfg,
        df,
        ts_col=data_cfg.ts_col,
        symbol_col=data_cfg.symbol_col,
        price_col=data_cfg.price_col,
        svc_cfg=sb_cfg,
    )
    out_path = cfg.out_reports or default_out
    _ensure_dir(out_path)
    if out_path.lower().endswith(".parquet"):
        pd.DataFrame(reports).to_parquet(out_path, index=False)
    else:
        pd.DataFrame(reports).to_csv(out_path, index=False)
    return out_path


# --------------------------- YAML helpers ---------------------------


def _load_yaml_file(path: str) -> tuple[Dict[str, Any], str]:
    if not path or not os.path.exists(path):
        return {}, ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except Exception:
        return {}, ""
    try:
        data = yaml.safe_load(content) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data, content


def _dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _show_diff(old: str, new: str, label: str) -> str:
    diff = "\n".join(
        difflib.unified_diff(
            (old or "").splitlines(),
            (new or "").splitlines(),
            fromfile=f"{label} (old)",
            tofile=f"{label} (new)",
            lineterm="",
        )
    )
    if diff.strip():
        st.code(diff, language="diff")
    else:
        st.info("Изменений нет")
    return diff


def _load_latest_metrics(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            file_size = fh.tell()
            if file_size <= 0:
                return {}
            chunk = 4096
            buffer = bytearray()
            position = file_size
            while position > 0 and buffer.count(b"\n") < 2:
                read = min(chunk, position)
                fh.seek(position - read)
                data = fh.read(read)
                if not data:
                    break
                buffer = data + buffer
                position -= read
            lines = [line for line in buffer.splitlines() if line.strip()]
            if not lines:
                return {}
            last_line = lines[-1].decode("utf-8")
        return json.loads(last_line)
    except Exception:
        return {}


def _extract_cache_ttl_days(config_path: str) -> float | None:
    data, _ = _load_yaml_file(config_path)
    if not data:
        return None

    def _dig(payload: Dict[str, Any], path: List[str]) -> Any:
        current: Any = payload
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    candidate_paths = [
        ["offline", "cache", "ttl_days"],
        ["offline", "cache_ttl_days"],
        ["rest_budget", "cache", "ttl_days"],
        ["rest_budget", "cache_ttl_days"],
        ["cache", "ttl_days"],
    ]
    for path in candidate_paths:
        value = _dig(data, path)
        if value is None:
            continue
        try:
            ttl = float(value)
        except (TypeError, ValueError):
            continue
        if ttl < 0:
            continue
        return ttl
    return None


def _json_preview(payload: Any, limit: int = 10) -> tuple[Any, bool]:
    truncated = False
    if isinstance(payload, list):
        if len(payload) > limit:
            truncated = True
        return payload[:limit], truncated

    if isinstance(payload, dict):
        preview: Dict[str, Any] = {}
        for key, value in payload.items():
            if key == "filters" and isinstance(value, dict):
                items = list(value.items())
                if len(items) > limit:
                    truncated = True
                preview[key] = {k: v for k, v in items[:limit]}
            else:
                preview[key] = value
        return preview, truncated

    return payload, truncated


# --------------------------- Streamlit UI ---------------------------

st.set_page_config(page_title="Trading Signals Control", layout="wide")

st.title("Панель управления проектом (сигнальный mid-freq)")

with st.sidebar:
    st.header("Глобальные пути")
    py = sys.executable
    st.caption(f"Python: `{py}`")

    cfg_ingest = st.text_input("configs/ingest.yaml", value="configs/ingest.yaml")
    cfg_sandbox = st.text_input("configs/sandbox.yaml", value="configs/sandbox.yaml")
    cfg_sim = st.text_input("configs/sim.yaml", value="configs/sim.yaml")
    cfg_realtime = st.text_input("configs/realtime.yaml", value="configs/realtime.yaml")
    rest_budget_cfg = st.text_input(
        "Rest budget (clock sync)", value="configs/rest_budget.yaml"
    )

    logs_dir = st.text_input("Каталог логов", value="logs")
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)

    trades_path = st.text_input(
        "Путь к трейдам (для evaluate)",
        value=os.path.join(logs_dir, "log_trades_*.csv"),
    )
    reports_path = st.text_input(
        "Путь к отчётам (для evaluate)", value=os.path.join(logs_dir, "reports.csv")
    )
    metrics_json = st.text_input(
        "Выход метрик JSON", value=os.path.join(logs_dir, "metrics.json")
    )
    snapshot_json = st.text_input(
        "Snapshot metrics JSON", value=os.path.join(logs_dir, "snapshot_metrics.json")
    )
    snapshot_csv = st.text_input(
        "Snapshot metrics CSV", value=os.path.join(logs_dir, "snapshot_metrics.csv")
    )
    equity_png = st.text_input(
        "PNG с equity", value=os.path.join(logs_dir, "equity.png")
    )
    signals_csv = st.text_input(
        "Файл сигналов (realtime)", value=os.path.join(logs_dir, "signals.csv")
    )
    realtime_log = st.text_input(
        "Лог realtime", value=os.path.join(logs_dir, "realtime.log")
    )
    realtime_pid = st.text_input(
        "PID-файл realtime", value=os.path.join(".run", "rt_signaler.pid")
    )

tabs = st.tabs(
    [
        "Статус",
        "Ingest",
        "Features",
        "Training Table",
        "Sandbox Backtest",
        "Evaluate",
        "Realtime Signaler",
        "Исполнение",
        "Логи",
        "Полный прогон",
        "Model Train",
        "YAML-редактор",
        "Sim Settings",
        "T-cost Calibrate",
        "Target Builder",
        "No-Trade Mask",
        "Walk-Forward Splits",
        "Threshold Tuner",
        "Probability Calibration",
        "Drift Monitor",
        "Monitoring",
        "Offline Jobs",
    ]
)
# --------------------------- Tab: Status ---------------------------

with tabs[0]:
    st.subheader("Ключевые показатели")

    col1, col2, col3 = st.columns(3)
    with col1:
        m = read_json(metrics_json)
        eq = m.get("equity", {})
        pnl_total = eq.get("pnl_total", None)
        sharpe = eq.get("sharpe", None)
        maxdd = eq.get("max_drawdown", None)
        st.metric(
            "PNL total",
            f"{pnl_total:.2f}" if isinstance(pnl_total, (int, float)) else "—",
        )
        st.metric(
            "Sharpe", f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "—"
        )
        st.metric(
            "Max Drawdown", f"{maxdd:.4f}" if isinstance(maxdd, (int, float)) else "—"
        )
    with col2:
        running = background_running(realtime_pid)
        st.metric("Realtime сигналер", "запущен" if running else "остановлен")
        sig_df = read_csv(signals_csv, n=1)
        last_sig_ts = (
            int(sig_df.iloc[-1]["ts_ms"])
            if not sig_df.empty and "ts_ms" in sig_df.columns
            else None
        )
        st.metric("Последний сигнал ts_ms", str(last_sig_ts) if last_sig_ts else "—")
    with col3:
        rep_df = read_csv(reports_path, n=1)
        last_eq = (
            float(rep_df.iloc[-1]["equity"])
            if not rep_df.empty and "equity" in rep_df.columns
            else None
        )
        st.metric(
            "Equity (последняя точка)",
            f"{last_eq:.2f}" if isinstance(last_eq, (int, float)) else "—",
        )

    st.divider()
    st.subheader("Equity (если есть)")
    if os.path.exists(equity_png):
        st.image(equity_png, caption="Equity curve")
    else:
        st.info("Файл equity.png пока не найден. Сгенерируйте через раздел Evaluate.")

    st.divider()
    st.subheader("Monitoring snapshot")
    snap = read_json(snapshot_json)
    if snap:
        st.json(snap)
    else:
        st.info("Snapshot metrics JSON не найден.")
    snap_df = read_csv(snapshot_csv)
    if not snap_df.empty:
        st.dataframe(snap_df)

    st.divider()
    st.subheader("Universe & Filters")

    offline_config_path = st.text_input(
        "Offline config (TTL)", value="configs/offline.yaml", key="offline_config_universe"
    )
    ttl_days = _extract_cache_ttl_days(offline_config_path)
    if offline_config_path and not os.path.exists(offline_config_path):
        st.warning(f"Offline config не найден: {offline_config_path}")
    elif ttl_days is not None:
        st.caption(f"TTL из offline config: {ttl_days:g} дней")
    else:
        st.caption("TTL (offline.cache.ttl_days) не найден в offline config.")

    universe_path = st.text_input(
        "Universe JSON", value="data/universe/symbols.json", key="universe_json_path"
    )
    filters_path = st.text_input(
        "Filters JSON", value="data/binance_filters.json", key="filters_json_path"
    )

    preview_limit = 10

    def _count_universe_entries(payload: Any) -> Optional[int]:
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            symbols = payload.get("symbols")
            if isinstance(symbols, list):
                return len(symbols)
        return None

    def _count_filter_entries(payload: Any) -> Optional[int]:
        if isinstance(payload, dict):
            filters_block = payload.get("filters")
            if isinstance(filters_block, dict):
                return len(filters_block)
        return None

    def _render_json_file(
        *,
        label: str,
        path: str,
        count_fn,
        log_name: str,
        refresh_cmd: List[str],
    ) -> None:
        st.markdown(f"### {label}")
        st.caption(f"`{path}`")

        button_key = f"refresh_{log_name}"
        if st.button("Refresh", key=button_key):
            rc = run_cmd(
                refresh_cmd,
                log_path=os.path.join(logs_dir, f"{log_name}.log"),
            )
            if rc == 0:
                st.success("Успешно обновлено")
            else:
                st.error(f"Refresh завершился с кодом {rc}")

        exists = os.path.exists(path)
        payload: Any = None
        if exists:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception as exc:
                st.error(f"Ошибка чтения {path}: {exc}")
                payload = None
        else:
            st.warning("Файл не найден. Нажмите Refresh, чтобы скачать данные.")

        mtime_ts = os.path.getmtime(path) if exists else None
        age_days = None
        if mtime_ts is not None:
            age_days = (time.time() - mtime_ts) / 86400.0

        count_value = count_fn(payload) if payload is not None else None
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric(
                "Количество",
                str(count_value) if count_value is not None else "—",
            )
        with info_cols[1]:
            mtime_text = (
                datetime.utcfromtimestamp(mtime_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
                if mtime_ts is not None
                else "—"
            )
            st.metric("mtime", mtime_text)
        with info_cols[2]:
            st.metric(
                "Возраст (дней)",
                f"{age_days:.2f}" if age_days is not None else "—",
            )

        if ttl_days is not None and ttl_days > 0 and age_days is not None and age_days > ttl_days:
            st.warning(
                "Файл старше TTL. Нажмите Refresh, чтобы обновить данные.",
            )

        if payload is not None:
            preview, truncated = _json_preview(payload, limit=preview_limit)
            st.code(
                json.dumps(preview, ensure_ascii=False, indent=2),
                language="json",
            )
            if truncated:
                st.caption(f"Показаны первые {preview_limit} элементов.")
            st.download_button(
                "Скачать JSON",
                data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=os.path.basename(path) or "data.json",
                mime="application/json",
                key=f"download_{log_name}",
            )

    _render_json_file(
        label="Universe (symbols)",
        path=universe_path,
        count_fn=_count_universe_entries,
        log_name="refresh_universe",
        refresh_cmd=[
            sys.executable,
            "scripts/refresh_universe.py",
            "--config",
            offline_config_path,
            "--out",
            universe_path,
        ],
    )

    _render_json_file(
        label="Binance filters",
        path=filters_path,
        count_fn=_count_filter_entries,
        log_name="refresh_filters",
        refresh_cmd=[
            sys.executable,
            "scripts/fetch_binance_filters.py",
            "--config",
            offline_config_path,
            "--out",
            filters_path,
        ],
    )

    st.divider()
    st.subheader("Quantizer — настройки фильтров")

    quantizer_targets = [
        ("configs/quantizer.yaml", "Quantizer (offline)"),
        ("configs/config_live.yaml", "Live config"),
    ]

    for idx, (cfg_path, label) in enumerate(quantizer_targets):
        with st.expander(f"{label} — `{cfg_path}`", expanded=(idx == 0)):
            data, _ = _load_yaml_file(cfg_path)
            quantizer_cfg: Dict[str, Any] = {}
            if isinstance(data, dict):
                section = data.get("quantizer")
                if isinstance(section, dict):
                    quantizer_cfg = section
            strict_default = bool(
                quantizer_cfg.get(
                    "strict_filters",
                    quantizer_cfg.get("strict", True),
                )
            )
            enforce_default = bool(
                quantizer_cfg.get("enforce_percent_price_by_side", True)
            )

            with st.form(f"quantizer_form_{idx}"):
                strict_value = st.checkbox(
                    "quantizer.strict_filters",
                    value=strict_default,
                )
                enforce_value = st.checkbox(
                    "quantizer.enforce_percent_price_by_side",
                    value=enforce_default,
                )
                submitted = st.form_submit_button("Сохранить")

            if submitted:
                new_payload: Dict[str, Any] = copy.deepcopy(data) if isinstance(data, dict) else {}
                if not isinstance(new_payload, dict):
                    new_payload = {}
                new_payload.setdefault("quantizer", {})
                if not isinstance(new_payload["quantizer"], dict):
                    new_payload["quantizer"] = {}
                new_payload["quantizer"]["strict_filters"] = bool(strict_value)
                new_payload["quantizer"]["enforce_percent_price_by_side"] = bool(
                    enforce_value
                )
                if "strict" in new_payload["quantizer"]:
                    new_payload["quantizer"]["strict"] = bool(strict_value)
                try:
                    atomic_write_with_retry(
                        cfg_path,
                        _dump_yaml(new_payload),
                    )
                except Exception as exc:
                    st.error(f"Не удалось сохранить {cfg_path}: {exc}")
                else:
                    st.success(f"Сохранено: {cfg_path}")

# --------------------------- Tab: Ingest ---------------------------

with tabs[1]:
    st.subheader("Публичный Ingest (orchestrator)")
    st.caption(
        "Читает configs/ingest.yaml и запускает полный цикл: klines → aggregate → funding/mark → prices."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Запустить ingest", type="primary"):
            rc = run_cmd(
                [
                    sys.executable,
                    _INGEST_SCRIPT,
                    "--config",
                    cfg_ingest,
                ],
                log_path=os.path.join(logs_dir, "ingest.log"),
            )
            if rc == 0:
                st.success("Ingest завершён успешно")
            else:
                st.error(f"Ingest завершился с кодом {rc}")
    with c2:
        if st.button("Показать configs/ingest.yaml"):
            try:
                with open(cfg_ingest, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="yaml")
            except Exception as e:
                st.error(str(e))

# --------------------------- Tab: Features ---------------------------

with tabs[2]:
    st.subheader("Оффлайн-фичи (единый код с онлайном)")
    prices_in = st.text_input("Входной prices.parquet/csv", value="data/prices.parquet")
    features_out = st.text_input(
        "Выход features.parquet", value="data/features.parquet"
    )
    lookbacks = st.text_input("Окна SMA/ret (через запятую)", value="5,15,60")
    rsi_period = st.number_input(
        "RSI period", min_value=2, max_value=200, value=14, step=1
    )

    if st.button("Собрать фичи (make_features.py)", type="primary"):
        rc = run_cmd(
            [
                sys.executable,
                _MAKE_FEATURES_SCRIPT,
                "--in",
                prices_in,
                "--out",
                features_out,
                "--lookbacks",
                lookbacks,
                "--rsi-period",
                str(int(rsi_period)),
            ],
            log_path=os.path.join(logs_dir, "features.log"),
        )
        if rc == 0:
            st.success("Фичи собраны")
        else:
            st.error(f"make_features завершился с кодом {rc}")

# --------------------------- Tab: Training Table ---------------------------

with tabs[3]:
    st.subheader("Training table (сбор меток и merge asof)")
    st.caption(
        "Вызов вашего build_training_table.py (аргументы подставьте под ваш проект)."
    )

    bt_base = st.text_input("features base (--base)", value="data/features.parquet")
    bt_prices = st.text_input("prices (--prices)", value="data/prices.parquet")
    bt_price_col = st.text_input("price col (--price-col)", value="price")
    bt_decision_delay = st.number_input(
        "decision_delay_ms", min_value=0, value=500, step=100
    )
    bt_horizon = st.number_input(
        "label_horizon_ms", min_value=60_000, value=3_600_000, step=60_000
    )
    bt_out = st.text_input("Выход train.parquet (--out)", value="data/train.parquet")

    extra_sources = st.text_area(
        "Доп. источники (--sources, JSON-список объектов) (опционально)",
        value="",
        placeholder='Например: [{"name":"funding","path":"data/futures/BTCUSDT_funding.parquet","time_col":"ts_ms","keys":["symbol"],"direction":"backward","tolerance_ms":86400000}]',
    )

    if st.button("Собрать training table", type="primary"):
        args = [
            sys.executable,
            _BUILD_TRAINING_TABLE_SCRIPT,
            "--base",
            bt_base,
            "--prices",
            bt_prices,
            "--price-col",
            bt_price_col,
            "--decision-delay-ms",
            str(int(bt_decision_delay)),
            "--label-horizon-ms",
            str(int(bt_horizon)),
            "--out",
            bt_out,
        ]
        if extra_sources.strip():
            args += ["--sources", extra_sources.strip()]
        rc = run_cmd(args, log_path=os.path.join(logs_dir, "train_table.log"))
        if rc == 0:
            st.success("Training table собрана")
        else:
            st.error(f"build_training_table завершился с кодом {rc}")

# --------------------------- Tab: Sandbox Backtest ---------------------------

with tabs[4]:
    st.subheader("Бэктест песочницы (ServiceBacktest)")
    default_rep = os.path.join(logs_dir, "sandbox_reports.csv")
    if st.button("Запустить бэктест", type="primary"):
        try:
            out = run_backtest_from_yaml(cfg_sandbox, default_rep, logs_dir)
            st.success(f"Бэктест завершён, отчёт: {out}")
        except Exception as e:
            st.error(str(e))

    st.caption("Текущий configs/sandbox.yaml:")
    try:
        with open(cfg_sandbox, "r", encoding="utf-8") as f:
            st.code(f.read(), language="yaml")
    except Exception as e:
        st.error(str(e))

# --------------------------- Tab: Evaluate ---------------------------

with tabs[5]:
    st.subheader("Оценка эффективности (ServiceEval)")
    artifacts_dir_eval = st.text_input(
        "Каталог артефактов", value=os.path.join(logs_dir, "eval")
    )
    if st.button("Посчитать метрики", type="primary"):
        cfg_eval = EvalConfig(
            trades_path=trades_path,
            reports_path=reports_path,
            artifacts_dir=artifacts_dir_eval,
            out_json=metrics_json,
            out_md=os.path.join(artifacts_dir_eval, "metrics.md"),
            equity_png=os.path.join(artifacts_dir_eval, "equity.png"),
        )
        try:
            ServiceEval(cfg_eval).run()
            st.success("Метрики готовы")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("metrics.json (если есть)")
    mj = read_json(metrics_json)
    if mj:
        st.json(mj)
    else:
        st.info("metrics.json пока не найден.")

# --------------------------- Tab: Realtime Signaler ---------------------------

with tabs[6]:
    st.subheader("Realtime сигналер (WebSocket Binance, без ключей)")

    runtime_yaml = st.text_input("Runtime YAML", value="configs/runtime.yaml")
    live_yaml = st.text_input(
        "Основной конфиг (config_live.yaml)", value="configs/config_live.yaml"
    )
    trade_cols = st.columns([3, 1])
    with trade_cols[0]:
        runtime_trade_yaml = st.text_input(
            "Runtime trade config",
            value=DEFAULT_RUNTIME_TRADE_PATH,
            help="YAML с дефолтами для execution/portfolio/costs",
        )
    with trade_cols[1]:
        if st.button("Reload trade defaults", type="secondary"):
            st.experimental_rerun()
    if runtime_trade_yaml and not Path(runtime_trade_yaml).exists():
        st.warning(f"Файл {runtime_trade_yaml} не найден")
    st.caption(f"Текущий runtime trade config: {runtime_trade_yaml}")
    runtime_trade_defaults = load_runtime_trade_defaults(
        runtime_trade_yaml, loader=_load_yaml_file
    )

    runner_status = read_json(os.path.join(logs_dir, "runner_status.json"))
    running = background_running(realtime_pid)

    safe_state = runner_status.get("safe_mode", {}) or {}
    queue_info = runner_status.get("queue", {}) or {}
    status_cols = st.columns(5)
    with status_cols[0]:
        st.metric("Safe mode", "ON" if safe_state.get("active") else "OFF")
    with status_cols[1]:
        st.metric(
            "Queue",
            f"{queue_info.get('size', 0)}/{queue_info.get('max', 0)}",
        )
    with status_cols[2]:
        st.metric("Workers", len(runner_status.get("workers", [])))
    with status_cols[3]:
        st.metric("Процесс", "запущен" if running else "остановлен")
    dirty_status = runner_status.get("dirty_restart") or {}
    dirty_active = bool(dirty_status.get("active"))
    detected_ms = dirty_status.get("detected_at_ms")
    detected_label = ""
    if detected_ms:
        try:
            ts = datetime.fromtimestamp(float(detected_ms) / 1000.0)
            detected_label = ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            detected_label = str(detected_ms)
    with status_cols[4]:
        st.metric(
            "Dirty restart",
            "YES" if dirty_active else "NO",
            help=f"detected at {detected_label}" if detected_label else None,
        )

    workers_payload = runner_status.get("workers") or []
    if workers_payload:
        st.dataframe(pd.DataFrame(workers_payload), use_container_width=True)
    else:
        st.info("Статус воркеров пока не доступен.")

    marker_path = os.path.join(logs_dir, "shutdown.marker")
    cfg_state = None
    state_error: str | None = None
    try:
        cfg_live_obj = load_config(live_yaml)
    except Exception as exc:
        state_error = f"Не удалось загрузить config: {exc}"
    else:
        cfg_state = getattr(cfg_live_obj, "state", None)

    control_cols = st.columns([2, 1])
    with control_cols[0]:
        confirm_clear = st.checkbox(
            "Подтверждаю очистку dirty marker", key="confirm_dirty_clear"
        )
    with control_cols[1]:
        clear_clicked = st.button(
            "Clear dirty marker",
            type="secondary",
            disabled=running or not confirm_clear or state_error is not None or cfg_state is None,
        )
    if clear_clicked:
        if state_error:
            st.error(state_error)
        elif cfg_state is None:
            st.error("StateConfig не найден в config_live.yaml")
        else:
            result = clear_dirty_restart(
                marker_path,
                cfg_state,
                runner_status_path=os.path.join(logs_dir, "runner_status.json"),
            )
            messages = []
            if result.get("marker_removed"):
                messages.append("marker удалён")
            if result.get("state_cleared"):
                messages.append("persistent state очищено")
            if result.get("status_updated"):
                messages.append("runner_status обновлён")
            if messages:
                st.success(
                    f"Очистка завершена: {', '.join(messages)}"
                )
            if result.get("errors"):
                st.error("Ошибки: " + "; ".join(map(str, result["errors"])))
    elif state_error:
        st.warning(state_error)

    last_reload = runner_status.get("last_reload") or {}
    reload_history = runner_status.get("reloads") or []
    if last_reload:
        ts_val = last_reload.get("timestamp_ms")
        timestamp_str = ""
        if ts_val:
            try:
                timestamp_str = datetime.fromtimestamp(float(ts_val) / 1000.0).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
                timestamp_str = str(ts_val)
        status_text = "успешно" if last_reload.get("success") else "ошибка"
        st.info(f"Последний reload: {status_text} ({timestamp_str})")
    if reload_history:
        table_rows = []
        for event in reversed(reload_history):
            ts_val = event.get("timestamp_ms")
            if ts_val:
                try:
                    ts_label = datetime.fromtimestamp(float(ts_val) / 1000.0).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                except Exception:
                    ts_label = str(ts_val)
            else:
                ts_label = ""
            table_rows.append(
                {
                    "timestamp": ts_label,
                    "success": bool(event.get("success")),
                    "errors": "; ".join(event.get("errors", [])) if event.get("errors") else "",
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        with st.expander("Детали последнего reload", expanded=False):
            st.json(last_reload)

    st.divider()
    cols = st.columns(3)
    with cols[0]:
        if st.button("Старт", disabled=running, type="primary"):
            try:
                pid = start_background(
                    [sys.executable, "script_live.py", "--config", cfg_realtime],
                    pid_file=realtime_pid,
                    log_file=realtime_log,
                )
                st.success(f"Сигналер запущен, PID={pid}")
            except Exception as e:
                st.error(str(e))
    with cols[1]:
        if st.button("Стоп", disabled=not running, type="secondary"):
            ok = stop_background(realtime_pid)
            if ok:
                st.success("Остановлено")
            else:
                st.error("Не удалось остановить процесс (возможно, уже не работает)")
    with cols[2]:
        if st.button("Показать configs/realtime.yaml"):
            try:
                with open(cfg_realtime, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="yaml")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown("#### Signal CSV Writer")
    signals_yaml_path = st.text_input("Signals YAML", value="configs/signals.yaml")
    signals_cfg, _signals_text = _load_yaml_file(signals_yaml_path)
    default_enabled = bool(signals_cfg.get("enabled", False))
    default_out_csv = str(signals_cfg.get("out_csv", "") or "")
    default_dedup = str(signals_cfg.get("dedup_persist", "") or "")
    try:
        default_ttl = int(signals_cfg.get("ttl_seconds", 0) or 0)
    except Exception:
        default_ttl = 0
    default_mode = str(signals_cfg.get("fsync_mode", "batch") or "batch").lower()
    if default_mode not in {"always", "batch", "off"}:
        default_mode = "batch"
    try:
        default_flush = float(signals_cfg.get("flush_interval_s", 5.0) or 0.0)
    except Exception:
        default_flush = 0.0
    default_rotate = bool(signals_cfg.get("rotate_daily", True))

    fsync_options = ["always", "batch", "off"]
    default_index = fsync_options.index(default_mode) if default_mode in fsync_options else 1

    with st.form("signals_writer_form"):
        enabled_input = st.checkbox("Включить сигнал-бас", value=default_enabled)
        out_csv_input = st.text_input("Файл сигналов CSV", value=default_out_csv)
        dedup_input = st.text_input(
            "Файл дедупликации", value=default_dedup, help="Путь к persist-файлу WS dedup"
        )
        ttl_input = st.number_input("TTL секунд", min_value=0, value=default_ttl, step=1)
        fsync_mode_input = st.selectbox(
            "fsync_mode",
            fsync_options,
            index=default_index,
            help="Режим fsync: always — каждый write, batch — по таймеру, off — только flush",
        )
        rotate_input = st.checkbox("rotate_daily", value=default_rotate)
        flush_interval_input = st.number_input(
            "flush_interval_s",
            min_value=0.0,
            value=float(default_flush),
            step=1.0,
        )
        submit_signals = st.form_submit_button("Сохранить signals.yaml")

    if submit_signals:
        new_cfg = dict(signals_cfg)
        new_cfg["enabled"] = bool(enabled_input)
        new_cfg["out_csv"] = out_csv_input or None
        new_cfg["dedup_persist"] = dedup_input or None
        new_cfg["ttl_seconds"] = int(ttl_input)
        new_cfg["fsync_mode"] = str(fsync_mode_input)
        new_cfg["rotate_daily"] = bool(rotate_input)
        new_cfg["flush_interval_s"] = float(flush_interval_input)
        try:
            _ensure_dir(signals_yaml_path)
            with open(signals_yaml_path, "w", encoding="utf-8") as wf:
                yaml.safe_dump(new_cfg, wf, sort_keys=False, allow_unicode=True)
        except Exception as exc:
            st.error(f"Не удалось сохранить {signals_yaml_path}: {exc}")
        else:
            st.success(f"Обновлено {signals_yaml_path}")
            signals_cfg = new_cfg
            default_out_csv = str(new_cfg.get("out_csv", "") or "")

    flag_path = os.path.join(logs_dir, "signal_writer_reopen.flag")
    reopen_cols = st.columns(3)
    with reopen_cols[0]:
        if st.button("Reopen", type="secondary"):
            try:
                _ensure_dir(flag_path)
                with open(flag_path, "w", encoding="utf-8") as fh:
                    fh.write(str(time.time()))
            except Exception as exc:
                st.error(f"Не удалось создать флаг reopen: {exc}")
            else:
                st.success(f"Флаг создан: {flag_path}")
    with reopen_cols[1]:
        st.write(" ")
    with reopen_cols[2]:
        st.caption("Создание файла signal_writer_reopen.flag перезапустит writer")

    writer_stats = runner_status.get("signal_writer") or {}
    if writer_stats:
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Written", int(writer_stats.get("written", 0)))
        metrics_cols[1].metric("Retries", int(writer_stats.get("retries", 0)))
        metrics_cols[2].metric("Errors", int(writer_stats.get("errors", 0)))
        metrics_cols[3].metric("Dropped", int(writer_stats.get("dropped", 0)))
        st.caption(
            f"Файл: {writer_stats.get('path', default_out_csv) or default_out_csv} | "
            f"fsync_mode={writer_stats.get('fsync_mode', '')}"
        )
        with st.expander("Полная статистика writer'а", expanded=False):
            st.json(writer_stats)
    else:
        st.info("Статистика writer'а пока не доступна.")

    preview_path = str(writer_stats.get("path") or default_out_csv or "").strip()
    if preview_path:
        preview_path = os.path.expanduser(preview_path)
        st.markdown(f"**Последние 100 строк CSV** `{preview_path}`")
        if os.path.exists(preview_path):
            tail_text = tail_file(preview_path, n=100)
            if tail_text:
                try:
                    with open(preview_path, "r", encoding="utf-8") as fh:
                        header_line = fh.readline().strip()
                except Exception:
                    header_line = ""
                lines = tail_text.splitlines()
                preview_text = tail_text
                if header_line and (not lines or lines[0] != header_line):
                    preview_text = "\n".join([header_line, *lines])
                try:
                    df_tail = pd.read_csv(io.StringIO(preview_text))
                except Exception:
                    st.text_area("Фрагмент CSV", preview_text, height=240)
                else:
                    st.dataframe(df_tail, use_container_width=True)
            else:
                st.info("Файл сигналов пока пуст.")
        else:
            st.info("Файл сигналов не найден.")
    else:
        st.info("Укажите путь к CSV в configs/signals.yaml для предпросмотра.")

    with st.expander(
        "Параметры стратегии (сохранить в configs/realtime.yaml)", expanded=False
    ):
        try:
            rt_cfg = load_config(cfg_realtime).model_dump()
            st.write("Текущая стратегия:")
            strat = rt_cfg.get("strategy", {}) or {}
            st.code(json.dumps(strat, ensure_ascii=False, indent=2), language="json")
            model_path = st.text_input(
                "Путь к модели (strategy.params.model_path)",
                value=str(strat.get("params", {}).get("model_path", "")),
            )
            thr = st.text_input(
                "Порог (strategy.params.threshold)",
                value=str(strat.get("params", {}).get("threshold", "0.0")),
            )
            if st.button("Сохранить изменения в configs/realtime.yaml"):
                new_cfg = copy.deepcopy(rt_cfg)
                new_cfg.setdefault("strategy", {}).setdefault("params", {})
                new_cfg["strategy"]["params"]["model_path"] = model_path
                try:
                    new_cfg["strategy"]["params"]["threshold"] = float(thr)
                except Exception:
                    new_cfg["strategy"]["params"]["threshold"] = thr
                _ensure_dir(cfg_realtime)
                with open(cfg_realtime, "w", encoding="utf-8") as wf:
                    yaml.safe_dump(new_cfg, wf, sort_keys=False, allow_unicode=True)
                st.success("Сохранено в configs/realtime.yaml")
        except Exception as e:
            st.error(f"Не удалось прочитать/изменить {cfg_realtime}: {e}")

    runtime_data, runtime_text = _load_yaml_file(runtime_yaml)
    live_data, live_text = _load_yaml_file(live_yaml)

    st.divider()
    st.markdown("#### Этапы пайплайна")
    stage_order = [
        "closed_bar",
        "windows",
        "anomaly",
        "extreme",
        "policy",
        "risk",
        "ttl",
        "dedup",
        "throttle",
        "publish",
    ]
    stage_labels = {
        "closed_bar": "Closed bar",
        "windows": "Windows",
        "anomaly": "Anomaly",
        "extreme": "Extreme",
        "policy": "Policy",
        "risk": "Risk",
        "ttl": "TTL",
        "dedup": "Dedup",
        "throttle": "Throttle",
        "publish": "Publish",
    }
    base_pipeline = live_data.get("pipeline") or {}
    pipeline_enabled = bool(base_pipeline.get("enabled", True))
    runtime_pipeline = runtime_data.get("pipeline") or {}
    if "enabled" in runtime_pipeline:
        pipeline_enabled = bool(runtime_pipeline.get("enabled", True))
    stage_states: Dict[str, bool] = {}
    base_stages = base_pipeline.get("stages") or {}
    for stage in stage_order:
        entry = base_stages.get(stage)
        if isinstance(entry, dict):
            stage_states[stage] = bool(entry.get("enabled", True))
        elif entry is not None:
            stage_states[stage] = bool(entry)
        else:
            stage_states[stage] = True
    runtime_stage_cfg = runtime_pipeline.get("stages") or {}
    for stage, cfg in runtime_stage_cfg.items():
        if isinstance(cfg, dict):
            stage_states[stage] = bool(cfg.get("enabled", True))
        else:
            stage_states[stage] = bool(cfg)
    with st.form("pipeline_stages_form"):
        pipeline_enabled_input = st.checkbox(
            "Pipeline включен", value=pipeline_enabled
        )
        stage_columns = st.columns(3)
        stage_inputs: Dict[str, bool] = {}
        for idx, stage in enumerate(stage_order):
            label = stage_labels.get(stage, stage)
            stage_inputs[stage] = stage_columns[idx % 3].checkbox(
                label, value=stage_states.get(stage, True)
            )
        if st.form_submit_button("Сохранить pipeline"):
            new_runtime = copy.deepcopy(runtime_data)
            pipeline_section = copy.deepcopy(runtime_pipeline)
            pipeline_section["enabled"] = bool(pipeline_enabled_input)
            existing_stages = pipeline_section.get("stages") or {}
            new_stages: Dict[str, Any] = {}
            for stage in stage_order:
                prev = existing_stages.get(stage)
                if isinstance(prev, dict):
                    params = {k: v for k, v in prev.items() if k != "enabled"}
                else:
                    params = {}
                entry = dict(params)
                entry["enabled"] = bool(stage_inputs[stage])
                new_stages[stage] = entry
            for stage, cfg in existing_stages.items():
                if stage not in new_stages:
                    new_stages[stage] = cfg
            pipeline_section["stages"] = new_stages
            new_runtime["pipeline"] = pipeline_section
            new_runtime_text = _dump_yaml(new_runtime)
            diff = _show_diff(runtime_text, new_runtime_text, runtime_yaml)
            if diff.strip():
                atomic_write_with_retry(runtime_yaml, new_runtime_text)
                st.success("Pipeline настройки сохранены")
                st.info("Запросите reload для применения изменений.")
            else:
                st.info("Нет изменений")

    st.divider()
    st.markdown("#### Execution / Costs / Portfolio overrides")
    runtime_exec_cfg = copy.deepcopy(runtime_data.get("execution") or {})
    runtime_exec_effective = copy.deepcopy(runtime_trade_defaults.get("execution") or {})
    runtime_exec_effective.update(copy.deepcopy(runtime_exec_cfg))

    runtime_portfolio_defaults = copy.deepcopy(runtime_trade_defaults.get("portfolio") or {})
    runtime_portfolio_cfg = copy.deepcopy(
        runtime_data.get("portfolio") or runtime_exec_cfg.get("portfolio") or {}
    )
    runtime_portfolio_effective = copy.deepcopy(runtime_portfolio_defaults)
    runtime_portfolio_effective.update(copy.deepcopy(runtime_exec_cfg.get("portfolio") or {}))
    runtime_portfolio_effective.update(copy.deepcopy(runtime_data.get("portfolio") or {}))

    runtime_costs_defaults = copy.deepcopy(runtime_trade_defaults.get("costs") or {})
    runtime_costs_cfg = copy.deepcopy(
        runtime_data.get("costs") or runtime_exec_cfg.get("costs") or {}
    )
    runtime_costs_effective = copy.deepcopy(runtime_costs_defaults)
    runtime_costs_effective.update(copy.deepcopy(runtime_exec_cfg.get("costs") or {}))
    runtime_costs_effective.update(copy.deepcopy(runtime_data.get("costs") or {}))
    runtime_impact_cfg = copy.deepcopy(runtime_costs_effective.get("impact") or {})

    mode_options = ["order", "bar"]
    current_mode = str(runtime_exec_effective.get("mode", "bar") or "bar").lower()
    if current_mode not in mode_options:
        current_mode = "bar"
    bar_price_default = str(runtime_exec_effective.get("bar_price") or "")

    def _format_optional_number(value: Any) -> str:
        if value in (None, ""):
            return ""
        return str(value)

    min_step_default = _format_optional_number(
        runtime_exec_effective.get("min_rebalance_step")
    )
    safety_default = _format_optional_number(runtime_exec_effective.get("safety_margin_bps"))
    equity_default = _format_optional_number(runtime_portfolio_effective.get("equity_usd"))
    taker_default = _format_optional_number(runtime_costs_effective.get("taker_fee_bps"))
    half_default = _format_optional_number(runtime_costs_effective.get("half_spread_bps"))
    impact_sqrt_default = _format_optional_number(runtime_impact_cfg.get("sqrt_coeff"))
    impact_linear_default = _format_optional_number(runtime_impact_cfg.get("linear_coeff"))

    def _parse_non_negative_float(
        raw_value: str, label: str, errors: List[str]
    ) -> float | None:
        text = str(raw_value).strip()
        if text == "":
            return None
        try:
            parsed = float(text)
        except ValueError:
            errors.append(f"{label}: не удалось преобразовать '{raw_value}' к числу")
            return None
        if parsed < 0:
            errors.append(f"{label}: значение {parsed} должно быть ≥ 0")
        return parsed

    with st.form("runtime_execution_form"):
        exec_cols = st.columns(2)
        mode_choice = exec_cols[0].selectbox(
            "execution.mode",
            mode_options,
            index=mode_options.index(current_mode),
            help="Режим исполнения: order — поведение по умолчанию, bar — сводное исполнение по барам.",
        )
        bar_price_input = exec_cols[1].text_input(
            "execution.bar_price",
            value=bar_price_default,
            help="Референсная цена для режима bar (например, close). Пусто = дефолт.",
        )
        min_step_input = exec_cols[0].text_input(
            "execution.min_rebalance_step",
            value=min_step_default,
            help="Минимальный шаг ребалансировки (доля позиции). Пусто = без ограничения.",
        )
        safety_input = exec_cols[1].text_input(
            "execution.safety_margin_bps",
            value=safety_default,
            help="Запас по безопасности в б.п. для бар-режима. Пусто = 0.",
        )
        equity_input = st.text_input(
            "portfolio.equity_usd",
            value=equity_default,
            help="Оценка капитала в USD. Пусто = использовать значение из базового конфига.",
        )
        cost_cols = st.columns(2)
        taker_input = cost_cols[0].text_input(
            "costs.taker_fee_bps",
            value=taker_default,
            help="Тейкерская комиссия в б.п. Пусто = оставить без изменений.",
        )
        half_spread_input = cost_cols[1].text_input(
            "costs.half_spread_bps",
            value=half_default,
            help="Половина спреда в б.п. Пусто = оставить без изменений.",
        )
        impact_cols = st.columns(2)
        impact_sqrt_input = impact_cols[0].text_input(
            "costs.impact.sqrt_coeff",
            value=impact_sqrt_default,
            help="Коэффициент квадратного влияния (б.п.).",
        )
        impact_linear_input = impact_cols[1].text_input(
            "costs.impact.linear_coeff",
            value=impact_linear_default,
            help="Линейный коэффициент влияния (б.п.).",
        )
        submit_runtime = st.form_submit_button(
            "Сохранить execution/cost/portfolio", type="secondary"
        )

    if submit_runtime:
        errors: List[str] = []
        min_step_value = _parse_non_negative_float(
            min_step_input, "execution.min_rebalance_step", errors
        )
        safety_value = _parse_non_negative_float(
            safety_input, "execution.safety_margin_bps", errors
        )
        equity_value = _parse_non_negative_float(
            equity_input, "portfolio.equity_usd", errors
        )
        taker_value = _parse_non_negative_float(
            taker_input, "costs.taker_fee_bps", errors
        )
        half_value = _parse_non_negative_float(
            half_spread_input, "costs.half_spread_bps", errors
        )
        impact_sqrt_value = _parse_non_negative_float(
            impact_sqrt_input, "costs.impact.sqrt_coeff", errors
        )
        impact_linear_value = _parse_non_negative_float(
            impact_linear_input, "costs.impact.linear_coeff", errors
        )

        if errors:
            for msg in errors:
                st.error(msg)
        else:
            new_runtime = copy.deepcopy(runtime_data)
            new_exec = copy.deepcopy(runtime_exec_cfg)
            new_exec["mode"] = str(mode_choice).strip().lower()
            bar_price_value = bar_price_input.strip()
            if bar_price_value:
                new_exec["bar_price"] = bar_price_value
            else:
                new_exec.pop("bar_price", None)

            if min_step_value is None:
                new_exec.pop("min_rebalance_step", None)
            else:
                new_exec["min_rebalance_step"] = float(min_step_value)

            if safety_value is None:
                new_exec.pop("safety_margin_bps", None)
            else:
                new_exec["safety_margin_bps"] = float(safety_value)

            new_portfolio = copy.deepcopy(runtime_portfolio_cfg)
            exec_portfolio = copy.deepcopy(
                runtime_exec_cfg.get("portfolio") or runtime_portfolio_cfg or {}
            )
            if equity_value is None:
                new_portfolio.pop("equity_usd", None)
                exec_portfolio.pop("equity_usd", None)
            else:
                new_portfolio["equity_usd"] = float(equity_value)
                exec_portfolio["equity_usd"] = float(equity_value)
            if exec_portfolio:
                new_exec["portfolio"] = exec_portfolio
            else:
                new_exec.pop("portfolio", None)
            if new_portfolio:
                new_runtime["portfolio"] = new_portfolio
            else:
                new_runtime.pop("portfolio", None)

            new_costs = copy.deepcopy(runtime_costs_cfg)
            exec_costs = copy.deepcopy(
                runtime_exec_cfg.get("costs") or runtime_costs_cfg or {}
            )
            if taker_value is None:
                new_costs.pop("taker_fee_bps", None)
                exec_costs.pop("taker_fee_bps", None)
            else:
                new_costs["taker_fee_bps"] = float(taker_value)
                exec_costs["taker_fee_bps"] = float(taker_value)
            if half_value is None:
                new_costs.pop("half_spread_bps", None)
                exec_costs.pop("half_spread_bps", None)
            else:
                new_costs["half_spread_bps"] = float(half_value)
                exec_costs["half_spread_bps"] = float(half_value)

            current_impact = copy.deepcopy(new_costs.get("impact") or {})
            exec_impact = copy.deepcopy(exec_costs.get("impact") or {})
            if impact_sqrt_value is None:
                current_impact.pop("sqrt_coeff", None)
                exec_impact.pop("sqrt_coeff", None)
            else:
                current_impact["sqrt_coeff"] = float(impact_sqrt_value)
                exec_impact["sqrt_coeff"] = float(impact_sqrt_value)
            if impact_linear_value is None:
                current_impact.pop("linear_coeff", None)
                exec_impact.pop("linear_coeff", None)
            else:
                current_impact["linear_coeff"] = float(impact_linear_value)
                exec_impact["linear_coeff"] = float(impact_linear_value)
            if current_impact:
                new_costs["impact"] = current_impact
            else:
                new_costs.pop("impact", None)
            if exec_impact:
                exec_costs["impact"] = exec_impact
            else:
                exec_costs.pop("impact", None)
            if exec_costs:
                new_exec["costs"] = exec_costs
            else:
                new_exec.pop("costs", None)
            if new_costs:
                new_runtime["costs"] = new_costs
            else:
                new_runtime.pop("costs", None)

            new_runtime["execution"] = new_exec
            new_runtime_text = _dump_yaml(new_runtime)
            diff = _show_diff(runtime_text, new_runtime_text, runtime_yaml)
            if diff.strip():
                atomic_write_with_retry(runtime_yaml, new_runtime_text)
                st.success("Настройки execution/cost/portfolio сохранены")
                st.info("Запросите reload для применения изменений.")
            else:
                st.info("Нет изменений")

    st.divider()
    st.markdown("#### TTL")
    ttl_cfg = copy.deepcopy(live_data.get("ttl") or {})
    ttl_mode_options = ["relative", "absolute", "off"]
    current_mode = str(ttl_cfg.get("mode", "relative"))
    if current_mode not in ttl_mode_options:
        current_mode = "relative"
    with st.form("ttl_form"):
        ttl_enabled = st.checkbox("TTL включён", value=bool(ttl_cfg.get("enabled", False)))
        ttl_seconds = st.number_input(
            "TTL секунд", min_value=0, max_value=86_400, value=int(ttl_cfg.get("ttl_seconds", 60))
        )
        ttl_mode = st.selectbox("Режим", ttl_mode_options, index=ttl_mode_options.index(current_mode))
        guard_ms = st.number_input(
            "Guard, мс", min_value=0, max_value=900_000, value=int(ttl_cfg.get("guard_ms", 5_000))
        )
        failsafe_ms = st.number_input(
            "Абсолютный предел, мс", min_value=0, max_value=3_600_000, value=int(ttl_cfg.get("absolute_failsafe_ms", 900_000))
        )
        state_path_val = st.text_input(
            "Файл состояния", value=str(ttl_cfg.get("state_path", "state/ttl_state.json"))
        )
        out_csv_val = st.text_input(
            "CSV для сигналов", value=str(ttl_cfg.get("out_csv") or "")
        )
        dedup_persist_val = st.text_input(
            "Файл кеша дедупликации", value=str(ttl_cfg.get("dedup_persist") or "")
        )
        if st.form_submit_button("Сохранить TTL"):
            new_live = copy.deepcopy(live_data)
            new_ttl = dict(ttl_cfg)
            new_ttl.update(
                {
                    "enabled": bool(ttl_enabled),
                    "ttl_seconds": int(ttl_seconds),
                    "mode": ttl_mode,
                    "guard_ms": int(guard_ms),
                    "absolute_failsafe_ms": int(failsafe_ms),
                    "state_path": state_path_val.strip(),
                    "out_csv": out_csv_val.strip() or None,
                    "dedup_persist": dedup_persist_val.strip() or None,
                }
            )
            new_live["ttl"] = new_ttl
            new_live_text = _dump_yaml(new_live)
            diff = _show_diff(live_text, new_live_text, live_yaml)
            if diff.strip():
                atomic_write_with_retry(live_yaml, new_live_text)
                st.success("TTL настройки сохранены")
            else:
                st.info("Нет изменений")

    st.divider()
    st.markdown("#### Throttle")
    throttle_cfg = copy.deepcopy(runtime_data.get("throttle") or {})
    mode_options = ["drop", "queue"]
    current_mode = str(throttle_cfg.get("mode", "drop"))
    if current_mode not in mode_options:
        current_mode = "drop"
    global_cfg = throttle_cfg.get("global") or {}
    symbol_cfg = throttle_cfg.get("symbol") or {}
    queue_cfg = throttle_cfg.get("queue") or {}
    with st.form("throttle_form"):
        thr_enabled = st.checkbox("Throttle включён", value=bool(throttle_cfg.get("enabled", False)))
        thr_mode = st.selectbox("Режим", mode_options, index=mode_options.index(current_mode))
        global_rps = st.number_input(
            "Global RPS", min_value=0.0, value=float(global_cfg.get("rps", 0.0))
        )
        global_burst = st.number_input(
            "Global burst", min_value=0, value=int(global_cfg.get("burst", 0))
        )
        symbol_rps = st.number_input(
            "Symbol RPS", min_value=0.0, value=float(symbol_cfg.get("rps", 0.0))
        )
        symbol_burst = st.number_input(
            "Symbol burst", min_value=0, value=int(symbol_cfg.get("burst", 0))
        )
        queue_max = st.number_input(
            "Queue max items", min_value=0, value=int(queue_cfg.get("max_items", 0))
        )
        queue_ttl = st.number_input(
            "Queue TTL, мс", min_value=0, value=int(queue_cfg.get("ttl_ms", 0))
        )
        time_source_val = st.text_input(
            "Источник времени", value=str(throttle_cfg.get("time_source", "monotonic"))
        )
        if st.form_submit_button("Сохранить throttle"):
            new_runtime = copy.deepcopy(runtime_data)
            new_throttle = dict(throttle_cfg)
            new_throttle.update(
                {
                    "enabled": bool(thr_enabled),
                    "mode": thr_mode,
                    "time_source": time_source_val.strip() or "monotonic",
                    "global": {"rps": float(global_rps), "burst": int(global_burst)},
                    "symbol": {"rps": float(symbol_rps), "burst": int(symbol_burst)},
                    "queue": {"max_items": int(queue_max), "ttl_ms": int(queue_ttl)},
                }
            )
            new_runtime["throttle"] = new_throttle
            new_runtime_text = _dump_yaml(new_runtime)
            diff = _show_diff(runtime_text, new_runtime_text, runtime_yaml)
            if diff.strip():
                atomic_write_with_retry(runtime_yaml, new_runtime_text)
                st.success("Throttle настройки сохранены")
                st.info("Запросите reload для применения изменений.")
            else:
                st.info("Нет изменений")

    st.divider()
    st.markdown("#### WebSocket дедупликация")
    ws_cfg = copy.deepcopy(runtime_data.get("ws") or {})
    with st.form("ws_form"):
        ws_enabled = st.checkbox("WS дедуп включён", value=bool(ws_cfg.get("enabled", False)))
        persist_path_val = st.text_input(
            "Файл persist", value=str(ws_cfg.get("persist_path") or "")
        )
        log_skips_val = st.checkbox(
            "Логировать пропуски", value=bool(ws_cfg.get("log_skips", False))
        )
        if st.form_submit_button("Сохранить дедупликацию"):
            new_runtime = copy.deepcopy(runtime_data)
            new_ws = dict(ws_cfg)
            new_ws.update(
                {
                    "enabled": bool(ws_enabled),
                    "persist_path": persist_path_val.strip() or None,
                    "log_skips": bool(log_skips_val),
                }
            )
            new_runtime["ws"] = new_ws
            new_runtime_text = _dump_yaml(new_runtime)
            diff = _show_diff(runtime_text, new_runtime_text, runtime_yaml)
            if diff.strip():
                atomic_write_with_retry(runtime_yaml, new_runtime_text)
                st.success("Настройки дедупликации сохранены")
                st.info("Запросите reload для применения изменений.")
            else:
                st.info("Нет изменений")

    st.divider()
    action_cols = st.columns(2)
    reload_flag = os.path.join(logs_dir, "reload_request.json")
    safe_stop_flag = os.path.join(logs_dir, "safe_stop.request")
    with action_cols[0]:
        if st.button("Запросить reload", type="primary"):
            payload = {
                "requested_at_ms": int(time.time() * 1000),
                "paths": [runtime_yaml],
            }
            atomic_write_with_retry(
                reload_flag, json.dumps(payload, ensure_ascii=False)
            )
            st.success("reload_request.json обновлён")
    with action_cols[1]:
        if st.button("Safe stop", type="secondary"):
            payload = {"requested_at_ms": int(time.time() * 1000)}
            atomic_write_with_retry(
                safe_stop_flag, json.dumps(payload, ensure_ascii=False)
            )
            st.success("Safe stop запрос записан")

    st.divider()
    st.subheader("Последние сигналы")
    sig_df = read_csv(signals_csv, n=200)
    if not sig_df.empty:
        st.dataframe(sig_df, use_container_width=True)
    else:
        st.info("Сигналы пока не найдены.")

    st.divider()
    st.markdown("### Демо ServiceSignalRunner")
    if st.button("Запустить демо ServiceSignalRunner"):

        class DummyAdapter:
            def run_events(self, provider):
                from core_models import Bar

                for i in range(3):
                    bar = Bar(
                        symbol="BTCUSDT",
                        ts=i,
                        open=0.0,
                        high=0.0,
                        low=0.0,
                        close=100.0 + i,
                        volume_base=1.0,
                    )
                    provider.on_bar(bar)
                    yield {"ts_ms": i, "symbol": "BTCUSDT"}

        class DummyFP:
            def warmup(self):
                pass

            def update(self, bar):
                return {"ref_price": float(bar.close)}

        class DummyStrat:
            def on_features(self, feats):
                pass

            def decide(self, ctx):
                return []

        try:
            adapter = DummyAdapter()
            runner = ServiceSignalRunner(
                adapter, DummyFP(), DummyStrat(), None, RunnerConfig()
            )
            list(runner.run())
            st.success("ServiceSignalRunner выполнен (демо)")
        except Exception as e:
            st.error(str(e))

with tabs[7]:
    st.subheader("Очередь на исполнение (manual approve)")

    st.caption(
        "Берём последние сигналы из logs/signals.csv. Можно Approve/Reject, экспортировать подтверждённые."
    )
    max_rows = st.number_input(
        "Сколько последних строк показывать",
        min_value=10,
        max_value=5000,
        value=200,
        step=10,
    )
    sig_df = load_signals_full(signals_csv, max_rows=max_rows)

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Всего сигналов (отображено)", len(sig_df))
    with colB:
        approved_path = os.path.join(logs_dir, "signals_approved.csv")
        rejected_path = os.path.join(logs_dir, "signals_rejected.csv")
        st.caption(f"Approved: `{approved_path}`")
        st.caption(f"Rejected: `{rejected_path}`")
    with colC:
        st.caption(
            "Формат файла approved/rejected совпадает с logs/signals.csv + колонка uid"
        )

    if sig_df.empty:
        st.info("Сигналов пока нет.")
    else:
        # покажем таблицу
        st.dataframe(sig_df, use_container_width=True, height=400)

        st.divider()
        st.subheader("Approve / Reject (по одному сигналу)")
        uid = st.text_input(
            "UID сигнала для действия (см. колонку uid)",
            value=str(sig_df.iloc[-1]["uid"]),
        )
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Approve выбранный UID", type="primary"):
                row = sig_df[sig_df["uid"] == uid]
                if row.empty:
                    st.error("UID не найден в текущей выборке.")
                else:
                    r = row.iloc[-1].to_dict()
                    header = list(sig_df.columns)
                    append_row_csv(
                        approved_path, header, [r.get(c, "") for c in header]
                    )
                    st.success("Добавлено в approved.")
        with action_cols[1]:
            if st.button("Reject выбранный UID", type="secondary"):
                row = sig_df[sig_df["uid"] == uid]
                if row.empty:
                    st.error("UID не найден в текущей выборке.")
                else:
                    r = row.iloc[-1].to_dict()
                    header = list(sig_df.columns)
                    append_row_csv(
                        rejected_path, header, [r.get(c, "") for c in header]
                    )
                    st.success("Добавлено в rejected.")

        st.divider()
        st.subheader("Экспорт подтверждённых сигналов")
        try:
            ap_df = read_csv(approved_path, n=10_000)
            if not ap_df.empty:
                csv_bytes = ap_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Скачать approved CSV",
                    data=csv_bytes,
                    file_name="signals_approved.csv",
                    mime="text/csv",
                )
            else:
                st.info("Файл approved пуст.")
        except Exception as e:
            st.error(str(e))

# --------------------------- Tab: Логи ---------------------------

with tabs[8]:
    st.subheader("Логи процессов (последние 200 строк)")
    log_names = {
        "ingest.log": os.path.join(logs_dir, "ingest.log"),
        "features.log": os.path.join(logs_dir, "features.log"),
        "train_table.log": os.path.join(logs_dir, "train_table.log"),
        "sandbox.log": os.path.join(logs_dir, "sandbox.log"),
        "evaluate.log": os.path.join(logs_dir, "evaluate.log"),
        "realtime.log": realtime_log,
    }
    for name, path in log_names.items():
        with st.expander(name, expanded=False):
            content = tail_file(path, n=200)
            st.code(content if content else "(пусто)")

with tabs[9]:
    st.subheader(
        "Полный прогон (ingest → features → train table → backtest → evaluate)"
    )
    st.caption(
        "Один клик запускает весь конвейер. Параметры ниже можно откорректировать."
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Ingest / Features**")
        prices_in = st.text_input(
            "prices (вход, после orchestrator)",
            value="data/prices.parquet",
            key="full_prices_in",
        )
        features_out = st.text_input(
            "features (выход make_features.py)",
            value="data/features.parquet",
            key="full_features_out",
        )
        lookbacks = st.text_input(
            "Окна SMA/ret (через запятую)", value="5,15,60", key="full_lookbacks"
        )
        rsi_period = st.number_input(
            "RSI period", min_value=2, max_value=200, value=14, step=1, key="full_rsi"
        )

    with colB:
        st.markdown("**Training Table**")
        bt_base = st.text_input(
            "features base (--base)", value="data/features.parquet", key="full_bt_base"
        )
        bt_prices = st.text_input(
            "prices (--prices)", value="data/prices.parquet", key="full_bt_prices"
        )
        bt_price_col = st.text_input(
            "price col (--price-col)", value="price", key="full_bt_price_col"
        )
        bt_decision_delay = st.number_input(
            "decision_delay_ms",
            min_value=0,
            value=500,
            step=100,
            key="full_decision_delay",
        )
        bt_horizon = st.number_input(
            "label_horizon_ms",
            min_value=60_000,
            value=3_600_000,
            step=60_000,
            key="full_horizon",
        )
        bt_out = st.text_input(
            "Выход train.parquet (--out)", value="data/train.parquet", key="full_bt_out"
        )

    st.markdown("**Evaluate**")
    out_md = st.text_input(
        "Выход markdown", value=os.path.join(logs_dir, "metrics.md"), key="full_out_md"
    )

    st.markdown("**Realtime (опционально)**")
    start_rt = st.checkbox(
        "Запустить realtime сигналер после завершения", value=False, key="full_start_rt"
    )

    if st.button("Запустить полный прогон", type="primary", key="full_build_btn"):
        build_all_pipeline(
            py=sys.executable,
            cfg_ingest=cfg_ingest,
            prices_in=prices_in,
            features_out=features_out,
            lookbacks=lookbacks,
            rsi_period=int(rsi_period),
            bt_base=bt_base,
            bt_prices=bt_prices,
            bt_price_col=bt_price_col,
            bt_decision_delay=int(bt_decision_delay),
            bt_horizon=int(bt_horizon),
            bt_out=bt_out,
            cfg_sandbox=cfg_sandbox,
            trades_path=trades_path,
            reports_path=reports_path,
            metrics_json=metrics_json,
            out_md=out_md,
            equity_png=equity_png,
            cfg_realtime=cfg_realtime,
            start_realtime=bool(start_rt),
            realtime_pid=realtime_pid,
            realtime_log=realtime_log,
            logs_dir=logs_dir,
        )

with tabs[10]:
    st.subheader("Model Train — обучение модели и выбор артефакта")

    st.caption(
        "Запуск обучения осуществляется через единый entrypoint `train_model_multi_patch.py`. После обучения можно записать путь к модели в configs/realtime.yaml."
    )

    st.markdown("### Запуск train_model_multi_patch")
    st.caption(
        "Введи точную команду запуска. Пример: "
        "`python train_model_multi_patch.py --config configs/config_train.yaml` "
        "или любая другая подходящая команда."
    )
    custom_cmd = st.text_input(
        "Команда запуска обучения",
        value="python train_model_multi_patch.py --config configs/config_train.yaml",
        key="mt_custom_cmd",
    )
    custom_log = os.path.join(logs_dir, "train_custom.log")
    st.caption(f"Лог обучения: `{custom_log}`")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Запустить тренер", type="primary", key="mt_run_custom"):
            # Разобьём команду по пробелам простым способом; при необходимости пользователь может обрамлять пути кавычками
            try:
                import shlex

                cmd_list = shlex.split(custom_cmd)
                rc = run_cmd(cmd_list, log_path=custom_log)
                if rc == 0:
                    st.success("Обучение завершено (кастомная команда)")
                else:
                    st.error(f"Команда завершилась с кодом {rc}")
            except Exception as e:
                st.error(str(e))
    with colB:
        if st.button("Показать лог обучения", key="mt_show_custom_log"):
            st.code(tail_file(custom_log, n=500) or "(пусто)")

    st.divider()
    st.markdown("### Указать артефакт модели и записать в configs/realtime.yaml")
    model_art = st.text_input(
        "Путь к готовому файлу модели (.pkl или др.)",
        value="artifacts/model.pkl",
        key="mt_art_path_a",
    )

    set_cols = st.columns(2)
    with set_cols[0]:
        if st.button(
            "Записать model_path в configs/realtime.yaml",
            type="primary",
            key="mt_set_model_a",
        ):
            try:
                rt_cfg = load_config(cfg_realtime).model_dump()
                new_cfg = copy.deepcopy(rt_cfg)
                new_cfg.setdefault("strategy", {}).setdefault("params", {})
                new_cfg["strategy"]["params"]["model_path"] = str(model_art)
                _ensure_dir(cfg_realtime)
                with open(cfg_realtime, "w", encoding="utf-8") as wf:
                    yaml.safe_dump(new_cfg, wf, sort_keys=False, allow_unicode=True)
                st.success("Путь к модели записан в configs/realtime.yaml")
            except Exception as e:
                st.error(f"Ошибка записи в configs/realtime.yaml: {e}")

    with set_cols[1]:
        if st.button("Показать текущий configs/realtime.yaml", key="mt_show_rt_yaml_a"):
            try:
                with open(cfg_realtime, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="yaml")
            except Exception as e:
                st.error(str(e))

with tabs[11]:
    st.subheader("YAML-редактор конфигов проекта")

    st.caption(
        "Редактируйте и сохраняйте конфиги: ingest.yaml, sandbox.yaml, sim.yaml, realtime.yaml. Есть проверка синтаксиса YAML и базовая валидация ключей."
    )

    files = {
        "configs/config_live.yaml": "configs/config_live.yaml",
        "configs/runtime.yaml": "configs/runtime.yaml",
        "configs/ops.yaml": "configs/ops.yaml",
        "configs/signals.yaml": "configs/signals.yaml",
        "configs/ingest.yaml": cfg_ingest,
        "configs/sandbox.yaml": cfg_sandbox,
        "configs/sim.yaml": cfg_sim,
        "configs/realtime.yaml": cfg_realtime,
    }

    choice = st.selectbox(
        "Выберите файл для редактирования",
        options=list(files.keys()),
        index=0,
        key="yaml_editor_choice",
    )
    path = os.path.expanduser(str(files.get(choice, choice) or "").strip())
    st.text_input(
        "Полный путь к выбранному файлу",
        value=path,
        key="yaml_editor_path",
        disabled=True,
    )

    def _read_yaml_file(target: str) -> str:
        if not target:
            return ""
        if not os.path.exists(target):
            return "# файл не существует — будет создан при сохранении\n"
        try:
            with open(target, "r", encoding="utf-8") as fh:
                return fh.read()
        except Exception as exc:
            return f"# ошибка чтения файла: {exc}\n"

    initial_text = _read_yaml_file(path)
    state_key = f"yaml_editor_content::{choice}"
    if state_key not in st.session_state:
        st.session_state[state_key] = initial_text
    content = st.text_area(
        "Содержимое YAML",
        value=st.session_state[state_key],
        height=500,
        key=state_key,
    )

    st.caption("Diff с содержимым на диске")
    _show_diff(initial_text, content, os.path.basename(path) or choice)

    action_cols = st.columns(3)
    validate_clicked = action_cols[0].button(
        "Validate", type="primary", key=f"yaml_validate::{choice}"
    )
    save_clicked = action_cols[1].button(
        "Save", type="secondary", key=f"yaml_save::{choice}"
    )
    reload_clicked = action_cols[2].button(
        "Reload", key=f"yaml_reload::{choice}"
    )

    if validate_clicked:
        fname = os.path.basename(path).lower()
        try:
            if not content.strip():
                yaml.safe_load("{}")
            elif fname == "ingest.yaml":
                parse_ingest_config(content)
            elif fname == "sandbox.yaml":
                parse_sandbox_config(content)
            elif fname in {"sim.yaml", "realtime.yaml", "config_live.yaml"}:
                load_config_from_str(content)
            else:
                yaml.safe_load(content)
        except Exception as exc:
            st.error(f"Ошибка проверки: {exc}")
        else:
            st.success("YAML синтаксически корректен")

    if save_clicked:
        try:
            _ensure_dir(path)
            atomic_write_with_retry(path, content)
        except Exception as exc:
            st.error(f"Не удалось сохранить: {exc}")
        else:
            st.session_state[state_key] = content
            st.success(f"Сохранено: {path}")

    if reload_clicked:
        st.session_state[state_key] = _read_yaml_file(path)
        st.session_state[f"yaml_reload_notice::{choice}"] = True
        st.experimental_rerun()

    notice_key = f"yaml_reload_notice::{choice}"
    if st.session_state.pop(notice_key, False):
        st.success(f"Перечитано с диска: {path}")

    st.divider()
    st.subheader("Подсказки по ключам (необязательные)")
    with st.expander("ingest.yaml — ожидаемые ключи", expanded=False):
        st.code(
            """symbols: ["BTCUSDT", "ETHUSDT"]
market: "futures"         # "spot" или "futures"
intervals: ["1m"]
aggregate_to: ["5m", "15m", "1h"]
period:
  start: "2024-01-01"
  end: "2024-12-31"
paths:
  klines_dir: "data/klines"
  futures_dir: "data/futures"
  prices_out: "data/prices.parquet"
futures:
  mark_interval: "1m"
slowness:
  api_limit: 1500
  sleep_ms: 350
""",
            language="yaml",
        )

    with st.expander("sandbox.yaml — ожидаемые ключи", expanded=False):
        st.code(
            """mode: "backtest"
symbol: "BTCUSDT"
latency_steps: 0
sim_config_path: "configs/sim.yaml"
strategy:
  module: "strategies.momentum"
  class: "MomentumStrategy"
  params:
    lookback: 5
    threshold: 0.0
    order_qty: 0.1  # доля позиции
data:
  path: "data/train.parquet"
  ts_col: "ts_ms"
  symbol_col: "symbol"
  price_col: "ref_price"
dynamic_spread:
  enabled: true
  base_bps: 3.0
  alpha_vol: 0.5
  beta_illiquidity: 1.0
  vol_mode: "hl"
  liq_col: "number_of_trades"
  liq_ref: 1000.0
  min_bps: 1.0
  max_bps: 25.0
out_reports: "logs/sandbox_reports.csv"
""",
            language="yaml",
        )

    with st.expander("sim.yaml — примерные блоки", expanded=False):
        st.code(
            """fees:
  maker_bps: 1.0
  taker_bps: 5.0
slippage:
  k: 0.8
  default_spread_bps: 3.0
  min_half_spread_bps: 0.0
pnl:
  mark_to: "side"
leakguard:
  decision_delay_ms: 500
risk:
  max_order_notional: 200.0
  max_abs_position_notional: 1000.0
  max_orders_per_min: 10
""",
            language="yaml",
        )

    with st.expander("realtime.yaml — ожидаемые ключи", expanded=False):
        st.code(
            """market: "futures"
symbols: ["BTCUSDT"]
interval: "1m"
strategy:
  module: "strategies.momentum"
  class: "MomentumStrategy"
  params:
    lookback: 5
    threshold: 0.0
    order_qty: 0.1  # доля позиции
    model_path: "artifacts/model.pkl"
features:
  lookbacks_prices: [5, 15, 60]
  rsi_period: 14
out_csv: "logs/signals.csv"
min_signal_gap_s: 300
backfill_on_gap: true
""",
            language="yaml",
        )


with tabs[12]:
    st.subheader("Sim Settings — симуляция, спред и комиссии")
    st.caption(
        "Редактируйте intrabar-логику, клиппинг, ограничения по ADV и параметры слиппеджа/комиссий. "
        "Изменения применяются сразу после сохранения формы."
    )

    cfg_cols = st.columns(3)
    with cfg_cols[0]:
        sim_yaml_path = st.text_input(
            "config_sim.yaml",
            value=cfg_sim,
            key="sim_settings_cfg_sim",
            help="Основной конфиг симулятора"
        )
    with cfg_cols[1]:
        slippage_yaml_path = st.text_input(
            "slippage.yaml",
            value="configs/slippage.yaml",
            key="sim_settings_slippage_yaml",
            help="Файл с параметрами слиппеджа"
        )
    with cfg_cols[2]:
        fees_yaml_path = st.text_input(
            "fees.yaml",
            value="configs/fees.yaml",
            key="sim_settings_fees_yaml",
            help="Файл с параметрами комиссий"
        )

    sim_data, _ = _load_yaml_file(sim_yaml_path)
    slip_data, _ = _load_yaml_file(slippage_yaml_path)
    fees_data, _ = _load_yaml_file(fees_yaml_path)

    if sim_yaml_path and not os.path.exists(sim_yaml_path):
        st.warning(
            f"Файл {sim_yaml_path} не найден — будет создан при сохранении.",
        )
    if slippage_yaml_path and not os.path.exists(slippage_yaml_path):
        st.info(
            f"Файл {slippage_yaml_path} пока не существует; значения будут записаны при сохранении.",
        )
    if fees_yaml_path and not os.path.exists(fees_yaml_path):
        st.info(
            f"Файл {fees_yaml_path} пока не существует; значения будут записаны при сохранении.",
        )

    def _save_yaml(path: str, payload: Dict[str, Any], label: str) -> bool:
        if not path:
            st.error(f"Не задан путь для {label}")
            return False
        try:
            atomic_write_with_retry(path, _dump_yaml(payload))
        except Exception as exc:
            st.error(f"Не удалось сохранить {label}: {exc}")
            return False
        st.success(f"Сохранено: {label}")
        return True

    def _parse_optional_float(text_value: str, field_name: str, errors: list[str], *, min_value: float | None = None, max_value: float | None = None) -> float | None:
        text = str(text_value).strip()
        if text == "":
            return None
        try:
            value = float(text)
        except ValueError:
            errors.append(f"{field_name}: не удалось преобразовать '{text_value}' к числу")
            return None
        if min_value is not None and value < min_value:
            errors.append(f"{field_name}: значение {value} < {min_value}")
        if max_value is not None and value > max_value:
            errors.append(f"{field_name}: значение {value} > {max_value}")
        return value

    def _parse_optional_int(text_value: str, field_name: str, errors: list[str], *, min_value: int | None = None) -> int | None:
        text = str(text_value).strip()
        if text == "":
            return None
        try:
            value = int(float(text))
        except ValueError:
            errors.append(f"{field_name}: не удалось преобразовать '{text_value}' к целому")
            return None
        if min_value is not None and value < min_value:
            errors.append(f"{field_name}: значение {value} < {min_value}")
        return value

    st.markdown("### Быстрый бэктест (bar-level отчёт)")
    tail_rows = int(
        st.number_input(
            "Сколько последних баров показать",
            min_value=5,
            max_value=200,
            value=20,
            step=5,
            key="sim_short_bt_tail",
        )
    )
    if st.button("Run short backtest", key="sim_run_short_backtest", type="primary"):
        if not cfg_sandbox or not os.path.exists(cfg_sandbox):
            st.error(f"Sandbox YAML не найден: {cfg_sandbox}")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                reports_tmp = os.path.join(tmpdir, "reports.csv")
                bar_report_tmp = os.path.join(tmpdir, "bar_report.csv")
                try:
                    run_backtest_from_yaml(
                        cfg_sandbox,
                        reports_tmp,
                        logs_dir,
                        bar_report_path=bar_report_tmp,
                    )
                except Exception as exc:
                    st.error(f"Ошибка короткого бэктеста: {exc}")
                else:
                    summary_path = os.path.splitext(bar_report_tmp)[0]
                    summary_path = f"{summary_path}_summary.csv" if summary_path else f"{bar_report_tmp}_summary.csv"
                    try:
                        summary_df = pd.read_csv(summary_path)
                    except FileNotFoundError:
                        summary_df = pd.DataFrame()
                    except Exception as exc:
                        st.warning(f"Не удалось прочитать summary: {exc}")
                        summary_df = pd.DataFrame()
                    if not summary_df.empty:
                        st.subheader("Bar summary (spread/impact/fees bps)")
                        st.dataframe(summary_df)
                        st.download_button(
                            "Скачать summary CSV",
                            summary_df.to_csv(index=False).encode("utf-8"),
                            "bar_summary.csv",
                            "text/csv",
                        )
                    else:
                        st.info("Summary пуст или не создан (нет баров).")

                    try:
                        bar_df = pd.read_csv(bar_report_tmp)
                    except FileNotFoundError:
                        bar_df = pd.DataFrame()
                    except Exception as exc:
                        st.warning(f"Не удалось прочитать bar-level отчёт: {exc}")
                        bar_df = pd.DataFrame()
                    if not bar_df.empty:
                        st.subheader("Bar report — последние строки")
                        st.dataframe(bar_df.tail(tail_rows))
                        st.download_button(
                            "Скачать bar report CSV",
                            bar_df.to_csv(index=False).encode("utf-8"),
                            "bar_report.csv",
                            "text/csv",
                        )
                    else:
                        st.info("Bar-level отчёт пуст.")

    st.divider()

    exec_cfg = copy.deepcopy(sim_data.get("execution") or {})
    clip_cfg = copy.deepcopy(exec_cfg.get("clip_to_bar") or {})
    bar_cap_cfg = copy.deepcopy(exec_cfg.get("bar_capacity_base") or {})

    def _normalise_intrabar(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        if text in {"", "default", "none"}:
            return ""
        if text in {"bridge", "brownian", "brownian_bridge"}:
            return "bridge"
        if text in {"linear", "open_close_linear", "open-close-linear", "oc_linear", "linear_oc"}:
            return "linear"
        if text in {"ohlc", "ohlc-linear", "ohlc_linear"}:
            return "ohlc"
        return text

    intrabar_options = [
        ("Отключено / legacy", ""),
        ("Mid (midpoint)", "mid"),
        ("Linear (open→close)", "linear"),
        ("OHLC экстремумы", "ohlc"),
        ("Bridge (Brownian)", "bridge"),
        ("Open", "open"),
        ("Close", "close"),
        ("High", "high"),
        ("Low", "low"),
    ]
    intrabar_map = {label: value for label, value in intrabar_options}
    current_intrabar = _normalise_intrabar(exec_cfg.get("intrabar_price_model"))
    intrabar_index = 0
    for idx, (_, val) in enumerate(intrabar_options):
        if val == current_intrabar:
            intrabar_index = idx
            break

    with st.expander("Execution: intrabar, clip и bar capacity", expanded=True):
        with st.form("sim_execution_form"):
            mode_choice = st.selectbox(
                "Модель intrabar",
                [label for label, _ in intrabar_options],
                index=intrabar_index,
                help="Определяет, как симулятор выбирает цену внутри бара",
            )
            use_latency_from = st.text_input(
                "use_latency_from (опционально)",
                value=str(exec_cfg.get("use_latency_from") or ""),
            )
            latency_value = st.number_input(
                "latency_constant_ms (опционально)",
                min_value=0.0,
                value=float(exec_cfg.get("latency_constant_ms") or 0.0),
                step=1.0,
            )
            latency_enabled = st.checkbox(
                "Применять latency_constant_ms",
                value=exec_cfg.get("latency_constant_ms") is not None,
            )

            clip_cols = st.columns(2)
            with clip_cols[0]:
                clip_enabled = st.checkbox(
                    "Включить clip_to_bar",
                    value=bool(clip_cfg.get("enabled", True)),
                )
            with clip_cols[1]:
                strict_open = st.checkbox(
                    "strict_open_fill",
                    value=bool(clip_cfg.get("strict_open_fill", False)),
                )

            cap_cols = st.columns(3)
            with cap_cols[0]:
                bar_cap_enabled = st.checkbox(
                    "Bar capacity (ADV)",
                    value=bool(bar_cap_cfg.get("enabled", False)),
                )
                capacity_frac = st.number_input(
                    "Доля ADV на бар",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(
                        min(
                            max(bar_cap_cfg.get("capacity_frac_of_ADV_base", 1.0), 0.0),
                            1.0,
                        )
                    ),
                )
            with cap_cols[1]:
                adv_path = st.text_input(
                    "adv_base_path",
                    value=str(bar_cap_cfg.get("adv_base_path") or ""),
                )
                floor_text = st.text_input(
                    "floor_base (опц.)",
                    value="" if bar_cap_cfg.get("floor_base") in (None, "") else str(bar_cap_cfg.get("floor_base")),
                )
            with cap_cols[2]:
                timeframe_text = st.text_input(
                    "timeframe_ms override (опц.)",
                    value="" if bar_cap_cfg.get("timeframe_ms") in (None, "") else str(bar_cap_cfg.get("timeframe_ms")),
                )

            submitted_exec = st.form_submit_button("Сохранить execution")

        if submitted_exec:
            errors: list[str] = []
            warnings: list[str] = []
            floor_value = _parse_optional_float(floor_text, "floor_base", errors, min_value=0.0)
            timeframe_value = _parse_optional_int(timeframe_text, "timeframe_ms", errors, min_value=0)
            if errors:
                for msg in errors:
                    st.error(msg)
            else:
                new_sim = copy.deepcopy(sim_data)
                exec_block = copy.deepcopy(new_sim.get("execution") or {})
                mode_value = intrabar_map.get(mode_choice, "")
                if mode_value:
                    exec_block["intrabar_price_model"] = mode_value
                else:
                    exec_block.pop("intrabar_price_model", None)

                latency_source = use_latency_from.strip()
                if latency_source:
                    exec_block["use_latency_from"] = latency_source
                else:
                    exec_block.pop("use_latency_from", None)
                if latency_enabled:
                    exec_block["latency_constant_ms"] = float(latency_value)
                else:
                    exec_block.pop("latency_constant_ms", None)

                clip_block = copy.deepcopy(exec_block.get("clip_to_bar") or {})
                clip_block["enabled"] = bool(clip_enabled)
                clip_block["strict_open_fill"] = bool(strict_open)
                exec_block["clip_to_bar"] = clip_block

                cap_block = copy.deepcopy(exec_block.get("bar_capacity_base") or {})
                cap_block["enabled"] = bool(bar_cap_enabled)
                cap_block["capacity_frac_of_ADV_base"] = float(capacity_frac)
                if adv_path.strip():
                    cap_block["adv_base_path"] = adv_path.strip()
                else:
                    cap_block.pop("adv_base_path", None)
                if floor_value is None:
                    cap_block.pop("floor_base", None)
                else:
                    cap_block["floor_base"] = float(floor_value)
                if timeframe_value is None:
                    cap_block.pop("timeframe_ms", None)
                else:
                    cap_block["timeframe_ms"] = int(timeframe_value)
                exec_block["bar_capacity_base"] = cap_block
                new_sim["execution"] = exec_block

                if _save_yaml(sim_yaml_path, new_sim, "config_sim.yaml"):
                    sim_data = new_sim
                    if bar_cap_enabled and not adv_path.strip():
                        warnings.append("Bar capacity включён, но путь к ADV пуст — потребуется заполнить adv_base_path.")
                    if not clip_enabled:
                        warnings.append("Clip_to_bar отключён — возможны цены вне диапазона бара.")
                    for msg in warnings:
                        st.warning(msg)

    st.divider()

    slip_section = copy.deepcopy(slip_data.get("slippage") or {})
    sim_slip_section = copy.deepcopy(sim_data.get("slippage") or {})
    base_k = float(slip_section.get("k", sim_slip_section.get("k", 0.8)))
    base_default_spread = float(slip_section.get("default_spread_bps", sim_slip_section.get("default_spread_bps", 2.0)))
    base_min_half = float(slip_section.get("min_half_spread_bps", sim_slip_section.get("min_half_spread_bps", 0.0)))
    dyn_section = copy.deepcopy(slip_section.get("dynamic") or {})

    with st.expander("Slippage: статический и динамический спред", expanded=True):
        with st.form("slippage_form"):
            slip_cols = st.columns(3)
            with slip_cols[0]:
                k_value = st.number_input(
                    "Impact k",
                    min_value=0.0,
                    value=float(base_k),
                    step=0.05,
                )
            with slip_cols[1]:
                default_spread_value = st.number_input(
                    "default_spread_bps",
                    min_value=0.0,
                    value=float(base_default_spread),
                    step=0.1,
                )
            with slip_cols[2]:
                min_half_value = st.number_input(
                    "min_half_spread_bps",
                    min_value=0.0,
                    value=float(base_min_half),
                    step=0.1,
                )

            dynamic_enabled = st.checkbox(
                "Включить динамический спред",
                value=bool(dyn_section.get("enabled", False)),
            )
            dyn_cols = st.columns(3)
            with dyn_cols[0]:
                alpha_bps_value = st.number_input(
                    "alpha_bps",
                    min_value=0.0,
                    value=float(dyn_section.get("alpha_bps", 0.0)),
                    step=0.1,
                )
            with dyn_cols[1]:
                beta_coef_value = st.number_input(
                    "beta_coef",
                    min_value=0.0,
                    value=float(dyn_section.get("beta_coef", 0.0)),
                    step=0.1,
                )
            with dyn_cols[2]:
                smoothing_text = st.text_input(
                    "smoothing_alpha (0..1, опц.)",
                    value="" if dyn_section.get("smoothing_alpha") in (None, "") else str(dyn_section.get("smoothing_alpha")),
                )

            dyn_cols2 = st.columns(3)
            with dyn_cols2[0]:
                min_spread_value = st.number_input(
                    "min_spread_bps",
                    min_value=0.0,
                    value=float(dyn_section.get("min_spread_bps", 0.0)),
                    step=0.1,
                )
            with dyn_cols2[1]:
                max_spread_value = st.number_input(
                    "max_spread_bps",
                    min_value=0.0,
                    value=float(dyn_section.get("max_spread_bps", 20.0)),
                    step=0.5,
                )
            with dyn_cols2[2]:
                fallback_text = st.text_input(
                    "fallback_spread_bps (опц.)",
                    value="" if dyn_section.get("fallback_spread_bps") in (None, "") else str(dyn_section.get("fallback_spread_bps")),
                )

            vol_metric_value = st.text_input(
                "vol_metric (опц.)",
                value=str(dyn_section.get("vol_metric") or ""),
            )
            vol_window_text = st.text_input(
                "vol_window (bars, опц.)",
                value="" if dyn_section.get("vol_window") in (None, "") else str(dyn_section.get("vol_window")),
            )

            submitted_slip = st.form_submit_button("Сохранить slippage")

        if submitted_slip:
            errors: list[str] = []
            warnings: list[str] = []
            smoothing_value = _parse_optional_float(smoothing_text, "smoothing_alpha", errors, min_value=0.0, max_value=1.0)
            fallback_value = _parse_optional_float(fallback_text, "fallback_spread_bps", errors, min_value=0.0)
            vol_window_value = _parse_optional_int(vol_window_text, "vol_window", errors, min_value=1)
            if max_spread_value < min_spread_value:
                errors.append("max_spread_bps должен быть ≥ min_spread_bps")
            if errors:
                for msg in errors:
                    st.error(msg)
            else:
                new_slip = copy.deepcopy(slip_section)
                new_slip["k"] = float(k_value)
                new_slip["default_spread_bps"] = float(default_spread_value)
                new_slip["min_half_spread_bps"] = float(min_half_value)

                new_dyn = copy.deepcopy(dyn_section)
                new_dyn["enabled"] = bool(dynamic_enabled)
                new_dyn["alpha_bps"] = float(alpha_bps_value)
                new_dyn["beta_coef"] = float(beta_coef_value)
                new_dyn["min_spread_bps"] = float(min_spread_value)
                new_dyn["max_spread_bps"] = float(max_spread_value)
                if smoothing_value is None:
                    new_dyn.pop("smoothing_alpha", None)
                else:
                    new_dyn["smoothing_alpha"] = float(smoothing_value)
                if fallback_value is None:
                    new_dyn.pop("fallback_spread_bps", None)
                else:
                    new_dyn["fallback_spread_bps"] = float(fallback_value)
                if vol_metric_value.strip():
                    new_dyn["vol_metric"] = vol_metric_value.strip()
                else:
                    new_dyn.pop("vol_metric", None)
                if vol_window_value is None:
                    new_dyn.pop("vol_window", None)
                else:
                    new_dyn["vol_window"] = int(vol_window_value)
                new_slip["dynamic"] = new_dyn

                sim_slip = copy.deepcopy(sim_slip_section)
                sim_slip.update({
                    "k": new_slip["k"],
                    "default_spread_bps": new_slip["default_spread_bps"],
                    "min_half_spread_bps": new_slip["min_half_spread_bps"],
                })
                sim_slip["dynamic"] = copy.deepcopy(new_dyn)

                slip_payload = copy.deepcopy(slip_data)
                slip_payload["slippage"] = new_slip
                sim_payload = copy.deepcopy(sim_data)
                sim_payload["slippage"] = sim_slip
                slip_saved = _save_yaml(slippage_yaml_path, slip_payload, "slippage.yaml")
                sim_saved = _save_yaml(sim_yaml_path, sim_payload, "config_sim.yaml (slippage)")
                if slip_saved and sim_saved:
                    slip_data = slip_payload
                    slip_section = new_slip
                    sim_data = sim_payload
                    warnings.append("Slippage обновлён. Проверьте, что динамический режим соответствует данным.")
                    if not dynamic_enabled and (
                        alpha_bps_value > 0 or beta_coef_value > 0 or vol_metric_value.strip()
                    ):
                        warnings.append(
                            "Динамический спред отключён, но заданы параметры > 0 — включите флаг 'Включить динамический спред'."
                        )
                    if dynamic_enabled and beta_coef_value > 0 and not vol_metric_value.strip():
                        warnings.append("beta_coef > 0, но vol_metric не задан — модель не будет использовать коэффициент волатильности.")
                    for msg in warnings:
                        st.warning(msg)

    st.divider()

    fees_section = copy.deepcopy(fees_data.get("fees") or {})
    sim_fees_section = copy.deepcopy(sim_data.get("fees") or {})
    maker_bps_value = float(fees_section.get("maker_bps", sim_fees_section.get("maker_bps", 1.0)))
    taker_bps_value = float(fees_section.get("taker_bps", sim_fees_section.get("taker_bps", 5.0)))
    maker_share_override = fees_section.get("maker_share_default", sim_fees_section.get("maker_share_default"))
    maker_share_override_text = "" if maker_share_override in (None, "") else str(maker_share_override)
    mt_section = copy.deepcopy(fees_section.get("maker_taker_share") or {})

    with st.expander("Fees: базовые ставки и maker/taker share", expanded=True):
        with st.form("fees_form"):
            fee_cols = st.columns(2)
            with fee_cols[0]:
                maker_bps_new = st.number_input(
                    "maker_bps",
                    min_value=0.0,
                    value=float(maker_bps_value),
                    step=0.1,
                )
            with fee_cols[1]:
                taker_bps_new = st.number_input(
                    "taker_bps",
                    min_value=0.0,
                    value=float(taker_bps_value),
                    step=0.1,
                )

            maker_share_override_input = st.text_input(
                "fees.maker_share_default (0..1, опц.)",
                value=maker_share_override_text,
            )

            mt_enabled = st.checkbox(
                "Включить maker/taker share",
                value=bool(mt_section.get("enabled", False)),
            )
            mode_options = ["fixed", "model", "predictor"]
            current_mode = str(mt_section.get("mode", "fixed")).lower()
            if current_mode not in mode_options:
                current_mode = "fixed"
            mt_mode = st.selectbox("Режим maker/taker share", mode_options, index=mode_options.index(current_mode))
            maker_share_default_value = st.number_input(
                "maker_share_default",
                min_value=0.0,
                max_value=1.0,
                value=float(mt_section.get("maker_share_default", 0.5)),
                step=0.05,
            )
            spread_cost_maker_value = st.number_input(
                "spread_cost_maker_bps",
                min_value=0.0,
                value=float(mt_section.get("spread_cost_maker_bps", 0.0)),
                step=0.1,
            )
            spread_cost_taker_value = st.number_input(
                "spread_cost_taker_bps",
                min_value=0.0,
                value=float(mt_section.get("spread_cost_taker_bps", 0.0)),
                step=0.1,
            )
            taker_override_text = st.text_input(
                "taker_fee_override_bps (опц.)",
                value="" if mt_section.get("taker_fee_override_bps") in (None, "") else str(mt_section.get("taker_fee_override_bps")),
            )
            coeffs = copy.deepcopy((mt_section.get("model") or {}).get("coefficients") or {})
            coeff_cols = st.columns(3)
            with coeff_cols[0]:
                intercept_text = st.text_input(
                    "intercept (опц.)",
                    value="" if coeffs.get("intercept") in (None, "") else str(coeffs.get("intercept")),
                )
            with coeff_cols[1]:
                dist_coef_text = st.text_input(
                    "distance_to_mid (опц.)",
                    value="" if coeffs.get("distance_to_mid") in (None, "") else str(coeffs.get("distance_to_mid")),
                )
            with coeff_cols[2]:
                latency_coef_text = st.text_input(
                    "latency (опц.)",
                    value="" if coeffs.get("latency") in (None, "") else str(coeffs.get("latency")),
                )

            submitted_fees = st.form_submit_button("Сохранить fees")

        if submitted_fees:
            errors: list[str] = []
            warnings: list[str] = []
            maker_share_override_value = _parse_optional_float(
                maker_share_override_input,
                "fees.maker_share_default",
                errors,
                min_value=0.0,
                max_value=1.0,
            )
            taker_override_value = _parse_optional_float(
                taker_override_text,
                "taker_fee_override_bps",
                errors,
                min_value=0.0,
            )
            intercept_value = _parse_optional_float(intercept_text, "intercept", errors)
            dist_coef_value = _parse_optional_float(dist_coef_text, "distance_to_mid", errors)
            latency_coef_value = _parse_optional_float(latency_coef_text, "latency", errors)
            if errors:
                for msg in errors:
                    st.error(msg)
            else:
                new_fees = copy.deepcopy(fees_section)
                new_fees["maker_bps"] = float(maker_bps_new)
                new_fees["taker_bps"] = float(taker_bps_new)
                if maker_share_override_value is None:
                    new_fees.pop("maker_share_default", None)
                else:
                    new_fees["maker_share_default"] = float(maker_share_override_value)

                new_mt = copy.deepcopy(mt_section)
                new_mt["enabled"] = bool(mt_enabled)
                new_mt["mode"] = mt_mode
                new_mt["maker_share_default"] = float(maker_share_default_value)
                new_mt["spread_cost_maker_bps"] = float(spread_cost_maker_value)
                new_mt["spread_cost_taker_bps"] = float(spread_cost_taker_value)
                if taker_override_value is None:
                    new_mt.pop("taker_fee_override_bps", None)
                else:
                    new_mt["taker_fee_override_bps"] = float(taker_override_value)
                if "model" not in new_mt or not isinstance(new_mt["model"], dict):
                    new_mt["model"] = {}
                model_block = copy.deepcopy(new_mt["model"])
                coeff_block = copy.deepcopy(model_block.get("coefficients") or {})
                if intercept_value is None:
                    coeff_block.pop("intercept", None)
                else:
                    coeff_block["intercept"] = float(intercept_value)
                if dist_coef_value is None:
                    coeff_block.pop("distance_to_mid", None)
                else:
                    coeff_block["distance_to_mid"] = float(dist_coef_value)
                if latency_coef_value is None:
                    coeff_block.pop("latency", None)
                else:
                    coeff_block["latency"] = float(latency_coef_value)
                model_block["coefficients"] = coeff_block
                new_mt["model"] = model_block
                new_fees["maker_taker_share"] = new_mt

                sim_fees = copy.deepcopy(sim_fees_section)
                sim_fees.update({
                    "maker_bps": new_fees["maker_bps"],
                    "taker_bps": new_fees["taker_bps"],
                })
                if maker_share_override_value is None:
                    sim_fees.pop("maker_share_default", None)
                else:
                    sim_fees["maker_share_default"] = float(maker_share_override_value)
                sim_fees["maker_taker_share"] = copy.deepcopy(new_mt)

                fees_payload = copy.deepcopy(fees_data)
                fees_payload["fees"] = new_fees
                sim_payload = copy.deepcopy(sim_data)
                sim_payload["fees"] = sim_fees
                fees_saved = _save_yaml(fees_yaml_path, fees_payload, "fees.yaml")
                sim_saved = _save_yaml(sim_yaml_path, sim_payload, "config_sim.yaml (fees)")
                if fees_saved and sim_saved:
                    fees_data = fees_payload
                    fees_section = new_fees
                    sim_data = sim_payload
                    if not mt_enabled and (
                        mt_mode != "fixed" or spread_cost_maker_value > 0 or spread_cost_taker_value > 0 or taker_override_value is not None
                    ):
                        warnings.append(
                            "Maker/taker share выключен, но заданы параметры — включите флаг, чтобы применить модель."
                        )
                    for msg in warnings:
                        st.warning(msg)

    st.divider()
    st.markdown("#### Калибровка слиппеджа и t-cost")

    calib_cols = st.columns(2)
    with calib_cols[0]:
        slippage_calib_cfg = st.text_input(
            "Slippage calibrate YAML",
            value="configs/slippage_calibrate.yaml",
            key="sim_slippage_calib_cfg",
        )
        apply_slip_update = st.checkbox(
            "Перезаписать slippage.yaml и config_sim",
            value=True,
            key="sim_apply_slip_update",
        )
        if st.button("Calibrate slippage", key="sim_calibrate_slippage"):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    report_path = os.path.join(tmpdir, "slippage_calibration.json")
                    report = calibrate_slippage_from_config(slippage_calib_cfg, out=report_path)
            except Exception as exc:
                st.error(f"Ошибка калибровки slippage: {exc}")
            else:
                st.success("Slippage калиброван")
                st.json(report)
                if apply_slip_update:
                    try:
                        slip_after, _ = _load_yaml_file(slippage_yaml_path)
                        sim_after, _ = _load_yaml_file(sim_yaml_path)
                        new_slip = copy.deepcopy(slip_after.get("slippage") or {})
                        for key in ["k", "default_spread_bps", "min_half_spread_bps"]:
                            if key in report:
                                new_slip[key] = float(report[key])
                        new_slip.setdefault("dynamic", copy.deepcopy(new_slip.get("dynamic") or {}))
                        slip_after["slippage"] = new_slip
                        sim_slip = copy.deepcopy(sim_after.get("slippage") or {})
                        for key in ["k", "default_spread_bps", "min_half_spread_bps"]:
                            if key in report:
                                sim_slip[key] = float(report[key])
                        sim_slip.setdefault("dynamic", copy.deepcopy(sim_slip.get("dynamic") or {}))
                        sim_after["slippage"] = sim_slip
                        saved_slip = _save_yaml(slippage_yaml_path, slip_after, "slippage.yaml")
                        saved_sim = _save_yaml(sim_yaml_path, sim_after, "config_sim.yaml")
                        if saved_slip and saved_sim:
                            slip_data = slip_after
                            sim_data = sim_after
                            st.success("Параметры slippage обновлены после калибровки")
                    except Exception as exc:
                        st.error(f"Не удалось обновить YAML после калибровки slippage: {exc}")

    with calib_cols[1]:
        tcost_target = st.selectbox(
            "T-cost target",
            options=["hl", "oc"],
            index=0,
            key="sim_tcost_target",
        )
        tcost_k = st.number_input(
            "T-cost k",
            min_value=0.0,
            value=0.25,
            step=0.05,
            key="sim_tcost_k",
        )
        tcost_update_sandbox = st.checkbox(
            "Обновить sandbox.yaml",
            value=False,
            key="sim_tcost_update_sandbox",
        )
        tcost_update_slippage = st.checkbox(
            "Обновить slippage.dynamic",
            value=True,
            key="sim_tcost_update_slip",
        )
        if st.button("Calibrate t-cost", key="sim_calibrate_tcost"):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    report_path = os.path.join(tmpdir, "tcost_calibration.json")
                    cfg = TCostCalibrateConfig(
                        sandbox_config=cfg_sandbox,
                        out=report_path,
                        target=str(tcost_target),
                        k=float(tcost_k),
                        dry_run=not tcost_update_sandbox,
                    )
                    report = calibrate_tcost_run(cfg)
            except Exception as exc:
                st.error(f"Ошибка калибровки t-cost: {exc}")
            else:
                st.success("T-cost калиброван")
                st.json(report)
                if tcost_update_slippage:
                    try:
                        fitted = report.get("fitted_params") or {}
                        alpha_bps = float(fitted.get("base_bps", 0.0))
                        beta_coef = float(fitted.get("alpha_vol", 0.0))
                        slip_after, _ = _load_yaml_file(slippage_yaml_path)
                        sim_after, _ = _load_yaml_file(sim_yaml_path)
                        slip_dyn = copy.deepcopy((slip_after.get("slippage") or {}).get("dynamic") or {})
                        slip_dyn["enabled"] = True
                        slip_dyn["alpha_bps"] = alpha_bps
                        slip_dyn["beta_coef"] = beta_coef
                        slip_block = copy.deepcopy(slip_after.get("slippage") or {})
                        slip_block["dynamic"] = slip_dyn
                        slip_after["slippage"] = slip_block

                        sim_slip_dyn = copy.deepcopy((sim_after.get("slippage") or {}).get("dynamic") or {})
                        sim_slip_dyn["enabled"] = True
                        sim_slip_dyn["alpha_bps"] = alpha_bps
                        sim_slip_dyn["beta_coef"] = beta_coef
                        sim_slip_block = copy.deepcopy(sim_after.get("slippage") or {})
                        sim_slip_block["dynamic"] = sim_slip_dyn
                        sim_after["slippage"] = sim_slip_block

                        saved_slip = _save_yaml(slippage_yaml_path, slip_after, "slippage.yaml")
                        saved_sim = _save_yaml(sim_yaml_path, sim_after, "config_sim.yaml")
                        if saved_slip and saved_sim:
                            slip_data = slip_after
                            sim_data = sim_after
                            st.success("Slippage.dynamic обновлён из t-cost калибровки")
                    except Exception as exc:
                        st.error(f"Не удалось обновить slippage после t-cost калибровки: {exc}")
with tabs[13]:
    st.subheader(
        "T-cost Calibrate — калибровка модели издержек (base_bps, alpha_vol, beta_illiquidity)"
    )

    st.caption(
        "Форма ниже читает ваш датасет (CSV/Parquet), строит прокси-спред по high/low (или |log return|), "
        "оценивает параметры линейной модели спреда и при желании записывает их в configs/sandbox.yaml → dynamic_spread."
    )

    import io
    import json
    import math
    from typing import Tuple

    import numpy as np
    import pandas as pd
    import yaml
    import os

    cfg_sandbox = "configs/sandbox.yaml"

    def _ensure_dir(path: str) -> None:
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)

    @st.cache_data(show_spinner=False)
    def _read_table(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        raise ValueError(f"Неизвестный формат файла: {ext}")

    def _winsorize(a: np.ndarray, p: float = 0.01) -> np.ndarray:
        if a.size == 0:
            return a
        lo = np.nanpercentile(a, p * 100.0)
        hi = np.nanpercentile(a, (1.0 - p) * 100.0)
        return np.clip(a, lo, hi)

    def _compute_features(
        df: pd.DataFrame,
        *,
        ts_col: str,
        symbol_col: str,
        price_col: str,
        vol_mode: str,
        liq_col: str,
        liq_ref: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = df.sort_values([symbol_col, ts_col]).reset_index(drop=True)

        if (
            vol_mode.lower() == "hl"
            and ("high" in df.columns)
            and ("low" in df.columns)
        ):
            hi = pd.to_numeric(df["high"], errors="coerce").astype(float)
            lo = pd.to_numeric(df["low"], errors="coerce").astype(float)
            ref = pd.to_numeric(df[price_col], errors="coerce").astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                vol_factor = np.maximum(0.0, (hi - lo) / np.where(ref > 0, ref, np.nan))
            y_bps = vol_factor * 10000.0
            v_bps = y_bps.copy()
        else:
            ref = pd.to_numeric(df[price_col], errors="coerce").astype(float)
            prev = ref.shift(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = np.abs(
                    np.log(np.where((ref > 0) & (prev > 0), ref / prev, np.nan))
                )
            vol_factor = ret
            v_bps = vol_factor * 10000.0
            y_bps = v_bps * 0.5

        if liq_col in df.columns:
            liq = pd.to_numeric(df[liq_col], errors="coerce").astype(float).to_numpy()
        elif "volume" in df.columns:
            liq = pd.to_numeric(df["volume"], errors="coerce").astype(float).to_numpy()
        else:
            liq = np.ones(len(df), dtype=float)

        if not liq_ref or liq_ref <= 0:
            finite = liq[np.isfinite(liq)]
            liq_ref = float(np.nanpercentile(finite, 75) if finite.size else 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            r_liq = np.maximum(0.0, (liq_ref - liq) / liq_ref)

        y_bps = y_bps.to_numpy() if isinstance(y_bps, pd.Series) else np.asarray(y_bps)
        v_bps = v_bps.to_numpy() if isinstance(v_bps, pd.Series) else np.asarray(v_bps)
        return v_bps, r_liq, y_bps

    def _fit_linear(
        y_bps: np.ndarray, v_bps: np.ndarray, r_liq: np.ndarray, winsor: float
    ):
        mask = np.isfinite(y_bps) & np.isfinite(v_bps) & np.isfinite(r_liq)
        y = y_bps[mask].astype(float)
        v = v_bps[mask].astype(float)
        r = r_liq[mask].astype(float)

        if winsor > 0:
            y = _winsorize(y, p=winsor)
            v = _winsorize(v, p=winsor)
            r = _winsorize(r, p=winsor)

        X = np.column_stack([np.ones_like(v), v, r])
        p, *_ = np.linalg.lstsq(X, y, rcond=None)
        p0, p1, p2 = [float(x) for x in p]
        base_bps = max(0.0, p0)
        alpha_vol = float(p1)
        beta_illiquidity = float(p2 / base_bps) if base_bps > 0 else 0.0

        y_hat = base_bps + alpha_vol * v + (base_bps * beta_illiquidity) * r
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2))) if y.size else float("nan")
        mae = float(np.mean(np.abs(y - y_hat))) if y.size else float("nan")
        corr = float(np.corrcoef(y, y_hat)[0, 1]) if y.size > 5 else float("nan")
        return base_bps, alpha_vol, beta_illiquidity, rmse, mae, corr

    # ---------- UI: текущие параметры из YAML ----------
    current_cfg = {}
    try:
        if os.path.exists(cfg_sandbox):
            current_cfg = load_sandbox_config(cfg_sandbox).model_dump()
    except Exception as e:
        st.error(f"Не удалось прочитать {cfg_sandbox}: {e}")
        current_cfg = {}

    dyn = current_cfg.get("dynamic_spread", {}) or {}
    col_now1, col_now2, col_now3 = st.columns(3)
    with col_now1:
        st.metric("base_bps (текущий)", f"{float(dyn.get('base_bps', 3.0)):.6f}")
    with col_now2:
        st.metric("alpha_vol (текущий)", f"{float(dyn.get('alpha_vol', 0.5)):.6f}")
    with col_now3:
        st.metric(
            "beta_illiquidity (текущий)",
            f"{float(dyn.get('beta_illiquidity', 1.0)):.6f}",
        )

    st.divider()

    # ---------- UI: форма калибровки ----------
    with st.form("tcost_calib_form"):
        data_path = st.text_input(
            "Путь к датасету (CSV/Parquet)",
            value=str(current_cfg.get("data", {}).get("path", "data/train.parquet")),
        )
        symbol = st.text_input(
            "Символ (опционально, напр. BTCUSDT)",
            value=str(current_cfg.get("symbol", "BTCUSDT")),
        )
        ts_col = st.text_input(
            "Колонка времени",
            value=str(current_cfg.get("data", {}).get("ts_col", "ts_ms")),
        )
        symbol_col = st.text_input(
            "Колонка символа",
            value=str(current_cfg.get("data", {}).get("symbol_col", "symbol")),
        )
        price_col = st.text_input(
            "Колонка цены-референса",
            value=str(current_cfg.get("data", {}).get("price_col", "ref_price")),
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            vol_mode = st.selectbox(
                "Источник волатильности",
                options=["hl", "ret"],
                index=(
                    0
                    if str(
                        (current_cfg.get("dynamic_spread", {}) or {}).get(
                            "vol_mode", "hl"
                        )
                    ).lower()
                    == "hl"
                    else 1
                ),
            )
        with col_b:
            winsor = st.number_input(
                "Винзоризация (доля, 0..0.2)",
                min_value=0.0,
                max_value=0.2,
                step=0.01,
                value=0.01,
            )
        with col_c:
            dry_run = st.checkbox("Только посмотреть (не записывать YAML)", value=False)

        liq_col = str(
            (current_cfg.get("dynamic_spread", {}) or {}).get(
                "liq_col", "number_of_trades"
            )
        )
        liq_ref = float(
            (current_cfg.get("dynamic_spread", {}) or {}).get("liq_ref", 1000.0)
        )

        st.caption(f"Ликвидность: liq_col='{liq_col}', liq_ref={liq_ref}")

        submitted = st.form_submit_button("Калибровать")

    if submitted:
        try:
            if not os.path.exists(data_path):
                st.error(f"Файл не найден: {data_path}")
            else:
                df = _read_table(data_path)
                if symbol:
                    df = df.loc[
                        df[symbol_col].astype(str).str.upper()
                        == str(symbol).strip().upper()
                    ].copy()
                    if df.empty:
                        st.error(f"В данных нет строк для символа {symbol}")
                        st.stop()

                v_bps, r_liq, y_bps = _compute_features(
                    df,
                    ts_col=ts_col,
                    symbol_col=symbol_col,
                    price_col=price_col,
                    vol_mode=vol_mode,
                    liq_col=liq_col,
                    liq_ref=liq_ref,
                )
                base_bps, alpha_vol, beta_ill, rmse, mae, corr = _fit_linear(
                    y_bps, v_bps, r_liq, winsor
                )

                st.success("Калибровка выполнена")
                st.json(
                    {
                        "base_bps": float(base_bps),
                        "alpha_vol": float(alpha_vol),
                        "beta_illiquidity": float(beta_ill),
                        "fit": {"rmse_bps": rmse, "mae_bps": mae, "corr": corr},
                    }
                )

                if not dry_run:
                    # обновим YAML
                    new_cfg = dict(current_cfg)
                    new_dyn = dict(new_cfg.get("dynamic_spread", {}) or {})
                    new_dyn["base_bps"] = float(base_bps)
                    new_dyn["alpha_vol"] = float(alpha_vol)
                    new_dyn["beta_illiquidity"] = float(beta_ill)
                    new_cfg["dynamic_spread"] = new_dyn

                    _ensure_dir(cfg_sandbox)
                    with open(cfg_sandbox, "w", encoding="utf-8") as f:
                        yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)
                    st.success(f"Обновлено: {cfg_sandbox}")

                    # покажем итоговые параметры
                    st.code(
                        yaml.safe_dump(
                            {"dynamic_spread": new_dyn},
                            sort_keys=False,
                            allow_unicode=True,
                        ),
                        language="yaml",
                    )
                else:
                    st.info("DRY-RUN: файл configs/sandbox.yaml не изменён.")
        except Exception as e:
            st.error(f"Ошибка калибровки: {e}")

    st.divider()
    st.markdown(
        "- **Рекомендация:** после записи параметров в YAML запусти бэктест (вкладка Sandbox Backtest) и сравни результаты.\n"
        "- **vol_mode='hl'** требует колонок `high` и `low`; иначе используй **'ret'** (будет слабее, но сработает."
    )

with tabs[14]:
    st.subheader(
        "Target Builder — cost-aware таргет с учётом комиссий и динамического спреда"
    )

    import os
    import pandas as pd
    import yaml
    from trainingtcost import effective_return_series

    def _read_table(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        raise ValueError(f"Неизвестный формат файла: {ext}")

    def _write_table(df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            df.to_parquet(path, index=False)
            return
        if ext in (".csv", ".txt"):
            df.to_csv(path, index=False)
            return
        raise ValueError(f"Неизвестный формат файла вывода: {ext}")

    with st.form("target_builder_form"):
        data_path = st.text_input(
            "Входной датасет (CSV/Parquet)", value="data/train.parquet"
        )
        out_path = st.text_input(
            "Куда сохранить (если пусто — рядом с суффиксом _costaware)", value=""
        )
        sandbox_yaml = st.text_input(
            "sandbox.yaml (динамический спред)", value="configs/sandbox.yaml"
        )
        sim_yaml = st.text_input(
            "sim.yaml (комиссии, опционально)", value="configs/sim.yaml"
        )
        fees_bps_total = st.text_input(
            "Комиссия (bps round-trip, опционально — перебивает sim.yaml)", value=""
        )
        horizon_bars = st.number_input(
            "Горизонт (в барах)", min_value=1, step=1, value=60
        )
        threshold = st.text_input(
            "Порог для бинарной метки (опционально, например 0.0005)", value=""
        )
        ts_col = st.text_input("Колонка времени", value="ts_ms")
        symbol_col = st.text_input("Колонка символа", value="symbol")
        price_col = st.text_input("Колонка цены", value="ref_price")
        submitted = st.form_submit_button("Сформировать таргет")

    if submitted:
        try:
            if not os.path.exists(data_path):
                st.error(f"Файл не найден: {data_path}")
                st.stop()
            df = _read_table(data_path)
            fees_val = float(fees_bps_total) if fees_bps_total.strip() else None
            thr_val = float(threshold) if threshold.strip() else None

            out_df = effective_return_series(
                df,
                horizon_bars=int(horizon_bars),
                fees_bps_total=(fees_val if fees_val is not None else 10.0),
                sandbox_yaml_path=sandbox_yaml,
                ts_col=ts_col,
                symbol_col=symbol_col,
                ref_price_col=price_col,
                label_threshold=thr_val,
                roundtrip_spread=True,
            )

            if not out_path.strip():
                base, ext = os.path.splitext(data_path)
                out_path = f"{base}_costaware{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"

            _write_table(out_df, out_path)
            st.success(f"Готово. Записано: {out_path}")
            cols = [
                c
                for c in out_df.columns
                if c.startswith("eff_ret_") or c.startswith("y_eff_")
            ] + ["slippage_bps", "fees_bps_total"]
            st.caption("Добавленные колонки:")
            st.code(", ".join(cols))
        except Exception as e:
            st.error(f"Ошибка: {e}")

with tabs[15]:
    st.subheader("No-Trade Mask — убрать или занулить запрещённые окна в данных")

    import os
    import pandas as pd
    from no_trade import compute_no_trade_mask

    def _read_table_mask(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        raise ValueError(f"Неизвестный формат файла: {ext}")

    def _write_table_mask(df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            df.to_parquet(path, index=False)
            return
        if ext in (".csv", ".txt"):
            df.to_csv(path, index=False)
            return
        raise ValueError(f"Неизвестный формат файла вывода: {ext}")

    with st.form("no_trade_mask_form"):
        data_path = st.text_input(
            "Входной датасет (после Target Builder)",
            value="data/train_costaware.parquet",
        )
        out_path = st.text_input(
            "Куда сохранить (если пусто — суффикс _masked)", value=""
        )
        sandbox_yaml = st.text_input(
            "sandbox.yaml (правила no_trade)", value="configs/sandbox.yaml"
        )
        ts_col = st.text_input("Колонка времени", value="ts_ms")
        mode = st.selectbox("Режим", options=["drop", "weight"], index=0)
        submitted = st.form_submit_button("Применить маску")

    if submitted:
        try:
            if not os.path.exists(data_path):
                st.error(f"Файл не найден: {data_path}")
                st.stop()
            df = _read_table_mask(data_path)
            mask_block = compute_no_trade_mask(
                df, sandbox_yaml_path=sandbox_yaml, ts_col=ts_col
            )

            if mode == "drop":
                out_df = df.loc[~mask_block].reset_index(drop=True)
            else:
                out_df = df.copy()
                out_df["train_weight"] = 1.0
                out_df.loc[mask_block, "train_weight"] = 0.0

            if not out_path.strip():
                base, ext = os.path.splitext(data_path)
                out_path = f"{base}_masked{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"

            _write_table_mask(out_df, out_path)

            total = int(len(df))
            blocked = int(mask_block.sum())
            kept = int(len(out_df))
            st.success(
                f"Готово. Всего строк: {total}. Запрещённых (no_trade): {blocked}. Вышло: {kept}."
            )
            if mode == "weight":
                z = int(
                    (out_df.get("train_weight", pd.Series(dtype=float)) == 0.0).sum()
                )
                st.info(f"Назначено train_weight=0 для {z} строк.")
        except Exception as e:
            st.error(f"Ошибка: {e}")

with tabs[16]:
    st.subheader("Walk-Forward Splits — сплиты с PURGE (горизонт) и EMBARGO (буфер)")

    import os
    import json
    import pandas as pd
    import yaml
    from splits import make_walkforward_splits

    def _read_table_wf(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        raise ValueError(f"Неизвестный формат файла: {ext}")

    def _write_table_wf(df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            df.to_parquet(path, index=False)
            return
        if ext in (".csv", ".txt"):
            df.to_csv(path, index=False)
            return
        raise ValueError(f"Неизвестный формат файла вывода: {ext}")

    def _write_manifest(manifest, json_path: str, yaml_path: str) -> None:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        data = [m.to_dict() for m in manifest]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    with st.form("walkforward_form"):
        data_path = st.text_input(
            "Входной датасет (после маски)", value="data/train_costaware_masked.parquet"
        )
        out_path = st.text_input("Куда сохранить (если пусто — суффикс _wf)", value="")
        ts_col = st.text_input("Колонка времени", value="ts_ms")
        symbol_col = st.text_input("Колонка символа (если есть)", value="symbol")
        interval_ms = st.text_input(
            "Интервал бара, мс (опционально, иначе оценим)", value=""
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            train_span_bars = st.number_input(
                "train_span_bars", min_value=1, step=1, value=7 * 24 * 60
            )
        with col2:
            val_span_bars = st.number_input(
                "val_span_bars", min_value=1, step=1, value=24 * 60
            )
        with col3:
            step_bars = st.number_input("step_bars", min_value=1, step=1, value=24 * 60)
        col4, col5 = st.columns(2)
        with col4:
            horizon_bars = st.number_input(
                "horizon_bars (PURGE)", min_value=1, step=1, value=60
            )
        with col5:
            embargo_bars = st.number_input(
                "embargo_bars (EMBARGO)", min_value=0, step=1, value=5
            )
        manifest_dir = st.text_input(
            "Куда писать манифест (JSON/YAML)", value="logs/walkforward"
        )
        submitted = st.form_submit_button("Сделать сплиты")

    if submitted:
        try:
            if not os.path.exists(data_path):
                st.error(f"Файл не найден: {data_path}")
                st.stop()

            df = _read_table_wf(data_path)
            iv_ms = int(interval_ms) if interval_ms.strip() else None

            df_out, manifest = make_walkforward_splits(
                df,
                ts_col=ts_col,
                symbol_col=(symbol_col if symbol_col in df.columns else None),
                interval_ms=iv_ms,
                train_span_bars=int(train_span_bars),
                val_span_bars=int(val_span_bars),
                step_bars=int(step_bars),
                horizon_bars=int(horizon_bars),
                embargo_bars=int(embargo_bars),
            )

            if not out_path.strip():
                base, ext = os.path.splitext(data_path)
                out_path = f"{base}_wf{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"

            _write_table_wf(df_out, out_path)

            json_path = os.path.join(manifest_dir, "walkforward_manifest.json")
            yaml_path = os.path.join(manifest_dir, "walkforward_manifest.yaml")
            _write_manifest(manifest, json_path=json_path, yaml_path=yaml_path)

            total = int(len(df_out))
            used = int((df_out["wf_role"] != "none").sum())
            n_train = int((df_out["wf_role"] == "train").sum())
            n_val = int((df_out["wf_role"] == "val").sum())

            st.success(f"Готово. Датасет со сплитами: {out_path}")
            st.info(
                f"Всего строк: {total}. В сплитах train: {n_train}, val: {n_val}, вне окон: {total - used}."
            )
            st.caption("Манифесты записаны:")
            st.code(json_path)
            st.code(yaml_path)
        except Exception as e:
            st.error(f"Ошибка: {e}")


with tabs[17]:
    st.subheader(
        "Threshold Tuner — подбор порога под целевую частоту с учётом кулдауна и no-trade"
    )

    import os
    import pandas as pd
    from threshold_tuner import (
        TuneConfig,
        tune_threshold,
        load_min_signal_gap_s_from_yaml,
    )

    def _read_table_thr(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        raise ValueError(f"Неизвестный формат файла: {ext}")

    def _write_csv_thr(df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)

    with st.form("threshold_tuner_form"):
        data_path = st.text_input(
            "Датасет предсказаний (CSV/Parquet):", value="data/val_predictions.parquet"
        )
        score_col = st.text_input("Колонка со скором/вероятностью:", value="score")
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.text_input(
                "Колонка бинарной метки (если классификация):", value="y_eff_60"
            )
        with col2:
            ret_col = st.text_input(
                "Колонка эффективного ретёрна (если регрессия):", value="eff_ret_60"
            )

        ts_col = st.text_input("Колонка времени:", value="ts_ms")
        symbol_col = st.text_input("Колонка символа:", value="symbol")
        direction = st.selectbox(
            "Правило сигнала", options=["greater", "less"], index=0
        )

        col3, col4, col5 = st.columns(3)
        with col3:
            target_signals_per_day = st.number_input(
                "Желаемые сигналы/день", min_value=0.1, step=0.1, value=1.5
            )
        with col4:
            tolerance = st.number_input(
                "Допуск по частоте", min_value=0.0, step=0.1, value=0.5
            )
        with col5:
            optimize_for = st.selectbox(
                "Метрика оптимизации", options=["sharpe", "precision", "f1"], index=0
            )

        col6, col7, col8 = st.columns(3)
        with col6:
            min_thr = st.number_input(
                "Минимальный порог", min_value=0.0, max_value=1.0, step=0.01, value=0.50
            )
        with col7:
            max_thr = st.number_input(
                "Максимальный порог",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=0.99,
            )
        with col8:
            steps = st.number_input("Число шагов сетки", min_value=5, step=1, value=50)

        col9, col10 = st.columns(2)
        with col9:
            realtime_yaml = st.text_input(
                "realtime.yaml (для чтения min_signal_gap_s):",
                value="configs/realtime.yaml",
            )
        with col10:
            manual_gap = st.text_input(
                "min_signal_gap_s (перебьёт realtime.yaml, опционально):", value=""
            )

        col11, col12 = st.columns(2)
        with col11:
            sandbox_yaml = st.text_input(
                "sandbox.yaml (no_trade):", value="configs/sandbox.yaml"
            )
        with col12:
            drop_no_trade = st.checkbox("Учитывать no-trade (фильтровать)", value=True)

        out_csv = st.text_input("Куда сохранить таблицу результатов (.csv):", value="")

        submitted = st.form_submit_button("Подобрать порог")

    if submitted:
        try:
            if not os.path.exists(data_path):
                st.error(f"Файл не найден: {data_path}")
                st.stop()
            df = _read_table_thr(data_path)

            # min gap
            if manual_gap.strip():
                min_gap = int(float(manual_gap))
            else:
                min_gap = load_min_signal_gap_s_from_yaml(realtime_yaml)

            cfg = TuneConfig(
                score_col=score_col,
                y_col=(y_col if y_col.strip() else None),
                ret_col=(ret_col if ret_col.strip() else None),
                ts_col=ts_col,
                symbol_col=symbol_col,
                direction=direction,
                target_signals_per_day=float(target_signals_per_day),
                tolerance=float(tolerance),
                min_signal_gap_s=int(min_gap or 0),
                min_thr=float(min_thr),
                max_thr=float(max_thr),
                steps=int(steps),
                sandbox_yaml_for_no_trade=(sandbox_yaml if drop_no_trade else None),
                drop_no_trade=bool(drop_no_trade),
                optimize_for=optimize_for,
            )

            res, best = tune_threshold(df, cfg)

            # сохраняем таблицу
            if not out_csv.strip():
                base, ext = os.path.splitext(data_path)
                out_csv = f"{base}_thrscan.csv"
            _write_csv_thr(res, out_csv)

            st.success("Готово. Рекомендованный порог и метрики ниже.")
            st.json(
                {
                    k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in best.items()
                }
            )
            st.caption("Первые 20 строк таблицы результатов:")
            st.dataframe(res.sort_values("signals_per_day").head(20))

            st.caption("Файл с полным сканом порогов записан:")
            st.code(out_csv)
        except Exception as e:
            st.error(f"Ошибка тюнинга порога: {e}")

with tabs[18]:
    st.subheader("Probability Calibration — Platt/Isotonic калибровка вероятностей")

    import os
    import json
    import numpy as np
    import pandas as pd
    from calibration import (
        fit_calibrator,
        BaseCalibrator,
        evaluate_before_after,
        calibration_table,
    )

    def _read_table_calib(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        raise ValueError(f"Неизвестный формат файла: {ext}")

    def _write_csv_calib(df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)

    st.markdown("### Обучение калибратора")
    with st.form("calibration_train_form"):
        data_path = st.text_input(
            "Файл с предсказаниями (CSV/Parquet):", value="data/val_predictions.parquet"
        )
        score_col = st.text_input("Колонка со скором/вероятностью:", value="score")
        y_col = st.text_input("Колонка бинарной метки 0/1:", value="y_eff_60")
        filter_val = st.checkbox("Фильтровать wf_role=='val'", value=True)
        wf_role_col = st.text_input(
            "Имя колонки роли (если фильтруем):", value="wf_role"
        )
        method = st.selectbox(
            "Метод калибровки:", options=["platt", "isotonic"], index=0
        )
        out_model = st.text_input(
            "Куда сохранить калибратор (.json):", value="models/calibrator.json"
        )
        report_csv = st.text_input(
            "Куда сохранить calibration-table (.csv, опционально):", value=""
        )
        submitted_train = st.form_submit_button("Обучить калибратор")

    if submitted_train:
        try:
            if not os.path.exists(data_path):
                st.error(f"Файл не найден: {data_path}")
                st.stop()
            df = _read_table_calib(data_path)
            if filter_val and wf_role_col in df.columns:
                df = df.loc[df[wf_role_col].astype(str) == "val"].reset_index(drop=True)
            if score_col not in df.columns or y_col not in df.columns:
                st.error(f"Нужны колонки: {score_col}, {y_col}")
                st.stop()

            s = pd.to_numeric(df[score_col], errors="coerce").astype(float).to_numpy()
            y = pd.to_numeric(df[y_col], errors="coerce").astype(float).to_numpy()

            cal = fit_calibrator(s, y, method=method)
            cal.save_json(out_model)

            metrics = evaluate_before_after(s, y, cal, bins=10)
            st.success(f"Готово. Калибратор сохранён: {out_model}")
            st.json(
                {
                    k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in metrics.items()
                }
            )

            # calibration-table после калибровки
            p_after = np.clip(cal.predict_proba(s), 0.0, 1.0)
            tbl = calibration_table(p_after, y, bins=10)
            if report_csv.strip():
                _write_csv_calib(tbl, report_csv.strip())
            st.caption("Calibration table (первые 10 строк):")
            st.dataframe(tbl.head(10))
        except Exception as e:
            st.error(f"Ошибка обучения калибратора: {e}")

    st.divider()
    st.markdown("### Применение калибратора к датасету")

    with st.form("calibration_apply_form"):
        data_path2 = st.text_input(
            "Файл с предсказаниями (CSV/Parquet) для применения:",
            value="data/val_predictions.parquet",
        )
        model_json = st.text_input("JSON калибратора:", value="models/calibrator.json")
        score_col2 = st.text_input("Имя колонки со скором:", value="score")
        out_col = st.text_input(
            "Имя новой колонки для калиброванной вероятности:", value="score_calibrated"
        )
        out_path2 = st.text_input(
            "Куда сохранить (если пусто — суффикс _calibrated):", value=""
        )
        submitted_apply = st.form_submit_button("Применить калибратор")

    if submitted_apply:
        try:
            if not os.path.exists(data_path2):
                st.error(f"Файл не найден: {data_path2}")
                st.stop()
            if not os.path.exists(model_json):
                st.error(f"Модель не найдена: {model_json}")
                st.stop()

            df2 = _read_table_calib(data_path2)
            if score_col2 not in df2.columns:
                st.error(f"Нет колонки: {score_col2}")
                st.stop()

            cal = BaseCalibrator.load_json(model_json)
            s2 = (
                pd.to_numeric(df2[score_col2], errors="coerce").astype(float).to_numpy()
            )
            p2 = cal.predict_proba(s2)
            df2[out_col] = p2

            if not out_path2.strip():
                base, ext = os.path.splitext(data_path2)
                out_path2 = f"{base}_calibrated{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"

            if out_path2.lower().endswith((".parquet", ".pq")):
                df2.to_parquet(out_path2, index=False)
            else:
                df2.to_csv(out_path2, index=False)
            st.success(f"Готово. Записано: {out_path2}")
            st.dataframe(df2.head(10))
        except Exception as e:
            st.error(f"Ошибка применения калибратора: {e}")

with tabs[19]:
    st.subheader("Drift Monitor — PSI по фичам и скору")

    import os
    import numpy as np
    import pandas as pd
    from drift import (
        make_baseline,
        save_baseline_json,
        load_baseline_json,
        compute_psi,
        default_feature_list,
    )

    def _read_table_dm(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        raise ValueError(f"Неизвестный формат файла: {ext}")

    def _write_csv_dm(df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### 1) Сформировать baseline")
        with st.form("drift_baseline_form"):
            base_data = st.text_input(
                "Файл для baseline (обычно валид. срез)",
                value="data/val_predictions.parquet",
            )
            base_features = st.text_input(
                "Фичи (через запятую, пусто — авто f_* и score)", value=""
            )
            base_bins = st.number_input(
                "Число бинов (числовые фичи)", min_value=2, step=1, value=10
            )
            base_topk = st.number_input(
                "Top-K категорий", min_value=5, step=1, value=20
            )
            base_out = st.text_input(
                "Куда сохранить baseline JSON", value="models/drift_baseline.json"
            )
            submitted_base = st.form_submit_button("Сделать baseline")

        if submitted_base:
            try:
                if not os.path.exists(base_data):
                    st.error(f"Файл не найден: {base_data}")
                    st.stop()
                dfb = _read_table_dm(base_data)
                if base_features.strip():
                    feats = [s.strip() for s in base_features.split(",") if s.strip()]
                else:
                    feats = default_feature_list(dfb)
                    if not feats:
                        st.error("Не удалось автодетектить фичи. Укажи их явно.")
                        st.stop()
                spec = make_baseline(
                    dfb, feats, bins=int(base_bins), top_k_cats=int(base_topk)
                )
                save_baseline_json(spec, base_out)
                st.success(f"Baseline сохранён: {base_out}")
                st.code(", ".join(feats))
            except Exception as e:
                st.error(f"Ошибка создания baseline: {e}")

    with colB:
        st.markdown("### 2) Проверить дрифт")
        with st.form("drift_check_form"):
            cur_data = st.text_input(
                "Текущий датасет (онлайн/последние дни)",
                value="data/online_last.parquet",
            )
            baseline_json = st.text_input(
                "Baseline JSON", value="models/drift_baseline.json"
            )
            features = st.text_input("Фичи (пусто — из baseline)", value="")
            ts_col = st.text_input("Колонка времени (UTC мс)", value="ts_ms")
            last_days = st.number_input(
                "Сколько последних дней взять", min_value=0, step=1, value=14
            )
            psi_warn = st.number_input(
                "Порог предупреждения PSI", min_value=0.0, step=0.01, value=0.10
            )
            psi_alert = st.number_input(
                "Порог алёрта PSI", min_value=0.0, step=0.01, value=0.25
            )
            out_csv = st.text_input("Куда сохранить CSV с PSI (опционально)", value="")
            submitted_check = st.form_submit_button("Посчитать PSI")

        if submitted_check:
            try:
                if not os.path.exists(cur_data):
                    st.error(f"Файл не найден: {cur_data}")
                    st.stop()
                if not os.path.exists(baseline_json):
                    st.error(f"Baseline JSON не найден: {baseline_json}")
                    st.stop()

                dfc = _read_table_dm(cur_data)
                if int(last_days) > 0 and ts_col in dfc.columns:
                    max_ts = int(pd.to_numeric(dfc[ts_col], errors="coerce").max())
                    cutoff = max_ts - int(last_days) * 86400000
                    dfc = dfc.loc[
                        pd.to_numeric(dfc[ts_col], errors="coerce") >= cutoff
                    ].reset_index(drop=True)

                base = load_baseline_json(baseline_json)
                if features.strip():
                    feats = [s.strip() for s in features.split(",") if s.strip()]
                else:
                    feats = list(base.keys())

                res = compute_psi(dfc, base, features=feats)

                if out_csv.strip():
                    _write_csv_dm(res, out_csv.strip())

                st.success("PSI посчитан")
                if not res.empty:
                    avg_psi = float(
                        res["psi"].replace([np.inf, -np.inf], np.nan).dropna().mean()
                    )
                    worst_feature = res.iloc[0]["feature"]
                    worst_psi = float(res.iloc[0]["psi"])
                    st.metric("Средний PSI", f"{avg_psi:.4f}")
                    st.metric(
                        "Максимальный PSI",
                        f"{worst_psi:.4f}",
                        help=f"Фича: {worst_feature}",
                    )

                    if worst_psi >= psi_alert or avg_psi >= psi_alert:
                        st.error(
                            "⚠️ Сильный дрифт: PSI > alert. Рекомендуется переобучение/перекалибровка."
                        )
                    elif worst_psi >= psi_warn or avg_psi >= psi_warn:
                        st.warning(
                            "ℹ️ Умеренный дрифт: PSI > warn. Наблюдать, возможно готовить переобучение."
                        )
                    else:
                        st.info("✅ Дрифт незначительный: PSI в норме.")

                st.caption("Таблица PSI (топ-50 по убыванию):")
                st.dataframe(res.head(50))
            except Exception as e:
                st.error(f"Ошибка расчёта PSI: {e}")

with tabs[20]:
    st.subheader("Monitoring overview")

    monitoring_logs = st.session_state.setdefault("monitoring_logs", [])
    latest_metrics = _load_latest_metrics(os.path.join(logs_dir, "metrics.jsonl"))
    queue_info = latest_metrics.get("throttle_queue") or {}
    cooldowns = latest_metrics.get("cooldowns_active") or {}
    ws_data = latest_metrics.get("ws") or {}

    def _fmt_ts(ts_ms: Any) -> str:
        try:
            ts_val = float(ts_ms)
        except (TypeError, ValueError):
            return "—"
        if ts_val <= 0:
            return "—"
        try:
            return datetime.utcfromtimestamp(ts_val / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "—"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clock drift (ms)", f"{clock.clock_skew():.2f}")
        age_sec = clock.last_sync_age_sec()
        age_text = f"{age_sec:.1f}" if age_sec != float("inf") else "∞"
        st.metric("Last sync age (s)", age_text)
    with col2:
        st.metric(
            "Queue size",
            f"{queue_info.get('size', 0)}/{queue_info.get('max', 0)}",
        )
        st.metric("Cooldowns active", str(cooldowns.get("count", 0)))
    with col3:
        st.metric("Last WS failure", _fmt_ts(ws_data.get("last_failure_ms")))
        st.metric("Last WS reconnect", _fmt_ts(ws_data.get("last_reconnect_ms")))

    st.divider()
    st.subheader("Snapshot summary")
    snapshot_summary = read_json(snapshot_json)
    if snapshot_summary:
        st.json(snapshot_summary)
    else:
        st.info("Snapshot metrics JSON не найден.")

    snap_df = read_csv(snapshot_csv)
    if not snap_df.empty:
        st.subheader("Snapshot metrics chart")
        numeric_df = snap_df.copy()
        numeric_df["value"] = pd.to_numeric(numeric_df["value"], errors="coerce")
        plot_df = numeric_df.dropna(subset=["value"]).set_index("metric")["value"]
        if not plot_df.empty:
            st.bar_chart(plot_df)
        st.dataframe(snap_df)
    else:
        st.info("Snapshot metrics CSV не найден или пуст.")

    st.divider()
    st.subheader("Clock sync")

    if st.button("Sync now", type="primary"):
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        run_cfg = None
        try:
            run_cfg = load_config(cfg_realtime)
        except Exception as exc:
            monitoring_logs.append(f"[{timestamp}] config error: {exc}")
            monitoring_logs[:] = monitoring_logs[-200:]
            st.warning(f"Не удалось загрузить конфиг: {exc}")
        if run_cfg is not None:
            clock_cfg = getattr(run_cfg, "clock_sync", ClockSyncConfig())
            rest_cfg_data: Dict[str, Any] = {}
            if rest_budget_cfg and os.path.exists(rest_budget_cfg):
                try:
                    with open(rest_budget_cfg, "r", encoding="utf-8") as fh:
                        raw_cfg = yaml.safe_load(fh) or {}
                    if isinstance(raw_cfg, dict):
                        rest_cfg_data = raw_cfg
                except Exception as exc:
                    monitoring_logs.append(
                        f"[{timestamp}] rest config error: {exc}"
                    )
                    monitoring_logs[:] = monitoring_logs[-200:]
                    st.warning(f"Ошибка rest-конфига: {exc}")
            elif rest_budget_cfg:
                monitoring_logs.append(
                    f"[{timestamp}] rest config not found: {rest_budget_cfg}"
                )
                monitoring_logs[:] = monitoring_logs[-200:]
                st.warning(f"Rest budget config не найден: {rest_budget_cfg}")

            try:
                with RestBudgetSession(rest_cfg_data) as sess:
                    drift, rtt = clock.manual_sync(clock_cfg, session=sess)
            except Exception as exc:
                monitoring_logs.append(f"[{timestamp}] sync failed: {exc}")
                monitoring_logs[:] = monitoring_logs[-200:]
                st.warning(f"Синхронизация не удалась: {exc}")
            else:
                monitoring.report_clock_sync(
                    drift, rtt, True, clock.system_utc_ms()
                )
                monitoring_logs.append(
                    f"[{timestamp}] sync drift={drift:.2f}ms rtt={rtt:.2f}ms"
                )
                monitoring_logs[:] = monitoring_logs[-200:]
                st.success(
                    f"Drift {drift:.2f} ms, median RTT {rtt:.2f} ms"
                )

    st.divider()
    st.subheader("HTTP metrics")
    http_data = latest_metrics.get("http") or {}
    http_rows: List[Dict[str, Any]] = []
    for window, stats in http_data.items():
        if not isinstance(stats, dict):
            continue
        row: Dict[str, Any] = {
            "window": window,
            "attempts": stats.get("attempts"),
            "success": stats.get("success"),
            "errors": stats.get("errors"),
            "total": stats.get("total"),
            "error_rate": stats.get("error_rate"),
        }
        codes = stats.get("by_code") or {}
        if isinstance(codes, dict):
            for code, value in codes.items():
                row[f"code_{code}"] = value
        http_rows.append(row)
    if http_rows:
        st.table(pd.DataFrame(http_rows).set_index("window"))
    else:
        st.info("HTTP статистика пока недоступна.")

    st.subheader("Websocket metrics")
    if ws_data:
        st.table(pd.DataFrame([ws_data]))
    else:
        st.info("Websocket статистика пока недоступна.")

    st.subheader("Signal metrics")
    signals_data = latest_metrics.get("signals") or {}
    signal_rows: List[Dict[str, Any]] = []
    for key in ("window_1m", "window_5m"):
        stats = signals_data.get(key)
        if isinstance(stats, dict):
            row = {"window": key}
            row.update(stats)
            signal_rows.append(row)
    if signal_rows:
        st.table(pd.DataFrame(signal_rows).set_index("window"))
    else:
        st.info("Signal статистика пока недоступна.")
    streaks = signals_data.get("zero_signal_streaks") or {}
    if isinstance(streaks, dict) and streaks:
        streak_df = (
            pd.DataFrame(
                [{"symbol": sym, "streak": streak} for sym, streak in streaks.items()]
            )
            .sort_values("streak", ascending=False)
            .set_index("symbol")
        )
        st.table(streak_df)

    st.subheader("Sync log")
    if monitoring_logs:
        st.text_area(
            "Log",
            "\n".join(monitoring_logs[-50:]),
            height=160,
        )
    else:
        st.info("Лог синхронизации пока пуст.")

with tabs[21]:
    st.subheader("Offline REST jobs")

    offline_config_path = st.text_input(
        "Offline config path", value="configs/offline.yaml", key="offline_jobs_config"
    )

    universe_default = st.session_state.get("universe_json_path", "data/universe/symbols.json")
    filters_default = st.session_state.get("filters_json_path", "data/binance_filters.json")

    jobs: list[dict[str, Any]] = [
        {
            "key": "refresh_universe",
            "title": "Refresh universe",
            "script": "scripts/refresh_universe.py",
            "stats": "logs/offline/refresh_universe_stats.json",
            "log": "logs/offline/refresh_universe.log",
            "default_args": f"--out {universe_default}",
            "supports_dry_run": True,
            "resume_flag": None,
            "description": "Fetch Binance exchange info and update USDT universe.",
        },
        {
            "key": "fetch_binance_filters",
            "title": "Fetch Binance filters",
            "script": "scripts/fetch_binance_filters.py",
            "stats": "logs/offline/fetch_binance_filters_stats.json",
            "log": "logs/offline/fetch_binance_filters.log",
            "default_args": f"--out {filters_default} --universe {universe_default}",
            "supports_dry_run": True,
            "resume_flag": None,
            "description": "Download and cache exchange filters for configured symbols.",
        },
        {
            "key": "build_adv",
            "title": "Build ADV dataset",
            "script": "build_adv.py",
            "stats": "logs/offline/build_adv_stats.json",
            "log": "logs/offline/build_adv.log",
            "default_args": "--market futures --interval 1h --start 2023-01-01T00:00:00Z --end 2023-01-08T00:00:00Z --symbols BTCUSDT,ETHUSDT --out data/adv/klines.parquet",
            "supports_dry_run": True,
            "resume_flag": "--resume-from-checkpoint",
            "description": "Fetch OHLCV history and build aggregated ADV parquet.",
        },
        {
            "key": "build_adv_hourly",
            "title": "Build ADV quotes (scripts/build_adv.py)",
            "script": "scripts/build_adv.py",
            "stats": "logs/offline/build_adv_hourly_stats.json",
            "log": "logs/offline/build_adv_hourly.log",
            "default_args": "--market futures --interval 1d --window-days 30 --out data/adv/adv_quotes.json",
            "supports_dry_run": False,
            "resume_flag": None,
            "description": "Generate ADV quotes from cached klines for reporting.",
        },
        {
            "key": "build_adv_base",
            "title": "Build ADV base volumes",
            "script": "scripts/build_adv_base.py",
            "stats": "logs/offline/build_adv_base_stats.json",
            "log": "logs/offline/build_adv_base.log",
            "default_args": "--market futures --interval 1h --window-days 30 --out data/adv/adv_base.json",
            "supports_dry_run": False,
            "resume_flag": None,
            "description": "Aggregate base volumes into ADV checkpoints.",
        },
    ]

    for job in jobs:
        with st.expander(job["title"], expanded=False):
            st.caption(job.get("description") or "")
            args_default = job.get("default_args", "")
            args_text = st.text_input(
                "Additional arguments",
                value=args_default,
                key=f"{job['key']}_args",
            )

            def _build_command(extra_flag: str | None = None) -> list[str]:
                base_args = shlex.split(args_text) if args_text.strip() else []
                cmd = [py, job["script"]]
                if offline_config_path:
                    if "--config" not in base_args:
                        cmd.extend(["--config", offline_config_path])
                cmd.extend(base_args)
                if extra_flag and extra_flag not in cmd:
                    cmd.append(extra_flag)
                return cmd

            log_path = job["log"]
            stats_path = job["stats"]

            col_run, col_dry, col_resume = st.columns(3)
            if col_run.button("Run", key=f"{job['key']}_run"):
                rc = run_cmd(_build_command(), log_path=log_path)
                if rc == 0:
                    st.success("Job completed successfully")
                else:
                    st.error(f"Job exited with code {rc}")
            if job.get("supports_dry_run"):
                if col_dry.button("Dry run", key=f"{job['key']}_dry"):
                    rc = run_cmd(_build_command("--dry-run"), log_path=log_path)
                    if rc == 0:
                        st.success("Dry run finished successfully")
                    else:
                        st.error(f"Dry run exited with code {rc}")
            else:
                col_dry.write("—")
            resume_flag = job.get("resume_flag")
            if resume_flag:
                if col_resume.button("Resume checkpoint", key=f"{job['key']}_resume"):
                    rc = run_cmd(_build_command(resume_flag), log_path=log_path)
                    if rc == 0:
                        st.success("Resume completed successfully")
                    else:
                        st.error(f"Resume exited with code {rc}")
            else:
                col_resume.write("—")

            stats = read_json(stats_path)
            if stats:
                requests_total = int(stats.get("requests_total", 0))
                total_retries = int(stats.get("total_retries", 0))
                total_wait = float(stats.get("total_wait_seconds", 0.0))
                cols = st.columns(3)
                cols[0].metric("Requests", f"{requests_total}")
                cols[1].metric("Retries", f"{total_retries}")
                cols[2].metric("Wait (s)", f"{total_wait:.2f}")

                checkpoint_meta = stats.get("checkpoint_meta")
                progress_pct: float | None = None
                if isinstance(checkpoint_meta, Mapping):
                    try:
                        progress_pct = float(checkpoint_meta.get("progress_pct"))
                    except (TypeError, ValueError):
                        progress_pct = None
                if progress_pct is not None:
                    st.progress(min(max(progress_pct / 100.0, 0.0), 1.0))
                    st.caption(f"Progress: {progress_pct:.2f}%")
                    last_symbol = checkpoint_meta.get("last_symbol") if isinstance(checkpoint_meta, Mapping) else None
                    if last_symbol:
                        st.caption(f"Last symbol: {last_symbol}")

                cooldowns_active = stats.get("cooldowns_active")
                if isinstance(cooldowns_active, Mapping):
                    active_count = int(cooldowns_active.get("count", 0))
                    if active_count > 0:
                        st.warning(
                            f"Cooldowns active: {active_count}. Requests will be delayed until limits recover."
                        )
                    endpoints_active = cooldowns_active.get("endpoints")
                    if isinstance(endpoints_active, Mapping) and endpoints_active:
                        cooldown_df = pd.DataFrame.from_dict(endpoints_active, orient="index")
                        st.dataframe(cooldown_df)

                requests = stats.get("requests") or {}
                wait_seconds = stats.get("wait_seconds") or {}
                retry_counts = stats.get("retry_counts") or {}
                if isinstance(requests, Mapping) or isinstance(wait_seconds, Mapping):
                    summary_df = pd.DataFrame(
                        {
                            "requests": pd.Series(requests, dtype="float64"),
                            "retries": pd.Series(retry_counts, dtype="float64"),
                            "wait_seconds": pd.Series(wait_seconds, dtype="float64"),
                        }
                    ).fillna(0.0)
                    if not summary_df.empty:
                        st.dataframe(summary_df.sort_values("requests", ascending=False))

                cooldown_counts = stats.get("cooldowns") or {}
                if isinstance(cooldown_counts, Mapping) and cooldown_counts:
                    st.dataframe(
                        pd.DataFrame.from_dict(cooldown_counts, orient="index", columns=["count"])
                    )

                session_meta = stats.get("session")
                if isinstance(session_meta, Mapping):
                    st.caption("Session configuration snapshot:")
                    st.json(session_meta)
            else:
                st.info("Stats file not found yet. Run the job to generate metrics.")

            log_tail = tail_file(log_path, n=200)
            if log_tail:
                st.text_area(
                    "Log tail",
                    log_tail,
                    height=220,
                    key=f"{job['key']}_log_tail",
                )
            else:
                st.info("Log file отсутствует или пуст.")
