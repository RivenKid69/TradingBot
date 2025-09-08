# app.py
from __future__ import annotations

import os
import sys
import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml

from core_config import load_config, load_config_from_str
from ingest_config import load_config as load_ingest_config, load_config_from_str as parse_ingest_config
from legacy_sandbox_config import (
    load_config as load_sandbox_config,
    load_config_from_str as parse_sandbox_config,
    SandboxConfig,
)

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
)
from service_backtest import BacktestConfig, from_config as backtest_from_config
from service_train import ServiceTrain, TrainConfig
from offline_feature_pipe import OfflineFeaturePipe
from transformers import FeatureSpec
from service_signal_runner import ServiceSignalRunner, RunnerConfig
from service_eval import ServiceEval, EvalConfig


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
    rc = run_cmd([py, "scripts/ingest_orchestrator.py", "--config", cfg_ingest],
                 log_path=os.path.join(logs_dir, "ingest.log"))
    if rc != 0:
        st.error(f"Ingest завершился с кодом {rc}")
        return

    rc = run_cmd([
        py, "scripts/make_features.py",
        "--in", prices_in,
        "--out", features_out,
        "--lookbacks", lookbacks,
        "--rsi-period", str(int(rsi_period)),
    ], log_path=os.path.join(logs_dir, "features.log"))
    if rc != 0:
        st.error(f"make_features завершился с кодом {rc}")
        return

    args = [
        py, "scripts/build_training_table.py",
        "--base", bt_base,
        "--prices", bt_prices,
        "--price-col", bt_price_col,
        "--decision-delay-ms", str(int(bt_decision_delay)),
        "--label-horizon-ms", str(int(bt_horizon)),
        "--out", bt_out,
    ]
    rc = run_cmd(args, log_path=os.path.join(logs_dir, "train_table.log"))
    if rc != 0:
        st.error(f"build_training_table завершился с кодом {rc}")
        return

    rc = run_cmd([py, "scripts/run_sandbox.py", "--config", cfg_sandbox],
                 log_path=os.path.join(logs_dir, "sandbox.log"))
    if rc != 0:
        st.error(f"run_sandbox завершился с кодом {rc}")
        return

    rc = run_cmd([
        py, "scripts/evaluate_performance.py",
        "--trades", trades_path,
        "--reports", reports_path,
        "--out-json", metrics_json,
        "--out-md", out_md,
        "--equity-png", equity_png,
        "--capital-base", "10000",
        "--rf-annual", "0.00",
    ], log_path=os.path.join(logs_dir, "evaluate.log"))
    if rc != 0:
        st.error(f"evaluate_performance завершился с кодом {rc}")
        return

    st.success("Полный прогон: метрики готовы")

    if start_realtime:
        if background_running(realtime_pid):
            st.info("Realtime сигналер уже запущен")
        else:
            try:
                pid = start_background(
                    [py, "scripts/run_realtime_signaler.py", "--config", cfg_realtime],
                    pid_file=realtime_pid,
                    log_file=realtime_log,
                )
                st.success(f"Realtime сигналер запущен, PID={pid}")
            except Exception as e:
                st.error(str(e))


# --------------------------- Service wrappers ---------------------------

def run_backtest_from_yaml(cfg_path: str, default_out: str) -> str:
    cfg: SandboxConfig = load_sandbox_config(cfg_path)
    sim_cfg = load_config(cfg.sim_config_path)

    sb_cfg = BacktestConfig(
        dynamic_spread_config=cfg.dynamic_spread,
        exchange_specs_path=cfg.exchange_specs_path,
        guards_config=cfg.sim_guards,
        signal_cooldown_s=int(cfg.min_signal_gap_s),
        no_trade_config=cfg.no_trade,
    )

    data_cfg = cfg.data
    path = data_cfg.path
    df = pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)
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

    logs_dir = st.text_input("Каталог логов", value="logs")
    _ensure_dir(logs_dir)

    trades_path = st.text_input("Путь к трейдам (для evaluate)", value=os.path.join(logs_dir, "log_trades_*.csv"))
    reports_path = st.text_input("Путь к отчётам (для evaluate)", value=os.path.join(logs_dir, "reports.csv"))
    metrics_json = st.text_input("Выход метрик JSON", value=os.path.join(logs_dir, "metrics.json"))
    equity_png = st.text_input("PNG с equity", value=os.path.join(logs_dir, "equity.png"))
    signals_csv = st.text_input("Файл сигналов (realtime)", value=os.path.join(logs_dir, "signals.csv"))
    realtime_log = st.text_input("Лог realtime", value=os.path.join(logs_dir, "realtime.log"))
    realtime_pid = st.text_input("PID-файл realtime", value=os.path.join(".run", "rt_signaler.pid"))

tabs = st.tabs([
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
])
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
        st.metric("PNL total", f"{pnl_total:.2f}" if isinstance(pnl_total, (int, float)) else "—")
        st.metric("Sharpe", f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "—")
        st.metric("Max Drawdown", f"{maxdd:.4f}" if isinstance(maxdd, (int, float)) else "—")
    with col2:
        running = background_running(realtime_pid)
        st.metric("Realtime сигналер", "запущен" if running else "остановлен")
        sig_df = read_csv(signals_csv, n=1)
        last_sig_ts = int(sig_df.iloc[-1]["ts_ms"]) if not sig_df.empty and "ts_ms" in sig_df.columns else None
        st.metric("Последний сигнал ts_ms", str(last_sig_ts) if last_sig_ts else "—")
    with col3:
        rep_df = read_csv(reports_path, n=1)
        last_eq = float(rep_df.iloc[-1]["equity"]) if not rep_df.empty and "equity" in rep_df.columns else None
        st.metric("Equity (последняя точка)", f"{last_eq:.2f}" if isinstance(last_eq, (int, float)) else "—")

    st.divider()
    st.subheader("Equity (если есть)")
    if os.path.exists(equity_png):
        st.image(equity_png, caption="Equity curve")
    else:
        st.info("Файл equity.png пока не найден. Сгенерируйте через раздел Evaluate.")

# --------------------------- Tab: Ingest ---------------------------

with tabs[1]:
    st.subheader("Публичный Ingest (orchestrator)")
    st.caption("Читает configs/ingest.yaml и запускает полный цикл: klines → aggregate → funding/mark → prices.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Запустить ingest", type="primary"):
            rc = run_cmd([sys.executable, "scripts/ingest_orchestrator.py", "--config", cfg_ingest],
                         log_path=os.path.join(logs_dir, "ingest.log"))
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
    features_out = st.text_input("Выход features.parquet", value="data/features.parquet")
    lookbacks = st.text_input("Окна SMA/ret (через запятую)", value="5,15,60")
    rsi_period = st.number_input("RSI period", min_value=2, max_value=200, value=14, step=1)

    if st.button("Собрать фичи (make_features.py)", type="primary"):
        rc = run_cmd([
            sys.executable, "scripts/make_features.py",
            "--in", prices_in,
            "--out", features_out,
            "--lookbacks", lookbacks,
            "--rsi-period", str(int(rsi_period)),
        ], log_path=os.path.join(logs_dir, "features.log"))
        if rc == 0:
            st.success("Фичи собраны")
        else:
            st.error(f"make_features завершился с кодом {rc}")

# --------------------------- Tab: Training Table ---------------------------

with tabs[3]:
    st.subheader("Training table (сбор меток и merge asof)")
    st.caption("Вызов вашего scripts/build_training_table.py (аргументы подставьте под ваш проект).")

    bt_base = st.text_input("features base (--base)", value="data/features.parquet")
    bt_prices = st.text_input("prices (--prices)", value="data/prices.parquet")
    bt_price_col = st.text_input("price col (--price-col)", value="price")
    bt_decision_delay = st.number_input("decision_delay_ms", min_value=0, value=500, step=100)
    bt_horizon = st.number_input("label_horizon_ms", min_value=60_000, value=3_600_000, step=60_000)
    bt_out = st.text_input("Выход train.parquet (--out)", value="data/train.parquet")

    extra_sources = st.text_area(
        "Доп. источники (--sources, JSON-список объектов) (опционально)",
        value='',
        placeholder='Например: [{"name":"funding","path":"data/futures/BTCUSDT_funding.parquet","time_col":"ts_ms","keys":["symbol"],"direction":"backward","tolerance_ms":86400000}]'
    )

    if st.button("Собрать training table", type="primary"):
        args = [
            sys.executable, "scripts/build_training_table.py",
            "--base", bt_base,
            "--prices", bt_prices,
            "--price-col", bt_price_col,
            "--decision-delay-ms", str(int(bt_decision_delay)),
            "--label-horizon-ms", str(int(bt_horizon)),
            "--out", bt_out,
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
            out = run_backtest_from_yaml(cfg_sandbox, default_rep)
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
    artifacts_dir_eval = st.text_input("Каталог артефактов", value=os.path.join(logs_dir, "eval"))
    if st.button("Посчитать метрики", type="primary"):
        cfg_eval = EvalConfig(
            trades_csv=trades_path,
            equity_csv=reports_path,
            artifacts_dir=artifacts_dir_eval,
        )
        try:
            svc = ServiceEval(cfg_eval)
            res = svc.run()
            pd.Series(res["metrics"]).to_json(metrics_json, force_ascii=False)
            st.success(f"Метрики готовы, отчёт: {res['report_path']}")
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

    with st.expander("Параметры стратегии (сохранить в configs/realtime.yaml)", expanded=False):
        try:
            import copy
            rt_cfg = load_config(cfg_realtime).model_dump()
            st.write("Текущая стратегия:")
            strat = rt_cfg.get("strategy", {}) or {}
            st.code(json.dumps(strat, ensure_ascii=False, indent=2), language="json")
            model_path = st.text_input("Путь к модели (strategy.params.model_path)", value=str(strat.get("params", {}).get("model_path", "")))
            thr = st.text_input("Порог (strategy.params.threshold)", value=str(strat.get("params", {}).get("threshold", "0.0")))
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

    running = background_running(realtime_pid)
    st.write(f"Статус: **{'запущен' if running else 'остановлен'}**")
    cols = st.columns(3)
    with cols[0]:
        if st.button("Старт", disabled=running, type="primary"):
            try:
                pid = start_background(
                    [sys.executable, "scripts/run_realtime_signaler.py", "--config", cfg_realtime],
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
                    bar = Bar(symbol="BTCUSDT", ts=i, open=0.0, high=0.0, low=0.0, close=100.0 + i, volume_base=1.0)
                    provider.on_bar(bar)
                    yield {"ts_ms": i, "symbol": "BTCUSDT"}

        class DummyFP:
            def warmup(self):
                pass

            def on_bar(self, bar):
                return {"ref_price": float(bar.close)}

        class DummyStrat:
            def on_features(self, feats):
                pass

            def decide(self, ctx):
                return []

        try:
            adapter = DummyAdapter()
            runner = ServiceSignalRunner(adapter, DummyFP(), DummyStrat(), None, RunnerConfig())
            list(runner.run())
            st.success("ServiceSignalRunner выполнен (демо)")
        except Exception as e:
            st.error(str(e))

with tabs[7]:
    st.subheader("Очередь на исполнение (manual approve)")

    st.caption("Берём последние сигналы из logs/signals.csv. Можно Approve/Reject, экспортировать подтверждённые.")
    max_rows = st.number_input("Сколько последних строк показывать", min_value=10, max_value=5000, value=200, step=10)
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
        st.caption("Формат файла approved/rejected совпадает с logs/signals.csv + колонка uid")

    if sig_df.empty:
        st.info("Сигналов пока нет.")
    else:
        # покажем таблицу
        st.dataframe(sig_df, use_container_width=True, height=400)

        st.divider()
        st.subheader("Approve / Reject (по одному сигналу)")
        uid = st.text_input("UID сигнала для действия (см. колонку uid)", value=str(sig_df.iloc[-1]["uid"]))
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Approve выбранный UID", type="primary"):
                row = sig_df[sig_df["uid"] == uid]
                if row.empty:
                    st.error("UID не найден в текущей выборке.")
                else:
                    r = row.iloc[-1].to_dict()
                    header = list(sig_df.columns)
                    append_row_csv(approved_path, header, [r.get(c, "") for c in header])
                    st.success("Добавлено в approved.")
        with action_cols[1]:
            if st.button("Reject выбранный UID", type="secondary"):
                row = sig_df[sig_df["uid"] == uid]
                if row.empty:
                    st.error("UID не найден в текущей выборке.")
                else:
                    r = row.iloc[-1].to_dict()
                    header = list(sig_df.columns)
                    append_row_csv(rejected_path, header, [r.get(c, "") for c in header])
                    st.success("Добавлено в rejected.")

        st.divider()
        st.subheader("Экспорт подтверждённых сигналов")
        try:
            ap_df = read_csv(approved_path, n=10_000)
            if not ap_df.empty:
                csv_bytes = ap_df.to_csv(index=False).encode("utf-8")
                st.download_button("Скачать approved CSV", data=csv_bytes, file_name="signals_approved.csv", mime="text/csv")
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
    st.subheader("Полный прогон (ingest → features → train table → backtest → evaluate)")
    st.caption("Один клик запускает весь конвейер. Параметры ниже можно откорректировать.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Ingest / Features**")
        prices_in = st.text_input("prices (вход, после orchestrator)", value="data/prices.parquet", key="full_prices_in")
        features_out = st.text_input("features (выход make_features.py)", value="data/features.parquet", key="full_features_out")
        lookbacks = st.text_input("Окна SMA/ret (через запятую)", value="5,15,60", key="full_lookbacks")
        rsi_period = st.number_input("RSI period", min_value=2, max_value=200, value=14, step=1, key="full_rsi")

    with colB:
        st.markdown("**Training Table**")
        bt_base = st.text_input("features base (--base)", value="data/features.parquet", key="full_bt_base")
        bt_prices = st.text_input("prices (--prices)", value="data/prices.parquet", key="full_bt_prices")
        bt_price_col = st.text_input("price col (--price-col)", value="price", key="full_bt_price_col")
        bt_decision_delay = st.number_input("decision_delay_ms", min_value=0, value=500, step=100, key="full_decision_delay")
        bt_horizon = st.number_input("label_horizon_ms", min_value=60_000, value=3_600_000, step=60_000, key="full_horizon")
        bt_out = st.text_input("Выход train.parquet (--out)", value="data/train.parquet", key="full_bt_out")

    st.markdown("**Evaluate**")
    out_md = st.text_input("Выход markdown", value=os.path.join(logs_dir, "metrics.md"), key="full_out_md")

    st.markdown("**Realtime (опционально)**")
    start_rt = st.checkbox("Запустить realtime сигналер после завершения", value=False, key="full_start_rt")

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

    st.caption("Выбери один из режимов: (A) запуск твоего train_model_multi_patch (произвольная команда), (B) baseline-тренер (опционально). После обучения можно записать путь к модели в configs/realtime.yaml.")

    mode = st.radio("Режим обучения", options=["A) Мой скрипт (train_model_multi_patch)", "B) Baseline тренер"], index=0, key="mt_mode")

    # -------------------- Режим A: кастомный train_model_multi_patch --------------------
    if mode.startswith("A"):
        st.markdown("### Режим A — запуск твоего train_model_multi_patch")

        st.caption("Введи точную команду запуска. Примеры: "
                   "`python scripts/train_model_multi_patch.py --config configs/train.yaml` "
                   "или `python train_model_multi_patch.py` "
                   "или любая другая подходящая команда.")
        custom_cmd = st.text_input("Команда запуска обучения", value="python scripts/train_model_multi_patch.py", key="mt_custom_cmd")
        custom_log = os.path.join(logs_dir, "train_custom.log")
        st.caption(f"Лог обучения: `{custom_log}`")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Запустить мой тренер", type="primary", key="mt_run_custom"):
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
            if st.button("Показать лог кастомного обучения", key="mt_show_custom_log"):
                st.code(tail_file(custom_log, n=500) or "(пусто)")

        st.divider()
        st.markdown("### Указать артефакт модели и записать в configs/realtime.yaml")
        model_art = st.text_input("Путь к готовому файлу модели (.pkl или др.)", value="artifacts/model.pkl", key="mt_art_path_a")

        set_cols = st.columns(2)
        with set_cols[0]:
            if st.button("Записать model_path в configs/realtime.yaml", type="primary", key="mt_set_model_a"):
                try:
                    import copy
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

    # -------------------- Режим B: baseline тренер (опционально) --------------------
    else:
        st.markdown("### Режим B — baseline-тренер (опционально)")
        st.caption("Это универсальный тренер на sklearn. Используй его только если тебе нужен простой эталон или быстрый старт. "
                   "Если тебе достаточно твоего train_model_multi_patch — этот раздел можно не использовать.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Параметры обучения**")
            train_data = st.text_input("Путь к train.parquet/csv (--data)", value="data/train.parquet", key="mt_data_b")
            price_col = st.text_input("Колонка цены (--price-col)", value="ref_price", key="mt_price_b")
            label_col = st.text_input("Колонка таргета (--label-col)", value="", key="mt_label_b")
            task = st.selectbox("Тип задачи (--task)", options=["classification", "regression"], index=0, key="mt_task_b")
            threshold = st.text_input("Порог для классификации (--threshold)", value="0.0", key="mt_thr_b")
            positive_rule = st.selectbox("Правило положительного класса (--positive-rule)", options=["gt", "ge"], index=0, key="mt_posrule_b")
            drop_cols = st.text_input("Исключить колонки (--drop-cols, через запятую)", value="", key="mt_dropcols_b")
            prefixes = st.text_input("Оставлять префиксы фич (--features-prefixes)", value="sma_,ret_,rsi", key="mt_prefixes_b")
            test_size = st.number_input("Доля валидации (--test-size)", min_value=0.05, max_value=0.9, value=0.2, step=0.05, key="mt_testsize_b")
            random_state = st.number_input("random_state (--random-state)", min_value=0, max_value=1_000_000, value=42, step=1, key="mt_rs_b")

        with col2:
            st.markdown("**Модель и выходы**")
            model_type = st.selectbox("Тип модели (--model)", options=["logreg", "rf_cls", "ridge", "rf_reg"], index=0, key="mt_model_b")
            out_dir = st.text_input("Каталог для артефактов (--out-dir)", value="artifacts", key="mt_outdir_b")
            model_name = st.text_input("Имя файла модели (--model-name)", value="model.pkl", key="mt_modelname_b")
            metrics_json = st.text_input("Куда сохранить метрики (--metrics-json)", value=os.path.join(logs_dir, "train_metrics.json"), key="mt_metricsjson_b")
            train_log = os.path.join(logs_dir, "train.log")
            st.caption(f"Лог обучения (baseline): `{train_log}`")

        run_cols = st.columns(2)
        with run_cols[0]:
            if st.button("Запустить baseline-тренер", type="primary", key="mt_run_b"):
                class DummyTrainer:
                    def __init__(self, mtype: str):
                        self.mtype = mtype
                        self.info: Dict[str, Any] = {}

                    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
                        self.info = {"model": self.mtype, "n": len(X)}

                    def save(self, path: str) -> str:
                        _ensure_dir(path)
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(self.info, f)
                        return path

                spec = FeatureSpec(lookbacks_prices=[5, 15, 60], rsi_period=14)
                fp = OfflineFeaturePipe(spec, price_col=price_col, label_col=(label_col or None))
                trainer = DummyTrainer(model_type)
                fmt = "parquet" if train_data.endswith(".parquet") else "csv"
                cfg_train = TrainConfig(
                    input_path=train_data,
                    input_format=fmt,
                    artifacts_dir=out_dir,
                    dataset_name="train_dataset",
                    model_name=model_name,
                )
                try:
                    svc = ServiceTrain(fp, trainer, cfg_train)
                    res = svc.run()
                    pd.Series({"n_samples": res["n_samples"], "n_features": res["n_features"]}).to_json(metrics_json)
                    st.success(f"Обучение завершено, модель: {res['model_path']}")
                except Exception as e:
                    st.error(str(e))

        with run_cols[1]:
            if st.button("Показать train_metrics.json (baseline)", key="mt_showmetrics_b"):
                mj = read_json(metrics_json)
                if mj:
                    st.json(mj)
                else:
                    st.info("Файл метрик пока не найден.")

        st.divider()
        st.subheader("Выбор baseline-модели и запись в configs/realtime.yaml")
        try:
            pkls = []
            if os.path.exists(out_dir) and os.path.isdir(out_dir):
                for name in os.listdir(out_dir):
                    if name.lower().endswith(".pkl"):
                        pkls.append(os.path.join(out_dir, name))
            pkls = sorted(pkls)
        except Exception:
            pkls = []

        chosen = st.selectbox("Выберите модельный .pkl", options=pkls if pkls else ["(нет файлов .pkl в каталоге артефактов)"], index=0, key="mt_choose_b")
        if pkls:
            st.caption(f"Выбрано: `{chosen}`")

        set_cols_b = st.columns(2)
        with set_cols_b[0]:
            if st.button("Записать model_path в configs/realtime.yaml (baseline)", type="primary", key="mt_set_model_b"):
                try:
                    import copy
                    rt_cfg = load_config(cfg_realtime).model_dump()
                    new_cfg = copy.deepcopy(rt_cfg)
                    new_cfg.setdefault("strategy", {}).setdefault("params", {})
                    new_cfg["strategy"]["params"]["model_path"] = str(chosen) if pkls else ""
                    _ensure_dir(cfg_realtime)
                    with open(cfg_realtime, "w", encoding="utf-8") as wf:
                        yaml.safe_dump(new_cfg, wf, sort_keys=False, allow_unicode=True)
                    st.success("Путь к модели записан в configs/realtime.yaml")
                except Exception as e:
                    st.error(f"Ошибка записи в configs/realtime.yaml: {e}")

        with set_cols_b[1]:
            if st.button("Показать текущий configs/realtime.yaml (baseline)", key="mt_show_rt_yaml_b"):
                try:
                    with open(cfg_realtime, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="yaml")
                except Exception as e:
                    st.error(str(e))

with tabs[11]:
    st.subheader("YAML-редактор конфигов проекта")

    st.caption("Редактируйте и сохраняйте конфиги: ingest.yaml, sandbox.yaml, sim.yaml, realtime.yaml. Есть проверка синтаксиса YAML и базовая валидация ключей.")

    files = {
        "configs/ingest.yaml": cfg_ingest,
        "configs/sandbox.yaml": cfg_sandbox,
        "configs/sim.yaml": cfg_sim,
        "configs/realtime.yaml": cfg_realtime,
    }

    col_top = st.columns(2)
    with col_top[0]:
        choice = st.selectbox(
            "Выберите файл для редактирования",
            options=list(files.keys()),
            index=0,
            key="yaml_editor_choice",
        )
    with col_top[1]:
        path = files.get(choice, choice)
        st.text_input("Полный путь к выбранному файлу", value=path, key="yaml_editor_path", disabled=True)

    # читаем содержимое
    initial_text = ""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                initial_text = f.read()
        else:
            initial_text = "# файл не существует — будет создан при сохранении\n"
    except Exception as e:
        initial_text = f"# ошибка чтения файла: {e}\n"

    content = st.text_area("Содержимое YAML", value=initial_text, height=500, key="yaml_editor_content")

    col_actions = st.columns(3)
    with col_actions[0]:
        if st.button("Проверить YAML", type="primary", key="yaml_editor_check"):
            try:
                fname = os.path.basename(path).lower()
                if not content.strip():
                    data = {}
                elif fname == "ingest.yaml":
                    data = parse_ingest_config(content).model_dump()
                elif fname == "sandbox.yaml":
                    data = parse_sandbox_config(content).model_dump()
                else:
                    data = load_config_from_str(content).model_dump()
                st.success("YAML синтаксически корректен")
                # базовая валидация по типичным ключам
                issues = []

                if fname == "ingest.yaml":
                    required = ["symbols", "market", "intervals", "period", "paths"]
                    for k in required:
                        if not isinstance(data, dict) or k not in data:
                            issues.append(f"нет ключа '{k}'")
                elif fname == "sandbox.yaml":
                    required = ["mode", "symbol", "sim_config_path", "strategy", "data"]
                    for k in required:
                        if not isinstance(data, dict) or k not in data:
                            issues.append(f"нет ключа '{k}'")
                elif fname == "sim.yaml":
                    soft_any = any(k in (data or {}) for k in ["fees", "slippage", "risk", "pnl", "leakguard"])
                    if not soft_any:
                        issues.append("не найден ни один из ожидаемых блоков: fees/slippage/risk/pnl/leakguard")
                elif fname == "realtime.yaml":
                    required = ["market", "symbols", "interval", "strategy", "out_csv"]
                    for k in required:
                        if not isinstance(data, dict) or k not in data:
                            issues.append(f"нет ключа '{k}'")

                if issues:
                    st.warning("Базовая проверка структуры: замечания ниже")
                    for it in issues:
                        st.write(f"- {it}")
                else:
                    st.info("Базовая проверка структуры: ок")
            except Exception as e:
                st.error(f"Ошибка синтаксиса YAML: {e}")

    with col_actions[1]:
        if st.button("Сохранить", type="secondary", key="yaml_editor_save"):
            try:
                fname = os.path.basename(path).lower()
                if content.strip():
                    if fname == "ingest.yaml":
                        parse_ingest_config(content)
                    elif fname == "sandbox.yaml":
                        parse_sandbox_config(content)
                    else:
                        load_config_from_str(content)
                _ensure_dir(path)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                st.success(f"Сохранено: {path}")
            except Exception as e:
                st.error(f"Не удалось сохранить: {e}")

    with col_actions[2]:
        if st.button("Открыть текущий файл", key="yaml_editor_open"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="yaml")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Подсказки по ключам (необязательные)")
    with st.expander("ingest.yaml — ожидаемые ключи", expanded=False):
        st.code("""symbols: ["BTCUSDT", "ETHUSDT"]
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
""", language="yaml")

    with st.expander("sandbox.yaml — ожидаемые ключи", expanded=False):
        st.code("""mode: "backtest"
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
""", language="yaml")

    with st.expander("sim.yaml — примерные блоки", expanded=False):
        st.code("""fees:
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
""", language="yaml")

    with st.expander("realtime.yaml — ожидаемые ключи", expanded=False):
        st.code("""market: "futures"
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
""", language="yaml")


with tabs[12]:
    st.subheader("Sim Settings — тонкая настройка симулятора (configs/sim.yaml)")

    st.caption("Редактируйте параметры симулятора через форму. Неизвестные ключи в sim.yaml будут сохранены без изменений. "
               "Это влияет на бэктест песочницы (run_sandbox.py).")

    # читаем текущий sim.yaml
    current = {}
    try:
        if os.path.exists(cfg_sim):
            current = load_config(cfg_sim).model_dump()
    except Exception as e:
        st.error(f"Не удалось прочитать {cfg_sim}: {e}")
        current = {}

    fees = current.get("fees", {}) or {}
    slippage = current.get("slippage", {}) or {}
    pnl = current.get("pnl", {}) or {}
    leakguard = current.get("leakguard", {}) or {}
    risk = current.get("risk", {}) or {}

    st.markdown("### Комиссии (fees)")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        maker_bps = st.number_input("maker_bps (bps)", min_value=0.0, value=float(fees.get("maker_bps", 1.0)), step=0.1)
    with col_f2:
        taker_bps = st.number_input("taker_bps (bps)", min_value=0.0, value=float(fees.get("taker_bps", 5.0)), step=0.1)

    st.markdown("### Слиппедж/спред (slippage)")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        k_slip = st.number_input("k (множитель слиппеджа)", min_value=0.0, value=float(slippage.get("k", 0.8)), step=0.1)
    with col_s2:
        default_spread_bps = st.number_input("default_spread_bps (bps)", min_value=0.0, value=float(slippage.get("default_spread_bps", 3.0)), step=0.5)
    with col_s3:
        min_half_spread_bps = st.number_input("min_half_spread_bps (bps)", min_value=0.0, value=float(slippage.get("min_half_spread_bps", 0.0)), step=0.1)

    st.markdown("### PnL")
    mark_to = st.selectbox("mark_to", options=["side", "mid", "close"], index=["side", "mid", "close"].index(str(pnl.get("mark_to", "side")) if pnl.get("mark_to") in ["side", "mid", "close"] else "side"))

    st.markdown("### Leakguard (защита от утечек времени)")
    decision_delay_ms = st.number_input("decision_delay_ms (мс)", min_value=0, value=int(leakguard.get("decision_delay_ms", 500)), step=50)

    st.markdown("### Риск-менеджмент")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        max_order_notional = st.number_input("max_order_notional", min_value=0.0, value=float(risk.get("max_order_notional", 200.0)), step=10.0, format="%.6f")
    with col_r2:
        max_abs_position_notional = st.number_input("max_abs_position_notional", min_value=0.0, value=float(risk.get("max_abs_position_notional", 1000.0)), step=10.0, format="%.6f")
    with col_r3:
        max_orders_per_min = st.number_input("max_orders_per_min", min_value=0, value=int(risk.get("max_orders_per_min", 10)), step=1)

    st.divider()
    col_act1, col_act2, col_act3 = st.columns(3)

    with col_act1:
        if st.button("Сохранить в configs/sim.yaml", type="primary", key="sim_save"):
            try:
                # обновим только известные секции, остальные ключи сохраним
                new_current = dict(current) if isinstance(current, dict) else {}
                new_current["fees"] = {
                    "maker_bps": float(maker_bps),
                    "taker_bps": float(taker_bps),
                }
                new_current["slippage"] = {
                    "k": float(k_slip),
                    "default_spread_bps": float(default_spread_bps),
                    "min_half_spread_bps": float(min_half_spread_bps),
                }
                new_current["pnl"] = {
                    "mark_to": str(mark_to),
                }
                new_current["leakguard"] = {
                    "decision_delay_ms": int(decision_delay_ms),
                }
                new_current["risk"] = {
                    "max_order_notional": float(max_order_notional),
                    "max_abs_position_notional": float(max_abs_position_notional),
                    "max_orders_per_min": int(max_orders_per_min),
                }

                _ensure_dir(cfg_sim)
                with open(cfg_sim, "w", encoding="utf-8") as f:
                    yaml.safe_dump(new_current, f, sort_keys=False, allow_unicode=True)
                st.success(f"Сохранено: {cfg_sim}")
            except Exception as e:
                st.error(f"Не удалось сохранить {cfg_sim}: {e}")

    with col_act2:
        if st.button("Показать текущий sim.yaml", key="sim_show"):
            try:
                with open(cfg_sim, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="yaml")
            except Exception as e:
                st.error(str(e))

    with col_act3:
        if st.button("Проверить структуру", key="sim_check"):
            issues = []
            try:
                data = load_config(cfg_sim).model_dump()
                # мягкая проверка ожидаемых ключей
                for sect in ["fees", "slippage", "pnl", "leakguard", "risk"]:
                    if sect not in data:
                        issues.append(f"нет блока '{sect}'")
                if "fees" in data:
                    for k in ["maker_bps", "taker_bps"]:
                        if k not in (data["fees"] or {}):
                            issues.append(f"fees: нет ключа '{k}'")
                if "slippage" in data:
                    for k in ["k", "default_spread_bps", "min_half_spread_bps"]:
                        if k not in (data["slippage"] or {}):
                            issues.append(f"slippage: нет ключа '{k}'")
                if "pnl" in data and "mark_to" not in (data["pnl"] or {}):
                    issues.append("pnl: нет ключа 'mark_to'")
                if "leakguard" in data and "decision_delay_ms" not in (data["leakguard"] or {}):
                    issues.append("leakguard: нет ключа 'decision_delay_ms'")
                if "risk" in data:
                    for k in ["max_order_notional", "max_abs_position_notional", "max_orders_per_min"]:
                        if k not in (data["risk"] or {}):
                            issues.append(f"risk: нет ключа '{k}'")
                if issues:
                    st.warning("Замечания по структуре:")
                    for it in issues:
                        st.write(f"- {it}")
                else:
                    st.success("Структура sim.yaml выглядит корректной")
            except Exception as e:
                st.error(f"Ошибка проверки: {e}")

    st.divider()
    st.subheader("Подсказка: связь параметров с симуляцией")
    st.markdown(
        "- **fees.maker_bps / taker_bps**: комиссии (базисные пункты). Влияют на цену исполнения и PnL.\n"
        "- **slippage.k**: множитель функции слиппеджа. Чем выше — тем дороже исполнение при той же волатильности/ликвидности.\n"
        "- **slippage.default_spread_bps**: базовый спред, используется когда нет bid/ask. С динамическими котировками служит нижней гранью.\n"
        "- **slippage.min_half_spread_bps**: минимальный half-spread для защиты от нереалистично узких спредов.\n"
        "- **pnl.mark_to**: расчёт PnL (side = по стороне сделки, mid/close — альтернативы).\n"
        "- **leakguard.decision_delay_ms**: моделирует задержку принятия решения.\n"
        "- **risk.***: ограничения по размеру ордера, общей позиции и частоте сигналов."
    )

with tabs[13]:
    st.subheader("T-cost Calibrate — калибровка модели издержек (base_bps, alpha_vol, beta_illiquidity)")

    st.caption("Форма ниже читает ваш датасет (CSV/Parquet), строит прокси-спред по high/low (или |log return|), "
               "оценивает параметры линейной модели спреда и при желании записывает их в configs/sandbox.yaml → dynamic_spread.")

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

        if vol_mode.lower() == "hl" and ("high" in df.columns) and ("low" in df.columns):
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
                ret = np.abs(np.log(np.where((ref > 0) & (prev > 0), ref / prev, np.nan)))
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

    def _fit_linear(y_bps: np.ndarray, v_bps: np.ndarray, r_liq: np.ndarray, winsor: float):
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
        st.metric("beta_illiquidity (текущий)", f"{float(dyn.get('beta_illiquidity', 1.0)):.6f}")

    st.divider()

    # ---------- UI: форма калибровки ----------
    with st.form("tcost_calib_form"):
        data_path = st.text_input("Путь к датасету (CSV/Parquet)", value=str(current_cfg.get("data", {}).get("path", "data/train.parquet")))
        symbol = st.text_input("Символ (опционально, напр. BTCUSDT)", value=str(current_cfg.get("symbol", "BTCUSDT")))
        ts_col = st.text_input("Колонка времени", value=str(current_cfg.get("data", {}).get("ts_col", "ts_ms")))
        symbol_col = st.text_input("Колонка символа", value=str(current_cfg.get("data", {}).get("symbol_col", "symbol")))
        price_col = st.text_input("Колонка цены-референса", value=str(current_cfg.get("data", {}).get("price_col", "ref_price")))

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            vol_mode = st.selectbox("Источник волатильности", options=["hl", "ret"], index=0 if str((current_cfg.get("dynamic_spread", {}) or {}).get("vol_mode", "hl")).lower() == "hl" else 1)
        with col_b:
            winsor = st.number_input("Винзоризация (доля, 0..0.2)", min_value=0.0, max_value=0.2, step=0.01, value=0.01)
        with col_c:
            dry_run = st.checkbox("Только посмотреть (не записывать YAML)", value=False)

        liq_col = str((current_cfg.get("dynamic_spread", {}) or {}).get("liq_col", "number_of_trades"))
        liq_ref = float((current_cfg.get("dynamic_spread", {}) or {}).get("liq_ref", 1000.0))

        st.caption(f"Ликвидность: liq_col='{liq_col}', liq_ref={liq_ref}")

        submitted = st.form_submit_button("Калибровать")

    if submitted:
        try:
            if not os.path.exists(data_path):
                st.error(f"Файл не найден: {data_path}")
            else:
                df = _read_table(data_path)
                if symbol:
                    df = df.loc[df[symbol_col].astype(str).str.upper() == str(symbol).strip().upper()].copy()
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
                base_bps, alpha_vol, beta_ill, rmse, mae, corr = _fit_linear(y_bps, v_bps, r_liq, winsor)

                st.success("Калибровка выполнена")
                st.json({
                    "base_bps": float(base_bps),
                    "alpha_vol": float(alpha_vol),
                    "beta_illiquidity": float(beta_ill),
                    "fit": {"rmse_bps": rmse, "mae_bps": mae, "corr": corr},
                })

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
                    st.code(yaml.safe_dump({"dynamic_spread": new_dyn}, sort_keys=False, allow_unicode=True), language="yaml")
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
    st.subheader("Target Builder — cost-aware таргет с учётом комиссий и динамического спреда")

    import os
    import pandas as pd
    import yaml
    from training.tcost import effective_return_series

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
        data_path = st.text_input("Входной датасет (CSV/Parquet)", value="data/train.parquet")
        out_path = st.text_input("Куда сохранить (если пусто — рядом с суффиксом _costaware)", value="")
        sandbox_yaml = st.text_input("sandbox.yaml (динамический спред)", value="configs/sandbox.yaml")
        sim_yaml = st.text_input("sim.yaml (комиссии, опционально)", value="configs/sim.yaml")
        fees_bps_total = st.text_input("Комиссия (bps round-trip, опционально — перебивает sim.yaml)", value="")
        horizon_bars = st.number_input("Горизонт (в барах)", min_value=1, step=1, value=60)
        threshold = st.text_input("Порог для бинарной метки (опционально, например 0.0005)", value="")
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
            cols = [c for c in out_df.columns if c.startswith("eff_ret_") or c.startswith("y_eff_")] + ["slippage_bps", "fees_bps_total"]
            st.caption("Добавленные колонки:")
            st.code(", ".join(cols))
        except Exception as e:
            st.error(f"Ошибка: {e}")

with tabs[15]:
    st.subheader("No-Trade Mask — убрать или занулить запрещённые окна в данных")

    import os
    import pandas as pd
    from training.no_trade import compute_no_trade_mask

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
        data_path = st.text_input("Входной датасет (после Target Builder)", value="data/train_costaware.parquet")
        out_path = st.text_input("Куда сохранить (если пусто — суффикс _masked)", value="")
        sandbox_yaml = st.text_input("sandbox.yaml (правила no_trade)", value="configs/sandbox.yaml")
        ts_col = st.text_input("Колонка времени", value="ts_ms")
        mode = st.selectbox("Режим", options=["drop", "weight"], index=0)
        submitted = st.form_submit_button("Применить маску")

    if submitted:
        try:
            if not os.path.exists(data_path):
                st.error(f"Файл не найден: {data_path}")
                st.stop()
            df = _read_table_mask(data_path)
            mask_block = compute_no_trade_mask(df, sandbox_yaml_path=sandbox_yaml, ts_col=ts_col)

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
            st.success(f"Готово. Всего строк: {total}. Запрещённых (no_trade): {blocked}. Вышло: {kept}.")
            if mode == "weight":
                z = int((out_df.get('train_weight', pd.Series(dtype=float)) == 0.0).sum())
                st.info(f"Назначено train_weight=0 для {z} строк.")
        except Exception as e:
            st.error(f"Ошибка: {e}")

with tabs[16]:
    st.subheader("Walk-Forward Splits — сплиты с PURGE (горизонт) и EMBARGO (буфер)")

    import os
    import json
    import pandas as pd
    import yaml
    from training.splits import make_walkforward_splits

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
        data_path = st.text_input("Входной датасет (после маски)", value="data/train_costaware_masked.parquet")
        out_path = st.text_input("Куда сохранить (если пусто — суффикс _wf)", value="")
        ts_col = st.text_input("Колонка времени", value="ts_ms")
        symbol_col = st.text_input("Колонка символа (если есть)", value="symbol")
        interval_ms = st.text_input("Интервал бара, мс (опционально, иначе оценим)", value="")
        col1, col2, col3 = st.columns(3)
        with col1:
            train_span_bars = st.number_input("train_span_bars", min_value=1, step=1, value=7*24*60)
        with col2:
            val_span_bars = st.number_input("val_span_bars", min_value=1, step=1, value=24*60)
        with col3:
            step_bars = st.number_input("step_bars", min_value=1, step=1, value=24*60)
        col4, col5 = st.columns(2)
        with col4:
            horizon_bars = st.number_input("horizon_bars (PURGE)", min_value=1, step=1, value=60)
        with col5:
            embargo_bars = st.number_input("embargo_bars (EMBARGO)", min_value=0, step=1, value=5)
        manifest_dir = st.text_input("Куда писать манифест (JSON/YAML)", value="logs/walkforward")
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
            st.info(f"Всего строк: {total}. В сплитах train: {n_train}, val: {n_val}, вне окон: {total - used}.")
            st.caption("Манифесты записаны:")
            st.code(json_path)
            st.code(yaml_path)
        except Exception as e:
            st.error(f"Ошибка: {e}")


with tabs[17]:
    st.subheader("Threshold Tuner — подбор порога под целевую частоту с учётом кулдауна и no-trade")

    import os
    import pandas as pd
    from training.threshold_tuner import TuneConfig, tune_threshold, load_min_signal_gap_s_from_yaml

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
        data_path = st.text_input("Датасет предсказаний (CSV/Parquet):", value="data/val_predictions.parquet")
        score_col = st.text_input("Колонка со скором/вероятностью:", value="score")
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.text_input("Колонка бинарной метки (если классификация):", value="y_eff_60")
        with col2:
            ret_col = st.text_input("Колонка эффективного ретёрна (если регрессия):", value="eff_ret_60")

        ts_col = st.text_input("Колонка времени:", value="ts_ms")
        symbol_col = st.text_input("Колонка символа:", value="symbol")
        direction = st.selectbox("Правило сигнала", options=["greater", "less"], index=0)

        col3, col4, col5 = st.columns(3)
        with col3:
            target_signals_per_day = st.number_input("Желаемые сигналы/день", min_value=0.1, step=0.1, value=1.5)
        with col4:
            tolerance = st.number_input("Допуск по частоте", min_value=0.0, step=0.1, value=0.5)
        with col5:
            optimize_for = st.selectbox("Метрика оптимизации", options=["sharpe", "precision", "f1"], index=0)

        col6, col7, col8 = st.columns(3)
        with col6:
            min_thr = st.number_input("Минимальный порог", min_value=0.0, max_value=1.0, step=0.01, value=0.50)
        with col7:
            max_thr = st.number_input("Максимальный порог", min_value=0.0, max_value=1.0, step=0.01, value=0.99)
        with col8:
            steps = st.number_input("Число шагов сетки", min_value=5, step=1, value=50)

        col9, col10 = st.columns(2)
        with col9:
            realtime_yaml = st.text_input("realtime.yaml (для чтения min_signal_gap_s):", value="configs/realtime.yaml")
        with col10:
            manual_gap = st.text_input("min_signal_gap_s (перебьёт realtime.yaml, опционально):", value="")

        col11, col12 = st.columns(2)
        with col11:
            sandbox_yaml = st.text_input("sandbox.yaml (no_trade):", value="configs/sandbox.yaml")
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
            st.json({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in best.items()})
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
    from training.calibration import (
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
        data_path = st.text_input("Файл с предсказаниями (CSV/Parquet):", value="data/val_predictions.parquet")
        score_col = st.text_input("Колонка со скором/вероятностью:", value="score")
        y_col = st.text_input("Колонка бинарной метки 0/1:", value="y_eff_60")
        filter_val = st.checkbox("Фильтровать wf_role=='val'", value=True)
        wf_role_col = st.text_input("Имя колонки роли (если фильтруем):", value="wf_role")
        method = st.selectbox("Метод калибровки:", options=["platt", "isotonic"], index=0)
        out_model = st.text_input("Куда сохранить калибратор (.json):", value="models/calibrator.json")
        report_csv = st.text_input("Куда сохранить calibration-table (.csv, опционально):", value="")
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
            st.json({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})

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
        data_path2 = st.text_input("Файл с предсказаниями (CSV/Parquet) для применения:", value="data/val_predictions.parquet")
        model_json = st.text_input("JSON калибратора:", value="models/calibrator.json")
        score_col2 = st.text_input("Имя колонки со скором:", value="score")
        out_col = st.text_input("Имя новой колонки для калиброванной вероятности:", value="score_calibrated")
        out_path2 = st.text_input("Куда сохранить (если пусто — суффикс _calibrated):", value="")
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
            s2 = pd.to_numeric(df2[score_col2], errors="coerce").astype(float).to_numpy()
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
    from training.drift import make_baseline, save_baseline_json, load_baseline_json, compute_psi, default_feature_list

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
            base_data = st.text_input("Файл для baseline (обычно валид. срез)", value="data/val_predictions.parquet")
            base_features = st.text_input("Фичи (через запятую, пусто — авто f_* и score)", value="")
            base_bins = st.number_input("Число бинов (числовые фичи)", min_value=2, step=1, value=10)
            base_topk = st.number_input("Top-K категорий", min_value=5, step=1, value=20)
            base_out = st.text_input("Куда сохранить baseline JSON", value="models/drift_baseline.json")
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
                spec = make_baseline(dfb, feats, bins=int(base_bins), top_k_cats=int(base_topk))
                save_baseline_json(spec, base_out)
                st.success(f"Baseline сохранён: {base_out}")
                st.code(", ".join(feats))
            except Exception as e:
                st.error(f"Ошибка создания baseline: {e}")

    with colB:
        st.markdown("### 2) Проверить дрифт")
        with st.form("drift_check_form"):
            cur_data = st.text_input("Текущий датасет (онлайн/последние дни)", value="data/online_last.parquet")
            baseline_json = st.text_input("Baseline JSON", value="models/drift_baseline.json")
            features = st.text_input("Фичи (пусто — из baseline)", value="")
            ts_col = st.text_input("Колонка времени (UTC мс)", value="ts_ms")
            last_days = st.number_input("Сколько последних дней взять", min_value=0, step=1, value=14)
            psi_warn = st.number_input("Порог предупреждения PSI", min_value=0.0, step=0.01, value=0.10)
            psi_alert = st.number_input("Порог алёрта PSI", min_value=0.0, step=0.01, value=0.25)
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
                    dfc = dfc.loc[pd.to_numeric(dfc[ts_col], errors="coerce") >= cutoff].reset_index(drop=True)

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
                    avg_psi = float(res["psi"].replace([np.inf, -np.inf], np.nan).dropna().mean())
                    worst_feature = res.iloc[0]["feature"]
                    worst_psi = float(res.iloc[0]["psi"])
                    st.metric("Средний PSI", f"{avg_psi:.4f}")
                    st.metric("Максимальный PSI", f"{worst_psi:.4f}", help=f"Фича: {worst_feature}")

                    if worst_psi >= psi_alert or avg_psi >= psi_alert:
                        st.error("⚠️ Сильный дрифт: PSI > alert. Рекомендуется переобучение/перекалибровка.")
                    elif worst_psi >= psi_warn or avg_psi >= psi_warn:
                        st.warning("ℹ️ Умеренный дрифт: PSI > warn. Наблюдать, возможно готовить переобучение.")
                    else:
                        st.info("✅ Дрифт незначительный: PSI в норме.")

                st.caption("Таблица PSI (топ-50 по убыванию):")
                st.dataframe(res.head(50))
            except Exception as e:
                st.error(f"Ошибка расчёта PSI: {e}")
