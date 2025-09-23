import json
from textwrap import dedent

import pandas as pd

from core_config import load_config
import service_eval
from service_eval import from_config
import di_registry


def _write_profiled_logs(tmp_path):
    trades = pd.DataFrame(
        {
            "ts_ms": [1, 2, 3, 4],
            "pnl": [1.0, -0.5, 2.0, -1.0],
            "side": ["BUY", "SELL", "BUY", "SELL"],
            "qty": [1, 1, 1, 1],
            "execution_profile": [
                "MKT_OPEN_NEXT_H1",
                "MKT_OPEN_NEXT_H1",
                "VWAP_CURRENT_H1",
                "VWAP_CURRENT_H1",
            ],
        }
    )
    equity = pd.DataFrame(
        {
            "ts_ms": [1, 2, 1, 2],
            "equity": [1.0, 0.5, 2.0, 1.0],
            "execution_profile": [
                "MKT_OPEN_NEXT_H1",
                "MKT_OPEN_NEXT_H1",
                "VWAP_CURRENT_H1",
                "VWAP_CURRENT_H1",
            ],
        }
    )
    trades_path = tmp_path / "trades.csv"
    equity_path = tmp_path / "equity.csv"
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    return trades_path, equity_path


def _write_config(tmp_path, trades_path, equity_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        dedent(
            f"""
            mode: eval
            run_id: test-run
            logs_dir: "{tmp_path / 'logs'}"
            artifacts_dir: "{tmp_path / 'artifacts'}"
            execution_profile: MKT_OPEN_NEXT_H1
            components:
              market_data:
                target: "x:y"
                params: {{}}
              executor:
                target: "x:y"
                params: {{}}
              feature_pipe:
                target: "x:y"
                params: {{}}
              policy:
                target: "x:y"
                params: {{}}
              risk_guards:
                target: "x:y"
                params: {{}}
            input:
              trades_path: "{trades_path}"
              equity_path: "{equity_path}"
            """
        ).strip()
    )
    return cfg_path


def test_from_config_filters_to_execution_profile(monkeypatch, tmp_path):
    trades_path, equity_path = _write_profiled_logs(tmp_path)
    cfg_path = _write_config(tmp_path, trades_path, equity_path)

    monkeypatch.setattr(di_registry, "build_graph", lambda components, run_config=None: {})
    monkeypatch.setattr(service_eval, "plot_equity_curve", lambda *args, **kwargs: None)

    cfg = load_config(str(cfg_path))
    metrics = from_config(cfg)

    payload = json.loads(json.dumps(metrics))
    assert set(payload.keys()) == {"equity", "trades"}
    assert payload["trades"]["n_trades"] == 2
