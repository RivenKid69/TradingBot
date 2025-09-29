import json
from pathlib import Path

import pandas as pd
import pytest

from aggregate_exec_logs import aggregate


def _make_row(ts: int, meta: dict[str, object]) -> dict[str, object]:
    return {
        "ts": ts,
        "run_id": "bar",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "MARKET",
        "price": 0.0,
        "quantity": 0.0,
        "fee": 0.0,
        "fee_asset": "USDT",
        "pnl": 0.0,
        "exec_status": "CANCELED",
        "liquidity": "UNKNOWN",
        "client_order_id": "",
        "order_id": "",
        "execution_profile": "bar",
        "market_regime": "",
        "meta_json": json.dumps(meta),
    }


def test_aggregate_accepts_bar_mode_logs(tmp_path: Path) -> None:
    meta_first = {
        "mode": "target",
        "decision": {
            "turnover_usd": 500.0,
            "act_now": True,
            "edge_bps": 10.0,
            "cost_bps": 1.0,
            "net_bps": 9.0,
        },
        "target_weight": 0.5,
        "delta_weight": 0.5,
        "adv_quote": 10_000.0,
        "bar_ts": 60_000,
    }
    meta_second = {
        "mode": "delta",
        "decision": {
            "turnover_usd": 300.0,
            "act_now": False,
            "edge_bps": 5.0,
            "cost_bps": 0.5,
            "net_bps": 4.5,
        },
        "target_weight": 0.2,
        "delta_weight": -0.3,
        "adv_quote": 20_000.0,
        "bar_ts": 60_000,
    }

    trades_df = pd.DataFrame(
        [
            _make_row(60_000, meta_first),
            _make_row(60_030, meta_second),
        ]
    )
    trades_path = tmp_path / "log_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    out_bars = tmp_path / "bars.csv"
    out_days = tmp_path / "days.csv"

    aggregate(str(trades_path), "", str(out_bars), str(out_days), bar_seconds=60)

    bars = pd.read_csv(out_bars)
    assert bars.shape[0] == 1
    row = bars.iloc[0]
    assert row["trades"] == 0
    assert row["bar_decisions"] == 2
    assert row["bar_act_now"] == 1
    assert row["bar_turnover_usd"] == pytest.approx(800.0)
    assert row["bar_cap_usd"] == pytest.approx(30_000.0)
    assert row["bar_act_now_rate"] == pytest.approx(0.5)
    assert row["bar_turnover_vs_cap"] == pytest.approx(800.0 / 30_000.0)
    assert "realized_slippage_bps" in row.index
    assert "modeled_cost_bps" in row.index
    assert "cost_bias_bps" in row.index
    assert pd.isna(row["realized_slippage_bps"])
    assert pd.isna(row["modeled_cost_bps"])

    days = pd.read_csv(out_days)
    assert days.shape[0] == 1
    day = days.iloc[0]
    assert day["bar_decisions"] == 2
    assert day["bar_act_now"] == 1
    assert day["bar_turnover_usd"] == pytest.approx(800.0)
    assert day["bar_cap_usd"] == pytest.approx(30_000.0)
    assert day["bar_turnover_vs_cap"] == pytest.approx(800.0 / 30_000.0)
    assert "cost_bias_bps" in day.index
