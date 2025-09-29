import json
from pathlib import Path

import pandas as pd
import pytest

from aggregate_exec_logs import aggregate


@pytest.mark.parametrize("bar_seconds", [60, 300])
def test_aggregate_cost_metrics(tmp_path: Path, bar_seconds: int) -> None:
    trades = pd.DataFrame(
        [
            {
                "ts": 1_000,
                "run_id": "order",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "order_type": "MARKET",
                "price": 101.0,
                "quantity": 1.0,
                "fee": 0.0,
                "fee_asset": "USDT",
                "pnl": 0.0,
                "exec_status": "FILLED",
                "liquidity": "TAKER",
                "client_order_id": "a",
                "order_id": "1",
                "execution_profile": "order",
                "market_regime": "NORMAL",
                "meta_json": json.dumps(
                    {
                        "reference_price": 100.0,
                        "decision": {"cost_bps": 80.0},
                    }
                ),
            },
            {
                "ts": 2_000,
                "run_id": "order",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "order_type": "MARKET",
                "price": 99.0,
                "quantity": 2.0,
                "fee": 0.0,
                "fee_asset": "USDT",
                "pnl": 0.0,
                "exec_status": "FILLED",
                "liquidity": "TAKER",
                "client_order_id": "b",
                "order_id": "2",
                "execution_profile": "order",
                "market_regime": "NORMAL",
                "meta_json": json.dumps(
                    {
                        "reference_price": 100.0,
                        "decision": {"cost_bps": 120.0},
                    }
                ),
            },
        ]
    )
    trades_path = tmp_path / "log_trades.csv"
    trades.to_csv(trades_path, index=False)

    out_bars = tmp_path / "bars.csv"
    out_days = tmp_path / "days.csv"
    metrics_md = tmp_path / "metrics.md"

    aggregate(
        str(trades_path),
        "",
        str(out_bars),
        str(out_days),
        bar_seconds=bar_seconds,
        metrics_md=str(metrics_md),
    )

    bars = pd.read_csv(out_bars)
    assert bars.shape[0] == 1
    row = bars.iloc[0]
    realized = row["realized_slippage_bps"]
    modeled = row["modeled_cost_bps"]
    bias = row["cost_bias_bps"]
    assert realized == pytest.approx(100.0)
    assert modeled == pytest.approx(320.0 / 3.0)
    assert bias == pytest.approx(100.0 - 320.0 / 3.0)

    days = pd.read_csv(out_days)
    assert days.shape[0] == 1
    day = days.iloc[0]
    assert day["realized_slippage_bps"] == pytest.approx(realized)
    assert day["modeled_cost_bps"] == pytest.approx(modeled)
    assert day["cost_bias_bps"] == pytest.approx(bias)

    md_text = metrics_md.read_text(encoding="utf-8")
    assert "Execution Costs" in md_text
    assert "realized_slippage_bps" in md_text
    assert "modeled_cost_bps" in md_text
