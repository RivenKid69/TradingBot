import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from aggregate_exec_logs import aggregate


def _make_row(
    ts: int,
    meta: dict[str, object],
    *,
    price: float | None = None,
    quantity: float = 0.0,
) -> dict[str, object]:
    if price is None:
        price_value = float(meta.get("reference_price", 0.0) or 0.0)
    else:
        price_value = price
    return {
        "ts": ts,
        "run_id": "bar",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "MARKET",
        "price": price_value,
        "quantity": quantity,
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
        "cap_usd": 12_000.0,
        "bar_ts": 60_000,
        "reference_price": 20_000.0,
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
        "cap_usd": 8_000.0,
        "bar_ts": 60_000,
        "reference_price": 19_500.0,
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
    assert row["bar_cap_usd"] == pytest.approx(12_000.0)
    assert row["bar_adv_quote"] == pytest.approx(10_000.0)
    assert row["bar_act_now_rate"] == pytest.approx(0.5)
    assert row["bar_turnover_vs_cap"] == pytest.approx(800.0 / 12_000.0)
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
    assert day["bar_cap_usd"] == pytest.approx(12_000.0)
    assert day["bar_adv_quote"] == pytest.approx(10_000.0)
    assert day["bar_turnover_vs_cap"] == pytest.approx(800.0 / 12_000.0)
    assert "cost_bias_bps" in day.index


def test_aggregate_skips_slippage_for_zero_notional(tmp_path: Path) -> None:
    meta = {
        "mode": "target",
        "decision": {"turnover_usd": 0.0, "act_now": False},
        "target_weight": 0.1,
        "delta_weight": 0.1,
        "adv_quote": 5_000.0,
        "cap_usd": 6_000.0,
        "bar_ts": 30_000,
        "reference_price": 15_000.0,
    }
    trades_df = pd.DataFrame([_make_row(30_000, meta, price=0.0, quantity=0.0)])
    trades_path = tmp_path / "log_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    out_bars = tmp_path / "bars.csv"
    out_days = tmp_path / "days.csv"

    aggregate(str(trades_path), "", str(out_bars), str(out_days), bar_seconds=60)

    bars = pd.read_csv(out_bars)
    assert bars.shape[0] == 1
    row = bars.iloc[0]
    assert pd.isna(row["realized_slippage_bps"])
    assert pd.isna(row["cost_bias_bps"])
    assert row["bar_adv_quote"] == pytest.approx(5_000.0)

    days = pd.read_csv(out_days)
    assert days.shape[0] == 1
    day = days.iloc[0]
    assert pd.isna(day["realized_slippage_bps"])
    assert pd.isna(day["cost_bias_bps"])
    assert day["bar_adv_quote"] == pytest.approx(5_000.0)


def test_aggregate_uses_single_cap_denominator(tmp_path: Path) -> None:
    shared_cap = 10_000.0
    meta_rows = [
        {
            "mode": "target",
            "decision": {"turnover_usd": 1_000.0, "act_now": True},
            "cap_usd": shared_cap,
            "bar_ts": 120_000,
        },
        {
            "mode": "delta",
            "decision": {"turnover_usd": 500.0, "act_now": False},
            "cap_usd": shared_cap,
            "bar_ts": 120_000,
        },
        {
            "mode": "delta",
            "decision": {"turnover_usd": 1_500.0, "act_now": True},
            "cap_usd": shared_cap,
            "bar_ts": 120_000,
        },
    ]

    trades_df = pd.DataFrame(
        [
            _make_row(120_000 + i * 10, meta)
            for i, meta in enumerate(meta_rows)
        ]
    )
    trades_path = tmp_path / "log_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    out_bars = tmp_path / "bars.csv"
    out_days = tmp_path / "days.csv"

    aggregate(str(trades_path), "", str(out_bars), str(out_days), bar_seconds=60)

    expected_turnover = 1_000.0 + 500.0 + 1_500.0
    bars = pd.read_csv(out_bars)
    assert bars.shape[0] == 1
    row = bars.iloc[0]
    assert row["bar_turnover_usd"] == pytest.approx(expected_turnover)
    assert row["bar_cap_usd"] == pytest.approx(shared_cap)
    assert row["bar_turnover_vs_cap"] == pytest.approx(expected_turnover / shared_cap)
    assert pd.isna(row["bar_adv_quote"])

    days = pd.read_csv(out_days)
    assert days.shape[0] == 1
    day = days.iloc[0]
    assert day["bar_turnover_usd"] == pytest.approx(expected_turnover)
    assert day["bar_cap_usd"] == pytest.approx(shared_cap)
    assert day["bar_turnover_vs_cap"] == pytest.approx(expected_turnover / shared_cap)
    assert pd.isna(day["bar_adv_quote"])


def test_aggregate_rejects_non_positive_bar_seconds(tmp_path: Path) -> None:
    trades_df = pd.DataFrame(
        [
            _make_row(
                60_000,
                {
                    "mode": "target",
                    "decision": {"turnover_usd": 100.0, "act_now": True},
                    "reference_price": 10_000.0,
                },
            )
        ]
    )
    trades_path = tmp_path / "log_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    out_bars = tmp_path / "bars.csv"
    out_days = tmp_path / "days.csv"

    with pytest.raises(ValueError) as excinfo:
        aggregate(str(trades_path), "", str(out_bars), str(out_days), bar_seconds=0)

    assert "bar_seconds must be a positive integer" in str(excinfo.value)


def test_cli_rejects_non_positive_bar_seconds(tmp_path: Path) -> None:
    trades_df = pd.DataFrame(
        [
            _make_row(
                60_000,
                {
                    "mode": "target",
                    "decision": {"turnover_usd": 100.0, "act_now": True},
                    "reference_price": 10_000.0,
                },
            )
        ]
    )
    trades_path = tmp_path / "log_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    out_bars = tmp_path / "bars.csv"
    out_days = tmp_path / "days.csv"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aggregate_exec_logs",
            "--trades",
            str(trades_path),
            "--out-bars",
            str(out_bars),
            "--out-days",
            str(out_days),
            "--bar-seconds",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "bar_seconds must be a positive integer" in result.stderr


def test_adv_does_not_fill_missing_cap(tmp_path: Path) -> None:
    meta = {
        "mode": "target",
        "decision": {"turnover_usd": 1_000.0, "act_now": True},
        "adv_quote": 50_000.0,
        "bar_ts": 180_000,
    }

    trades_df = pd.DataFrame([_make_row(180_000, meta)])
    trades_path = tmp_path / "log_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    out_bars = tmp_path / "bars.csv"
    out_days = tmp_path / "days.csv"

    aggregate(str(trades_path), "", str(out_bars), str(out_days), bar_seconds=60)

    bars = pd.read_csv(out_bars)
    assert bars.shape[0] == 1
    row = bars.iloc[0]
    assert pd.isna(row["bar_cap_usd"])
    assert pd.isna(row["bar_turnover_vs_cap"])
    assert row["bar_adv_quote"] == pytest.approx(50_000.0)

    days = pd.read_csv(out_days)
    assert days.shape[0] == 1
    day = days.iloc[0]
    assert pd.isna(day["bar_cap_usd"])
    assert pd.isna(day["bar_turnover_vs_cap"])
    assert day["bar_adv_quote"] == pytest.approx(50_000.0)


def test_aggregate_accepts_string_act_now_column(tmp_path: Path) -> None:
    base_meta = {
        "mode": "target",
        "decision": {"turnover_usd": 200.0, "act_now": False},
        "cap_usd": 5_000.0,
        "bar_ts": 180_000,
    }

    rows: list[dict[str, object]] = []
    act_now_values = ["TRUE", "false", "1", "0"]
    for idx, act_now in enumerate(act_now_values):
        meta = dict(base_meta)
        meta["decision"] = dict(base_meta["decision"])
        meta["decision"]["turnover_usd"] = 200.0 + idx * 50.0
        row = _make_row(180_000 + idx * 5, meta)
        row["act_now"] = act_now
        rows.append(row)

    trades_df = pd.DataFrame(rows)
    trades_path = tmp_path / "log_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    out_bars = tmp_path / "bars.csv"
    out_days = tmp_path / "days.csv"

    aggregate(str(trades_path), "", str(out_bars), str(out_days), bar_seconds=60)

    bars = pd.read_csv(out_bars)
    assert bars.shape[0] == 1
    row = bars.iloc[0]
    assert row["bar_decisions"] == 4
    assert row["bar_act_now"] == 2
    assert row["bar_act_now_rate"] == pytest.approx(0.5)

    days = pd.read_csv(out_days)
    assert days.shape[0] == 1
    day = days.iloc[0]
    assert day["bar_decisions"] == 4
    assert day["bar_act_now"] == 2
    assert day["bar_act_now_rate"] == pytest.approx(0.5)
