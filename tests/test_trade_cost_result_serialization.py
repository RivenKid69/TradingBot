from execution_sim import _TradeCostResult


def test_trade_cost_result_to_dict_serializes_fields():
    result = _TradeCostResult(
        bps=12.5,
        mid=100.25,
        base_price=None,
        inputs={"order_qty": 10, "side": "buy"},
        metrics={"slippage": 0.12, "fill_ratio": 0.95},
        expected_spread_bps=None,
        latency_timeout_ratio=0.2,
        execution_profile="aggressive",
        vol_raw={"short": 1, "long": 2.5},
    )

    payload = result.to_dict()

    assert payload == {
        "bps": 12.5,
        "mid": 100.25,
        "base_price": None,
        "inputs": {"order_qty": 10, "side": "buy"},
        "metrics": {"slippage": 0.12, "fill_ratio": 0.95},
        "expected_spread_bps": None,
        "latency_timeout_ratio": 0.2,
        "execution_profile": "aggressive",
        "vol_raw": {"short": 1.0, "long": 2.5},
    }
