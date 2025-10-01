"""Focused unit tests for :mod:`impl_bar_executor` helper utilities."""

from __future__ import annotations

import importlib
import sys
import types
from decimal import Decimal

import pytest

from api.spot_signals import (
    SpotSignalEconomics,
    SpotSignalEnvelope,
    SpotSignalTargetWeightPayload,
)


def _ensure_stubbed_universe_module() -> None:
    if "services.universe" in sys.modules:
        return
    stub = types.ModuleType("services.universe")
    stub.get_symbols = lambda *args, **kwargs: []
    stub.run = lambda *args, **kwargs: []
    sys.modules["services.universe"] = stub


_ensure_stubbed_universe_module()


impl_bar_executor = importlib.import_module("impl_bar_executor")


BarExecutor = impl_bar_executor.BarExecutor
PortfolioState = impl_bar_executor.PortfolioState
SymbolSpec = impl_bar_executor.SymbolSpec


@pytest.fixture(scope="module")
def executor() -> BarExecutor:
    return BarExecutor(run_id="test-utils", default_equity_usd=1000.0)


def test_extract_payload_supports_envelopes_and_model_dump(executor: BarExecutor) -> None:
    economics = SpotSignalEconomics(
        edge_bps=15.0,
        cost_bps=5.0,
        net_bps=10.0,
        turnover_usd=200.0,
        act_now=True,
    )
    payload = SpotSignalTargetWeightPayload(target_weight=0.6, economics=economics)
    envelope = SpotSignalEnvelope(
        symbol="BTCUSDT",
        bar_close_ms=1,
        expires_at_ms=2,
        payload=payload,
    )

    class Dumping:
        def model_dump(self) -> dict[str, float]:
            return {"delta_weight": 0.1, "normalized": True}

    direct = executor._extract_payload(envelope)
    wrapped = executor._extract_payload({"payload": envelope})
    dumped = executor._extract_payload({"payload": Dumping()})
    rebalance = executor._extract_payload({"rebalance": {"target_weight": 0.3}})

    assert direct == wrapped
    assert direct["target_weight"] == pytest.approx(0.6)
    assert direct["economics"]["turnover_usd"] == pytest.approx(200.0)
    assert dumped == {"delta_weight": 0.1, "normalized": True}
    assert rebalance == {"target_weight": 0.3}


def test_normalize_symbol_specs_and_cli_enrichment() -> None:
    base_specs = {
        "btcusdt": {"min_notional": "10", "step_size": "0.001", "tick_size": "0.01"},
        "adausdt": {"tick_size": "0.0001"},
    }
    cli_input = [
        "ethusdt:min_notional=5,step_size=0.01",
        {"adausdt": {"min_notional": "7"}},
    ]

    built = impl_bar_executor._build_symbol_specs(base_specs, cli_input)
    assert set(built) == {"BTCUSDT", "ETHUSDT", "ADAUSDT"}

    btc_spec = built["BTCUSDT"]
    assert isinstance(btc_spec, SymbolSpec)
    assert btc_spec.min_notional == Decimal("10")
    assert btc_spec.step_size == Decimal("0.001")

    eth_spec = built["ETHUSDT"]
    assert eth_spec.step_size == Decimal("0.01")
    assert eth_spec.min_notional == Decimal("5")

    enriched = impl_bar_executor._maybe_enrich_symbol_specs(
        built,
        ["ethusdt:tick_size=0.005", "xrpusdt:min_notional=12"],
    )
    assert enriched["ETHUSDT"].tick_size == Decimal("0.005")
    assert enriched["ADAUSDT"].tick_size == Decimal("0.0001")
    assert enriched["XRPUSDT"].min_notional == Decimal("12")


def test_normalize_cli_input_parses_strings_and_mappings() -> None:
    normalized = impl_bar_executor._normalize_cli_input(
        [
            "btcusdt:min_notional=15,step_size=0.002",
            {"ethusdt": {"tick_size": "0.05"}},
            {"ltcusdt": {"min_notional": 3, "step_size": 0.1}},
        ]
    )
    assert normalized == {
        "BTCUSDT": {"min_notional": "15", "step_size": "0.002"},
        "ETHUSDT": {"tick_size": "0.05"},
        "LTCUSDT": {"min_notional": 3, "step_size": 0.1},
    }


def test_numeric_helpers_handle_edges() -> None:
    assert impl_bar_executor._clamp_01(-1.5) == 0.0
    assert impl_bar_executor._clamp_01(1.2) == 1.0
    assert impl_bar_executor._round6("0.12345678") == Decimal("0.123457")
    assert impl_bar_executor._round6(0.1234564) == Decimal("0.123456")
    assert impl_bar_executor._as_decimal(0.5) == Decimal("0.5")


def test_collection_normalizers_flatten_and_deduplicate() -> None:
    assert impl_bar_executor._as_list(["a", ["b", "c"], "a"]) == ["a", "b", "c", "a"]
    assert impl_bar_executor._as_list("a, b,,c") == ["a", "b", "c"]
    assert impl_bar_executor._as_set(["x", ["x", "y"], "x"]) == {"x", "y"}


def test_weight_normalizers_accept_multiple_formats() -> None:
    weights = impl_bar_executor._weights_as_dict(
        [
            "btcusdt:0.4,ethusdt=0.25",
            {"adausdt": "0.2"},
            ("xrpusdt", 0.05),
        ]
    )
    assert weights == {
        "BTCUSDT": pytest.approx(0.4),
        "ETHUSDT": pytest.approx(0.25),
        "ADAUSDT": pytest.approx(0.2),
        "XRPUSDT": pytest.approx(0.05),
    }

    normalized = impl_bar_executor._normalize_weights({"btcusdt": 1.2, "ethusdt": -0.5, "adausdt": 0.3})
    assert normalized == {
        "BTCUSDT": pytest.approx(1.0),
        "ETHUSDT": pytest.approx(0.0),
        "ADAUSDT": pytest.approx(0.3),
    }


def test_portfolio_state_methods_preserve_original() -> None:
    state = PortfolioState(symbol="BTCUSDT", weight=0.1, equity_usd=100.0, price=Decimal("100"), ts=1)
    bar_state = state.with_bar(
        types.SimpleNamespace(ts=2),
        Decimal("105"),
    )
    intrabar_state = state.with_intrabar(ts=3, price="110")

    assert state.ts == 1 and state.price == Decimal("100")
    assert bar_state.ts == 2 and bar_state.price == Decimal("105")
    assert intrabar_state.ts == 3 and intrabar_state.price == Decimal("110")


def test_turnover_caps_helpers_clamp_negative_values() -> None:
    payload = {
        "per_symbol": {"bps": -5, "usd": "25", "daily_bps": float("nan")},
        "portfolio": {"daily_usd": -1, "usd": 50},
    }
    sanitized = impl_bar_executor._turnover_caps(payload)

    assert sanitized["per_symbol"]["bps"] == pytest.approx(0.0)
    assert sanitized["per_symbol"]["usd"] == pytest.approx(25.0)
    assert sanitized["per_symbol"]["daily_bps"] == pytest.approx(0.0)
    assert sanitized["portfolio"]["daily_usd"] == pytest.approx(0.0)
    assert sanitized["portfolio"]["usd"] == pytest.approx(50.0)
