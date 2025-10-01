"""Focused tests covering BarExecutor payload helpers."""

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
    """Install a lightweight ``services.universe`` stub for unit tests."""

    if "services.universe" in sys.modules:
        return

    stub = types.ModuleType("services.universe")
    stub.get_symbols = lambda *args, **kwargs: []
    stub.run = lambda *args, **kwargs: []
    sys.modules["services.universe"] = stub


_ensure_stubbed_universe_module()


impl_bar_executor = importlib.import_module("impl_bar_executor")


BarExecutor = impl_bar_executor.BarExecutor
SymbolSpec = impl_bar_executor.SymbolSpec


@pytest.fixture()
def executor() -> BarExecutor:
    return BarExecutor(run_id="test")


def test_extract_payload_supports_various_meta_containers(executor: BarExecutor) -> None:
    economics = SpotSignalEconomics(
        edge_bps=12.5,
        cost_bps=1.5,
        net_bps=11.0,
        turnover_usd=250.0,
        act_now=True,
    )
    signal_payload = SpotSignalTargetWeightPayload(
        target_weight=0.55,
        economics=economics,
    )
    envelope = SpotSignalEnvelope(
        symbol="BTCUSDT",
        bar_close_ms=1,
        expires_at_ms=2,
        payload=signal_payload,
    )

    class DumpingPayload:
        def model_dump(self) -> dict[str, float]:
            return {"weight": 0.8, "normalized": True}

    class AttributeContainer:
        def __init__(self, payload: object) -> None:
            self.payload = payload

    dict_result = executor._extract_payload(
        {"payload": {"delta_weight": 0.1, "edge_bps": 9}}
    )
    model_result = executor._extract_payload({"payload": DumpingPayload()})
    envelope_direct = executor._extract_payload(envelope)
    envelope_nested = executor._extract_payload({"payload": envelope})
    attribute_result = executor._extract_payload(AttributeContainer({"delta": 0.2}))
    rebalance_result = executor._extract_payload({"rebalance": {"target": 0.7}})

    assert dict_result == {"delta_weight": 0.1, "edge_bps": 9}
    assert model_result == {"weight": 0.8, "normalized": True}
    assert envelope_direct["kind"] == "target_weight"
    assert envelope_direct["target_weight"] == pytest.approx(0.55)
    assert envelope_direct["economics"]["edge_bps"] == pytest.approx(12.5)
    assert envelope_nested == envelope_direct
    assert attribute_result == {"delta": 0.2}
    assert rebalance_result == {"target": 0.7}


def test_extract_payload_gracefully_handles_invalid_types(executor: BarExecutor) -> None:
    class Unsupported:
        pass

    assert executor._extract_payload({"payload": 5}) == {}
    assert executor._extract_payload(Unsupported()) == {}
    assert executor._extract_payload(None) == {}


def test_normalize_symbol_specs_recognizes_aliases_and_sanitizes(
    executor: BarExecutor,
) -> None:
    class SpecContainer:
        def model_dump(self) -> dict[str, object]:
            return {
                "payload": {"lot_step": "0.02"},
                "details": {"nested": {"tickSize": "0.05"}},
                "min_notional": -12,
            }

    specs = executor._normalize_symbol_specs(
        {
            "btcusdt": {
                "minNotional": "15",
                "filters": {"stepSize": "0.001"},
                "meta": [{"PRICE_TICK": "0.01"}],
            },
            "ethusdt": SpecContainer(),
            "adausdt": {
                "min_notional": "abc",
                "step_size": None,
                "tick_size": -0.1,
            },
            "dogeusdt": 5,
            None: {"min_notional": 1},
            "  ": {"min_notional": 1},
        }
    )

    assert set(specs) == {"BTCUSDT", "ETHUSDT", "ADAUSDT"}

    btc_spec = specs["BTCUSDT"]
    assert isinstance(btc_spec, SymbolSpec)
    assert btc_spec.min_notional == Decimal("15")
    assert btc_spec.step_size == Decimal("0.001")
    assert btc_spec.tick_size == Decimal("0.01")

    eth_spec = specs["ETHUSDT"]
    assert eth_spec.min_notional == Decimal("0")
    assert eth_spec.step_size == Decimal("0.02")
    assert eth_spec.tick_size == Decimal("0.05")

    ada_spec = specs["ADAUSDT"]
    assert ada_spec.min_notional == Decimal("0")
    assert ada_spec.step_size == Decimal("0")
    assert ada_spec.tick_size == Decimal("0")

