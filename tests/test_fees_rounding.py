import pytest

from impl_fees import FeesImpl
from fees import FeeComputation, FeesModel, _sanitize_rounding_config


def test_rounding_config_allows_zero_minimum_and_maximum_fee():
    config = _sanitize_rounding_config({"minimum_fee": 0.0, "maximum_fee": 0.0})

    assert config.get("minimum_fee") == pytest.approx(0.0)
    assert config.get("maximum_fee") == pytest.approx(0.0)


def test_rounding_config_preserves_zero_fee_from_aliases():
    config = _sanitize_rounding_config(
        {
            "minimum_fee": None,
            "min_fee": 0.0,
            "maximum_fee": None,
            "max_fee": 0.0,
        }
    )

    assert config.get("minimum_fee") == pytest.approx(0.0)
    assert config.get("maximum_fee") == pytest.approx(0.0)


def test_rounding_nested_options_normalized():
    fees = FeesImpl.from_dict(
        {
            "rounding": {
                "enabled": True,
                "mode": "STEP",
                "step": "0.0005",
                "per_symbol": {
                    "btcusdt": {"step": 0.001},
                    "ethusdt": {"enabled": False},
                },
            }
        }
    )

    assert fees.model_payload["fee_rounding_step"] == pytest.approx(0.0005)
    rounding_payload = fees.model_payload.get("rounding") or {}
    assert rounding_payload.get("mode") == "step"
    assert rounding_payload.get("step") == pytest.approx(0.0005)
    assert rounding_payload.get("per_symbol", {}).get("BTCUSDT", {}).get("step") == pytest.approx(0.001)

    rounding_meta = fees.metadata.get("rounding") or {}
    assert rounding_meta.get("enabled") is True
    assert rounding_meta.get("step") == pytest.approx(0.0005)
    assert rounding_meta.get("per_symbol", {}).get("BTCUSDT", {}).get("step") == pytest.approx(0.001)
    assert rounding_meta.get("per_symbol", {}).get("ETHUSDT", {}).get("enabled") is False

    expected_rounding = fees.expected_payload.get("rounding") or {}
    assert expected_rounding.get("step") == pytest.approx(0.0005)

    model = fees.model
    assert model is not None
    assert model.fee_rounding_step == pytest.approx(0.0005)
    assert model.rounding.get("step") == pytest.approx(0.0005)
    assert model.rounding.get("per_symbol", {}).get("BTCUSDT", {}).get("step") == pytest.approx(0.001)


def test_rounding_disabled_drops_fee_step():
    fees = FeesImpl.from_dict({"rounding": {"enabled": False, "step": 0.1}})

    assert "fee_rounding_step" not in fees.model_payload
    rounding_payload = fees.model_payload.get("rounding") or {}
    assert rounding_payload.get("enabled") is False
    assert rounding_payload.get("step") is None

    rounding_meta = fees.metadata.get("rounding") or {}
    assert rounding_meta.get("enabled") is False
    assert "fee_rounding_step" not in fees.metadata

    model = fees.model
    assert model is not None
    assert model.fee_rounding_step == pytest.approx(0.0)
    assert model.rounding.get("enabled") is False
    assert "step" not in model.rounding


def test_settlement_options_propagated():
    fees = FeesImpl.from_dict(
        {
            "settlement": {
                "mode": "bnb",
                "currency": "bnb",
                "prefer_discount_asset": True,
            }
        }
    )

    settlement_payload = fees.model_payload.get("settlement") or {}
    assert settlement_payload.get("mode") == "bnb"
    assert settlement_payload.get("currency") == "BNB"
    assert settlement_payload.get("prefer_discount_asset") is True

    settlement_meta = fees.metadata.get("settlement") or {}
    assert settlement_meta.get("currency") == "BNB"
    assert settlement_meta.get("mode") == "bnb"

    expected_settlement = fees.expected_payload.get("settlement") or {}
    assert expected_settlement.get("currency") == "BNB"

    model = fees.model
    assert model is not None
    assert model.settlement.get("currency") == "BNB"
    assert model.settlement.get("prefer_discount_asset") is True


def test_fees_model_rounding_from_nested_only():
    model = FeesModel.from_dict({"rounding": {"step": 0.25}})

    assert model.fee_rounding_step == pytest.approx(0.25)
    assert model.rounding.get("step") == pytest.approx(0.25)
    assert model.rounding.get("enabled") is True


def test_symbol_commission_step_rounds_up():
    model = FeesModel.from_dict(
        {
            "maker_bps": 10,
            "taker_bps": 10,
            "symbol_fee_table": {
                "BTCUSDT": {
                    "maker_bps": 10,
                    "taker_bps": 10,
                    "quantizer": {"commission_step": 0.0001},
                }
            },
        }
    )

    fee = model.compute(
        side="BUY",
        price=100.0,
        qty=0.123456,
        liquidity="maker",
        symbol="BTCUSDT",
    )

    assert fee == pytest.approx(0.0124)


def test_bnb_settlement_converts_with_rounding():
    model = FeesModel.from_dict(
        {
            "maker_bps": 10,
            "taker_bps": 10,
            "settlement": {"mode": "bnb", "currency": "BNB"},
            "symbol_fee_table": {
                "BTCUSDT": {
                    "maker_bps": 10,
                    "taker_bps": 10,
                    "quantizer": {"commission_step": 0.0001},
                }
            },
        }
    )

    # Without conversion rate the result stays in quote currency with step rounding.
    fee_quote = model.compute(
        side="SELL",
        price=100.0,
        qty=0.123456,
        liquidity="taker",
        symbol="BTCUSDT",
    )
    assert fee_quote == pytest.approx(0.0124)

    # When conversion is provided the fee is converted to BNB and rounded up to the tick.
    fee_bnb = model.compute(
        side="SELL",
        price=100.0,
        qty=0.123456,
        liquidity="taker",
        symbol="BTCUSDT",
        bnb_conversion_rate=200.0,
    )
    assert fee_bnb == pytest.approx(0.000062)


def test_fee_compute_return_details():
    model = FeesModel.from_dict(
        {
            "maker_bps": 10,
            "taker_bps": 10,
            "settlement": {"mode": "bnb", "currency": "BNB"},
            "symbol_fee_table": {
                "BTCUSDT": {
                    "maker_bps": 10,
                    "taker_bps": 10,
                    "quantizer": {"commission_step": 0.0001},
                }
            },
        }
    )

    details_quote = model.compute(
        side="BUY",
        price=100.0,
        qty=0.123456,
        liquidity="maker",
        symbol="BTCUSDT",
        return_details=True,
    )

    assert isinstance(details_quote, FeeComputation)
    assert details_quote.fee == pytest.approx(0.0124)
    assert details_quote.fee_before_rounding == pytest.approx(0.0123456)
    assert details_quote.commission_step == pytest.approx(0.0001)
    assert details_quote.rounding_step == pytest.approx(0.0001)
    assert details_quote.rounding_enabled is True
    assert details_quote.use_bnb_settlement is True
    assert details_quote.requires_bnb_conversion is True

    details_bnb = model.compute(
        side="BUY",
        price=100.0,
        qty=0.123456,
        liquidity="maker",
        symbol="BTCUSDT",
        bnb_conversion_rate=200.0,
        return_details=True,
    )

    assert isinstance(details_bnb, FeeComputation)
    assert details_bnb.fee == pytest.approx(0.000062)
    assert details_bnb.bnb_conversion_rate == pytest.approx(200.0)
    assert details_bnb.requires_bnb_conversion is False
    assert details_bnb.commission_step == pytest.approx(0.0000005)
    assert details_bnb.rounding_step == pytest.approx(0.0000005)


def test_bnb_rounding_scales_with_conversion():
    model = FeesModel.from_dict(
        {
            "taker_bps": 100,
            "settlement": {"mode": "bnb", "currency": "BNB"},
            "rounding": {
                "enabled": True,
                "step": 0.01,
                "minimum_fee": 0.01,
                "maximum_fee": 1.0,
            },
        }
    )

    details = model.compute(
        side="SELL",
        price=10.0,
        qty=1.0,
        liquidity="taker",
        bnb_conversion_rate=200.0,
        return_details=True,
    )

    assert isinstance(details, FeeComputation)
    assert details.fee_before_rounding == pytest.approx(0.0005)
    assert details.fee == pytest.approx(0.0005)
    assert details.rounding_step == pytest.approx(0.00005)
    assert details.commission_step == pytest.approx(0.00005)


def test_decimal_rounding_uses_half_up():
    model = FeesModel.from_dict(
        {
            "maker_bps": 0,
            "taker_bps": 250,
            "rounding": {"decimals": 2},
        }
    )

    fee = model.compute(
        side="SELL",
        price=1.0,
        qty=1.0,
        liquidity="taker",
    )

    assert fee == pytest.approx(0.03)

