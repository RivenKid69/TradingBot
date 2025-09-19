import math


from quantizer import Quantizer


def test_commission_step_from_quote_commission_precision():
    filters = {
        "BTCUSDT": {
            "quoteCommissionPrecision": 4,
            "PRICE_FILTER": {},
        }
    }
    quantizer = Quantizer(filters, strict=True)
    assert math.isclose(quantizer.get_commission_step("BTCUSDT"), 10.0 ** -4)


def test_commission_step_falls_back_to_quote_precision():
    filters = {
        "ETHUSDT": {
            "quotePrecision": "5",
            "PRICE_FILTER": {},
        }
    }
    quantizer = Quantizer(filters, strict=True)
    assert math.isclose(quantizer.get_commission_step("ETHUSDT"), 10.0 ** -5)


def test_commission_step_uses_direct_value_when_present():
    filters = {
        "BNBUSDT": {
            "commission_step": "0.0005",
            "PRICE_FILTER": {},
        }
    }
    quantizer = Quantizer(filters, strict=True)
    assert math.isclose(quantizer.get_commission_step("BNBUSDT"), 0.0005)


def test_commission_step_missing_symbol_returns_zero():
    quantizer = Quantizer({}, strict=True)
    assert quantizer.get_commission_step("UNKNOWN") == 0.0

