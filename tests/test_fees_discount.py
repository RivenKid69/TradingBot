import pytest

from binance_fee_refresh import DEFAULT_BNB_DISCOUNT_RATE
from impl_fees import FeesConfig, FeesImpl


def test_bnb_discount_applied_by_default():
    price = 100.0
    qty = 1.0

    # базовая комиссия без скидки
    base = FeesImpl.from_dict({})
    fee_base_maker = base.model.compute(side="BUY", price=price, qty=qty, liquidity="maker")
    fee_base_taker = base.model.compute(side="BUY", price=price, qty=qty, liquidity="taker")

    # включаем BNB скидку без явных мультипликаторов
    disc = FeesImpl.from_dict({"use_bnb_discount": True})
    fee_disc_maker = disc.model.compute(side="BUY", price=price, qty=qty, liquidity="maker")
    fee_disc_taker = disc.model.compute(side="BUY", price=price, qty=qty, liquidity="taker")

    expected_mult = 1.0 - float(DEFAULT_BNB_DISCOUNT_RATE)
    assert disc.cfg.maker_discount_mult == pytest.approx(expected_mult)
    assert disc.cfg.taker_discount_mult == pytest.approx(expected_mult)
    assert fee_disc_maker == pytest.approx(fee_base_maker * expected_mult)
    assert fee_disc_taker == pytest.approx(fee_base_taker * expected_mult)


def test_bnb_discount_auto_data_ignored_when_disabled():
    price = 100.0
    qty = 1.0

    baseline = FeesImpl(FeesConfig(use_bnb_discount=False))
    fee_base_maker = baseline.model.compute(
        side="BUY", price=price, qty=qty, liquidity="maker"
    )
    fee_base_taker = baseline.model.compute(
        side="BUY", price=price, qty=qty, liquidity="taker"
    )

    cfg = FeesConfig(use_bnb_discount=False)
    cfg.auto_maker_discount_mult = 0.25
    cfg.auto_taker_discount_mult = 0.5
    cfg.auto_use_bnb_discount = True

    auto = FeesImpl(cfg)
    fee_auto_maker = auto.model.compute(
        side="BUY", price=price, qty=qty, liquidity="maker"
    )
    fee_auto_taker = auto.model.compute(
        side="BUY", price=price, qty=qty, liquidity="taker"
    )

    assert auto.model_payload["use_bnb_discount"] is False
    assert auto.model_payload["maker_discount_mult"] == pytest.approx(1.0)
    assert auto.model_payload["taker_discount_mult"] == pytest.approx(1.0)
    assert fee_auto_maker == pytest.approx(fee_base_maker)
    assert fee_auto_taker == pytest.approx(fee_base_taker)

