import pytest

from impl_fees import FeesImpl


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

    assert disc.cfg.maker_discount_mult == 0.75
    assert disc.cfg.taker_discount_mult == 0.75
    assert fee_disc_maker == pytest.approx(fee_base_maker * 0.75)
    assert fee_disc_taker == pytest.approx(fee_base_taker * 0.75)

