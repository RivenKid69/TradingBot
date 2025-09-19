import pytest

from adapters.binance_spot_private import AccountFeeInfo
from impl_fees import FeesImpl


def test_account_info_overrides_base_rates(monkeypatch):
    calls = {}

    def _fake_fetch(**kwargs):
        calls["kwargs"] = kwargs
        return AccountFeeInfo(vip_tier=3, maker_bps=7.5, taker_bps=9.1)

    monkeypatch.setattr("impl_fees.fetch_account_fee_info", _fake_fetch)

    fees = FeesImpl.from_dict(
        {
            "maker_bps": 1.0,
            "taker_bps": 5.0,
            "account_info": {
                "enabled": True,
                "api_key": "key",
                "api_secret": "secret",
            },
        }
    )

    assert calls["kwargs"]["api_key"] == "key"
    assert fees.model_payload["maker_bps"] == pytest.approx(7.5)
    assert fees.model_payload["taker_bps"] == pytest.approx(9.1)
    assert fees.model_payload["vip_tier"] == 3

    account_meta = fees.metadata.get("account_fetch") or {}
    assert account_meta.get("status") == "ok"
    applied = account_meta.get("applied") or {}
    assert applied.get("maker_bps") is True
    assert applied.get("taker_bps") is True
    assert applied.get("vip_tier") is True


def test_account_info_missing_credentials(monkeypatch):
    def _fail_fetch(**kwargs):  # pragma: no cover - should not be triggered
        raise AssertionError("fetch_account_fee_info should not be called")

    monkeypatch.setattr("impl_fees.fetch_account_fee_info", _fail_fetch)

    fees = FeesImpl.from_dict({"account_info": {"enabled": True}})

    account_meta = fees.metadata.get("account_fetch") or {}
    assert account_meta.get("status") == "missing_credentials"
    applied = account_meta.get("applied") or {}
    assert applied.get("maker_bps") is False
    assert applied.get("taker_bps") is False
    assert applied.get("vip_tier") is False
