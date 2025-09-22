import json

import pytest

from impl_fees import FeesImpl


def test_public_snapshot_overrides_applied_without_account_info(tmp_path):
    snapshot_payload = {
        "metadata": {
            "built_at": "2024-01-01T00:00:00Z",
            "source": "unit-test",
            "vip_tier": "VIP 3",
            "schema_version": 1,
            "account_overrides": {
                "vip_tier": 3,
                "use_bnb_discount": True,
                "maker_discount_mult": 0.75,
                "taker_discount_mult": 0.75,
            },
        },
        "fees": {
            "BTCUSDT": {
                "maker_bps": 1.0,
                "taker_bps": 5.0,
            }
        },
    }

    snapshot_path = tmp_path / "fees.json"
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

    fees = FeesImpl.from_dict(
        {
            "path": str(snapshot_path),
            "account_info": {"enabled": True},
        }
    )

    assert fees.model_payload["vip_tier"] == 3
    assert fees.model_payload["use_bnb_discount"] is True
    assert fees.model_payload["maker_discount_mult"] == pytest.approx(0.75)
    assert fees.model_payload["taker_discount_mult"] == pytest.approx(0.75)

    expected = fees.expected_payload
    assert expected["maker_fee_bps"] == pytest.approx(0.75)
    assert expected["taker_fee_bps"] == pytest.approx(3.75)
    assert expected["use_bnb_discount"] is True

    account_meta = fees.metadata.get("account_fetch") or {}
    assert account_meta.get("status") == "missing_credentials"
    applied = account_meta.get("applied") or {}
    assert applied.get("vip_tier") is False
    assert applied.get("maker_bps") is False
    assert applied.get("taker_bps") is False

    table_meta = fees.metadata.get("table") or {}
    overrides = table_meta.get("account_overrides") or {}
    assert overrides.get("vip_tier") == 3
    assert overrides.get("use_bnb_discount") is True
    assert overrides.get("maker_discount_mult") == pytest.approx(0.75)
    assert overrides.get("taker_discount_mult") == pytest.approx(0.75)

    assert fees.metadata.get("maker_discount_mult") == pytest.approx(0.75)
    assert fees.metadata.get("taker_discount_mult") == pytest.approx(0.75)
