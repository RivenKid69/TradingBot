from __future__ import annotations

import numpy as np
import pytest

from action_proto import ActionProto, ActionType
from domain.adapters import ActionV1, action_v1_to_proto, gym_to_action_v1


def test_gym_to_action_v1_normalises_payload():
    payload = {
        "type": 2,
        "volume_frac": np.array([0.25], dtype=np.float32),
        "price_offset_ticks": np.int32(5),
        "ttl_steps": np.int64(3),
        "tif": "IOC",
        "client_tag": "test",
    }

    action = gym_to_action_v1(payload)

    assert isinstance(action, ActionV1)
    assert action.action_type == ActionType.LIMIT
    assert action.volume_frac == pytest.approx(0.25)
    assert action.price_offset_ticks == 5
    assert action.ttl_steps == 3
    assert action.tif == "IOC"
    assert action.client_tag == "test"


def test_action_v1_to_proto_roundtrip():
    v1 = ActionV1(
        action_type=ActionType.MARKET,
        volume_frac=0.75,
        ttl_steps=2,
        client_tag="abc",
    )

    proto = action_v1_to_proto(v1)

    assert isinstance(proto, ActionProto)
    assert proto.action_type is ActionType.MARKET
    assert proto.volume_frac == pytest.approx(0.75)
    assert proto.ttl_steps == 2
    assert proto.client_tag == "abc"


def test_mapping_conversion_returns_proto():
    payload = {
        "type": ActionType.CANCEL_ALL,
        "volume_frac": 0.0,
    }

    proto = action_v1_to_proto(payload)
    assert isinstance(proto, ActionProto)
    assert proto.action_type is ActionType.CANCEL_ALL
    assert proto.volume_frac == 0.0


def test_gym_to_action_v1_rejects_invalid_mapping():
    with pytest.raises(ValueError):
        gym_to_action_v1({"volume_frac": 0.1})


def test_action_v1_to_proto_type_error():
    with pytest.raises(TypeError):
        action_v1_to_proto(123)
