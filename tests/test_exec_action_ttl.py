from types import SimpleNamespace

from execution_sim import ActionType, as_exec_action


def _make_proto(ttl_steps):
    return SimpleNamespace(
        action_type=ActionType.LIMIT,
        volume_frac=1.0,
        abs_price=100.0,
        ttl_steps=ttl_steps,
    )


def test_as_exec_action_rounds_fractional_ttl_half_up():
    action = as_exec_action(_make_proto(0.5), step_ms=100)
    assert action.ttl_steps == 1


def test_as_exec_action_rounds_fractional_ttl_to_next_int():
    action = as_exec_action(_make_proto(2.5), step_ms=100)
    assert action.ttl_steps == 3
