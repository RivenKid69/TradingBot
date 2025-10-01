import math
from decimal import Decimal

import pytest

from core_config import SpotTurnoverCaps, SpotTurnoverLimit
from core_models import Bar
from impl_bar_executor import BarExecutor, PortfolioState


@pytest.fixture
def base_state() -> PortfolioState:
    return PortfolioState(
        symbol="BTCUSDT",
        weight=0.2,
        equity_usd=1000.0,
        quantity=Decimal("0.02"),
        price=Decimal("100"),
        ts=1,
    )


def make_bar(ts: int = 1, price: str = "100") -> Bar:
    dec_price = Decimal(price)
    return Bar(
        ts=ts,
        symbol="BTCUSDT",
        open=dec_price,
        high=dec_price,
        low=dec_price,
        close=dec_price,
        volume_base=Decimal("1"),
        volume_quote=dec_price,
    )


def test_resolve_target_weight_modes(base_state: PortfolioState) -> None:
    executor = BarExecutor(default_equity_usd=base_state.equity_usd)

    target, mode, delta = executor._resolve_target_weight(
        base_state, {"target_weight": 1.5}
    )
    assert target == pytest.approx(1.0)
    assert mode == "target"
    assert delta == pytest.approx(0.8)

    target, mode, delta = executor._resolve_target_weight(
        base_state, {"delta_weight": -0.4}
    )
    assert target == pytest.approx(0.0)
    assert mode == "delta"
    assert delta == pytest.approx(-0.2)

    target, mode, delta = executor._resolve_target_weight(base_state, {})
    assert target == pytest.approx(base_state.weight)
    assert mode == "none"
    assert delta == pytest.approx(0.0)


def test_turnover_caps_and_registration(base_state: PortfolioState) -> None:
    caps = SpotTurnoverCaps(
        per_symbol=SpotTurnoverLimit(usd=100.0),
        portfolio=SpotTurnoverLimit(usd=150.0),
    )
    executor = BarExecutor(default_equity_usd=base_state.equity_usd, turnover_caps=caps)
    bar = make_bar(ts=base_state.ts)

    initial = executor._evaluate_turnover_caps(base_state.symbol, base_state, bar)
    assert initial["symbol_limit"] == pytest.approx(100.0)
    assert initial["portfolio_limit"] == pytest.approx(150.0)
    assert initial["effective_cap"] == pytest.approx(100.0)

    executor._register_turnover(base_state.symbol, bar.ts, 60.0)
    after_first = executor._evaluate_turnover_caps(base_state.symbol, base_state, bar)
    assert after_first["symbol_remaining"] == pytest.approx(40.0)
    assert after_first["portfolio_remaining"] == pytest.approx(90.0)
    assert after_first["effective_cap"] == pytest.approx(40.0)

    executor._register_turnover(base_state.symbol, bar.ts, 50.0)
    exhausted = executor._evaluate_turnover_caps(base_state.symbol, base_state, bar)
    assert exhausted["symbol_remaining"] == pytest.approx(0.0)
    assert exhausted["portfolio_remaining"] == pytest.approx(40.0)
    assert exhausted["effective_cap"] == pytest.approx(0.0)

    next_state = base_state.with_bar(make_bar(ts=2, price="105"), Decimal("105"))
    executor._register_turnover(base_state.symbol, 2, 10.0)
    refreshed = executor._evaluate_turnover_caps(next_state.symbol, next_state, make_bar(ts=2, price="105"))
    assert refreshed["symbol_remaining"] == pytest.approx(90.0)
    assert refreshed["portfolio_remaining"] == pytest.approx(140.0)


def test_build_instructions_twap_participation(base_state: PortfolioState) -> None:
    executor = BarExecutor(
        default_equity_usd=base_state.equity_usd,
        symbol_specs={"BTCUSDT": {"step_size": "0"}},
    )
    bar = make_bar(ts=10, price="200")
    state = base_state.with_bar(bar, bar.close)

    payload = {
        "delta_weight": 0.4,
        "twap": {"parts": 2, "interval_ms": 60000},
        "max_participation": 0.05,
    }

    instructions, final_weight, total_notional, _, reason = executor._build_instructions(
        state=state,
        target_weight=state.weight + 0.4,
        delta_weight=0.4,
        payload=payload,
        bar=bar,
        adv_quote=2000.0,
    )

    assert reason is None
    assert len(instructions) == 4  # participation forces additional slices
    assert all(instr.slices_total == len(instructions) for instr in instructions)
    assert all(instr.slice_index == idx for idx, instr in enumerate(instructions))
    expected_ts = [bar.ts + i * 60000 for i in range(4)]
    assert [instr.ts for instr in instructions] == expected_ts
    assert final_weight == pytest.approx(state.weight + 0.4)
    assert total_notional == pytest.approx(400.0)


def test_build_instructions_rejects_below_min_notional(base_state: PortfolioState) -> None:
    executor = BarExecutor(
        default_equity_usd=base_state.equity_usd,
        symbol_specs={"BTCUSDT": {"min_notional": "50"}},
    )
    payload = {"delta_weight": 0.01}
    instructions, final_weight, total_notional, _, reason = executor._build_instructions(
        state=base_state,
        target_weight=base_state.weight + 0.01,
        delta_weight=0.01,
        payload=payload,
        bar=None,
        adv_quote=None,
    )

    assert instructions == []
    assert reason == "below_min_notional"
    assert final_weight == pytest.approx(base_state.weight)
    assert total_notional == pytest.approx(0.0)


def test_build_instructions_quantizes_step_size(base_state: PortfolioState) -> None:
    executor = BarExecutor(
        default_equity_usd=base_state.equity_usd,
        symbol_specs={"BTCUSDT": {"step_size": "0.1"}},
    )
    price = Decimal("123.45")
    state = base_state.with_bar(make_bar(ts=5, price=str(price)), price)
    target_weight = 0.35
    payload = {"target_weight": target_weight}

    instructions, final_weight, total_notional, _, reason = executor._build_instructions(
        state=state,
        target_weight=target_weight,
        delta_weight=target_weight - state.weight,
        payload=payload,
        bar=None,
        adv_quote=None,
    )

    assert reason is None
    assert len(instructions) == 1
    instruction = instructions[0]
    assert math.isclose(float(instruction.quantity), 1.2, rel_tol=1e-9)
    expected_delta = float((Decimal("1.2") * price) / Decimal(str(state.equity_usd)))
    expected_notional = float(Decimal("1.2") * price)
    assert final_weight == pytest.approx(state.weight + expected_delta)
    assert total_notional == pytest.approx(expected_notional)


def test_build_instructions_twap_skips_zero_quantity_slices(base_state: PortfolioState) -> None:
    executor = BarExecutor(
        default_equity_usd=base_state.equity_usd,
        symbol_specs={"BTCUSDT": {"step_size": "0.7"}},
    )
    bar = make_bar(ts=20, price="100")
    state = base_state.with_bar(bar, bar.close)

    payload = {"delta_weight": 0.1, "twap": {"parts": 3, "interval_ms": 120000}}

    instructions, final_weight, total_notional, _, reason = executor._build_instructions(
        state=state,
        target_weight=state.weight + 0.1,
        delta_weight=0.1,
        payload=payload,
        bar=bar,
        adv_quote=None,
    )

    assert reason is None
    # Only the final slice should execute due to rounding
    assert len(instructions) == 1
    instruction = instructions[0]
    assert instruction.slice_index == 0
    assert instruction.slices_total == 1
    assert instruction.ts == bar.ts + 2 * 120000
    assert instruction.quantity == Decimal("0.7")
    assert total_notional == pytest.approx(float(Decimal("0.7") * bar.close))
    expected_delta = float((Decimal("0.7") * bar.close) / Decimal(str(state.equity_usd)))
    assert final_weight == pytest.approx(state.weight + expected_delta)


def test_build_instructions_respects_participation_cap_with_large_step_size(
    base_state: PortfolioState,
) -> None:
    executor = BarExecutor(
        default_equity_usd=base_state.equity_usd,
        symbol_specs={"BTCUSDT": {"step_size": "2"}},
    )
    bar = make_bar(ts=30, price="100")
    state = base_state.with_bar(bar, bar.close)

    payload = {
        "delta_weight": 0.6,
        "twap": {"parts": 10, "interval_ms": 60000},
        "max_participation": 0.35,
    }

    instructions, final_weight, total_notional, _, reason = executor._build_instructions(
        state=state,
        target_weight=state.weight + 0.6,
        delta_weight=0.6,
        payload=payload,
        bar=bar,
        adv_quote=1000.0,
    )

    cap = 1000.0 * 0.35
    assert reason is None
    assert instructions
    assert all(instr.notional_usd <= cap + 1e-9 for instr in instructions)
    assert instructions[-1].notional_usd == pytest.approx(200.0)
    assert instructions[-1].quantity == Decimal("2")
    executed_notional = sum(instr.notional_usd for instr in instructions)
    assert total_notional == pytest.approx(executed_notional)
    expected_final = state.weight + executed_notional / state.equity_usd
    assert final_weight == pytest.approx(expected_final)
