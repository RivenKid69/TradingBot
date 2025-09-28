import pytest

from execution_sim import ExecutionSimulator


def _make_sim(mark_to: str) -> ExecutionSimulator:
    return ExecutionSimulator(
        filters_path=None,
        pnl_config={"mark_to": mark_to},
        use_seasonality=False,
    )


@pytest.mark.parametrize(
    "mark_to, position, expected_mark",
    [
        ("side", 1.0, 99.0),
        ("mid", 1.0, 100.0),
        ("bid", 1.0, 99.0),
        ("ask", 1.0, 101.0),
        ("side", -1.0, 101.0),
        ("mid", -1.0, 100.0),
        ("bid", -1.0, 99.0),
        ("ask", -1.0, 101.0),
        ("side", 0.0, 100.0),
        ("mid", 0.0, 100.0),
        ("bid", 0.0, 99.0),
        ("ask", 0.0, 101.0),
    ],
)
def test_mark_price_modes(mark_to: str, position: float, expected_mark: float) -> None:
    sim = _make_sim(mark_to)
    sim.position_qty = position
    sim._avg_entry_price = 100.0
    sim.realized_pnl_cum = 0.0

    mark = sim._mark_price(ref=100.0, bid=99.0, ask=101.0)
    assert mark == pytest.approx(expected_mark)

    unrealized = sim._unrealized_pnl(mark)
    expected_total = position * (expected_mark - 100.0)
    assert sim.realized_pnl_cum + unrealized == pytest.approx(expected_total)


@pytest.mark.parametrize(
    "mark_to, position, quotes, expected_mark",
    [
        ("mid", 1.0, {"ref": 100.0}, 100.0),
        ("bid", 1.0, {"ref": 100.0}, 100.0),
        ("ask", -1.0, {"ref": 100.0}, 100.0),
        ("side", 1.0, {"ref": 100.0}, 100.0),
        ("side", -1.0, {"ref": 100.0}, 100.0),
        ("mid", 1.0, {"bid": None, "ask": None, "ref": None}, None),
        ("bid", 1.0, {"ask": 101.0}, 101.0),
        ("ask", -1.0, {"bid": 99.0}, 99.0),
    ],
)
def test_mark_price_fallbacks(
    mark_to: str, position: float, quotes: dict[str, float | None], expected_mark: float | None
) -> None:
    sim = _make_sim(mark_to)
    sim.position_qty = position
    sim._avg_entry_price = 100.0
    sim.realized_pnl_cum = 0.0

    mark = sim._mark_price(
        ref=quotes.get("ref"),
        bid=quotes.get("bid"),
        ask=quotes.get("ask"),
    )

    if expected_mark is None:
        assert mark is None
        assert sim._unrealized_pnl(mark) == pytest.approx(0.0)
        return

    assert mark == pytest.approx(expected_mark)
    unrealized = sim._unrealized_pnl(mark)
    expected_total = position * (expected_mark - 100.0)
    assert sim.realized_pnl_cum + unrealized == pytest.approx(expected_total)
