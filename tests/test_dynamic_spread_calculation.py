import math

import pytest

from impl_slippage import _DynamicSpreadProfile, _calc_dynamic_spread, SlippageCfg, SlippageImpl
from slippage import DynamicSpreadConfig


@pytest.fixture
def dynamic_cfg() -> DynamicSpreadConfig:
    return DynamicSpreadConfig(
        enabled=True,
        alpha_bps=5.0,
        beta_coef=1.5,
        min_spread_bps=1.0,
        max_spread_bps=50.0,
        smoothing_alpha=0.4,
    )


def test_calc_dynamic_spread_produces_finite_values(dynamic_cfg: DynamicSpreadConfig) -> None:
    profile = _DynamicSpreadProfile(cfg=dynamic_cfg, default_spread_bps=dynamic_cfg.alpha_bps or 5.0)
    for _ in range(5):
        spread = _calc_dynamic_spread(
            cfg=dynamic_cfg,
            default_spread_bps=dynamic_cfg.alpha_bps or 5.0,
            bar_high=101.0,
            bar_low=99.0,
            mid_price=100.0,
            seasonal_multiplier=1.2,
            vol_multiplier=1.1,
            profile=profile,
        )
        assert spread is not None
        assert math.isfinite(spread)
        assert spread >= 0.0
        assert spread <= dynamic_cfg.max_spread_bps


def test_get_spread_bps_fallback_to_base(dynamic_cfg: DynamicSpreadConfig) -> None:
    cfg = SlippageCfg(default_spread_bps=4.0, dynamic=dynamic_cfg)
    impl = SlippageImpl(cfg)

    # missing bar data forces fallback to base spread
    fallback = impl.get_spread_bps(ts_ms=0, base_spread_bps=6.0, bar_high=None, bar_low=None, mid_price=None)
    assert math.isfinite(fallback)
    assert fallback >= 0.0
    assert fallback == pytest.approx(6.0)

    # valid bar data should still return finite non-negative spread
    spread = impl.get_spread_bps(
        ts_ms=0,
        base_spread_bps=6.0,
        bar_high=102.0,
        bar_low=99.0,
        mid_price=100.0,
        vol_metrics={"range_ratio_bps": 40.0},
    )
    assert math.isfinite(spread)
    assert spread >= 0.0
