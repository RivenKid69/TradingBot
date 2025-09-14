import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from pipeline import MetricKillSwitch, Stage, Reason


def test_metric_kill_switch_hysteresis_and_cooldown():
    ks = MetricKillSwitch(upper=2.0, lower=1.0, cooldown_bars=2)

    # Below thresholds -> pass
    res = ks.update("BTCUSDT", 0.5)
    assert res.action == "pass"
    assert not ks.is_active("BTCUSDT")

    # Exceed upper -> enter disabled state
    res = ks.update("BTCUSDT", 3.0)
    assert res.action == "drop"
    assert res.stage is Stage.POLICY
    assert res.reason is Reason.MAINTENANCE
    assert ks.is_active("BTCUSDT")

    # During cooldown still disabled even if metric low
    res = ks.update("BTCUSDT", 0.2)
    assert res.action == "drop"
    assert ks.is_active("BTCUSDT")

    # After cooldown but metric above lower -> still disabled
    res = ks.update("BTCUSDT", 1.5)
    assert res.action == "drop"
    assert ks.is_active("BTCUSDT")

    # Metric below lower after cooldown -> resume trading
    res = ks.update("BTCUSDT", 0.5)
    assert res.action == "pass"
    assert not ks.is_active("BTCUSDT")
    assert ks.last_metric_value("BTCUSDT") == pytest.approx(0.5)
