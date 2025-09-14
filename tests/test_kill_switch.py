import pathlib
import sys

# Ensure stdlib logging is used instead of local logging.py
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_orig_sys_path = list(sys.path)
sys.path = [p for p in sys.path if p not in ("", str(REPO_ROOT))]
import logging as std_logging  # type: ignore
sys.modules["logging"] = std_logging
sys.path = _orig_sys_path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_config import KillSwitchConfig
from services import monitoring


def test_kill_switch_feed_lag_trigger():
    monitoring.configure_kill_switch(KillSwitchConfig(feed_lag_ms=100.0))
    monitoring.report_feed_lag("BTCUSDT", 150.0)
    assert monitoring.kill_switch_triggered()
    info = monitoring.kill_switch_info()
    assert info["metric"] == "feed_lag_ms"
    assert info["symbol"] == "BTCUSDT"


def test_kill_switch_reset_on_configure():
    monitoring.configure_kill_switch(KillSwitchConfig(feed_lag_ms=100.0))
    monitoring.report_feed_lag("BTCUSDT", 50.0)
    assert not monitoring.kill_switch_triggered()
    monitoring.configure_kill_switch(KillSwitchConfig(feed_lag_ms=10.0))
    monitoring.report_feed_lag("BTCUSDT", 5.0)
    assert not monitoring.kill_switch_triggered()
