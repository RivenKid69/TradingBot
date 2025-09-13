import textwrap
import sys
import pathlib

# Ensure stdlib logging is used instead of local logging.py
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_orig_sys_path = list(sys.path)
sys.path = [p for p in sys.path if p not in ("", str(REPO_ROOT))]
import logging as std_logging  # type: ignore
sys.modules["logging"] = std_logging
sys.path = _orig_sys_path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_config import load_config_from_str


def test_clock_sync_config_loading():
    yaml_cfg = textwrap.dedent(
        """
        mode: sim
        symbols: ["BTCUSDT"]
        components:
          market_data:
            target: "module:Cls"
            params: {}
          executor:
            target: "module:Cls"
            params: {}
          feature_pipe:
            target: "module:Cls"
            params: {}
          policy:
            target: "module:Cls"
            params: {}
          risk_guards:
            target: "module:Cls"
            params: {}
        data:
          symbols: ["BTCUSDT"]
          timeframe: "1m"
        clock_sync:
          refresh_sec: 123
          warn_threshold_ms: 456
          kill_threshold_ms: 789
        """
    )
    cfg = load_config_from_str(yaml_cfg)
    assert cfg.clock_sync.refresh_sec == 123
    assert cfg.clock_sync.warn_threshold_ms == 456
    assert cfg.clock_sync.kill_threshold_ms == 789
