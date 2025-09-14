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


BASE_YAML = textwrap.dedent(
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
    """
)


def test_ws_dedup_defaults_and_override():
    cfg = load_config_from_str(BASE_YAML)
    assert cfg.ws_dedup.enabled is False
    assert cfg.ws_dedup.persist_path == "state/last_bar_seen.json"
    assert cfg.ws_dedup.log_skips is False

    yaml_with_dedup = BASE_YAML + textwrap.dedent(
        """
        ws_dedup:
          enabled: true
          persist_path: "custom.json"
          log_skips: true
        """
    )
    cfg2 = load_config_from_str(yaml_with_dedup)
    assert cfg2.ws_dedup.enabled is True
    assert cfg2.ws_dedup.persist_path == "custom.json"
    assert cfg2.ws_dedup.log_skips is True
