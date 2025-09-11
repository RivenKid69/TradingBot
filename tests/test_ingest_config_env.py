import logging  # ensure stdlib logging is loaded before path hacks
import pathlib
import sys


BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


def test_paths_config_env_override(monkeypatch):
    from ingest_config import load_config_from_str

    yaml_cfg = """
    symbols: ["BTCUSDT"]
    period:
      start: '2021-01-01'
      end: '2021-01-02'
    paths:
      klines_dir: yaml_klines
      prices_out: yaml_prices.parquet
    """

    monkeypatch.setenv("KLINES_DIR", "/tmp/klines")
    cfg = load_config_from_str(yaml_cfg)
    assert cfg.paths.klines_dir == "/tmp/klines"
    assert cfg.paths.prices_out == "yaml_prices.parquet"
