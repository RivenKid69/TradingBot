from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
import yaml
import os


class PeriodConfig(BaseModel):
    start: str
    end: str


class PathsConfig(BaseModel):
    """Configuration of file-system paths used during data ingestion.

    Values can be overridden via environment variables:

    * ``KLINES_DIR`` – override ``klines_dir``
    * ``FUTURES_DIR`` – override ``futures_dir``
    * ``PRICES_OUT`` – override ``prices_out``
    """

    klines_dir: str = Field("data/klines", description="Directory for klines output")
    futures_dir: str = Field("data/futures", description="Directory for futures data")
    prices_out: str = Field(
        "data/prices.parquet", description="Output path for normalized prices"
    )

    @classmethod
    def from_env(cls, **kwargs) -> "PathsConfig":
        """Build ``PathsConfig`` reading overrides from environment variables."""

        data = dict(kwargs)
        env_map = {
            "klines_dir": "KLINES_DIR",
            "futures_dir": "FUTURES_DIR",
            "prices_out": "PRICES_OUT",
        }
        for field, env_name in env_map.items():
            env_val = os.getenv(env_name)
            if env_val:
                data[field] = env_val
        return cls(**data)


class FuturesConfig(BaseModel):
    mark_interval: str = "1m"


class SlownessConfig(BaseModel):
    api_limit: int = 1500
    sleep_ms: int = 350


class IngestConfig(BaseModel):
    symbols: List[str]
    market: str = "spot"
    intervals: List[str] = Field(default_factory=lambda: ["1m"])
    aggregate_to: List[str] = Field(default_factory=list)
    period: PeriodConfig
    paths: PathsConfig = PathsConfig()
    futures: FuturesConfig = FuturesConfig()
    slowness: SlownessConfig = Field(default_factory=SlownessConfig)


def load_config(path: str) -> IngestConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    paths = PathsConfig.from_env(**data.get("paths", {}))
    data["paths"] = paths
    return IngestConfig(**data)


def load_config_from_str(content: str) -> IngestConfig:
    data = yaml.safe_load(content) or {}
    paths = PathsConfig.from_env(**data.get("paths", {}))
    data["paths"] = paths
    return IngestConfig(**data)


__all__ = ["IngestConfig", "load_config", "load_config_from_str"]
