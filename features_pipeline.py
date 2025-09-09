# -*- coding: utf-8 -*-
"""
features_pipeline.py
------------------------------------------------------------------
Single source of truth for feature normalization both in training
and inference. Works over dict[str, pandas.DataFrame] where each DF
follows the canonical schema established in prepare_and_run.py.
- Adds standardized columns with suffix '_z' (z-score).
- Leaves original columns intact.
- Saves/loads stats to/from JSON for reproducibility.

Usage:
    pipe = FeaturePipeline()
    pipe.fit(all_dfs_dict)                 # during training
    pipe.save("models/preproc_pipeline.json")
    all_dfs_dict = pipe.transform_dict(all_dfs_dict, add_suffix="_z")

    pipe = FeaturePipeline.load("models/preproc_pipeline.json")  # inference
    all_dfs_dict = pipe.transform_dict(all_dfs_dict, add_suffix="_z")
"""
import os
import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

CANON_PREFIX = [
    "timestamp","symbol","open","high","low","close","volume","quote_asset_volume",
    "number_of_trades","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"
]

# Additional optional features we may standardize if present
OPTIONAL_NUMERIC = [
    "fear_greed_value","fear_greed_value_norm",
    "recent_event_high_96h","recent_event_medium_96h","time_since_last_event_hours",
]

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)

def _columns_to_scale(df: pd.DataFrame) -> List[str]:
    # Key columns which are numeric but shouldn't be z-scored directly:
    exclude = {"timestamp"}  # 'symbol' non-numeric already excluded
    cols: List[str] = []
    for c in df.columns:
        if c in exclude: 
            continue
        if c == "symbol":
            continue
        if c.endswith("_z"):  # already standardized
            continue
        if _is_numeric(df[c]):
            cols.append(c)
    return cols

class FeaturePipeline:
    def __init__(self, stats: Optional[Dict[str, Dict[str, float]]] = None):
        # stats: {col: {"mean": float, "std": float}}
        self.stats: Dict[str, Dict[str, float]] = stats or {}

    def reset(self) -> None:
        """Drop previously computed statistics.

        Creating a fresh instance for each ``TradingEnv`` or clearing the
        state on episode reset avoids cross‑environment leakage of
        normalization parameters.
        """
        self.stats.clear()

    def fit(self, dfs: Dict[str, pd.DataFrame]) -> "FeaturePipeline":
        # Fit over concatenation of all symbols (row-wise)
        frames = []
        for _, df in dfs.items():
            frames.append(df)
        if not frames:
            raise ValueError("No dataframes to fit FeaturePipeline.")
        big = pd.concat(frames, axis=0, ignore_index=True)
        if "close_orig" in big.columns:
            pass
        elif "close" in big.columns:
            big["close"] = big["close"].shift(1)
        cols = _columns_to_scale(big)
        stats = {}
        for c in cols:
            v = big[c].astype(float).to_numpy()
            m = float(np.nanmean(v))
            s = float(np.nanstd(v, ddof=0))
            if not np.isfinite(s) or s == 0.0:
                s = 1.0  # avoid division by zero
            if not np.isfinite(m):
                m = 0.0
            stats[c] = {"mean": m, "std": s}
        self.stats = stats
        return self

    def transform_df(self, df: pd.DataFrame, add_suffix: str = "_z") -> pd.DataFrame:
        if not self.stats:
            raise ValueError("FeaturePipeline is empty; call fit() or load().")
        out = df.copy()
        if "close_orig" in out.columns:
            pass
        elif "close" in out.columns:
            out["close"] = out["close"].shift(1)
        for c, ms in self.stats.items():
            if c not in out.columns:
                # silently skip columns missing in this DF
                continue
            v = out[c].astype(float).to_numpy()
            z = (v - ms["mean"]) / ms["std"]
            out[c + add_suffix] = z
        return out

    def transform_dict(self, dfs: Dict[str, pd.DataFrame], add_suffix: str = "_z") -> Dict[str, pd.DataFrame]:
        return {k: self.transform_df(v, add_suffix=add_suffix) for k, v in dfs.items()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls(stats=stats)
