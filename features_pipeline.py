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
from datetime import UTC, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    def __init__(
        self,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ):
        """Container for feature normalization statistics.

        Parameters
        ----------
        stats:
            Mapping from column name to ``{"mean": float, "std": float}``.
        metadata:
            Additional information persisted alongside the statistics (for
            example the training window bounds or split version).
        """

        # stats: {col: {"mean": float, "std": float}}
        self.stats: Dict[str, Dict[str, float]] = stats or {}
        self.metadata: Dict[str, object] = metadata or {}

    def reset(self) -> None:
        """Drop previously computed statistics.

        Creating a fresh instance for each ``TradingEnv`` or clearing the
        state on episode reset avoids cross‑environment leakage of
        normalization parameters.
        """

        self.stats.clear()
        self.metadata.clear()

    def fit(
        self,
        dfs: Dict[str, pd.DataFrame],
        *,
        train_mask_column: Optional[str] = None,
        train_mask_values: Optional[Iterable] = None,
        train_start_ts: Optional[int] = None,
        train_end_ts: Optional[int] = None,
        timestamp_column: str = "timestamp",
        split_version: Optional[str] = None,
        train_intervals: Optional[Sequence[Tuple[Optional[int], Optional[int]]]] = None,
    ) -> "FeaturePipeline":
        """Fit normalization statistics from the provided dataframes.

        The caller may either provide a boolean/role mask identifying the
        training rows (for example a ``wf_role`` column equal to ``"train"``)
        or explicit ``train_start_ts``/``train_end_ts`` bounds. When both are
        supplied the intersection is used.
        """

        frames: List[pd.DataFrame] = []
        per_symbol_counts: Dict[str, int] = {}

        if train_mask_column is None and (train_start_ts is not None or train_end_ts is not None):
            # Ensure the timestamp column exists up-front when time bounds are used.
            for name, df in dfs.items():
                if timestamp_column not in df.columns:
                    raise KeyError(
                        f"DataFrame '{name}' is missing timestamp column '{timestamp_column}' required for training window filter."
                    )

        mask_values: Optional[Sequence] = None
        if train_mask_values is not None:
            mask_values = tuple(train_mask_values)

        for name, df in dfs.items():
            if df is None:
                continue
            cur = df
            if train_mask_column is not None:
                if train_mask_column not in cur.columns:
                    raise KeyError(
                        f"DataFrame '{name}' is missing training mask column '{train_mask_column}'."
                    )
                mask_series = cur[train_mask_column]
                if mask_values is None:
                    if pd.api.types.is_bool_dtype(mask_series):
                        mask = mask_series.astype(bool)
                    else:
                        mask = mask_series.astype(str).str.lower() == "train"
                else:
                    mask = mask_series.isin(mask_values)
                cur = cur.loc[mask]

            if train_start_ts is not None or train_end_ts is not None:
                ts = pd.to_numeric(cur[timestamp_column], errors="coerce")
                time_mask = pd.Series(True, index=cur.index)
                if train_start_ts is not None:
                    time_mask &= ts >= int(train_start_ts)
                if train_end_ts is not None:
                    time_mask &= ts <= int(train_end_ts)
                cur = cur.loc[time_mask]

            if not cur.empty:
                per_symbol_counts[name] = int(len(cur))
                frames.append(cur)

        if not frames:
            raise ValueError("No rows available to fit FeaturePipeline after applying training filters.")

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

        intervals_payload: Optional[List[Dict[str, Optional[int]]]] = None
        if train_intervals:
            intervals_payload = [
                {
                    "start_ts": int(start) if start is not None else None,
                    "end_ts": int(end) if end is not None else None,
                }
                for start, end in train_intervals
            ]

        metadata: Dict[str, object] = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "filters": {
                "train_mask_column": train_mask_column,
                "train_mask_values": list(mask_values) if mask_values is not None else None,
                "train_start_ts": int(train_start_ts) if train_start_ts is not None else None,
                "train_end_ts": int(train_end_ts) if train_end_ts is not None else None,
                "timestamp_column": timestamp_column,
                "train_intervals": intervals_payload,
            },
            "train_rows_by_symbol": per_symbol_counts,
            "train_rows_total": int(sum(per_symbol_counts.values())),
        }
        if split_version is not None:
            metadata["split_version"] = str(split_version)

        self.stats = stats
        self.metadata = metadata
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

    def get_metadata(self) -> Dict[str, object]:
        """Return metadata captured during :meth:`fit`."""

        return dict(self.metadata)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"stats": self.stats, "metadata": self.metadata}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "stats" in payload:
            stats = payload.get("stats", {})
            metadata = payload.get("metadata", {})
        else:
            # Backwards compatibility for legacy artifacts containing only stats.
            stats = payload
            metadata = {}
        return cls(stats=stats, metadata=metadata)
