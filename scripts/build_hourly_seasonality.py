"""Utility helpers for deriving hour-of-week multipliers.

The script scans a trade/latency log and computes relative multipliers for
liquidity, latency and bid-ask spread.  The output JSON can then be consumed by
the simulator to modulate these parameters during backtests.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_logs(path: Path) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_multipliers(df: pd.DataFrame) -> dict[str, np.ndarray]:
    ts_col = 'ts_ms' if 'ts_ms' in df.columns else 'ts'
    if ts_col not in df.columns:
        raise ValueError('ts or ts_ms column required')
    ts = pd.to_datetime(df[ts_col], unit='ms', utc=True)
    how = ts.dt.dayofweek * 24 + ts.dt.hour
    df = df.assign(hour_of_week=how)
    metrics: dict[str, np.ndarray] = {}
    cols_map = {
        'liquidity': ['liquidity', 'order_size', 'qty', 'quantity'],
        'latency': ['latency_ms'],
        # Some datasets may label spread either in absolute terms or in bps.
        'spread': ['spread', 'spread_bps'],
    }
    for key, candidates in cols_map.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            arr = np.ones(168, dtype=float)
        else:
            grouped = df.groupby('hour_of_week')[col].mean()
            overall = df[col].mean()
            if overall:
                mult = grouped / overall
            else:
                mult = grouped * 0.0 + 1.0
            arr = mult.reindex(range(168), fill_value=1.0).to_numpy(dtype=float)
        metrics[key] = arr
    return metrics


def write_checksum(path: Path) -> Path:
    """Compute sha256 checksum for *path* and write `<path>.sha256`."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path = path.with_suffix(path.suffix + '.sha256')
    checksum_path.write_text(digest)
    return checksum_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build hourly seasonality multipliers for liquidity, latency and spread'
    )
    parser.add_argument(
        '--data',
        default='data/seasonality_source/latest.parquet',
        help='Path to trade/latency logs (csv or parquet)',
    )
    parser.add_argument(
        '--out',
        default='configs/liquidity_latency_seasonality.json',
        help='Output JSON path',
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    df = load_logs(data_path)
    multipliers = compute_multipliers(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({k: v.tolist() for k, v in multipliers.items()}, f, indent=2)
    checksum_path = write_checksum(data_path)
    print(f'Saved seasonality multipliers to {args.out}')
    print(f'Input data checksum written to {checksum_path}')


if __name__ == '__main__':
    main()
