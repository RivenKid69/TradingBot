"""Extract liquidity seasonality multipliers.

The hour-of-week index uses ``0 = Monday 00:00 UTC``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

from utils.time import hour_of_week


def load_ohlcv(path: Path) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_multipliers(df: pd.DataFrame) -> np.ndarray:
    if 'ts_ms' not in df.columns:
        raise ValueError('ts_ms column required')
    vol_col = next((c for c in ['quote_asset_volume', 'quote_volume', 'volume'] if c in df.columns), None)
    if vol_col is None:
        raise ValueError('volume column not found')
    # ``hour_of_week`` uses Monday 00:00 UTC as index 0
    ts_ms = df['ts_ms'].to_numpy(dtype=np.int64)
    df = df.assign(hour_of_week=hour_of_week(ts_ms))
    grouped = df.groupby('hour_of_week')[vol_col].mean()
    overall = df[vol_col].mean()
    mult = grouped / overall if overall else grouped * 0.0 + 1.0
    mult = mult.reindex(range(168), fill_value=1.0)
    return mult.to_numpy(dtype=float)


def write_checksum(path: Path) -> Path:
    """Compute sha256 checksum for *path* and write `<path>.sha256`."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path = path.with_suffix(path.suffix + '.sha256')
    checksum_path.write_text(digest)
    return checksum_path


def main():
    parser = argparse.ArgumentParser(description='Extract liquidity seasonality multipliers')
    parser.add_argument(
        '--data',
        default='data/seasonality_source/latest.parquet',
        help='Path to OHLCV data (csv or parquet)',
    )
    parser.add_argument('--out', default='configs/liquidity_seasonality.json', help='Output JSON path')
    args = parser.parse_args()

    data_path = Path(args.data)
    df = load_ohlcv(data_path)
    multipliers = compute_multipliers(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        'hour_of_week_definition': '0=Monday 00:00 UTC',
        'liquidity': multipliers.tolist(),
    }
    with open(args.out, 'w') as f:
        json.dump(out_data, f, indent=2)
    checksum_path = write_checksum(data_path)
    print(f'Saved liquidity seasonality to {args.out}')
    print(f'Input data checksum written to {checksum_path}')


if __name__ == '__main__':
    main()
