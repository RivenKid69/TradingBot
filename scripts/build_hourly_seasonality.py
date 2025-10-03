"""Utility helpers for deriving hour-of-week multipliers.

The script scans a trade/latency log and computes relative multipliers for
liquidity, latency and bid-ask spread. The hour-of-week index uses
``0 = Monday 00:00 UTC``. The output JSON can then be consumed by the simulator
to modulate these parameters during backtests.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from utils.time import hour_of_week
from scripts.offline_utils import (
    SplitArtifact,
    apply_split_tag,
    load_offline_payload,
    ms_to_iso,
    resolve_split_artifact,
    window_days as compute_window_days,
)


DEFAULT_OUTPUT = Path("data/latency/liquidity_latency_seasonality.json")


def load_logs(path: Path) -> pd.DataFrame:
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _fill_missing(arr: np.ndarray, period: int = 168) -> tuple[np.ndarray, list[int]]:
    """Replace NaNs in *arr* using neighbouring-period averages or global mean."""

    out = arr.copy()
    imputed: list[int] = []
    for i, val in enumerate(out):
        if not np.isnan(val):
            continue
        imputed.append(i)
        left = right = None
        for j in range(1, period):
            lval = out[(i - j) % period]
            if not np.isnan(lval):
                left = lval
                break
        for j in range(1, period):
            rval = out[(i + j) % period]
            if not np.isnan(rval):
                right = rval
                break
        if left is not None and right is not None:
            out[i] = (left + right) / 2.0
        elif left is not None:
            out[i] = left
        elif right is not None:
            out[i] = right
        else:
            out[i] = 1.0
    return out, imputed


def _rolling_mean_circular(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply circular rolling mean with *window* size to *arr*."""
    if window <= 1:
        return arr
    k = window
    kernel = np.ones(k) / k
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    extended = np.concatenate([arr[-pad_left:], arr, arr[:pad_right]])
    return np.convolve(extended, kernel, mode="valid")


def compute_multipliers(
    df: pd.DataFrame,
    min_samples: int = 30,
    prior_metrics: Optional[Dict[str, np.ndarray]] = None,
    trim_bottom_pct: float = 0.0,
    trim_top_pct: float = 0.0,
    *,
    by_day: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    ts_col = 'ts_ms' if 'ts_ms' in df.columns else 'ts'
    if ts_col not in df.columns:
        raise ValueError('ts or ts_ms column required')
    # ``hour_of_week`` uses Monday 00:00 UTC as index 0
    ts_ms = df[ts_col].to_numpy(dtype=np.int64)
    how = hour_of_week(ts_ms)
    group_col = 'day_of_week' if by_day else 'hour_of_week'
    if by_day:
        df = df.assign(day_of_week=(how // 24))
    else:
        df = df.assign(hour_of_week=how)
    metrics: dict[str, np.ndarray] = {}
    imputed_hours: dict[str, list[int]] = {}
    cols_map = {
        'liquidity': ['liquidity', 'order_size', 'qty', 'quantity'],
        'latency': ['latency_ms'],
        # Some datasets may label spread either in absolute terms or in bps.
        'spread': ['spread', 'spread_bps'],
    }
    period = 7 if by_day else 168
    for key, candidates in cols_map.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            arr = np.ones(period, dtype=float)
        else:
            data = df[[group_col, col]].copy()
            if trim_bottom_pct > 0.0 or trim_top_pct > 0.0:
                series = data[col]
                lower = series.quantile(trim_bottom_pct / 100.0) if trim_bottom_pct > 0.0 else series.min()
                upper = series.quantile(1 - trim_top_pct / 100.0) if trim_top_pct > 0.0 else series.max()
                data = data[(data[col] >= lower) & (data[col] <= upper)]
            grouped = data.groupby(group_col)[col].agg(['mean', 'count'])
            overall = data[col].mean()
            if overall:
                mult = grouped['mean'] / overall
            else:
                mult = grouped['mean'] * 0.0 + 1.0
            mult[grouped['count'] < min_samples] = np.nan
            arr, imp = _fill_missing(
                mult.reindex(range(period)).to_numpy(dtype=float), period=period
            )
            if imp:
                imputed_hours[key] = imp
        if by_day:
            from utils_time import interpolate_daily_multipliers
            arr = interpolate_daily_multipliers(arr)
        if prior_metrics and key in prior_metrics:
            weights = np.asarray(prior_metrics[key], dtype=float)
            if weights.shape[0] != 168:
                raise ValueError(f"prior_metrics for {key} must have length 168")
            w = 1.0 / (1.0 + weights)
            arr = arr * w
            mean = float(np.mean(arr)) or 1.0
            arr = arr / mean
        metrics[key] = arr
    return metrics, imputed_hours


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
        default=None,
        help='Output JSON path (backups stored in data/latency/backups)',
    )
    parser.add_argument(
        '--symbol',
        default=None,
        help='If provided, wrap multipliers under this instrument symbol',
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=30,
        help='Minimum samples per hour required before imputation',
    )
    parser.add_argument(
        '--window-days',
        type=int,
        default=0,
        help='Number of most recent days to include (0 to use the full dataset)',
    )
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=0,
        help='Apply circular rolling mean with this window (0 to disable)',
    )
    parser.add_argument(
        '--smooth-alpha',
        type=float,
        default=0.0,
        help='Regularisation strength towards 1.0 (0 to disable)',
    )
    parser.add_argument(
        '--prior-metrics',
        default=None,
        help='Optional path to JSON with per-hour validation metrics',
    )
    parser.add_argument(
        '--trim-top',
        type=float,
        default=0.0,
        help='Percentile of highest values to discard before averaging (0-100)',
    )
    parser.add_argument(
        '--trim-bottom',
        type=float,
        default=0.0,
        help='Percentile of lowest values to discard before averaging (0-100)',
    )
    parser.add_argument(
        '--by-day',
        action='store_true',
        help='Aggregate by day-of-week and interpolate to 168-hour array',
    )
    parser.add_argument(
        '--config',
        default='configs/offline.yaml',
        help='Offline configuration with dataset split definitions',
    )
    parser.add_argument(
        '--split',
        default=None,
        help='Dataset split identifier providing window bounds and output path',
    )
    args = parser.parse_args()

    split_info: SplitArtifact | None = None
    if args.split:
        try:
            payload = load_offline_payload(args.config)
        except FileNotFoundError:
            raise SystemExit(f'Offline config not found: {args.config}')
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        try:
            split_info = resolve_split_artifact(payload, args.split, 'seasonality')
        except KeyError as exc:
            raise SystemExit(f"Unknown split {args.split!r} in offline config") from exc
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if (
            split_info.config_start_ms is not None
            and split_info.config_end_ms is not None
            and split_info.config_end_ms < split_info.config_start_ms
        ):
            raise SystemExit(
                f"Configured window for split {split_info.split_name!r} has end before start"
            )

    base_out = Path(args.out) if args.out else DEFAULT_OUTPUT
    if split_info:
        if split_info.output_path is not None and not args.out:
            base_out = split_info.output_path
        out_path = apply_split_tag(base_out, split_info.tag)
    else:
        out_path = base_out

    if split_info:
        config_window_days = compute_window_days(
            split_info.config_start_ms, split_info.config_end_ms
        )
    else:
        config_window_days = None

    data_path = Path(args.data)
    df = load_logs(data_path)
    if df.empty:
        raise SystemExit(f'No rows found in input data at {data_path}')

    ts_column = 'ts_ms' if 'ts_ms' in df.columns else 'ts' if 'ts' in df.columns else None
    if ts_column is None:
        raise SystemExit('Input data must contain ts or ts_ms column')

    ts_numeric = pd.to_numeric(df[ts_column], errors='coerce')
    if ts_numeric.isna().all():
        parsed = pd.to_datetime(df[ts_column], utc=True, errors='coerce')
        ts_numeric = (parsed.view('int64') // 1_000_000)
    mask = ts_numeric.notna()
    if split_info and split_info.config_start_ms is not None:
        mask &= ts_numeric >= split_info.config_start_ms
    if split_info and split_info.config_end_ms is not None:
        mask &= ts_numeric <= split_info.config_end_ms
    if not mask.any():
        raise SystemExit('No data available after applying split window filters')
    if not mask.all():
        df = df.loc[mask].copy()
        ts_numeric = ts_numeric.loc[mask]
    if args.window_days and args.window_days > 0:
        cutoff = ts_numeric.max() - (args.window_days * 86_400_000)
        mask_window = ts_numeric >= cutoff
        filtered_df = df.loc[mask_window].copy()
        if filtered_df.empty:
            raise SystemExit(
                'No data available after applying --window-days '
                f'={args.window_days}; consider lowering the value.'
            )
        removed = len(df) - len(filtered_df)
        df = filtered_df
        ts_numeric = ts_numeric.loc[mask_window]
        print(
            f'Applied --window-days={args.window_days}: kept {len(df)} rows, '
            f'removed {removed} rows.'
        )

    observed = ts_numeric.dropna()
    actual_start_ms = int(observed.min()) if not observed.empty else None
    actual_end_ms = int(observed.max()) if not observed.empty else None
    prior: Optional[Dict[str, np.ndarray]] = None
    if args.prior_metrics:
        with open(args.prior_metrics, 'r') as f:
            raw = json.load(f)
        prior = {k: np.asarray(v, dtype=float) for k, v in raw.items() if isinstance(v, list)}
    multipliers, imputed = compute_multipliers(
        df,
        args.min_samples,
        prior,
        trim_bottom_pct=args.trim_bottom,
        trim_top_pct=args.trim_top,
        by_day=args.by_day,
    )
    if args.smooth_window > 1:
        for key, arr in multipliers.items():
            multipliers[key] = _rolling_mean_circular(arr, args.smooth_window)
    if args.smooth_alpha > 0.0:
        for key, arr in multipliers.items():
            multipliers[key] = arr * (1.0 - args.smooth_alpha) + args.smooth_alpha

    for key, arr in multipliers.items():
        mean = float(np.mean(arr))
        if not np.isfinite(mean) or mean == 0.0:
            mean = 1.0
        multipliers[key] = arr / mean

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.window_days and args.window_days > 0:
        window_days_meta = int(args.window_days)
    elif config_window_days:
        window_days_meta = int(config_window_days)
    else:
        computed_window = compute_window_days(actual_start_ms, actual_end_ms)
        window_days_meta = int(computed_window) if computed_window else 0
    meta = {
        'built_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'window_days': window_days_meta,
        'smoothing': {
            'rolling_window': args.smooth_window,
            'regularization_alpha': args.smooth_alpha,
        },
        'trim_percentiles': {
            'top': args.trim_top,
            'bottom': args.trim_bottom,
        },
    }
    data_window_meta = {
        'actual': {
            'start_ms': actual_start_ms,
            'end_ms': actual_end_ms,
            'start': ms_to_iso(actual_start_ms),
            'end': ms_to_iso(actual_end_ms),
        }
    }
    if split_info:
        data_window_meta['config'] = split_info.configured_window
        meta['split'] = split_info.split_metadata
    meta['data_window'] = data_window_meta
    if args.symbol:
        out_data = {
            str(args.symbol): {k: v.tolist() for k, v in multipliers.items()},
            'hour_of_week_definition': '0=Monday 00:00 UTC',
            'metadata': meta,
        }
    else:
        out_data = {k: v.tolist() for k, v in multipliers.items()}
        out_data['hour_of_week_definition'] = '0=Monday 00:00 UTC'
        out_data['metadata'] = meta
    backup_dir = Path('data/latency/backups')
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    temp_file = tempfile.NamedTemporaryFile(
        'w', delete=False, dir=out_path.parent, prefix=f'.{out_path.stem}.', suffix='.tmp'
    )
    try:
        json.dump(out_data, temp_file, indent=2)
        temp_file.flush()
        os.fsync(temp_file.fileno())
    finally:
        temp_file.close()
    temp_path = Path(temp_file.name)
    if out_path.exists():
        backup_path = backup_dir / f'{out_path.stem}-{timestamp}{out_path.suffix}'
        shutil.copy2(out_path, backup_path)
        print(f'Backed up previous config to {backup_path}')
    os.replace(temp_path, out_path)
    if imputed:
        for key, hours in imputed.items():
            print(f'Imputed {key} multipliers for hours: {sorted(hours)}')
    checksum_path = write_checksum(data_path)
    print(f'Saved seasonality multipliers to {out_path} (backups in {backup_dir})')
    print(f'Input data checksum written to {checksum_path}')


if __name__ == '__main__':
    main()
