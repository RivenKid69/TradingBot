"""Generate deterministic portfolio weights from model scores.

This module keeps backwards-compatible helpers used in the unit tests while
repurposing the CLI to emit target weights produced by
:class:`~portfolio_allocator.DeterministicPortfolioAllocator`.  The workflow no
longer depends on Gym environments or discrete action wrappers – instead the
allocator operates directly on score tables and writes the resulting weights to
an artifact file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core_config import TrainConfig, load_config
from portfolio_allocator import DeterministicPortfolioAllocator
from scripts.offline_utils import load_offline_payload, resolve_split_bundle


def _coerce_timestamp(value) -> int | None:
    """Convert supported timestamp representations to Unix seconds."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "none":
            return None
        try:
            return int(text)
        except ValueError:
            try:
                ts = pd.Timestamp(text)
            except Exception as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"Unable to parse timestamp string '{value}'.") from exc
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return int(ts.value // 10**9)
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.value // 10**9)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return None
        number = float(value)
        if abs(number) >= 1e11:
            seconds = number / 1000.0
        else:
            seconds = number
        try:
            return int(seconds)
        except (OverflowError, ValueError):
            return None
    raise ValueError(f"Unsupported timestamp value: {value!r}")


def _normalize_interval(item) -> tuple[int | None, int | None]:
    if item is None:
        return (None, None)
    if isinstance(item, Mapping):
        start = item.get("start_ts")
        if start is None:
            start = item.get("start") or item.get("from")
        end = item.get("end_ts")
        if end is None:
            end = item.get("end") or item.get("to")
    elif isinstance(item, Sequence) and len(item) == 2:
        start, end = item
    else:
        raise TypeError(f"Unsupported interval specification: {item!r}")
    start_ts = _coerce_timestamp(start)
    end_ts = _coerce_timestamp(end)
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError(f"Invalid interval with start {start_ts} after end {end_ts}")
    return (start_ts, end_ts)


def _extract_offline_split_overrides(
    payload: Mapping[str, Any] | None,
    dataset_key: str,
    *,
    fallback_split: str = "time",
) -> dict[str, list[dict[str, int | None]]]:
    """Extract inline time split overrides from an offline payload."""

    if payload is None:
        return {}

    datasets = payload.get("datasets") if isinstance(payload, Mapping) else None
    dataset_entry = datasets.get(dataset_key) if isinstance(datasets, Mapping) else None

    def _select_split_block(entry: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not isinstance(entry, Mapping):
            return None
        splits_block = entry.get("splits")
        if isinstance(splits_block, Mapping):
            return splits_block
        direct = {k: entry.get(k) for k in ("train", "val", "test") if entry.get(k) is not None}
        return direct or None

    def _pick_split_entry(block: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not isinstance(block, Mapping) or not block:
            return None
        if dataset_key in block and isinstance(block[dataset_key], Mapping):
            return block[dataset_key]
        if fallback_split in block and isinstance(block[fallback_split], Mapping):
            return block[fallback_split]
        for value in block.values():
            if isinstance(value, Mapping):
                return value
        return None

    split_entry = _pick_split_entry(_select_split_block(dataset_entry))
    if split_entry is None:
        split_entry = _pick_split_entry(payload.get("splits") if isinstance(payload, Mapping) else None)

    if not isinstance(split_entry, Mapping):
        return {}

    overrides: dict[str, list[dict[str, int | None]]] = {}
    for phase, entries in split_entry.items():
        if phase not in {"train", "val", "test"}:
            continue
        if isinstance(entries, (list, tuple)):
            iterable: Sequence[Any] = entries
        else:
            iterable = (entries,)
        normalized_items: list[dict[str, int | None]] = []
        for item in iterable:
            try:
                start_ts, end_ts = _normalize_interval(item)
            except Exception:
                continue
            normalized_items.append({"start_ts": start_ts, "end_ts": end_ts})
        if normalized_items:
            overrides[phase] = normalized_items
    return overrides


def _load_time_splits(data_cfg) -> tuple[str | None, dict[str, list[tuple[int | None, int | None]]]]:
    """Derive train/val/test time windows from config or an external manifest."""

    splits: dict[str, list[tuple[int | None, int | None]]] = {"train": [], "val": [], "test": []}
    version: str | None = getattr(data_cfg, "split_version", None)

    overrides = getattr(data_cfg, "split_overrides", None)
    if isinstance(overrides, Mapping):
        for phase, entries in overrides.items():
            if phase not in splits:
                splits[phase] = []
            if isinstance(entries, (list, tuple)):
                iterable = entries
            else:
                iterable = [entries]
            splits[phase].extend(_normalize_interval(item) for item in iterable)

    split_path = getattr(data_cfg, "split_path", None)
    if split_path:
        manifest_path = Path(split_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_path}")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            if manifest_path.suffix.lower() in (".yaml", ".yml"):
                raw = yaml.safe_load(fh) or {}
            else:
                raw = json.load(fh)
        version = raw.get("version") or raw.get("name") or version or manifest_path.stem
        raw_splits = raw.get("splits")
        if raw_splits is None:
            raw_splits = {k: raw.get(k) for k in ("train", "val", "test") if raw.get(k) is not None}
        if raw_splits is None:
            raise ValueError("Split manifest must contain 'splits' or per-phase keys (train/val/test).")
        for phase in ("train", "val", "test"):
            entries = raw_splits.get(phase)
            if entries is None:
                continue
            if isinstance(entries, (list, tuple)):
                iterable = entries
            else:
                iterable = [entries]
            splits[phase].extend(_normalize_interval(item) for item in iterable)

    for phase in ("train", "val", "test"):
        start_attr = getattr(data_cfg, f"{phase}_start_ts", None)
        end_attr = getattr(data_cfg, f"{phase}_end_ts", None)
        if start_attr is not None or end_attr is not None:
            splits[phase].append(_normalize_interval({"start_ts": start_attr, "end_ts": end_attr}))

    if not splits["train"]:
        fallback = _normalize_interval({"start_ts": getattr(data_cfg, "start_ts", None), "end_ts": getattr(data_cfg, "end_ts", None)})
        if fallback != (None, None):
            splits["train"].append(fallback)

    for phase, items in splits.items():
        cleaned = [item for item in items if not (item[0] is None and item[1] is None)]
        cleaned.sort(key=lambda it: it[0] if it[0] is not None else -float("inf"))
        splits[phase] = cleaned

    if not splits["train"]:
        raise ValueError("Training split is empty: provide train_start/train_end or a manifest")

    return version, splits


def _apply_role_column(
    df: pd.DataFrame,
    intervals: dict[str, list[tuple[int | None, int | None]]],
    timestamp_column: str,
    role_column: str,
) -> tuple[pd.DataFrame, bool]:
    """Annotate ``df`` with walk-forward roles using the provided intervals."""

    if timestamp_column not in df.columns:
        raise KeyError(f"DataFrame is missing timestamp column '{timestamp_column}'")
    ts = pd.to_numeric(df[timestamp_column], errors="coerce")
    roles = pd.Series(["none"] * len(df), index=df.index, dtype=object)

    for phase in ("train", "val", "test"):
        masks = []
        for start, end in intervals.get(phase, []):
            cur = pd.Series(True, index=df.index)
            if start is not None:
                cur &= ts >= start
            if end is not None:
                cur &= ts <= end
            masks.append(cur)
        if not masks:
            continue
        phase_mask = masks[0]
        for extra in masks[1:]:
            phase_mask |= extra
        assignable = (roles == "none") & phase_mask
        if assignable.any():
            roles.loc[assignable] = phase

    inferred_test = False
    if not intervals.get("test"):
        leftover = roles == "none"
        if leftover.any():
            roles.loc[leftover] = "test"
            inferred_test = True

    df_out = df.copy()
    df_out[role_column] = roles.values
    return df_out, inferred_test


def _phase_bounds(mapping: dict[str, pd.DataFrame], ts_col: str) -> tuple[int | None, int | None]:
    start: int | None = None
    end: int | None = None
    for df in mapping.values():
        ts = pd.to_numeric(df[ts_col], errors="coerce").dropna()
        if ts.empty:
            continue
        cur_start = int(ts.min())
        cur_end = int(ts.max())
        start = cur_start if start is None or cur_start < start else start
        end = cur_end if end is None or cur_end > end else end
    return start, end


def _fmt_ts(ts: int | None) -> str:
    if ts is None:
        return "None"
    return pd.to_datetime(int(ts), unit="s", utc=True).isoformat()


def _format_interval(interval: tuple[int | None, int | None]) -> str:
    start, end = interval
    return f"[{_fmt_ts(start)} .. {_fmt_ts(end)}]"


def _ensure_validation_split_present(
    dfs_with_roles: dict[str, pd.DataFrame],
    intervals: dict[str, list[tuple[int | None, int | None]]],
    timestamp_column: str,
    role_column: str,
) -> None:
    """Abort execution when the validation split yields zero rows."""

    val_rows = 0
    for df in dfs_with_roles.values():
        val_rows += int((df[role_column].astype(str) == "val").sum())
    if val_rows > 0:
        return

    configured = intervals.get("val", [])
    configured_desc = ", ".join(_format_interval(it) for it in configured) if configured else "(not configured)"
    observed_start, observed_end = _phase_bounds(dfs_with_roles, timestamp_column)
    coverage_desc = f"[{_fmt_ts(observed_start)} .. {_fmt_ts(observed_end)}]"

    msg_lines = [
        "Validation split is empty after applying configured intervals.",
        f"Configured validation intervals: {configured_desc}",
        f"Observed data coverage: {coverage_desc}",
    ]

    if configured:
        overlap_detected = None
        if observed_start is not None and observed_end is not None:
            overlap_detected = False
            for start, end in configured:
                start_cmp = observed_start if start is None else start
                end_cmp = observed_end if end is None else end
                if start_cmp <= observed_end and end_cmp >= observed_start:
                    overlap_detected = True
                    break
        if overlap_detected is False:
            msg_lines.append(
                "Configured validation window does not overlap with available data; regenerate or refresh the offline dataset."
            )
        else:
            msg_lines.append(
                "Adjust the validation split configuration or refresh the offline dataset to include the desired range."
            )
    else:
        msg_lines.append("Provide validation split overrides or refresh the offline dataset to include validation data.")

    raise SystemExit("\n".join(msg_lines))


def _load_scores_frame(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")

    if file_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Scores file is empty")

    if {"symbol", "score"}.issubset(df.columns):
        if "timestamp" in df.columns:
            latest_ts = df["timestamp"].max()
            df = df[df["timestamp"] == latest_ts]
        latest_scores = df.groupby("symbol")[["score"]].last()["score"].astype(float)
        return latest_scores.to_frame().T

    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c.lower() not in {"timestamp", "time", "ts"}]
    if len(numeric_cols) == 0:
        raise ValueError("Scores file does not contain numeric columns")
    return df[numeric_cols]


def _load_prev_weights(path: str | None) -> pd.Series | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Previous weights file not found: {path}")
    if file_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    if df.empty:
        return None
    if {"symbol", "weight"}.issubset(df.columns):
        series = pd.Series(df["weight"].astype(float).values, index=df["symbol"].astype(str))
    else:
        numeric_cols = [
            c for c in df.select_dtypes(include=["number"]).columns if c.lower() not in {"timestamp", "time", "ts"}
        ]
        if not numeric_cols:
            return None
        series = df[numeric_cols].iloc[0]
    return series.astype(float)


def _save_weights(weights: pd.Series, path: Path) -> None:
    if path.suffix.lower() in {".parquet", ".pq"}:
        weights.to_frame("weight").to_parquet(path)
    else:
        weights.to_frame("weight").to_csv(path, index_label="symbol")


def _resolve_output_path(cfg: TrainConfig, cli_output: str | None) -> Path:
    if cli_output:
        return Path(cli_output).expanduser()
    base_dir = Path(getattr(cfg, "artifacts_dir", "artifacts")).expanduser()
    return base_dir / "portfolio_weights.parquet"


def _extract_portfolio_params(cfg: TrainConfig) -> dict[str, float | int | None]:
    portfolio_cfg = getattr(cfg, "portfolio", None)
    params = {
        "top_n": None,
        "threshold": 0.0,
        "max_weight_per_symbol": 1.0,
        "max_gross_exposure": 1.0,
        "realloc_threshold": 0.0,
    }
    if portfolio_cfg is None:
        return params
    top_n = getattr(portfolio_cfg, "top_n", None)
    if top_n is not None:
        params["top_n"] = int(top_n)
    params["threshold"] = float(getattr(portfolio_cfg, "threshold", params["threshold"]))
    params["max_weight_per_symbol"] = float(getattr(portfolio_cfg, "max_weight_per_symbol", params["max_weight_per_symbol"]))
    params["max_gross_exposure"] = float(getattr(portfolio_cfg, "max_gross_exposure", params["max_gross_exposure"]))
    params["realloc_threshold"] = float(getattr(portfolio_cfg, "realloc_threshold", params["realloc_threshold"]))
    return params


def _should_resolve_offline_bundle(value: str | None) -> bool:
    if value is None:
        return False
    text = value.strip().lower()
    return bool(text and text not in {"none", "null"})


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate deterministic portfolio weights")
    parser.add_argument("--config", default="configs/config_train.yaml", help="Path to YAML config")
    parser.add_argument("--scores", help="Path to CSV/Parquet with model scores")
    parser.add_argument("--output", default=None, help="Destination for portfolio weights (CSV or Parquet)")
    parser.add_argument("--prev-weights", default=None, help="Optional previous weights for reallocation threshold checks")
    parser.add_argument("--dataset-split", default="val", help="Dataset split declared in the offline config")
    parser.add_argument("--offline-config", default="configs/offline.yaml", help="Offline dataset configuration")
    parser.add_argument("--liquidity-seasonality", default=None, help="Override path to liquidity seasonality coefficients")
    args = parser.parse_args(argv)

    split_key = args.dataset_split or ""
    offline_bundle = None
    offline_payload: Mapping[str, Any] | None = None
    offline_split_overrides: dict[str, list[dict[str, int | None]]] = {}
    if _should_resolve_offline_bundle(split_key):
        try:
            offline_bundle = resolve_split_bundle(args.offline_config, split_key)
        except FileNotFoundError as exc:
            raise SystemExit(f"Offline config not found: {args.offline_config}") from exc
        except KeyError as exc:
            raise SystemExit(f"Dataset split '{split_key}' not found in offline config {args.offline_config}") from exc
        except ValueError as exc:
            raise SystemExit(f"Failed to resolve offline split '{split_key}': {exc}") from exc
        offline_payload = load_offline_payload(args.offline_config)
        offline_split_overrides = _extract_offline_split_overrides(offline_payload, split_key, fallback_split="time")

    seasonality_path = args.liquidity_seasonality
    seasonality_hash: str | None = None
    if offline_bundle is not None:
        if offline_bundle.version:
            print(f"Resolved offline dataset split '{offline_bundle.name}' version {offline_bundle.version}")
        else:
            print(f"Resolved offline dataset split '{offline_bundle.name}'")
        seasonality_art = offline_bundle.artifacts.get("seasonality")
        if seasonality_art:
            if seasonality_path is None:
                seasonality_path = seasonality_art.path.as_posix()
            raw_hash = seasonality_art.info.artifact.get("verification_hash")
            if raw_hash:
                seasonality_hash = str(raw_hash)
    if seasonality_path is None:
        seasonality_path = "configs/liquidity_seasonality.json"
    if not Path(seasonality_path).exists():
        raise FileNotFoundError(
            f"Liquidity seasonality file not found: {seasonality_path}. Run offline builders first."
        )
    if seasonality_hash:
        print(f"Seasonality verification hash: {seasonality_hash}")

    cfg = load_config(args.config)
    if not isinstance(cfg, TrainConfig):
        raise TypeError("Loaded config is not a TrainConfig; check the 'mode' field")
    if offline_split_overrides:
        setattr(cfg.data, "split_overrides", offline_split_overrides)

    if not args.scores:
        raise SystemExit("--scores is required to compute portfolio weights")

    scores_df = _load_scores_frame(args.scores)
    prev_weights = _load_prev_weights(args.prev_weights)
    params = _extract_portfolio_params(cfg)

    allocator = DeterministicPortfolioAllocator()
    weights = allocator.compute_weights(
        scores_df,
        prev_weights=prev_weights,
        top_n=params["top_n"],
        threshold=params["threshold"],
        max_weight_per_symbol=params["max_weight_per_symbol"],
        max_gross_exposure=params["max_gross_exposure"],
        realloc_threshold=params["realloc_threshold"],
    )

    output_path = _resolve_output_path(cfg, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_weights(weights, output_path)

    if weights.empty:
        print(f"No symbols passed the allocation filters. Empty weights saved to '{output_path}'.")
    else:
        gross = float(weights.abs().sum())
        print(f"Allocated {len(weights)} symbols with gross exposure {gross:.4f}. Output → {output_path}")

    return weights


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
