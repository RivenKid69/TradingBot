import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from services.metrics import (
    read_any,
    calculate_metrics,
    compute_equity_metrics,
    equity_from_trades,
)


def _bucket_stats(df: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """Return per-order-size bucket spread/slippage statistics."""
    required = {"order_size", "spread_bps", "slippage_bps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    labels, bins = pd.qcut(
        df["order_size"], quantiles, labels=False, retbins=True, duplicates="drop"
    )
    df = df.assign(_bucket=labels)
    stats = (
        df.groupby("_bucket")[{"spread_bps", "slippage_bps"}]
        .agg(["mean", "median"])
        .rename(
            columns={
                ("spread_bps", "mean"): "spread_bps_mean",
                ("spread_bps", "median"): "spread_bps_median",
                ("slippage_bps", "mean"): "slippage_bps_mean",
                ("slippage_bps", "median"): "slippage_bps_median",
            }
        )
    )
    stats.columns = stats.columns.droplevel(0)
    stats = stats.reset_index(drop=True)
    mids = (bins[:-1] + bins[1:]) / 2
    stats["order_size_mid"] = mids[: len(stats)]
    return stats[
        [
            "order_size_mid",
            "spread_bps_mean",
            "spread_bps_median",
            "slippage_bps_mean",
            "slippage_bps_median",
        ]
    ]


def _latency_stats(df: pd.DataFrame) -> dict:
    """Return latency percentiles in milliseconds."""
    if "latency_ms" not in df.columns:
        raise ValueError("missing 'latency_ms' column")
    latencies = df["latency_ms"].dropna()
    p50, p95 = np.percentile(latencies, [50, 95])
    return {"p50_ms": float(p50), "p95_ms": float(p95)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reality check report for simulated vs benchmark logs",
    )
    parser.add_argument(
        "--trades", required=True, help="Path to simulated trade log (CSV or Parquet)",
    )
    parser.add_argument(
        "--historical-trades",
        required=True,
        help="Path to historical trade log (CSV or Parquet)",
    )
    parser.add_argument(
        "--equity",
        required=False,
        help="Path to simulated equity log (CSV or Parquet). If omitted, equity is built from trades",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark equity log (CSV or Parquet)",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=10,
        help="Number of order size quantiles for bucket stats",
    )
    args = parser.parse_args()

    trades_path = Path(args.trades)
    trades_df = read_any(trades_path.as_posix())
    hist_trades_df = read_any(args.historical_trades)

    equity_df = read_any(args.equity) if args.equity else equity_from_trades(trades_df)
    benchmark_df = read_any(args.benchmark)

    sim_metrics = calculate_metrics(trades_df, equity_df)
    benchmark_metrics = compute_equity_metrics(benchmark_df).to_dict()
    sim_latency = _latency_stats(trades_df)
    hist_latency = _latency_stats(hist_trades_df)

    sim_buckets = _bucket_stats(trades_df, args.quantiles)
    sim_buckets.insert(0, "dataset", "simulation")
    hist_buckets = _bucket_stats(hist_trades_df, args.quantiles)
    hist_buckets.insert(0, "dataset", "historical")
    bucket_df = pd.concat([sim_buckets, hist_buckets], ignore_index=True)

    out_dir = trades_path.parent
    out_base = out_dir / "sim_reality_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_base = out_dir / "sim_reality_check_buckets"
    bucket_csv = bucket_base.with_suffix(".csv")
    bucket_json = bucket_base.with_suffix(".json")
    bucket_png = bucket_base.with_suffix(".png")

    bucket_df.to_csv(bucket_csv, index=False)
    bucket_df.to_json(bucket_json, orient="records", indent=2)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, grp in bucket_df.groupby("dataset"):
            axes[0].plot(grp["order_size_mid"], grp["spread_bps_mean"], label=name)
            axes[1].plot(grp["order_size_mid"], grp["slippage_bps_mean"], label=name)
        axes[0].set_xlabel("order size")
        axes[0].set_ylabel("spread (bps)")
        axes[1].set_xlabel("order size")
        axes[1].set_ylabel("slippage (bps)")
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        fig.savefig(bucket_png)
        plt.close(fig)
    except Exception:
        pass

    latency_summary = {"simulation": sim_latency, "historical": hist_latency}
    summary = {
        "simulation": sim_metrics,
        "benchmark": benchmark_metrics,
        "latency_ms": latency_summary,
    }

    json_path = out_base.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = out_base.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# Simulation Reality Check\n\n")
        f.write("## Simulation Metrics\n")
        for section, metrics in sim_metrics.items():
            f.write(f"### {section.capitalize()}\n")
            for k, v in metrics.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        f.write("## Benchmark Metrics\n")
        for k, v in benchmark_metrics.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")
        f.write("## Latency Metrics\n")
        for name, stats in latency_summary.items():
            f.write(f"### {name.capitalize()}\n")
            for k, v in stats.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

    print(f"Saved reports to {json_path} and {md_path}")
    print(f"Saved bucket stats to {bucket_csv} and {bucket_png}")


if __name__ == "__main__":
    main()

