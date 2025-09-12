import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from services.metrics import (
    read_any,
    calculate_metrics,
    compute_equity_metrics,
    equity_from_trades,
)


DEFAULT_SCENARIOS = {
    "Low": {"fee_mult": 0.5, "spread_mult": 0.5},
    "Med": {"fee_mult": 1.0, "spread_mult": 1.0},
    "High": {"fee_mult": 1.5, "spread_mult": 1.5},
}


def apply_fee_spread(
    trades_df: pd.DataFrame, fee_mult: float, spread_mult: float
) -> pd.DataFrame:
    """Return a copy of ``trades_df`` with fee and spread adjustments applied.

    Parameters
    ----------
    trades_df:
        DataFrame containing trade information with ``fees`` and ``spread_bps``
        columns.
    fee_mult:
        Multiplier applied to the ``fees`` column.
    spread_mult:
        Multiplier applied to the ``spread_bps`` column.

    Raises
    ------
    ValueError
        If either ``fees`` or ``spread_bps`` column is missing.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with adjusted ``fees`` and ``spread_bps``.
    """

    required = {"fees", "spread_bps"}
    missing = required - set(trades_df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = trades_df.copy()
    df["fees"] *= fee_mult
    df["spread_bps"] *= spread_mult
    return df


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


def _order_fill_stats(df: pd.DataFrame) -> dict:
    """Return fractions of partially filled and unfilled orders."""
    if "exec_status" not in df.columns:
        raise ValueError("missing 'exec_status' column")
    status = df["exec_status"].astype(str)
    total = len(status)
    if total == 0:
        return {"fraction_partially_filled": 0.0, "fraction_unfilled": 0.0}
    partial = (status == "PARTIALLY_FILLED").sum()
    unfilled = (status == "CANCELED").sum()
    return {
        "fraction_partially_filled": float(partial / total),
        "fraction_unfilled": float(unfilled / total),
    }


def _cancel_stats(df: pd.DataFrame) -> dict:
    """Return cancellation counts by reason and their shares."""
    if "exec_status" not in df.columns:
        raise ValueError("missing 'exec_status' column")
    if "meta" not in df.columns:
        raise ValueError("missing 'meta' column")
    cancelled = df[df["exec_status"].astype(str) == "CANCELED"]
    total = len(cancelled)
    if total == 0:
        return {"counts": {}, "shares": {}}

    def _parse(m: Any) -> dict:
        if isinstance(m, dict):
            return m
        if isinstance(m, str):
            try:
                return json.loads(m)
            except Exception:
                try:
                    import ast

                    return ast.literal_eval(m)
                except Exception:
                    return {}
        return {}

    reasons = cancelled["meta"].apply(lambda m: str(_parse(m).get("reason", "OTHER")))
    counts = reasons.value_counts().to_dict()
    shares = {k: float(v / total) for k, v in counts.items()}
    return {"counts": counts, "shares": shares}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reality check report for simulated vs benchmark logs",
    )
    parser.add_argument(
        "--trades",
        required=True,
        help="Path to simulated trade log (CSV or Parquet)",
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
        "--kpi-thresholds",
        default="benchmarks/sim_kpi_thresholds.json",
        help="Path to JSON file with KPI thresholds",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=10,
        help="Number of order size quantiles for bucket stats",
    )
    parser.add_argument(
        "--scenario-config",
        default=None,
        help="Path to JSON file with scenario configuration",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated list of scenarios to include",
    )
    parser.add_argument(
        "--sensitivity-threshold",
        type=float,
        default=0.2,
        help="Relative KPI change threshold triggering sensitivity flag",
    )
    args = parser.parse_args()

    scenarios = DEFAULT_SCENARIOS.copy()
    if args.scenario_config:
        try:
            with open(Path(args.scenario_config)) as f:
                scenarios.update(json.load(f))
        except Exception:
            pass
    scenario_names = (
        [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios
        else list(scenarios.keys())
    )

    trades_path = Path(args.trades)
    trades_df = read_any(trades_path.as_posix())
    hist_trades_df = read_any(args.historical_trades)

    baseline_params = scenarios.get("Med", {})
    base_df = apply_fee_spread(
        trades_df.copy(),
        baseline_params.get("fee_mult", 1.0),
        baseline_params.get("spread_mult", 1.0),
    )
    base_eq = equity_from_trades(base_df)
    baseline_metrics = calculate_metrics(base_df, base_eq)

    scenario_metrics: dict[str, dict] = {"Med": baseline_metrics}
    for name in scenario_names:
        if name == "Med":
            continue
        params = scenarios.get(name, {})
        df = trades_df.copy()
        df = apply_fee_spread(
            df,
            params.get("fee_mult", 1.0),
            params.get("spread_mult", 1.0),
        )
        eq_df = equity_from_trades(df)
        scenario_metrics[name] = calculate_metrics(df, eq_df)

    baseline_kpi = baseline_metrics["equity"].get("pnl_total", 0.0)
    degradation_rows = []
    flags: dict[str, str] = {}
    for name, metrics in scenario_metrics.items():
        kpi = metrics.get("equity", {}).get("pnl_total", float("nan"))
        rel_change = (
            (kpi - baseline_kpi) / baseline_kpi if baseline_kpi else float("nan")
        )
        flag = name != "Med" and abs(rel_change) > args.sensitivity_threshold
        degradation_rows.append(
            {
                "scenario": name,
                "kpi": kpi,
                "relative_change": rel_change,
                "flag": flag,
            }
        )
        if flag:
            flags[f"scenario.{name}"] = "чрезмерная чувствительность"

    degradation_ranking = pd.DataFrame(degradation_rows).sort_values(
        "kpi", ascending=False
    )

    equity_df = read_any(args.equity) if args.equity else equity_from_trades(trades_df)
    benchmark_df = read_any(args.benchmark)

    sim_metrics = calculate_metrics(trades_df, equity_df)
    benchmark_metrics = compute_equity_metrics(benchmark_df).to_dict()
    sim_latency = _latency_stats(trades_df)
    hist_latency = _latency_stats(hist_trades_df)
    sim_fill = _order_fill_stats(trades_df)
    hist_fill = _order_fill_stats(hist_trades_df)
    sim_cancel = _cancel_stats(trades_df)
    hist_cancel = _cancel_stats(hist_trades_df)

    # Load KPI thresholds and check simulated metrics
    thresholds = {}
    try:
        with open(Path(args.kpi_thresholds)) as f:
            thresholds = json.load(f)
    except Exception:
        thresholds = {}

    kpi_values = {
        "equity": sim_metrics.get("equity", {}),
        "trades": sim_metrics.get("trades", {}),
        "latency_ms": sim_latency,
        "order_fill": sim_fill,
        "cancellations": sim_cancel,
    }

    def _check(values: dict, specs: dict, prefix: str = "") -> None:
        for key, spec in specs.items():
            if isinstance(spec, dict) and {"min", "max"} <= set(spec.keys()):
                actual = values.get(key)
                if actual is None or not (spec["min"] <= actual <= spec["max"]):
                    flags[prefix + key] = "нереалистично"
            elif isinstance(spec, dict):
                _check(values.get(key, {}), spec, prefix + key + ".")

    _check(kpi_values, thresholds)

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

    degr_base = out_dir / "sim_reality_check_degradation"
    degr_csv = degr_base.with_suffix(".csv")
    degr_json = degr_base.with_suffix(".json")
    degradation_ranking.to_csv(degr_csv, index=False)
    degradation_ranking.to_json(degr_json, orient="records", indent=2)

    scen_base = out_dir / "sim_reality_check_scenarios"
    scen_csv = scen_base.with_suffix(".csv")
    scen_json = scen_base.with_suffix(".json")
    scen_df = pd.json_normalize(
        [
            {
                "scenario": name,
                **{f"equity.{k}": v for k, v in m.get("equity", {}).items()},
                **{f"trades.{k}": v for k, v in m.get("trades", {}).items()},
            }
            for name, m in scenario_metrics.items()
        ]
    )
    scen_df.to_csv(scen_csv, index=False)
    scen_df.to_json(scen_json, orient="records", indent=2)

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
    fill_summary = {"simulation": sim_fill, "historical": hist_fill}
    cancel_summary = {"simulation": sim_cancel, "historical": hist_cancel}
    summary = {
        "simulation": sim_metrics,
        "benchmark": benchmark_metrics,
        "latency_ms": latency_summary,
        "order_fill": fill_summary,
        "cancellations": cancel_summary,
        "scenario_metrics": scenario_metrics,
        "degradation_ranking": degradation_ranking.to_dict(orient="records"),
        "flags": flags,
    }

    json_path = out_base.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = out_base.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# Simulation Reality Check\n\n")
        if flags:
            f.write("## KPI Flags\n")
            for k, v in flags.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        f.write("## Sensitivity Analysis\n")
        try:
            f.write(degradation_ranking.to_markdown(index=False))
            f.write("\n\n")
        except Exception:
            for row in degradation_ranking.to_dict(orient="records"):
                f.write(
                    "- {scenario}: kpi={kpi}, change={change:.2%}, flag={flag}\n".format(
                        scenario=row["scenario"],
                        kpi=row["kpi"],
                        change=row["relative_change"],
                        flag=row.get("flag", False),
                    )
                )
            f.write("\n")
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
        f.write("## Order Fill KPIs\n")
        for name, stats in fill_summary.items():
            f.write(f"### {name.capitalize()}\n")
            for k, v in stats.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        f.write("## Cancellation Breakdown\n")
        for name, stats in cancel_summary.items():
            f.write(f"### {name.capitalize()}\n")
            for reason, cnt in stats.get("counts", {}).items():
                share = stats.get("shares", {}).get(reason, 0.0)
                f.write(f"- {reason}: {cnt} ({share:.2%})\n")
            f.write("\n")

    print(f"Saved reports to {json_path} and {md_path}")
    print(f"Saved bucket stats to {bucket_csv} and {bucket_png}")
    print(f"Saved sensitivity analysis to {degr_csv}")
    print(f"Saved scenario metrics to {scen_csv}")
    if flags:
        print("Unrealistic KPIs detected:")
        for k, v in flags.items():
            print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
