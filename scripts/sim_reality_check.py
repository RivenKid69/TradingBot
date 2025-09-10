import argparse
import json
from pathlib import Path

from services.metrics import (
    read_any,
    calculate_metrics,
    compute_equity_metrics,
    equity_from_trades,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reality check report for simulated vs benchmark logs"
    )
    parser.add_argument(
        "--trades", required=True, help="Path to simulated trade log (CSV or Parquet)"
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
    args = parser.parse_args()

    trades_path = Path(args.trades)
    trades_df = read_any(trades_path.as_posix())

    equity_df = read_any(args.equity) if args.equity else equity_from_trades(trades_df)
    benchmark_df = read_any(args.benchmark)

    sim_metrics = calculate_metrics(trades_df, equity_df)
    benchmark_metrics = compute_equity_metrics(benchmark_df).to_dict()

    summary = {"simulation": sim_metrics, "benchmark": benchmark_metrics}

    out_dir = trades_path.parent
    out_base = out_dir / "sim_reality_check"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Saved reports to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
