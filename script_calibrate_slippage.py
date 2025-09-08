"""CLI wrapper for slippage calibration service."""

from __future__ import annotations

import argparse

from service_calibrate_slippage import from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate slippage coefficient")
    parser.add_argument("--config", required=True, help="Path to YAML config with trades settings")
    parser.add_argument("--out", required=True, help="Where to write JSON report")
    args = parser.parse_args()

    from_config(args.config, out=args.out)


if __name__ == "__main__":
    main()

