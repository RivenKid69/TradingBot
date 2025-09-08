"""CLI wrapper for T-cost calibration service."""

from __future__ import annotations

import argparse

from service_calibrate_tcost import from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate T-cost parameters")
    parser.add_argument("--config", required=True, help="Path to legacy_sandbox.yaml")
    parser.add_argument("--out", required=True, help="Where to store JSON report")
    args = parser.parse_args()

    from_config(args.config, out=args.out)


if __name__ == "__main__":
    main()

