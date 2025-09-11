#!/usr/bin/env python3
"""Convert legacy seasonality multipliers JSON to the new format.

Legacy files may contain a bare list or a mapping with a single
``"multipliers"`` key. This utility rewrites them into a mapping with a
specific target key (defaults to ``"liquidity"``).
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Path to legacy multipliers JSON")
    p.add_argument("output", help="Where to write the converted JSON")
    p.add_argument(
        "--key",
        default="liquidity",
        help="Destination key for the multipliers in the new file",
    )
    args = p.parse_args()

    src = Path(args.input)
    data = json.loads(src.read_text())
    if isinstance(data, list):
        arr = data
    elif isinstance(data, dict):
        arr = (
            data.get("multipliers")
            or data.get("liquidity")
            or data.get("latency")
            or data.get("spread")
        )
    else:
        raise ValueError("Unsupported legacy format")

    if not isinstance(arr, list):
        raise ValueError("No multipliers array found in legacy file")

    out = {args.key: arr}
    dst = Path(args.output)
    dst.write_text(json.dumps(out, indent=2))
    print(f"Wrote {dst}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
