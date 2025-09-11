#!/usr/bin/env python3
"""Edit specific hourly multipliers within a JSON file.

The JSON file must contain an array of length 168 under a specific key
(e.g. ``"liquidity"`` or ``"latency"``). This CLI allows overriding
individual hour-of-week multipliers by specifying ``HOUR=VALUE`` pairs via
``--set`` arguments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

HOURS_IN_WEEK = 168
SEASONALITY_MULT_MIN = 0.1
SEASONALITY_MULT_MAX = 10.0


def _parse_set(value: str) -> Tuple[int, float]:
    """Parse ``HOUR=VALUE`` pair from command line."""
    try:
        hour_str, val_str = value.split("=", 1)
        hour = int(hour_str)
        val = float(val_str)
    except ValueError as exc:  # pragma: no cover - argparse handles
        raise argparse.ArgumentTypeError("Expected HOUR=VALUE") from exc
    return hour, val


def _load_array(data: dict, key: str | None) -> Tuple[list, dict]:
    """Return multipliers array and the container holding it."""
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    if not key or key not in data:
        raise ValueError(f"Key '{key}' not found in JSON file")
    arr = data[key]
    if not isinstance(arr, list):
        raise ValueError("Multipliers must be stored as a list")
    return arr, data


def _apply_overrides(arr: list, pairs: Iterable[Tuple[int, float]]) -> None:
    for hour, val in pairs:
        if not 0 <= hour < HOURS_IN_WEEK:
            raise ValueError(f"Hour {hour} out of range 0..{HOURS_IN_WEEK - 1}")
        if not SEASONALITY_MULT_MIN <= val <= SEASONALITY_MULT_MAX:
            raise ValueError(
                f"Value {val} out of range {SEASONALITY_MULT_MIN}..{SEASONALITY_MULT_MAX}"
            )
        arr[hour] = float(val)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", help="Path to multipliers JSON file")
    parser.add_argument(
        "--key",
        required=True,
        help="Key within JSON mapping containing the multipliers array",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        required=True,
        action="append",
        type=_parse_set,
        help="Override in the form HOUR=VALUE; can be repeated",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write updated JSON (defaults to overwriting input)",
    )
    args = parser.parse_args()

    path = Path(args.json)
    data = json.loads(path.read_text())

    arr, _ = _load_array(data, args.key)
    if len(arr) != HOURS_IN_WEEK:
        raise ValueError(f"Multipliers array must contain {HOURS_IN_WEEK} values")

    _apply_overrides(arr, args.sets)

    out_path = Path(args.output) if args.output else path
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
