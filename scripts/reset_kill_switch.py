#!/usr/bin/env python3
"""Reset operational kill switch counters and remove flag file."""

from __future__ import annotations

import argparse
from pathlib import Path

from services import ops_kill_switch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete the kill switch flag file and reset counters."
    )
    parser.add_argument(
        "--flag-path",
        default="state/ops_kill_switch.flag",
        help="Path to kill switch flag file",
    )
    parser.add_argument(
        "--state-path",
        default="state/ops_state.json",
        help="Path to kill switch state file",
    )
    args = parser.parse_args()

    flag = Path(args.flag_path)
    try:
        flag.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass

    ops_kill_switch.init({"flag_path": args.flag_path, "state_path": args.state_path})
    ops_kill_switch.manual_reset()
    print("Kill switch counters reset and flag file removed")


if __name__ == "__main__":
    main()
