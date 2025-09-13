import os
import sys
import logging  # noqa: F401 ensures stdlib is loaded before path append

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO)

from apply_no_trade_mask import main


def run(mode: str | None) -> None:
    out = f"/tmp/no_trade_sample_{mode or 'mask'}.csv"
    sys.argv = [
        "no-trade-mask",
        "--data", os.path.join(REPO, "tests/data/no_trade_sample.csv"),
        "--out", out,
        "--sandbox_config", os.path.join(REPO, "configs/legacy_sandbox.yaml"),
        "--timeframe", "1m",
        "--close-lag-ms", "0",
    ]
    if mode:
        sys.argv += ["--mode", mode]
    else:
        sys.argv += ["--mask-only"]
    main()


if __name__ == "__main__":
    run("drop")
    run("weight")
    run(None)
