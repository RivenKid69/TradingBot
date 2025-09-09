import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load historical OHLCV data from CSV/Parquet."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def label_regimes(df: pd.DataFrame) -> pd.Series:
    """Label market regimes using volatility and trend filters."""
    df = df.copy()
    df["ret"] = np.log(df["close"]).diff()
    df["vol"] = df["ret"].rolling(24).std()
    df["trend"] = df["ret"].rolling(24).mean()
    vol_hi = df["vol"] > df["vol"].quantile(0.66)
    vol_lo = df["vol"] < df["vol"].quantile(0.33)
    trend_hi = df["trend"].abs() > df["trend"].abs().quantile(0.66)
    liq_lo = df["volume"].rolling(24).mean() < df["volume"].rolling(24).mean().quantile(0.25)
    regime = pd.Series("NORMAL", index=df.index)
    regime[trend_hi] = "STRONG_TREND"
    regime[vol_lo & ~trend_hi] = "CHOPPY_FLAT"
    regime[liq_lo] = "ILLIQUID"
    return regime


def estimate_params(df: pd.DataFrame, regime: pd.Series) -> dict:
    out = {}
    total = len(df)
    returns = np.log(df["close"]).diff()
    spread = df.get("spread")
    if spread is None:
        spread = (df["high"] - df["low"]) / df["close"]
    for name in ["NORMAL", "CHOPPY_FLAT", "STRONG_TREND", "ILLIQUID"]:
        mask = regime == name
        r = returns[mask].dropna()
        if len(r) < 2:
            mu = sigma = kappa = 0.0
        else:
            mu = r.mean()
            sigma = r.std()
            prev = r.shift(1).dropna()
            corr = np.corrcoef(prev, r.loc[prev.index])[0, 1]
            kappa = max(0.0, -np.log(abs(corr))) if np.isfinite(corr) and abs(corr) < 1 else 0.0
        avg_vol = df.loc[mask, "volume"].mean()
        avg_spread = spread.loc[mask].mean()
        out[name] = {
            "mu": float(mu),
            "sigma": float(sigma),
            "kappa": float(kappa),
            "avg_volume": float(avg_vol if pd.notna(avg_vol) else 0.0),
            "avg_spread": float(avg_spread if pd.notna(avg_spread) else 0.0),
        }
    counts = regime.value_counts()
    probs = [float(counts.get(name, 0) / total) for name in ["NORMAL", "CHOPPY_FLAT", "STRONG_TREND", "ILLIQUID"]]
    out["regime_probs"] = probs
    # flash shocks
    shock_threshold = returns.std() * 5
    shocks = returns[np.abs(returns) > shock_threshold]
    out["flash_shock"] = {
        "probability": float(len(shocks) / max(len(returns), 1)),
        "magnitudes": np.abs(shocks).dropna().tolist(),
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to OHLCV data (csv or parquet)")
    p.add_argument("--out", default="configs/market_regimes.json", help="Output JSON path")
    args = p.parse_args()
    df = load_ohlcv(Path(args.data))
    regime = label_regimes(df)
    params = estimate_params(df, regime)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"regimes": {k: v for k, v in params.items() if k not in {"regime_probs", "flash_shock"}},
                   "regime_probs": params["regime_probs"],
                   "flash_shock": params["flash_shock"]}, f, indent=2)
    print(f"Saved regime parameters to {args.out}")


if __name__ == "__main__":
    main()
