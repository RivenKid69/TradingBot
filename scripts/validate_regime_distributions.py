import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces

from wrappers.action_space import DictToMultiDiscreteActionWrapper


class _DummyEnv:
    """Simple environment used when real TradingEnv is unavailable."""

    def __init__(self):
        self._rng = np.random.default_rng(0)
        self._regime = "normal"
        # Mean values for each regime; std is 10% of the mean
        self._params = {
            "normal": {"volatility": 1.0, "volume": 100.0, "spread": 0.01},
            "choppy_flat": {"volatility": 0.5, "volume": 90.0, "spread": 0.015},
            "strong_trend": {"volatility": 1.5, "volume": 130.0, "spread": 0.02},
        }
        self.action_space = type("AS", (), {"sample": lambda self: 0})()

    def set_market_regime(self, regime: str = "normal", duration: int = 0):
        self._regime = regime

    def reset(self):
        return 0

    def step(self, action):
        p = self._params[self._regime]
        vol = self._rng.normal(p["volatility"], 0.1 * p["volatility"])
        volume = self._rng.normal(p["volume"], 0.1 * p["volume"])
        spread = self._rng.normal(p["spread"], 0.1 * p["spread"])
        info = {"volatility": float(vol), "volume": float(volume), "spread": float(spread)}
        return 0, 0.0, False, info


def _wrap_action_space_if_needed(env, bins_vol: int = 101):
    try:
        if isinstance(env.action_space, spaces.Dict):
            keys = set(getattr(env.action_space, "spaces", {}).keys())
            expected = {"price_offset_ticks", "ttl_steps", "type", "volume_frac"}
            if expected.issubset(keys):
                return DictToMultiDiscreteActionWrapper(env, bins_vol=bins_vol)
    except Exception:
        return env
    return env


def _set_regime(env, regime: str, duration: int):
    if hasattr(env, "env_method"):
        env.env_method("set_market_regime", regime=regime, duration=duration)
    elif hasattr(env, "set_market_regime"):
        env.set_market_regime(regime=regime, duration=duration)
    else:
        raise AttributeError("Environment does not support regime control")


def _collect_stats(env, regime: str, steps: int) -> Dict[str, np.ndarray]:
    _set_regime(env, regime, steps)
    vol, volume, spread = [], [], []
    obs = env.reset()
    for _ in range(steps):
        action = env.action_space.sample() if hasattr(env, "action_space") else None
        result = env.step(action)
        # Handle Gymnasium (5-tuple) and Gym (4-tuple)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        if isinstance(info, list):
            info = info[0]
        vol.append(info.get("volatility", 0.0))
        volume.append(info.get("volume", 0.0))
        spread.append(info.get("spread", 0.0))
        if np.any(done):
            obs = env.reset()
    return {
        "volatility": np.asarray(vol),
        "volume": np.asarray(volume),
        "spread": np.asarray(spread),
    }


def compare_regime_distributions(env, reference_path: str, n_steps: int = 1000, tolerance: float = 0.1,
                                 quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
                                 raise_on_fail: bool = True) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Compare sampled regime distributions with reference quantiles.

    Returns a dictionary of metrics and a boolean flag whether all metrics are within tolerance.
    """
    ref = json.loads(Path(reference_path).read_text())
    metrics: Dict[str, Dict[str, float]] = {}
    all_ok = True
    for regime, expected in ref.items():
        data = _collect_stats(env, regime, n_steps)
        regime_metrics: Dict[str, float] = {}
        for key in ("volatility", "volume", "spread"):
            sample = data[key]
            q_actual = np.quantile(sample, quantiles)
            q_ref = np.array([expected[key][f"q{int(q*100)}"] for q in quantiles])
            rel_diff = np.max(np.abs(q_actual - q_ref) / (np.abs(q_ref) + 1e-9))
            regime_metrics[f"{key}_qdiff"] = float(rel_diff)
            if rel_diff > tolerance:
                all_ok = False
        metrics[regime] = regime_metrics
    if raise_on_fail and not all_ok:
        raise RuntimeError("Regime distributions diverged beyond tolerance")
    return metrics, all_ok


def make_env(use_dummy: bool):
    if not use_dummy:
        try:
            from trading_patchnew import TradingEnv  # type: ignore
            return _wrap_action_space_if_needed(TradingEnv())
        except Exception:
            print("⚠️  Falling back to dummy environment")
    return _DummyEnv()


def main(argv=None) -> bool:
    parser = argparse.ArgumentParser(description="Validate regime distributions")
    parser.add_argument("--ref", default="configs/reference_regime_distributions.json", help="Path to reference JSON")
    parser.add_argument("--steps", type=int, default=1000, help="Number of samples per regime")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Maximum allowed relative diff")
    parser.add_argument("--use-dummy-env", action="store_true", help="Force using dummy environment")
    args = parser.parse_args(argv)

    env = make_env(args.use_dummy_env or os.getenv("USE_DUMMY_ENV") == "1")
    metrics, ok = compare_regime_distributions(env, args.ref, args.steps, args.tolerance, raise_on_fail=False)
    for regime, vals in metrics.items():
        print(f"Regime: {regime}")
        for name, value in vals.items():
            print(f"  {name}: {value:.4f}")
    if not ok:
        print("❌ Distribution check failed")
    else:
        print("✅ Distribution check passed")
    return ok


if __name__ == "__main__":  # pragma: no cover - CLI
    import sys
    success = main()
    if not success:
        sys.exit(1)
