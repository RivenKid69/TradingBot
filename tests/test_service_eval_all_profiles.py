import sys
import pathlib
import pandas as pd

base = pathlib.Path(__file__).resolve().parent.parent
if "" in sys.path:
    sys.path.remove("")
if str(base) in sys.path:
    sys.path.remove(str(base))
sys.path.append(str(base))

from core_config import EvalConfig as RunEvalConfig, EvalInputConfig, ComponentSpec, Components, ExecutionProfile
from service_eval import from_config
import di_registry


def _make_trades_df():
    return pd.DataFrame({
        "ts_ms": [1, 2],
        "pnl": [1.0, -0.5],
        "side": ["BUY", "SELL"],
        "qty": [1, 1],
    })


def _make_equity_df():
    return pd.DataFrame({
        "ts_ms": [1, 2],
        "equity": [1.0, 0.5],
    })


def _dummy_components() -> Components:
    spec = ComponentSpec(target="x:y", params={})
    return Components(
        market_data=spec,
        executor=spec,
        feature_pipe=spec,
        policy=spec,
        risk_guards=spec,
    )


def test_from_config_all_profiles(monkeypatch, tmp_path):
    # prepare csv logs for each execution profile
    for prof in ExecutionProfile:
        _make_trades_df().to_csv(tmp_path / f"trades_{prof.value}.csv", index=False)
        _make_equity_df().to_csv(tmp_path / f"equity_{prof.value}.csv", index=False)

    cfg = RunEvalConfig(
        components=_dummy_components(),
        input=EvalInputConfig(
            trades_path=f"{tmp_path}/trades_<profile>.csv",
            equity_path=f"{tmp_path}/equity_<profile>.csv",
        ),
        logs_dir=str(tmp_path),
        artifacts_dir=str(tmp_path / "artifacts"),
        all_profiles=True,
    )

    monkeypatch.setattr(di_registry, "build_graph", lambda components, run_config=None: {})

    metrics = from_config(cfg, all_profiles=True)
    assert set(metrics.keys()) == {p.value for p in ExecutionProfile}
