import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.metrics import equity_from_trades
import scripts.sim_reality_check as sim


def test_apply_fee_spread_scaling():
    df = pd.DataFrame({'fees': [2.0], 'spread_bps': [10.0]})
    scaled = sim.apply_fee_spread(df, fee_mult=0.5, spread_mult=2.0)
    assert scaled.loc[0, 'fees'] == pytest.approx(1.0)
    assert scaled.loc[0, 'spread_bps'] == pytest.approx(20.0)


def test_main_sensitivity(tmp_path, monkeypatch):
    trades = pd.DataFrame({
        'ts_ms': [1, 2],
        'pnl': [5.0, 5.0],
        'fees': [1.0, 1.0],
        'spread_bps': [1.0, 1.0],
        'side': ['BUY', 'SELL'],
        'slippage_bps': [0.0, 0.0],
        'price': [100.0, 100.0],
        'qty': [1.0, 1.0],
    })

    trades_path = tmp_path / 'trades.csv'
    hist_path = tmp_path / 'hist.csv'
    bench_path = tmp_path / 'bench.csv'

    trades.to_csv(trades_path, index=False)
    trades.to_csv(hist_path, index=False)
    equity_from_trades(trades).to_csv(bench_path, index=False)

    # patch fee spread to also scale PnL for scenarios so KPI differs
    orig_apply = sim.apply_fee_spread

    def apply_with_pnl(df, fee_mult, spread_mult):
        df = orig_apply(df, fee_mult, spread_mult)
        df['pnl'] = df['pnl'] * fee_mult
        return df

    monkeypatch.setattr(sim, 'apply_fee_spread', apply_with_pnl)

    # simplify heavy helpers
    def fake_bucket(df, q):
        return pd.DataFrame({
            'order_size_mid': [],
            'spread_bps_mean': [],
            'spread_bps_median': [],
            'slippage_bps_mean': [],
            'slippage_bps_median': [],
        })
    monkeypatch.setattr(sim, '_bucket_stats', fake_bucket)
    monkeypatch.setattr(sim, '_latency_stats', lambda df: {'p50_ms': 0.0, 'p95_ms': 0.0})
    monkeypatch.setattr(sim, '_order_fill_stats', lambda df: {'fraction_partially_filled': 0.0, 'fraction_unfilled': 0.0})
    monkeypatch.setattr(sim, '_cancel_stats', lambda df: {'counts': {}, 'shares': {}})

    args = [
        'sim_reality_check',
        '--trades', str(trades_path),
        '--historical-trades', str(hist_path),
        '--benchmark', str(bench_path),
        '--quantiles', '1',
        '--sensitivity-threshold', '0.2',
    ]
    monkeypatch.setattr(sys, 'argv', args)

    sim.main()

    degr_path = trades_path.with_name('sim_reality_check_degradation.json')
    ranking = json.loads(degr_path.read_text())
    scenarios = [r['scenario'] for r in ranking]
    assert scenarios == ['High', 'Med', 'Low']
    kpis = [r['kpi'] for r in ranking]
    assert kpis == sorted(kpis, reverse=True)

    summary_path = trades_path.with_name('sim_reality_check.json')
    summary = json.loads(summary_path.read_text())
    assert 'scenario.High' in summary['flags']
