#!/usr/bin/env python3
"""Run filtered vs unfiltered replay comparison for split3 and split4."""

import logging
import sys
import os

logging.basicConfig(level=logging.WARNING, format='%(message)s')
for name in ['wyckoff_effort.pipeline.wyckoff_features',
             'wyckoff_effort.pipeline', 'wyckoff_effort',
             'wyckoff_trader', 'wyckoff_rl.live.adapters',
             'wyckoff_rl', 'wyckoff_features']:
    logging.getLogger(name).setLevel(logging.ERROR)

sys.path.insert(0, '/opt/ElegantRL')

import pandas as pd
from wyckoff_rl.live.adapters import SCIDReplayAdapter, SimExecutor
from wyckoff_rl.live.trader import WyckoffTrader, VETO_PRESETS

SCID_PATH = '/opt/SierraChart/Data/NQH26-CME.scid'
START_DATE = '2026-01-15'
END_DATE = '2026-03-18'
CKPT_DIR = '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous'
WAVE_DIR = '/opt/ElegantRL/wyckoff_rl/live/checkpoints/wavenet_ppo'

runs = [
    ("split3", f"{CKPT_DIR}/split3_actor__000001488896_00491.096.pt", None),
    ("split3+veto", f"{CKPT_DIR}/split3_actor__000001488896_00491.096.pt", "split3"),
    ("split4", f"{CKPT_DIR}/split4_actor__000000874496_00444.927.pt", None),
    ("split4+veto", f"{CKPT_DIR}/split4_actor__000000874496_00444.927.pt", "split4"),
    ("wavenet_s0", f"{WAVE_DIR}/actor__000001949696_00453.865.pt", None),
    ("wavenet+veto", f"{WAVE_DIR}/actor__000001949696_00453.865.pt", "wavenet_s0"),
]

results = []

for label, ckpt_path, preset in runs:
    print(f"\n{'='*60}", flush=True)
    print(f"RUNNING: {label}" + (f"  veto={VETO_PRESETS[preset]}" if preset else ""), flush=True)
    print(f"{'='*60}", flush=True)

    data = SCIDReplayAdapter(
        scid_path=SCID_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
        speed=0.0,
    )
    sim = SimExecutor(initial_capital=250_000.0)
    veto_rules = VETO_PRESETS.get(preset) if preset else None

    trader = WyckoffTrader(
        checkpoint_path=ckpt_path,
        data_adapter=data,
        order_adapter=sim,
        range_size=40.0,
        continuous_sizing=True,
        max_contracts=1,
        log_dir=f'wyckoff_rl/live_logs/filtered_{label}',
        veto_rules=veto_rules,
    )
    trader.start()

    log_path = trader._trade_log_path
    total_pnl = trader._total_pnl
    n_trades = trader._n_trades
    n_vetoed = trader._n_vetoed
    bars = trader._bar_count

    win_rate = 0; max_dd = 0; n_round_trips = 0
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if len(df) > 0:
            realized = df.pnl_realized_usd.astype(float)
            closes = realized[realized != 0]
            winners = (closes > 0).sum()
            losers = (closes < 0).sum()
            n_round_trips = winners + losers
            win_rate = winners / max(1, n_round_trips) * 100
            equity = df.equity.astype(float)
            peak = equity.cummax()
            max_dd = float((equity - peak).min())

    results.append({
        'label': label, 'bars': bars, 'trades': n_trades,
        'round_trips': n_round_trips, 'vetoed': n_vetoed,
        'total_pnl_usd': total_pnl, 'win_rate': win_rate,
        'max_dd_usd': max_dd, 'equity': sim.get_account_value(),
    })

    print(f"  → trades={n_trades} RT={n_round_trips} vetoed={n_vetoed} "
          f"PnL=${total_pnl:+,.0f} WR={win_rate:.0f}% MaxDD=${max_dd:+,.0f}", flush=True)

print(f"\n\n{'='*100}", flush=True)
print(f"VETO FILTER COMPARISON — OOS Replay (Jan 15 – Mar 18, 2026)", flush=True)
print(f"{'='*100}", flush=True)
print(f"{'Label':<16} {'Bars':>6} {'Trades':>7} {'RT':>5} {'Vetoed':>7} "
      f"{'PnL $':>12} {'WinRate':>8} {'MaxDD $':>12} {'Equity':>12}", flush=True)
print("-"*100, flush=True)
for r in results:
    print(f"{r['label']:<16} {r['bars']:>6} {r['trades']:>7} {r['round_trips']:>5} "
          f"{r['vetoed']:>7} {r['total_pnl_usd']:>+12,.0f} {r['win_rate']:>7.1f}% "
          f"{r['max_dd_usd']:>+12,.0f} {r['equity']:>12,.0f}", flush=True)
print("="*100, flush=True)

# Show improvement for pairs that exist
for base, filtered in [("split3", "split3+veto"), ("split4", "split4+veto"), ("wavenet_s0", "wavenet+veto")]:
    rb = next((r for r in results if r['label'] == base), None)
    rf = next((r for r in results if r['label'] == filtered), None)
    if rb is None or rf is None:
        continue
    delta_pnl = rf['total_pnl_usd'] - rb['total_pnl_usd']
    delta_wr = rf['win_rate'] - rb['win_rate']
    print(f"  {filtered}: PnL {'+' if delta_pnl >= 0 else ''}{delta_pnl:,.0f} "
          f"({rf['total_pnl_usd']/rb['total_pnl_usd']*100:.0f}% of base), "
          f"WR {'+' if delta_wr >= 0 else ''}{delta_wr:.1f}pp, "
          f"vetoed {rf['vetoed']} entries", flush=True)
