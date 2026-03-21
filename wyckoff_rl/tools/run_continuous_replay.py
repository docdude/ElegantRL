#!/usr/bin/env python3
"""Run SCID replay for continuous checkpoints — splits 3 & 5 only."""

import logging
import sys
import os
import csv

# Suppress ALL verbose loggers BEFORE imports trigger them
logging.basicConfig(level=logging.WARNING, format='%(message)s')
for name in ['wyckoff_effort.pipeline.wyckoff_features',
             'wyckoff_effort.pipeline', 'wyckoff_effort',
             'wyckoff_trader', 'wyckoff_rl.live.adapters',
             'wyckoff_rl', 'wyckoff_features']:
    logging.getLogger(name).setLevel(logging.ERROR)

sys.path.insert(0, '/opt/ElegantRL')

import pandas as pd
import numpy as np

from wyckoff_rl.live.adapters import SCIDReplayAdapter, SimExecutor
from wyckoff_rl.live.trader import WyckoffTrader

SCID_PATH = '/opt/SierraChart/Data/NQH26-CME.scid'
START_DATE = '2026-01-15'
END_DATE = '2026-03-18'
CKPT_DIR = '/opt/ElegantRL/wyckoff_rl/live/checkpoints/continuous'

checkpoints = [
    ("split3_s1.48", f"{CKPT_DIR}/split3_actor__000000618496_00450.382.pt"),
    ("split5_s1.55", f"{CKPT_DIR}/split5_actor__000000157696_00221.198.pt"),
]

results = []

for label, ckpt_path in checkpoints:
    print(f"\n{'='*60}")
    print(f"RUNNING: {label}")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"{'='*60}\n")

    data = SCIDReplayAdapter(
        scid_path=SCID_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
        speed=0.0,
    )
    sim = SimExecutor(initial_capital=250_000.0)

    trader = WyckoffTrader(
        checkpoint_path=ckpt_path,
        data_adapter=data,
        order_adapter=sim,
        range_size=40.0,
        continuous_sizing=True,  # KEY: continuous sizing
        max_contracts=1,
        log_dir='wyckoff_rl/live_logs',
    )
    trader.start()

    # Collect results
    log_path = trader._trade_log_path
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if len(df) > 0:
            total_pnl = float(df.total_pnl_usd.iloc[-1])
            final_equity = float(df.equity.iloc[-1])
            n_trades = len(df)
            pnl_pts = total_pnl / 20.0

            realized = df.pnl_realized_usd.astype(float)
            winners = (realized > 0).sum()
            losers = (realized < 0).sum()
            win_rate = winners / max(1, winners + losers) * 100

            equity = df.equity.astype(float)
            peak = equity.cummax()
            max_dd = float((equity - peak).min())
        else:
            total_pnl = 0; final_equity = 250000; n_trades = 0
            pnl_pts = 0; win_rate = 0; max_dd = 0
    else:
        total_pnl = 0; final_equity = 250000; n_trades = 0
        pnl_pts = 0; win_rate = 0; max_dd = 0

    results.append({
        'label': label,
        'bars': trader._bar_count,
        'trades': n_trades,
        'total_pnl_usd': total_pnl,
        'pnl_pts': pnl_pts,
        'final_equity': final_equity,
        'win_rate': win_rate,
        'max_dd_usd': max_dd,
        'log_file': log_path,
    })

# Summary
print(f"\n\n{'='*80}")
print(f"COMPARISON SUMMARY — Continuous Sizing (Jan 15 – Mar 18, 2026)")
print(f"{'='*80}")
print(f"{'Label':<20} {'Bars':>6} {'Trades':>7} {'PnL $':>12} {'PnL pts':>10} {'WinRate':>8} {'MaxDD $':>12} {'Equity':>12}")
print("-"*80)
# Split 0 already completed in prior run
print(f"{'split0_s1.67':<20} {'6074':>6} {'1088':>7} {'$-48,135':>12} {'-2406.8':>10} {'---':>8} {'---':>12} {'$201,865':>12}")
for r in results:
    print(f"{r['label']:<20} {r['bars']:>6} {r['trades']:>7} {r['total_pnl_usd']:>+12,.0f} {r['pnl_pts']:>+10.1f} {r['win_rate']:>7.1f}% {r['max_dd_usd']:>+12,.0f} {r['final_equity']:>12,.0f}")
print("-"*80)
print(f"{'binary_split5':<20} {'6074':>6} {'472':>7} {'$-12,170':>12} {'-608.5':>10} {'52.0%':>8} {'$-45,315':>12} {'$234,270':>12}")
print(f"{'='*80}")
