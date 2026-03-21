#!/usr/bin/env python3
"""Run 2-month SCID replay for Studio 2 binary (all-in) checkpoints."""

import logging
import sys
import os

# Suppress ALL verbose loggers BEFORE imports
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
CKPT_DIR = '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_binary'

checkpoints = [
    ("s2_split7_s2.01", f"{CKPT_DIR}/split7_actor__000000874496_00232.490.pt"),
    ("s2_split4_s1.78", f"{CKPT_DIR}/split4_actor__000000823296_00381.176.pt"),
    ("s2_split5_s1.54", f"{CKPT_DIR}/split5_actor__000001232896_00305.635.pt"),
]

results = []

for label, ckpt_path in checkpoints:
    print(f"\n{'='*60}", flush=True)
    print(f"RUNNING: {label}", flush=True)
    print(f"Checkpoint: {os.path.basename(ckpt_path)}", flush=True)
    print(f"{'='*60}", flush=True)

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
        continuous_sizing=False,  # BINARY all-in
        max_contracts=1,
        log_dir=f'wyckoff_rl/live_logs/studio2_{label}',
    )
    trader.start()

    # Collect results
    log_path = trader._trade_log_path
    total_pnl = trader._total_pnl
    n_trades = trader._n_trades
    bars = trader._bar_count
    final_equity = sim.get_account_value()
    pnl_pts = total_pnl / 20.0

    win_rate = 0; max_dd = 0
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if len(df) > 0:
            realized = df.pnl_realized_usd.astype(float)
            winners = (realized > 0).sum()
            losers = (realized < 0).sum()
            win_rate = winners / max(1, winners + losers) * 100
            equity = df.equity.astype(float)
            peak = equity.cummax()
            max_dd = float((equity - peak).min())

    results.append({
        'label': label,
        'bars': bars,
        'trades': n_trades,
        'total_pnl_usd': total_pnl,
        'pnl_pts': pnl_pts,
        'final_equity': final_equity,
        'win_rate': win_rate,
        'max_dd_usd': max_dd,
    })

    cumR = pnl_pts / 1000.0 * 256
    print(f"\nRESULT {label}: bars={bars} trades={n_trades} "
          f"P&L=${total_pnl:+,.0f} ({pnl_pts:+.1f}pts) equity=${final_equity:,.0f} "
          f"cumR={cumR:+.1f} WR={win_rate:.0f}% MaxDD=${max_dd:+,.0f}", flush=True)

# Summary
print(f"\n\n{'='*90}", flush=True)
print(f"STUDIO 2 BINARY — OOS Replay (Jan 15 – Mar 18, 2026)", flush=True)
print(f"{'='*90}", flush=True)
print(f"{'Label':<22} {'Bars':>6} {'Trades':>7} {'PnL $':>12} {'PnL pts':>10} {'WinRate':>8} {'MaxDD $':>12} {'Equity':>12}", flush=True)
print("-"*90, flush=True)
for r in results:
    print(f"{r['label']:<22} {r['bars']:>6} {r['trades']:>7} {r['total_pnl_usd']:>+12,.0f} {r['pnl_pts']:>+10.1f} {r['win_rate']:>7.1f}% {r['max_dd_usd']:>+12,.0f} {r['final_equity']:>12,.0f}", flush=True)
print("-"*90, flush=True)
# Reference: Studio 1 results
print(f"{'s1_cont_split5':<22} {'6074':>6} {'750':>7} {'$-8,145':>12} {'-407.2':>10} {'47.6%':>8} {'$-41,965':>12} {'$238,105':>12}", flush=True)
print(f"{'s1_binary_split5':<22} {'6074':>6} {'472':>7} {'$-12,170':>12} {'-608.5':>10} {'52.0%':>8} {'$-45,315':>12} {'$234,270':>12}", flush=True)
print(f"{'='*90}", flush=True)
