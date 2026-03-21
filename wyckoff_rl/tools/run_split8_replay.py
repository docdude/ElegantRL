#!/usr/bin/env python3
"""Run 2-month SCID replay for split 8 continuous checkpoint."""

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
import numpy as np

from wyckoff_rl.live.adapters import SCIDReplayAdapter, SimExecutor
from wyckoff_rl.live.trader import WyckoffTrader

SCID_PATH = '/opt/SierraChart/Data/NQH26-CME.scid'
START_DATE = '2026-01-15'
END_DATE = '2026-03-18'

label = "s2c_split8_s0.80"
ckpt_path = "/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous/split8_actor__000000260096_00271.155.pt"

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
    continuous_sizing=True,
    max_contracts=1,
    log_dir=f'wyckoff_rl/live_logs/studio2c_{label}',
)
trader.start()

# Results
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

cumR = pnl_pts / 1000.0 * 256
print(f"\n{'='*60}", flush=True)
print(f"RESULT {label}:", flush=True)
print(f"  Bars:     {bars}", flush=True)
print(f"  Trades:   {n_trades}", flush=True)
print(f"  PnL:      ${total_pnl:+,.0f} ({pnl_pts:+.1f} pts)", flush=True)
print(f"  Equity:   ${final_equity:,.0f}", flush=True)
print(f"  Win Rate: {win_rate:.1f}%", flush=True)
print(f"  Max DD:   ${max_dd:+,.0f}", flush=True)
print(f"  cumR:     {cumR:+.1f}", flush=True)
print(f"{'='*60}", flush=True)
