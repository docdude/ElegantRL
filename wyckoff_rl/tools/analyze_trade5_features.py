#!/usr/bin/env python3
"""
Extract the exact 30-bar x 36-feature window the RL model saw at trade #5.

Uses the same monkey-patch approach as analyze_continuous_features.py:
run the full replay with WyckoffTrader, intercept _on_bar_complete to
capture inference._window at every bar, then inspect the specific trade.

Trade #5: Feb 20 14:38 BUY @ 24852.50 → 14:46 SELL @ 24945.25 (+$1,855)
"""
import logging, sys, os
import numpy as np
from datetime import datetime, timezone

logging.basicConfig(level=logging.WARNING, format='%(message)s')
for name in ['wyckoff_effort.pipeline.wyckoff_features',
             'wyckoff_effort.pipeline', 'wyckoff_effort',
             'wyckoff_trader', 'wyckoff_rl.live.adapters',
             'wyckoff_rl', 'wyckoff_features']:
    logging.getLogger(name).setLevel(logging.ERROR)

sys.path.insert(0, '/opt/ElegantRL')

from wyckoff_rl.live.adapters import SCIDReplayAdapter, SimExecutor
from wyckoff_rl.live.trader import WyckoffTrader
from wyckoff_rl.feature_config import ALL_FEATURES, SELECTED_INDICES

FEATURE_NAMES = [ALL_FEATURES[i] for i in SELECTED_INDICES]

SCID_PATH = '/opt/SierraChart/Data/NQH26-CME.scid'
CKPT = '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous/split4_actor__000000874496_00444.927.pt'

# ── Monkey-patch to capture features at every bar ──
capture = {
    'windows': [],      # (30, 36) feature window at each bar
    'actions': [],       # raw action value
    'prices': [],        # bar close price
    'timestamps': [],    # bar timestamp
    'positions': [],     # position before action
    'trade_bars': [],    # (bar_idx, action_str, delta, price, pnl)
}

original_on_bar = WyckoffTrader._on_bar_complete

def patched_on_bar(self, bar):
    self._bar_count += 1
    self._last_bar_price = bar.close

    features = self.feature_engine.add_bar(bar)
    if features is None:
        return

    self.inference.push_features(features)
    if not self.inference.ready:
        return

    bar_idx = len(capture['windows'])
    capture['windows'].append(self.inference._window.copy())
    capture['prices'].append(bar.close)
    capture['timestamps'].append(bar.timestamp)
    capture['positions'].append(self._position)

    target_pos, raw_action = self.inference.get_action(
        position=self._position,
        unrealized_pnl=self._unrealized_pnl,
        cash=self._cash,
    )
    capture['actions'].append(raw_action)

    target_pos = max(-self.max_contracts, min(self.max_contracts, target_pos))
    delta = int(round(target_pos - self._position))
    if delta != 0:
        action_str = "BUY" if delta > 0 else "SELL"
        # Compute PnL
        pnl = 0.0
        if self._position != 0:
            if (self._position > 0 and delta < 0) or (self._position < 0 and delta > 0):
                closed = min(abs(self._position), abs(delta))
                pnl_pts = (bar.close - self._entry_price) * (1 if self._position > 0 else -1)
                pnl = closed * pnl_pts * 20.0
        capture['trade_bars'].append((bar_idx, action_str, delta, bar.close, pnl))
        self._execute_trade(delta, bar.close, raw_action, bar)

WyckoffTrader._on_bar_complete = patched_on_bar

# ── Run replay ──
print("Running split 4 continuous replay with feature capture...", flush=True)
data = SCIDReplayAdapter(scid_path=SCID_PATH, start_date='2026-01-15', end_date='2026-03-18', speed=0.0)
sim = SimExecutor(initial_capital=250_000.0)
trader = WyckoffTrader(
    checkpoint_path=CKPT, data_adapter=data, order_adapter=sim,
    range_size=40.0, continuous_sizing=True, max_contracts=1,
    log_dir='/tmp/trade5_analysis',
)
trader.start()

print(f"\nCapture complete: {len(capture['windows'])} bars, {len(capture['trade_bars'])} trades", flush=True)

# ── List all trades to identify #5 ──
print(f"\n{'='*90}")
print(f"ALL TRADES")
print(f"{'='*90}")
cumulative_pnl = 0.0
for i, (bidx, act, delta, price, pnl) in enumerate(capture['trade_bars']):
    cumulative_pnl += pnl
    ts = datetime.fromtimestamp(capture['timestamps'][bidx], tz=timezone.utc)
    pos_before = capture['positions'][bidx]
    raw_act = capture['actions'][bidx]
    pnl_str = f"  PnL=${pnl:+,.0f}" if pnl != 0 else ""
    print(f"  Trade {i+1:3d}: bar {bidx:4d} | {ts} | {act} {abs(delta)} @ {price:.2f} | "
          f"pos {pos_before:+.0f}→{pos_before+delta:+.0f} | action={raw_act:+.4f}{pnl_str} | cum=${cumulative_pnl:+,.0f}")

# ── Find the target trade (Feb 20 17:40 BUY @ 24968) ──
TARGET_TRADE = None
for i, (bidx, act, delta, price, pnl) in enumerate(capture['trade_bars']):
    ts = datetime.fromtimestamp(capture['timestamps'][bidx], tz=timezone.utc)
    if ts.month == 2 and ts.day == 20 and ts.hour == 17 and 38 <= ts.minute <= 42 and act == "BUY":
        TARGET_TRADE = i
        break

if TARGET_TRADE is None:
    # Fallback: use trade index 4 (0-based for trade #5)
    TARGET_TRADE = 4 if len(capture['trade_bars']) > 4 else 0
    print(f"\nWARNING: Could not find Feb 20 14:38 BUY, using trade index {TARGET_TRADE}")

entry_trade = capture['trade_bars'][TARGET_TRADE]
entry_bidx = entry_trade[0]
entry_ts = datetime.fromtimestamp(capture['timestamps'][entry_bidx], tz=timezone.utc)
entry_price = entry_trade[3]

# Find the corresponding exit (next SELL after this BUY)
exit_trade = None
exit_bidx = None
for j in range(TARGET_TRADE + 1, len(capture['trade_bars'])):
    if capture['trade_bars'][j][1] == "SELL":
        exit_trade = capture['trade_bars'][j]
        exit_bidx = exit_trade[0]
        break

print(f"\n{'='*90}")
print(f"TRADE #{TARGET_TRADE+1} DETAILS")
print(f"{'='*90}")
print(f"  Entry: bar {entry_bidx} | {entry_ts} | BUY @ {entry_price:.2f} | action={capture['actions'][entry_bidx]:+.6f}")
if exit_trade:
    exit_ts = datetime.fromtimestamp(capture['timestamps'][exit_bidx], tz=timezone.utc)
    print(f"  Exit:  bar {exit_bidx} | {exit_ts} | SELL @ {exit_trade[3]:.2f} | action={capture['actions'][exit_bidx]:+.6f} | PnL=${exit_trade[4]:+,.0f}")

# ── Print the 30-bar x 36-feature window at ENTRY ──
all_windows = np.array(capture['windows'])
all_prices = np.array(capture['prices'])

print(f"\n{'='*90}")
print(f"30-BAR FEATURE WINDOW AT ENTRY (bar {entry_bidx}, {entry_ts})")
print(f"{'='*90}")

window = all_windows[entry_bidx]  # (30, 36)
print(f"\nCurrent bar (window[-1]) features:")
current = window[-1]
for i, fname in enumerate(FEATURE_NAMES):
    print(f"  {fname:30s} = {current[i]:+.6f}")

# Show feature evolution across the 30-bar window for key features
print(f"\n{'='*90}")
print(f"FEATURE EVOLUTION ACROSS 30-BAR WINDOW AT ENTRY")
print(f"{'='*90}")
print(f"Shows bars 1,5,10,15,20,25,30 of the window")
print(f"{'Feature':<30} {'bar1':>8} {'bar5':>8} {'bar10':>8} {'bar15':>8} {'bar20':>8} {'bar25':>8} {'bar30':>8}")
print('-'*90)

for i, fname in enumerate(FEATURE_NAMES):
    vals = [window[j][i] for j in [0, 4, 9, 14, 19, 24, 29]]
    print(f"  {fname:<28} {vals[0]:>+8.3f} {vals[1]:>+8.3f} {vals[2]:>+8.3f} "
          f"{vals[3]:>+8.3f} {vals[4]:>+8.3f} {vals[5]:>+8.3f} {vals[6]:>+8.3f}")

# ── Compare entry window vs exit window ──
if exit_bidx is not None:
    print(f"\n{'='*90}")
    print(f"CURRENT-BAR FEATURES: ENTRY vs EXIT")
    print(f"{'='*90}")
    entry_current = all_windows[entry_bidx, -1, :]
    exit_current = all_windows[exit_bidx, -1, :]
    print(f"{'Feature':<30} {'Entry':>10} {'Exit':>10} {'Delta':>10}")
    print('-'*65)
    for i, fname in enumerate(FEATURE_NAMES):
        diff = exit_current[i] - entry_current[i]
        marker = " ***" if abs(diff) > 0.1 else " *" if abs(diff) > 0.03 else ""
        print(f"  {fname:<28} {entry_current[i]:>+10.4f} {exit_current[i]:>+10.4f} {diff:>+10.4f}{marker}")

# ── Compare entry features vs all-bar baseline ──
print(f"\n{'='*90}")
print(f"ENTRY FEATURES vs ALL-BAR BASELINE (Z-scores)")
print(f"{'='*90}")
current_bar_all = all_windows[:, -1, :]
mean_all = current_bar_all.mean(axis=0)
std_all = current_bar_all.std(axis=0)
entry_current = all_windows[entry_bidx, -1, :]

deviations = []
for i, fname in enumerate(FEATURE_NAMES):
    z = (entry_current[i] - mean_all[i]) / (std_all[i] + 1e-8)
    deviations.append((abs(z), fname, mean_all[i], entry_current[i], z))

deviations.sort(reverse=True)
print(f"{'Feature':<30} {'All Mean':>10} {'Entry Val':>10} {'Z-score':>8}")
print('-'*65)
for absz, fname, ma, ev, z in deviations:
    marker = "***" if absz > 1.0 else " * " if absz > 0.5 else "   "
    print(f"{marker}{fname:<27} {ma:>+10.4f} {ev:>+10.4f} {z:>+8.2f}")

# ── Bars around the trade (price action context) ──
print(f"\n{'='*90}")
print(f"PRICE ACTION AROUND TRADE #{TARGET_TRADE+1}")
print(f"{'='*90}")
ctx_start = max(0, entry_bidx - 10)
ctx_end = min(len(capture['prices']) - 1, (exit_bidx or entry_bidx) + 5)

for idx in range(ctx_start, ctx_end + 1):
    ts = datetime.fromtimestamp(capture['timestamps'][idx], tz=timezone.utc)
    price = capture['prices'][idx]
    action_val = capture['actions'][idx]
    pos = capture['positions'][idx]
    marker = ""
    if idx == entry_bidx:
        marker = " <<< ENTRY (BUY)"
    elif idx == exit_bidx:
        marker = " <<< EXIT (SELL)"
    print(f"  Bar {idx:4d}: {ts} close={price:.2f} pos={pos:+.0f} action={action_val:+.4f}{marker}")

print("\nDone.", flush=True)
