#!/usr/bin/env python3
"""
Feature analysis: capture what the model 'sees' at each trade entry.

Runs split 7 replay, captures the 30-bar × 36-feature window at each
trade bar, then compares entry-bar features vs all-bar averages to identify
which Wyckoff signals drive the model's decisions.
"""

import logging
import sys
import os
import numpy as np

# Suppress logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
for name in ['wyckoff_effort.pipeline.wyckoff_features',
             'wyckoff_effort.pipeline', 'wyckoff_effort',
             'wyckoff_trader', 'wyckoff_rl.live.adapters',
             'wyckoff_rl', 'wyckoff_features']:
    logging.getLogger(name).setLevel(logging.ERROR)

sys.path.insert(0, '/opt/ElegantRL')

import pandas as pd
from wyckoff_rl.live.adapters import SCIDReplayAdapter, SimExecutor
from wyckoff_rl.live.trader import WyckoffTrader
from wyckoff_rl.feature_config import ALL_FEATURES, SELECTED_INDICES

FEATURE_NAMES = [ALL_FEATURES[i] for i in SELECTED_INDICES]

SCID_PATH = '/opt/SierraChart/Data/NQH26-CME.scid'
CKPT = '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_binary/split7_actor__000000874496_00232.490.pt'

# Monkey-patch the trader to capture features at every bar
all_bar_features = []     # features at every bar (after window warm-up)
all_bar_actions = []      # raw action at every bar
all_bar_prices = []       # close price at each bar
entry_bar_indices = []    # indices into above lists where BUY trades happen
exit_bar_indices = []     # indices where SELL trades happen

original_on_bar = WyckoffTrader._on_bar_complete

def patched_on_bar(self, bar):
    """Intercept bar completion to capture features + actions."""
    self._bar_count += 1
    self._last_bar_price = bar.close

    features = self.feature_engine.add_bar(bar)
    if features is None:
        return

    self.inference.push_features(features)

    if not self.inference.ready:
        return

    # Capture the current feature window (30 bars × 36 features)
    bar_idx = len(all_bar_features)
    all_bar_features.append(self.inference._window.copy())  # (30, 36)
    all_bar_prices.append(bar.close)

    target_pos, raw_action = self.inference.get_action(
        position=self._position,
        unrealized_pnl=self._unrealized_pnl,
        cash=self._cash,
    )
    all_bar_actions.append(raw_action)

    target_pos = max(-self.max_contracts, min(self.max_contracts, target_pos))

    delta = int(round(target_pos - self._position))
    if delta != 0:
        if delta > 0:
            entry_bar_indices.append(bar_idx)
        else:
            exit_bar_indices.append(bar_idx)
        self._execute_trade(delta, bar.close, raw_action, bar)

WyckoffTrader._on_bar_complete = patched_on_bar

# Run replay
print("Running split 7 replay with feature capture...", flush=True)
data = SCIDReplayAdapter(scid_path=SCID_PATH, start_date='2026-01-15', end_date='2026-03-18', speed=0.0)
sim = SimExecutor(initial_capital=250_000.0)
trader = WyckoffTrader(
    checkpoint_path=CKPT, data_adapter=data, order_adapter=sim,
    range_size=40.0, continuous_sizing=False, max_contracts=1, log_dir='/tmp/feat_analysis',
)
trader.start()

print(f"\nCapture complete: {len(all_bar_features)} bars, {len(entry_bar_indices)} entries, {len(exit_bar_indices)} exits", flush=True)

# Convert to arrays
all_windows = np.array(all_bar_features)  # (N_bars, 30, 36)
actions = np.array(all_bar_actions)       # (N_bars,)
prices = np.array(all_bar_prices)

# ── Analysis 1: Current-bar features at entry vs all bars ──
# The most recent bar in the window is index -1 (window[29])
current_bar_all = all_windows[:, -1, :]  # (N_bars, 36) — current bar features for ALL bars
current_bar_entry = all_windows[entry_bar_indices][:, -1, :]  # entries only
current_bar_exit = all_windows[exit_bar_indices][:, -1, :]    # exits only

print(f"\n{'='*90}")
print(f"FEATURE ANALYSIS: Split 7 Entry Bars vs All Bars")
print(f"{'='*90}")
print(f"{'Feature':<30} {'All Mean':>10} {'All Std':>10} {'Entry Mean':>10} {'Entry Std':>10} {'Z-score':>8} {'Direction':>10}")
print('-'*90)

mean_all = current_bar_all.mean(axis=0)
std_all = current_bar_all.std(axis=0)
mean_entry = current_bar_entry.mean(axis=0)

deviations = []
for i, fname in enumerate(FEATURE_NAMES):
    z = (mean_entry[i] - mean_all[i]) / (std_all[i] + 1e-8)
    direction = "HIGHER" if z > 0.3 else "LOWER" if z < -0.3 else ""
    deviations.append((abs(z), i, fname, mean_all[i], std_all[i], mean_entry[i], z, direction))

# Sort by absolute z-score to see most distinctive features
deviations.sort(reverse=True)

for absz, i, fname, ma, sa, me, z, d in deviations:
    marker = "***" if absz > 0.5 else " * " if absz > 0.3 else "   "
    print(f"{marker}{fname:<27} {ma:>+10.4f} {sa:>10.4f} {me:>+10.4f} {current_bar_entry.std(axis=0)[i]:>10.4f} {z:>+8.2f} {d:>10}")

# ── Analysis 2: Window-average features (temporal context) ──
print(f"\n{'='*90}")
print(f"WINDOW CONTEXT: 30-bar Average at Entry vs All Bars")
print(f"{'='*90}")

window_avg_all = all_windows.mean(axis=1)  # (N_bars, 36)
window_avg_entry = all_windows[entry_bar_indices].mean(axis=1)  # (N_entries, 36)

mean_wavg_all = window_avg_all.mean(axis=0)
std_wavg_all = window_avg_all.std(axis=0)
mean_wavg_entry = window_avg_entry.mean(axis=0)

print(f"{'Feature':<30} {'All Mean':>10} {'Entry Mean':>10} {'Z-score':>8} {'Direction':>10}")
print('-'*70)

w_deviations = []
for i, fname in enumerate(FEATURE_NAMES):
    z = (mean_wavg_entry[i] - mean_wavg_all[i]) / (std_wavg_all[i] + 1e-8)
    direction = "HIGHER" if z > 0.3 else "LOWER" if z < -0.3 else ""
    w_deviations.append((abs(z), fname, mean_wavg_all[i], mean_wavg_entry[i], z, direction))

w_deviations.sort(reverse=True)
for absz, fname, ma, me, z, d in w_deviations:
    marker = "***" if absz > 0.5 else " * " if absz > 0.3 else "   "
    print(f"{marker}{fname:<27} {ma:>+10.4f} {me:>+10.4f} {z:>+8.2f} {d:>10}")

# ── Analysis 3: Feature trend within window at entry (last 5 bars vs first 5) ──
print(f"\n{'='*90}")
print(f"INTRA-WINDOW TREND at Entry: Last 5 bars vs First 5 bars")
print(f"{'='*90}")
entry_windows = all_windows[entry_bar_indices]  # (N_entries, 30, 36)
early = entry_windows[:, :5, :].mean(axis=(0, 1))   # first 5 bars avg
late = entry_windows[:, -5:, :].mean(axis=(0, 1))    # last 5 bars avg

print(f"{'Feature':<30} {'Early(1-5)':>10} {'Late(26-30)':>10} {'Delta':>10} {'Trend':>10}")
print('-'*70)

trends = []
for i, fname in enumerate(FEATURE_NAMES):
    delta = late[i] - early[i]
    trend = "RISING" if delta > 0.05 else "FALLING" if delta < -0.05 else ""
    trends.append((abs(delta), fname, early[i], late[i], delta, trend))

trends.sort(reverse=True)
for absd, fname, e, l, d, t in trends:
    marker = "***" if absd > 0.15 else " * " if absd > 0.05 else "   "
    print(f"{marker}{fname:<27} {e:>+10.4f} {l:>+10.4f} {d:>+10.4f} {t:>10}")

# ── Analysis 4: Action distribution ──
print(f"\n{'='*90}")
print(f"ACTION DISTRIBUTION")
print(f"{'='*90}")
print(f"  All bars:   mean={actions.mean():.4f}  std={actions.std():.4f}  min={actions.min():.4f}  max={actions.max():.4f}")
print(f"  Entry bars: {[f'{actions[i]:.4f}' for i in entry_bar_indices]}")
print(f"  Exit bars:  {[f'{actions[i]:.4f}' for i in exit_bar_indices]}")

# Histogram of actions
bins = np.linspace(-1, 1, 21)
counts, _ = np.histogram(actions, bins=bins)
print(f"\n  Action histogram:")
for j in range(len(counts)):
    bar_char = '#' * (counts[j] * 50 // max(counts))
    print(f"  [{bins[j]:+.1f},{bins[j+1]:+.1f}) {counts[j]:>5} {bar_char}")
