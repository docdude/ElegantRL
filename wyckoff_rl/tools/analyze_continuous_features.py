#!/usr/bin/env python3
"""
Feature analysis for the 2 winning continuous models (splits 4 and 3).
Captures 30-bar x 36-feature windows, compares entry/exit features vs baseline.
Also compares what the two models have in common.
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

CHECKPOINTS = {
    'split4_s2.37': '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous/split4_actor__000000874496_00444.927.pt',
    'split3_s2.29': '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous/split3_actor__000001488896_00491.096.pt',
}

def run_feature_capture(label, ckpt_path):
    """Run replay with monkey-patched feature capture."""
    # Reset capture state
    capture = {
        'all_bar_features': [],
        'all_bar_actions': [],
        'all_bar_prices': [],
        'entry_bar_indices': [],
        'exit_bar_indices': [],
        'entry_actions': [],
        'exit_actions': [],
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

        bar_idx = len(capture['all_bar_features'])
        capture['all_bar_features'].append(self.inference._window.copy())
        capture['all_bar_prices'].append(bar.close)

        target_pos, raw_action = self.inference.get_action(
            position=self._position,
            unrealized_pnl=self._unrealized_pnl,
            cash=self._cash,
        )
        capture['all_bar_actions'].append(raw_action)

        target_pos = max(-self.max_contracts, min(self.max_contracts, target_pos))

        delta = int(round(target_pos - self._position))
        if delta != 0:
            if delta > 0:
                capture['entry_bar_indices'].append(bar_idx)
                capture['entry_actions'].append(raw_action)
            else:
                capture['exit_bar_indices'].append(bar_idx)
                capture['exit_actions'].append(raw_action)
            self._execute_trade(delta, bar.close, raw_action, bar)

    WyckoffTrader._on_bar_complete = patched_on_bar

    print(f"\n{'='*90}", flush=True)
    print(f"Running {label} replay with feature capture...", flush=True)
    data = SCIDReplayAdapter(scid_path=SCID_PATH, start_date='2026-01-15', end_date='2026-03-18', speed=0.0)
    sim = SimExecutor(initial_capital=250_000.0)
    trader = WyckoffTrader(
        checkpoint_path=ckpt_path, data_adapter=data, order_adapter=sim,
        range_size=40.0, continuous_sizing=True, max_contracts=1,
        log_dir=f'/tmp/feat_analysis_{label}',
    )
    trader.start()

    WyckoffTrader._on_bar_complete = original_on_bar

    n_bars = len(capture['all_bar_features'])
    n_entries = len(capture['entry_bar_indices'])
    n_exits = len(capture['exit_bar_indices'])
    print(f"Capture complete: {n_bars} bars, {n_entries} entries, {n_exits} exits", flush=True)

    return {
        'all_windows': np.array(capture['all_bar_features']),
        'actions': np.array(capture['all_bar_actions']),
        'prices': np.array(capture['all_bar_prices']),
        'entry_idx': capture['entry_bar_indices'],
        'exit_idx': capture['exit_bar_indices'],
        'entry_actions': capture['entry_actions'],
        'exit_actions': capture['exit_actions'],
    }


def analyze_model(label, data):
    """Run complete feature analysis for one model."""
    all_windows = data['all_windows']
    actions = data['actions']
    entry_idx = data['entry_idx']
    exit_idx = data['exit_idx']

    if len(entry_idx) == 0:
        print(f"  No entries for {label}, skipping analysis", flush=True)
        return None

    # ── Analysis 1: Current-bar features at entry vs all bars ──
    current_bar_all = all_windows[:, -1, :]
    current_bar_entry = all_windows[entry_idx][:, -1, :]

    print(f"\n{'='*90}", flush=True)
    print(f"FEATURE ANALYSIS: {label} — Entry Bars vs All Bars", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Feature':<30} {'All Mean':>10} {'All Std':>10} {'Entry Mean':>10} {'Entry Std':>10} {'Z-score':>8} {'Direction':>10}", flush=True)
    print('-'*90, flush=True)

    mean_all = current_bar_all.mean(axis=0)
    std_all = current_bar_all.std(axis=0)
    mean_entry = current_bar_entry.mean(axis=0)
    std_entry = current_bar_entry.std(axis=0)

    deviations = []
    for i, fname in enumerate(FEATURE_NAMES):
        z = (mean_entry[i] - mean_all[i]) / (std_all[i] + 1e-8)
        direction = "HIGHER" if z > 0.3 else "LOWER" if z < -0.3 else ""
        deviations.append((abs(z), i, fname, mean_all[i], std_all[i], mean_entry[i], std_entry[i], z, direction))

    deviations.sort(reverse=True)
    for absz, i, fname, ma, sa, me, se, z, d in deviations:
        marker = "***" if absz > 0.5 else " * " if absz > 0.3 else "   "
        print(f"{marker}{fname:<27} {ma:>+10.4f} {sa:>10.4f} {me:>+10.4f} {se:>10.4f} {z:>+8.2f} {d:>10}", flush=True)

    # ── Analysis 2: Window-average features ──
    print(f"\n{'='*90}", flush=True)
    print(f"WINDOW CONTEXT: {label} — 30-bar Average at Entry vs All Bars", flush=True)
    print(f"{'='*90}", flush=True)

    window_avg_all = all_windows.mean(axis=1)
    window_avg_entry = all_windows[entry_idx].mean(axis=1)

    mean_wavg_all = window_avg_all.mean(axis=0)
    std_wavg_all = window_avg_all.std(axis=0)
    mean_wavg_entry = window_avg_entry.mean(axis=0)

    print(f"{'Feature':<30} {'All Mean':>10} {'Entry Mean':>10} {'Z-score':>8} {'Direction':>10}", flush=True)
    print('-'*70, flush=True)

    w_deviations = []
    for i, fname in enumerate(FEATURE_NAMES):
        z = (mean_wavg_entry[i] - mean_wavg_all[i]) / (std_wavg_all[i] + 1e-8)
        direction = "HIGHER" if z > 0.3 else "LOWER" if z < -0.3 else ""
        w_deviations.append((abs(z), fname, mean_wavg_all[i], mean_wavg_entry[i], z, direction))

    w_deviations.sort(reverse=True)
    for absz, fname, ma, me, z, d in w_deviations:
        marker = "***" if absz > 0.5 else " * " if absz > 0.3 else "   "
        print(f"{marker}{fname:<27} {ma:>+10.4f} {me:>+10.4f} {z:>+8.2f} {d:>10}", flush=True)

    # ── Analysis 3: Intra-window trend ──
    print(f"\n{'='*90}", flush=True)
    print(f"INTRA-WINDOW TREND: {label} — Last 5 bars vs First 5 bars at Entry", flush=True)
    print(f"{'='*90}", flush=True)
    entry_windows = all_windows[entry_idx]
    early = entry_windows[:, :5, :].mean(axis=(0, 1))
    late = entry_windows[:, -5:, :].mean(axis=(0, 1))

    print(f"{'Feature':<30} {'Early(1-5)':>10} {'Late(26-30)':>10} {'Delta':>10} {'Trend':>10}", flush=True)
    print('-'*70, flush=True)

    trends = []
    for i, fname in enumerate(FEATURE_NAMES):
        delta = late[i] - early[i]
        trend = "RISING" if delta > 0.05 else "FALLING" if delta < -0.05 else ""
        trends.append((abs(delta), fname, early[i], late[i], delta, trend))

    trends.sort(reverse=True)
    for absd, fname, e, l, d, t in trends:
        marker = "***" if absd > 0.15 else " * " if absd > 0.05 else "   "
        print(f"{marker}{fname:<27} {e:>+10.4f} {l:>+10.4f} {d:>+10.4f} {t:>10}", flush=True)

    # ── Analysis 4: Action distribution ──
    print(f"\n{'='*90}", flush=True)
    print(f"ACTION DISTRIBUTION: {label}", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"  All bars:   mean={actions.mean():.4f}  std={actions.std():.4f}  min={actions.min():.4f}  max={actions.max():.4f}", flush=True)
    if len(data['entry_actions']) > 0:
        ea = np.array(data['entry_actions'])
        print(f"  Entry bars: mean={ea.mean():.4f}  std={ea.std():.4f}  min={ea.min():.4f}  max={ea.max():.4f}  n={len(ea)}", flush=True)
    if len(data['exit_actions']) > 0:
        xa = np.array(data['exit_actions'])
        print(f"  Exit bars:  mean={xa.mean():.4f}  std={xa.std():.4f}  min={xa.min():.4f}  max={xa.max():.4f}  n={len(xa)}", flush=True)

    bins = np.linspace(-1, 1, 21)
    counts, _ = np.histogram(actions, bins=bins)
    print(f"\n  Action histogram:", flush=True)
    for j in range(len(counts)):
        bar_char = '#' * (counts[j] * 50 // max(counts))
        print(f"  [{bins[j]:+.1f},{bins[j+1]:+.1f}) {counts[j]:>5} {bar_char}", flush=True)

    # Return z-scores for comparison
    z_scores = {}
    for absz, i, fname, ma, sa, me, se, z, d in deviations:
        z_scores[fname] = z
    return z_scores


# ═══════════════════════════════════════════════════════════════════════════
# Main: run both models and compare
# ═══════════════════════════════════════════════════════════════════════════

all_results = {}
all_z_scores = {}

for label, ckpt in CHECKPOINTS.items():
    data = run_feature_capture(label, ckpt)
    all_results[label] = data
    z = analyze_model(label, data)
    if z is not None:
        all_z_scores[label] = z

# ── Cross-model comparison ──
if len(all_z_scores) >= 2:
    labels = list(all_z_scores.keys())
    print(f"\n{'='*90}", flush=True)
    print(f"CROSS-MODEL COMPARISON: Entry-bar Z-scores", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Feature':<30} {labels[0]:>12} {labels[1]:>12} {'Agree?':>8} {'Avg Z':>8}", flush=True)
    print('-'*75, flush=True)

    comparisons = []
    for fname in FEATURE_NAMES:
        z0 = all_z_scores[labels[0]].get(fname, 0)
        z1 = all_z_scores[labels[1]].get(fname, 0)
        avg_z = (z0 + z1) / 2
        agree = "YES" if (z0 > 0 and z1 > 0) or (z0 < 0 and z1 < 0) else "NO"
        comparisons.append((abs(avg_z), fname, z0, z1, agree, avg_z))

    comparisons.sort(reverse=True)
    for absz, fname, z0, z1, agree, avgz in comparisons:
        marker = "***" if absz > 0.3 and agree == "YES" else " * " if absz > 0.15 else "   "
        print(f"{marker}{fname:<27} {z0:>+12.3f} {z1:>+12.3f} {agree:>8} {avgz:>+8.3f}", flush=True)

    # Summary: shared strong signals
    print(f"\n{'='*90}", flush=True)
    print(f"SHARED SIGNALS (both models agree, |avg Z| > 0.2)", flush=True)
    print(f"{'='*90}", flush=True)
    shared = [(absz, fname, z0, z1, avgz) for absz, fname, z0, z1, agree, avgz in comparisons
              if agree == "YES" and absz > 0.2]
    for absz, fname, z0, z1, avgz in shared:
        direction = "HIGHER at entry" if avgz > 0 else "LOWER at entry"
        print(f"  {fname:<30} avg Z={avgz:>+.3f}  ({labels[0]}={z0:>+.3f}, {labels[1]}={z1:>+.3f})  → {direction}", flush=True)

print("\nDone.", flush=True)
