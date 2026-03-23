#!/usr/bin/env python3
"""
Unified trade analysis for Wyckoff RL checkpoints.

Replaces: analyze_bad_trades.py, analyze_continuous_features.py,
          analyze_features.py, analyze_trade5_features.py,
          analyze_trade_features.py

Modes:
    profile   — Entry feature Z-scores vs baseline, window context, action dist
    losses    — Win vs loss feature comparison, clustering by hold/dir/session
    deep-dive — Single trade window inspection + feature evolution
    all       — Run profile + losses together

Usage:
    # Feature profile for one checkpoint
    python -m wyckoff_rl.tools.analyze_trades profile \\
        --checkpoints path/to/actor.pt

    # Win/loss analysis across two checkpoints
    python -m wyckoff_rl.tools.analyze_trades losses \\
        --checkpoints split3.pt split4.pt --continuous

    # Deep-dive on trade #5
    python -m wyckoff_rl.tools.analyze_trades deep-dive \\
        --checkpoints split4.pt --continuous --trade 5

    # Everything
    python -m wyckoff_rl.tools.analyze_trades all \\
        --checkpoints split3.pt split4.pt --continuous
"""

import argparse
import logging
import os
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(message)s')
for name in ['wyckoff_effort.pipeline.wyckoff_features',
             'wyckoff_effort.pipeline', 'wyckoff_effort',
             'wyckoff_trader', 'wyckoff_rl.live.adapters',
             'wyckoff_rl', 'wyckoff_features']:
    logging.getLogger(name).setLevel(logging.ERROR)

_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from wyckoff_rl.live.live_features import TRAINING_FEATURE_INDICES
from wyckoff_rl.feature_config import ALL_FEATURES

# Feature names for the 36 training features
FEATURE_NAMES = [ALL_FEATURES[i] for i in TRAINING_FEATURE_INDICES]
N_FEATURES = len(FEATURE_NAMES)

# Defaults
SCID_PATH = os.environ.get('SCID_PATH', '/opt/SierraChart/Data/NQH26-CME.scid')
START_DATE = '2026-01-15'
END_DATE = '2026-03-18'
INITIAL_CAPITAL = 250_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip trade builder
# ─────────────────────────────────────────────────────────────────────────────

def build_round_trips(capture: dict) -> list[dict]:
    """Pair trade events into round-trip trades with entry/exit windows."""
    events = capture['trade_events']
    windows = capture['windows']
    timestamps = capture['timestamps']
    prices = capture['prices']
    actions = capture['actions']

    trades = []
    i = 0
    while i < len(events) - 1:
        entry = events[i]
        exit_ = events[i + 1]

        entry_idx, entry_action, entry_delta, entry_price, _, _ = entry
        exit_idx, exit_action, exit_delta, exit_price, exit_pnl, _ = exit_

        # Entry opens, exit closes/flips
        if (entry_action == "BUY" and exit_action == "SELL") or \
           (entry_action == "SELL" and exit_action == "BUY"):
            direction = "LONG" if entry_action == "BUY" else "SHORT"
            entry_time = datetime.fromtimestamp(timestamps[entry_idx], tz=timezone.utc)
            exit_time = datetime.fromtimestamp(timestamps[exit_idx], tz=timezone.utc)

            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_window': windows[entry_idx],   # (30, 36)
                'exit_window': windows[exit_idx],      # (30, 36)
                'entry_features': windows[entry_idx, -1, :],  # (36,) current bar
                'exit_features': windows[exit_idx, -1, :],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_hour': entry_time.hour,
                'entry_action': entry_action,
                'entry_raw_action': actions[entry_idx],
                'direction': direction,
                'pnl_usd': exit_pnl,
                'hold_bars': exit_idx - entry_idx,
                'is_winner': exit_pnl > 0,
            })
            i += 2
        else:
            i += 1

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Profile mode
# ─────────────────────────────────────────────────────────────────────────────

def analyze_profile(captures: list[dict]):
    """Entry feature Z-scores vs baseline, window context, action distribution."""
    model_zscores = {}

    for cap in captures:
        label = cap['label']
        windows = cap['windows']
        actions = cap['actions']
        n_bars = cap['n_bars']

        # Find entry bars (position changes from flat to non-flat)
        events = cap['trade_events']
        entry_indices = [e[0] for e in events if e[5] == 0 and e[2] != 0]

        if not entry_indices:
            print(f"\n  {label}: No entry events found")
            continue

        # All-bar baseline: current-bar features (last bar in window)
        all_current = windows[:, -1, :]    # (N, 36)
        all_mean = all_current.mean(axis=0)
        all_std = all_current.std(axis=0) + 1e-8

        # Entry-bar features
        entry_current = windows[entry_indices, -1, :]  # (n_entries, 36)
        entry_mean = entry_current.mean(axis=0)

        # Z-scores
        z_scores = (entry_mean - all_mean) / all_std
        ranked = np.argsort(np.abs(z_scores))[::-1]

        print(f"\n{'='*70}")
        print(f"FEATURE PROFILE: {label}")
        print(f"  Bars: {n_bars}, Entries: {len(entry_indices)}, "
              f"Total PnL: ${cap['total_pnl']:+,.0f}")
        print(f"{'='*70}")

        # Section 1: Current-bar Z-scores
        print(f"\n  CURRENT-BAR FEATURES AT ENTRY (top 15 by |Z|):")
        print(f"  {'Feature':<28} {'Entry':>8} {'All':>8} {'Z':>7}")
        print(f"  {'-'*55}")
        for i in ranked[:15]:
            print(f"  {FEATURE_NAMES[i]:<28} {entry_mean[i]:>+8.4f} "
                  f"{all_mean[i]:>+8.4f} {z_scores[i]:>+7.3f}")

        # Section 2: 30-bar window average
        entry_windows = windows[entry_indices]  # (n_entries, 30, 36)
        all_window_avgs = windows.mean(axis=1)  # (N, 36)
        entry_window_avgs = entry_windows.mean(axis=1)  # (n_entries, 36)

        aw_mean = all_window_avgs.mean(axis=0)
        aw_std = all_window_avgs.std(axis=0) + 1e-8
        ew_mean = entry_window_avgs.mean(axis=0)
        wz = (ew_mean - aw_mean) / aw_std
        w_ranked = np.argsort(np.abs(wz))[::-1]

        print(f"\n  WINDOW-AVERAGE CONTEXT (top 10 by |Z|):")
        print(f"  {'Feature':<28} {'Entry':>8} {'All':>8} {'Z':>7}")
        print(f"  {'-'*55}")
        for i in w_ranked[:10]:
            print(f"  {FEATURE_NAMES[i]:<28} {ew_mean[i]:>+8.4f} "
                  f"{aw_mean[i]:>+8.4f} {wz[i]:>+7.3f}")

        # Section 3: Intra-window momentum
        entry_first5 = entry_windows[:, :5, :].mean(axis=(0, 1))
        entry_last5 = entry_windows[:, -5:, :].mean(axis=(0, 1))
        momentum = entry_last5 - entry_first5
        m_ranked = np.argsort(np.abs(momentum))[::-1]

        print(f"\n  INTRA-WINDOW MOMENTUM (last5 - first5, top 10):")
        print(f"  {'Feature':<28} {'First5':>8} {'Last5':>8} {'Delta':>8}")
        print(f"  {'-'*55}")
        for i in m_ranked[:10]:
            print(f"  {FEATURE_NAMES[i]:<28} {entry_first5[i]:>+8.4f} "
                  f"{entry_last5[i]:>+8.4f} {momentum[i]:>+8.4f}")

        # Section 4: Action distribution
        print(f"\n  ACTION DISTRIBUTION:")
        print(f"  Mean={actions.mean():+.4f}  Std={actions.std():.4f}  "
              f"Min={actions.min():+.4f}  Max={actions.max():+.4f}")

        model_zscores[label] = z_scores

    # Cross-model comparison
    if len(model_zscores) >= 2:
        labels_list = list(model_zscores.keys())
        print(f"\n{'='*70}")
        print(f"CROSS-MODEL COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Feature':<28} ", end='')
        for lbl in labels_list:
            print(f" {lbl[:12]:>12}", end='')
        print(f"  {'Agree?':>8}")
        print(f"  {'-'*70}")

        all_z = np.array([model_zscores[l] for l in labels_list])
        avg_abs_z = np.abs(all_z).mean(axis=0)
        ranked_shared = np.argsort(avg_abs_z)[::-1]

        for i in ranked_shared[:15]:
            vals = all_z[:, i]
            agree = "✓" if np.all(np.sign(vals) == np.sign(vals[0])) else "✗"
            print(f"  {FEATURE_NAMES[i]:<28} ", end='')
            for v in vals:
                print(f" {v:>+12.3f}", end='')
            print(f"  {agree:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Losses mode
# ─────────────────────────────────────────────────────────────────────────────

def analyze_losses(captures: list[dict]):
    """Win vs loss comparison, clustering by hold time, direction, session."""
    model_loss_profiles = {}

    for cap in captures:
        label = cap['label']
        windows = cap['windows']
        trades = build_round_trips(cap)

        if not trades:
            print(f"\n  {label}: No round-trip trades found")
            continue

        winners = [t for t in trades if t['is_winner']]
        losers = [t for t in trades if not t['is_winner']]

        total_pnl = sum(t['pnl_usd'] for t in trades)
        win_pnl = sum(t['pnl_usd'] for t in winners)
        loss_pnl = sum(t['pnl_usd'] for t in losers)

        print(f"\n{'='*70}")
        print(f"WIN/LOSS ANALYSIS: {label}")
        print(f"  Trades: {len(trades)} (W:{len(winners)} L:{len(losers)})")
        print(f"  PnL: ${total_pnl:+,.0f} (Win:${win_pnl:+,.0f} Loss:${loss_pnl:+,.0f})")
        print(f"{'='*70}")

        if not losers or not winners:
            print("  Need both winners and losers for comparison")
            continue

        # All-bar baseline
        all_current = windows[:, -1, :]
        all_mean = all_current.mean(axis=0)
        all_std = all_current.std(axis=0) + 1e-8

        # Section 1: Current-bar features (win vs loss vs all)
        win_features = np.array([t['entry_features'] for t in winners])
        loss_features = np.array([t['entry_features'] for t in losers])
        win_mean = win_features.mean(axis=0)
        loss_mean = loss_features.mean(axis=0)
        win_z = (win_mean - all_mean) / all_std
        loss_z = (loss_mean - all_mean) / all_std
        gap = loss_z - win_z

        ranked = np.argsort(np.abs(gap))[::-1]

        print(f"\n  CURRENT-BAR: LOSS vs WIN (top 15 gap):")
        print(f"  {'Feature':<28} {'LossZ':>7} {'WinZ':>7} {'Gap':>7}")
        print(f"  {'-'*55}")
        for i in ranked[:15]:
            print(f"  {FEATURE_NAMES[i]:<28} {loss_z[i]:>+7.3f} "
                  f"{win_z[i]:>+7.3f} {gap[i]:>+7.3f}")

        model_loss_profiles[label] = {'loss_z': loss_z, 'win_z': win_z, 'gap': gap}

        # Section 2: Window context (win vs loss)
        win_windows = np.array([t['entry_window'] for t in winners])
        loss_windows = np.array([t['entry_window'] for t in losers])
        win_wm = win_windows.mean(axis=(0, 1))
        loss_wm = loss_windows.mean(axis=(0, 1))
        all_wm = windows.mean(axis=(0, 1))
        all_ws = windows.std(axis=(0, 1)) + 1e-8
        w_gap = (loss_wm - win_wm) / all_ws
        w_ranked = np.argsort(np.abs(w_gap))[::-1]

        print(f"\n  WINDOW-AVG: LOSS vs WIN (top 10):")
        print(f"  {'Feature':<28} {'Loss':>8} {'Win':>8} {'Gap/σ':>7}")
        print(f"  {'-'*55}")
        for i in w_ranked[:10]:
            print(f"  {FEATURE_NAMES[i]:<28} {loss_wm[i]:>+8.4f} "
                  f"{win_wm[i]:>+8.4f} {w_gap[i]:>+7.3f}")

        # Section 3: Intra-window momentum (win vs loss)
        win_mom = (win_windows[:, -5:, :].mean(axis=(0, 1)) -
                   win_windows[:, :5, :].mean(axis=(0, 1)))
        loss_mom = (loss_windows[:, -5:, :].mean(axis=(0, 1)) -
                    loss_windows[:, :5, :].mean(axis=(0, 1)))
        mom_diff = loss_mom - win_mom
        mom_ranked = np.argsort(np.abs(mom_diff))[::-1]

        print(f"\n  INTRA-WINDOW MOMENTUM DIFF (loss - win, top 10):")
        print(f"  {'Feature':<28} {'LossMom':>8} {'WinMom':>8} {'Diff':>8}")
        print(f"  {'-'*55}")
        for i in mom_ranked[:10]:
            print(f"  {FEATURE_NAMES[i]:<28} {loss_mom[i]:>+8.4f} "
                  f"{win_mom[i]:>+8.4f} {mom_diff[i]:>+8.4f}")

        # Section 4: Loss clustering by hold time
        print(f"\n  LOSS CLUSTERING BY HOLD TIME:")
        buckets = {'Short(≤10)': [], 'Medium(11-50)': [], 'Long(>50)': []}
        for t in losers:
            if t['hold_bars'] <= 10:
                buckets['Short(≤10)'].append(t)
            elif t['hold_bars'] <= 50:
                buckets['Medium(11-50)'].append(t)
            else:
                buckets['Long(>50)'].append(t)

        for bname, btrades in buckets.items():
            if not btrades:
                continue
            bpnl = sum(t['pnl_usd'] for t in btrades)
            print(f"  {bname}: {len(btrades)} trades, ${bpnl:+,.0f}")
            bf = np.array([t['entry_features'] for t in btrades])
            bz = (bf.mean(axis=0) - all_mean) / all_std
            top3 = np.argsort(np.abs(bz))[::-1][:3]
            for i in top3:
                print(f"    {FEATURE_NAMES[i]}: Z={bz[i]:+.3f}")

        # Section 5: Loss clustering by direction
        print(f"\n  LOSS CLUSTERING BY DIRECTION:")
        for direction in ("LONG", "SHORT"):
            dir_losses = [t for t in losers if t['direction'] == direction]
            if not dir_losses:
                continue
            dpnl = sum(t['pnl_usd'] for t in dir_losses)
            print(f"  {direction}: {len(dir_losses)} losses, ${dpnl:+,.0f}")
            df = np.array([t['entry_features'] for t in dir_losses])
            dz = (df.mean(axis=0) - all_mean) / all_std
            top3 = np.argsort(np.abs(dz))[::-1][:3]
            for i in top3:
                print(f"    {FEATURE_NAMES[i]}: Z={dz[i]:+.3f}")

        # Section 6: Loss clustering by session
        print(f"\n  LOSS CLUSTERING BY SESSION (UTC hour):")
        session_buckets = defaultdict(list)
        for t in trades:
            h = t['entry_hour']
            if h < 6:
                session = "00-06 (Asia)"
            elif h < 12:
                session = "06-12 (Euro AM)"
            elif h < 15:
                session = "12-15 (Euro PM)"
            elif h < 20:
                session = "15-20 (US AM)"
            else:
                session = "20-24 (US PM)"
            session_buckets[session].append(t)

        for sess, strades in sorted(session_buckets.items()):
            spnl = sum(t['pnl_usd'] for t in strades)
            sw = sum(1 for t in strades if t['is_winner'])
            sl = len(strades) - sw
            print(f"  {sess}: {len(strades)} trades (W:{sw} L:{sl}) ${spnl:+,.0f}")

        # Section 7: Worst 10 trades
        sorted_by_pnl = sorted(trades, key=lambda t: t['pnl_usd'])
        print(f"\n  WORST 10 TRADES:")
        print(f"  {'#':>3} {'Dir':<6} {'PnL':>10} {'Hold':>5} "
              f"{'Time':>20} {'Entry':>10} {'Exit':>10}")
        print(f"  {'-'*70}")
        for rank, t in enumerate(sorted_by_pnl[:10], 1):
            print(f"  {rank:>3} {t['direction']:<6} ${t['pnl_usd']:>+9,.0f} "
                  f"{t['hold_bars']:>5} "
                  f"{t['entry_time'].strftime('%m/%d %H:%M'):>20} "
                  f"{t['entry_price']:>10.2f} {t['exit_price']:>10.2f}")

    # Cross-model shared failure signals
    if len(model_loss_profiles) >= 2:
        labels_list = list(model_loss_profiles.keys())
        print(f"\n{'='*70}")
        print(f"CROSS-MODEL SHARED FAILURE SIGNALS")
        print(f"{'='*70}")

        gaps = np.array([model_loss_profiles[l]['gap'] for l in labels_list])
        same_sign = np.all(np.sign(gaps) == np.sign(gaps[0:1, :]), axis=0)
        avg_abs_gap = np.abs(gaps).mean(axis=0)
        shared_rank = np.argsort(avg_abs_gap * same_sign.astype(float))[::-1]

        print(f"  {'Feature':<28} ", end='')
        for lbl in labels_list:
            print(f" {lbl[:10]:>10}", end='')
        print(f"  {'Shared?':>8}")
        print(f"  {'-'*70}")

        shown = 0
        for i in shared_rank:
            if not same_sign[i]:
                continue
            if shown >= 10:
                break
            print(f"  {FEATURE_NAMES[i]:<28} ", end='')
            for g in gaps[:, i]:
                print(f" {g:>+10.3f}", end='')
            print(f"  {'✓':>8}")
            shown += 1


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: Deep-dive mode
# ─────────────────────────────────────────────────────────────────────────────

def analyze_deep_dive(capture: dict, trade_num: int):
    """Deep-dive into a single trade: window features, evolution, context."""
    label = capture['label']
    trades = build_round_trips(capture)

    if not trades:
        print(f"  {label}: No round-trip trades found")
        return

    # List all trades
    print(f"\n{'='*70}")
    print(f"ALL TRADES: {label}")
    print(f"{'='*70}")
    cum_pnl = 0.0
    print(f"  {'#':>3} {'Dir':<6} {'PnL':>10} {'CumPnL':>10} {'Hold':>5} "
          f"{'Time':>20} {'Entry':>10}")
    print(f"  {'-'*70}")
    for i, t in enumerate(trades, 1):
        cum_pnl += t['pnl_usd']
        marker = " ←" if i == trade_num else ""
        print(f"  {i:>3} {t['direction']:<6} ${t['pnl_usd']:>+9,.0f} "
              f"${cum_pnl:>+9,.0f} {t['hold_bars']:>5} "
              f"{t['entry_time'].strftime('%m/%d %H:%M'):>20} "
              f"{t['entry_price']:>10.2f}{marker}")

    if trade_num < 1 or trade_num > len(trades):
        print(f"\n  Trade #{trade_num} out of range (1-{len(trades)})")
        return

    t = trades[trade_num - 1]
    windows = capture['windows']
    all_current = windows[:, -1, :]
    all_mean = all_current.mean(axis=0)
    all_std = all_current.std(axis=0) + 1e-8

    print(f"\n{'='*70}")
    print(f"DEEP-DIVE: Trade #{trade_num}")
    print(f"  {t['direction']} @ {t['entry_price']:.2f} → "
          f"{t['exit_price']:.2f} | PnL: ${t['pnl_usd']:+,.0f}")
    print(f"  Entry: {t['entry_time']} | Exit: {t['exit_time']}")
    print(f"  Hold: {t['hold_bars']} bars | Action: {t['entry_raw_action']:+.4f}")
    print(f"{'='*70}")

    # 30-bar window at entry
    w = t['entry_window']  # (30, 36)
    current = w[-1, :]     # last bar

    # Entry Z-scores
    z = (current - all_mean) / all_std
    ranked = np.argsort(np.abs(z))[::-1]

    print(f"\n  ENTRY-BAR FEATURES (top 15 by |Z|):")
    print(f"  {'Feature':<28} {'Value':>8} {'AvgAll':>8} {'Z':>7}")
    print(f"  {'-'*55}")
    for i in ranked[:15]:
        print(f"  {FEATURE_NAMES[i]:<28} {current[i]:>+8.4f} "
              f"{all_mean[i]:>+8.4f} {z[i]:>+7.3f}")

    # Feature evolution across window
    sample_bars = [0, 4, 9, 14, 19, 24, 29]
    print(f"\n  FEATURE EVOLUTION ACROSS 30-BAR WINDOW:")
    print(f"  {'Feature':<22}", end='')
    for b in sample_bars:
        print(f" {'Bar'+str(b+1):>8}", end='')
    print()
    print(f"  {'-'*80}")

    # Show top 10 most variable features in the window
    var = w.std(axis=0)
    var_ranked = np.argsort(var)[::-1]
    for i in var_ranked[:10]:
        print(f"  {FEATURE_NAMES[i]:<22}", end='')
        for b in sample_bars:
            print(f" {w[b, i]:>+8.4f}", end='')
        print()

    # Entry vs Exit comparison
    if t['exit_idx'] < len(windows):
        exit_current = t['exit_features']
        print(f"\n  ENTRY vs EXIT FEATURES (top changes):")
        delta = exit_current - current
        d_ranked = np.argsort(np.abs(delta))[::-1]
        print(f"  {'Feature':<28} {'Entry':>8} {'Exit':>8} {'Delta':>8}")
        print(f"  {'-'*55}")
        for i in d_ranked[:10]:
            print(f"  {FEATURE_NAMES[i]:<28} {current[i]:>+8.4f} "
                  f"{exit_current[i]:>+8.4f} {delta[i]:>+8.4f}")

    # Price context around trade
    prices = capture['prices']
    entry_idx = t['entry_idx']
    print(f"\n  PRICE ACTION CONTEXT:")
    start = max(0, entry_idx - 10)
    end = min(len(prices), t['exit_idx'] + 5)
    for bi in range(start, end):
        marker = ""
        if bi == entry_idx:
            marker = f" ← ENTRY ({t['entry_action']})"
        elif bi == t['exit_idx']:
            marker = " ← EXIT"
        print(f"  Bar {bi:>5}: {prices[bi]:>10.2f}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Unified trade analysis for Wyckoff RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('mode', choices=['profile', 'losses', 'deep-dive', 'all'],
                   help='Analysis mode')
    p.add_argument('--checkpoints', '-c', nargs='+', required=True,
                   help='Checkpoint .pt file paths')
    p.add_argument('--labels', '-l', nargs='+', default=None,
                   help='Labels for each checkpoint')
    p.add_argument('--continuous', action='store_true',
                   help='Use continuous position sizing')
    p.add_argument('--trade', type=int, default=1,
                   help='Trade number for deep-dive mode (1-based)')
    p.add_argument('--scid', default=SCID_PATH)
    p.add_argument('--start-date', default=START_DATE)
    p.add_argument('--end-date', default=END_DATE)
    p.add_argument('--capital', type=float, default=INITIAL_CAPITAL)
    return p


def main():
    args = build_parser().parse_args()

    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"ERROR: Checkpoint not found: {path}", file=sys.stderr)
            sys.exit(1)

    if args.labels:
        if len(args.labels) != len(args.checkpoints):
            print("ERROR: --labels count must match --checkpoints count",
                  file=sys.stderr)
            sys.exit(1)
        labels = args.labels
    else:
        labels = [os.path.basename(p).replace('.pt', '')[:20]
                  for p in args.checkpoints]

    print(f"\n{'='*60}", flush=True)
    print(f"WYCKOFF TRADE ANALYSIS — mode: {args.mode}", flush=True)
    print(f"Period: {args.start_date} → {args.end_date}", flush=True)
    print(f"Checkpoints: {len(args.checkpoints)}", flush=True)
    print(f"{'='*60}", flush=True)

    # Precompute bars + features ONCE
    from wyckoff_rl.live.precompute import PrecomputedReplay
    print(f"\n  Precomputing bars + features...", flush=True)
    replay = PrecomputedReplay.from_scid(
        args.scid, args.start_date, args.end_date,
    )
    print(f"  → {replay.n_bars} bars, {replay.n_windows} windows\n", flush=True)

    # Run captures (fast — inference only)
    captures = []
    for label, ckpt_path in zip(labels, args.checkpoints):
        print(f"  Capturing: {label}...", flush=True)
        cap = replay.run_capture(ckpt_path, continuous=args.continuous,
                                 initial_capital=args.capital)
        cap['label'] = label
        n_rt = len(build_round_trips(cap))
        print(f"  → {cap['n_bars']} bars, {cap['n_trades']} trades, "
              f"{n_rt} round-trips, ${cap['total_pnl']:+,.0f}",
              flush=True)
        captures.append(cap)

    # Run analysis
    if args.mode in ('profile', 'all'):
        analyze_profile(captures)

    if args.mode in ('losses', 'all'):
        analyze_losses(captures)

    if args.mode == 'deep-dive':
        analyze_deep_dive(captures[0], args.trade)


if __name__ == '__main__':
    main()
