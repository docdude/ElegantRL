#!/usr/bin/env python3
"""
Bad Trade Analysis: Why did the winning models lose money on specific trades?

Runs split3 and split4 continuous replays with monkey-patched feature capture,
then clusters losing trades by:
  1. Feature regime at entry (z-scores vs all-bar baseline)
  2. Hold time (short churn vs long drawdown)
  3. Direction (LONG vs SHORT)
  4. Time of day (UTC hours)
  5. Intra-window feature trends (momentum building or fading)

Compares losing-trade feature profiles vs winning-trade feature profiles
to find what the model gets wrong.
"""
import logging, sys, os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import defaultdict

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

CHECKPOINTS = {
    'split3_s2.29': '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous/split3_actor__000001488896_00491.096.pt',
    'split4_s2.37': '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous/split4_actor__000000874496_00444.927.pt',
}


def run_feature_capture(label, ckpt_path):
    """Run replay with monkey-patched feature capture, returning per-trade data."""
    capture = {
        'windows': [],
        'actions': [],
        'prices': [],
        'timestamps': [],
        'positions': [],
        'trade_events': [],   # (bar_idx, action_str, delta, price, pnl, position_before)
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
            pnl = 0.0
            if self._position != 0:
                if (self._position > 0 and delta < 0) or (self._position < 0 and delta > 0):
                    closed = min(abs(self._position), abs(delta))
                    pnl_pts = (bar.close - self._entry_price) * (1 if self._position > 0 else -1)
                    pnl = closed * pnl_pts * 20.0
            capture['trade_events'].append((bar_idx, action_str, delta, bar.close, pnl, self._position))
            self._execute_trade(delta, bar.close, raw_action, bar)

    WyckoffTrader._on_bar_complete = patched_on_bar

    print(f"\nRunning {label} replay with feature capture...", flush=True)
    data = SCIDReplayAdapter(scid_path=SCID_PATH, start_date='2026-01-15', end_date='2026-03-18', speed=0.0)
    sim = SimExecutor(initial_capital=250_000.0)
    trader = WyckoffTrader(
        checkpoint_path=ckpt_path, data_adapter=data, order_adapter=sim,
        range_size=40.0, continuous_sizing=True, max_contracts=1,
        log_dir=f'/tmp/bad_trade_analysis_{label}',
    )
    trader.start()
    WyckoffTrader._on_bar_complete = original_on_bar

    print(f"  Captured {len(capture['windows'])} bars, {len(capture['trade_events'])} trade events", flush=True)
    return capture


def pair_round_trips(capture):
    """Convert trade events into round-trip trades with entry/exit feature windows."""
    events = capture['trade_events']
    all_windows = np.array(capture['windows'])
    trades = []
    entry_event = None

    for i, (bidx, act, delta, price, pnl, pos_before) in enumerate(events):
        if entry_event is None:
            # Opening trade
            entry_event = (bidx, act, delta, price, pnl, pos_before)
        elif pnl != 0:
            # Closing trade (has realized PnL)
            e_bidx, e_act, e_delta, e_price, _, e_pos = entry_event
            direction = 'LONG' if e_act == 'BUY' else 'SHORT'
            hold_bars = bidx - e_bidx

            entry_ts = capture['timestamps'][e_bidx]
            exit_ts = capture['timestamps'][bidx]
            entry_dt = datetime.fromtimestamp(entry_ts, tz=timezone.utc)
            exit_dt = datetime.fromtimestamp(exit_ts, tz=timezone.utc)

            trades.append({
                'trade_num': len(trades) + 1,
                'direction': direction,
                'entry_bar': e_bidx,
                'exit_bar': bidx,
                'entry_price': e_price,
                'exit_price': price,
                'entry_time': entry_dt,
                'exit_time': exit_dt,
                'hold_bars': hold_bars,
                'pnl_usd': pnl,
                'entry_action': capture['actions'][e_bidx],
                'exit_action': capture['actions'][bidx],
                'entry_window': all_windows[e_bidx],       # (30, 36)
                'exit_window': all_windows[bidx],           # (30, 36)
                'entry_hour': entry_dt.hour,
            })

            # Check if this exit also opens a new position
            new_pos = pos_before + delta
            if new_pos != 0:
                entry_event = (bidx, act, delta, price, 0, pos_before + delta)
            else:
                entry_event = None
        else:
            # Flip: close + open in one action
            entry_event = (bidx, act, delta, price, pnl, pos_before)

    return trades


def analyze_bad_trades(label, trades, capture):
    """Full analysis of losing vs winning trades."""
    all_windows = np.array(capture['windows'])
    tdf = pd.DataFrame([{k: v for k, v in t.items() if k not in ('entry_window', 'exit_window')} for t in trades])

    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] < 0]
    n_total = len(trades)

    print(f"\n{'#'*90}")
    print(f"  BAD TRADE ANALYSIS: {label}")
    print(f"  {n_total} round trips: {len(wins)} wins ({len(wins)/n_total*100:.1f}%), {len(losses)} losses ({len(losses)/n_total*100:.1f}%)")
    print(f"{'#'*90}")

    if not losses:
        print("  No losses to analyze!")
        return

    # ═══════════════════════════════════════════════════════════════════
    # 1. FEATURE Z-SCORES: Losing entries vs Winning entries
    # ═══════════════════════════════════════════════════════════════════
    win_entry_feats = np.array([t['entry_window'][-1] for t in wins])   # current bar at entry
    loss_entry_feats = np.array([t['entry_window'][-1] for t in losses])
    all_bar_feats = all_windows[:, -1, :]  # baseline: all bars

    mean_all = all_bar_feats.mean(axis=0)
    std_all = all_bar_feats.std(axis=0)
    mean_win = win_entry_feats.mean(axis=0)
    mean_loss = loss_entry_feats.mean(axis=0)

    print(f"\n{'='*100}")
    print(f"1. CURRENT-BAR FEATURES AT ENTRY: Losing vs Winning vs All-bars Baseline")
    print(f"{'='*100}")
    print(f"{'Feature':<30} {'All Mean':>9} {'Win Mean':>9} {'Loss Mean':>10} {'Win Z':>7} {'Loss Z':>7} {'Gap':>7} {'Signal':<20}")
    print('-'*100)

    comparisons = []
    for i, fname in enumerate(FEATURE_NAMES):
        s = std_all[i] + 1e-8
        z_win = (mean_win[i] - mean_all[i]) / s
        z_loss = (mean_loss[i] - mean_all[i]) / s
        gap = z_loss - z_win
        comparisons.append((abs(gap), i, fname, mean_all[i], mean_win[i], mean_loss[i], z_win, z_loss, gap))

    comparisons.sort(reverse=True)
    for absg, i, fname, ma, mw, ml, zw, zl, gap in comparisons:
        signal = ""
        if absg > 0.3:
            if gap > 0:
                signal = "LOSSES HIGHER"
            else:
                signal = "LOSSES LOWER"
        marker = "***" if absg > 0.5 else " * " if absg > 0.3 else "   "
        print(f"{marker}{fname:<27} {ma:>+9.4f} {mw:>+9.4f} {ml:>+10.4f} {zw:>+7.2f} {zl:>+7.2f} {gap:>+7.2f} {signal:<20}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. WINDOW-AVERAGE FEATURES (30-bar context)
    # ═══════════════════════════════════════════════════════════════════
    win_wavg = np.array([t['entry_window'].mean(axis=0) for t in wins])
    loss_wavg = np.array([t['entry_window'].mean(axis=0) for t in losses])
    all_wavg = all_windows.mean(axis=1)

    mean_wavg_all = all_wavg.mean(axis=0)
    std_wavg_all = all_wavg.std(axis=0)

    print(f"\n{'='*100}")
    print(f"2. WINDOW-AVERAGE (30-bar context): Losing vs Winning")
    print(f"{'='*100}")
    print(f"{'Feature':<30} {'Win WAvg':>9} {'Loss WAvg':>10} {'Win Z':>7} {'Loss Z':>7} {'Gap':>7} {'Signal':<20}")
    print('-'*100)

    w_comparisons = []
    for i, fname in enumerate(FEATURE_NAMES):
        s = std_wavg_all[i] + 1e-8
        z_win = (win_wavg.mean(axis=0)[i] - mean_wavg_all[i]) / s
        z_loss = (loss_wavg.mean(axis=0)[i] - mean_wavg_all[i]) / s
        gap = z_loss - z_win
        w_comparisons.append((abs(gap), fname, win_wavg.mean(axis=0)[i], loss_wavg.mean(axis=0)[i], z_win, z_loss, gap))

    w_comparisons.sort(reverse=True)
    for absg, fname, mw, ml, zw, zl, gap in w_comparisons:
        signal = ""
        if absg > 0.3:
            signal = "LOSSES HIGHER" if gap > 0 else "LOSSES LOWER"
        marker = "***" if absg > 0.5 else " * " if absg > 0.3 else "   "
        print(f"{marker}{fname:<27} {mw:>+9.4f} {ml:>+10.4f} {zw:>+7.2f} {zl:>+7.2f} {gap:>+7.2f} {signal:<20}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. INTRA-WINDOW MOMENTUM: Are losses entering on fading momentum?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"3. INTRA-WINDOW MOMENTUM: Last 5 bars minus First 5 bars at Entry")
    print(f"{'='*100}")
    print(f"{'Feature':<30} {'Win Δ':>9} {'Loss Δ':>10} {'Diff':>8} {'Pattern':<25}")
    print('-'*90)

    win_early = np.array([t['entry_window'][:5].mean(axis=0) for t in wins])
    win_late = np.array([t['entry_window'][-5:].mean(axis=0) for t in wins])
    loss_early = np.array([t['entry_window'][:5].mean(axis=0) for t in losses])
    loss_late = np.array([t['entry_window'][-5:].mean(axis=0) for t in losses])

    win_delta = (win_late - win_early).mean(axis=0)
    loss_delta = (loss_late - loss_early).mean(axis=0)

    momentum_diffs = []
    for i, fname in enumerate(FEATURE_NAMES):
        diff = loss_delta[i] - win_delta[i]
        pattern = ""
        if abs(diff) > 0.03:
            if win_delta[i] > 0.02 and loss_delta[i] < -0.02:
                pattern = "WIN RISING, LOSS FALLING"
            elif win_delta[i] < -0.02 and loss_delta[i] > 0.02:
                pattern = "WIN FALLING, LOSS RISING"
            elif abs(loss_delta[i]) < abs(win_delta[i]) * 0.5:
                pattern = "LOSS MOMENTUM WEAK"
            else:
                pattern = "DIVERGENT"
        momentum_diffs.append((abs(diff), fname, win_delta[i], loss_delta[i], diff, pattern))

    momentum_diffs.sort(reverse=True)
    for absd, fname, wd, ld, diff, pat in momentum_diffs[:20]:
        marker = "***" if absd > 0.06 else " * " if absd > 0.03 else "   "
        print(f"{marker}{fname:<27} {wd:>+9.4f} {ld:>+10.4f} {diff:>+8.4f} {pat:<25}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. LOSS CLUSTERING BY HOLD TIME
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"4. LOSS CLUSTERING BY HOLD TIME")
    print(f"{'='*100}")

    short_losses = [t for t in losses if t['hold_bars'] <= 10]
    medium_losses = [t for t in losses if 10 < t['hold_bars'] <= 50]
    long_losses = [t for t in losses if t['hold_bars'] > 50]

    for bucket, name in [(short_losses, 'SHORT HOLD (≤10 bars)'),
                         (medium_losses, 'MEDIUM HOLD (11-50 bars)'),
                         (long_losses, 'LONG HOLD (>50 bars)')]:
        if not bucket:
            continue
        pnls = np.array([t['pnl_usd'] for t in bucket])
        holds = np.array([t['hold_bars'] for t in bucket])
        dirs = [t['direction'] for t in bucket]
        n_long = sum(1 for d in dirs if d == 'LONG')
        n_short = sum(1 for d in dirs if d == 'SHORT')
        print(f"\n  {name}: {len(bucket)} trades")
        print(f"    Avg PnL: ${pnls.mean():,.0f}, Total: ${pnls.sum():,.0f}")
        print(f"    Avg hold: {holds.mean():.1f} bars, Median: {np.median(holds):.0f}")
        print(f"    Direction: {n_long} LONG, {n_short} SHORT")

        # Top features that distinguish this bucket
        bucket_feats = np.array([t['entry_window'][-1] for t in bucket])
        bucket_mean = bucket_feats.mean(axis=0)
        print(f"    Top distinguishing features vs all-bar baseline:")
        devs = []
        for i, fname in enumerate(FEATURE_NAMES):
            z = (bucket_mean[i] - mean_all[i]) / (std_all[i] + 1e-8)
            devs.append((abs(z), fname, z))
        devs.sort(reverse=True)
        for absz, fname, z in devs[:8]:
            dir_str = "HIGH" if z > 0 else "LOW"
            print(f"      {fname:<28} z={z:>+.2f} ({dir_str})")

    # ═══════════════════════════════════════════════════════════════════
    # 5. LOSS CLUSTERING BY DIRECTION
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"5. LOSS CLUSTERING BY DIRECTION")
    print(f"{'='*100}")

    for direction in ['LONG', 'SHORT']:
        dir_losses = [t for t in losses if t['direction'] == direction]
        dir_wins = [t for t in wins if t['direction'] == direction]
        if not dir_losses:
            continue

        pnls = np.array([t['pnl_usd'] for t in dir_losses])
        print(f"\n  {direction} LOSSES: {len(dir_losses)} trades, total ${pnls.sum():,.0f}, avg ${pnls.mean():,.0f}")
        print(f"  {direction} WINS:   {len(dir_wins)} trades" + (f", avg ${np.mean([t['pnl_usd'] for t in dir_wins]):,.0f}" if dir_wins else ""))

        if dir_wins:
            loss_feats = np.array([t['entry_window'][-1] for t in dir_losses])
            win_feats = np.array([t['entry_window'][-1] for t in dir_wins])
            print(f"    Features where losing {direction}s differ from winning {direction}s:")
            devs = []
            for i, fname in enumerate(FEATURE_NAMES):
                diff = loss_feats.mean(axis=0)[i] - win_feats.mean(axis=0)[i]
                s = all_bar_feats[:, i].std() + 1e-8
                z = diff / s
                devs.append((abs(z), fname, z, loss_feats.mean(axis=0)[i], win_feats.mean(axis=0)[i]))
            devs.sort(reverse=True)
            for absz, fname, z, lm, wm in devs[:10]:
                signal = "LOSS HIGHER" if z > 0 else "LOSS LOWER"
                marker = "***" if absz > 0.3 else " * " if absz > 0.15 else "   "
                print(f"    {marker}{fname:<26} win={wm:>+.4f} loss={lm:>+.4f} z={z:>+.2f} {signal}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. LOSS CLUSTERING BY TIME OF DAY
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"6. LOSS CLUSTERING BY TIME OF DAY (UTC)")
    print(f"{'='*100}")

    # Group losses into sessions
    sessions = {
        'Asia/Overnight (00-08 UTC)': (0, 8),
        'EU Open (08-13 UTC)': (8, 13),
        'US Open (13-16 UTC)': (13, 16),
        'US Afternoon (16-20 UTC)': (16, 20),
        'US Close/Evening (20-24 UTC)': (20, 24),
    }

    for session_name, (h_start, h_end) in sessions.items():
        sess_losses = [t for t in losses if h_start <= t['entry_hour'] < h_end]
        sess_wins = [t for t in wins if h_start <= t['entry_hour'] < h_end]
        if not sess_losses and not sess_wins:
            continue

        n_loss = len(sess_losses)
        n_win = len(sess_wins)
        n_tot = n_loss + n_win
        loss_pnl = sum(t['pnl_usd'] for t in sess_losses) if sess_losses else 0
        win_pnl = sum(t['pnl_usd'] for t in sess_wins) if sess_wins else 0
        net = loss_pnl + win_pnl
        wr = n_win / n_tot * 100 if n_tot > 0 else 0

        marker = " <<<" if net < -5000 else ""
        print(f"\n  {session_name}: {n_tot} trades, WR={wr:.0f}%, net=${net:>+,.0f}{marker}")
        print(f"    Wins: {n_win} (${win_pnl:>+,.0f})  Losses: {n_loss} (${loss_pnl:>+,.0f})")

        if sess_losses and len(sess_losses) >= 3:
            sess_feats = np.array([t['entry_window'][-1] for t in sess_losses])
            print(f"    Loss-entry features vs baseline:")
            devs = []
            for i, fname in enumerate(FEATURE_NAMES):
                z = (sess_feats.mean(axis=0)[i] - mean_all[i]) / (std_all[i] + 1e-8)
                devs.append((abs(z), fname, z))
            devs.sort(reverse=True)
            for absz, fname, z in devs[:5]:
                print(f"      {fname:<28} z={z:>+.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # 7. WORST TRADES: Individual deep dives
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"7. WORST 10 TRADES — INDIVIDUAL FEATURE PROFILES")
    print(f"{'='*100}")

    sorted_losses = sorted(losses, key=lambda t: t['pnl_usd'])
    for t in sorted_losses[:10]:
        entry_feats = t['entry_window'][-1]
        print(f"\n  Trade #{t['trade_num']:3d}: {t['direction']} | "
              f"{t['entry_time'].strftime('%m/%d %H:%M')} → {t['exit_time'].strftime('%m/%d %H:%M')} | "
              f"Hold={t['hold_bars']} bars | PnL=${t['pnl_usd']:>+,.0f} | "
              f"Entry@{t['entry_price']:.0f} Exit@{t['exit_price']:.0f}")

        # Z-scores vs all-bar baseline
        outliers = []
        for i, fname in enumerate(FEATURE_NAMES):
            z = (entry_feats[i] - mean_all[i]) / (std_all[i] + 1e-8)
            outliers.append((abs(z), fname, z, entry_feats[i]))
        outliers.sort(reverse=True)
        print(f"    Top feature outliers at entry:")
        for absz, fname, z, val in outliers[:8]:
            dir_str = "HIGH" if z > 0 else "LOW"
            print(f"      {fname:<28} val={val:>+.4f} z={z:>+.2f} ({dir_str})")

        # Window momentum (last 5 - first 5)
        early = t['entry_window'][:5].mean(axis=0)
        late = t['entry_window'][-5:].mean(axis=0)
        delta = late - early
        mom_outliers = [(abs(delta[i]), FEATURE_NAMES[i], delta[i]) for i in range(len(FEATURE_NAMES))]
        mom_outliers.sort(reverse=True)
        print(f"    Window momentum (last5 - first5):")
        for absd, fname, d in mom_outliers[:5]:
            trend = "RISING" if d > 0 else "FALLING"
            print(f"      {fname:<28} Δ={d:>+.4f} ({trend})")

    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    all_model_trades = {}

    for label, ckpt in CHECKPOINTS.items():
        capture = run_feature_capture(label, ckpt)
        trades = pair_round_trips(capture)
        all_model_trades[label] = analyze_bad_trades(label, trades, capture)

    # ═══════════════════════════════════════════════════════════════════
    # Cross-model: shared failure patterns
    # ═══════════════════════════════════════════════════════════════════
    if len(all_model_trades) == 2:
        labels = list(all_model_trades.keys())
        all_windows_combined = []

        print(f"\n{'#'*100}")
        print(f"  CROSS-MODEL SHARED FAILURE PATTERNS")
        print(f"{'#'*100}")

        for label in labels:
            trades = all_model_trades[label]
            losses = [t for t in trades if t['pnl_usd'] < 0]
            wins = [t for t in trades if t['pnl_usd'] > 0]

            loss_feats = np.array([t['entry_window'][-1] for t in losses])
            win_feats = np.array([t['entry_window'][-1] for t in wins])

            # Compute per-model loss-vs-win z-scores
            combined = np.vstack([loss_feats, win_feats])
            combined_std = combined.std(axis=0) + 1e-8
            z_per_feat = (loss_feats.mean(0) - win_feats.mean(0)) / combined_std
            all_windows_combined.append(z_per_feat)

        z_model0 = all_windows_combined[0]
        z_model1 = all_windows_combined[1]

        print(f"\n{'Feature':<30} {labels[0]:>14} {labels[1]:>14} {'Agree?':>8} {'Avg Gap':>9} {'Interpretation':<25}")
        print('-'*110)

        cross = []
        for i, fname in enumerate(FEATURE_NAMES):
            agree = "YES" if (z_model0[i] > 0 and z_model1[i] > 0) or (z_model0[i] < 0 and z_model1[i] < 0) else "NO"
            avg = (z_model0[i] + z_model1[i]) / 2
            interp = ""
            if agree == "YES" and abs(avg) > 0.15:
                interp = "LOSSES HIGHER" if avg > 0 else "LOSSES LOWER"
            cross.append((abs(avg) if agree == "YES" else 0, fname, z_model0[i], z_model1[i], agree, avg, interp))

        cross.sort(reverse=True)
        for score, fname, z0, z1, agree, avg, interp in cross:
            marker = "***" if score > 0.3 else " * " if score > 0.15 else "   "
            print(f"{marker}{fname:<27} {z0:>+14.3f} {z1:>+14.3f} {agree:>8} {avg:>+9.3f} {interp:<25}")

        # Summary
        print(f"\n{'='*100}")
        print(f"SHARED SIGNALS: Features where BOTH models' losses show the same bias")
        print(f"{'='*100}")
        shared = [(score, fname, z0, z1, avg, interp) for score, fname, z0, z1, agree, avg, interp in cross
                  if agree == "YES" and score > 0.1]
        if shared:
            for score, fname, z0, z1, avg, interp in shared:
                print(f"  {fname:<30} avg_z={avg:>+.3f}  ({labels[0]}={z0:>+.3f}, {labels[1]}={z1:>+.3f})  → {interp}")
        else:
            print("  No strong shared signals found (models may fail on different conditions)")

    print("\n\nDone.", flush=True)
