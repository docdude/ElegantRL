#!/usr/bin/env python3
"""
Veto rule discovery for Wyckoff RL checkpoints.

Replaces: analyze_filter_candidates.py

Three discovery approaches:
  1. CRITIC VALUE GATE: Load cri.pth, compute V(state) at each entry.
     Sweep thresholds to find where vetoing low-V entries helps.
  2. FEATURE-BASED VETOES: For each of 36 features, sweep percentile
     thresholds for rules like "don't trade when feature > X".
  3. COMBINED: Critic gate + best feature veto together.

Usage:
    # Feature-only sweep (no critic needed)
    python -m wyckoff_rl.tools.discover_veto \\
        --checkpoints path/to/actor.pt --continuous

    # With critic value gate
    python -m wyckoff_rl.tools.discover_veto \\
        --checkpoints path/to/actor.pt \\
        --critics path/to/cri.pth --continuous

    # Multiple models
    python -m wyckoff_rl.tools.discover_veto \\
        --checkpoints split3.pt split4.pt \\
        --critics split3_cri.pth split4_cri.pth --continuous
"""

import argparse
import logging
import os
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timezone

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

FEATURE_NAMES = [ALL_FEATURES[i] for i in TRAINING_FEATURE_INDICES]
N_FEATURES = len(FEATURE_NAMES)

SCID_PATH = os.environ.get('SCID_PATH', '/opt/SierraChart/Data/NQH26-CME.scid')
START_DATE = '2026-01-15'
END_DATE = '2026-03-18'
INITIAL_CAPITAL = 250_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip trade builder
# ─────────────────────────────────────────────────────────────────────────────

def build_trades(capture: dict) -> list[dict]:
    """Build round-trip trades with entry features + optional critic values."""
    events = capture['trade_events']
    all_windows = capture['windows']
    has_critic = capture['has_critic']
    critic_vals = capture.get('critic_values', np.array([]))
    trades = []
    entry_event = None

    for ev in events:
        bidx, act, delta, price, pnl, pos_before = ev

        if entry_event is None:
            entry_event = ev
        elif pnl != 0:
            e_bidx, e_act, _, e_price, _, _ = entry_event
            direction = 'LONG' if e_act == 'BUY' else 'SHORT'

            entry_ts = capture['timestamps'][e_bidx]
            entry_dt = datetime.fromtimestamp(entry_ts, tz=timezone.utc)

            entry_window = all_windows[e_bidx]
            entry_features = entry_window[-1]

            trade = {
                'direction': direction,
                'entry_bar': e_bidx,
                'exit_bar': bidx,
                'entry_price': e_price,
                'exit_price': price,
                'pnl': pnl,
                'entry_hour': entry_dt.hour,
                'entry_time': entry_dt,
                'entry_raw_action': capture['actions'][e_bidx],
                'entry_features': entry_features,
                'hold_bars': bidx - e_bidx,
            }
            if has_critic and e_bidx < len(critic_vals):
                trade['entry_critic_value'] = critic_vals[e_bidx]

            trades.append(trade)

            new_pos = pos_before + delta
            entry_event = ev if new_pos != 0 else None
        else:
            entry_event = ev

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Critic gate analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_critic_gate(label: str, trades: list[dict]):
    """Test critic V(s) as a trade gate with threshold sweep."""
    if not trades or 'entry_critic_value' not in trades[0]:
        print(f"\n  {label}: No critic values — skipping critic gate analysis")
        return

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    win_v = np.array([t['entry_critic_value'] for t in wins])
    loss_v = np.array([t['entry_critic_value'] for t in losses])
    all_v = np.array([t['entry_critic_value'] for t in trades])
    pnls = np.array([t['pnl'] for t in trades])

    print(f"\n{'='*80}")
    print(f"  CRITIC VALUE GATE — {label}")
    print(f"{'='*80}")
    print(f"\n  Critic V(s) at entry:")
    print(f"    ALL    (n={len(all_v)}): mean={all_v.mean():.4f}  "
          f"std={all_v.std():.4f}  range=[{all_v.min():.4f}, {all_v.max():.4f}]")
    if len(win_v):
        print(f"    WINS   (n={len(win_v)}): mean={win_v.mean():.4f}  "
              f"std={win_v.std():.4f}")
    if len(loss_v):
        print(f"    LOSSES (n={len(loss_v)}): mean={loss_v.mean():.4f}  "
              f"std={loss_v.std():.4f}")

    if len(win_v) and len(loss_v):
        sep = ((win_v.mean() - loss_v.mean()) /
               np.sqrt((win_v.std()**2 + loss_v.std()**2) / 2 + 1e-8))
        print(f"    Cohen's d: {sep:.3f}")

    # Threshold sweep
    print(f"\n  {'Threshold':>10} {'Kept':>6} {'Wins':>6} {'Loss':>6} {'WR':>7} "
          f"{'Net PnL':>12} {'Sacr.Win$':>12} {'Avoid.Loss$':>12}")

    for p in [5, 10, 15, 20, 25, 30, 40, 50]:
        thresh = np.percentile(all_v, p)
        kept_mask = all_v >= thresh
        kept = pnls[kept_mask]
        filtered = pnls[~kept_mask]
        is_win = pnls > 0

        k_w = (kept_mask & is_win).sum()
        k_l = (kept_mask & ~is_win).sum()
        wr = k_w / max(1, k_w + k_l) * 100
        net = kept.sum()
        sac = pnls[~kept_mask & is_win].sum()
        avd = pnls[~kept_mask & ~is_win].sum()

        print(f"  P{p:02d}={thresh:>7.3f} {kept_mask.sum():>6} {k_w:>6} "
              f"{k_l:>6} {wr:>6.1f}% {net:>11,.0f} {sac:>11,.0f} {avd:>11,.0f}")

    # Worst/best trades
    for label_sec, subset, reverse in [("Worst 10 losses", losses, False),
                                        ("Best 10 wins", wins, True)]:
        sorted_sub = sorted(subset, key=lambda t: t['pnl'], reverse=reverse)[:10]
        print(f"\n  {label_sec}:")
        for t in sorted_sub:
            pctl = (all_v < t['entry_critic_value']).mean() * 100
            print(f"    PnL=${t['pnl']:>+9,.0f}  V={t['entry_critic_value']:>8.4f} "
                  f"(P{pctl:.0f})  {t['direction']}  hour={t['entry_hour']}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature veto sweep
# ─────────────────────────────────────────────────────────────────────────────

def analyze_feature_vetoes(label: str, trades: list[dict]) -> list[dict]:
    """Sweep single-feature threshold vetoes, find best rules + combinations."""
    features_matrix = np.array([t['entry_features'] for t in trades])
    pnls = np.array([t['pnl'] for t in trades])
    is_win = pnls > 0
    total_pnl = pnls.sum()

    print(f"\n{'='*80}")
    print(f"  FEATURE-BASED VETOES — {label}")
    print(f"  Baseline: {len(trades)} trades, ${total_pnl:+,.0f}")
    print(f"{'='*80}")

    results = []
    for fi, fname in enumerate(FEATURE_NAMES):
        fvals = features_matrix[:, fi]

        for direction, dir_label in [('high', 'VETO if >'), ('low', 'VETO if <')]:
            best_net = 0
            best_info = None

            for pct in [70, 75, 80, 85, 90, 95]:
                if direction == 'high':
                    thresh = np.percentile(fvals, pct)
                    veto_mask = fvals > thresh
                else:
                    thresh = np.percentile(fvals, 100 - pct)
                    veto_mask = fvals < thresh

                if veto_mask.sum() < 3:
                    continue

                vetoed_pnl = pnls[veto_mask]
                sac = vetoed_pnl[vetoed_pnl > 0].sum()
                avd = vetoed_pnl[vetoed_pnl <= 0].sum()
                net_improvement = -avd - sac

                if net_improvement > best_net:
                    best_net = net_improvement
                    best_info = {
                        'feature': fname,
                        'fi': fi,
                        'direction': dir_label,
                        'op': '>' if direction == 'high' else '<',
                        'percentile': pct,
                        'threshold': thresh,
                        'n_vetoed': int(veto_mask.sum()),
                        'n_vetoed_wins': int((veto_mask & is_win).sum()),
                        'n_vetoed_losses': int((veto_mask & ~is_win).sum()),
                        'sacrificed_wins': float(sac),
                        'avoided_losses': float(avd),
                        'net_improvement': float(net_improvement),
                        'kept_pnl': float(pnls[~veto_mask].sum()),
                    }

            if best_info:
                results.append(best_info)

    results.sort(key=lambda x: -x['net_improvement'])

    # Top 20 single rules
    print(f"\n  Top 20 single-feature veto rules:")
    print(f"  {'Feature':<24} {'Rule':<12} {'Pctl':>5} {'Thresh':>8} "
          f"{'#Veto':>6} {'#VW':>5} {'#VL':>5} "
          f"{'Sacr$':>10} {'Avoid$':>10} {'Net+$':>10} {'Kept$':>12}")
    for r in results[:20]:
        print(f"  {r['feature']:<24} {r['direction']:<12} "
              f"P{r['percentile']:>3} {r['threshold']:>8.3f} "
              f"{r['n_vetoed']:>6} {r['n_vetoed_wins']:>5} "
              f"{r['n_vetoed_losses']:>5} "
              f"{r['sacrificed_wins']:>9,.0f} {r['avoided_losses']:>9,.0f} "
              f"{r['net_improvement']:>9,.0f} {r['kept_pnl']:>11,.0f}")

    # Pairwise veto combinations
    if len(results) >= 2:
        print(f"\n  Pairwise veto combinations (top 5 candidates):")
        top = results[:6]
        combos = []

        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                r1, r2 = top[i], top[j]
                m1 = (features_matrix[:, r1['fi']] > r1['threshold']
                       if r1['op'] == '>' else
                       features_matrix[:, r1['fi']] < r1['threshold'])
                m2 = (features_matrix[:, r2['fi']] > r2['threshold']
                       if r2['op'] == '>' else
                       features_matrix[:, r2['fi']] < r2['threshold'])
                combo = m1 | m2
                kept_pnl = pnls[~combo].sum()
                nv = combo.sum()
                sac = pnls[combo & is_win].sum()
                avd = pnls[combo & ~is_win].sum()
                net_imp = -avd - sac
                wr = is_win[~combo].mean() * 100 if (~combo).any() else 0
                combos.append((r1, r2, nv, sac, avd, net_imp, kept_pnl, wr))

        combos.sort(key=lambda x: -x[5])
        for r1, r2, nv, sac, avd, ni, kp, wr in combos[:5]:
            print(f"    {r1['feature']} {r1['op']} {r1['threshold']:.3f} "
                  f"+ {r2['feature']} {r2['op']} {r2['threshold']:.3f}")
            print(f"      vetoed={nv}, sacr=${sac:,.0f}, avoid=${avd:,.0f}, "
                  f"net+=${ni:,.0f}, kept=${kp:,.0f}, WR={wr:.1f}%")

    # Print as veto rule string for run_replay.py
    if results:
        best = results[0]
        print(f"\n  Suggested --veto flag for run_replay.py:")
        print(f'    --veto "{best["feature"]} {best["op"]} {best["threshold"]:.3f}"')
        if len(results) >= 2:
            r2 = results[1]
            print(f'    --veto "{best["feature"]} {best["op"]} {best["threshold"]:.3f}, '
                  f'{r2["feature"]} {r2["op"]} {r2["threshold"]:.3f}"')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Combined critic + feature
# ─────────────────────────────────────────────────────────────────────────────

def analyze_combined(label: str, trades: list[dict], feature_results: list[dict]):
    """Test critic gate + best feature veto together."""
    if not trades or 'entry_critic_value' not in trades[0]:
        return
    if not feature_results:
        return

    all_v = np.array([t['entry_critic_value'] for t in trades])
    pnls = np.array([t['pnl'] for t in trades])
    is_win = pnls > 0
    features_matrix = np.array([t['entry_features'] for t in trades])
    total_pnl = pnls.sum()

    best = feature_results[0]
    feat_mask = (features_matrix[:, best['fi']] > best['threshold']
                 if best['op'] == '>' else
                 features_matrix[:, best['fi']] < best['threshold'])

    print(f"\n{'='*80}")
    print(f"  COMBINED: CRITIC + FEATURE — {label}")
    print(f"  Feature rule: {best['feature']} {best['op']} {best['threshold']:.3f}")
    print(f"{'='*80}")

    print(f"  {'Critic':>12} {'#Veto':>6} {'Kept':>6} {'WR':>7} "
          f"{'Kept PnL':>12} {'Sacr$':>10} {'Avoid$':>10}")

    for p in [10, 20, 25, 30]:
        v_thresh = np.percentile(all_v, p)
        critic_mask = all_v < v_thresh
        combo = feat_mask | critic_mask
        nv = combo.sum()
        kept_pnl = pnls[~combo].sum()
        sac = pnls[combo & is_win].sum()
        avd = pnls[combo & ~is_win].sum()
        wr = is_win[~combo].mean() * 100 if (~combo).any() else 0

        print(f"  P{p:02d}={v_thresh:>7.3f} {nv:>6} {len(trades)-nv:>6} "
              f"{wr:>6.1f}% {kept_pnl:>11,.0f} {sac:>9,.0f} {avd:>9,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Veto rule discovery for Wyckoff RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--checkpoints', '-c', nargs='+', required=True,
                   help='Actor checkpoint .pt file paths')
    p.add_argument('--critics', nargs='+', default=None,
                   help='Critic .pth file paths (same order as checkpoints)')
    p.add_argument('--labels', '-l', nargs='+', default=None,
                   help='Labels for each checkpoint')
    p.add_argument('--continuous', action='store_true',
                   help='Use continuous position sizing')
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

    if args.critics:
        if len(args.critics) != len(args.checkpoints):
            print("ERROR: --critics count must match --checkpoints count",
                  file=sys.stderr)
            sys.exit(1)
        for path in args.critics:
            if not os.path.exists(path):
                print(f"ERROR: Critic not found: {path}", file=sys.stderr)
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
    print(f"VETO RULE DISCOVERY", flush=True)
    print(f"Period: {args.start_date} → {args.end_date}", flush=True)
    print(f"Checkpoints: {len(args.checkpoints)}", flush=True)
    print(f"Critics: {'yes' if args.critics else 'no'}", flush=True)
    print(f"{'='*60}", flush=True)

    # Precompute bars + features ONCE
    from wyckoff_rl.live.precompute import PrecomputedReplay
    print(f"\n  Precomputing bars + features...", flush=True)
    replay = PrecomputedReplay.from_scid(
        args.scid, args.start_date, args.end_date,
    )
    print(f"  → {replay.n_bars} bars, {replay.n_windows} windows\n", flush=True)

    for i, (label, ckpt_path) in enumerate(zip(labels, args.checkpoints)):
        critic_path = args.critics[i] if args.critics else None
        extra = " + critic" if critic_path else ""
        print(f"\n  Capturing: {label}{extra}...", flush=True)

        cap = replay.run_capture(
            ckpt_path,
            continuous=args.continuous,
            initial_capital=args.capital,
            critic_path=critic_path,
        )
        cap['label'] = label

        trades = build_trades(cap)
        n_w = sum(1 for t in trades if t['pnl'] > 0)
        n_l = len(trades) - n_w
        net = sum(t['pnl'] for t in trades)
        print(f"  → {len(trades)} round trips (W:{n_w} L:{n_l}), "
              f"${net:+,.0f}", flush=True)

        # Critic gate
        if cap['has_critic']:
            analyze_critic_gate(label, trades)

        # Feature vetoes
        feature_results = analyze_feature_vetoes(label, trades)

        # Combined
        if cap['has_critic'] and feature_results:
            analyze_combined(label, trades, feature_results)


if __name__ == '__main__':
    main()
