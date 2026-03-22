#!/usr/bin/env python3
"""
Filter Candidate Analysis: Can critic V(s) or feature thresholds
selectively kill losing trades without destroying winning trades?

Two approaches tested:
  1. CRITIC VALUE GATE: Load cri.pth, run V(state) at each entry.
     If V(entry) < threshold → veto the trade.
  2. FEATURE-BASED VETOES: For each of 36 features, sweep thresholds
     and find rules like "don't trade when wave_progress > X" that
     remove more loss $ than win $.

Outputs: threshold sweep tables showing PnL impact at each cutoff.
"""
import logging, sys, os
import numpy as np
import pandas as pd
import torch
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

CKPT_DIR = '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous'
WAVE_DIR = '/opt/ElegantRL/wyckoff_rl/live/checkpoints/wavenet_ppo'
MODELS = {
    'split3': {
        'actor': f'{CKPT_DIR}/split3_actor__000001488896_00491.096.pt',
        'critic': f'{CKPT_DIR}/critics/split3_cri.pth',
    },
    'split4': {
        'actor': f'{CKPT_DIR}/split4_actor__000000874496_00444.927.pt',
        'critic': f'{CKPT_DIR}/critics/split4_cri.pth',
    },
    'wavenet_s0': {
        'actor': f'{WAVE_DIR}/actor__000001949696_00453.865.pt',
        'critic': f'{WAVE_DIR}/critics/split0_cri.pth',
    },
}
START_DATE = '2026-01-15'
END_DATE = '2026-03-18'


def run_replay_with_critic(label, actor_path, critic_path):
    """Replay with monkey-patch to capture features + critic values at every bar."""
    
    # Load critic
    critic = torch.load(critic_path, map_location='cpu', weights_only=False)
    critic.eval()
    
    capture = {
        'windows': [],
        'actions': [],
        'prices': [],
        'timestamps': [],
        'positions': [],
        'critic_values': [],
        'trade_events': [],
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
        window_copy = self.inference._window.copy()
        capture['windows'].append(window_copy)
        capture['prices'].append(bar.close)
        capture['timestamps'].append(bar.timestamp)
        capture['positions'].append(self._position)

        # Build state for actor
        target_pos, raw_action = self.inference.get_action(
            position=self._position,
            unrealized_pnl=self._unrealized_pnl,
            cash=self._cash,
        )
        capture['actions'].append(raw_action)

        # Build state for critic (same state as actor)
        agent_state = np.array([
            self._position,
            np.tanh(self._unrealized_pnl / 1000.0),
            np.tanh(self._cash / 1000.0),
        ], dtype=np.float32)
        window_flat = window_copy.flatten()
        state = np.concatenate([agent_state, window_flat])
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            v = critic(state_t).squeeze().item()
        capture['critic_values'].append(v)

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
    print(f"\nRunning {label} replay with critic value capture...", flush=True)
    data = SCIDReplayAdapter(scid_path=SCID_PATH, start_date=START_DATE, end_date=END_DATE, speed=0.0)
    sim = SimExecutor(initial_capital=250_000.0)
    trader = WyckoffTrader(
        checkpoint_path=actor_path, data_adapter=data, order_adapter=sim,
        range_size=40.0, continuous_sizing=True, max_contracts=1,
        log_dir=f'/tmp/filter_analysis_{label}',
    )
    trader.start()
    WyckoffTrader._on_bar_complete = original_on_bar
    print(f"  {len(capture['windows'])} bars, {len(capture['trade_events'])} events", flush=True)
    return capture


def build_trades(capture):
    """Build round-trip trades with entry features + critic values."""
    events = capture['trade_events']
    all_windows = np.array(capture['windows'])
    critic_vals = np.array(capture['critic_values'])
    trades = []
    entry_event = None

    for i, (bidx, act, delta, price, pnl, pos_before) in enumerate(events):
        if entry_event is None:
            entry_event = (bidx, act, delta, price, pnl, pos_before)
        elif pnl != 0:
            e_bidx, e_act, e_delta, e_price, _, e_pos = entry_event
            direction = 'LONG' if e_act == 'BUY' else 'SHORT'
            
            entry_ts = capture['timestamps'][e_bidx]
            entry_dt = datetime.fromtimestamp(entry_ts, tz=timezone.utc)

            # Entry features: last bar of the 30-bar window (most recent)
            entry_window = all_windows[e_bidx]  # (30, 36)
            entry_features = entry_window[-1]    # (36,) — latest bar features at entry

            trades.append({
                'direction': direction,
                'entry_bar': e_bidx,
                'exit_bar': bidx,
                'entry_price': e_price,
                'exit_price': price,
                'pnl': pnl,
                'entry_hour': entry_dt.hour,
                'entry_time': entry_dt,
                'entry_critic_value': critic_vals[e_bidx],
                'entry_raw_action': capture['actions'][e_bidx],
                'entry_features': entry_features,
                'hold_bars': bidx - e_bidx,
            })

            new_pos = pos_before + delta
            if new_pos != 0:
                entry_event = (bidx, act, delta, price, 0, new_pos)
            else:
                entry_event = None
        else:
            entry_event = (bidx, act, delta, price, pnl, pos_before)

    return trades


def analyze_critic_gate(label, trades):
    """Test critic V(s) as a trade gate."""
    print(f"\n{'='*80}")
    print(f"  CRITIC VALUE GATE — {label}")
    print(f"{'='*80}")

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    win_v = np.array([t['entry_critic_value'] for t in wins])
    loss_v = np.array([t['entry_critic_value'] for t in losses])
    all_v = np.array([t['entry_critic_value'] for t in trades])

    print(f"\nCritic V(s) at entry:")
    print(f"  ALL   (n={len(all_v)}): mean={all_v.mean():.4f}  std={all_v.std():.4f}  min={all_v.min():.4f}  max={all_v.max():.4f}")
    print(f"  WINS  (n={len(win_v)}): mean={win_v.mean():.4f}  std={win_v.std():.4f}  min={win_v.min():.4f}  max={win_v.max():.4f}")
    print(f"  LOSSES(n={len(loss_v)}): mean={loss_v.mean():.4f}  std={loss_v.std():.4f}  min={loss_v.min():.4f}  max={loss_v.max():.4f}")
    
    # Separation test
    separation = (win_v.mean() - loss_v.mean()) / np.sqrt((win_v.std()**2 + loss_v.std()**2) / 2)
    print(f"  Cohen's d (separation): {separation:.3f}")

    # Threshold sweep
    print(f"\n  Threshold sweep (veto entries with V(s) < threshold):")
    print(f"  {'Threshold':>10} {'Kept':>6} {'Wins':>6} {'Losses':>8} {'WR':>7} {'Net PnL':>12} {'Sacr.Wins':>12} {'Avoid.Loss':>12}")
    
    percentiles = [5, 10, 15, 20, 25, 30, 40, 50]
    thresholds = [np.percentile(all_v, p) for p in percentiles]
    
    for p, thresh in zip(percentiles, thresholds):
        kept = [t for t in trades if t['entry_critic_value'] >= thresh]
        filtered = [t for t in trades if t['entry_critic_value'] < thresh]
        
        k_wins = sum(1 for t in kept if t['pnl'] > 0)
        k_losses = sum(1 for t in kept if t['pnl'] <= 0)
        wr = k_wins / len(kept) * 100 if kept else 0
        net = sum(t['pnl'] for t in kept)
        
        sac = sum(t['pnl'] for t in filtered if t['pnl'] > 0)
        avoid = sum(t['pnl'] for t in filtered if t['pnl'] <= 0)
        
        print(f"  P{p:02d}={thresh:>7.3f} {len(kept):>6} {k_wins:>6} {k_losses:>8} {wr:>6.1f}% {net:>11,.0f} {sac:>11,.0f} {avoid:>11,.0f}")
    
    # Worst 10 losses — critic values
    print(f"\n  Worst 10 losses — critic V(s):")
    worst = sorted(losses, key=lambda t: t['pnl'])[:10]
    for t in worst:
        percentile = (all_v < t['entry_critic_value']).mean() * 100
        print(f"    PnL={t['pnl']:>9,.0f}  V={t['entry_critic_value']:>8.4f} (P{percentile:.0f})  {t['direction']}  hour={t['entry_hour']}")

    # Best 10 wins — critic values
    print(f"\n  Best 10 wins — critic V(s):")
    best = sorted(wins, key=lambda t: -t['pnl'])[:10]
    for t in best:
        percentile = (all_v < t['entry_critic_value']).mean() * 100
        print(f"    PnL={t['pnl']:>9,.0f}  V={t['entry_critic_value']:>8.4f} (P{percentile:.0f})  {t['direction']}  hour={t['entry_hour']}")


def analyze_feature_vetoes(label, trades):
    """Test single-feature threshold vetoes."""
    print(f"\n{'='*80}")
    print(f"  FEATURE-BASED VETOES — {label}")
    print(f"{'='*80}")

    n = len(trades)
    features_matrix = np.array([t['entry_features'] for t in trades])  # (n_trades, 36)
    pnls = np.array([t['pnl'] for t in trades])
    is_win = pnls > 0

    total_pnl = pnls.sum()
    
    # For each feature, find the best "veto if feature > X" and "veto if feature < X"
    results = []
    
    for fi, fname in enumerate(FEATURE_NAMES):
        fvals = features_matrix[:, fi]
        
        # Try percentile thresholds for "veto HIGH" (feature > threshold)
        for direction, label_dir in [('high', 'VETO if >'), ('low', 'VETO if <')]:
            best_net_improvement = 0
            best_info = None
            
            for pct in [70, 75, 80, 85, 90, 95]:
                if direction == 'high':
                    thresh = np.percentile(fvals, pct)
                    veto_mask = fvals > thresh
                else:
                    thresh = np.percentile(fvals, 100 - pct)
                    veto_mask = fvals < thresh
                
                vetoed_pnl = pnls[veto_mask]
                vetoed_wins = vetoed_pnl[vetoed_pnl > 0].sum()
                vetoed_losses = vetoed_pnl[vetoed_pnl <= 0].sum()
                net_improvement = -vetoed_losses - vetoed_wins  # positive = filter helps
                
                n_vetoed = veto_mask.sum()
                n_vetoed_wins = (veto_mask & is_win).sum()
                n_vetoed_losses = (veto_mask & ~is_win).sum()
                
                if net_improvement > best_net_improvement and n_vetoed >= 3:
                    best_net_improvement = net_improvement
                    best_info = {
                        'feature': fname,
                        'direction': label_dir,
                        'percentile': pct,
                        'threshold': thresh,
                        'n_vetoed': n_vetoed,
                        'n_vetoed_wins': n_vetoed_wins,
                        'n_vetoed_losses': n_vetoed_losses,
                        'sacrificed_wins': vetoed_wins,
                        'avoided_losses': vetoed_losses,
                        'net_improvement': net_improvement,
                        'new_pnl': total_pnl + net_improvement - vetoed_wins,  # wait, simpler:
                        'kept_pnl': pnls[~veto_mask].sum(),
                    }
            
            if best_info:
                results.append(best_info)
    
    # Sort by net improvement
    results.sort(key=lambda x: -x['net_improvement'])
    
    print(f"\n  Top 20 single-feature veto rules (by net $ improvement):")
    print(f"  {'Feature':<28} {'Rule':<12} {'Pctl':>5} {'Thresh':>8} {'#Veto':>6} {'#VWin':>6} {'#VLoss':>7} {'Sacr$':>10} {'Avoid$':>10} {'Net+$':>10} {'KeptPnL':>12}")
    
    for r in results[:20]:
        print(f"  {r['feature']:<28} {r['direction']:<12} P{r['percentile']:>3} {r['threshold']:>8.3f} "
              f"{r['n_vetoed']:>6} {r['n_vetoed_wins']:>6} {r['n_vetoed_losses']:>7} "
              f"{r['sacrificed_wins']:>9,.0f} {r['avoided_losses']:>9,.0f} "
              f"{r['net_improvement']:>9,.0f} {r['kept_pnl']:>11,.0f}")

    # Test COMBINATION of top 2-3 non-overlapping vetoes
    print(f"\n  Testing top veto combinations:")
    
    if len(results) >= 2:
        top = results[:5]  # top 5 individual rules
        
        # Try all pairs
        best_combo = None
        best_combo_net = 0
        
        for i in range(len(top)):
            for j in range(i+1, len(top)):
                r1, r2 = top[i], top[j]
                fi1 = FEATURE_NAMES.index(r1['feature'])
                fi2 = FEATURE_NAMES.index(r2['feature'])
                
                mask1 = features_matrix[:, fi1] > r1['threshold'] if '>' in r1['direction'] else features_matrix[:, fi1] < r1['threshold']
                mask2 = features_matrix[:, fi2] > r2['threshold'] if '>' in r2['direction'] else features_matrix[:, fi2] < r2['threshold']
                
                combo_mask = mask1 | mask2  # veto if EITHER rule fires
                
                kept_pnl = pnls[~combo_mask].sum()
                n_vetoed = combo_mask.sum()
                n_vw = (combo_mask & is_win).sum()
                n_vl = (combo_mask & ~is_win).sum()
                sac = pnls[combo_mask & is_win].sum()
                avd = pnls[combo_mask & ~is_win].sum()
                net_imp = -avd - sac
                
                wr_kept = is_win[~combo_mask].mean() * 100 if (~combo_mask).any() else 0
                
                if net_imp > best_combo_net:
                    best_combo_net = net_imp
                    best_combo = (r1, r2, n_vetoed, n_vw, n_vl, sac, avd, net_imp, kept_pnl, wr_kept)
        
        if best_combo:
            r1, r2, nv, nvw, nvl, sac, avd, nimp, kpnl, wr = best_combo
            print(f"  Best pair:")
            print(f"    Rule 1: {r1['feature']} {r1['direction']} P{r1['percentile']} (thresh={r1['threshold']:.3f})")
            print(f"    Rule 2: {r2['feature']} {r2['direction']} P{r2['percentile']} (thresh={r2['threshold']:.3f})")
            print(f"    Vetoed: {nv} trades ({nvw} wins, {nvl} losses)")
            print(f"    Sacrificed wins: ${sac:,.0f}, Avoided losses: ${avd:,.0f}")
            print(f"    Net improvement: ${nimp:,.0f}")
            print(f"    Kept PnL: ${kpnl:,.0f} (from ${total_pnl:,.0f}), WR={wr:.1f}%")

    # Also test: combined critic + best feature veto
    return results


def main():
    for model_name, paths in MODELS.items():
        capture = run_replay_with_critic(model_name, paths['actor'], paths['critic'])
        trades = build_trades(capture)
        
        print(f"\n  {model_name}: {len(trades)} round trips, "
              f"{sum(1 for t in trades if t['pnl']>0)} wins, "
              f"{sum(1 for t in trades if t['pnl']<=0)} losses, "
              f"net ${sum(t['pnl'] for t in trades):,.0f}")
        
        analyze_critic_gate(model_name, trades)
        feature_results = analyze_feature_vetoes(model_name, trades)
        
        # Combined: critic + best feature
        print(f"\n{'='*80}")
        print(f"  COMBINED: CRITIC + FEATURE VETO — {model_name}")
        print(f"{'='*80}")
        
        all_v = np.array([t['entry_critic_value'] for t in trades])
        pnls = np.array([t['pnl'] for t in trades])
        is_win = pnls > 0
        features_matrix = np.array([t['entry_features'] for t in trades])
        
        if feature_results:
            best_feat = feature_results[0]
            fi = FEATURE_NAMES.index(best_feat['feature'])
            feat_mask = features_matrix[:, fi] > best_feat['threshold'] if '>' in best_feat['direction'] else features_matrix[:, fi] < best_feat['threshold']
            
            for p in [10, 20, 25, 30]:
                v_thresh = np.percentile(all_v, p)
                critic_mask = all_v < v_thresh
                combo = feat_mask | critic_mask
                
                kept_pnl = pnls[~combo].sum()
                n_vetoed = combo.sum()
                sac = pnls[combo & is_win].sum()
                avd = pnls[combo & ~is_win].sum()
                wr = is_win[~combo].mean() * 100 if (~combo).any() else 0
                
                print(f"  Critic P{p:02d} + {best_feat['feature']} {best_feat['direction']}: "
                      f"veto {n_vetoed} trades, kept PnL=${kept_pnl:,.0f}, WR={wr:.1f}%, "
                      f"sacr=${sac:,.0f}, avoid=${avd:,.0f}")


if __name__ == '__main__':
    main()
