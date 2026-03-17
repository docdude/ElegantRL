#!/usr/bin/env python3
"""
Evaluate ALL checkpoints in each Wyckoff CPCV split directory.

Compares OOS performance against NQ buy-and-hold baseline.
Best avgR during training often != best OOS Sharpe/return.

Usage -- single split:
    python -m wyckoff_rl.eval_all_checkpoints \
        --results-dir wyckoff_effort/rl_results/20260317_... --split 0

Usage -- all splits:
    python -m wyckoff_rl.eval_all_checkpoints \
        --results-dir wyckoff_effort/rl_results/20260317_...

Options:
    --gpu 0         GPU device (-1 for CPU)
    --force         Re-evaluate even if checkpoint_results.csv exists
    --top-k 5       Highlight top-k checkpoints in plots
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wyckoff_rl.config import (
    GPU_ID, DEFAULT_ERL_PARAMS, DEFAULT_ENV_PARAMS, WYCKOFF_NPZ_PATH,
)
from wyckoff_rl.function_train_test import load_wyckoff_data

import torch as th


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_checkpoints(split_dir: str) -> list[dict]:
    """Find all actor__*.pt and act.pth checkpoints."""
    pattern = re.compile(r'^actor__(\d+)(?:_(-?\d+(?:\.\d+)?))?\.pt$')
    checkpoints = []
    for f in os.listdir(split_dir):
        m = pattern.match(f)
        if m:
            step = int(m.group(1))
            avgR = float(m.group(2)) if m.group(2) else None
            checkpoints.append({
                'path': os.path.join(split_dir, f),
                'filename': f,
                'step': step,
                'avgR': avgR,
            })
    act_path = os.path.join(split_dir, 'act.pth')
    if os.path.exists(act_path):
        max_step = max((c['step'] for c in checkpoints), default=0)
        checkpoints.append({
            'path': act_path,
            'filename': 'act.pth',
            'step': max_step + 1,
            'avgR': None,
        })
    checkpoints.sort(key=lambda c: c['step'])
    return checkpoints


# ═══════════════════════════════════════════════════════════════════════════
# NQ buy-and-hold baseline
# ═══════════════════════════════════════════════════════════════════════════

def compute_nq_baseline(close_ary: np.ndarray, test_indices: np.ndarray,
                        initial_amount: float) -> dict:
    """Compute NQ buy-and-hold baseline over test indices.

    Returns dict with 'values' (account curve) and 'return_pct'.
    """
    prices = close_ary[test_indices].ravel()
    # Normalized to start at initial_amount
    bh_values = prices / prices[0] * initial_amount
    bh_return = (bh_values[-1] / initial_amount - 1) * 100
    return {'values': bh_values, 'return_pct': bh_return}


# ═══════════════════════════════════════════════════════════════════════════
# Single-checkpoint evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single_checkpoint(
    checkpoint_path: str,
    test_npz: str,
    test_len: int,
    env_params: dict,
    initial_amount: float,
    device: str,
    baseline_values: np.ndarray = None,
    baseline_return: float = None,
) -> dict | None:
    """Load actor from checkpoint, run deterministic eval on test env."""
    from elegantrl.envs.WyckoffTradingEnv import WyckoffTradingEnv

    try:
        actor = th.load(checkpoint_path, map_location=device, weights_only=False)
        actor.eval()
    except Exception as e:
        print(f"    WARN: Failed to load {os.path.basename(checkpoint_path)}: {e}")
        return None

    # Build test env
    env = WyckoffTradingEnv(
        npz_path=test_npz,
        initial_amount=initial_amount,
        cost_per_trade=env_params.get('cost_per_trade', 0.5),
        reward_mode=env_params.get('reward_mode', 'pnl'),
        reward_scale=env_params.get('reward_scale', 1.0),
        max_position=env_params.get('max_position', 1),
        beg_idx=0,
        end_idx=test_len,
    )

    # Deterministic rollout
    state, _ = env.reset()
    account_values = [initial_amount]
    done = False
    total_reward = 0.0

    while not done:
        state_t = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)
        with th.no_grad():
            action = actor(state_t).squeeze(0).cpu().numpy()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        account_values.append(env.total_asset)

    account_values = np.array(account_values)

    # Metrics
    daily_returns = np.diff(account_values) / (account_values[:-1] + 1e-12)
    final_return = (account_values[-1] / initial_amount - 1)
    max_dd = _max_drawdown(account_values)

    mean_r = daily_returns.mean()
    std_r = daily_returns.std()
    sharpe = mean_r / std_r * np.sqrt(252 * 6.5 * 60) if std_r > 1e-12 else 0.0
    # Approximate annualization: range bars ≈ 1 per ~2min, so ~195 bars/hour
    # 252 days × 6.5 hours × 195 bars ≈ 320K bars/year
    ann_factor = np.sqrt(min(len(daily_returns), 320_000))
    sharpe_simple = mean_r / std_r * ann_factor if std_r > 1e-12 else 0.0

    neg_r = daily_returns[daily_returns < 0]
    downside_std = np.sqrt(np.mean(neg_r ** 2)) if len(neg_r) > 0 else 1e-12
    sortino = mean_r / downside_std * ann_factor

    calmar = final_return / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    # Baseline comparison
    alpha = 0.0
    days_beating = 0
    pct_beating = 0.0
    if baseline_values is not None and baseline_return is not None:
        alpha = final_return * 100 - baseline_return
        strategy_pct = (account_values / account_values[0] - 1) * 100
        baseline_pct = (baseline_values / baseline_values[0] - 1) * 100
        min_len = min(len(strategy_pct), len(baseline_pct))
        beating_mask = strategy_pct[:min_len] > baseline_pct[:min_len]
        days_beating = int(beating_mask.sum())
        pct_beating = days_beating / min_len * 100

    return {
        'account_values': account_values,
        'final_return': final_return,
        'sharpe': sharpe_simple,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': max_dd,
        'ann_return': final_return,  # approximation for range bars
        'total_reward': total_reward,
        'n_trades': getattr(env, 'total_trades', 0),
        'alpha': alpha,
        'days_beating': days_beating,
        'pct_beating': pct_beating,
    }


def _max_drawdown(values: np.ndarray) -> float:
    """Compute max drawdown as negative fraction."""
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / (peak + 1e-12)
    return float(dd.min())


# ═══════════════════════════════════════════════════════════════════════════
# Per-split evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_split(
    split_dir: str,
    close_ary: np.ndarray,
    tech_ary: np.ndarray,
    gpu_id: int = 0,
    force: bool = False,
    top_k: int = 5,
) -> pd.DataFrame | None:
    """Evaluate all checkpoints in one split directory."""
    csv_path = os.path.join(split_dir, 'checkpoint_results.csv')
    if os.path.exists(csv_path) and not force:
        print(f"  {os.path.basename(split_dir)}: results exist, skipping (--force)")
        return pd.read_csv(csv_path)

    split_name = os.path.basename(split_dir)
    meta_path = os.path.join(split_dir, 'split_meta.json')
    if not os.path.exists(meta_path):
        print(f"  {split_name}: no split_meta.json, skipping")
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    test_npz = meta.get('test_npz', '')
    n_test = meta.get('n_test', 0)
    env_params = meta.get('env_params', DEFAULT_ENV_PARAMS)
    initial_amount = env_params.get('initial_amount', 1000.0)

    if not os.path.exists(test_npz):
        print(f"  {split_name}: test NPZ not found at {test_npz}, skipping")
        return None

    # Load test data for baseline
    test_data = np.load(test_npz)
    test_close = test_data['close_ary']

    # Buy-and-hold baseline
    baseline = compute_nq_baseline(test_close, np.arange(n_test), initial_amount)

    # Discover checkpoints
    checkpoints = discover_checkpoints(split_dir)
    if not checkpoints:
        print(f"  {split_name}: no checkpoints found, skipping")
        return None

    device = f'cuda:{gpu_id}' if gpu_id >= 0 and th.cuda.is_available() else 'cpu'

    print(f"\n  {split_name}: {len(checkpoints)} checkpoints, "
          f"test={n_test:,} bars, B&H={baseline['return_pct']:+.2f}%")

    # Evaluate each checkpoint
    rows = []
    all_curves = {}
    for ckpt in checkpoints:
        result = evaluate_single_checkpoint(
            ckpt['path'], test_npz, n_test, env_params,
            initial_amount, device,
            baseline_values=baseline['values'],
            baseline_return=baseline['return_pct'],
        )
        if result is None:
            continue

        row = {
            'checkpoint': ckpt['filename'],
            'step': ckpt['step'],
            'train_avgR': ckpt['avgR'],
            'final_return': result['final_return'] * 100,
            'alpha': result['alpha'],
            'sharpe': result['sharpe'],
            'sortino': result['sortino'],
            'calmar': result['calmar'],
            'max_drawdown': result['max_drawdown'] * 100,
            'total_reward': result['total_reward'],
            'n_trades': result['n_trades'],
            'pct_beating': result['pct_beating'],
        }
        rows.append(row)
        all_curves[ckpt['filename']] = result['account_values']

        avgR_str = f"avgR={ckpt['avgR']:.1f}" if ckpt['avgR'] else "final"
        print(f"    {ckpt['filename']:<45} "
              f"Ret={result['final_return']*100:>+7.2f}%  "
              f"Alpha={result['alpha']:>+6.2f}  "
              f"Sharpe={result['sharpe']:>7.2f}  "
              f"MaxDD={result['max_drawdown']*100:>6.1f}%  "
              f"Trades={result['n_trades']}  ({avgR_str})")

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(['sharpe', 'final_return'], ascending=[False, False])
    df.to_csv(csv_path, index=False)
    print(f"\n    Saved: {csv_path}")

    # Best checkpoint summary
    best = df.iloc[0]
    print(f"\n    BEST checkpoint: {best['checkpoint']}")
    print(f"      Return={best['final_return']:+.2f}%, "
          f"Alpha={best['alpha']:+.2f}, "
          f"Sharpe={best['sharpe']:.2f}, "
          f"MaxDD={best['max_drawdown']:.1f}%, "
          f"Trades={best['n_trades']}")

    if best['checkpoint'] != 'act.pth':
        final_row = df[df['checkpoint'] == 'act.pth']
        if not final_row.empty:
            final = final_row.iloc[0]
            print(f"    FINAL model:     act.pth")
            print(f"      Return={final['final_return']:+.2f}%, "
                  f"Sharpe={final['sharpe']:.2f} "
                  f"← {'worse' if final['sharpe'] < best['sharpe'] else 'same/better'} "
                  f"than best checkpoint")

    # Plot
    _plot_split_comparison(
        df, all_curves, split_dir, initial_amount, top_k,
        baseline_values=baseline['values'],
        baseline_return=baseline['return_pct'],
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def _plot_split_comparison(
    df: pd.DataFrame,
    all_curves: dict,
    split_dir: str,
    initial_amount: float,
    top_k: int,
    baseline_values: np.ndarray = None,
    baseline_return: float = None,
):
    """4-panel comparison plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    split_name = os.path.basename(split_dir)
    has_baseline = baseline_values is not None
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: All equity curves + B&H baseline
    ax = axes[0, 0]
    for name, curve in all_curves.items():
        ax.plot(curve, alpha=0.15, linewidth=0.5, color='steelblue')
    if has_baseline:
        ax.plot(baseline_values, color='black', linestyle='--',
                linewidth=2, label=f'NQ B&H ({baseline_return:+.2f}%)')
    # Highlight top-k by Sharpe
    top_names = df.head(top_k)['checkpoint'].tolist()
    colors = plt.cm.Set1(np.linspace(0, 1, min(top_k, 9)))
    for name, color in zip(top_names, colors):
        if name in all_curves:
            row = df[df['checkpoint'] == name].iloc[0]
            ax.plot(all_curves[name], color=color, linewidth=1.5, alpha=0.9,
                    label=f"{name} (Sharpe={row['sharpe']:.2f})")
    ax.axhline(initial_amount, color='grey', linestyle=':', alpha=0.3)
    ax.set_title(f'{split_name}: All Checkpoints ({len(all_curves)} total)')
    ax.set_xlabel('Bar')
    ax.set_ylabel('Portfolio Value (NQ points)')
    ax.legend(fontsize=6, loc='upper left')

    # Panel 2: Return vs Step
    ax = axes[0, 1]
    ax.scatter(df['step'], df['final_return'], c=df['sharpe'], cmap='RdYlGn',
               s=30, alpha=0.7, edgecolors='grey', linewidths=0.5)
    for name in top_names:
        row = df[df['checkpoint'] == name].iloc[0]
        ax.annotate(f"S={row['sharpe']:.1f}", (row['step'], row['final_return']),
                    fontsize=5, alpha=0.8)
    if has_baseline:
        ax.axhline(baseline_return, color='black', linestyle='--', linewidth=1,
                    label=f'NQ B&H ({baseline_return:+.2f}%)')
    ax.set_title('OOS Return vs Training Step')
    ax.set_xlabel('Step')
    ax.set_ylabel('Return (%)')
    ax.legend(fontsize=6)

    # Panel 3: Sharpe vs MaxDD scatter
    ax = axes[1, 0]
    sc = ax.scatter(df['max_drawdown'], df['sharpe'], c=df['final_return'],
                    cmap='RdYlGn', s=30, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Return (%)', shrink=0.8)
    ax.axhline(0, linewidth=0.5, color='grey', linestyle=':')
    ax.set_title('Sharpe vs Max Drawdown')
    ax.set_xlabel('Max Drawdown (%)')
    ax.set_ylabel('Sharpe Ratio')

    # Panel 4: Train avgR vs OOS Sharpe (best != highest training reward)
    ax = axes[1, 1]
    has_avgR = df['train_avgR'].notna()
    if has_avgR.sum() > 1:
        ax.scatter(df.loc[has_avgR, 'train_avgR'], df.loc[has_avgR, 'sharpe'],
                   s=30, alpha=0.7, color='steelblue')
        corr = df.loc[has_avgR, ['train_avgR', 'sharpe']].corr().iloc[0, 1]
        ax.set_title(f'Train avgR vs OOS Sharpe (corr={corr:.2f})')
        ax.set_xlabel('Training avgR')
        ax.set_ylabel('OOS Sharpe')
    else:
        ax.text(0.5, 0.5, 'Insufficient avgR data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Train avgR vs OOS Sharpe')

    plt.suptitle(f'Wyckoff NQ — {split_name} Checkpoint Analysis', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(split_dir, 'checkpoint_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot: {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate summary across all splits
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(results_dir: str, split_dfs: dict[str, pd.DataFrame]):
    """Compute aggregate statistics across all evaluated splits."""
    if not split_dfs:
        return

    print(f"\n{'='*80}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*80}")

    all_best = []
    all_final = []
    for split_name, df in sorted(split_dfs.items()):
        best = df.iloc[0]  # already sorted by Sharpe
        all_best.append({
            'split': split_name,
            'checkpoint': best['checkpoint'],
            'return': best['final_return'],
            'alpha': best['alpha'],
            'sharpe': best['sharpe'],
            'max_dd': best['max_drawdown'],
            'n_trades': best.get('n_trades', 0),
        })
        final_row = df[df['checkpoint'] == 'act.pth']
        if not final_row.empty:
            final = final_row.iloc[0]
            all_final.append({
                'split': split_name,
                'return': final['final_return'],
                'sharpe': final['sharpe'],
            })

    best_df = pd.DataFrame(all_best)
    print(f"\nBest checkpoint per split:")
    print(f"  {'Split':<12} {'Checkpoint':<35} {'Return':>8} {'Alpha':>8} "
          f"{'Sharpe':>7} {'MaxDD':>7} {'Trades':>7}")
    print(f"  {'-'*90}")
    for _, r in best_df.iterrows():
        print(f"  {r['split']:<12} {r['checkpoint']:<35} "
              f"{r['return']:>+7.2f}% {r['alpha']:>+7.2f} "
              f"{r['sharpe']:>7.2f} {r['max_dd']:>6.1f}% "
              f"{r['n_trades']:>7.0f}")

    print(f"\n  Mean best Sharpe: {best_df['sharpe'].mean():.3f} "
          f"± {best_df['sharpe'].std():.3f}")
    print(f"  Mean best return: {best_df['return'].mean():+.2f}% "
          f"± {best_df['return'].std():.2f}%")
    print(f"  Mean best alpha:  {best_df['alpha'].mean():+.2f}")

    if all_final:
        final_df = pd.DataFrame(all_final)
        n_best_not_final = sum(1 for b in all_best if b['checkpoint'] != 'act.pth')
        print(f"\n  Best ≠ final model in {n_best_not_final}/{len(all_best)} splits "
              f"({n_best_not_final/len(all_best)*100:.0f}%)")
        print(f"  Mean final Sharpe: {final_df['sharpe'].mean():.3f} "
              f"vs best {best_df['sharpe'].mean():.3f}")

    # Save aggregate
    agg_path = os.path.join(results_dir, 'aggregate_checkpoint_results.csv')
    best_df.to_csv(agg_path, index=False)
    print(f"\n  Saved: {agg_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Wyckoff RL checkpoints vs NQ buy-and-hold"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Run results directory (contains split_* dirs)")
    parser.add_argument("--split", type=int, default=None,
                        help="Evaluate single split (default: all)")
    parser.add_argument("--gpu", type=int, default=GPU_ID)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--npz", default=WYCKOFF_NPZ_PATH,
                        help="Path to Wyckoff NPZ data")
    return parser.parse_args()


def main():
    args = parse_args()

    close_ary, tech_ary = load_wyckoff_data(args.npz)
    print(f"Data: {close_ary.shape[0]:,} bars, {tech_ary.shape[1]} features")

    # Find split directories
    if args.split is not None:
        split_dirs = [os.path.join(args.results_dir, f'split_{args.split}')]
    else:
        split_dirs = sorted([
            os.path.join(args.results_dir, d)
            for d in os.listdir(args.results_dir)
            if d.startswith('split_') and os.path.isdir(
                os.path.join(args.results_dir, d))
        ])

    if not split_dirs:
        print(f"No split directories found in {args.results_dir}")
        sys.exit(1)

    print(f"Evaluating {len(split_dirs)} split(s) in {args.results_dir}")

    split_dfs = {}
    for split_dir in split_dirs:
        if not os.path.isdir(split_dir):
            print(f"  {split_dir} not found, skipping")
            continue
        df = evaluate_split(
            split_dir, close_ary, tech_ary,
            gpu_id=args.gpu, force=args.force, top_k=args.top_k,
        )
        if df is not None:
            split_dfs[os.path.basename(split_dir)] = df

    if len(split_dfs) > 1:
        aggregate_results(args.results_dir, split_dfs)

    print(f"\nDone. Evaluated {len(split_dfs)} splits.")


if __name__ == '__main__':
    main()
