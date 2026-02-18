"""
Evaluate all checkpoints in a directory and compare their performance curves.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.demo_FinRL_Alpaca_VecEnv_CV import (
    AlpacaStockVecEnv, download_and_preprocess, df_to_arrays,
    TEST_START, TEST_END
)

def evaluate_checkpoint(actor_path, val_env, num_days, initial_amount=1e6, gpu_id=0):
    """Evaluate a single checkpoint and return daily account values."""
    device = f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu'
    
    # Load actor
    try:
        actor = th.load(actor_path, map_location=device, weights_only=False)
        actor.eval()
    except Exception as e:
        print(f"  ⚠️ Failed to load {actor_path}: {e}")
        return None
    
    # Reset env
    val_env.if_random_reset = False
    state, _ = val_env.reset()
    
    # Track account values
    account_values = [initial_amount]
    max_step = num_days - 1
    
    with th.no_grad():
        for t in range(max_step):
            action = actor(state.to(device) if hasattr(state, 'to') else state)
            state, reward, terminal, truncate, info = val_env.step(action)
            
            if hasattr(val_env, 'total_asset'):
                if t < max_step - 1:
                    account_value = val_env.total_asset[0].cpu().item()
                    account_values.append(account_value)
                else:
                    if hasattr(val_env, 'cumulative_returns') and val_env.cumulative_returns is not None:
                        cr = val_env.cumulative_returns
                        if hasattr(cr, 'cpu'):
                            final_value = initial_amount * cr[0].cpu().item() / 100
                        else:
                            final_value = initial_amount * float(cr[0]) / 100
                        account_values.append(final_value)
                    else:
                        account_values.append(account_values[-1])
    
    return np.array(account_values)


def main(checkpoint_dir, gpu_id=0, beg_idx=None, end_idx=None):
    """Main evaluation loop."""
    print(f"\n{'='*70}")
    print(f"CHECKPOINT COMPARISON: {checkpoint_dir}")
    print(f"{'='*70}")
    
    # Find all checkpoints
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('actor__') and f.endswith('.pt'):
            checkpoint_files.append(f)
        elif f == 'act.pth':
            checkpoint_files.append(f)
    
    checkpoint_files = sorted(checkpoint_files)
    print(f"\nFound {len(checkpoint_files)} checkpoints")
    
    # Load data and create env
    print("\nLoading data...")
    df = download_and_preprocess(force_download=False)
    close_ary, tech_ary = df_to_arrays(df)
    num_days = close_ary.shape[0]
    
    # Determine eval period: CLI args > fold_meta.json > 80/20 default
    if beg_idx is not None and end_idx is not None:
        train_end_idx = beg_idx
        eval_end_idx = end_idx
        print(f"Eval period (from CLI): [{beg_idx}:{end_idx}]")
    else:
        # Try to load fold metadata saved by CV script
        import json
        fold_meta_path = os.path.join(checkpoint_dir, 'fold_meta.json')
        if os.path.exists(fold_meta_path):
            with open(fold_meta_path) as f:
                meta = json.load(f)
            train_end_idx = meta['val_start']
            eval_end_idx = meta['val_end']
            print(f"Eval period (from fold_meta.json): [{train_end_idx}:{eval_end_idx}]")
            print(f"  CV method: {meta.get('cv_method', 'unknown')}, "
                  f"Fold: {meta.get('fold', '?')}")
        else:
            # Default: 80/20 split (matches main demo_FinRL_Alpaca_VecEnv.py)
            train_end_idx = int(num_days * 0.8)
            eval_end_idx = num_days
            print(f"Eval period (default 80/20): [{train_end_idx}:{eval_end_idx}]")
            print(f"  Tip: Use --beg-idx/--end-idx for CV fold evaluation,")
            print(f"       or run CV script which saves fold_meta.json")
    
    test_days = eval_end_idx - train_end_idx
    print(f"Test period: {test_days} days")
    
    # Create validation env
    val_env = AlpacaStockVecEnv(
        initial_amount=1e6,
        max_stock=100,
        cost_pct=1e-3,
        gamma=0.99,
        beg_idx=train_end_idx,
        end_idx=eval_end_idx,
        num_envs=1,  # Single env for consistent evaluation
        gpu_id=gpu_id
    )
    
    # Check for VecNormalize stats (backward compatible)
    vec_norm_path = os.path.join(checkpoint_dir, 'vec_normalize.pt')
    if os.path.exists(vec_norm_path):
        from elegantrl.envs.vec_normalize import VecNormalize
        val_env = VecNormalize(val_env, training=False)
        val_env.load(vec_norm_path, verbose=True)
        print(f"✓ Loaded VecNormalize stats from {vec_norm_path}")
    elif '_norm' in checkpoint_dir:
        # Old _norm runs without saved stats
        print(f"⚠️  WARNING: This appears to be a normalized run but vec_normalize.pt not found!")
        print(f"   Evaluation with raw observations may produce incorrect results.")
    
    # Get baseline - map eval indices back to dates for baseline fetching
    dates = sorted(df['date'].unique())
    eval_start_date = str(dates[train_end_idx])[:10] if train_end_idx < len(dates) else TEST_START
    eval_end_date = str(dates[min(eval_end_idx - 1, len(dates) - 1)])[:10] if eval_end_idx <= len(dates) else TEST_END
    print(f"Eval date range: {eval_start_date} to {eval_end_date}")
    
    try:
        from finrl.plot import get_baseline
        baseline_df = get_baseline(ticker='^DJI', start=eval_start_date, end=eval_end_date)
        baseline_values = baseline_df['close'] / baseline_df['close'].iloc[0] * 1e6
        baseline_return = (baseline_values.iloc[-1] / 1e6 - 1) * 100
        has_baseline = True
        print(f"DJIA baseline return: {baseline_return:.2f}%")
    except:
        has_baseline = False
        baseline_values = None
        print("Baseline not available")
    
    # Evaluate each checkpoint
    results = []
    all_curves = {}
    
    for ckpt in checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        print(f"\n  Evaluating: {ckpt}...", end=' ')
        
        account_values = evaluate_checkpoint(ckpt_path, val_env, test_days, gpu_id=gpu_id)
        
        if account_values is not None:
            final_return = (account_values[-1] / 1e6 - 1) * 100
            
            # Calculate when it beats baseline
            if has_baseline:
                strategy_returns = (account_values / account_values[0] - 1) * 100
                baseline_x = np.linspace(0, len(strategy_returns)-1, len(baseline_values))
                baseline_resampled = np.interp(range(len(strategy_returns)), baseline_x, 
                                               (baseline_values.values / baseline_values.iloc[0] - 1) * 100)
                
                # Find first day beating baseline
                beating_mask = strategy_returns > baseline_resampled
                if beating_mask.any():
                    first_beat_day = np.argmax(beating_mask)
                    days_beating = beating_mask.sum()
                    pct_beating = days_beating / len(strategy_returns) * 100
                else:
                    first_beat_day = -1
                    days_beating = 0
                    pct_beating = 0
                
                alpha = final_return - baseline_return
            else:
                first_beat_day = -1
                days_beating = 0
                pct_beating = 0
                alpha = 0
            
            # Calculate Sharpe (simplified)
            daily_returns = np.diff(account_values) / account_values[:-1]
            sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8)
            
            # Max drawdown
            peak = np.maximum.accumulate(account_values)
            drawdown = (account_values - peak) / peak
            max_dd = drawdown.min() * 100
            
            results.append({
                'checkpoint': ckpt,
                'final_return': final_return,
                'alpha': alpha,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'first_beat_day': first_beat_day,
                'days_beating': days_beating,
                'pct_beating': pct_beating,
            })
            all_curves[ckpt] = account_values
            
            print(f"Return: {final_return:.2f}%, Alpha: {alpha:+.2f}%, Sharpe: {sharpe:.2f}, "
                  f"Beats baseline: {pct_beating:.0f}% of days")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY (sorted by % days beating baseline)")
    print(f"{'='*70}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('pct_beating', ascending=False)
    
    print(f"\n{'Checkpoint':<45} {'Return':>8} {'Alpha':>8} {'Sharpe':>7} {'MaxDD':>8} {'Beat%':>7} {'FirstBeat':>10}")
    print("-" * 100)
    for _, row in df_results.iterrows():
        print(f"{row['checkpoint']:<45} {row['final_return']:>7.2f}% {row['alpha']:>+7.2f}% "
              f"{row['sharpe']:>7.2f} {row['max_drawdown']:>7.2f}% {row['pct_beating']:>6.0f}% "
              f"{'Day '+str(int(row['first_beat_day'])) if row['first_beat_day'] >= 0 else 'Never':>10}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All curves (cumulative return %)
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_curves)))
    for i, (ckpt, values) in enumerate(all_curves.items()):
        returns = (values / values[0] - 1) * 100
        label = ckpt.replace('actor__', '').replace('.pt', '').replace('.pth', '')
        ax1.plot(returns, label=label, alpha=0.7, linewidth=1.5, color=colors[i])
    
    if has_baseline:
        baseline_x = np.linspace(0, len(list(all_curves.values())[0])-1, len(baseline_values))
        baseline_returns = (baseline_values.values / baseline_values.iloc[0] - 1) * 100
        ax1.plot(baseline_x, baseline_returns, 'k--', linewidth=2, label=f'DJIA ({baseline_return:.1f}%)')
    
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title('All Checkpoints: Cumulative Returns vs DJIA')
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend(loc='upper left', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Best by different metrics
    ax2 = axes[0, 1]
    best_return = df_results.loc[df_results['final_return'].idxmax(), 'checkpoint']
    best_sharpe = df_results.loc[df_results['sharpe'].idxmax(), 'checkpoint']
    best_consistency = df_results.loc[df_results['pct_beating'].idxmax(), 'checkpoint']
    
    for ckpt, label, color in [(best_return, 'Best Return', 'blue'),
                                (best_sharpe, 'Best Sharpe', 'green'),
                                (best_consistency, 'Most Consistent', 'orange')]:
        if ckpt in all_curves:
            values = all_curves[ckpt]
            returns = (values / values[0] - 1) * 100
            final_ret = returns[-1]
            ax2.plot(returns, label=f'{label}: {ckpt[-20:]} ({final_ret:.1f}%)', 
                    linewidth=2, color=color)
    
    if has_baseline:
        ax2.plot(baseline_x, baseline_returns, 'k--', linewidth=2, label=f'DJIA ({baseline_return:.1f}%)')
    
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_title('Best Checkpoints by Different Metrics')
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Return vs Consistency scatter
    ax3 = axes[1, 0]
    ax3.scatter(df_results['pct_beating'], df_results['final_return'], 
                c=df_results['sharpe'], cmap='RdYlGn', s=100, alpha=0.7)
    ax3.set_xlabel('% Days Beating Baseline')
    ax3.set_ylabel('Final Return (%)')
    ax3.set_title('Return vs Consistency (color = Sharpe)')
    ax3.axhline(baseline_return, color='red', linestyle='--', label=f'DJIA Return ({baseline_return:.1f}%)')
    ax3.axvline(50, color='gray', linestyle=':', alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(ax3.collections[0], ax=ax3, label='Sharpe Ratio')
    
    # Plot 4: Metrics bar chart
    ax4 = axes[1, 1]
    x = range(len(df_results))
    width = 0.35
    ax4.bar([i - width/2 for i in x], df_results['final_return'], width, label='Return %', color='blue', alpha=0.7)
    ax4.bar([i + width/2 for i in x], df_results['pct_beating'], width, label='Beat Baseline %', color='green', alpha=0.7)
    ax4.axhline(baseline_return, color='red', linestyle='--', label=f'DJIA ({baseline_return:.1f}%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([c[-15:] for c in df_results['checkpoint']], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Percentage')
    ax4.set_title('Return vs Consistency by Checkpoint')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(checkpoint_dir, 'checkpoint_comparison.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Saved comparison plot to: {save_path}")
    plt.close()
    
    # Save results CSV
    csv_path = os.path.join(checkpoint_dir, 'checkpoint_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"✅ Saved results to: {csv_path}")
    
    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"  🏆 Highest Return: {best_return}")
    print(f"  📊 Best Sharpe:    {best_sharpe}")
    print(f"  ✅ Most Consistent: {best_consistency}")
    
    return df_results, all_curves


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints in a directory')
    parser.add_argument('--dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--beg-idx', type=int, default=None,
                        help='Start index for eval period (overrides auto-detection)')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='End index for eval period (overrides auto-detection)')
    args = parser.parse_args()
    
    main(args.dir, args.gpu, beg_idx=args.beg_idx, end_idx=args.end_idx)
