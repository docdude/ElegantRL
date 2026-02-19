"""
Test the DRL Evaluation framework on existing PPO checkpoints.

This test uses:
- AlpacaStockVecEnv (28 stocks, state_dim=283, action_dim=28)
- Pre-trained checkpoints from /mnt/ssd_backup/ElegantRL/AlpacaStockVecEnv-v1_PPO_1943_v2_norm/
- Evaluation approach matching eval_all_checkpoints.py (NO VecNormalize)
"""

import sys
import os

# Add paths
sys.path.insert(0, '/mnt/ssd_backup/ElegantRL')
sys.path.insert(0, '/mnt/ssd_backup/ElegantRL/lopez_de_prado_analysis')

import torch as th
import numpy as np
import pandas as pd
from pathlib import Path

# Import from the same source as eval_all_checkpoints.py
from examples.demo_FinRL_Alpaca_VecEnv_CV import (
    AlpacaStockVecEnv, download_and_preprocess, df_to_arrays
)
from lopez_de_prado_evaluation_drl import DRLEvaluator


def evaluate_checkpoint(actor_path: str, val_env, num_days: int, 
                        initial_amount: float = 1e6, gpu_id: int = 0):
    """
    Evaluate a single checkpoint and return daily account values.
    MATCHING eval_all_checkpoints.py exactly.
    """
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


def evaluate_multiple_checkpoints(checkpoint_dir: str, gpu_id: int = 0, 
                                  max_checkpoints: int = None):
    """
    Evaluate checkpoints and return results dict compatible with DRLEvaluator.
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find all checkpoints
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('actor__') and f.endswith('.pt'):
            checkpoint_files.append(f)
        elif f == 'act.pth':
            checkpoint_files.append(f)
    
    checkpoint_files = sorted(checkpoint_files)
    
    if max_checkpoints and len(checkpoint_files) > max_checkpoints:
        indices = np.linspace(0, len(checkpoint_files)-1, max_checkpoints, dtype=int)
        checkpoint_files = [checkpoint_files[i] for i in indices]
    
    print(f"\n{'='*60}")
    print(f"Evaluating {len(checkpoint_files)} checkpoints")
    print('='*60)
    
    # Load data and create env (MATCHING eval_all_checkpoints.py)
    df = download_and_preprocess(force_download=False)
    close_ary, tech_ary = df_to_arrays(df)
    num_days = close_ary.shape[0]
    train_end_idx = int(num_days * 0.8)
    test_days = num_days - train_end_idx
    
    print(f"Test period: {test_days} days")
    
    # Create validation env
    val_env = AlpacaStockVecEnv(
        initial_amount=1e6,
        max_stock=100,
        cost_pct=1e-3,
        gamma=0.99,
        beg_idx=train_end_idx,
        end_idx=num_days,
        num_envs=1,
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
        print(f"⚠️  WARNING: Normalized run but vec_normalize.pt not found - results may be incorrect")
    
    # Evaluate each checkpoint
    results = {}
    
    for i, ckpt in enumerate(checkpoint_files):
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        print(f"\n[{i+1}/{len(checkpoint_files)}] {ckpt}...", end=' ')
        
        account_values = evaluate_checkpoint(ckpt_path, val_env, test_days, gpu_id=gpu_id)
        
        if account_values is not None:
            # Calculate metrics (MATCHING eval_all_checkpoints.py)
            final_return = (account_values[-1] / 1e6 - 1) * 100
            daily_returns = np.diff(account_values) / account_values[:-1]
            sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8)
            peak = np.maximum.accumulate(account_values)
            drawdown = (account_values - peak) / peak
            max_dd = drawdown.min() * 100
            
            results[ckpt] = {
                'daily_returns': daily_returns,
                'account_values': account_values,
                'sharpe_ratio': sharpe,
                'cumulative_return': final_return,
                'max_drawdown': abs(max_dd),  # Store as positive for display
                'n_days': len(daily_returns)
            }
            
            print(f"Return: {final_return:.2f}%, Sharpe: {sharpe:.2f}, MDD: {max_dd:.2f}%")
        else:
            results[ckpt] = {'error': 'Failed to load'}
    
    return results, close_ary, train_end_idx, num_days


def run_pbo_analysis(checkpoint_results: dict, close_ary=None, 
                     beg_idx: int = None, end_idx: int = None,
                     min_variance: float = 1e-10):
    """
    Run PBO analysis on checkpoint results.
    
    Args:
        checkpoint_results: Dict of checkpoint evaluation results
        close_ary: Close price array for EQW benchmark Sharpe calculation
        beg_idx: Start index for eval period (for EQW benchmark)
        end_idx: End index for eval period (for EQW benchmark)
        min_variance: Minimum variance to include a checkpoint (filters zero-trading checkpoints)
    """
    print(f"\n{'='*60}")
    print("PBO ANALYSIS")
    print('='*60)
    
    # Filter out checkpoints with zero variance (early checkpoints that don't trade)
    valid_results = {}
    for name, res in checkpoint_results.items():
        if 'daily_returns' not in res or 'error' in res:
            continue
        variance = res['daily_returns'].var()
        if variance > min_variance:
            valid_results[name] = res
        else:
            print(f"   ⚠️ Skipping {name}: zero variance (not trading)")
    
    print(f"   Using {len(valid_results)}/{len(checkpoint_results)} checkpoints (filtered zero-variance)")
    
    if len(valid_results) < 2:
        print("   ❌ Need at least 2 valid checkpoints for PBO")
        return None
    
    # Create evaluator
    evaluator = DRLEvaluator(embargo_td=pd.Timedelta(days=5), n_splits=5)
    
    # Compute EQW benchmark Sharpe for PBO threshold (FinRL_Crypto approach)
    benchmark_sharpe = 0.0
    if close_ary is not None and beg_idx is not None and end_idx is not None:
        prices = close_ary[beg_idx:end_idx]
        if len(prices) > 1:
            daily_stock_returns = np.diff(prices, axis=0) / prices[:-1]
            daily_portfolio_returns = daily_stock_returns.mean(axis=1)
            if daily_portfolio_returns.std() > 1e-8:
                benchmark_sharpe = float(
                    daily_portfolio_returns.mean() / daily_portfolio_returns.std() * np.sqrt(252)
                )
            print(f"   EQW benchmark Sharpe: {benchmark_sharpe:.4f}")
    
    # Compute PBO with valid results only
    pbo_results = evaluator.compute_drl_pbo(
        valid_results, n_splits=8, benchmark_sharpe=benchmark_sharpe
    )
    
    return pbo_results


def main():
    """Main test function."""
    
    # Select checkpoint directory
    checkpoint_dir = '/mnt/ssd_backup/ElegantRL/AlpacaStockVecEnv-v1_PPO_1943_v2_norm'
    
    print("🚀 López de Prado DRL Evaluation Test")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    print(f"   NOTE: Using same evaluation as eval_all_checkpoints.py (NO VecNormalize)")
    
    # Evaluate all checkpoints (no limit - include all)
    all_results, close_ary, train_end_idx, num_days = evaluate_multiple_checkpoints(
        checkpoint_dir, max_checkpoints=None
    )
    
    if not all_results:
        print("No results!")
        return
    
    # Filter valid results
    valid_results = {k: v for k, v in all_results.items() 
                    if 'daily_returns' in v and 'error' not in v}
    
    print(f"\n✅ Valid checkpoints: {len(valid_results)}/{len(all_results)}")
    
    # Summary table
    print("\n📊 Checkpoint Summary:")
    print("-" * 80)
    print(f"{'Checkpoint':<45} {'Return%':>10} {'Sharpe':>8} {'MDD%':>8}")
    print("-" * 80)
    for name, res in sorted(valid_results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True):
        print(f"{name:<45} {res['cumulative_return']:>10.2f} "
              f"{res['sharpe_ratio']:>8.2f} {res['max_drawdown']:>8.2f}")
    
    # Run PBO (with zero-variance filtering + EQW benchmark Sharpe)
    if len(valid_results) >= 2:
        pbo_results = run_pbo_analysis(
            all_results, close_ary=close_ary,
            beg_idx=train_end_idx, end_idx=num_days
        )
        
        # DSR for best checkpoint
        print("\n" + "="*60)
        print("DEFLATED SHARPE RATIO (DSR)")
        print("="*60)
        
        # Find best by Sharpe (excluding zero-variance)
        best_name = max(valid_results.keys(), 
                       key=lambda k: valid_results[k]['sharpe_ratio'] 
                       if valid_results[k]['daily_returns'].var() > 1e-10 else -999)
        best = valid_results[best_name]
        
        print(f"   Best checkpoint: {best_name}")
        print(f"   Sharpe: {best['sharpe_ratio']:.3f}")
        
        evaluator = DRLEvaluator()
        dsr_results = evaluator.compute_deflated_sharpe(
            sharpe_ratio=best['sharpe_ratio'],
            n_trials=len([v for v in valid_results.values() if v['daily_returns'].var() > 1e-10]),
            returns=best['daily_returns']
        )
    else:
        print("\n⚠️ Need at least 2 valid checkpoints for PBO analysis")


if __name__ == "__main__":
    main()
