"""
Multi-Agent López de Prado Evaluation Framework

This script evaluates multiple DRL agents/HPO runs using:
- PBO (Probability of Backtest Overfitting) - for comparing independent trials
- DSR (Deflated Sharpe Ratio) - for assessing selection bias
- Standard metrics (Sharpe, Return, MDD)

Proper Usage:
- PBO/DSR require INDEPENDENT trials (different HPO configs, seeds, or algorithms)
- Do NOT use PBO on checkpoints from a single training run (they're correlated)

Supported input formats:
1. Multiple HPO run directories (each with best checkpoint)
2. Multiple algorithm directories (PPO, SAC, TD3, etc.)
3. CSV file with pre-computed results from HPO sweep
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch as th

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / 'pypbo'))

from pypbo.pbo import pbo as compute_pbo, dsr, expected_max

# Import environment
from examples.demo_FinRL_Alpaca_VecEnv_CV import (
    AlpacaStockVecEnv, download_and_preprocess, df_to_arrays
)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_checkpoint(
    actor_path: str,
    env: AlpacaStockVecEnv,
    num_days: int,
    initial_amount: float = 1e6,
    gpu_id: int = 0,
    vec_normalize_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single checkpoint and return metrics.
    
    Matches eval_all_checkpoints.py approach (NO VecNormalize by default).
    If vec_normalize_path is provided, loads saved normalization stats.
    """
    device = f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu'
    
    # Load actor
    try:
        actor = th.load(actor_path, map_location=device, weights_only=False)
        actor.eval()
    except Exception as e:
        print(f"  ⚠️ Failed to load {actor_path}: {e}")
        return None
    
    # Load VecNormalize if provided
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        from elegantrl.envs.vec_normalize import VecNormalize
        # Wrap env with VecNormalize and load stats
        env = VecNormalize(env, training=False)
        env.load(vec_normalize_path)
        print(f"  ✓ Loaded VecNormalize from {vec_normalize_path}")
    
    # Reset env
    env.if_random_reset = False
    state, _ = env.reset()
    
    # Track account values
    account_values = [initial_amount]
    max_step = num_days - 1
    
    with th.no_grad():
        for t in range(max_step):
            action = actor(state.to(device) if hasattr(state, 'to') else state)
            state, reward, terminal, truncate, info = env.step(action)
            
            if hasattr(env, 'total_asset'):
                if t < max_step - 1:
                    account_values.append(env.total_asset[0].cpu().item())
                else:
                    # Final step
                    if hasattr(env, 'cumulative_returns') and env.cumulative_returns is not None:
                        cr = env.cumulative_returns
                        if hasattr(cr, 'cpu'):
                            final_value = initial_amount * cr[0].cpu().item() / 100
                        else:
                            final_value = initial_amount * float(cr[0]) / 100
                        account_values.append(final_value)
                    else:
                        account_values.append(account_values[-1])
    
    account_values = np.array(account_values)
    
    # Calculate metrics
    daily_returns = np.diff(account_values) / account_values[:-1]
    final_return = (account_values[-1] / initial_amount - 1) * 100
    sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-8)
    
    peak = np.maximum.accumulate(account_values)
    drawdown = (account_values - peak) / peak
    max_dd = drawdown.min() * 100
    
    return {
        'daily_returns': daily_returns,
        'account_values': account_values,
        'sharpe_ratio': sharpe,
        'cumulative_return': final_return,
        'max_drawdown': max_dd,
        'n_days': len(daily_returns),
        'skewness': pd.Series(daily_returns).skew(),
        'kurtosis': pd.Series(daily_returns).kurtosis() + 3,  # Excess -> regular
    }


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the best checkpoint in a directory (by filename convention or act.pth)."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for act.pth first (final model)
    if (checkpoint_dir / 'act.pth').exists():
        return str(checkpoint_dir / 'act.pth')
    
    # Look for actor checkpoints with score in filename
    actor_files = list(checkpoint_dir.glob('actor__*.pt'))
    if actor_files:
        # Sort by score (higher is better) - format: actor__STEP_SCORE.pt
        def get_score(f):
            parts = f.stem.split('_')
            if len(parts) >= 4:
                try:
                    return float(parts[-1])
                except ValueError:
                    return 0
            return 0
        
        best = max(actor_files, key=get_score)
        return str(best)
    
    return None


def find_vec_normalize(checkpoint_dir: str) -> Optional[str]:
    """Find VecNormalize stats file in checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    
    for name in ['vec_normalize.pt', 'vec_normalize.pth', 'vecnorm.pt']:
        path = checkpoint_dir / name
        if path.exists():
            return str(path)
    
    return None


# =============================================================================
# MULTI-AGENT EVALUATION
# =============================================================================

def evaluate_multiple_agents(
    agent_dirs: List[str],
    gpu_id: int = 0,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate multiple independent agents/HPO runs.
    
    Args:
        agent_dirs: List of directories, each containing a trained model
        gpu_id: GPU to use for evaluation
        verbose: Print progress
    
    Returns:
        Dict mapping agent_name -> evaluation results
    """
    # Load data
    df = download_and_preprocess(force_download=False)
    close_ary, tech_ary = df_to_arrays(df)
    num_days = close_ary.shape[0]
    train_end_idx = int(num_days * 0.8)
    test_days = num_days - train_end_idx
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-Agent Evaluation")
        print(f"{'='*60}")
        print(f"Test period: {test_days} days")
        print(f"Agents to evaluate: {len(agent_dirs)}")
    
    # Create base validation env
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
    
    results = {}
    
    for i, agent_dir in enumerate(agent_dirs):
        agent_name = Path(agent_dir).name
        
        if verbose:
            print(f"\n[{i+1}/{len(agent_dirs)}] {agent_name}...", end=' ')
        
        # Find best checkpoint
        ckpt_path = find_best_checkpoint(agent_dir)
        if not ckpt_path:
            if verbose:
                print("❌ No checkpoint found")
            continue
        
        # Find VecNormalize (if saved)
        vec_norm_path = find_vec_normalize(agent_dir)
        
        # Evaluate
        eval_result = evaluate_checkpoint(
            ckpt_path, val_env, test_days, 
            gpu_id=gpu_id,
            vec_normalize_path=vec_norm_path
        )
        
        if eval_result:
            results[agent_name] = eval_result
            results[agent_name]['checkpoint_path'] = ckpt_path
            
            if verbose:
                print(f"Return: {eval_result['cumulative_return']:.2f}%, "
                      f"Sharpe: {eval_result['sharpe_ratio']:.2f}, "
                      f"MDD: {eval_result['max_drawdown']:.2f}%")
        else:
            if verbose:
                print("❌ Evaluation failed")
    
    return results


def load_hpo_results_csv(csv_path: str) -> Dict[str, Dict]:
    """
    Load HPO results from a CSV file (e.g., from Hypersweeper).
    
    Expected columns: trial_id/name, sharpe, return, max_drawdown, ...
    """
    df = pd.read_csv(csv_path)
    
    results = {}
    for _, row in df.iterrows():
        name = row.get('trial_id') or row.get('name') or row.get('config_id') or f"trial_{len(results)}"
        
        results[str(name)] = {
            'sharpe_ratio': row.get('sharpe') or row.get('sharpe_ratio') or 0,
            'cumulative_return': row.get('return') or row.get('cumulative_return') or row.get('final_return') or 0,
            'max_drawdown': row.get('max_drawdown') or row.get('mdd') or 0,
            # Daily returns not available from CSV - PBO won't work
            'from_csv': True,
        }
    
    return results


# =============================================================================
# PBO / DSR ANALYSIS
# =============================================================================

def compute_pbo_analysis(
    results: Dict[str, Dict],
    n_splits: int = 8,
    min_variance: float = 1e-10,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Compute PBO (Probability of Backtest Overfitting) for multiple agents.
    
    REQUIRES: daily_returns for each agent (can't use CSV-only results).
    """
    # Filter agents with valid daily returns
    valid_agents = {}
    for name, res in results.items():
        if res.get('from_csv'):
            if verbose:
                print(f"  ⚠️ Skipping {name}: no daily returns (from CSV)")
            continue
        if 'daily_returns' not in res:
            continue
        if res['daily_returns'].var() < min_variance:
            if verbose:
                print(f"  ⚠️ Skipping {name}: zero variance")
            continue
        valid_agents[name] = res
    
    if verbose:
        print(f"\n{'='*60}")
        print("PBO ANALYSIS")
        print(f"{'='*60}")
        print(f"Valid agents for PBO: {len(valid_agents)}/{len(results)}")
    
    if len(valid_agents) < 2:
        print("  ❌ Need at least 2 agents with daily returns for PBO")
        return None
    
    # Build returns matrix (T x N)
    agent_names = list(valid_agents.keys())
    n_days = valid_agents[agent_names[0]]['n_days']
    
    # Verify all have same length
    for name in agent_names:
        if valid_agents[name]['n_days'] != n_days:
            print(f"  ⚠️ {name} has different number of days, skipping")
            agent_names.remove(name)
    
    M = np.column_stack([valid_agents[name]['daily_returns'] for name in agent_names])
    
    if verbose:
        print(f"  Returns matrix: {M.shape[0]} days × {M.shape[1]} agents")
    
    # Compute PBO
    def sharpe_metric(returns):
        """Sharpe ratio metric for PBO."""
        if returns.std() < 1e-10:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    try:
        pbo_result = compute_pbo(
            M, S=n_splits, 
            metric_func=sharpe_metric,
            threshold=0,
            verbose=False
        )
        
        if verbose:
            print(f"\n✅ PBO Results:")
            print(f"   Agents: {len(agent_names)}")
            print(f"   CSCV splits: {pbo_result.Cs}")
            print(f"   PBO: {pbo_result.pbo:.3f}")
            print(f"   Prob OOS Loss: {pbo_result.prob_oos_loss:.3f}")
            
            if hasattr(pbo_result, 'linear_model') and pbo_result.linear_model:
                slope = pbo_result.linear_model[0]
                print(f"   Degradation: slope={slope:.3f}")
            
            if pbo_result.pbo > 0.5:
                print(f"   🔴 High overfitting risk")
            elif pbo_result.pbo > 0.25:
                print(f"   🟡 Moderate overfitting risk")
            else:
                print(f"   🟢 Low overfitting risk")
        
        return {
            'pbo': pbo_result.pbo,
            'prob_oos_loss': pbo_result.prob_oos_loss,
            'n_agents': len(agent_names),
            'n_splits': n_splits,
            'agent_names': agent_names,
        }
        
    except Exception as e:
        print(f"  ❌ PBO computation failed: {e}")
        return None


def compute_dsr_analysis(
    results: Dict[str, Dict],
    verbose: bool = True,
) -> Dict:
    """
    Compute Deflated Sharpe Ratio for the best agent.
    
    DSR accounts for multiple testing bias when selecting the "best" agent.
    """
    # Get valid agents
    valid_agents = {k: v for k, v in results.items() 
                   if 'sharpe_ratio' in v and v['sharpe_ratio'] > 0}
    
    if not valid_agents:
        return {'error': 'No valid agents'}
    
    # Find best by Sharpe
    best_name = max(valid_agents.keys(), key=lambda k: valid_agents[k]['sharpe_ratio'])
    best = valid_agents[best_name]
    
    if verbose:
        print(f"\n{'='*60}")
        print("DEFLATED SHARPE RATIO (DSR)")
        print(f"{'='*60}")
        print(f"Best agent: {best_name}")
        print(f"Sharpe: {best['sharpe_ratio']:.3f}")
    
    # Get Sharpe std across all agents
    all_sharpes = np.array([v['sharpe_ratio'] for v in valid_agents.values()])
    sharpe_std = all_sharpes.std(ddof=1) if len(all_sharpes) > 1 else 0.5
    
    N = len(valid_agents)
    T = best.get('n_days', 150)
    skew = best.get('skewness', 0)
    kurt = best.get('kurtosis', 3)
    
    if verbose:
        print(f"N trials: {N}")
        print(f"T observations: {T}")
        print(f"Sharpe std: {sharpe_std:.4f}")
        print(f"Expected max under null: {sharpe_std * expected_max(N):.3f}")
    
    # Compute DSR
    dsr_value = dsr(
        test_sharpe=best['sharpe_ratio'],
        sharpe_std=sharpe_std,
        N=N,
        T=T,
        skew=skew,
        kurtosis=kurt
    )
    
    if verbose:
        print(f"\n✅ DSR: {dsr_value:.4f}")
        if dsr_value > 0.95:
            print(f"   🟢 High confidence - unlikely due to luck")
        elif dsr_value > 0.5:
            print(f"   🟡 Moderate confidence")
        else:
            print(f"   🔴 Low confidence - may be due to luck")
    
    return {
        'best_agent': best_name,
        'best_sharpe': best['sharpe_ratio'],
        'dsr': dsr_value,
        'n_trials': N,
        'sharpe_std': sharpe_std,
    }


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary_table(results: Dict[str, Dict]):
    """Print summary table of all agents."""
    print(f"\n{'='*80}")
    print("AGENT SUMMARY")
    print("="*80)
    print(f"{'Agent':<40} {'Return%':>10} {'Sharpe':>8} {'MDD%':>8}")
    print("-"*80)
    
    sorted_agents = sorted(
        results.items(), 
        key=lambda x: x[1].get('sharpe_ratio', 0), 
        reverse=True
    )
    
    for name, res in sorted_agents:
        ret = res.get('cumulative_return', 0)
        sr = res.get('sharpe_ratio', 0)
        mdd = res.get('max_drawdown', 0)
        print(f"{name:<40} {ret:>10.2f} {sr:>8.2f} {mdd:>8.2f}")


def save_results(results: Dict, pbo_results: Optional[Dict], dsr_results: Dict, output_path: str):
    """Save analysis results to JSON."""
    output = {
        'agents': {
            name: {
                'sharpe_ratio': res.get('sharpe_ratio'),
                'cumulative_return': res.get('cumulative_return'),
                'max_drawdown': res.get('max_drawdown'),
                'n_days': res.get('n_days'),
                'checkpoint_path': res.get('checkpoint_path'),
            }
            for name, res in results.items()
        },
        'pbo': pbo_results,
        'dsr': dsr_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent López de Prado Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate multiple HPO runs
  python multi_agent_evaluation.py --dirs run1/ run2/ run3/
  
  # Evaluate all subdirectories matching a pattern
  python multi_agent_evaluation.py --pattern "AlpacaStockVecEnv-*"
  
  # Load from HPO results CSV
  python multi_agent_evaluation.py --csv hpo_results.csv
        """
    )
    
    parser.add_argument('--dirs', nargs='+', help='Agent checkpoint directories')
    parser.add_argument('--pattern', type=str, help='Glob pattern for finding directories')
    parser.add_argument('--csv', type=str, help='CSV file with HPO results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--output', type=str, default='multi_agent_analysis.json', help='Output JSON file')
    parser.add_argument('--n-splits', type=int, default=8, help='Number of CSCV splits for PBO')
    
    args = parser.parse_args()
    
    # Collect agent directories
    agent_dirs = []
    
    if args.dirs:
        agent_dirs = args.dirs
    elif args.pattern:
        import glob
        base_dir = Path(__file__).parent.parent
        agent_dirs = sorted(glob.glob(str(base_dir / args.pattern)))
    elif args.csv:
        # Load from CSV
        results = load_hpo_results_csv(args.csv)
        print(f"Loaded {len(results)} trials from {args.csv}")
    else:
        # Default: find all AlpacaStockVecEnv directories
        base_dir = Path(__file__).parent.parent
        agent_dirs = sorted([
            str(d) for d in base_dir.iterdir() 
            if d.is_dir() and d.name.startswith('AlpacaStockVecEnv')
        ])
    
    if agent_dirs:
        print(f"Found {len(agent_dirs)} agent directories")
        results = evaluate_multiple_agents(agent_dirs, gpu_id=args.gpu)
    
    if not results:
        print("No valid results!")
        return
    
    # Print summary
    print_summary_table(results)
    
    # PBO analysis (requires daily returns)
    has_returns = any('daily_returns' in r for r in results.values())
    pbo_results = None
    if has_returns and len(results) >= 2:
        pbo_results = compute_pbo_analysis(results, n_splits=args.n_splits)
    else:
        print("\n⚠️ Skipping PBO: need daily returns from at least 2 agents")
    
    # DSR analysis
    dsr_results = compute_dsr_analysis(results)
    
    # Save results
    save_results(results, pbo_results, dsr_results, args.output)


if __name__ == "__main__":
    main()
