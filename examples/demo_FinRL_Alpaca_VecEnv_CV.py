"""
ElegantRL VecEnv Demo with FinRL Alpaca Data

This script adapts the China A-shares VecEnv demo to use live market data
from Alpaca via FinRL's AlpacaProcessor, supporting:
- Technical indicators (MACD, RSI, Bollinger Bands, etc.)
- VIX (volatility index)
- Turbulence index
- Multiple CV strategies: holdout, anchored walk-forward, CPCV
- Equal-weight benchmark comparison (excess Sharpe)

CV Methods:
    holdout:  Simple train/val split (fast, no CV)
    wf:       Anchored walk-forward (expanding window, respects time order)
    cpcv:     Combinatorial Purged K-Fold CV (Lopez de Prado 2018)

Usage:
    python examples/demo_FinRL_Alpaca_VecEnv_CV.py
    python examples/demo_FinRL_Alpaca_VecEnv_CV.py --download
    python examples/demo_FinRL_Alpaca_VecEnv_CV.py --agent sac
    python examples/demo_FinRL_Alpaca_VecEnv_CV.py --cv-method wf --folds 3
    python examples/demo_FinRL_Alpaca_VecEnv_CV.py --cv-method cpcv
"""
import os
import itertools
from math import comb
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from elegantrl import Config
from elegantrl import train_agent
from elegantrl.train.config import build_env
from elegantrl.agents import AgentPPO, AgentA2C
from elegantrl.agents import AgentSAC, AgentModSAC, AgentTD3, AgentDDPG
from elegantrl.envs.StockTradingEnv import StockTradingVecEnv
from elegantrl.envs.vec_normalize import VecNormalize

# FinRL backtest imports
try:
    from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
    FINRL_AVAILABLE = True
except ImportError:
    FINRL_AVAILABLE = False
    print("Warning: finrl.plot not available. Install finrl for full backtest stats.")


# =============================================================================
# CROSS-VALIDATION SPLIT GENERATORS
# =============================================================================

def get_holdout_splits(
    total_days: int,
    train_ratio: float = 0.7,
    gap_days: int = 0,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Simple train/val holdout split. Returns a single split."""
    train_end = int(total_days * train_ratio)
    val_start = train_end + gap_days
    return [((0, train_end), (val_start, total_days))]


def get_anchored_walk_forward_splits(
    total_days: int,
    n_folds: int = 3,
    val_ratio: float = 0.2,
    gap_days: int = 0,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Anchored (expanding window) walk-forward splits.
    
    Unlike chunked walk-forward, the training window always starts at day 0
    and grows with each fold. This is the standard approach for time series
    because it uses all available historical data.
    
    Example with n_folds=3, 1000 days, val_ratio=0.2:
        Fold 0: Train [0:267]  -> Val [267:333]    (267 train days)
        Fold 1: Train [0:533]  -> Val [533:667]    (533 train days)
        Fold 2: Train [0:800]  -> Val [800:1000]   (800 train days)
    
    Args:
        total_days: Total number of trading days
        n_folds: Number of walk-forward folds
        val_ratio: Fraction of each fold period used for validation
        gap_days: Embargo gap between train and val (purging)
    
    Returns:
        List of ((train_start, train_end), (val_start, val_end))
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    
    splits = []
    fold_size = total_days // n_folds
    
    for fold in range(n_folds):
        val_end = (fold + 1) * fold_size
        val_days = int(fold_size * val_ratio)
        val_start = val_end - val_days
        train_end = val_start - gap_days
        train_start = 0  # Always anchored at start
        
        if train_end > train_start and val_end > val_start:
            splits.append(((train_start, train_end), (val_start, val_end)))
    
    return splits


def get_cpcv_splits(
    total_days: int,
    n_groups: int = 5,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
) -> List[Tuple[Tuple, Tuple]]:
    """
    Combinatorial Purged Cross-Validation (CPCV) splits.
    
    Based on Lopez de Prado (2018) "Advances in Financial Machine Learning".
    Generates C(n_groups, n_test_groups) train/test combinations with:
    - Purging: removes train samples adjacent to test boundaries
    - Embargo: adds gap after each test fold boundary
    
    Args:
        total_days: Total number of trading days
        n_groups: Total number of groups to divide data into (N)
        n_test_groups: Number of groups used for testing per split (K)
        embargo_pct: Fraction of total data to embargo after each test fold
    
    Returns:
        List of (train_ranges_tuple, test_ranges_tuple) where each is
        a tuple of (start, end) pairs for contiguous blocks.
    """
    if n_test_groups >= n_groups:
        raise ValueError(
            f"n_test_groups ({n_test_groups}) must be < n_groups ({n_groups})"
        )
    
    fold_size = total_days // n_groups
    fold_bounds = []
    for i in range(n_groups):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_groups - 1 else total_days
        fold_bounds.append((start, end))
    
    embargo_days = max(1, int(total_days * embargo_pct))
    
    all_test_combos = list(
        itertools.combinations(range(n_groups), n_test_groups)
    )
    
    splits = []
    for test_group_ids in all_test_combos:
        train_group_ids = [
            i for i in range(n_groups) if i not in test_group_ids
        ]
        
        test_ranges = [fold_bounds[i] for i in test_group_ids]
        
        train_ranges = []
        for train_id in train_group_ids:
            t_start, t_end = fold_bounds[train_id]
            
            for test_id in test_group_ids:
                test_start, test_end = fold_bounds[test_id]
                
                if (t_end <= test_start
                        and t_end > test_start - embargo_days):
                    t_end = max(t_start, test_start - embargo_days)
                
                if (t_start >= test_end
                        and t_start < test_end + embargo_days):
                    t_start = min(t_end, test_end + embargo_days)
            
            if t_end > t_start:
                train_ranges.append((t_start, t_end))
        
        if train_ranges and test_ranges:
            splits.append((tuple(train_ranges), tuple(test_ranges)))
    
    return splits


def get_cv_splits(
    total_days: int,
    cv_method: str = 'holdout',
    n_folds: int = 3,
    n_groups: int = 5,
    n_test_groups: int = 2,
    gap_days: int = 0,
    embargo_pct: float = 0.01,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> dict:
    """
    Unified CV split generator. Returns splits in a standard format.
    
    All splits are normalized to (train_ranges, val_ranges) where each
    is a tuple of (start, end) pairs.
    
    Returns:
        dict with 'splits', 'method', 'n_splits'
    """
    if cv_method == 'holdout':
        raw = get_holdout_splits(total_days, train_ratio, gap_days)
        splits = [
            (((s[0][0], s[0][1]),), ((s[1][0], s[1][1]),))
            for s in raw
        ]
    elif cv_method == 'wf':
        raw = get_anchored_walk_forward_splits(
            total_days, n_folds, val_ratio, gap_days
        )
        splits = [
            (((s[0][0], s[0][1]),), ((s[1][0], s[1][1]),))
            for s in raw
        ]
    elif cv_method == 'cpcv':
        splits = get_cpcv_splits(
            total_days, n_groups, n_test_groups, embargo_pct
        )
    else:
        raise ValueError(
            f"Unknown cv_method '{cv_method}'. Choose: holdout, wf, cpcv"
        )
    
    n_splits = len(splits)
    print(f"\n📊 CV Method: {cv_method.upper()} ({n_splits} splits)")
    
    if cv_method == 'cpcv':
        print(f"   C({n_groups},{n_test_groups}) = "
              f"{comb(n_groups, n_test_groups)} combinations")
        print(f"   Embargo: {embargo_pct*100:.1f}% = "
              f"{int(total_days * embargo_pct)} days")
    elif cv_method == 'wf':
        print(f"   {n_folds} anchored walk-forward folds")
        if gap_days > 0:
            print(f"   Gap (purge): {gap_days} days")
    
    for i, (train_r, val_r) in enumerate(splits[:5]):
        td = sum(e - s for s, e in train_r)
        vd = sum(e - s for s, e in val_r)
        print(f"   Split {i}: Train {td}d {train_r} -> "
              f"Val {vd}d {val_r}")
    if n_splits > 5:
        print(f"   ... and {n_splits - 5} more splits")
    
    return {'splits': splits, 'method': cv_method, 'n_splits': n_splits}


def compute_equal_weight_sharpe(
    close_ary: np.ndarray,
    beg_idx: int,
    end_idx: int,
) -> float:
    """
    Compute annualized Sharpe of an equal-weight buy-and-hold portfolio.
    Used as benchmark for excess Sharpe calculation.
    """
    prices = close_ary[beg_idx:end_idx]
    if len(prices) < 2:
        return 0.0
    
    daily_stock_returns = np.diff(prices, axis=0) / prices[:-1]
    daily_portfolio_returns = daily_stock_returns.mean(axis=1)
    
    if daily_portfolio_returns.std() > 1e-8:
        sharpe = (daily_portfolio_returns.mean()
                  / daily_portfolio_returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    
    return float(sharpe)


# =============================================================================
# AGENT REGISTRY
# =============================================================================

AGENT_REGISTRY = {
    # On-policy agents (can handle larger num_envs)
    'ppo': AgentPPO,
    'a2c': AgentA2C,
    # Off-policy agents (need reduced params for GPU memory)
    'sac': AgentSAC,
    'modsac': AgentModSAC,  # Modified SAC: negative target_entropy + reliable_lambda (better for finance)
    'td3': AgentTD3,
    'ddpg': AgentDDPG,
}

ON_POLICY_AGENTS = {'ppo', 'a2c'}
OFF_POLICY_AGENTS = {'sac', 'modsac', 'td3', 'ddpg'}


# =============================================================================
# BACKTEST UTILITIES (using FinRL infrastructure)
# =============================================================================

def run_backtest_with_stats(env, actor, num_days: int, num_envs: int, 
                            initial_amount: float = 1e6, save_path: str = None,
                            baseline_ticker: str = '^DJI',
                            trade_start_date: str = '2023-06-01',
                            trade_end_date: str = '2024-01-01',
                            agent_name: str = 'PPO',
                            checkpoint_name: str = None):
    """
    Run backtest using FinRL's backtest infrastructure.
    
    Uses env_id=0 as representative trajectory for detailed stats,
    also reports aggregate statistics across all VecEnv parallel runs.
    """
    print(f"\n{'='*60}")
    print(f"BACKTEST VALIDATION")
    print(f"{'='*60}")
    
    # Disable random reset for fair backtest comparison
    env.if_random_reset = False
    
    device = env.device
    
    # Track account values for env 0 (representative trajectory)
    account_values = [initial_amount]
    
    # Run episode
    state, _ = env.reset()
    max_step = num_days - 1
    for t in range(max_step):
        action = actor(state.to(device) if hasattr(state, 'to') else state)
        state, reward, terminal, truncate, info = env.step(action)
        
        # Track total_asset for env 0
        # Note: VecEnv auto-resets on terminal, so we must handle the final step specially
        if hasattr(env, 'total_asset'):
            if t < max_step - 1:
                # Normal step: read total_asset directly
                account_value = env.total_asset[0].cpu().item()
                account_values.append(account_value)
            else:
                # Final step: env has reset, use cumulative_returns to compute final value
                if hasattr(env, 'cumulative_returns') and env.cumulative_returns is not None:
                    cr = env.cumulative_returns
                    if isinstance(cr, list):
                        final_value = initial_amount * cr[0] / 100
                    elif hasattr(cr, 'cpu'):
                        final_value = initial_amount * cr[0].cpu().item() / 100
                    else:
                        final_value = initial_amount * float(cr) / 100
                    account_values.append(final_value)
                else:
                    account_values.append(account_values[-1])  # fallback
    
    # Get final cumulative returns across all envs
    if hasattr(env, 'cumulative_returns'):
        returns = env.cumulative_returns
        if hasattr(returns, 'cpu'):
            returns = returns.cpu().numpy()
        else:
            returns = np.array(returns)
    else:
        returns = np.array([0.0])
    
    print(f"\n{'='*60}")
    print(f"VECENV RESULTS (across {num_envs} parallel envs)")
    print(f"{'='*60}")
    print(f"| Avg Cumulative Return: {returns.mean():.2f}%")
    print(f"| Std Cumulative Return: {returns.std():.2f}%")
    print(f"| Min Return: {returns.min():.2f}%")
    print(f"| Max Return: {returns.max():.2f}%")
    
    # Create DataFrame for FinRL backtest
    dates = pd.bdate_range(start=trade_start_date, periods=len(account_values))
    df_account = pd.DataFrame({
        'date': dates.astype(str),
        'account_value': account_values
    })
    
    print(f"\n{'='*60}")
    print(f"BACKTEST STATISTICS (Env 0 Representative)")
    print(f"{'='*60}")
    print(f"| Trading Days: {len(account_values)}")
    print(f"| Initial Amount: ${initial_amount:,.2f}")
    print(f"| Final Amount: ${account_values[-1]:,.2f}")
    print(f"| Total Return: {(account_values[-1]/initial_amount - 1)*100:.2f}%")
    
    # Use FinRL backtest_stats
    if FINRL_AVAILABLE and len(account_values) > 2:
        print(f"\n{'='*60}")
        print(f"FINRL BACKTEST STATS")
        print(f"{'='*60}")
        try:
            perf_stats = backtest_stats(account_value=df_account)
            print(perf_stats)
        except Exception as e:
            print(f"| Warning: FinRL backtest_stats failed: {e}")
        
        # Get baseline comparison
        if baseline_ticker:
            print(f"\n{'='*60}")
            print(f"BASELINE COMPARISON: {baseline_ticker}")
            print(f"{'='*60}")
            try:
                baseline_df = get_baseline(
                    ticker=baseline_ticker,
                    start=trade_start_date,
                    end=trade_end_date
                )
                if baseline_df is not None and len(baseline_df) > 0:
                    # Normalize baseline to initial_amount
                    baseline_df['account_value'] = baseline_df['close'] / baseline_df['close'].iloc[0] * initial_amount
                    baseline_return = (baseline_df['account_value'].iloc[-1] / initial_amount - 1) * 100
                    print(f"| {baseline_ticker} Return: {baseline_return:.2f}%")
                    print(f"| Strategy vs Baseline: {(account_values[-1]/initial_amount - 1)*100 - baseline_return:+.2f}%")
                    
                    # Baseline stats
                    baseline_for_stats = pd.DataFrame({
                        'date': baseline_df['date'],
                        'account_value': baseline_df['account_value']
                    })
                    print(f"\n| {baseline_ticker} Stats:")
                    try:
                        baseline_perf_stats = backtest_stats(account_value=baseline_for_stats)
                        print(baseline_perf_stats)
                    except Exception as e:
                        print(f"| Warning: Baseline stats failed: {e}")
            except Exception as e:
                print(f"| Warning: Could not fetch baseline {baseline_ticker}: {e}")
    
    # Save results
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save account values
        df_account.to_csv(f"{save_path}/account_values.csv", index=False)
        print(f"\n| Saved account values to: {save_path}/account_values.csv")
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Account value over time with baseline comparison
        ax1 = axes[0, 0]
        ax1.plot(range(len(account_values)), account_values, label=f'{agent_name.upper()} Strategy ({(account_values[-1]/initial_amount - 1)*100:.1f}%)', linewidth=2, color='blue')
        
        # Add baseline if available
        if baseline_ticker and FINRL_AVAILABLE:
            try:
                baseline_df = get_baseline(
                    ticker=baseline_ticker,
                    start=trade_start_date,
                    end=trade_end_date
                )
                if baseline_df is not None and len(baseline_df) > 0:
                    baseline_values = baseline_df['close'] / baseline_df['close'].iloc[0] * initial_amount
                    baseline_return = (baseline_values.iloc[-1] / initial_amount - 1) * 100
                    # Align baseline to strategy timeline
                    baseline_x = np.linspace(0, len(account_values)-1, len(baseline_values))
                    ax1.plot(baseline_x, baseline_values.values, label=f'{baseline_ticker} ({baseline_return:.1f}%)', linewidth=2, color='orange', linestyle='--')
            except Exception as e:
                print(f"| Warning: Could not plot baseline: {e}")
        
        ax1.axhline(initial_amount, color='gray', linestyle=':', alpha=0.5, label='Initial $1M')
        ax1.set_title('Portfolio Value: Strategy vs DJIA Baseline')
        ax1.set_xlabel('Trading Day')
        ax1.set_ylabel('Account Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add annotation for alpha
        strategy_return = (account_values[-1]/initial_amount - 1)*100
        if 'baseline_return' in dir():
            alpha = strategy_return - baseline_return
            ax1.annotate(f'Alpha: {alpha:+.2f}%', xy=(0.98, 0.02), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=11, fontweight='bold',
                        color='green' if alpha > 0 else 'red')
        
        # Plot 2: Return distribution across envs
        ax2 = axes[0, 1]
        ax2.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax2.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax2.set_title(f'Return Distribution ({num_envs} envs)')
        ax2.set_xlabel('Cumulative Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Daily returns
        ax3 = axes[1, 0]
        account_arr = np.array(account_values)
        daily_returns = np.diff(account_arr) / account_arr[:-1] * 100
        ax3.bar(range(len(daily_returns)), daily_returns, alpha=0.7, color='purple')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Daily Returns')
        ax3.set_xlabel('Trading Day')
        ax3.set_ylabel('Daily Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Drawdown
        ax4 = axes[1, 1]
        peak = np.maximum.accumulate(account_arr)
        drawdown = (account_arr - peak) / peak * 100
        ax4.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
        ax4.set_title('Drawdown')
        ax4.set_xlabel('Trading Day')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        if checkpoint_name:
            fig.suptitle(f'Checkpoint: {checkpoint_name}', fontsize=10, y=1.01)
        plt.tight_layout()
        plot_path = f"{save_path}/backtest_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"| Saved backtest plot to: {plot_path}")
        plt.close()
        
        # Create dedicated baseline comparison plot (matching China demo style)
        if baseline_ticker and FINRL_AVAILABLE:
            try:
                baseline_df = get_baseline(
                    ticker=baseline_ticker,
                    start=trade_start_date,
                    end=trade_end_date
                )
                if baseline_df is not None and len(baseline_df) > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Calculate cumulative returns as percentages
                    strategy_returns = (np.array(account_values) / account_values[0] - 1) * 100
                    baseline_prices = baseline_df['close'].values
                    baseline_returns = (baseline_prices / baseline_prices[0] - 1) * 100
                    
                    # Plot strategy
                    ax.plot(range(len(strategy_returns)), strategy_returns, 
                           label=f'{agent_name.upper()} Strategy ({strategy_returns[-1]:.1f}%)', linewidth=2, color='blue')
                    
                    # Plot baseline (resample to match strategy length)
                    baseline_x = np.linspace(0, len(strategy_returns)-1, len(baseline_returns))
                    ax.plot(baseline_x, baseline_returns, 
                           label=f'{baseline_ticker} ({baseline_returns[-1]:.1f}%)', linewidth=2, color='orange')
                    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                    
                    # Fill between to show alpha (resample baseline to match strategy)
                    baseline_resampled = np.interp(range(len(strategy_returns)), baseline_x, baseline_returns)
                    ax.fill_between(range(len(strategy_returns)), 
                                   strategy_returns, 
                                   baseline_resampled,
                                   alpha=0.2, color='green', 
                                   where=strategy_returns > baseline_resampled,
                                   label='Outperformance')
                    ax.fill_between(range(len(strategy_returns)), 
                                   strategy_returns, 
                                   baseline_resampled,
                                   alpha=0.2, color='red', 
                                   where=strategy_returns < baseline_resampled,
                                   label='Underperformance')
                    
                    title = f'Strategy vs {baseline_ticker} Baseline'
                    if checkpoint_name:
                        title += f'\nCheckpoint: {checkpoint_name}'
                    ax.set_title(title)
                    ax.set_xlabel('Trading Day')
                    ax.set_ylabel('Cumulative Return (%)')
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    comparison_path = f"{save_path}/baseline_comparison.png"
                    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                    print(f"| Saved baseline comparison to: {comparison_path}")
                    plt.close()
            except Exception as e:
                print(f"| Warning: Could not create baseline comparison plot: {e}")
    
    return df_account, returns


# =============================================================================
# CONFIGURATION
# =============================================================================

# Alpaca API Credentials (replace with your own)
API_KEY = "PKP504QUJP2PBLNQ0XWT"
API_SECRET = "OdbB6bGJRiHjeZoQEoBgQQD5pj7VJG2vxPNk9bY3"

# Stock universe - DOW 30 (or customize)
TICKERS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT'
]

# Date ranges
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-06-01"  # 80% for training
TEST_START = "2023-06-01"
TEST_END = "2024-01-01"   # 20% for validation

# Technical indicators from FinRL
TECH_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma'
]

# Data cache
DATA_CACHE_DIR = "./datasets"
DATA_CACHE_FILE = f"{DATA_CACHE_DIR}/alpaca_processed.csv"


# =============================================================================
# DATA DOWNLOAD & PREPROCESSING
# =============================================================================

def download_and_preprocess(force_download: bool = False) -> pd.DataFrame:
    """
    Download and preprocess data using FinRL AlpacaProcessor.
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    if os.path.exists(DATA_CACHE_FILE) and not force_download:
        print(f"📂 Loading cached data from {DATA_CACHE_FILE}")
        df = pd.read_csv(DATA_CACHE_FILE)
        df['date'] = pd.to_datetime(df['date'])
        print(f"   ✓ Loaded {len(df)} rows, {df['tic'].nunique()} stocks")
        return df
    
    print(f"📊 Downloading from Alpaca...")
    print(f"   Tickers: {len(TICKERS)}, Range: {TRAIN_START} → {TEST_END}")
    
    from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor
    dp = AlpacaProcessor(API_KEY=API_KEY, API_SECRET=API_SECRET)
    
    # Download raw data
    df = dp.download_data(
        ticker_list=TICKERS,
        start_date=TRAIN_START,
        end_date=TEST_END,
        time_interval='1D'
    )
    print(f"   ✓ Downloaded {len(df)} rows")
    
    # Clean data
    df = dp.clean_data(df)
    
    # Add technical indicators
    df = dp.add_technical_indicator(df, TECH_INDICATORS)
    print(f"   ✓ Added {len(TECH_INDICATORS)} technical indicators")
    
    # Add VIX
    try:
        df = dp.add_vix(df)
        if 'VIXY' in df.columns and 'vix' not in df.columns:
            df = df.rename(columns={'VIXY': 'vix'})
        print(f"   ✓ Added VIX index")
    except Exception as e:
        print(f"   ⚠ VIX not available: {e}")
        df['vix'] = 0
    
    # Add turbulence
    try:
        df = dp.add_turbulence(df)
        print(f"   ✓ Added turbulence index")
    except Exception as e:
        print(f"   ⚠ Turbulence calculation failed: {e}")
        df['turbulence'] = 0
    
    # Clean NaN/Inf values
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Drop redundant 'timestamp' column (we use 'date' for daily data)
    # and reorder columns with 'date' first for readability
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    cols = ['date', 'tic'] + [c for c in df.columns if c not in ['date', 'tic']]
    df = df[cols]
    
    # Save cache
    df.to_csv(DATA_CACHE_FILE, index=False)
    print(f"   ✓ Saved to {DATA_CACHE_FILE}")
    
    return df


def df_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert FinRL DataFrame to close_ary and tech_ary format for StockTradingVecEnv.
    
    Returns:
        close_ary: shape [num_days, num_stocks]
        tech_ary: shape [num_days, num_stocks * num_indicators + market_indicators]
    """
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    dates = sorted(df['date'].unique())
    tickers = sorted(df['tic'].unique())
    num_days = len(dates)
    num_stocks = len(tickers)
    
    # Build close price array [time, stocks]
    close_ary = df.pivot_table(
        index='date', columns='tic', values='close', aggfunc='first'
    ).reindex(columns=tickers).values.astype(np.float32)
    
    # Build tech indicator array
    tech_arys = []
    available_techs = [t for t in TECH_INDICATORS if t in df.columns]
    
    for tech in available_techs:
        tech_pivot = df.pivot_table(
            index='date', columns='tic', values=tech, aggfunc='first'
        ).reindex(columns=tickers).values.astype(np.float32)
        tech_arys.append(tech_pivot)
    
    if tech_arys:
        tech_ary = np.stack(tech_arys, axis=-1)  # [time, stocks, num_techs]
        tech_ary = tech_ary.reshape(num_days, -1)  # [time, stocks * num_techs]
    else:
        tech_ary = np.zeros((num_days, num_stocks), dtype=np.float32)
    
    # Add VIX and turbulence as market-wide indicators
    market_indicators = []
    
    if 'vix' in df.columns:
        vix_ary = df.groupby('date')['vix'].first().reindex(dates).values.astype(np.float32)
        market_indicators.append(vix_ary.reshape(-1, 1))
        
    if 'turbulence' in df.columns:
        turb_ary = df.groupby('date')['turbulence'].first().reindex(dates).values.astype(np.float32)
        market_indicators.append(turb_ary.reshape(-1, 1))
    
    if market_indicators:
        market_ary = np.concatenate(market_indicators, axis=1)
        tech_ary = np.concatenate([tech_ary, market_ary], axis=1)
    
    # Handle NaN values
    close_ary = np.nan_to_num(close_ary, nan=0.0)
    tech_ary = np.nan_to_num(tech_ary, nan=0.0)
    
    print(f"📊 Data arrays created:")
    print(f"   close_ary: {close_ary.shape} (days × stocks)")
    print(f"   tech_ary: {tech_ary.shape} (days × features)")
    print(f"   Features: {len(available_techs)} per-stock + {len(market_indicators)} market-wide")
    
    # Save to npz for the VecEnv to load
    npz_path = f"{DATA_CACHE_DIR}/alpaca_stock_data.numpy.npz"
    np.savez_compressed(npz_path, close_ary=close_ary, tech_ary=tech_ary)
    print(f"   ✓ Saved arrays to {npz_path}")
    
    return close_ary, tech_ary


# =============================================================================
# ADAPTED VECENV FOR ALPACA DATA (Module-level for pickling)
# =============================================================================

# Path where the npz data is stored (set before creating env)
ALPACA_NPZ_PATH = f"{DATA_CACHE_DIR}/alpaca_stock_data.numpy.npz"


class AlpacaStockVecEnv(StockTradingVecEnv):
    """
    StockTradingVecEnv adapted for Alpaca/FinRL data.
    
    This class overrides load_data_from_disk() to load from our pre-processed
    npz file instead of the default China_A_shares files.
    
    Note: Must be defined at module level for multiprocessing pickle support.
    """
    
    def load_data_from_disk(self, tech_id_list=None):
        """Load pre-processed Alpaca data from npz file."""
        if os.path.exists(ALPACA_NPZ_PATH):
            ary_dict = np.load(ALPACA_NPZ_PATH, allow_pickle=True)
            return ary_dict['close_ary'], ary_dict['tech_ary']
        else:
            raise FileNotFoundError(
                f"Alpaca data not found at {ALPACA_NPZ_PATH}. "
                f"Run download_and_preprocess() first."
            )


def find_best_checkpoint(directory: str) -> str:
    """Find the checkpoint with highest avgR from a directory.
    
    Checkpoint naming convention:
    - Best checkpoints: actor__000000120472_00170.385.pt (step_avgR)
    - Periodic checkpoints: actor__000000461390.pt (step only, no avgR)
    
    Returns path to best checkpoint, or most recent if no avgR checkpoints found.
    
    Note: If checkpoint_results.csv exists, uses that for better metric-based selection.
    """
    import re
    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    if not pt_files:
        return None
    
    # Check for checkpoint_results.csv (generated by eval_all_checkpoints.py)
    # This has actual metrics: sharpe, max_drawdown, pct_beating (consistency)
    csv_path = os.path.join(directory, 'checkpoint_results.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0 and 'checkpoint' in df.columns:
                # Selection criteria (in priority order):
                # 1. Filter to checkpoints with positive Sharpe and < -10% drawdown
                # 2. Sort by pct_beating (consistency) then sharpe
                df_good = df[
                    (df['sharpe'] > 0.5) & 
                    (df['max_drawdown'] > -15)  # drawdown is negative, so > -15 means better than -15%
                ].copy()
                
                if len(df_good) > 0:
                    # Sort by consistency (pct_beating) then sharpe
                    df_good = df_good.sort_values(
                        ['pct_beating', 'sharpe'], 
                        ascending=[False, False]
                    )
                    best_ckpt = df_good.iloc[0]['checkpoint']
                    row = df_good.iloc[0]
                    print(f"   🏆 Best checkpoint (from results.csv): {best_ckpt}")
                    print(f"      Return: {row['final_return']:.2f}%, Sharpe: {row['sharpe']:.2f}, "
                          f"MaxDD: {row['max_drawdown']:.1f}%, Consistency: {row['pct_beating']:.0f}%")
                    return f"{directory}/{best_ckpt}"
                else:
                    # Fallback: best sharpe with reasonable drawdown
                    df_ok = df[df['max_drawdown'] > -20].copy()
                    if len(df_ok) > 0:
                        best_idx = df_ok['sharpe'].idxmax()
                        best_ckpt = df_ok.loc[best_idx, 'checkpoint']
                        row = df_ok.loc[best_idx]
                        print(f"   📊 Best checkpoint (by Sharpe, from results.csv): {best_ckpt}")
                        print(f"      Return: {row['final_return']:.2f}%, Sharpe: {row['sharpe']:.2f}, "
                              f"MaxDD: {row['max_drawdown']:.1f}%")
                        return f"{directory}/{best_ckpt}"
        except Exception as e:
            print(f"   ⚠️ Could not read checkpoint_results.csv: {e}")
    
    # Fallback to avgR-based selection (original method)
    # Pattern to match avgR in filename: actor__STEP_AVGR.pt
    # e.g., actor__000000120472_00170.385.pt -> avgR = 170.385
    avgr_pattern = re.compile(r'actor__\d+_(\d+\.\d+)\.pt$')
    
    best_file = None
    best_avgr = float('-inf')
    
    for f in pt_files:
        match = avgr_pattern.match(f)
        if match:
            avgr = float(match.group(1))
            if avgr > best_avgr:
                best_avgr = avgr
                best_file = f
    
    if best_file:
        print(f"   🏆 Found best checkpoint (by avgR): {best_file} (avgR={best_avgr:.3f})")
        print(f"      ⚠️ Note: Run eval_all_checkpoints.py for better metric-based selection")
        return f"{directory}/{best_file}"
    else:
        # No avgR checkpoints found, fall back to most recent
        most_recent = sorted(pt_files)[-1]
        print(f"   ⚠️  No avgR checkpoints found, using most recent: {most_recent}")
        return f"{directory}/{most_recent}"


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def run(gpu_id: int = 0, force_download: bool = False, agent_name: str = 'ppo',
        eval_only: bool = False, checkpoint: str = None, use_vec_normalize: bool = False,
        run_suffix: str = None, use_best_checkpoint: bool = False,
        cv_method: str = 'holdout', n_folds: int = 3, gap_days: int = 0,
        n_groups: int = 5, n_test_groups: int = 2, embargo_pct: float = 0.01,
        continue_train: bool = False):
    """Main training function using Alpaca data with ElegantRL VecEnv.
    
    Args:
        gpu_id: GPU device ID
        force_download: Force re-download data from Alpaca
        agent_name: Agent to use ('ppo', 'a2c', 'sac', 'td3', 'ddpg')
        eval_only: Skip training, run validation only
        checkpoint: Path to checkpoint directory or .pt file (for eval_only)
        use_vec_normalize: Enable observation/reward normalization
        run_suffix: Optional suffix for checkpoint directory
        use_best_checkpoint: Load best checkpoint instead of most recent
        cv_method: CV strategy: 'holdout', 'wf' (anchored walk-forward), 'cpcv'
        n_folds: Number of walk-forward folds (for cv_method='wf')
        gap_days: Purging gap between train/val (holdout/wf)
        n_groups: CPCV: number of groups N (for cv_method='cpcv')
        n_test_groups: CPCV: test groups K -> C(N,K) splits
        embargo_pct: CPCV: embargo fraction after test groups
        continue_train: Resume training from latest checkpoint
    """
    
    # Validate agent
    agent_name = agent_name.lower()
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose from: {list(AGENT_REGISTRY.keys())}")
    
    agent_class = AGENT_REGISTRY[agent_name]
    is_off_policy = agent_name in OFF_POLICY_AGENTS
    
    # 1. Download/load data
    print("\n" + "="*60)
    print("STEP 1: Data Download & Preprocessing")
    print("="*60)
    df = download_and_preprocess(force_download=force_download)
    
    # 2. Convert to arrays
    close_ary, tech_ary = df_to_arrays(df)
    num_days = close_ary.shape[0]
    num_stocks = close_ary.shape[1]
    
    # Calculate dimensions
    amount_dim = 1
    state_dim = num_stocks + close_ary.shape[1] + tech_ary.shape[1] + amount_dim
    action_dim = num_stocks
    
    # Generate CV splits using unified dispatcher
    cv_info = get_cv_splits(
        total_days=num_days,
        cv_method=cv_method,
        n_folds=n_folds,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        gap_days=gap_days,
        embargo_pct=embargo_pct,
    )
    splits = cv_info['splits']
    n_splits = cv_info['n_splits']
    
    print(f"   state_dim: {state_dim}, action_dim: {action_dim}")
    
    # 3. Environment Setup (using module-level AlpacaStockVecEnv class)
    print("\n" + "="*60)
    print("STEP 2: Environment Setup")
    print("="*60)
    
    # 4. Configure training (same settings for all folds)
    gamma = 0.99
    
    # Off-policy agents need reduced parameters to avoid OOM
    # ReplayBuffer shape: (buffer_size, num_envs, state_dim) - memory scales with BOTH!
    # L4 (24GB): num_envs=96, buffer_size=100K = ~12GB buffer + ~10GB overhead = fits
    # Prioritize large buffer for SAC - more important than num_envs for off-policy
    # Note: horizon_len will be set per-fold based on val period length
    if is_off_policy:
        num_envs = 96       # 96 envs (fits on L4 24GB with large buffer)
        batch_size = 256
        buffer_size = int(1e5)  # 100K - buffer is (100K, 96, 283) = ~12GB
        repeat_times = 2.0      # more gradient updates per sample (like demo_DDPG_TD3_SAC)
        learning_rate = 1e-4    # slightly lower for stability
        net_dims = [256, 128]   # slightly larger network to compensate
        state_value_tau = 0
        soft_update_tau = 5e-3  # target network update rate
        lambda_entropy = None   # Not used by off-policy
        break_step = int(5e5)
        print(f"   ⚡ Off-policy mode: num_envs={num_envs}, buffer_size={buffer_size:,} (~12GB buffer)")
    else:
        num_envs = 2 ** 11  # 2048 parallel envs
        batch_size = None   # use default
        buffer_size = None  # on-policy doesn't use large buffer
        soft_update_tau = None  # Not used by on-policy
        repeat_times = 16
        learning_rate = 1e-4  # Reduced from 2e-4 for stability with large advantages
        net_dims = [128, 64]
        state_value_tau = 0.1  # Will be overridden to 0 if using VecNormalize
        lambda_entropy = 0.01  # Entropy coefficient for exploration (default 0.001 is too low)
        break_step = int(4e6)  # Extended from 1e6 - PPO benefits from longer training
        print(f"   🚀 On-policy mode: num_envs={num_envs}, lr={learning_rate}, entropy={lambda_entropy}")
    
    # If using VecNormalize, disable agent's internal state normalization to avoid double-normalization
    if use_vec_normalize and state_value_tau > 0:
        print(f"   ⚠️  Disabling state_value_tau (was {state_value_tau}) - VecNormalize handles normalization")
        state_value_tau = 0
    
    # VecNormalize settings
    # ElegantRL envs already apply internal reward scaling (reward_scale=2^-12 for stocks,
    # HPO-tuned norm_reward for crypto). Adding VecNormalize norm_reward on top creates
    # double-normalization that causes divergence for ALL agent types:
    #   - TD3 with norm_reward=True: objC exploded to 1,360,436 by 281K steps
    #   - TD3 without VecNorm: objC stable at ~1.0 over 496K steps
    #   - A2C with norm_reward=True: objA exploded at ~450K steps
    #   - A2C with norm_reward=False: objC stable over 1M steps
    # RL Zoo also never uses VecNormalize for TD3/SAC/DDPG.
    # PPO with norm_reward=True: stable (clip prevents gradient explosion)
    # A2C with norm_reward=True: objA exploded at ~420K steps (no clip)
    # Off-policy: always False (never use VecNormalize reward scaling)
    norm_reward = not is_off_policy  # True for PPO/A2C, False for SAC/TD3/DDPG
    
    vec_normalize_kwargs = {
        'norm_obs': True,
        'norm_reward': norm_reward,
        'clip_obs': 10.0,
        'clip_reward': 10.0 if norm_reward else None,
        'gamma': gamma,
        'training': True,
    }
    
    if use_vec_normalize:
        print(f"   📊 VecNormalize: norm_obs=True, norm_reward={norm_reward} (env already scales rewards internally)")
    
    # Dynamic num_workers based on CPU count and policy type
    cpu_count = os.cpu_count() or 8
    num_workers = min(cpu_count // 2, 6) if is_off_policy else min(cpu_count // 4, 4)
    
    # =========================================================================
    # CROSS-VALIDATION TRAINING LOOP
    # =========================================================================
    
    fold_results = []  # Store results for each fold
    
    for fold_idx, (train_ranges, val_ranges) in enumerate(splits):
        fold_num = fold_idx + 1
        # Extract contiguous boundaries from ranges
        train_start = train_ranges[0][0]
        train_end = train_ranges[-1][1]
        val_start = val_ranges[0][0]
        val_end = val_ranges[-1][1]
        max_step = val_end - val_start - 1
        
        if max_step < 2:
            print(f"\n  Fold {fold_num}: skipping (val too short: {max_step+1} days)")
            fold_results.append({'fold': fold_num, 'return': None, 'sharpe': None})
            continue
        
        print("\n" + "="*60)
        print(f"FOLD {fold_num}/{n_splits}: Train [{train_start}:{train_end}] -> Val [{val_start}:{val_end}]")
        print("="*60)
        
        # Create env_args for this fold
        env_args = {
            'env_name': 'AlpacaStockVecEnv-v1',
            'num_envs': num_envs,
            'max_step': max_step,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'if_discrete': False,
            'gamma': gamma,
            'beg_idx': train_start,
            'end_idx': train_end,
            # VecNormalize: running observation/reward normalization
            'use_vec_normalize': use_vec_normalize,
            'vec_normalize_kwargs': vec_normalize_kwargs,
        }
        
        # Create config for this fold
        args = Config(agent_class, AlpacaStockVecEnv, env_args)
        
        args.gpu_id = gpu_id
        args.random_seed = gpu_id + 1943 + fold_idx  # Different seed per fold
        
        # Custom checkpoint directory with fold number
        base_cwd = f'./AlpacaStockVecEnv-v1_{agent_class.__name__[5:]}_{args.random_seed}'
        if n_splits > 1:
            base_cwd = f'{base_cwd}_fold{fold_num}'
        if run_suffix:
            base_cwd = f'{base_cwd}_{run_suffix}'
        args.cwd = base_cwd
        print(f"   📁 Checkpoint dir: {args.cwd}")
        
        # SAFETY: Don't auto-remove existing checkpoints
        args.if_remove = False
        args.continue_train = continue_train  # Resume from latest checkpoint if True
        
        args.break_step = break_step
        args.net_dims = net_dims
        args.gamma = gamma
        # horizon_len: on-policy uses full episode, off-policy uses 1/4 for batching
        args.horizon_len = max_step if not is_off_policy else max_step // 4
        args.repeat_times = repeat_times
        args.learning_rate = learning_rate
        args.state_value_tau = state_value_tau
        
        # Off-policy specific settings
        if is_off_policy:
            args.batch_size = batch_size
            args.buffer_size = buffer_size
            args.soft_update_tau = soft_update_tau
        else:
            # On-policy specific settings (PPO/A2C)
            # Explicit defaults matching HPO script (agent getattr fallbacks are the same,
            # but being explicit prevents silent drift if agent defaults ever change)
            args.lambda_entropy = lambda_entropy
            args.ratio_clip = 0.25           # PPO clip range: ratio.clamp(1-clip, 1+clip)
            args.lambda_gae_adv = 0.95       # GAE lambda for advantage estimation
            args.if_use_v_trace = True        # V-trace for on-policy advantage calculation
            args.clip_grad_norm = 3.0         # Gradient clipping norm
        
        args.eval_times = 32
        args.eval_per_step = int(5e3)  # More frequent feedback (matches main script)
        args.num_workers = num_workers
        
        # Eval env uses validation period
        args.eval_env_class = AlpacaStockVecEnv
        args.eval_env_args = {
            'env_name': 'AlpacaStockVecEnv-v1',
            'num_envs': num_envs,
            'max_step': max_step,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'if_discrete': False,
            'beg_idx': val_start,
            'end_idx': val_end,
            'use_vec_normalize': use_vec_normalize,
            'vec_normalize_kwargs': {**vec_normalize_kwargs, 'training': False},
        }
        
        # 5. Train (or skip if eval_only)
        if not eval_only:
            print(f"\n   Training Fold {fold_num}...")
            print(f"   Agent: {agent_class.__name__}")
            print(f"   Type: {'Off-policy' if is_off_policy else 'On-policy'}")
            print(f"   num_envs: {num_envs}")
            print(f"   horizon_len: {args.horizon_len}")
            print(f"   break_step: {break_step:,}")
            if is_off_policy:
                print(f"   batch_size: {batch_size}")
                print(f"   buffer_size: {buffer_size:,}")
            if continue_train:
                print(f"   🔄 RESUMING from latest checkpoint in: {args.cwd}")
            
            # Save fold metadata for eval_all_checkpoints.py auto-detection
            import json
            os.makedirs(args.cwd, exist_ok=True)
            fold_meta = {
                'cv_method': cv_method,
                'fold': fold_num,
                'n_splits': n_splits,
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'agent': agent_name,
                'use_vec_normalize': use_vec_normalize,
            }
            with open(os.path.join(args.cwd, 'fold_meta.json'), 'w') as f:
                json.dump(fold_meta, f, indent=2)
            
            train_agent(args, if_single_process=False)
            # Note: VecNormalize stats are automatically saved by run.py if use_vec_normalize=True
        
        # 6. Validation for this fold
        # Find checkpoint to evaluate
        actor_path = None
        
        # In eval_only mode with provided checkpoint: use SAME checkpoint for ALL folds
        # This allows evaluating a single trained model across multiple validation windows
        if checkpoint:
            if checkpoint.endswith('.pt') or checkpoint.endswith('.pth'):
                actor_path = checkpoint
            elif os.path.isdir(checkpoint):
                actor_path = find_best_checkpoint(checkpoint) if use_best_checkpoint else None
                if not actor_path:
                    pt_files = sorted([s for s in os.listdir(checkpoint)
                                       if s.endswith('.pt') and s.startswith('actor')])
                    if pt_files:
                        actor_path = f"{checkpoint}/{pt_files[-1]}"
        
        # If no checkpoint provided (or not found), try fold-specific checkpoint dir
        if not actor_path and os.path.isdir(args.cwd):
            if use_best_checkpoint:
                actor_path = find_best_checkpoint(args.cwd)
            else:
                pt_files = sorted([s for s in os.listdir(args.cwd)
                                   if s.endswith('.pt') and s.startswith('actor')])
                if pt_files:
                    actor_path = f"{args.cwd}/{pt_files[-1]}"
        
        if not actor_path:
            print(f"   ⚠️  No checkpoint found for fold {fold_num}, skipping validation")
            fold_results.append({'fold': fold_num, 'return': None, 'sharpe': None})
            continue
        
        print(f"\n   📂 Loading checkpoint: {os.path.basename(actor_path)}")
        
        # Create validation environment
        val_env = AlpacaStockVecEnv(
            initial_amount=1e6,
            max_stock=100,
            cost_pct=1e-3,
            gamma=gamma,
            beg_idx=val_start,
            end_idx=val_end,
            num_envs=num_envs,
            gpu_id=gpu_id
        )
        
        # Wrap with VecNormalize if training used it (load stats from fold checkpoint dir)
        # training=False: freeze stats, still apply saved obs normalization to match training
        vec_norm_path = os.path.join(args.cwd, 'vec_normalize.pt') if os.path.isdir(args.cwd) else None
        if vec_norm_path and os.path.exists(vec_norm_path):
            val_env = VecNormalize(val_env, training=False)
            val_env.load(vec_norm_path, verbose=True)
            print(f"   ✓ Loaded VecNormalize stats from {vec_norm_path}")
        elif use_vec_normalize:
            print(f"   ⚠️  WARNING: --normalize was used but vec_normalize.pt not found in {args.cwd}!")
            print(f"      Evaluation with raw observations may produce INCORRECT results.")
        
        # Load actor
        try:
            actor = th.load(actor_path, map_location=f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu', weights_only=False)
            actor.eval()
        except:
            agent = agent_class(net_dims, state_dim, action_dim, gpu_id=gpu_id)
            agent.act.load_state_dict(th.load(actor_path, map_location='cpu', weights_only=False))
            actor = agent.act
            actor.eval()
        
        # Disable random reset for fair/reproducible evaluation
        val_env.if_random_reset = False
        
        # Run validation episode and track actual account values
        state, _ = val_env.reset()
        account_values = [1e6]  # Track actual portfolio value for proper Sharpe
        device = f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu'
        
        for t in range(max_step):
            with th.no_grad():
                action = actor(state.to(device) if hasattr(state, 'to') else state)
            state, reward, terminal, truncate, info = val_env.step(action)
            
            # Track actual total_asset for env 0 (like eval_all_checkpoints.py)
            if hasattr(val_env, 'total_asset'):
                if t < max_step - 1:
                    account_values.append(val_env.total_asset[0].cpu().item())
                else:
                    # Final step: use cumulative_returns to compute final value
                    if hasattr(val_env, 'cumulative_returns') and val_env.cumulative_returns is not None:
                        cr = val_env.cumulative_returns
                        if hasattr(cr, 'cpu'):
                            final_value = 1e6 * cr[0].cpu().item() / 100
                        elif isinstance(cr, (list, np.ndarray)):
                            final_value = 1e6 * float(cr[0]) / 100
                        else:
                            final_value = 1e6 * float(cr) / 100
                        account_values.append(final_value)
                    else:
                        account_values.append(account_values[-1])
        
        # Calculate fold metrics
        # Use env's cumulative_returns which is (total_asset / initial_amount) * 100
        # So cumulative_returns=122.31 means 122.31% of initial = 22.31% gain
        # We subtract 100 to get the actual gain percentage
        if hasattr(val_env, 'cumulative_returns') and val_env.cumulative_returns is not None:
            cr = val_env.cumulative_returns
            if hasattr(cr, 'cpu'):
                final_return = cr[0].cpu().item() - 100.0  # Subtract 100 to get gain %
            elif isinstance(cr, (list, np.ndarray)):
                final_return = float(cr[0]) - 100.0
            else:
                final_return = float(cr) - 100.0
        else:
            # Fallback: compute from account values
            final_return = (account_values[-1] / account_values[0] - 1) * 100
        
        # Sharpe from actual daily percentage returns (matches eval_all_checkpoints.py)
        account_arr = np.array(account_values)
        daily_pct_returns = np.diff(account_arr) / account_arr[:-1]
        if daily_pct_returns.std() > 1e-8:
            sharpe = daily_pct_returns.mean() / daily_pct_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Get DJI baseline for this fold's validation period
        dji_return = None
        alpha = None
        if FINRL_AVAILABLE:
            try:
                # Get actual dates for validation period from the dataframe
                dates = sorted(df['date'].unique())
                val_start_date = str(dates[val_start])[:10]
                val_end_date = str(dates[min(val_end-1, len(dates)-1)])[:10]
                
                baseline_df = get_baseline(ticker='^DJI', start=val_start_date, end=val_end_date)
                if baseline_df is not None and len(baseline_df) > 1:
                    dji_return = (baseline_df['close'].iloc[-1] / baseline_df['close'].iloc[0] - 1) * 100
                    alpha = final_return - dji_return
            except Exception as e:
                pass  # Baseline not available for this period
        
        # EQW benchmark Sharpe for this fold's val period
        eqw_sharpe = compute_equal_weight_sharpe(close_ary, val_start, val_end)
        excess_sharpe = sharpe - eqw_sharpe
        
        # Save daily returns for PBO analysis (matches FinRL_Crypto approach)
        # PBO needs per-strategy (checkpoint/trial) return time series
        fold_save_dir = os.path.dirname(actor_path) if actor_path else args.cwd
        if fold_save_dir and os.path.isdir(fold_save_dir):
            np.save(os.path.join(fold_save_dir, 'daily_returns.npy'), daily_pct_returns)
            np.save(os.path.join(fold_save_dir, 'account_values.npy'), account_arr)
            print(f"   | Saved daily_returns.npy ({len(daily_pct_returns)} days) for PBO")
        
        fold_results.append({
            'fold': fold_num,
            'return': final_return,
            'sharpe': sharpe,
            'eqw_sharpe': eqw_sharpe,
            'excess_sharpe': excess_sharpe,
            'dji_return': dji_return,
            'alpha': alpha,
            'checkpoint': actor_path,
            'daily_returns': daily_pct_returns,  # Keep in-memory for PBO
        })
        
        print(f"\n   Fold {fold_num} Results:")
        print(f"   | Cumulative Return: {final_return:.2f}%")
        print(f"   | Agent Sharpe: {sharpe:.3f}")
        print(f"   | EQW Benchmark Sharpe: {eqw_sharpe:.3f}")
        print(f"   | Excess Sharpe: {excess_sharpe:+.3f}")
        if dji_return is not None:
            print(f"   | DJI Baseline: {dji_return:.2f}%")
            print(f"   | Alpha (vs DJI): {alpha:+.2f}%")
    
    # =========================================================================
    # AGGREGATE CROSS-VALIDATION RESULTS
    # =========================================================================
    
    if len(splits) > 1:
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION SUMMARY ({cv_info['method'].upper()}, "
              f"{len([r for r in fold_results if r['return'] is not None])}/{n_splits} splits)")
        print("="*60)
        
        valid_results = [r for r in fold_results if r['return'] is not None]
        if valid_results:
            returns = [r['return'] for r in valid_results]
            sharpes = [r['sharpe'] for r in valid_results]
            eqw_sharpes = [r['eqw_sharpe'] for r in valid_results]
            excess_sharpes = [r['excess_sharpe'] for r in valid_results]
            
            # Check if we have DJI data
            has_dji = any(r.get('dji_return') is not None for r in valid_results)
            
            # Always show excess Sharpe (EQW benchmark)
            print(f"\n{'Fold':<6} {'Return':>10} {'Sharpe':>8} {'EQW':>8} {'Excess':>8}", end='')
            if has_dji:
                print(f" {'DJI':>10} {'Alpha':>10}", end='')
            print()
            print("-" * (42 + (22 if has_dji else 0)))
            
            for r in fold_results:
                if r['return'] is not None:
                    line = (f"{r['fold']:<6} {r['return']:>9.2f}% "
                            f"{r['sharpe']:>8.3f} {r['eqw_sharpe']:>8.3f} "
                            f"{r['excess_sharpe']:>+8.3f}")
                    if has_dji:
                        dji_str = f"{r['dji_return']:.2f}%" if r.get('dji_return') is not None else "N/A"
                        alpha_str = f"{r['alpha']:+.2f}%" if r.get('alpha') is not None else "N/A"
                        line += f" {dji_str:>10} {alpha_str:>10}"
                    print(line)
                else:
                    print(f"{r['fold']:<6} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            
            print("-" * (42 + (22 if has_dji else 0)))
            mean_line = (f"{'Mean':<6} {np.mean(returns):>8.2f}% "
                         f"{np.mean(sharpes):>8.3f} {np.mean(eqw_sharpes):>8.3f} "
                         f"{np.mean(excess_sharpes):>+8.3f}")
            std_line = (f"{'Std':<6} {np.std(returns):>8.2f}% "
                        f"{np.std(sharpes):>8.3f} {np.std(eqw_sharpes):>8.3f} "
                        f"{np.std(excess_sharpes):>8.3f}")
            
            if has_dji:
                alphas = [r['alpha'] for r in valid_results if r.get('alpha') is not None]
                dji_returns = [r['dji_return'] for r in valid_results if r.get('dji_return') is not None]
                if dji_returns:
                    mean_line += f" {np.mean(dji_returns):>8.2f}% {np.mean(alphas):>+8.2f}%"
                    std_line += f" {np.std(dji_returns):>8.2f}% {np.std(alphas):>8.2f}%"
            
            print(mean_line)
            print(std_line)
            
            # Summary note with excess Sharpe
            print(f"\n📊 CV provides a more robust OOS estimate.")
            print(f"   Mean return: {np.mean(returns):.2f}% ± {np.std(returns):.2f}%")
            print(f"   Mean Sharpe: {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")
            print(f"   Mean Excess Sharpe (vs EQW): {np.mean(excess_sharpes):+.3f} ± {np.std(excess_sharpes):.3f}")
            if has_dji and alphas:
                print(f"   Mean Alpha vs DJI: {np.mean(alphas):+.2f}% ± {np.std(alphas):.2f}%")
        else:
            print("   ⚠️  No valid fold results to aggregate")
    
    # For single fold (simple holdout), offer detailed FinRL backtest with plots
    elif len(splits) == 1 and fold_results and fold_results[0].get('checkpoint'):
        train_ranges, val_ranges = splits[0]
        val_start_idx = val_ranges[0][0]
        val_end_idx = val_ranges[-1][1]
        
        # Ask user if they want detailed backtest (skip prompt in eval_only mode)
        run_detailed = eval_only  # Auto-run in eval mode
        if not eval_only:
            run_detailed = input("\n| Press 'y' for detailed FinRL backtest with plots: ").lower() == 'y'
        
        if run_detailed:
            actor_path = fold_results[0]['checkpoint']
            save_path = os.path.dirname(actor_path)
            
            print(f"\n| Running detailed backtest with FinRL stats and plots...")
            
            # Create validation environment
            val_env = AlpacaStockVecEnv(
                initial_amount=1e6,
                max_stock=100,
                cost_pct=1e-3,
                gamma=gamma,
                beg_idx=val_start_idx,
                end_idx=val_end_idx,
                num_envs=num_envs,
                gpu_id=gpu_id
            )
            
            # Wrap with VecNormalize if training used it
            # training=False: freeze stats, still apply saved obs normalization
            vec_norm_path = os.path.join(save_path, 'vec_normalize.pt') if save_path else None
            if vec_norm_path and os.path.exists(vec_norm_path):
                val_env = VecNormalize(val_env, training=False)
                val_env.load(vec_norm_path, verbose=True)
                print(f"| ✓ Loaded VecNormalize stats from {vec_norm_path}")
            elif use_vec_normalize:
                print(f"| ⚠️  WARNING: --normalize was used but vec_normalize.pt not found!")
                print(f"|    Evaluation with raw observations may produce INCORRECT results.")
            
            # Load actor
            try:
                actor = th.load(actor_path, map_location=f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu', weights_only=False)
                actor.eval()
            except:
                agent = agent_class(net_dims, state_dim, action_dim, gpu_id=gpu_id)
                agent.act.load_state_dict(th.load(actor_path, map_location='cpu', weights_only=False))
                actor = agent.act
                actor.eval()
            
            # Run comprehensive backtest with FinRL stats and plots
            run_backtest_with_stats(
                env=val_env,
                actor=actor,
                num_days=val_end_idx - val_start_idx,
                num_envs=num_envs,
                initial_amount=1e6,
                save_path=save_path,
                baseline_ticker='^DJI',
                trade_start_date=TEST_START,
                trade_end_date=TEST_END,
                agent_name=agent_name,
                checkpoint_name=os.path.basename(actor_path) if actor_path else None
            )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ElegantRL Alpaca Stock Trading with VecEnv')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--download', action='store_true', help='Force re-download data')
    parser.add_argument('--agent', type=str, default='ppo', 
                        choices=list(AGENT_REGISTRY.keys()),
                        help='Agent: ppo, a2c (on-policy) | sac, modsac, td3, ddpg (off-policy). ModSAC recommended for finance.')
    parser.add_argument('--eval', action='store_true', dest='eval_only',
                        help='Skip training, run validation only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint dir or .pt file (for --eval mode)')
    parser.add_argument('--normalize', action='store_true', dest='use_vec_normalize',
                        help='Enable VecNormalize for obs/reward normalization (recommended for off-policy agents)')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Suffix to append to checkpoint directory (e.g., v2, norm, test)')
    parser.add_argument('--best', action='store_true',
                        help='Load best checkpoint (highest avgR) instead of most recent')
    # Cross-validation arguments
    parser.add_argument('--cv-method', type=str, default='holdout',
                        choices=['holdout', 'wf', 'cpcv'],
                        help='CV method: holdout, wf (anchored walk-forward), cpcv')
    parser.add_argument('--folds', type=int, default=3,
                        help='Number of walk-forward folds (for --cv-method wf)')
    parser.add_argument('--n-groups', type=int, default=5,
                        help='CPCV: number of groups N (for --cv-method cpcv)')
    parser.add_argument('--n-test-groups', type=int, default=2,
                        help='CPCV: test groups K -> C(N,K) splits')
    parser.add_argument('--embargo-pct', type=float, default=0.01,
                        help='CPCV: embargo fraction after test groups')
    parser.add_argument('--gap-days', type=int, default=0,
                        help='Purging gap between train/val (holdout/wf only)')
    parser.add_argument('--continue', action='store_true', dest='continue_train',
                        help='Resume training from latest checkpoint (loads actor + VecNormalize stats)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"ElegantRL VecEnv {'Validation' if args.eval_only else 'Training'}")
    print(f"{'='*60}")
    print(f"Agent: {args.agent.upper()}")
    print(f"GPU: {args.gpu}")
    print(f"VecNormalize: {'Enabled' if args.use_vec_normalize else 'Disabled'}")
    print(f"CV Method: {args.cv_method.upper()}")
    if args.cv_method == 'wf':
        print(f"  Walk-forward: {args.folds} folds, gap={args.gap_days} days")
    elif args.cv_method == 'cpcv':
        print(f"  CPCV: C({args.n_groups},{args.n_test_groups}) = "
              f"{comb(args.n_groups, args.n_test_groups)} splits, "
              f"embargo={args.embargo_pct*100:.1f}%")
    if args.eval_only:
        print(f"Mode: Eval-only")
        if args.checkpoint:
            print(f"Checkpoint: {args.checkpoint}")
    elif args.continue_train:
        print(f"Mode: Resume training from checkpoint")
    else:
        print(f"Available agents: {list(AGENT_REGISTRY.keys())}")
    
    run(gpu_id=args.gpu, force_download=args.download, agent_name=args.agent,
        eval_only=args.eval_only, checkpoint=args.checkpoint,
        use_vec_normalize=args.use_vec_normalize, run_suffix=args.suffix,
        use_best_checkpoint=args.best, cv_method=args.cv_method,
        n_folds=args.folds, gap_days=args.gap_days,
        n_groups=args.n_groups, n_test_groups=args.n_test_groups,
        embargo_pct=args.embargo_pct, continue_train=args.continue_train)
