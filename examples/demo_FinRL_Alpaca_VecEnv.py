"""
ElegantRL VecEnv Demo with FinRL Alpaca Data

This script adapts the China A-shares VecEnv demo to use live market data
from Alpaca via FinRL's AlpacaProcessor, supporting:
- Technical indicators (MACD, RSI, Bollinger Bands, etc.)
- VIX (volatility index)
- Turbulence index

Usage:
    python examples/demo_FinRL_Alpaca_VecEnv.py
    python examples/demo_FinRL_Alpaca_VecEnv.py --download  # Force re-download
    python examples/demo_FinRL_Alpaca_VecEnv.py --agent sac  # Use SAC agent
    python examples/demo_FinRL_Alpaca_VecEnv.py --eval --checkpoint ./AlpacaStockVecEnv-v1_PPO_1943
"""
import os
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
                            agent_name: str = 'PPO'):
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
        
        plt.tight_layout()
        plot_path = f"{save_path}/backtest_results.png"
        plt.savefig(plot_path, dpi=150)
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
                    
                    ax.set_title(f'Strategy vs {baseline_ticker} Baseline')
                    ax.set_xlabel('Trading Day')
                    ax.set_ylabel('Cumulative Return (%)')
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    comparison_path = f"{save_path}/baseline_comparison.png"
                    plt.savefig(comparison_path, dpi=150)
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
    """
    import re
    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    if not pt_files:
        return None
    
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
        print(f"   🏆 Found best checkpoint: {best_file} (avgR={best_avgr:.3f})")
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
        run_suffix: str = None, use_best_checkpoint: bool = False, continue_train: bool = False):
    """Main training function using Alpaca data with ElegantRL VecEnv.
    
    Args:
        gpu_id: GPU device ID
        force_download: Force re-download data from Alpaca
        agent_name: Agent to use ('ppo', 'a2c', 'sac', 'td3', 'ddpg')
        eval_only: Skip training, run validation only
        checkpoint: Path to checkpoint directory or .pt file (for eval_only)
        use_vec_normalize: Enable observation/reward normalization (recommended for SAC/TD3)
        run_suffix: Optional suffix to append to checkpoint directory name (e.g., 'v2', 'norm')
        use_best_checkpoint: Load best checkpoint (highest avgR) instead of most recent
        continue_train: Resume training from latest checkpoint in cwd
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
    
    # Train/val split (80/20)
    train_end_idx = int(num_days * 0.8)
    
    print(f"\n📊 Dataset split:")
    print(f"   Total days: {num_days}")
    print(f"   Training: days 0-{train_end_idx-1} ({train_end_idx} days)")
    print(f"   Validation: days {train_end_idx}-{num_days-1} ({num_days - train_end_idx} days)")
    print(f"   state_dim: {state_dim}, action_dim: {action_dim}")
    
    # 3. Environment Setup (using module-level AlpacaStockVecEnv class)
    print("\n" + "="*60)
    print("STEP 2: Environment Setup")
    print("="*60)
    
    # 4. Configure training
    gamma = 0.99
    max_step = num_days - train_end_idx - 1  # validation period length
    
    # Off-policy agents need reduced parameters to avoid OOM
    # ReplayBuffer shape: (buffer_size, num_envs, state_dim) - memory scales with BOTH!
    # L4 (24GB): num_envs=96, buffer_size=100K = ~12GB buffer + ~10GB overhead = fits
    # Prioritize large buffer for SAC - more important than num_envs for off-policy
    if is_off_policy:
        num_envs = 96       # 96 envs (fits on L4 24GB with large buffer)
        batch_size = 256
        horizon_len = max_step // 4
        buffer_size = int(1.9e5)  # 100K - buffer is (100K, 96, 283) = ~12GB
        repeat_times = 2.0      # more gradient updates per sample (like demo_DDPG_TD3_SAC)
        learning_rate = 1e-4    # slightly lower for stability
        net_dims = [256, 128]   # slightly larger network to compensate
        state_value_tau = 0
        soft_update_tau = 5e-3  # target network update rate
        break_step = int(5e5)
        print(f"   ⚡ Off-policy mode: num_envs={num_envs}, buffer_size={buffer_size:,} (~12GB buffer)")
    else:
        num_envs = 2 ** 11  # 2048 parallel envs
        batch_size = None   # use default
        horizon_len = max_step
        buffer_size = None  # on-policy doesn't use large buffer
        repeat_times = 16
        learning_rate = 1e-4  # Reduced from 2e-4 for stability with large advantages
        net_dims = [128, 64]
        state_value_tau = 0.1  # Will be overridden to 0 if using VecNormalize
        lambda_entropy = 0.01  # Entropy coefficient for exploration (default 0.001 is too low)
        break_step = int(1e6)  # Extended from 1e6 - PPO benefits from longer training
        print(f"   🚀 On-policy mode: num_envs={num_envs}, lr={learning_rate}, entropy={lambda_entropy}")
    
    # If using VecNormalize, disable agent's internal state normalization to avoid double-normalization
    if use_vec_normalize and state_value_tau > 0:
        print(f"   ⚠️  Disabling state_value_tau (was {state_value_tau}) - VecNormalize handles normalization")
        state_value_tau = 0
    
    # VecNormalize settings - different for on-policy vs off-policy agents
    # On-policy (PPO/A2C): Only normalize observations - they have internal advantage normalization
    # Off-policy (SAC/TD3/DDPG): Normalize both obs and rewards - no internal reward handling
    norm_reward = is_off_policy  # Only normalize rewards for off-policy agents
    
    vec_normalize_kwargs = {
        'norm_obs': True,
        'norm_reward': norm_reward,
        'clip_obs': 10.0,
        'clip_reward': 10.0 if norm_reward else None,
        'gamma': gamma,
        'training': True,
    }
    
    if use_vec_normalize:
        if is_off_policy:
            print(f"   📊 VecNormalize: norm_obs=True, norm_reward=True (off-policy)")
        else:
            print(f"   📊 VecNormalize: norm_obs=True, norm_reward=False (on-policy has GAE)")
    
    env_args = {
        'env_name': 'AlpacaStockVecEnv-v1',
        'num_envs': num_envs,
        'max_step': max_step,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'if_discrete': False,
        'gamma': gamma,
        'beg_idx': 0,
        'end_idx': train_end_idx,
        # VecNormalize: running observation/reward normalization
        'use_vec_normalize': use_vec_normalize,
        'vec_normalize_kwargs': vec_normalize_kwargs,
    }
    
    # Use module-level AlpacaStockVecEnv (picklable for multiprocessing)
    args = Config(agent_class, AlpacaStockVecEnv, env_args)
    
    args.gpu_id = gpu_id
    args.random_seed = gpu_id + 1943
    
    # Custom checkpoint directory with optional suffix
    base_cwd = f'./AlpacaStockVecEnv-v1_{agent_class.__name__[5:]}_{args.random_seed}'
    if run_suffix:
        args.cwd = f'{base_cwd}_{run_suffix}'
        print(f"   📁 Checkpoint dir: {args.cwd}")
    
    # SAFETY: Don't auto-remove existing checkpoints - ask user or keep them
    args.if_remove = False  # Changed from default True to prevent accidental data loss
    args.continue_train = continue_train  # Resume from latest checkpoint if True
    
    args.break_step = break_step
    args.net_dims = net_dims
    args.gamma = gamma
    args.horizon_len = horizon_len
    args.repeat_times = repeat_times
    args.learning_rate = learning_rate
    args.state_value_tau = state_value_tau
    # Note: reward_scale stays at default 1.0 (env already scales internally)
    
    # Off-policy specific settings
    if is_off_policy:
        args.batch_size = batch_size
        args.buffer_size = buffer_size
        args.soft_update_tau = soft_update_tau
        # buffer_init_size uses default: batch_size * 8 = 2048
    else:
        # On-policy specific settings (PPO/A2C)
        args.lambda_entropy = lambda_entropy  # Entropy coefficient for exploration
    
    #args.eval_times = 2 ** 14  # 16384 evaluations
    #args.eval_per_step = int(2e4)
    args.eval_times = 32  # Match demo_DDPG_TD3_SAC.py (was 16384 - way too slow!)
    args.eval_per_step = int(5e3)  # More frequent feedback (was 2e4)
     
    # Dynamic num_workers based on CPU count and policy type
    # Off-policy: more workers (CPU-bound gradient updates benefit from parallel rollouts)
    # On-policy: fewer workers (GPU-bound, data collected in large batches)
    cpu_count = os.cpu_count() or 8
    args.num_workers = min(cpu_count // 2, 6) if is_off_policy else min(cpu_count // 4, 4)
    args.eval_env_class = AlpacaStockVecEnv  # module-level class
    args.eval_env_args = {
        'env_name': 'AlpacaStockVecEnv-v1',
        'num_envs': num_envs,
        'max_step': max_step,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'if_discrete': False,
        'beg_idx': train_end_idx,
        'end_idx': num_days,
        # VecNormalize for eval env - uses same settings but will load train stats
        'use_vec_normalize': use_vec_normalize,
        'vec_normalize_kwargs': {**vec_normalize_kwargs, 'training': False},  # Don't update stats during eval
    }
    
    # 5. Train (or skip if eval_only)
    if not eval_only:
        print("\n" + "="*60)
        print("STEP 3: Training")
        print("="*60)
        print(f"   Agent: {agent_class.__name__}")
        print(f"   Type: {'Off-policy' if is_off_policy else 'On-policy'}")
        print(f"   num_envs: {num_envs}")
        print(f"   horizon_len: {horizon_len}")
        print(f"   break_step: {break_step:,}")
        if is_off_policy:
            print(f"   batch_size: {batch_size}")
            print(f"   buffer_size: {buffer_size:,}")
        if continue_train:
            print(f"   🔄 RESUMING from latest checkpoint in: {args.cwd}")
        
        train_agent(args, if_single_process=False)
        cwd = args.cwd
        # Note: VecNormalize stats are automatically saved by run.py if use_vec_normalize=True
    else:
        print("\n" + "="*60)
        print("STEP 3: Skipping Training (eval-only mode)")
        print("="*60)
        cwd = checkpoint
    
    # 6. Validation
    # Find checkpoint - use best (highest avgR) or most recent based on flag
    actor_path = None
    if checkpoint:
        if checkpoint.endswith('.pt') or checkpoint.endswith('.pth'):
            actor_path = checkpoint
            print(f"   📂 Using specified checkpoint: {os.path.basename(checkpoint)}")
        elif os.path.isdir(checkpoint):
            if use_best_checkpoint:
                actor_path = find_best_checkpoint(checkpoint)
            else:
                pt_files = sorted([s for s in os.listdir(checkpoint)
                                   if s.endswith('.pt') and s.startswith('actor')])
                if pt_files:
                    actor_path = f"{checkpoint}/{pt_files[-1]}"
                    print(f"   📂 Using most recent checkpoint: {pt_files[-1]}")
    
    # Initialize save_path
    save_path = None
    
    if not eval_only:
        # After training, ask user
        if input("\n| Press 'y' to load actor.pt and validate: ") != 'y':
            return
        if not actor_path:
            if use_best_checkpoint:
                actor_path = find_best_checkpoint(args.cwd)
            else:
                pt_files = sorted([s for s in os.listdir(args.cwd)
                                   if s.endswith('.pt') and s.startswith('actor')])
                if pt_files:
                    actor_path = f"{args.cwd}/{pt_files[-1]}"
                    print(f"   📂 Using most recent checkpoint: {pt_files[-1]}")
            if not actor_path:
                print(f"❌ No .pt checkpoint files found in {args.cwd}")
                return
        save_path = args.cwd
    else:
        # Eval-only mode requires a checkpoint
        if not actor_path:
            # Try to find latest checkpoint directory
            pattern = f"*_{agent_name.upper()}_*"
            import glob
            dirs = sorted(glob.glob(f"./{pattern}"))
            if dirs:
                cwd = dirs[-1]
                if use_best_checkpoint:
                    actor_path = find_best_checkpoint(cwd)
                else:
                    pt_files = sorted([s for s in os.listdir(cwd)
                                       if s.endswith('.pt') and s.startswith('actor')])
                    if pt_files:
                        actor_path = f"{cwd}/{pt_files[-1]}"
                        print(f"   📂 Using most recent checkpoint: {pt_files[-1]}")
                if actor_path:
                    save_path = cwd
        
        if not actor_path:
            print(f"\n❌ No checkpoint found! Provide --checkpoint path")
            print(f"   Example: --checkpoint ./AlpacaStockVecEnv-v1_PPO_1943/actor_00000981000_00122.38.pt")
            print(f"   Use --best to automatically select the best checkpoint (highest avgR)")
            return
    
    # Determine save path from checkpoint if not set
    if checkpoint and not save_path:
        if checkpoint.endswith('.pt'):
            save_path = os.path.dirname(checkpoint)
        else:
            save_path = checkpoint
    
    print(f"\n| Loading: {actor_path}")
    
    # Create validation environment on HELD-OUT TEST DATA ONLY
    # train_end_idx to num_days is the test split not seen during training
    test_days = num_days - train_end_idx
    print(f"| Test data range: idx {train_end_idx} to {num_days} ({test_days} trading days)")
    print(f"| NOTE: Evaluating on HELD-OUT test data only (not seen during training)")
    
    val_env = AlpacaStockVecEnv(
        initial_amount=1e6,
        max_stock=100,
        cost_pct=1e-3,
        gamma=gamma,
        beg_idx=train_end_idx,  # Start from test data
        end_idx=num_days,        # End at last day
        num_envs=num_envs,
        gpu_id=gpu_id
    )
    
    # Check for VecNormalize stats and wrap if available
    # This ensures backward compatibility:
    # - New runs with vec_normalize.pt: Load stats and wrap env
    # - Old _norm runs without stats: Warn user, use raw env (results may be incorrect)
    # - Non-normalized runs: Use raw env (correct behavior)
    vec_norm_path = os.path.join(save_path, 'vec_normalize.pt') if save_path else None
    if vec_norm_path and os.path.exists(vec_norm_path):
        # New runs: Load VecNormalize stats
        val_env = VecNormalize(val_env, training=False)
        val_env.load(vec_norm_path)
        print(f"| ✓ Loaded VecNormalize stats from {vec_norm_path}")
    elif use_vec_normalize:
        # Old _norm runs: Stats not saved, can't properly evaluate
        print(f"| ⚠️  WARNING: --normalize was used but vec_normalize.pt not found!")
        print(f"|    This model was trained with normalized observations but stats weren't saved.")
        print(f"|    Evaluation with raw observations may produce INCORRECT results.")
        print(f"|    Consider retraining to save normalization stats.")
    # else: Non-normalized run, use raw env (correct)
    
    # Load actor - handle both full model and state_dict formats
    try:
        # Try loading as full module first (ElegantRL default)
        actor = th.load(actor_path, map_location=f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu', weights_only=False)
        actor.eval()
    except:
        # Fall back to state_dict loading
        agent = agent_class(net_dims, state_dim, action_dim, gpu_id=gpu_id)
        agent.act.load_state_dict(th.load(actor_path, map_location='cpu', weights_only=False))
        actor = agent.act
        actor.eval()
    
    # Run comprehensive backtest with stats and plots using FinRL
    run_backtest_with_stats(
        env=val_env,
        actor=actor,
        num_days=test_days,  # Only test period
        num_envs=num_envs,
        initial_amount=1e6,
        save_path=save_path,
        baseline_ticker='^DJI',  # Compare to Dow Jones
        trade_start_date=TEST_START,
        trade_end_date=TEST_END,
        agent_name=agent_name
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
    parser.add_argument('--continue', action='store_true', dest='continue_train',
                        help='Resume training from latest checkpoint (loads actor + VecNormalize stats)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"ElegantRL VecEnv {'Validation' if args.eval_only else 'Training'}")
    print(f"{'='*60}")
    print(f"Agent: {args.agent.upper()}")
    print(f"GPU: {args.gpu}")
    print(f"VecNormalize: {'Enabled' if args.use_vec_normalize else 'Disabled'}")
    if args.eval_only:
        print(f"Mode: Eval-only")
        if args.checkpoint:
            print(f"Checkpoint: {args.checkpoint}")
    else:
        print(f"Available agents: {list(AGENT_REGISTRY.keys())}")
    
    run(gpu_id=args.gpu, force_download=args.download, agent_name=args.agent,
        eval_only=args.eval_only, checkpoint=args.checkpoint, 
        use_vec_normalize=args.use_vec_normalize, run_suffix=args.suffix,
        use_best_checkpoint=args.best, continue_train=args.continue_train)
