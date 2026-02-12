"""
Hypersweeper HPO Training Script for ElegantRL Stock Trading

This is an HPO-compatible version of demo_FinRL_Alpaca_VecEnv.py with:
- Hydra configuration for Hypersweeper integration
- Multiple CV strategies: holdout, anchored walk-forward, CPCV
- Proper held-out test period separation
- Per-trial averaging across all CV folds (not one-fold-per-trial)
- Returns validation Sharpe ratio for HPO optimization

CV Methods:
    holdout:  Simple train/val split (fast, no CV)
    wf:       Anchored walk-forward (expanding window, respects time order)
    cpcv:     Combinatorial Purged K-Fold CV (López de Prado 2018)

Data Layout:
    [──────── TRAIN+VAL (HPO pool) ────────][── TEST (held-out) ──]
    The HPO pool is further split by the chosen CV method.
    TEST is NEVER seen during HPO - only for final OOS evaluation.

Usage:
    # Single run (test config)
    python examples/hpo_alpaca_vecenv.py --config-name=alpaca_ppo
    
    # HPO sweep with SMAC
    python examples/hpo_alpaca_vecenv.py -m --config-name=alpaca_ppo
    
    # CPCV mode
    python examples/hpo_alpaca_vecenv.py cv_method=cpcv n_groups=5 n_test_groups=2
    
    # Walk-forward mode
    python examples/hpo_alpaca_vecenv.py cv_method=wf n_folds=3

References:
    - López de Prado (2018): "Advances in Financial Machine Learning" - CPCV
    - Bailey et al. (2014): "The Probability of Backtest Overfitting" - PBO
    - Schneider et al. (2024): arXiv:2405.15393 - HPO resampling strategies
    - Bischl et al. (2023): HPO best practices
"""
import os
import sys
import random
import itertools
from math import comb
import numpy as np
import torch as th
from typing import Tuple, Optional, List
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for elegantrl imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from elegantrl import Config
from elegantrl import train_agent
from elegantrl.agents import AgentPPO, AgentA2C
from elegantrl.agents import AgentSAC, AgentModSAC, AgentTD3, AgentDDPG
from elegantrl.envs.StockTradingEnv import StockTradingVecEnv
from elegantrl.envs.vec_normalize import VecNormalize

# =============================================================================
# AGENT REGISTRY
# =============================================================================

AGENT_REGISTRY = {
    'ppo': AgentPPO,
    'a2c': AgentA2C,
    'sac': AgentSAC,
    'modsac': AgentModSAC,
    'td3': AgentTD3,
    'ddpg': AgentDDPG,
}

ON_POLICY_AGENTS = {'ppo', 'a2c'}
OFF_POLICY_AGENTS = {'sac', 'modsac', 'td3', 'ddpg'}

# Data paths (use absolute path since Hydra changes working directory)
_SCRIPT_DIR = Path(__file__).parent.parent.resolve()
DATA_CACHE_DIR = _SCRIPT_DIR / "datasets"
ALPACA_NPZ_PATH = DATA_CACHE_DIR / "alpaca_stock_data.numpy.npz"


# =============================================================================
# SEED MANAGEMENT
# =============================================================================

def set_all_seeds(seed: int, cuda_deterministic: bool = False):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    
    if cuda_deterministic:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


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
        Fold 0: Train [0:267]  → Val [267:333]    (267 train days)
        Fold 1: Train [0:533]  → Val [533:667]    (533 train days) 
        Fold 2: Train [0:800]  → Val [800:1000]   (800 train days)
    
    Args:
        total_days: Total number of trading days in HPO pool
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
        # Validation window for this fold
        val_end = (fold + 1) * fold_size
        val_days = int(fold_size * val_ratio)
        val_start = val_end - val_days
        
        # Training window anchored at 0, ends before gap
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
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Combinatorial Purged Cross-Validation (CPCV) splits.
    
    Based on López de Prado (2018) "Advances in Financial Machine Learning".
    Generates C(n_groups, n_test_groups) train/test combinations with:
    - Purging: removes train samples adjacent to test boundaries
    - Embargo: adds gap after each test fold boundary
    
    Example with n_groups=5, n_test_groups=2:
        10 combinations of (3 train groups, 2 test groups)
        Each combination has ~60% train, ~40% test (minus purge+embargo)
    
    Args:
        total_days: Total number of trading days in HPO pool
        n_groups: Total number of groups to divide data into (N)
        n_test_groups: Number of groups used for testing per split (K)
        embargo_pct: Fraction of total data to embargo after each test fold
    
    Returns:
        List of (train_indices_tuple, test_indices_tuple) where each is
        a tuple of (start, end) pairs for the contiguous blocks.
        
    Note:
        Number of splits = C(n_groups, n_test_groups)
        For n_groups=5, n_test_groups=2: 10 splits
        For n_groups=6, n_test_groups=2: 15 splits
    """
    if n_test_groups >= n_groups:
        raise ValueError(f"n_test_groups ({n_test_groups}) must be < n_groups ({n_groups})")
    
    # Divide into n_groups equal folds
    fold_size = total_days // n_groups
    fold_bounds = []
    for i in range(n_groups):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_groups - 1 else total_days
        fold_bounds.append((start, end))
    
    embargo_days = max(1, int(total_days * embargo_pct))
    
    # Generate all C(N, K) test-fold combinations
    all_test_combos = list(itertools.combinations(range(n_groups), n_test_groups))
    
    splits = []
    for test_group_ids in all_test_combos:
        train_group_ids = [i for i in range(n_groups) if i not in test_group_ids]
        
        # Build test index ranges
        test_ranges = [fold_bounds[i] for i in test_group_ids]
        
        # Build train index ranges with purging + embargo
        # Remove train samples that are within embargo_days of any test boundary
        train_ranges = []
        for train_id in train_group_ids:
            t_start, t_end = fold_bounds[train_id]
            
            # Purge: remove train samples near test boundaries
            for test_id in test_group_ids:
                test_start, test_end = fold_bounds[test_id]
                
                # If train fold is right before a test fold, clip the end
                if t_end <= test_start and t_end > test_start - embargo_days:
                    t_end = max(t_start, test_start - embargo_days)
                
                # If train fold is right after a test fold, clip the start
                if t_start >= test_end and t_start < test_end + embargo_days:
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
    
    Args:
        total_days: Total trading days in the HPO pool (excludes held-out test)
        cv_method: 'holdout', 'wf' (anchored walk-forward), or 'cpcv'
        Other args: Method-specific parameters
    
    Returns:
        dict with:
            'splits': list of (train_ranges, val_ranges) 
            'method': cv method name
            'n_splits': number of splits
    """
    if cv_method == 'holdout':
        raw_splits = get_holdout_splits(total_days, train_ratio, gap_days)
        # Convert to standard format: list of (train_ranges, val_ranges)
        splits = [(((s[0][0], s[0][1]),), ((s[1][0], s[1][1]),)) for s in raw_splits]
        
    elif cv_method == 'wf':
        raw_splits = get_anchored_walk_forward_splits(total_days, n_folds, val_ratio, gap_days)
        splits = [(((s[0][0], s[0][1]),), ((s[1][0], s[1][1]),)) for s in raw_splits]
        
    elif cv_method == 'cpcv':
        splits = get_cpcv_splits(total_days, n_groups, n_test_groups, embargo_pct)
        
    else:
        raise ValueError(f"Unknown cv_method '{cv_method}'. Choose: holdout, wf, cpcv")
    
    n_splits = len(splits)
    print(f"\n📊 CV Method: {cv_method.upper()} ({n_splits} splits)")
    
    if cv_method == 'cpcv':
        print(f"   C({n_groups},{n_test_groups}) = {comb(n_groups, n_test_groups)} combinations")
        print(f"   Embargo: {embargo_pct*100:.1f}% of data = {int(total_days * embargo_pct)} days")
    elif cv_method == 'wf':
        print(f"   {n_folds} anchored walk-forward folds")
        if gap_days > 0:
            print(f"   Gap (purge): {gap_days} days")
    
    for i, (train_r, val_r) in enumerate(splits[:5]):  # Show first 5
        train_days = sum(e - s for s, e in train_r)
        val_days = sum(e - s for s, e in val_r)
        print(f"   Split {i}: Train {train_days}d {train_r} → Val {val_days}d {val_r}")
    if n_splits > 5:
        print(f"   ... and {n_splits - 5} more splits")
    
    return {
        'splits': splits,
        'method': cv_method,
        'n_splits': n_splits,
    }


# =============================================================================
# VECENV FOR ALPACA DATA
# =============================================================================

class AlpacaStockVecEnv(StockTradingVecEnv):
    """StockTradingVecEnv adapted for Alpaca/FinRL data."""
    
    def load_data_from_disk(self, tech_id_list=None):
        """Load pre-processed Alpaca data from npz file."""
        if os.path.exists(ALPACA_NPZ_PATH):
            ary_dict = np.load(ALPACA_NPZ_PATH, allow_pickle=True)
            return ary_dict['close_ary'], ary_dict['tech_ary']
        else:
            raise FileNotFoundError(
                f"Alpaca data not found at {ALPACA_NPZ_PATH}. "
                f"Run demo_FinRL_Alpaca_VecEnv.py first to download data."
            )


# =============================================================================
# VALIDATION / SHARPE CALCULATION
# =============================================================================

def evaluate_agent(
    actor,
    env_class,
    env_args: dict,
    num_episodes: int = 1,
    device: str = 'cuda:0',
    vec_normalize_path: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate agent and compute Sharpe ratio.
    
    Args:
        actor: Trained actor network
        env_class: Environment class
        env_args: Environment arguments
        num_episodes: Number of evaluation episodes
        device: Device to run on
        vec_normalize_path: Path to vec_normalize.pt stats file (if training used VecNormalize)
    
    Returns:
        sharpe_ratio: Annualized Sharpe ratio (computed from actual portfolio returns)
        mean_return: Mean cumulative return (%)
        std_return: Std of returns across envs
    """
    env = env_class(**env_args)
    initial_amount = env_args.get('initial_amount', 1e6)
    
    # Load VecNormalize stats if available (critical for correct evaluation)
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during eval
        env.norm_reward = False  # Don't normalize rewards during eval
        print(f"| Loaded VecNormalize stats from {vec_normalize_path}")
    
    # Disable random reset for fair/reproducible evaluation
    env.if_random_reset = False
    
    # Run episode and track actual account values (not scaled rewards)
    state, _ = env.reset()
    account_values = [initial_amount]  # Track actual portfolio value
    
    for t in range(env.max_step):
        with th.no_grad():
            action = actor(state.to(device))
        state, reward, terminal, truncate, info = env.step(action)
        
        # Track actual total_asset for env 0 (like eval_all_checkpoints.py)
        if hasattr(env, 'total_asset'):
            if t < env.max_step - 1:
                account_values.append(env.total_asset[0].cpu().item())
            else:
                # Final step: use cumulative_returns to compute final value
                if hasattr(env, 'cumulative_returns') and env.cumulative_returns is not None:
                    cr = env.cumulative_returns
                    if hasattr(cr, 'cpu'):
                        final_value = initial_amount * cr[0].cpu().item() / 100
                    elif isinstance(cr, (list, np.ndarray)):
                        final_value = initial_amount * float(cr[0]) / 100
                    else:
                        final_value = initial_amount * float(cr) / 100
                    account_values.append(final_value)
                else:
                    account_values.append(account_values[-1])
    
    # Convert to numpy for calculations
    account_values = np.array(account_values)
    
    # Compute daily returns as percentage (standard financial calculation)
    daily_returns = np.diff(account_values) / account_values[:-1]
    
    # Annualized Sharpe ratio (matches eval_all_checkpoints.py)
    if daily_returns.std() > 1e-8:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Cumulative return (gain %)
    # cumulative_returns = (total_asset / initial_amount) * 100
    # So 122.31 means 122.31% of initial = 22.31% gain
    if hasattr(env, 'cumulative_returns') and env.cumulative_returns is not None:
        returns = env.cumulative_returns
        if hasattr(returns, 'cpu'):
            returns = returns.cpu().numpy()
        elif isinstance(returns, list):
            returns = np.array(returns)
        mean_return = float(np.mean(returns)) - 100.0  # Subtract 100 for gain %
        std_return = float(np.std(returns))
    else:
        mean_return = (account_values[-1] / initial_amount - 1) * 100
        std_return = 0.0
    
    return sharpe, mean_return, std_return


# =============================================================================
# BENCHMARK: Buy-and-Hold Equal-Weight Portfolio
# =============================================================================

def compute_equal_weight_sharpe(
    close_ary: np.ndarray,
    beg_idx: int,
    end_idx: int,
    initial_amount: float = 1e6,
) -> float:
    """
    Compute annualized Sharpe of a buy-and-hold equal-weight portfolio.
    
    This serves as the benchmark for the HPO objective.
    Agent Sharpe is compared against this to measure excess performance.
    
    Args:
        close_ary: Full close price array [total_days, num_stocks]
        beg_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        initial_amount: Starting capital
    
    Returns:
        Annualized Sharpe ratio of equal-weight buy-and-hold
    """
    prices = close_ary[beg_idx:end_idx]  # [days, stocks]
    if len(prices) < 2:
        return 0.0
    
    # Equal-weight portfolio: invest equally in all stocks
    # Daily portfolio return = mean of individual stock daily returns
    daily_stock_returns = np.diff(prices, axis=0) / prices[:-1]  # [days-1, stocks]
    daily_portfolio_returns = daily_stock_returns.mean(axis=1)    # [days-1]
    
    if daily_portfolio_returns.std() > 1e-8:
        sharpe = daily_portfolio_returns.mean() / daily_portfolio_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    
    return float(sharpe)


# =============================================================================
# MAIN HPO TRAINING FUNCTION
# =============================================================================

@hydra.main(config_path="configs", config_name="alpaca_ppo", version_base="1.1")
def train_and_evaluate(cfg: DictConfig) -> float:
    """
    Hydra-compatible training function for Hypersweeper.
    
    For each HPO trial, trains and evaluates across ALL CV folds,
    then returns the average excess Sharpe (agent - benchmark).
    
    This follows FinRL_Crypto's approach:
    - CPCV/WF: train on each split, evaluate, average
    - Objective = mean(sharpe_agent) - mean(sharpe_benchmark)
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        float: Negative excess Sharpe ratio (Hypersweeper minimizes)
    """
    # === CONFIGURATION ===
    agent_name = cfg.get('agent', 'ppo').lower()
    gpu_id = cfg.get('gpu_id', 0)
    seed = cfg.get('seed', 42)
    
    # CV settings
    cv_method = cfg.get('cv_method', 'holdout')  # holdout, wf, cpcv
    n_folds = cfg.get('n_folds', 3)              # for wf
    n_groups = cfg.get('n_groups', 5)             # for cpcv (N)
    n_test_groups = cfg.get('n_test_groups', 2)   # for cpcv (K)
    gap_days = cfg.get('gap_days', 0)             # purge gap (holdout/wf)
    embargo_pct = cfg.get('embargo_pct', 0.01)    # embargo fraction (cpcv)
    test_ratio = cfg.get('test_ratio', 0.2)       # held-out test fraction
    
    # === SEED MANAGEMENT ===
    set_all_seeds(seed)
    
    # === AGENT SETUP ===
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose from: {list(AGENT_REGISTRY.keys())}")
    
    agent_class = AGENT_REGISTRY[agent_name]
    is_off_policy = agent_name in OFF_POLICY_AGENTS
    
    # === LOAD DATA ===
    if not os.path.exists(ALPACA_NPZ_PATH):
        raise FileNotFoundError(
            f"Data not found at {ALPACA_NPZ_PATH}. "
            f"Run 'python examples/demo_FinRL_Alpaca_VecEnv.py' first to download."
        )
    
    data = np.load(ALPACA_NPZ_PATH)
    close_ary = data['close_ary']
    tech_ary = data['tech_ary']
    
    num_days_total = close_ary.shape[0]
    num_stocks = close_ary.shape[1]
    
    # === HELD-OUT TEST PERIOD ===
    # Reserve final test_ratio of data for true OOS evaluation.
    # HPO only sees the train+val pool.
    hpo_pool_end = int(num_days_total * (1.0 - test_ratio))
    num_days_hpo = hpo_pool_end  # HPO pool: [0, hpo_pool_end)
    # test_start = hpo_pool_end   # Test:     [hpo_pool_end, num_days_total)
    
    print(f"\n📊 Data Layout:")
    print(f"   Total days: {num_days_total}")
    print(f"   HPO pool:   [0:{hpo_pool_end}] ({hpo_pool_end} days)")
    print(f"   Test (held-out): [{hpo_pool_end}:{num_days_total}] ({num_days_total - hpo_pool_end} days)")
    
    # Calculate dimensions
    amount_dim = 1
    state_dim = num_stocks + close_ary.shape[1] + tech_ary.shape[1] + amount_dim
    action_dim = num_stocks
    
    # === GENERATE CV SPLITS (within HPO pool only) ===
    cv_info = get_cv_splits(
        total_days=num_days_hpo,
        cv_method=cv_method,
        n_folds=n_folds,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        gap_days=gap_days,
        embargo_pct=embargo_pct,
    )
    splits = cv_info['splits']
    n_splits = cv_info['n_splits']
    
    # === HYPERPARAMETERS FROM CONFIG ===
    # Network architecture (categorical or per-layer)
    net_arch = cfg.get('net_arch', None)
    if net_arch:
        net_dims = {
            "small": [64, 64],
            "medium": [256, 128],
            "large": [512, 256],
            "big": [400, 300],
        }.get(net_arch, [256, 128])
    else:
        net_dims_0 = cfg.get('net_dims_0', 256)
        net_dims_1 = cfg.get('net_dims_1', 128)
        net_dims = [net_dims_0, net_dims_1]
    
    # Core hyperparameters
    learning_rate = cfg.get('learning_rate', 3e-4)
    gamma = cfg.get('gamma', 0.99)
    batch_size = cfg.get('batch_size', 512)
    repeat_times = cfg.get('repeat_times', 16)
    clip_grad_norm = cfg.get('clip_grad_norm', 3.0)
    
    # Agent-specific hyperparameters
    if is_off_policy:
        num_envs = cfg.get('num_envs', 96)
        buffer_size = cfg.get('buffer_size', int(1e5))
        soft_update_tau = cfg.get('soft_update_tau', 5e-3)
        explore_noise_std = cfg.get('explore_noise_std', 0.1)
        policy_noise_std = cfg.get('policy_noise_std', 0.2)
        update_freq = cfg.get('update_freq', 2)
        num_ensembles = cfg.get('num_ensembles', 4)
        critic_tau = cfg.get('critic_tau', 0.995)
    else:
        num_envs = cfg.get('num_envs', 2048)
        ratio_clip = cfg.get('ratio_clip', 0.25)
        lambda_gae_adv = cfg.get('lambda_gae_adv', 0.95)
        lambda_entropy = cfg.get('lambda_entropy', 0.01)
        if_use_v_trace = cfg.get('if_use_v_trace', True)
    
    # VecNormalize settings
    # On-policy: norm_reward=True (data consumed once, safe)
    # Off-policy: norm_reward=False (stale replay buffer causes critic divergence)
    use_vec_normalize = cfg.get('use_vec_normalize', True)
    norm_obs = cfg.get('norm_obs', True)
    norm_reward = cfg.get('norm_reward', not is_off_policy)
    
    vec_normalize_kwargs = {
        'norm_obs': norm_obs,
        'norm_reward': norm_reward,
        'clip_obs': 10.0,
        'clip_reward': 10.0 if norm_reward else None,
        'gamma': gamma,
        'training': True,
    }
    
    break_step = cfg.get('break_step', int(5e5))
    device = f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu'
    cpu_count = os.cpu_count() or 8
    num_workers = min(cpu_count // 4, 4) if not is_off_policy else min(cpu_count // 2, 6)
    
    # =========================================================================
    # TRAIN + EVALUATE ACROSS ALL CV SPLITS
    # =========================================================================
    # Following FinRL_Crypto: each HPO trial trains on EVERY split and averages.
    # Objective = mean(sharpe_agent) - mean(sharpe_benchmark) across all splits.
    
    sharpe_agent_list = []
    sharpe_bench_list = []
    
    print(f"\n{'='*60}")
    print(f"HPO Trial: {agent_name.upper()} | Seed: {seed} | CV: {cv_method} ({n_splits} splits)")
    print(f"{'='*60}")
    print(f"| net_dims: {net_dims}, lr: {learning_rate}, gamma: {gamma}")
    print(f"| break_step: {break_step:,}, num_envs: {num_envs}")
    
    for split_idx, (train_ranges, val_ranges) in enumerate(splits):
        # Get first train range start and last val range end for this split
        # CPCV may have multiple train/val ranges per split
        train_start = train_ranges[0][0]
        train_end = train_ranges[-1][1]
        val_start = val_ranges[0][0]
        val_end = val_ranges[-1][1]
        
        # For CPCV with non-contiguous train blocks, use the full range
        # (the env loads contiguous data, purging happens at split level)
        # For simplicity, use first_train_start to last_train_end as train range
        # and first_val_start to last_val_end as val range
        max_step = val_end - val_start - 1
        if max_step < 2:
            print(f"  Split {split_idx}: skipping (val too short: {max_step+1} days)")
            continue
        
        # horizon_len: on-policy uses full episode, off-policy uses config
        if is_off_policy:
            horizon_len = cfg.get('horizon_len', 256)
        else:
            horizon_len = max_step
        
        print(f"\n  Split {split_idx}/{n_splits}: Train [{train_start}:{train_end}] → Val [{val_start}:{val_end}]")
        
        # --- Build Config for this split ---
        env_args = {
            'env_name': 'AlpacaStockVecEnv-HPO',
            'num_envs': num_envs,
            'max_step': max_step,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'if_discrete': False,
            'gamma': gamma,
            'beg_idx': train_start,
            'end_idx': train_end,
            'use_vec_normalize': use_vec_normalize,
            'vec_normalize_kwargs': vec_normalize_kwargs,
        }
        
        args = Config(agent_class, AlpacaStockVecEnv, env_args)
        args.gpu_id = gpu_id
        args.random_seed = seed + split_idx  # Different seed per split
        args.break_step = break_step
        args.net_dims = net_dims
        args.gamma = gamma
        args.horizon_len = horizon_len
        args.repeat_times = repeat_times
        args.learning_rate = learning_rate
        args.clip_grad_norm = clip_grad_norm
        args.cwd = f"./checkpoints_{agent_name}_seed{seed}_split{split_idx}"
        args.if_remove = True
        args.eval_times = 8
        args.eval_per_step = int(2e4)
        args.num_workers = num_workers
        
        # Agent-specific settings
        if is_off_policy:
            args.batch_size = batch_size
            args.buffer_size = buffer_size
            args.soft_update_tau = soft_update_tau
            args.explore_noise_std = explore_noise_std
            if agent_name == 'td3':
                args.policy_noise_std = policy_noise_std
                args.update_freq = update_freq
                args.num_ensembles = num_ensembles
            elif agent_name in ('sac', 'modsac'):
                args.num_ensembles = num_ensembles
            if agent_name == 'modsac':
                args.critic_tau = critic_tau
        else:
            args.batch_size = batch_size
            args.ratio_clip = ratio_clip
            args.lambda_gae_adv = lambda_gae_adv
            args.lambda_entropy = lambda_entropy
            args.if_use_v_trace = if_use_v_trace
        
        # Eval env on val period
        args.eval_env_class = AlpacaStockVecEnv
        args.eval_env_args = {
            'env_name': 'AlpacaStockVecEnv-HPO-Val',
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
        
        # --- Train ---
        try:
            train_agent(args, if_single_process=False)
        except Exception as e:
            print(f"  Split {split_idx}: training failed: {e}")
            continue
        
        # --- Evaluate ---
        pt_files = sorted([f for f in os.listdir(args.cwd)
                         if f.endswith('.pt') and f.startswith('actor')])
        if not pt_files:
            print(f"  Split {split_idx}: no checkpoint found!")
            continue
        
        actor_path = f"{args.cwd}/{pt_files[-1]}"
        actor = th.load(actor_path, map_location=device, weights_only=False)
        actor.eval()
        
        val_env_args = {
            'initial_amount': 1e6, 'max_stock': 100, 'cost_pct': 1e-3,
            'gamma': gamma, 'beg_idx': val_start, 'end_idx': val_end,
            'num_envs': num_envs, 'gpu_id': gpu_id,
        }
        
        vec_normalize_path = None
        if use_vec_normalize:
            vnp = os.path.join(args.cwd, 'vec_normalize.pt')
            if os.path.exists(vnp):
                vec_normalize_path = vnp
        
        sharpe_agent, mean_ret, std_ret = evaluate_agent(
            actor=actor, env_class=AlpacaStockVecEnv, env_args=val_env_args,
            device=device, vec_normalize_path=vec_normalize_path,
        )
        
        # Benchmark: equal-weight buy-and-hold on same val period
        sharpe_bench = compute_equal_weight_sharpe(close_ary, val_start, val_end)
        
        sharpe_agent_list.append(sharpe_agent)
        sharpe_bench_list.append(sharpe_bench)
        
        print(f"  Split {split_idx}: Agent Sharpe={sharpe_agent:.4f}, Bench Sharpe={sharpe_bench:.4f}, "
              f"Excess={sharpe_agent - sharpe_bench:+.4f}, Return={mean_ret:.2f}%")
        
        # --- Cleanup ---
        if cfg.get('cleanup_checkpoints', True):
            import shutil
            try:
                shutil.rmtree(args.cwd)
            except:
                pass
    
    # =========================================================================
    # AGGREGATE RESULTS ACROSS FOLDS
    # =========================================================================
    if not sharpe_agent_list:
        print("| ERROR: No successful splits!")
        return 0.0
    
    mean_sharpe_agent = np.mean(sharpe_agent_list)
    mean_sharpe_bench = np.mean(sharpe_bench_list)
    excess_sharpe = mean_sharpe_agent - mean_sharpe_bench
    
    print(f"\n{'='*60}")
    print(f"HPO TRIAL RESULTS ({len(sharpe_agent_list)}/{n_splits} splits)")
    print(f"{'='*60}")
    print(f"| Mean Agent Sharpe:  {mean_sharpe_agent:.4f} ± {np.std(sharpe_agent_list):.4f}")
    print(f"| Mean Bench Sharpe:  {mean_sharpe_bench:.4f} ± {np.std(sharpe_bench_list):.4f}")
    print(f"| Excess Sharpe:      {excess_sharpe:+.4f}")
    print(f"| Objective (neg):    {-excess_sharpe:.4f}")
    
    # Return NEGATIVE excess Sharpe (Hypersweeper minimizes by default)
    return -excess_sharpe
