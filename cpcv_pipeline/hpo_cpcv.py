"""
Hydra + Hypersweeper HPO Training Script with Leak-Free CPCV.

This mirrors examples/hpo_alpaca_vecenv.py but fixes the CPCV leakage bug
by using pre-sliced .npz files + dynamic env classes (no range flattening).

CV Methods:
    holdout:  Simple train/val split (fast, 1 split)
    wf:       Anchored walk-forward (expanding window, respects time order)
    cpcv:     Combinatorial Purged K-Fold CV (López de Prado 2018)

Data Layout:
    [──────── TRAIN+VAL (HPO pool) ────────][── TEST (held-out) ──]
    The HPO pool is further split by the chosen CV method.
    TEST is NEVER seen during HPO — only for final OOS evaluation.

Usage:
    # Single run (test config)
    python cpcv_pipeline/hpo_cpcv.py --config-name=cpcv_ppo

    # HPO sweep with SMAC
    python cpcv_pipeline/hpo_cpcv.py -m --config-name=cpcv_ppo

    # CPCV mode (default)
    python cpcv_pipeline/hpo_cpcv.py cv_method=cpcv n_groups=6 n_test_groups=2

    # Walk-forward mode
    python cpcv_pipeline/hpo_cpcv.py cv_method=wf n_folds=3

References:
    - López de Prado (2018): "Advances in Financial Machine Learning" — CPCV
    - Bailey et al. (2014): "The Probability of Backtest Overfitting" — PBO
    - Schneider et al. (2024): arXiv:2405.15393 — HPO resampling strategies
"""
import os
import sys
import random
import shutil
import numpy as np
import torch as th
from typing import Tuple, Optional, List
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root
_SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

from elegantrl import Config
from elegantrl import train_agent
from elegantrl.agents import AgentPPO, AgentA2C
from elegantrl.agents import AgentSAC, AgentModSAC, AgentTD3, AgentDDPG
from elegantrl.envs.StockTradingEnv import StockTradingVecEnv

from cpcv_pipeline.function_CPCV import (
    CombPurgedKFoldCV,
    AdaptiveCombPurgedKFoldCV,
    BaggedCombPurgedKFoldCV,
    verify_no_leakage,
    compute_external_feature,
    FEATURE_CHOICES,
)
from cpcv_pipeline.function_train_test import (
    load_full_data,
    save_sliced_data,
)
from elegantrl.envs.StockTradingEnv import StockTradingVecEnv

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

DATA_CACHE_DIR = _SCRIPT_DIR / "datasets"
ALPACA_NPZ_PATH = DATA_CACHE_DIR / "alpaca_stock_data.numpy.npz"


# =============================================================================
# SEED MANAGEMENT
# =============================================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


# =============================================================================
# CV SPLIT GENERATORS (returning INDEX ARRAYS, not ranges)
# =============================================================================

def get_holdout_index_splits(
    total_days: int,
    train_ratio: float = 0.7,
    gap_days: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Simple holdout: single (train_indices, val_indices) pair."""
    train_end = int(total_days * train_ratio)
    val_start = train_end + gap_days
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(val_start, total_days)
    return [(train_idx, val_idx)]


def get_wf_index_splits(
    total_days: int,
    n_folds: int = 3,
    val_ratio: float = 0.2,
    gap_days: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Anchored walk-forward: expanding window, returns index arrays."""
    splits = []
    fold_size = total_days // n_folds
    for fold in range(n_folds):
        val_end = (fold + 1) * fold_size
        val_days = int(fold_size * val_ratio)
        val_start = val_end - val_days
        train_end = val_start - gap_days
        if train_end > 0 and val_end > val_start:
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))
    return splits


def get_cpcv_index_splits(
    total_days: int,
    n_groups: int = 5,
    n_test_groups: int = 2,
    embargo_days: int = 7,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    CPCV splits returning proper INDEX ARRAYS.

    This is the leak-free version: no range flattening.
    Uses CombPurgedKFoldCV.split() which yields actual np.ndarray indices.
    """
    cv = CombPurgedKFoldCV(
        n_splits=n_groups,
        n_test_splits=n_test_groups,
        embargo_days=embargo_days,
    )

    # Verify no leakage before returning
    verify_no_leakage(cv, total_days)

    return list(cv.split(total_days))


def get_acpcv_index_splits(
    total_days: int,
    n_groups: int = 5,
    n_test_groups: int = 2,
    embargo_days: int = 7,
    external_feature: np.ndarray = None,
    n_subsplits: int = 3,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Adaptive CPCV splits with feature-aware group boundaries.

    Shifts group boundaries based on an external feature (e.g. rolling
    volatility, drawdown) to avoid splitting at regime transitions.
    """
    if external_feature is None:
        raise ValueError(
            "Adaptive CPCV requires an external_feature array. "
            "Set acpcv_feature in config (options: volatility, drawdown)."
        )
    cv = AdaptiveCombPurgedKFoldCV(
        n_splits=n_groups,
        n_test_splits=n_test_groups,
        embargo_days=embargo_days,
        external_feature=external_feature,
        n_subsplits=n_subsplits,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    verify_no_leakage(cv, total_days)
    return list(cv.split(total_days))


def get_bcpcv_index_splits(
    total_days: int,
    n_groups: int = 5,
    n_test_groups: int = 2,
    embargo_days: int = 7,
    n_bags: int = 5,
    base_seed: int = 1943,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """
    Bagged CPCV: same splits as standard CPCV, plus bag seed list.

    Returns the same splits (identical to standard CPCV) — the "bagging"
    is done by training each split multiple times with different seeds.
    The caller is responsible for the multi-seed training loop.

    Returns
    -------
    splits : list of (train_idx, val_idx)
    bag_seeds : list of int
    """
    cv = BaggedCombPurgedKFoldCV(
        n_splits=n_groups,
        n_test_splits=n_test_groups,
        embargo_days=embargo_days,
        n_bags=n_bags,
        base_seed=base_seed,
    )
    verify_no_leakage(cv, total_days)
    return list(cv.split(total_days)), cv.bag_seeds()


def get_cv_splits(
    total_days: int,
    cv_method: str,
    n_folds: int = 3,
    n_groups: int = 5,
    n_test_groups: int = 2,
    gap_days: int = 0,
    embargo_days: int = 7,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    external_feature: np.ndarray = None,
    n_subsplits: int = 3,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    n_bags: int = 5,
    base_seed: int = 1943,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Unified CV split generator.

    ALL methods return List[Tuple[np.ndarray, np.ndarray]] — train/val
    index arrays, never (start, end) range tuples.

    CV methods:
        holdout : Simple train/val split
        wf      : Anchored walk-forward
        cpcv    : Combinatorial Purged K-Fold CV
        acpcv   : Adaptive CPCV (feature-aware boundaries)
        bcpcv   : Bagged CPCV (multi-seed; returns same splits as cpcv)
    """
    if cv_method == 'holdout':
        splits = get_holdout_index_splits(total_days, train_ratio, gap_days)
    elif cv_method == 'wf':
        splits = get_wf_index_splits(total_days, n_folds, val_ratio, gap_days)
    elif cv_method == 'cpcv':
        splits = get_cpcv_index_splits(
            total_days, n_groups, n_test_groups, embargo_days
        )
    elif cv_method == 'acpcv':
        splits = get_acpcv_index_splits(
            total_days, n_groups, n_test_groups, embargo_days,
            external_feature=external_feature,
            n_subsplits=n_subsplits,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
    elif cv_method == 'bcpcv':
        splits, bag_seeds = get_bcpcv_index_splits(
            total_days, n_groups, n_test_groups, embargo_days,
            n_bags=n_bags, base_seed=base_seed,
        )
        print(f"\n🎒 Bagged CPCV: {n_bags} bags per split "
              f"(seeds: {bag_seeds})")
    else:
        raise ValueError(
            f"Unknown cv_method '{cv_method}'. "
            f"Choose: holdout, wf, cpcv, acpcv, bcpcv"
        )

    print(f"\n📊 CV Method: {cv_method.upper()} ({len(splits)} splits)")
    for i, (train_idx, val_idx) in enumerate(splits[:5]):
        overlap = np.intersect1d(train_idx, val_idx)
        leak = f" ⚠ LEAK={len(overlap)}" if len(overlap) > 0 else " ✓"
        print(f"   Split {i}: Train {len(train_idx)}d  →  Val {len(val_idx)}d{leak}")
    if len(splits) > 5:
        print(f"   ... and {len(splits) - 5} more splits")

    return splits


# =============================================================================
# EVALUATE AGENT (same logic as hpo_alpaca_vecenv.py)
# =============================================================================

def evaluate_agent(
    actor,
    env_class,
    env_args: dict,
    device: str = 'cuda:0',
    vec_normalize_path: Optional[str] = None,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Evaluate agent and compute Sharpe ratio.

    Returns:
        sharpe_ratio, mean_return, std_return, daily_returns
    """
    env = env_class(**env_args)
    initial_amount = env_args.get('initial_amount', 1e6)

    if vec_normalize_path and os.path.exists(vec_normalize_path):
        from elegantrl.envs.vec_normalize import VecNormalize
        env = VecNormalize(env, training=False, norm_reward=False)
        env.load(vec_normalize_path)
        # load() now restores saved flags; override for eval safety
        env.training = False
        env.norm_reward = False

    env.if_random_reset = False

    state, _ = env.reset()
    account_values = [initial_amount]

    for t in range(env.max_step):
        with th.no_grad():
            action = actor(state.to(device))
        state, reward, terminal, truncate, info = env.step(action)

        if hasattr(env, 'total_asset'):
            if t < env.max_step - 1:
                account_values.append(env.total_asset[0].cpu().item())
            else:
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

    account_values = np.array(account_values)
    daily_returns = np.diff(account_values) / account_values[:-1]

    if daily_returns.std() > 1e-8:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    if hasattr(env, 'cumulative_returns') and env.cumulative_returns is not None:
        returns = env.cumulative_returns
        if hasattr(returns, 'cpu'):
            returns = returns.cpu().numpy()
        elif isinstance(returns, list):
            returns = np.array(returns)
        mean_return = float(np.mean(returns)) - 100.0
        std_return = float(np.std(returns))
    else:
        mean_return = (account_values[-1] / initial_amount - 1) * 100
        std_return = 0.0

    return sharpe, mean_return, std_return, daily_returns


def compute_equal_weight_sharpe(
    close_ary: np.ndarray,
    indices: np.ndarray,
) -> float:
    """
    Buy-and-hold equal-weight benchmark Sharpe (index-based, not range-based).
    """
    prices = close_ary[np.sort(indices)]
    if len(prices) < 2:
        return 0.0
    daily_stock_returns = np.diff(prices, axis=0) / prices[:-1]
    daily_portfolio_returns = daily_stock_returns.mean(axis=1)
    if daily_portfolio_returns.std() > 1e-8:
        sharpe = daily_portfolio_returns.mean() / daily_portfolio_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    return float(sharpe)


# =============================================================================
# MAIN HPO TRAINING FUNCTION
# =============================================================================

@hydra.main(config_path="configs", config_name="cpcv_ppo", version_base="1.1")
def train_and_evaluate(cfg: DictConfig) -> float:
    """
    Hydra-compatible training function for Hypersweeper.

    For each HPO trial, trains and evaluates across ALL CV folds,
    then returns the average excess Sharpe (agent - benchmark).

    KEY DIFFERENCE from hpo_alpaca_vecenv.py:
    - Uses pre-sliced .npz + dynamic env classes for CPCV
    - No range flattening — disjoint train sets stay disjoint
    - Agent sees non-contiguous days as contiguous (accepted approach)

    Returns:
        float: Negative excess Sharpe ratio (Hypersweeper minimizes)
    """
    # === CONFIGURATION ===
    agent_name = cfg.get('agent', 'ppo').lower()
    gpu_id = cfg.get('gpu_id', 0)
    seed = cfg.get('seed', 42)

    cv_method = cfg.get('cv_method', 'cpcv')
    n_folds = cfg.get('n_folds', 3)
    n_groups = cfg.get('n_groups', 5)
    n_test_groups = cfg.get('n_test_groups', 2)
    gap_days = cfg.get('gap_days', 7)
    embargo_days = cfg.get('embargo_days', 7)
    test_ratio = cfg.get('test_ratio', 0.2)

    # Adaptive CPCV parameters
    acpcv_feature = cfg.get('acpcv_feature', 'drawdown')  # volatility | drawdown
    acpcv_window = cfg.get('acpcv_window', 63)
    n_subsplits = cfg.get('n_subsplits', 3)
    lower_quantile = cfg.get('lower_quantile', 0.25)
    upper_quantile = cfg.get('upper_quantile', 0.75)

    # Bagged CPCV parameters
    n_bags = cfg.get('n_bags', 5)
    base_seed = cfg.get('base_seed', 1943)

    set_all_seeds(seed)

    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'")
    agent_class = AGENT_REGISTRY[agent_name]
    is_off_policy = agent_name in OFF_POLICY_AGENTS

    # === LOAD DATA ===
    close_ary, tech_ary = load_full_data()
    num_days_total = close_ary.shape[0]
    num_stocks = close_ary.shape[1]

    # === HELD-OUT TEST PERIOD ===
    hpo_pool_end = int(num_days_total * (1.0 - test_ratio))
    hpo_close = close_ary[:hpo_pool_end]
    hpo_tech = tech_ary[:hpo_pool_end]
    num_days_hpo = hpo_pool_end

    print(f"\n📊 Data Layout:")
    print(f"   Total days: {num_days_total}")
    print(f"   HPO pool:   [0:{hpo_pool_end}] ({num_days_hpo} days)")
    print(f"   Test (held-out): [{hpo_pool_end}:{num_days_total}] "
          f"({num_days_total - hpo_pool_end} days)")

    amount_dim = 1
    state_dim = num_stocks + close_ary.shape[1] + tech_ary.shape[1] + amount_dim
    action_dim = num_stocks

    # === GENERATE CV SPLITS (index arrays) ===
    # Compute external feature for Adaptive CPCV if needed
    external_feature = None
    if cv_method == 'acpcv':
        external_feature = compute_external_feature(
            hpo_close, feature_name=acpcv_feature, window=acpcv_window,
            tech_ary=hpo_tech,
        )
        print(f"   A-CPCV feature: {acpcv_feature} (window={acpcv_window})")

    splits = get_cv_splits(
        total_days=num_days_hpo,
        cv_method=cv_method,
        n_folds=n_folds,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        gap_days=gap_days,
        embargo_days=embargo_days,
        external_feature=external_feature,
        n_subsplits=n_subsplits,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_bags=n_bags,
        base_seed=base_seed,
    )
    n_splits = len(splits)

    # For Bagged CPCV, get the bag seeds for multi-seed training
    bag_seeds_list = None
    if cv_method == 'bcpcv':
        _, bag_seeds_list = get_bcpcv_index_splits(
            num_days_hpo, n_groups, n_test_groups, embargo_days,
            n_bags=n_bags, base_seed=base_seed,
        )

    # === HYPERPARAMETERS FROM CONFIG ===
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

    learning_rate = cfg.get('learning_rate', 3e-4)
    gamma = cfg.get('gamma', 0.99)
    batch_size = cfg.get('batch_size', 512)
    repeat_times = cfg.get('repeat_times', 16)
    clip_grad_norm = cfg.get('clip_grad_norm', 3.0)

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

    use_vec_normalize = cfg.get('use_vec_normalize', False)
    norm_obs = cfg.get('norm_obs', True)
    norm_reward = cfg.get('norm_reward', False)

    vec_normalize_kwargs = {
        'norm_obs': norm_obs,
        'norm_reward': norm_reward,
        'clip_obs': 10.0,
        'clip_reward': None,
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

    sharpe_agent_list = []
    sharpe_bench_list = []

    print(f"\n{'='*60}")
    print(f"HPO Trial: {agent_name.upper()} | Seed: {seed} | "
          f"CV: {cv_method} ({n_splits} splits)")
    print(f"{'='*60}")
    print(f"| net_dims: {net_dims}, lr: {learning_rate}, gamma: {gamma}")
    print(f"| break_step: {break_step:,}, num_envs: {num_envs}")

    # Pre-slice data directory
    split_data_dir = Path(DATA_CACHE_DIR) / "hpo_splits"
    split_data_dir.mkdir(parents=True, exist_ok=True)

    for split_idx, (train_indices, val_indices) in enumerate(splits):
        n_train = len(train_indices)
        n_val = len(val_indices)
        max_step = n_val - 1

        if max_step < 2:
            print(f"  Split {split_idx}: skipping (val too short: {n_val} days)")
            continue

        # Double-check no leakage
        overlap = np.intersect1d(train_indices, val_indices)
        assert len(overlap) == 0, (
            f"LEAKAGE in split {split_idx}: {len(overlap)} overlapping indices!"
        )

        # For Bagged CPCV, iterate over bag seeds; otherwise single pass
        seeds_for_split = bag_seeds_list if cv_method == 'bcpcv' else [seed + split_idx]
        bag_sharpes = []

        for bag_seed in seeds_for_split:
            bag_tag = f" bag_seed={bag_seed}" if cv_method == 'bcpcv' else ""
            print(f"\n  Split {split_idx}/{n_splits}: "
                  f"Train {n_train}d → Val {n_val}d  ✓ no leak{bag_tag}")

            # ── Pre-slice and save to .npz ──────────────────────────────
            train_npz = str(split_data_dir / f"hpo_s{bag_seed}_split{split_idx}_train.npz")
            val_npz = str(split_data_dir / f"hpo_s{bag_seed}_split{split_idx}_val.npz")

            save_sliced_data(train_indices, train_npz, hpo_close, hpo_tech)
            save_sliced_data(val_indices, val_npz, hpo_close, hpo_tech)

            if is_off_policy:
                horizon_len = cfg.get('horizon_len', 256)
            else:
                horizon_len = n_train - 1

            # ── Build Config ────────────────────────────────────────────
            env_args = {
                'env_name': 'AlpacaStockVecEnv-HPO',
                'num_envs': num_envs,
                'max_step': n_train - 1,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': False,
                'gamma': gamma,
                'beg_idx': 0,          # Pre-sliced: always 0
                'end_idx': n_train,    # Pre-sliced: always full length
                'npz_path': train_npz, # forkserver-safe: plain str in env_args
                'gpu_id': gpu_id,
                'use_vec_normalize': use_vec_normalize,
                'vec_normalize_kwargs': vec_normalize_kwargs,
            }

            args = Config(agent_class, StockTradingVecEnv, env_args)
            args.gpu_id = gpu_id
            args.random_seed = bag_seed
            args.break_step = break_step
            args.net_dims = net_dims
            args.gamma = gamma
            args.horizon_len = horizon_len
            args.repeat_times = repeat_times
            args.learning_rate = learning_rate
            args.clip_grad_norm = clip_grad_norm
            args.cwd = f"./checkpoints_{agent_name}_seed{bag_seed}_split{split_idx}"
            args.if_remove = True
            args.eval_times = 8
            args.eval_per_step = int(2e4)
            args.num_workers = num_workers

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

            # Eval env on val period (also pre-sliced)
            args.eval_env_class = StockTradingVecEnv
            args.eval_env_args = {
                'env_name': 'AlpacaStockVecEnv-HPO-Val',
                'num_envs': num_envs,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': False,
                'beg_idx': 0,          # Pre-sliced
                'end_idx': n_val,      # Pre-sliced
                'npz_path': val_npz,   # forkserver-safe
                'gpu_id': gpu_id,
                'use_vec_normalize': use_vec_normalize,
                'vec_normalize_kwargs': {**vec_normalize_kwargs, 'training': False},
            }

            # --- Train ---
            _single = (gpu_id < 0)  # CPU mode: avoid forkserver detach_() issue
            try:
                train_agent(args, if_single_process=_single)
            except Exception as e:
                print(f"  Split {split_idx}: training failed: {e}")
                continue

            # --- Evaluate ---
            pt_files = sorted([
                f for f in os.listdir(args.cwd)
                if f.endswith('.pt') and f.startswith('actor')
            ])
            if not pt_files:
                print(f"  Split {split_idx}: no checkpoint found!")
                continue

            actor_path = f"{args.cwd}/{pt_files[-1]}"
            actor = th.load(actor_path, map_location=device, weights_only=False)
            actor.eval()

            val_env_args = {
                'initial_amount': 1e6, 'max_stock': 100, 'cost_pct': 1e-3,
                'gamma': gamma,
                'beg_idx': 0,          # Pre-sliced
                'end_idx': n_val,      # Pre-sliced
                'npz_path': val_npz,   # forkserver-safe
                'num_envs': num_envs,
                'gpu_id': gpu_id,
            }

            vec_normalize_path = None
            if use_vec_normalize:
                vnp = os.path.join(args.cwd, 'vec_normalize.pt')
                if os.path.exists(vnp):
                    vec_normalize_path = vnp

            sharpe_agent, mean_ret, std_ret, val_daily_returns = evaluate_agent(
                actor=actor, env_class=StockTradingVecEnv, env_args=val_env_args,
                device=device, vec_normalize_path=vec_normalize_path,
            )

            # Benchmark: equal-weight buy-and-hold on same val indices
            sharpe_bench = compute_equal_weight_sharpe(hpo_close, val_indices)

            bag_sharpes.append(sharpe_agent)

            # Save daily returns for PBO analysis
            trial_returns_dir = os.path.join(
                os.path.dirname(args.cwd), 'pbo_returns'
            )
            os.makedirs(trial_returns_dir, exist_ok=True)
            np.save(
                os.path.join(
                    trial_returns_dir,
                    f'trial_{cfg.get("trial_id", "unknown")}_split_{split_idx}_seed{bag_seed}.npy'
                ),
                val_daily_returns,
            )

            print(f"  Split {split_idx}{bag_tag}: Agent Sharpe={sharpe_agent:.4f}, "
                  f"Bench Sharpe={sharpe_bench:.4f}, "
                  f"Excess={sharpe_agent - sharpe_bench:+.4f}, "
                  f"Return={mean_ret:.2f}%")

            # --- Cleanup ---
            if cfg.get('cleanup_checkpoints', True):
                try:
                    shutil.rmtree(args.cwd)
                except Exception:
                    pass

            # Clean up temp .npz files
            for f in [train_npz, val_npz]:
                if os.path.exists(f):
                    os.remove(f)

        # --- End of bag loop ---
        # Aggregate bag results: average Sharpe across bags for this split
        if bag_sharpes:
            avg_bag_sharpe = float(np.mean(bag_sharpes))
            sharpe_agent_list.append(avg_bag_sharpe)
            sharpe_bench_list.append(sharpe_bench)
            if cv_method == 'bcpcv' and len(bag_sharpes) > 1:
                print(f"  Split {split_idx} bagged avg: {avg_bag_sharpe:.4f} "
                      f"(from {len(bag_sharpes)} bags, "
                      f"std={np.std(bag_sharpes):.4f})")

    # =========================================================================
    # AGGREGATE RESULTS
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
    print(f"| Mean Agent Sharpe:  {mean_sharpe_agent:.4f} "
          f"± {np.std(sharpe_agent_list):.4f}")
    print(f"| Mean Bench Sharpe:  {mean_sharpe_bench:.4f} "
          f"± {np.std(sharpe_bench_list):.4f}")
    print(f"| Excess Sharpe:      {excess_sharpe:+.4f}")
    print(f"| Objective (neg):    {-excess_sharpe:.4f}")

    # Return NEGATIVE excess Sharpe (Hypersweeper minimizes by default)
    return -excess_sharpe


if __name__ == "__main__":
    train_and_evaluate()
