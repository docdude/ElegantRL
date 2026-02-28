"""
Hydra + Hypersweeper HPO Training Script with Leak-Free CPCV.

Delegates training to ``function_train_test.train_split()`` — the same
function used by optimize_cpcv.py and optimize_adapt_cpcv.py — so GPU
memory scaling, VecNormalize, and multi-worker logic are identical.

CV Methods:
    holdout:  Simple train/val split (fast, 1 split)
    wf:       Anchored walk-forward (expanding window, respects time order)
    cpcv:     Combinatorial Purged K-Fold CV (López de Prado 2018)
    acpcv:    Adaptive CPCV (feature-aware boundaries)
    bcpcv:    Bagged CPCV (multi-seed)

Data Layout:
    [──────── TRAIN+VAL (HPO pool) ────────][── TEST (held-out) ──]
    The HPO pool is further split by the chosen CV method.
    TEST is NEVER seen during HPO — only for final OOS evaluation.

Usage:
    # Single run (test config)
    python cpcv_pipeline/hpo_cpcv.py --config-name=cpcv_ppo

    # HPO sweep with SMAC
    python cpcv_pipeline/hpo_cpcv.py -m --config-name=cpcv_ppo

    # ACPCV with RSI feature
    python cpcv_pipeline/hpo_cpcv.py --config-name=cpcv_ppo_acpcv

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
from typing import Tuple, List
from pathlib import Path

import json
import time as _time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

try:
    import wandb
except ImportError:
    wandb = None

# Add project root
_SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

from cpcv_pipeline.function_CPCV import (
    CombPurgedKFoldCV,
    AdaptiveCombPurgedKFoldCV,
    BaggedCombPurgedKFoldCV,
    verify_no_leakage,
    compute_external_feature,
)
from cpcv_pipeline.function_train_test import (
    load_full_data,
    train_split,
    evaluate_agent_on_indices,
)

# =============================================================================
# AGENT REGISTRY
# =============================================================================

ON_POLICY_AGENTS = {'ppo', 'a2c'}
OFF_POLICY_AGENTS = {'sac', 'modsac', 'td3', 'ddpg'}


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

# Weight for alpha (excess return) in composite objective: SR + w * alpha.
# Calibrated so Sharpe dominates but alpha breaks ties among similar-SR configs.
# Analysis on 261 checkpoints across 10 ACPCV splits showed 99.9% return
# capture and 9/10 agreement with best-return checkpoint.
ALPHA_WEIGHT = 0.1


def compute_equal_weight_benchmark(
    close_ary: np.ndarray,
    indices: np.ndarray,
) -> dict:
    """
    Buy-and-hold equal-weight benchmark stats (index-based).

    Returns
    -------
    dict with keys:
        sharpe       : annualised Sharpe ratio
        total_return : total return as fraction (e.g. 0.15 = 15%)
    """
    prices = close_ary[np.sort(indices)]
    if len(prices) < 2:
        return {'sharpe': 0.0, 'total_return': 0.0}
    daily_stock_returns = np.diff(prices, axis=0) / prices[:-1]
    daily_portfolio_returns = daily_stock_returns.mean(axis=1)
    total_return = float(np.prod(1 + daily_portfolio_returns) - 1)
    if daily_portfolio_returns.std() > 1e-8:
        sharpe = daily_portfolio_returns.mean() / daily_portfolio_returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    return {'sharpe': float(sharpe), 'total_return': total_return}


# =============================================================================
# MAIN HPO TRAINING FUNCTION
# =============================================================================

@hydra.main(config_path="configs", config_name="cpcv_ppo", version_base="1.1")
def train_and_evaluate(cfg: DictConfig) -> float:
    """
    Hydra-compatible training function for Hypersweeper.

    For each HPO trial, delegates training to ``train_split()`` — the same
    function used by optimize_cpcv.py and optimize_adapt_cpcv.py. This ensures
    GPU memory scaling, VecNormalize, and multi-worker logic are identical.

    Returns:
        float: Negative composite objective (Hypersweeper minimizes).
               Composite = SR + 0.1 * alpha, where alpha = agent_return - bench_return.
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
    acpcv_feature = cfg.get('acpcv_feature', 'drawdown')
    acpcv_window = cfg.get('acpcv_window', 63)
    n_subsplits = cfg.get('n_subsplits', 3)
    lower_quantile = cfg.get('lower_quantile', 0.25)
    upper_quantile = cfg.get('upper_quantile', 0.75)

    # Bagged CPCV parameters
    n_bags = cfg.get('n_bags', 5)
    base_seed = cfg.get('base_seed', 1943)

    set_all_seeds(seed)

    is_off_policy = agent_name in OFF_POLICY_AGENTS

    # === LOAD DATA ===
    close_ary, tech_ary = load_full_data()
    num_days_total = close_ary.shape[0]

    # === HELD-OUT TEST PERIOD ===
    hpo_pool_end = int(num_days_total * (1.0 - test_ratio))
    hpo_close = close_ary[:hpo_pool_end]
    hpo_tech = tech_ary[:hpo_pool_end]
    num_days_hpo = hpo_pool_end

    print(f"\n{'='*60}")
    print(f"HPO Data Layout")
    print(f"{'='*60}")
    print(f"   Total days: {num_days_total}")
    print(f"   HPO pool:   [0:{hpo_pool_end}] ({num_days_hpo} days)")
    print(f"   Test (held-out): [{hpo_pool_end}:{num_days_total}] "
          f"({num_days_total - hpo_pool_end} days)")

    # === GENERATE CV SPLITS ===
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

    # === BUILD HYPERPARAMETERS FROM CONFIG ===
    # Map Hydra config → erl_params/env_params dicts used by train_split()
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
    break_step = cfg.get('break_step', int(5e5))
    num_envs = cfg.get('num_envs', 2048)

    # Build erl_params dict (same format as DEFAULT_ERL_PARAMS in config.py)
    erl_params = {
        'net_dims': net_dims,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'batch_size': batch_size,
        'repeat_times': repeat_times,
        'clip_grad_norm': clip_grad_norm,
        'break_step': break_step,
    }

    if not is_off_policy:
        erl_params['ratio_clip'] = cfg.get('ratio_clip', 0.25)
        erl_params['lambda_gae_adv'] = cfg.get('lambda_gae_adv', 0.95)
        erl_params['lambda_entropy'] = cfg.get('lambda_entropy', 0.01)
        erl_params['if_use_v_trace'] = cfg.get('if_use_v_trace', True)
    else:
        erl_params['buffer_size'] = cfg.get('buffer_size', int(1e5))
        erl_params['soft_update_tau'] = cfg.get('soft_update_tau', 5e-3)

    env_params = {
        'num_envs': num_envs,
        'initial_amount': 1e6,
        'max_stock': 100,
        'cost_pct': 1e-3,
    }

    # VecNormalize
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

    # num_workers: passed to train_split(), which handles GPU scaling
    num_workers_cfg = cfg.get('num_workers', None)
    # None = let train_split() compute from CPU count (same as optimize scripts)

    # HPO output directory
    hpo_cwd = os.path.join(
        str(_SCRIPT_DIR / "train_results"),
        f"hpo_{cv_method}_{agent_name}_seed{seed}",
    )

    # PBO / DSR persistence: save per-split daily returns for post-hoc analysis
    pbo_returns_dir = os.path.join(hpo_cwd, "pbo_returns")
    os.makedirs(pbo_returns_dir, exist_ok=True)

    # Trial ID: Hydra multirun job number, or fallback to timestamp
    try:
        trial_id = str(HydraConfig.get().job.num)
    except Exception:
        import time
        trial_id = str(int(time.time()))

    print(f"\n{'='*60}")
    print(f"HPO Trial {trial_id}: {agent_name.upper()} | Seed: {seed} | "
          f"CV: {cv_method} ({n_splits} splits)")
    print(f"{'='*60}")
    print(f"| net_dims: {net_dims}, lr: {learning_rate}, gamma: {gamma}")
    print(f"| break_step: {break_step:,}, num_envs: {num_envs}")
    print(f"| vec_normalize: {use_vec_normalize}, "
          f"norm_obs: {norm_obs}, norm_reward: {norm_reward}")

    # =========================================================================
    # TRAIN + EVALUATE ACROSS ALL CV SPLITS
    # Uses train_split() — identical to optimize_cpcv / optimize_adapt_cpcv
    # =========================================================================

    sharpe_agent_list = []
    sharpe_bench_list = []
    alpha_list = []
    composite_list = []

    for split_idx, (train_indices, val_indices) in enumerate(splits):
        n_val = len(val_indices)
        if n_val < 3:
            print(f"  Split {split_idx}: skipping (val too short: {n_val} days)")
            continue

        # ── Call train_split() — same function used by optimize scripts ──
        result = train_split(
            split_idx=split_idx,
            train_indices=train_indices,
            test_indices=val_indices,
            close_ary=hpo_close,
            tech_ary=hpo_tech,
            model_name=agent_name,
            erl_params=erl_params,
            env_params=env_params,
            cwd_base=hpo_cwd,
            gpu_id=gpu_id,
            random_seed=seed,
            use_vec_normalize=use_vec_normalize,
            vec_normalize_kwargs=vec_normalize_kwargs,
            continue_train=False,
            num_workers=num_workers_cfg,
        )

        # ── Evaluate best checkpoint with proper OOS Sharpe ──────────────
        # avgR (from recorder) is % of initial capital — good for checkpoint
        # selection during training, but Sharpe is the proper risk-adjusted
        # metric for HPO.  We find the best checkpoint by avgR, then run a
        # deterministic eval episode on the val fold to compute Sharpe from
        # actual daily returns.  This matches eval_all_checkpoints logic.
        split_cwd = result.get('cwd', '')
        sharpe_agent = 0.0
        agent_return = 0.0
        best_avgR = result.get('best_avgR', 0.0)

        # Find the best checkpoint file (saved by evaluator when avgR improves)
        best_ckpt = None
        if split_cwd and os.path.isdir(split_cwd):
            # act.pth is always updated to the latest best actor
            act_path = os.path.join(split_cwd, 'act.pth')
            if os.path.exists(act_path):
                best_ckpt = act_path

        if best_ckpt:
            # VecNormalize stats (if used)
            vec_norm_path = os.path.join(split_cwd, 'vec_normalize.pt') \
                if use_vec_normalize else None

            try:
                eval_result = evaluate_agent_on_indices(
                    checkpoint_path=best_ckpt,
                    indices=val_indices,
                    close_ary=hpo_close,
                    tech_ary=hpo_tech,
                    model_name=agent_name,
                    net_dims=net_dims,
                    gpu_id=gpu_id,
                    vec_normalize_path=vec_norm_path,
                    num_envs=1,  # deterministic single-env eval
                )
                sharpe_agent = eval_result['sharpe']
                agent_return = eval_result['final_return'] * 100  # as %

                # Save daily returns for PBO matrix construction
                daily_rets = np.array(eval_result['daily_returns'])
                npy_path = os.path.join(
                    pbo_returns_dir,
                    f"trial_{trial_id}_split_{split_idx}.npy",
                )
                np.save(npy_path, daily_rets)
            except Exception as e:
                print(f"  Split {split_idx}: eval failed ({e}), using Sharpe=0")

        # Benchmark: equal-weight buy-and-hold on same val indices
        bench = compute_equal_weight_benchmark(hpo_close, val_indices)
        sharpe_bench = bench['sharpe']
        bench_return = bench['total_return']

        # Composite objective: SR + 0.1 * alpha  (alpha = excess return)
        alpha = agent_return / 100.0 - bench_return  # both as fractions
        composite = sharpe_agent + ALPHA_WEIGHT * alpha

        sharpe_agent_list.append(sharpe_agent)
        sharpe_bench_list.append(sharpe_bench)
        alpha_list.append(alpha)
        composite_list.append(composite)

        print(f"  Split {split_idx}: SR={sharpe_agent:.4f}, "
              f"ret={agent_return:.1f}%, alpha={alpha:+.4f}, "
              f"composite={composite:.4f}, avgR={best_avgR:.1f}, "
              f"Bench SR={sharpe_bench:.4f}")

        # ── Cleanup checkpoints (HPO doesn't need them) ──────────────────
        if cfg.get('cleanup_checkpoints', True) and os.path.isdir(split_cwd):
            try:
                shutil.rmtree(split_cwd)
            except Exception:
                pass

        # Free GPU memory between splits
        th.cuda.empty_cache()

    # =========================================================================
    # AGGREGATE RESULTS
    # =========================================================================
    if not sharpe_agent_list:
        print("| ERROR: No successful splits!")
        return 0.0

    mean_sharpe_agent = np.mean(sharpe_agent_list)
    std_sharpe_agent = np.std(sharpe_agent_list)
    mean_sharpe_bench = np.mean(sharpe_bench_list)
    mean_alpha = np.mean(alpha_list)
    mean_composite = np.mean(composite_list)

    print(f"\n{'='*60}")
    print(f"HPO TRIAL RESULTS ({len(sharpe_agent_list)}/{n_splits} splits)")
    print(f"{'='*60}")
    print(f"| Mean Agent Sharpe:  {mean_sharpe_agent:.4f} "
          f"± {std_sharpe_agent:.4f}")
    print(f"| Mean Bench Sharpe:  {mean_sharpe_bench:.4f} "
          f"± {np.std(sharpe_bench_list):.4f}")
    print(f"| Mean Alpha:         {mean_alpha:+.4f}")
    print(f"| Mean Composite:     {mean_composite:.4f}  "
          f"(SR + {ALPHA_WEIGHT} × alpha)")
    print(f"| Objective return:   {mean_composite:.4f}  (maximize=true)")

    # ── Save trial summary for DSR post-hoc analysis ────────────────
    trial_summary = {
        'trial_id': trial_id,
        'mean_sharpe': float(mean_sharpe_agent),
        'std_sharpe': float(std_sharpe_agent),
        'mean_alpha': float(mean_alpha),
        'mean_composite': float(mean_composite),
        'per_split_sharpe': [float(s) for s in sharpe_agent_list],
        'per_split_alpha': [float(a) for a in alpha_list],
        'per_split_composite': [float(c) for c in composite_list],
        'n_splits': len(sharpe_agent_list),
        'hyperparams': {
            'net_dims': net_dims,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'batch_size': batch_size,
            'repeat_times': repeat_times,
            'num_envs': num_envs,
        },
    }
    summary_path = os.path.join(pbo_returns_dir, f"trial_{trial_id}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(trial_summary, f, indent=2)

    # ── Per-trial wandb logging (Hypersweeper logs optimizer-level stats;
    #    this adds richer per-trial metrics for dashboards) ────────────
    if wandb is not None and wandb.run is not None:
        wandb.log({
            "trial/id": trial_id,
            "trial/mean_sharpe": mean_sharpe_agent,
            "trial/std_sharpe": std_sharpe_agent,
            "trial/mean_bench_sharpe": mean_sharpe_bench,
            "trial/mean_alpha": mean_alpha,
            "trial/mean_composite": mean_composite,
            "trial/n_splits_ok": len(sharpe_agent_list),
            "trial/n_splits_total": n_splits,
            # Hyperparams for this trial
            "trial/hp_lr": learning_rate,
            "trial/hp_gamma": gamma,
            "trial/hp_batch_size": batch_size,
            "trial/hp_repeat_times": repeat_times,
            "trial/hp_num_envs": num_envs,
        })

    # Return composite (Hypersweeper handles maximize=true internally
    # by negating before passing to SMAC, and using argmax for incumbents)
    # Composite = SR + 0.1 * alpha: Sharpe dominates, alpha breaks ties
    return mean_composite


if __name__ == "__main__":
    train_and_evaluate()
