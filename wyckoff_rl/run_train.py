#!/usr/bin/env python3
"""
Wyckoff RL — Adaptive CPCV Training Script.

Trains a DRL agent on NQ range bars with Wyckoff features using
Adaptive Combinatorial Purged K-Fold Cross-Validation.

Uses the GPU-vectorized WyckoffTradingVecEnv with multiprocessing.
Designed to run on Lightning AI Studio or any GPU machine.

Usage:
    # Default: PPO, 10 splits, pnl reward
    python -m wyckoff_rl.run_train --gpu 0

    # Different reward mode
    python -m wyckoff_rl.run_train --gpu 0 --reward sharpe

    # Single split (for testing)
    python -m wyckoff_rl.run_train --gpu 0 --split 0

    # SAC instead of PPO
    python -m wyckoff_rl.run_train --gpu 0 --model sac

    # Dry run (print splits, no training)
    python -m wyckoff_rl.run_train --dry-run

    # Resume after crash
    python -m wyckoff_rl.run_train --gpu 0 --continue --cwd /path/to/results
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cpcv_pipeline.function_CPCV import (
    AdaptiveCombPurgedKFoldCV,
    CombPurgedKFoldCV,
    verify_no_leakage,
    format_segments,
)
from wyckoff_rl.function_train_test import (
    load_wyckoff_data,
    train_split,
)
from wyckoff_rl.config import (
    WYCKOFF_NPZ_PATH, RESULTS_DIR,
    N_GROUPS, K_TEST_GROUPS, EMBARGO_BARS,
    ADAPTIVE_FEATURE, ADAPTIVE_SMOOTH_WINDOW,
    ADAPTIVE_N_SUBSPLITS, ADAPTIVE_LOWER_Q, ADAPTIVE_UPPER_Q,
    DEFAULT_ERL_PARAMS, DEFAULT_ENV_PARAMS,
    RANDOM_SEED, GPU_ID,
)


def compute_adaptive_feature(
    close_ary: np.ndarray,
    tech_ary: np.ndarray,
    feature_names: np.ndarray,
    feature_name: str = "ER_Ratio",
    smooth_window: int = 50,
) -> np.ndarray:
    """Compute external feature for adaptive CPCV boundary shifting.

    Extracts a named column from tech_ary and applies rolling-mean smoothing.
    Falls back to rolling volatility if the feature isn't found in tech_ary.
    """
    idx = np.where(feature_names == feature_name)[0]
    if len(idx) > 0:
        raw = tech_ary[:, idx[0]].astype(np.float64)
        if smooth_window > 1 and raw.std() > 0:
            kernel = np.ones(smooth_window) / smooth_window
            return np.convolve(raw, kernel, mode='same')
        return raw
    # Fallback: rolling volatility
    print(f"  ⚠ Feature '{feature_name}' not in tech_ary, falling back to rolling vol")
    prices = close_ary.ravel()
    log_ret = np.diff(np.log(prices + 1e-12))
    log_ret = np.concatenate([[0.0], log_ret])
    vol = np.full_like(log_ret, np.nan)
    window = smooth_window
    for i in range(window, len(log_ret)):
        vol[i] = log_ret[i - window:i].std()
    vol[:window] = vol[window] if window < len(vol) else 0.0
    return vol


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wyckoff RL — Adaptive CPCV Training"
    )
    parser.add_argument("--model", type=str, default="ppo",
                        choices=["ppo", "wyckoff_ppo", "wyckoff_wave_ppo", "sac", "td3"])
    parser.add_argument("--reward", type=str, default=DEFAULT_ENV_PARAMS['reward_mode'],
                        choices=["pnl", "log_ret", "sharpe", "sortino"],
                        help="Reward function")
    parser.add_argument("--npz", type=str, default=WYCKOFF_NPZ_PATH,
                        help="Path to Wyckoff NPZ data")

    # CPCV
    parser.add_argument("--n-groups", type=int, default=N_GROUPS)
    parser.add_argument("--k-test", type=int, default=K_TEST_GROUPS)
    parser.add_argument("--embargo", type=int, default=EMBARGO_BARS)
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Use standard CPCV (no boundary adaptation)")

    # Training
    parser.add_argument("--gpu", type=int, default=GPU_ID)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--break-step", type=int,
                        default=DEFAULT_ERL_PARAMS['break_step'])
    parser.add_argument("--net-dims", type=str, default="128,64")
    parser.add_argument("--lr", type=float,
                        default=DEFAULT_ERL_PARAMS['learning_rate'])
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULT_ERL_PARAMS['batch_size'])
    parser.add_argument("--continuous", action="store_true",
                        help="Use continuous [-1,+1] position sizing instead of binary {-1,0,+1}")
    parser.add_argument("--loss-weight", type=float,
                        default=DEFAULT_ERL_PARAMS['loss_weight'],
                        help="Asymmetric advantage weight for losses (1.0=symmetric, 2.0=2x loss penalty)")
    parser.add_argument("--trade-reward-weight", type=float,
                        default=DEFAULT_ENV_PARAMS['trade_reward_weight'],
                        help="Trade-close bonus weight (0.0=bar-only, 0.5=adds trade PnL bonus)")

    # Execution
    parser.add_argument("--split", type=int, default=None,
                        help="Run only this split (0-based)")
    parser.add_argument("--splits", type=str, default=None,
                        help="Run range/list of splits: '0-4' or '0,2,5'")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue", dest="continue_train", action="store_true")
    parser.add_argument("--cwd", type=str, default=None)

    return parser.parse_args()


def parse_split_spec(spec: str, n_total: int) -> list:
    indices = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    for i in indices:
        if i < 0 or i >= n_total:
            raise ValueError(f"Split {i} out of range [0, {n_total - 1}]")
    return sorted(set(indices))


def main():
    args = parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    close_ary, tech_ary = load_wyckoff_data(args.npz)
    total_bars = close_ary.shape[0]
    n_features = tech_ary.shape[1]

    # Load feature names for adaptive CPCV
    _raw = np.load(args.npz, allow_pickle=True)
    feature_names = _raw['feature_names'] if 'feature_names' in _raw else np.array([])

    print(f"\n{'='*60}")
    print(f"Wyckoff RL — Adaptive CPCV Training")
    print(f"{'='*60}")
    print(f"  Data: {args.npz}")
    print(f"  Bars: {total_bars:,}, Features: {n_features}")
    sizing = "continuous [-1,+1]" if args.continuous else "binary {-1,0,+1}"
    print(f"  Model: {args.model.upper()}, Reward: {args.reward}, Sizing: {sizing}")
    print(f"  CPCV: N={args.n_groups}, K={args.k_test}, "
          f"embargo={args.embargo} bars")

    # ── Create splitter ──────────────────────────────────────────────────
    if args.no_adaptive:
        cv = CombPurgedKFoldCV(
            n_splits=args.n_groups,
            n_test_splits=args.k_test,
            embargo_days=args.embargo,
        )
        print(f"  Mode: Standard CPCV")
    else:
        ext_feature = compute_adaptive_feature(
            close_ary, tech_ary, feature_names,
            feature_name=ADAPTIVE_FEATURE,
            smooth_window=ADAPTIVE_SMOOTH_WINDOW,
        )
        cv = AdaptiveCombPurgedKFoldCV(
            n_splits=args.n_groups,
            n_test_splits=args.k_test,
            embargo_days=args.embargo,
            external_feature=ext_feature,
            n_subsplits=ADAPTIVE_N_SUBSPLITS,
            lower_quantile=ADAPTIVE_LOWER_Q,
            upper_quantile=ADAPTIVE_UPPER_Q,
        )
        print(f"  Mode: Adaptive CPCV ({ADAPTIVE_FEATURE}, smooth={ADAPTIVE_SMOOTH_WINDOW})")

    n_splits = cv.n_combinations
    n_paths = cv.n_paths
    print(f"  Splits: {n_splits}, Paths: {n_paths}")

    # ── Verify ───────────────────────────────────────────────────────────
    all_splits = list(cv.split(total_bars))
    print(f"\nVerifying splits...")
    for i, (tr, te) in enumerate(all_splits):
        overlap = np.intersect1d(tr, te)
        if len(overlap) > 0:
            print(f"  ✗ LEAKAGE in split {i}: {len(overlap)} overlapping")
            return 1
    print(f"  ✓ All {n_splits} splits leak-free")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\nSplit summary:")
    for i, (tr, te) in enumerate(all_splits):
        marker = " ← selected" if args.split == i else ""
        print(f"  Split {i:>2}: Train {len(tr):>6,} bars  |  "
              f"Test {len(te):>6,} bars{marker}")

    if args.dry_run:
        print(f"\n  [DRY RUN] No training performed.")
        return 0

    # ── Prepare hyperparams ──────────────────────────────────────────────
    net_dims = [int(x) for x in args.net_dims.split(",")]

    erl_params = DEFAULT_ERL_PARAMS.copy()
    erl_params.update({
        'net_dims': net_dims,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'break_step': args.break_step,
        'loss_weight': args.loss_weight,
    })

    env_params = DEFAULT_ENV_PARAMS.copy()
    env_params['reward_mode'] = args.reward
    env_params['continuous_sizing'] = args.continuous
    env_params['trade_reward_weight'] = args.trade_reward_weight

    # ── Output directory ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.cwd:
        cwd_base = args.cwd
    else:
        cwd_base = os.path.join(
            RESULTS_DIR,
            f"{timestamp}_Wyckoff_{args.model.upper()}_"
            f"{args.reward}_seed{args.seed}"
        )
    os.makedirs(cwd_base, exist_ok=True)

    # Save run config
    run_config = {
        'timestamp': timestamp,
        'npz_path': args.npz,
        'total_bars': total_bars,
        'n_features': n_features,
        'model': args.model,
        'reward_mode': args.reward,
        'n_groups': args.n_groups,
        'k_test': args.k_test,
        'embargo_bars': args.embargo,
        'adaptive': not args.no_adaptive,
        'n_splits': n_splits,
        'seed': args.seed,
        'erl_params': erl_params,
        'env_params': env_params,
    }
    with open(os.path.join(cwd_base, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2, default=str)

    # ── Determine which splits to run ────────────────────────────────────
    if args.split is not None and args.splits is not None:
        print("ERROR: --split and --splits are mutually exclusive")
        return 1

    if args.split is not None:
        split_indices = [args.split]
    elif args.splits is not None:
        split_indices = parse_split_spec(args.splits, n_splits)
    else:
        split_indices = list(range(n_splits))

    # ── Train ────────────────────────────────────────────────────────────
    results = []
    for idx in split_indices:
        train_idx, test_idx = all_splits[idx]

        # Skip completed when resuming
        if args.continue_train:
            recorder = os.path.join(cwd_base, f"split_{idx}", "recorder.npy")
            if os.path.exists(recorder):
                rec = np.load(recorder)
                if len(rec) > 0:
                    print(f"\n  Split {idx}: already completed "
                          f"(best_cumR={rec[np.argmax(rec[:,1]),1]:.2f}), skipping")
                    continue

        result = train_split(
            split_idx=idx,
            train_indices=train_idx,
            test_indices=test_idx,
            close_ary=close_ary,
            tech_ary=tech_ary,
            model_name=args.model,
            erl_params=erl_params,
            env_params=env_params,
            cwd_base=cwd_base,
            gpu_id=args.gpu,
            random_seed=args.seed,
            continue_train=args.continue_train,
        )
        results.append(result)
        print(f"\n  Split {idx} done: {result}")

    # ── Summary ──────────────────────────────────────────────────────────
    if results:
        with open(os.path.join(cwd_base, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"Training Complete — {len(results)} splits")
        print(f"{'='*60}")
        for r in results:
            best = r.get('best_cumR', 'N/A')
            print(f"  Split {r['split_idx']}: best_cumR={best}")
        print(f"\nResults saved to: {cwd_base}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
