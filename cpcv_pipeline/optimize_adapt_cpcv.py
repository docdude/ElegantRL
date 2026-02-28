#!/usr/bin/env python3
"""
Adaptive CPCV Training Script — Feature-aware group boundaries.

Same combinatorial split structure as standard CPCV, but group boundaries
are shifted based on an external feature (default: 63-day rolling drawdown)
so that splits avoid regime transitions.  This follows RiskLabAI's
A-CPCV extension.

Algorithm:
  1. Compute external feature from close_ary (default: 63-day drawdown)
  2. Create fine-grained subsplits (n_subsplits per group)
  3. At each initial boundary, check feature level:
     - Low feature  → shift boundary right (+1 subsplit)
     - High feature → shift boundary left  (-1 subsplit)
     Boundaries move *away* from extremes, keeping similar regimes together.
  4. Train DRL agent on each of the C(N,K) adaptive splits

Drawdown (63d) was selected as default because it achieves the lowest
regime gap (AvgGap=8.9%) and 0 strict mismatches vs 3 for standard CPCV.

Usage:
    # Standard run (drawdown 63d, obs-only norm)
    python -m cpcv_pipeline.optimize_adapt_cpcv --gpu 0

    # Custom feature and window
    python -m cpcv_pipeline.optimize_adapt_cpcv --gpu 0 --feature volatility --feature-window 42

    # Custom quantile thresholds
    python -m cpcv_pipeline.optimize_adapt_cpcv --gpu 0 \\
        --lower-quantile 0.2 --upper-quantile 0.8

    # Dry run to see adaptive boundaries
    python -m cpcv_pipeline.optimize_adapt_cpcv --dry-run

    # Resume after crash
    python -m cpcv_pipeline.optimize_adapt_cpcv --gpu 0 --cwd $CWD --continue
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
    verify_complete_oos_coverage,
    format_segments,
    compute_external_feature,
    FEATURE_CHOICES,
    FEATURE_DEFAULT_WINDOWS,
)
from cpcv_pipeline.function_train_test import (
    load_full_data,
    train_split,
)
from cpcv_pipeline.config import (
    N_GROUPS, K_TEST_GROUPS, EMBARGO_DAYS,
    DEFAULT_ERL_PARAMS, DEFAULT_ENV_PARAMS,
    USE_VEC_NORMALIZE, VEC_NORMALIZE_KWARGS,
    RANDOM_SEED, GPU_ID, RESULTS_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive CPCV Training — Feature-aware boundaries"
    )
    # Model
    parser.add_argument("--model", type=str, default="ppo",
                        choices=["ppo", "a2c", "sac", "td3", "ddpg"],
                        help="DRL algorithm")
    # CPCV params
    parser.add_argument("--n-groups", type=int, default=N_GROUPS,
                        help=f"Number of fold groups N (default: {N_GROUPS})")
    parser.add_argument("--k-test", type=int, default=K_TEST_GROUPS,
                        help=f"Test groups per split K (default: {K_TEST_GROUPS})")
    parser.add_argument("--embargo", type=int, default=EMBARGO_DAYS,
                        help=f"Embargo days (default: {EMBARGO_DAYS})")
    # Training params
    parser.add_argument("--gpu", type=int, default=GPU_ID,
                        help=f"GPU ID (default: {GPU_ID})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument("--break-step", type=int,
                        default=DEFAULT_ERL_PARAMS['break_step'],
                        help="Break step for training")
    parser.add_argument("--net-dims", type=str, default="128,64",
                        help="Network dimensions (comma-separated)")
    parser.add_argument("--lr", type=float,
                        default=DEFAULT_ERL_PARAMS['learning_rate'],
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULT_ERL_PARAMS['batch_size'],
                        help="Batch size")
    parser.add_argument("--num-envs", type=int,
                        default=DEFAULT_ENV_PARAMS['num_envs'],
                        help="Base num_envs (auto-scaled for horizon)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override num_workers (default: nproc//4, max 4). "
                             "Useful on low-CPU machines like L4 (8 cores → 2 workers)")
    # A-CPCV specific
    parser.add_argument("--feature", type=str, default="drawdown",
                        choices=FEATURE_CHOICES,
                        help="External feature for boundary adjustment "
                             "(default: drawdown — 0 regime mismatches)")
    parser.add_argument("--feature-window", type=int, default=None,
                        help="Rolling feature window in days "
                             "(default: auto per feature — drawdown=63, volatility=21)")
    parser.add_argument("--n-subsplits", type=int, default=3,
                        help="Sub-intervals per group for boundary adjustment")
    parser.add_argument("--lower-quantile", type=float, default=0.25,
                        help="Lower feature quantile threshold")
    parser.add_argument("--upper-quantile", type=float, default=0.75,
                        help="Upper feature quantile threshold")
    # Execution control
    parser.add_argument("--split", type=int, default=None,
                        help="Run only this split index (0-based)")
    parser.add_argument("--splits", type=str, default=None,
                        help="Run a range of splits, e.g. '0-4' or "
                             "'0,2,5' (for multi-GPU parallel)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print split info without training")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable VecNormalize")
    parser.add_argument("--norm-obs-only", action="store_true",
                        help="VecNormalize obs only (no reward norm)")
    parser.add_argument("--continue", dest="continue_train", action="store_true",
                        help="Resume from existing checkpoints (skip completed splits, "
                             "reload weights for partial splits)")
    parser.add_argument("--cwd", type=str, default=None,
                        help="Custom output directory")
    return parser.parse_args()


def _parse_splits(spec: str, n_splits: int) -> list:
    indices = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    for i in indices:
        if i < 0 or i >= n_splits:
            raise ValueError(f"Split {i} out of range [0, {n_splits - 1}]")
    return sorted(set(indices))


def main():
    args = parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    close_ary, tech_ary = load_full_data()
    total_days = close_ary.shape[0]
    n_stocks = close_ary.shape[1]
    n_tech = tech_ary.shape[1]

    print(f"\n{'='*60}")
    print(f"Adaptive CPCV Training Pipeline")
    print(f"{'='*60}")
    print(f"  Data: {total_days} days × {n_stocks} stocks, "
          f"{n_tech} tech features")
    print(f"  Model: {args.model.upper()}")
    print(f"  CPCV: N={args.n_groups}, K={args.k_test}, "
          f"embargo={args.embargo}d")

    # Resolve feature window: use CLI value if given, else feature default
    feature_window = args.feature_window
    if feature_window is None:
        feature_window = FEATURE_DEFAULT_WINDOWS.get(args.feature)
    window_str = f"{feature_window}d" if feature_window is not None else "N/A"
    print(f"  Adaptive: feature={args.feature}, window={window_str}, "
          f"subsplits={args.n_subsplits}, "
          f"quantiles=[{args.lower_quantile}, {args.upper_quantile}]")

    # ── Compute external feature ─────────────────────────────────────────
    print(f"\nComputing {args.feature} feature (window={feature_window})...")
    ext_feature = compute_external_feature(
        close_ary, feature_name=args.feature, window=feature_window,
        tech_ary=tech_ary,
    )
    print(f"  Feature range: [{ext_feature.min():.6f}, {ext_feature.max():.6f}]")
    print(f"  Feature mean:  {ext_feature.mean():.6f}, std: {ext_feature.std():.6f}")

    # ── Create Adaptive CPCV splitter ────────────────────────────────────
    cv = AdaptiveCombPurgedKFoldCV(
        n_splits=args.n_groups,
        n_test_splits=args.k_test,
        embargo_days=args.embargo,
        external_feature=ext_feature,
        n_subsplits=args.n_subsplits,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )

    n_splits = cv.n_combinations
    n_paths = cv.n_paths
    print(f"  Splits: {n_splits} (C({args.n_groups},{args.k_test}))")
    print(f"  Paths:  {n_paths}")

    # Compare with standard CPCV
    std_cv = CombPurgedKFoldCV(
        n_splits=args.n_groups,
        n_test_splits=args.k_test,
        embargo_days=args.embargo,
    )
    std_bounds = std_cv.get_fold_bounds(total_days)
    adp_bounds = cv.get_fold_bounds(total_days)

    print(f"\n  Group boundaries (standard vs adaptive):")
    print(f"  {'Group':>6}  {'Standard':>20}  {'Adaptive':>20}  {'Shift':>8}")
    for i, ((ss, se), (as_, ae)) in enumerate(zip(std_bounds, adp_bounds)):
        shift_start = as_ - ss
        shift_end = ae - se
        print(f"  {i:>6}  [{ss:>4}:{se:>4}] ({se-ss:>3}d)  "
              f"[{as_:>4}:{ae:>4}] ({ae-as_:>3}d)  "
              f"{shift_start:>+4},{shift_end:>+4}")

    # ── Verify no leakage ────────────────────────────────────────────────
    print(f"\nVerifying no leakage...")
    try:
        verify_no_leakage(cv, total_days)
        verify_complete_oos_coverage(cv, total_days)
        print(f"  ✓ No leakage detected across all {n_splits} splits")
        print(f"  ✓ Complete OOS coverage verified")
    except AssertionError as e:
        print(f"  ✗ LEAKAGE DETECTED: {e}")
        print(f"  ABORTING — fix the A-CPCV configuration first")
        return 1

    # ── Prepare run directory ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    net_dims = [int(x) for x in args.net_dims.split(",")]

    if args.cwd:
        cwd_base = args.cwd
    else:
        cwd_base = os.path.join(
            RESULTS_DIR,
            f"{timestamp}_AdaptCPCV_{args.model.upper()}_N{args.n_groups}"
            f"K{args.k_test}_seed{args.seed}"
        )

    os.makedirs(cwd_base, exist_ok=True)

    # ── Prepare hyperparameters ──────────────────────────────────────────
    erl_params = DEFAULT_ERL_PARAMS.copy()
    erl_params.update({
        'net_dims': net_dims,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'break_step': args.break_step,
    })

    env_params = DEFAULT_ENV_PARAMS.copy()
    env_params['num_envs'] = args.num_envs

    use_vec_normalize = not args.no_normalize

    # Override norm kwargs if --norm-obs-only
    vec_normalize_kwargs = VEC_NORMALIZE_KWARGS.copy()
    if args.norm_obs_only:
        use_vec_normalize = True  # override --no-normalize if both set
        vec_normalize_kwargs['norm_obs'] = True
        vec_normalize_kwargs['norm_reward'] = False

    # ── Save run config ──────────────────────────────────────────────────
    run_config = {
        'timestamp': timestamp,
        'method': 'adaptive_cpcv',
        'model': args.model,
        'n_groups': args.n_groups,
        'k_test': args.k_test,
        'embargo_days': args.embargo,
        'n_splits': n_splits,
        'n_paths': n_paths,
        'total_days': total_days,
        'n_stocks': n_stocks,
        'seed': args.seed,
        'gpu_id': args.gpu,
        'erl_params': erl_params,
        'env_params': env_params,
        'use_vec_normalize': use_vec_normalize,
        'net_dims': net_dims,
        'adaptive_params': {
            'feature': args.feature,
            'feature_window': feature_window,
            'n_subsplits': args.n_subsplits,
            'lower_quantile': args.lower_quantile,
            'upper_quantile': args.upper_quantile,
            'feature_mean': float(ext_feature.mean()),
            'feature_std': float(ext_feature.std()),
        },
        'standard_fold_bounds': [list(b) for b in std_bounds],
        'adaptive_fold_bounds': [list(b) for b in adp_bounds],
    }
    with open(os.path.join(cwd_base, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2, default=str)

    # Save external feature for reproducibility
    np.save(os.path.join(cwd_base, 'external_feature.npy'), ext_feature)

    # ── Print split summary ──────────────────────────────────────────────
    print(f"\nSplit summary:")
    all_splits = list(cv.split(total_days))
    for i, (train_idx, test_idx) in enumerate(all_splits):
        marker = " ← selected" if args.split == i else ""
        print(f"  Split {i + 1:>2}: "
              f"Train {len(train_idx):>4}d {format_segments(train_idx)}  |  "
              f"Test {len(test_idx):>4}d {format_segments(test_idx)}{marker}")

    if args.dry_run:
        print(f"\n  [DRY RUN] Would save results to: {cwd_base}")
        print(cv.describe_splits(total_days))
        return 0

    # ── Train ────────────────────────────────────────────────────────────
    if args.split is not None and args.splits is not None:
        print("ERROR: --split and --splits are mutually exclusive")
        return 1

    if args.split is not None:
        split_indices = [args.split]
    elif args.splits is not None:
        split_indices = _parse_splits(args.splits, n_splits)
    else:
        split_indices = list(range(n_splits))

    splits_to_run = [(i, all_splits[i]) for i in split_indices]

    results = []
    for split_idx, (train_idx, test_idx) in splits_to_run:
        split_cwd = os.path.join(cwd_base, f"split_{split_idx}")

        # ── Skip completed splits when resuming ─────────────────────────
        if args.continue_train:
            recorder_path = os.path.join(split_cwd, 'recorder.npy')
            if os.path.exists(recorder_path):
                rec = np.load(recorder_path)
                if len(rec) > 0:
                    max_step = int(rec[-1, 0])
                    if max_step >= erl_params['break_step']:
                        best_idx = int(np.argmax(rec[:, 1]))
                        print(f"\n  [SKIP] Split {split_idx + 1}: already "
                              f"completed ({max_step:,} steps, "
                              f"best_avgR={rec[best_idx, 1]:.3f})")
                        results.append({
                            'split_idx': split_idx,
                            'cwd': split_cwd,
                            'train_days': len(train_idx),
                            'test_days': len(test_idx),
                            'model_name': args.model,
                            'seed': args.seed,
                            'best_step': int(rec[best_idx, 0]),
                            'best_avgR': float(rec[best_idx, 1]),
                            'skipped': True,
                        })
                        continue
                    else:
                        print(f"\n  [RESUME] Split {split_idx + 1}: "
                              f"partial run ({max_step:,} / "
                              f"{erl_params['break_step']:,} steps)")

        print(f"\n{'#'*60}")
        print(f"# TRAINING SPLIT {split_idx + 1} / {n_splits}")
        print(f"{'#'*60}")

        result = train_split(
            split_idx=split_idx,
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
            use_vec_normalize=use_vec_normalize,
            vec_normalize_kwargs=vec_normalize_kwargs,
            continue_train=args.continue_train,
            num_workers=args.num_workers,
        )
        results.append(result)

        # Save per-split result (multi-GPU safe — no shared file)
        split_result_path = os.path.join(
            cwd_base, f'result_split_{split_idx}.json'
        )
        with open(split_result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # Also save combined results for this process
        with open(os.path.join(cwd_base, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    for r in results:
        best_r = r.get('best_avgR', 'N/A')
        print(f"  Split {r['split_idx'] + 1}: "
              f"train={r['train_days']}d, test={r['test_days']}d, "
              f"best_avgR={best_r}")
    print(f"\n  Results saved to: {cwd_base}")
    print(f"\n  Next steps:")
    print(f"  1. Compare ALL checkpoints:  python -m cpcv_pipeline.eval_all_checkpoints --results-dir {cwd_base}")
    print(f"  2. Evaluate best per split:  python -m cpcv_pipeline.evaluate_splits --results-dir {cwd_base}")
    print(f"  3. Run DSR analysis:         python -m cpcv_pipeline.run_dsr --results-dir {cwd_base} --per-split --full --gpu 0")

    return 0


if __name__ == "__main__":
    sys.exit(main())
