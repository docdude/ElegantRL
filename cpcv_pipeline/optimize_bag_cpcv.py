#!/usr/bin/env python3
"""
Bagged CPCV Training Script — Multi-seed ensemble per split.

Same CPCV splits as standard, but each split trains ``n_bags`` agents
with different random seeds.  At evaluation time, all bag actors are
loaded and their actions averaged ("ensemble inference").

This tests whether bagging reduces checkpoint-selection overfitting and
improves OOS stability, following RiskLabAI's B-CPCV extension.

Usage:
    # Train all 10 splits × 5 bags = 50 training runs
    python -m cpcv_pipeline.optimize_bag_cpcv --gpu 0

    # Fewer bags for a quick test
    python -m cpcv_pipeline.optimize_bag_cpcv --gpu 0 --n-bags 3

    # Single split, all bags
    python -m cpcv_pipeline.optimize_bag_cpcv --gpu 0 --split 0

    # Resume after crash
    python -m cpcv_pipeline.optimize_bag_cpcv --gpu 0 --cwd /path/to/run --continue

    # Multi-GPU
    CWD=/path/to/results
    python -m cpcv_pipeline.optimize_bag_cpcv --gpu 0 --splits 0-4 --cwd $CWD &
    python -m cpcv_pipeline.optimize_bag_cpcv --gpu 1 --splits 5-9 --cwd $CWD &
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
    BaggedCombPurgedKFoldCV,
    verify_no_leakage,
    verify_complete_oos_coverage,
    format_segments,
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
        description="Bagged CPCV Training — Multi-seed ensemble per split"
    )
    parser.add_argument("--model", type=str, default="ppo",
                        choices=["ppo", "a2c", "sac", "td3", "ddpg"])
    parser.add_argument("--n-groups", type=int, default=N_GROUPS)
    parser.add_argument("--k-test", type=int, default=K_TEST_GROUPS)
    parser.add_argument("--embargo", type=int, default=EMBARGO_DAYS)
    parser.add_argument("--gpu", type=int, default=GPU_ID)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Base seed (bag b uses seed + b)")
    parser.add_argument("--n-bags", type=int, default=5,
                        help="Number of bagged agents per split")
    parser.add_argument("--break-step", type=int,
                        default=DEFAULT_ERL_PARAMS['break_step'])
    parser.add_argument("--net-dims", type=str, default="128,64")
    parser.add_argument("--lr", type=float,
                        default=DEFAULT_ERL_PARAMS['learning_rate'])
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULT_ERL_PARAMS['batch_size'])
    parser.add_argument("--num-envs", type=int,
                        default=DEFAULT_ENV_PARAMS['num_envs'])
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override num_workers (default: nproc//4, max 4)")
    parser.add_argument("--split", type=int, default=None,
                        help="Run only this split index")
    parser.add_argument("--splits", type=str, default=None,
                        help="Run a range of splits, e.g. '0-4'")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--norm-obs-only", action="store_true")
    parser.add_argument("--continue", dest="continue_train", action="store_true")
    parser.add_argument("--cwd", type=str, default=None)
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

    close_ary, tech_ary = load_full_data()
    total_days = close_ary.shape[0]
    n_stocks = close_ary.shape[1]

    print(f"\n{'='*60}")
    print(f"Bagged CPCV Training Pipeline")
    print(f"{'='*60}")
    print(f"  Data: {total_days} days × {n_stocks} stocks")
    print(f"  Model: {args.model.upper()}")
    print(f"  CPCV: N={args.n_groups}, K={args.k_test}, "
          f"embargo={args.embargo}d")
    print(f"  Bags: {args.n_bags} (seeds: "
          f"{[args.seed + b for b in range(args.n_bags)]})")

    cv = BaggedCombPurgedKFoldCV(
        n_splits=args.n_groups,
        n_test_splits=args.k_test,
        embargo_days=args.embargo,
        n_bags=args.n_bags,
        base_seed=args.seed,
    )

    n_splits = cv.n_combinations
    n_paths = cv.n_paths
    total_runs = n_splits * args.n_bags
    print(f"  Splits: {n_splits}, Paths: {n_paths}")
    print(f"  Total training runs: {n_splits} × {args.n_bags} = {total_runs}")

    # Verify
    print(f"\nVerifying no leakage...")
    try:
        verify_no_leakage(cv, total_days)
        verify_complete_oos_coverage(cv, total_days)
        print(f"  ✓ No leakage, complete OOS coverage")
    except AssertionError as e:
        print(f"  ✗ LEAKAGE: {e}")
        return 1

    # Prepare directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    net_dims = [int(x) for x in args.net_dims.split(",")]

    if args.cwd:
        cwd_base = args.cwd
    else:
        cwd_base = os.path.join(
            RESULTS_DIR,
            f"{timestamp}_BagCPCV_{args.model.upper()}_N{args.n_groups}"
            f"K{args.k_test}_bags{args.n_bags}_seed{args.seed}"
        )
    os.makedirs(cwd_base, exist_ok=True)

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
    vec_normalize_kwargs = VEC_NORMALIZE_KWARGS.copy()
    if args.norm_obs_only:
        use_vec_normalize = True
        vec_normalize_kwargs['norm_obs'] = True
        vec_normalize_kwargs['norm_reward'] = False

    # Save config
    run_config = {
        'timestamp': timestamp,
        'method': 'bagged_cpcv',
        'model': args.model,
        'n_groups': args.n_groups,
        'k_test': args.k_test,
        'embargo_days': args.embargo,
        'n_splits': n_splits,
        'n_paths': n_paths,
        'n_bags': args.n_bags,
        'bag_seeds': cv.bag_seeds(),
        'total_runs': total_runs,
        'total_days': total_days,
        'n_stocks': n_stocks,
        'seed': args.seed,
        'gpu_id': args.gpu,
        'erl_params': erl_params,
        'env_params': env_params,
        'use_vec_normalize': use_vec_normalize,
        'net_dims': net_dims,
    }
    with open(os.path.join(cwd_base, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2, default=str)

    # Print splits
    all_splits = list(cv.split(total_days))
    print(f"\nSplit summary:")
    for i, (train_idx, test_idx) in enumerate(all_splits):
        print(f"  Split {i:>2}: "
              f"Train {len(train_idx):>4}d {format_segments(train_idx)}  |  "
              f"Test {len(test_idx):>4}d {format_segments(test_idx)}")

    if args.dry_run:
        print(f"\n  [DRY RUN] Would train {total_runs} runs to: {cwd_base}")
        return 0

    # Determine splits to run
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
    bag_seeds = cv.bag_seeds()

    results = []
    for split_idx, (train_idx, test_idx) in splits_to_run:
        for bag_idx, bag_seed in enumerate(bag_seeds):
            # Each bag gets its own subdirectory
            bag_cwd = os.path.join(cwd_base, f"split_{split_idx}",
                                   f"bag_{bag_idx}")

            # Skip completed bags when resuming
            if args.continue_train:
                recorder_path = os.path.join(bag_cwd, 'recorder.npy')
                if os.path.exists(recorder_path):
                    rec = np.load(recorder_path)
                    if len(rec) > 0:
                        max_step = int(rec[-1, 0])
                        if max_step >= erl_params['break_step']:
                            best_idx = int(np.argmax(rec[:, 1]))
                            print(f"\n  [SKIP] Split {split_idx} Bag {bag_idx} "
                                  f"(seed {bag_seed}): completed "
                                  f"({max_step:,} steps, "
                                  f"best_avgR={rec[best_idx, 1]:.3f})")
                            results.append({
                                'split_idx': split_idx,
                                'bag_idx': bag_idx,
                                'bag_seed': bag_seed,
                                'cwd': bag_cwd,
                                'skipped': True,
                            })
                            continue

            print(f"\n{'#'*60}")
            print(f"# SPLIT {split_idx}/{n_splits}  "
                  f"BAG {bag_idx}/{args.n_bags}  "
                  f"(seed {bag_seed})")
            print(f"{'#'*60}")

            # train_split saves to cwd_base/cwd_suffix
            # Use cwd_suffix="bag_{bag_idx}" inside split_{split_idx}/
            bag_cwd_base = os.path.join(cwd_base, f"split_{split_idx}")

            result = train_split(
                split_idx=split_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                close_ary=close_ary,
                tech_ary=tech_ary,
                model_name=args.model,
                erl_params=erl_params,
                env_params=env_params,
                cwd_base=bag_cwd_base,
                gpu_id=args.gpu,
                random_seed=bag_seed,
                use_vec_normalize=use_vec_normalize,
                vec_normalize_kwargs=vec_normalize_kwargs,
                continue_train=args.continue_train,
                cwd_suffix=f"bag_{bag_idx}",
                num_workers=args.num_workers,
            )
            result['split_idx'] = split_idx
            result['bag_idx'] = bag_idx
            result['bag_seed'] = bag_seed
            results.append(result)

        # Save per-split result
        split_result_path = os.path.join(
            cwd_base, f'result_split_{split_idx}.json'
        )
        split_bags = [r for r in results if r.get('split_idx') == split_idx]
        with open(split_result_path, 'w') as f:
            json.dump(split_bags, f, indent=2, default=str)

    # Save combined results
    with open(os.path.join(cwd_base, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"Bagged CPCV Training Complete")
    print(f"{'='*60}")
    for split_idx in split_indices:
        split_bags = [r for r in results if r.get('split_idx') == split_idx]
        avgRs = [r.get('best_avgR', 0) for r in split_bags
                 if not r.get('skipped')]
        if avgRs:
            print(f"  Split {split_idx}: {len(split_bags)} bags, "
                  f"best_avgR range [{min(avgRs):.1f}, {max(avgRs):.1f}]")
    print(f"\n  Results: {cwd_base}")
    print(f"\n  Next: evaluate with ensemble inference")
    print(f"    python -m cpcv_pipeline.eval_all_checkpoints "
          f"--results-dir {cwd_base}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
