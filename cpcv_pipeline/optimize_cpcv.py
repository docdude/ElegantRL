#!/usr/bin/env python3
"""
CPCV Training Script — Combinatorial Purged K-Fold Cross-Validation.

Trains a DRL agent across all C(N,K) CPCV splits, then assembles
backtest paths for PBO analysis.

This is the CPCV equivalent of FinRL_Crypto's 1_optimize_cpcv.py.

Usage:
    # Verify no leakage first (always do this before GPU runs)
    python -m cpcv_pipeline.test_leakage --verbose

    # Run all splits sequentially
    python -m cpcv_pipeline.optimize_cpcv --model ppo --gpu 0

    # Run a single split
    python -m cpcv_pipeline.optimize_cpcv --model ppo --gpu 0 --split 0

    # Dry run (print splits, don't train)
    python -m cpcv_pipeline.optimize_cpcv --dry-run

    # With custom settings
    python -m cpcv_pipeline.optimize_cpcv --model ppo --n-groups 6 --k-test 2 \\
        --embargo 7 --break-step 2000000 --seed 1943

    # Multi-GPU: split work across GPUs (run in parallel terminals)
    CWD=/path/to/results
    python -m cpcv_pipeline.optimize_cpcv --gpu 0 --splits 0-4 --cwd $CWD &
    python -m cpcv_pipeline.optimize_cpcv --gpu 1 --splits 5-9 --cwd $CWD &

    # Resume after crash (skips completed, reloads partial)
    python -m cpcv_pipeline.optimize_cpcv --gpu 0 --splits 0-4 --cwd $CWD --continue
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cpcv_pipeline.function_CPCV import (
    CombPurgedKFoldCV,
    back_test_paths_generator,
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
    print_config as print_pipeline_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPCV Training — Combinatorial Purged K-Fold CV"
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
    """Parse split spec: '0-4', '0,2,5', or '0-4,7,9'."""
    indices = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    # Validate
    for i in indices:
        if i < 0 or i >= n_splits:
            raise ValueError(
                f"Split {i} out of range [0, {n_splits - 1}]"
            )
    return sorted(set(indices))


def main():
    args = parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    close_ary, tech_ary = load_full_data()
    total_days = close_ary.shape[0]
    n_stocks = close_ary.shape[1]
    n_tech = tech_ary.shape[1]

    print(f"\n{'='*60}")
    print(f"CPCV Training Pipeline")
    print(f"{'='*60}")
    print(f"  Data: {total_days} days × {n_stocks} stocks, "
          f"{n_tech} tech features")
    print(f"  Model: {args.model.upper()}")
    print(f"  CPCV: N={args.n_groups}, K={args.k_test}, "
          f"embargo={args.embargo}d")

    # ── Create CPCV splitter ─────────────────────────────────────────────
    cv = CombPurgedKFoldCV(
        n_splits=args.n_groups,
        n_test_splits=args.k_test,
        embargo_days=args.embargo,
    )

    n_splits = cv.n_combinations
    n_paths = cv.n_paths
    print(f"  Splits: {n_splits} (C({args.n_groups},{args.k_test}))")
    print(f"  Paths:  {n_paths}")

    # ── Verify no leakage ────────────────────────────────────────────────
    print(f"\nVerifying no leakage...")
    try:
        verify_no_leakage(cv, total_days)
        verify_complete_oos_coverage(cv, total_days)
        print(f"  ✓ No leakage detected across all {n_splits} splits")
        print(f"  ✓ Complete OOS coverage verified")
    except AssertionError as e:
        print(f"  ✗ LEAKAGE DETECTED: {e}")
        print(f"  ABORTING — fix the CPCV configuration first")
        return 1

    # ── Prepare run directory ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    net_dims = [int(x) for x in args.net_dims.split(",")]

    if args.cwd:
        cwd_base = args.cwd
    else:
        cwd_base = os.path.join(
            RESULTS_DIR,
            f"{timestamp}_CPCV_{args.model.upper()}_N{args.n_groups}"
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
    }
    with open(os.path.join(cwd_base, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2, default=str)

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
    print(f"  3. Run PBO analysis:         python -m cpcv_pipeline.run_pbo --cwd {cwd_base}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
