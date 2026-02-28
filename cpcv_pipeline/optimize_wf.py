#!/usr/bin/env python3
"""
Walk-Forward Training Script.

Trains a DRL agent using anchored walk-forward cross-validation.
Unlike CPCV, walk-forward always trains on data before the test period,
making it naturally leakage-free.

Usage:
    python -m cpcv_pipeline.optimize_wf --model ppo --gpu 0
    python -m cpcv_pipeline.optimize_wf --model ppo --n-folds 5 --gap 7
    python -m cpcv_pipeline.optimize_wf --dry-run

    # Multi-GPU
    CWD=train_results/WF_PPO
    python -m cpcv_pipeline.optimize_wf --gpu 0 --splits 0-2 --cwd $CWD &
    python -m cpcv_pipeline.optimize_wf --gpu 1 --splits 3-4 --cwd $CWD &

    # Resume after crash
    python -m cpcv_pipeline.optimize_wf --cwd $CWD --continue
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

from cpcv_pipeline.function_train_test import (
    load_full_data,
    train_split,
)
from cpcv_pipeline.config import (
    DEFAULT_ERL_PARAMS, DEFAULT_ENV_PARAMS,
    RANDOM_SEED, GPU_ID, RESULTS_DIR,
    WF_GAP_DAYS,
)


def get_wf_splits(total_days, n_folds=5, gap_days=7):
    """
    Anchored walk-forward splits.

    Fold i: Train [0 : (i+1)*fold_size]
            Test  [(i+1)*fold_size + gap : (i+2)*fold_size + gap]

    Returns list of (train_indices, test_indices) np.ndarray pairs.
    """
    fold_size = total_days // (n_folds + 1)
    splits = []

    for fold_idx in range(n_folds):
        train_end = (fold_idx + 1) * fold_size
        test_start = train_end + gap_days
        test_end = min(test_start + fold_size, total_days)

        if test_end <= test_start:
            break

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)

        assert len(np.intersect1d(train_indices, test_indices)) == 0
        splits.append((train_indices, test_indices))

    return splits


def _parse_splits(spec, n_splits):
    """Parse split spec: '0-4', '0,2,5', or '0-2,4'."""
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
            raise ValueError(
                f"Split {i} out of range [0, {n_splits - 1}]"
            )
    return sorted(set(indices))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Training"
    )
    # Model & CV
    parser.add_argument("--model", type=str, default="ppo")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--gap", type=int, default=WF_GAP_DAYS)
    # Hardware
    parser.add_argument("--gpu", type=int, default=GPU_ID)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    # Hyperparameters
    parser.add_argument("--break-step", type=int,
                        default=DEFAULT_ERL_PARAMS['break_step'])
    parser.add_argument("--net-dims", type=str, default="128,64")
    parser.add_argument("--lr", type=float,
                        default=DEFAULT_ERL_PARAMS['learning_rate'])
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULT_ERL_PARAMS['batch_size'])
    parser.add_argument("--num-envs", type=int,
                        default=DEFAULT_ENV_PARAMS['num_envs'])
    # Execution control
    parser.add_argument("--split", type=int, default=None,
                        help="Run only this split (0-based)")
    parser.add_argument("--splits", type=str, default=None,
                        help="Range of splits, e.g. '0-2' or '0,2,4'")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable VecNormalize")
    parser.add_argument(
        "--continue", dest="continue_train", action="store_true",
        help="Resume from checkpoints (skip completed, "
             "reload partial)")
    parser.add_argument("--cwd", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    close_ary, tech_ary = load_full_data()
    total_days = close_ary.shape[0]

    print(f"\n{'='*60}")
    print("Walk-Forward Training Pipeline")
    print(f"{'='*60}")
    print(f"  Data: {total_days} days "
          f"x {close_ary.shape[1]} stocks")
    print(f"  Model: {args.model.upper()}")
    print(f"  Folds: {args.n_folds}, Gap: {args.gap}d")

    splits = get_wf_splits(total_days, args.n_folds, args.gap)
    n_splits = len(splits)

    print(f"\nSplit summary:")
    for i, (train_idx, test_idx) in enumerate(splits):
        marker = " <- selected" if args.split == i else ""
        print(
            f"  Fold {i}: "
            f"Train [0:{train_idx[-1]+1}] ({len(train_idx)}d)  "
            f"|  Test [{test_idx[0]}:{test_idx[-1]+1}] "
            f"({len(test_idx)}d){marker}")

    if args.dry_run:
        return 0

    # ── Output directory ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    net_dims = [int(x) for x in args.net_dims.split(",")]

    if args.cwd:
        cwd_base = args.cwd
    else:
        cwd_base = os.path.join(
            RESULTS_DIR,
            f"{timestamp}_WF_{args.model.upper()}"
            f"_seed{args.seed}"
        )
    os.makedirs(cwd_base, exist_ok=True)

    # ── Hyperparameters ──────────────────────────────────────────
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

    # ── Save config ──────────────────────────────────────────────
    run_config = {
        'cv_method': 'walk_forward',
        'n_folds': args.n_folds,
        'gap_days': args.gap,
        'total_days': total_days,
        'model': args.model,
        'seed': args.seed,
        'gpu_id': args.gpu,
        'erl_params': erl_params,
        'env_params': env_params,
        'use_vec_normalize': use_vec_normalize,
        'net_dims': net_dims,
    }
    with open(os.path.join(cwd_base, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2, default=str)

    # ── Which splits to run ──────────────────────────────────────
    if args.split is not None and args.splits is not None:
        print("ERROR: --split and --splits are mutually exclusive")
        return 1

    if args.split is not None:
        split_indices = [args.split]
    elif args.splits is not None:
        split_indices = _parse_splits(args.splits, n_splits)
    else:
        split_indices = list(range(n_splits))

    splits_to_run = [(i, splits[i]) for i in split_indices]

    # ── Train ────────────────────────────────────────────────────
    results = []
    for split_idx, (train_idx, test_idx) in splits_to_run:
        split_cwd = os.path.join(cwd_base, f"split_{split_idx}")

        # Skip / resume logic
        if args.continue_train:
            rec_path = os.path.join(split_cwd, 'recorder.npy')
            if os.path.exists(rec_path):
                rec = np.load(rec_path)
                if len(rec) > 0:
                    max_step = int(rec[-1, 0])
                    if max_step >= erl_params['break_step']:
                        best_i = int(np.argmax(rec[:, 1]))
                        print(
                            f"\n  [SKIP] Fold {split_idx}: "
                            f"done ({max_step:,} steps, "
                            f"avgR={rec[best_i,1]:.3f})")
                        results.append({
                            'split_idx': split_idx,
                            'cwd': split_cwd,
                            'train_days': len(train_idx),
                            'test_days': len(test_idx),
                            'model_name': args.model,
                            'seed': args.seed,
                            'best_step': int(rec[best_i, 0]),
                            'best_avgR': float(
                                rec[best_i, 1]),
                            'skipped': True,
                        })
                        continue
                    else:
                        print(
                            f"\n  [RESUME] Fold "
                            f"{split_idx}: partial "
                            f"({max_step:,} / "
                            f"{erl_params['break_step']:,})")

        print(f"\n{'#'*60}")
        print(f"# TRAINING FOLD {split_idx} / {n_splits}")
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
            continue_train=args.continue_train,
        )
        results.append(result)

        # Per-split result (multi-GPU safe)
        with open(os.path.join(
            cwd_base, f'result_split_{split_idx}.json'
        ), 'w') as f:
            json.dump(result, f, indent=2, default=str)
        with open(os.path.join(
            cwd_base, 'results.json'
        ), 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    for r in results:
        best_r = r.get('best_avgR', 'N/A')
        sk = ' (skipped)' if r.get('skipped') else ''
        print(f"  Fold {r['split_idx']}: "
              f"train={r['train_days']}d, "
              f"test={r['test_days']}d, "
              f"best_avgR={best_r}{sk}")
    print(f"\n  Results: {cwd_base}")
    print(f"  Eval: python -m cpcv_pipeline.eval_all_checkpoints"
          f" --results-dir {cwd_base}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
