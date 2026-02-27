#!/usr/bin/env python3
"""
Post-Training OOS Evaluation Script.

After training all CPCV splits, run this to evaluate each split's
best checkpoint on its test set and collect OOS returns for PBO analysis.

Usage:
    python -m cpcv_pipeline.evaluate_splits --results-dir train_results/20250101_CPCV_PPO_N5K2_seed1943
    python -m cpcv_pipeline.evaluate_splits --results-dir ... --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cpcv_pipeline.config import GPU_ID
from cpcv_pipeline.function_train_test import (
    load_full_data,
    evaluate_agent_on_indices,
)


def find_best_checkpoint(split_dir: str) -> str:
    """Find the best actor checkpoint in a split directory."""
    # Look for actor__*.pt files, pick the one with highest return
    actor_files = [
        f for f in os.listdir(split_dir)
        if f.startswith("actor__") and f.endswith(".pt")
    ]

    if actor_files:
        # Parse return from filename: actor__000000001200_00155.540.pt
        best_file = None
        best_return = -float('inf')
        for f in actor_files:
            try:
                parts = f.replace(".pt", "").split("_")
                ret_str = parts[-1]
                ret_val = float(ret_str)
                if ret_val > best_return:
                    best_return = ret_val
                    best_file = f
            except (ValueError, IndexError):
                continue
        if best_file:
            return os.path.join(split_dir, best_file)

    # Fallback: act.pth
    act_path = os.path.join(split_dir, "act.pth")
    if os.path.exists(act_path):
        return act_path

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CPCV splits post-training"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Path to CPCV results directory"
    )
    parser.add_argument("--gpu", type=int, default=GPU_ID)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate even if eval_results.json exists"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    close_ary, tech_ary = load_full_data()
    total_days = close_ary.shape[0]

    print(f"Data: {total_days} days x {close_ary.shape[1]} stocks")

    # Find all split directories
    split_dirs = sorted([
        d for d in os.listdir(args.results_dir)
        if d.startswith("split_")
        and os.path.isdir(os.path.join(args.results_dir, d))
    ])

    if not split_dirs:
        print(f"No split_* directories found in {args.results_dir}")
        return 1

    print(f"Found {len(split_dirs)} splits to evaluate")

    all_results = []

    for split_dir_name in split_dirs:
        split_dir = os.path.join(args.results_dir, split_dir_name)
        eval_path = os.path.join(split_dir, "eval_results.json")

        if os.path.exists(eval_path) and not args.force:
            print(f"  {split_dir_name}: eval_results.json exists, skipping "
                  f"(use --force to re-evaluate)")
            with open(eval_path) as f:
                all_results.append(json.load(f))
            continue

        # Load split metadata
        meta_path = os.path.join(split_dir, "split_meta.json")
        if not os.path.exists(meta_path):
            print(f"  {split_dir_name}: no split_meta.json, skipping")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        test_indices = np.array(meta['test_indices'])
        model_name = meta['model_name']
        net_dims = meta.get('erl_params', {}).get('net_dims', [128, 64])

        # Find best checkpoint
        checkpoint = find_best_checkpoint(split_dir)
        if checkpoint is None:
            print(f"  {split_dir_name}: no checkpoint found, skipping")
            continue

        print(f"\n  Evaluating {split_dir_name}:")
        print(f"    Checkpoint: {os.path.basename(checkpoint)}")
        print(f"    Test indices: [{test_indices[0]}..{test_indices[-1]}] "
              f"({len(test_indices)} days)")

        # Check for VecNormalize stats saved during training
        vec_norm_path = os.path.join(split_dir, 'vec_normalize.pt')
        if not os.path.exists(vec_norm_path):
            vec_norm_path = None

        try:
            result = evaluate_agent_on_indices(
                checkpoint_path=checkpoint,
                indices=test_indices,
                close_ary=close_ary,
                tech_ary=tech_ary,
                model_name=model_name,
                net_dims=net_dims,
                gpu_id=args.gpu,
                vec_normalize_path=vec_norm_path,
            )

            result['split_idx'] = meta['split_idx']
            result['test_indices'] = test_indices.tolist()
            result['checkpoint'] = os.path.basename(checkpoint)
            result['model_name'] = model_name

            # Save per-split results
            with open(eval_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            print(f"    Return: {result['final_return']*100:.2f}%")
            print(f"    Sharpe: {result['sharpe']:.4f}")
            print(f"    Max DD: {result['max_drawdown']*100:.2f}%")

            all_results.append(result)

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    summary_path = os.path.join(args.results_dir, "eval_summary.json")
    summary = {
        'n_splits_evaluated': len(all_results),
        'splits': [],
    }

    for r in all_results:
        summary['splits'].append({
            'split_idx': r.get('split_idx'),
            'final_return': r.get('final_return'),
            'sharpe': r.get('sharpe'),
            'max_drawdown': r.get('max_drawdown'),
        })

    if all_results:
        sharpes = [r.get('sharpe', 0) for r in all_results]
        returns = [r.get('final_return', 0) for r in all_results]
        summary['avg_sharpe'] = float(np.mean(sharpes))
        summary['avg_return'] = float(np.mean(returns))
        summary['std_sharpe'] = float(np.std(sharpes))

        print(f"\n{'='*60}")
        print(f"Evaluation Summary")
        print(f"{'='*60}")
        print(f"  Splits evaluated: {len(all_results)}")
        print(f"  Avg Sharpe: {summary['avg_sharpe']:.4f} "
              f"± {summary['std_sharpe']:.4f}")
        print(f"  Avg Return: {summary['avg_return']*100:.2f}%")

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
