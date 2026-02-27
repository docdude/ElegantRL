#!/usr/bin/env python3
"""
Per-split PBO: Checkpoint Selection Overfitting Analysis.

For each CPCV split, treats every saved checkpoint as an independent
"strategy" and runs CSCV-PBO to test whether selecting the best checkpoint
(early-stopping) leads to overfitting.

This answers: "Am I overfitting by picking the best checkpoint?"

Matrix M per split: (T_test, N_checkpoints)
  - Rows = OOS test days for that split (~300)
  - Columns = one checkpoint's daily returns on that test set

This is NOT the same as FinRL_Crypto's PBO (which uses HPO trials as
columns). This tests checkpoint/early-stopping selection overfitting,
not hyperparameter overfitting.

Requires re-evaluating ALL checkpoints per split to capture daily returns.
Each checkpoint evaluation is fast (~300 days of inference).

Usage:
    python -m cpcv_pipeline.run_checkpoint_pbo \
        --results-dir train_results/20260224_062644_CPCV_PPO_N5K2_seed1943

    # Single split only (faster for testing)
    python -m cpcv_pipeline.run_checkpoint_pbo --results-dir ... --split 0

    # Adjust CSCV chunks
    python -m cpcv_pipeline.run_checkpoint_pbo --results-dir ... --S 8
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "lopez_de_prado_analysis", "pypbo"))

from cpcv_pipeline.config import (
    N_SPLITS, DEFAULT_ENV_PARAMS, DEFAULT_ERL_PARAMS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_metric(returns: np.ndarray) -> np.ndarray:
    """Annualised Sharpe ratio for each column."""
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    mean_r = np.mean(returns, axis=0)
    std_r = np.std(returns, axis=0, ddof=1)
    std_r[std_r < 1e-10] = 1e-10
    sharpe = mean_r / std_r * np.sqrt(252)
    if sharpe.shape[0] == 1:
        return float(sharpe[0])
    return sharpe


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate all checkpoints → M matrix for one split
# ─────────────────────────────────────────────────────────────────────────────

def build_M_for_split(
    split_dir: str,
    close_ary: np.ndarray,
    tech_ary: np.ndarray,
    all_dates: np.ndarray,
    gpu_id: int = 0,
    cache: bool = True,
) -> Tuple[np.ndarray, list, dict]:
    """Build M = (T_test, N_checkpoints) for one split.

    Re-evaluates every checkpoint on the OOS test set and collects
    daily returns.  Reuses ``evaluate_single_checkpoint`` from
    eval_all_checkpoints to avoid code duplication.

    Caches results as ``checkpoint_daily_returns.npz``.

    Returns
    -------
    M : ndarray (T_test, N_checkpoints)
    checkpoint_names : list of str
    meta : dict with split info
    """
    import torch as th
    from elegantrl.envs.StockTradingEnv import StockTradingVecEnv
    from elegantrl.envs.vec_normalize import VecNormalize
    from cpcv_pipeline.function_train_test import save_sliced_data, DATA_CACHE_DIR
    from cpcv_pipeline.eval_all_checkpoints import (
        _split_contiguous_segments,
        evaluate_single_checkpoint,
        discover_checkpoints as discover_ckpts,
    )

    split_name = os.path.basename(split_dir)
    cache_path = os.path.join(split_dir, "checkpoint_daily_returns.npz")

    # ── Check cache ───────────────────────────────────────────────────
    if cache and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        M = cached['M']
        names = cached['names'].tolist()
        print(f"  {split_name}: loaded cached M {M.shape} "
              f"({len(names)} checkpoints)")
        meta_path = os.path.join(split_dir, "split_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        return M, names, meta

    # ── Load split metadata ───────────────────────────────────────────
    meta_path = os.path.join(split_dir, "split_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No split_meta.json in {split_dir}")
    with open(meta_path) as f:
        meta = json.load(f)
    test_indices = np.array(meta['test_indices'])

    # ── Discover checkpoints ──────────────────────────────────────────
    checkpoints = discover_ckpts(split_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {split_dir}")

    print(f"  {split_name}: {len(checkpoints)} checkpoints, "
          f"{len(test_indices)} test days")

    # ── Build per-segment environments (mirrors evaluate_split) ───────
    initial_amount = DEFAULT_ENV_PARAMS.get('initial_amount', 1e6)
    gamma = DEFAULT_ERL_PARAMS.get('gamma', 0.99)
    device = f'cuda:{gpu_id}' if gpu_id >= 0 and th.cuda.is_available() else 'cpu'

    segments = _split_contiguous_segments(test_indices)
    vec_norm_path = os.path.join(split_dir, 'vec_normalize.pt')
    has_vec_norm = os.path.exists(vec_norm_path)

    segment_envs = []
    tmp_npzs = []
    for seg_idx, seg_indices in enumerate(segments):
        num_days = len(seg_indices)
        seg_npz = os.path.join(
            DATA_CACHE_DIR, "cpcv_splits",
            f"_pbo_ckpt_{split_name}_seg{seg_idx}_{os.getpid()}.npz",
        )
        save_sliced_data(seg_indices, seg_npz, close_ary, tech_ary,
                         dates_ary=all_dates)
        tmp_npzs.append(seg_npz)

        env = StockTradingVecEnv(
            npz_path=seg_npz,
            initial_amount=initial_amount,
            max_stock=DEFAULT_ENV_PARAMS.get('max_stock', 1e2),
            cost_pct=DEFAULT_ENV_PARAMS.get('cost_pct', 1e-3),
            gamma=gamma, beg_idx=0, end_idx=num_days,
            num_envs=1, gpu_id=gpu_id,
        )
        env.if_random_reset = False

        if has_vec_norm:
            env = VecNormalize(env, training=False)
            env.load(vec_norm_path, verbose=False)

        segment_envs.append({'env': env, 'num_days': num_days})

    # ── Evaluate each checkpoint via evaluate_single_checkpoint ───────
    M_cols = []
    valid_names = []

    for ckpt in checkpoints:
        result = evaluate_single_checkpoint(
            checkpoint_path=ckpt['path'],
            segment_envs=segment_envs,
            initial_amount=initial_amount,
            device=device,
        )
        if result is None:
            continue

        # Derive daily returns from account_values
        av = result['account_values']
        daily_returns = np.diff(av) / (av[:-1] + 1e-9)
        M_cols.append(daily_returns)
        valid_names.append(ckpt['filename'])

        sr = sharpe_metric(daily_returns)
        ret = result['final_return'] * 100
        avgR_str = f"avgR={ckpt['avgR']:.1f}" if ckpt['avgR'] else "periodic"
        print(f"    {ckpt['filename']:<45} SR={sr:>6.2f}  "
              f"Ret={ret:>+7.2f}%  ({avgR_str})")

    # ── Cleanup temp NPZs ────────────────────────────────────────────
    for p in tmp_npzs:
        if os.path.exists(p):
            os.remove(p)

    if not M_cols:
        raise RuntimeError(f"No valid checkpoints evaluated in {split_dir}")

    # All columns should have equal length (same test set)
    M = np.column_stack(M_cols)

    # ── Cache ─────────────────────────────────────────────────────────
    if cache:
        np.savez_compressed(cache_path, M=M, names=np.array(valid_names))
        print(f"    Cached → {cache_path}")

    return M, valid_names, meta


# ─────────────────────────────────────────────────────────────────────────────
# PBO analysis on one split's M
# ─────────────────────────────────────────────────────────────────────────────

def run_split_pbo(
    M: np.ndarray,
    checkpoint_names: list,
    split_name: str,
    S: int = 8,
) -> dict:
    """Run CSCV-PBO on a single split's checkpoint matrix.

    Parameters
    ----------
    M : ndarray (T, N_checkpoints)
    checkpoint_names : list, length N_checkpoints
    split_name : str
    S : int, CSCV chunks (must be even)
    """
    T, N = M.shape

    print(f"\n  PBO for {split_name}: {T} days × {N} checkpoints, S={S}")

    if N < 3:
        print(f"    SKIP: only {N} checkpoints, need ≥ 3")
        return {'error': f'too_few_checkpoints_{N}', 'N': N}

    # Adjust S
    while T // S < 2 and S > 2:
        S -= 2
    if S < 2:
        return {'error': 'insufficient_data', 'T': T}

    # Import pypbo
    try:
        from pypbo.pbo import pbo as pbo_func
    except ImportError:
        sys.path.insert(0, os.path.join(
            PROJECT_ROOT, "lopez_de_prado_analysis", "pypbo"
        ))
        from pypbo.pbo import pbo as pbo_func

    pbo_result = pbo_func(
        M=M, S=S,
        metric_func=sharpe_metric,
        threshold=0.0,
        n_jobs=1, verbose=False, plot=False, hist=False,
    )

    logits = np.array(pbo_result.logits)
    lm = pbo_result.linear_model
    col_sharpes = sharpe_metric(M)
    if isinstance(col_sharpes, (int, float)):
        col_sharpes = np.array([col_sharpes])

    best_is_idx = int(np.argmax(col_sharpes))
    best_ckpt = checkpoint_names[best_is_idx]

    interpretation = (
        "LOW overfitting risk"
        if pbo_result.pbo < 0.30 else
        "MODERATE overfitting risk"
        if pbo_result.pbo < 0.50 else
        "HIGH overfitting risk"
        if pbo_result.pbo < 0.70 else
        "VERY HIGH overfitting risk"
    )

    print(f"    PBO = {pbo_result.pbo*100:.1f}%  — {interpretation}")
    print(f"    Prob(OOS loss) = {pbo_result.prob_oos_loss*100:.1f}%")
    print(f"    Logits: mean={logits.mean():.3f}, "
          f"min={logits.min():.3f}, max={logits.max():.3f}")
    print(f"    Perf. degradation: slope={lm.slope:.4f}, "
          f"R²={lm.rvalue**2:.4f}")
    print(f"    Best checkpoint: {best_ckpt} (SR={col_sharpes[best_is_idx]:.3f})")

    return {
        'split': split_name,
        'pbo': float(pbo_result.pbo),
        'pbo_pct': float(pbo_result.pbo * 100),
        'prob_oos_loss': float(pbo_result.prob_oos_loss),
        'logits_mean': float(logits.mean()),
        'logits_std': float(logits.std()),
        'logits_min': float(logits.min()),
        'logits_max': float(logits.max()),
        'perf_degradation_slope': float(lm.slope),
        'perf_degradation_r2': float(lm.rvalue**2),
        'perf_degradation_p': float(lm.pvalue),
        'n_checkpoints': N,
        'T': T,
        'S': S,
        'best_checkpoint': best_ckpt,
        'best_sharpe': float(col_sharpes[best_is_idx]),
        'all_sharpes': col_sharpes.tolist(),
        'interpretation': interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-split PBO on checkpoint selection (early-stopping overfitting)"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Path to CPCV results directory"
    )
    parser.add_argument(
        "--split", type=int, default=None,
        help="Evaluate only this split (0-based)"
    )
    parser.add_argument(
        "--S", type=int, default=8,
        help="CSCV chunks (must be even, default 8)"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device (-1 for CPU)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-evaluation (ignore cached checkpoint_daily_returns.npz)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save JSON results (default: results_dir/checkpoint_pbo_results.json)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir

    # Load dataset once
    from cpcv_pipeline.function_train_test import load_full_data
    from cpcv_pipeline.eval_all_checkpoints import load_dates

    print(f"\n{'='*65}")
    print(f"Per-Split Checkpoint PBO Analysis")
    print(f"{'='*65}")
    print(f"  Testing: does selecting the best checkpoint overfit?")
    print(f"  Each checkpoint = one 'strategy' in the CSCV framework\n")

    print("Loading dataset ...")
    close_ary, tech_ary = load_full_data()
    all_dates = load_dates()
    print(f"  {close_ary.shape[0]} days, {close_ary.shape[1]} stocks\n")

    # Which splits to evaluate
    if args.split is not None:
        split_indices = [args.split]
    else:
        split_indices = list(range(N_SPLITS))

    all_results = {}
    for split_idx in split_indices:
        split_dir = os.path.join(results_dir, f"split_{split_idx}")
        if not os.path.isdir(split_dir):
            print(f"  split_{split_idx}: directory not found, skipping")
            continue

        M, names, meta = build_M_for_split(
            split_dir=split_dir,
            close_ary=close_ary,
            tech_ary=tech_ary,
            all_dates=all_dates,
            gpu_id=args.gpu,
            cache=not args.no_cache,
        )

        result = run_split_pbo(M, names, f"split_{split_idx}", S=args.S)
        all_results[f"split_{split_idx}"] = result

    # Summary
    valid = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid:
        print(f"\n{'='*65}")
        print(f"SUMMARY")
        print(f"{'='*65}")
        print(f"  {'Split':<10} {'PBO%':>6} {'P(loss)':>8} {'Slope':>7} "
              f"{'R²':>6} {'N_ckpt':>7} {'Best SR':>8} Interpretation")
        print(f"  {'─'*75}")
        for name, r in sorted(valid.items()):
            print(f"  {name:<10} {r['pbo_pct']:>5.1f}% "
                  f"{r['prob_oos_loss']*100:>7.1f}% "
                  f"{r['perf_degradation_slope']:>+7.3f} "
                  f"{r['perf_degradation_r2']:>5.3f} "
                  f"{r['n_checkpoints']:>7} "
                  f"{r['best_sharpe']:>8.3f}  "
                  f"{r['interpretation']}")

        pbos = [v['pbo'] for v in valid.values()]
        print(f"\n  Mean PBO: {np.mean(pbos)*100:.1f}%")
        print(f"  Low risk (<30%): {sum(1 for p in pbos if p < 0.30)}/{len(pbos)}")
        print(f"  Moderate (30-50%): {sum(1 for p in pbos if 0.30 <= p < 0.50)}/{len(pbos)}")
        print(f"  High (>50%): {sum(1 for p in pbos if p >= 0.50)}/{len(pbos)}")

    # Save
    save_path = args.save or os.path.join(results_dir, "checkpoint_pbo_results.json")
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
