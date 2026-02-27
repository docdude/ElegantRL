#!/usr/bin/env python3
"""
Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR) analysis.

Computes DSR for a single-config CPCV run using checkpoint_results.csv
from each split. No HPO required — checkpoints within each split are
treated as the "independent trials" for the multiple-testing correction.

Two modes:
  1. Quick (default): Uses only aggregate metrics from checkpoint_results.csv.
     Approximates daily return skew/kurtosis as 0/3 (normal assumption).
  2. Full (--full): Re-evaluates the best checkpoint per split to capture
     daily returns for exact skew/kurtosis.

Usage:
    # Quick DSR from existing CSVs (no GPU needed)
    python -m cpcv_pipeline.run_dsr \
        --results-dir train_results/20260224_062644_CPCV_PPO_N5K2_seed1943

    # Full DSR with exact skew/kurtosis (re-evaluates best checkpoints)
    python -m cpcv_pipeline.run_dsr --results-dir ... --full --gpu 0

    # Per-split breakdown
    python -m cpcv_pipeline.run_dsr --results-dir ... --per-split
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cpcv_pipeline.config import N_SPLITS


# ─────────────────────────────────────────────────────────────────────────────
# Core DSR / PSR functions (Bailey & López de Prado, 2014)
# ─────────────────────────────────────────────────────────────────────────────

def expected_max_sr(N: int) -> float:
    """E[max] of N iid standard normals.

    Bailey & López de Prado (2014), Eq. (4).
    Approximation valid for N >> 1 (requires N >= 5).
    """
    if N < 2:
        return 0.0
    if N < 5:
        # Small-N fallback: use exact E[max] for small samples
        # E[max(N)] ≈ Φ^{-1}(1 - 1/(N+1)) for small N
        return stats.norm.ppf(1 - 1.0 / (N + 1))
    return ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / N)
            + np.euler_gamma * stats.norm.ppf(1 - np.exp(-1) / N))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    T: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR).

    Prob(true SR > benchmark_sr | observed data).
    Bailey & López de Prado (2014), Eq. (1).

    Parameters
    ----------
    observed_sr : float
        Observed (annualised) Sharpe ratio.
    benchmark_sr : float
        Benchmark Sharpe (e.g. 0 for "better than flat").
    T : int
        Number of return observations.
    skew, kurtosis : float
        Sample skewness and kurtosis (excess) of returns.
        kurtosis=3 is normal (non-excess); use fisher=False.
    """
    denom_sq = (1.0
                - skew * observed_sr
                + observed_sr**2 * (kurtosis - 1) / 4.0)
    if denom_sq <= 0:
        return 0.5
    z = (observed_sr - benchmark_sr) * np.sqrt(T - 1) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_std: float,
    N: int,
    T: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """Deflated Sharpe Ratio (DSR).

    DSR = PSR(SR_0) where SR_0 = sr_std * E[max(N)].
    Tests whether the best-observed Sharpe survives the multiple-testing
    correction from having examined N strategy variants.

    Parameters
    ----------
    observed_sr : float
        Best observed (annualised) Sharpe ratio.
    sr_std : float
        Std of all observed Sharpe ratios.
    N : int
        Number of independent trials/strategies tested.
    T : int
        Number of return observations for the best strategy.
    skew, kurtosis : float
        Return distribution moments.
    """
    e_max = expected_max_sr(N)
    target_sr = sr_std * e_max

    dsr = probabilistic_sharpe_ratio(observed_sr, target_sr, T, skew, kurtosis)

    # Also compute PSR vs 0 (no benchmark)
    psr_vs_zero = probabilistic_sharpe_ratio(observed_sr, 0.0, T, skew, kurtosis)

    # MinTRL: minimum track record length to achieve 95% PSR
    # Solved from PSR formula: T* s.t. PSR(SR, 0, T*, skew, kurt) = 0.95
    z_95 = stats.norm.ppf(0.95)
    denom_inner = (1.0
                   - skew * observed_sr
                   + observed_sr**2 * (kurtosis - 1) / 4.0)
    if observed_sr > 1e-8 and denom_inner > 0:
        min_trl = 1 + (z_95 / observed_sr) ** 2 * denom_inner
    else:
        min_trl = float('inf')

    interpretation = (
        "PASS — strategy likely not overfit"
        if dsr > 0.95 else
        "MARGINAL — some overfitting risk"
        if dsr > 0.5 else
        "FAIL — high overfitting risk"
    )

    return {
        'dsr': dsr,
        'psr_vs_zero': psr_vs_zero,
        'observed_sr': observed_sr,
        'target_sr': target_sr,
        'e_max_N': e_max,
        'sr_std': sr_std,
        'N': N,
        'T': T,
        'skewness': skew,
        'kurtosis': kurtosis,
        'min_trl': min_trl,
        'interpretation': interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint data
# ─────────────────────────────────────────────────────────────────────────────

def load_all_checkpoint_sharpes(results_dir: str) -> dict:
    """Load Sharpe ratios from all splits' checkpoint_results.csv.

    Returns dict with:
      - per_split: {split_idx: {'sharpes': [...], 'best_sharpe': ..., ...}}
      - all_sharpes: flat array of all Sharpe ratios across all splits
      - best_overall: global best
    """
    per_split = {}
    all_sharpes = []

    for split_idx in range(N_SPLITS):
        split_dir = os.path.join(results_dir, f"split_{split_idx}")
        csv_path = os.path.join(split_dir, "checkpoint_results.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping split {split_idx}")
            continue

        df = pd.read_csv(csv_path)
        if df.empty or 'sharpe' not in df.columns:
            continue

        sharpes = df['sharpe'].values
        best_idx = df['sharpe'].idxmax()
        best_row = df.loc[best_idx]

        per_split[split_idx] = {
            'sharpes': sharpes.tolist(),
            'n_checkpoints': len(sharpes),
            'best_sharpe': float(best_row['sharpe']),
            'best_checkpoint': str(best_row['checkpoint']),
            'best_return': float(best_row['final_return']),
            'best_alpha': float(best_row.get('alpha', 0)),
            'test_days': None,  # filled below if meta exists
        }
        all_sharpes.extend(sharpes.tolist())

        # Get test set size from metadata
        meta_path = os.path.join(split_dir, "split_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            per_split[split_idx]['test_days'] = len(meta.get('test_indices', []))

    all_sharpes = np.array(all_sharpes)
    best_overall_idx = np.argmax(all_sharpes)

    return {
        'per_split': per_split,
        'all_sharpes': all_sharpes,
        'n_total_checkpoints': len(all_sharpes),
        'best_overall_sharpe': float(all_sharpes[best_overall_idx]),
        'sharpe_std': float(np.std(all_sharpes, ddof=1)) if len(all_sharpes) > 1 else 0.0,
        'n_splits_found': len(per_split),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI and main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deflated Sharpe Ratio (DSR) analysis for CPCV results"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Path to CPCV results directory (contains split_0/ ... split_9/)"
    )
    parser.add_argument(
        "--per-split", action="store_true",
        help="Show per-split DSR breakdown"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Re-evaluate best checkpoint to get exact skew/kurtosis "
             "(slower, requires GPU)"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device for --full mode (-1 for CPU)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save results to JSON (default: results_dir/dsr_results.json)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir

    print(f"\n{'='*65}")
    print(f"Deflated Sharpe Ratio (DSR) Analysis")
    print(f"{'='*65}")
    print(f"  Results dir: {results_dir}")

    # Load all checkpoint Sharpe ratios
    data = load_all_checkpoint_sharpes(results_dir)

    if data['n_splits_found'] == 0:
        print("ERROR: No checkpoint_results.csv found. Run eval_all_checkpoints first.")
        return 1

    print(f"  Splits found: {data['n_splits_found']}")
    print(f"  Total checkpoints: {data['n_total_checkpoints']}")
    print(f"  Best overall Sharpe: {data['best_overall_sharpe']:.4f}")
    print(f"  Sharpe std: {data['sharpe_std']:.4f}")

    # ── Global DSR (all checkpoints across all splits as "trials") ────
    # This answers: "Given we tested N checkpoints across 10 splits,
    # does our best Sharpe survive the multiple-testing correction?"

    # Find which split has the best checkpoint for T
    best_split = max(data['per_split'].items(),
                     key=lambda x: x[1]['best_sharpe'])
    best_split_idx = best_split[0]
    best_split_data = best_split[1]
    T = best_split_data.get('test_days', 300) or 300

    # Default: assume normal returns (quick mode)
    skew, kurt = 0.0, 3.0
    # Cache per-split daily returns to avoid re-evaluating a checkpoint twice
    _cached_daily_returns = {}  # split_idx -> np.ndarray

    if args.full:
        print(f"\n  Re-evaluating best checkpoint for exact moments...")
        dr = _get_daily_returns_for_best(
            results_dir, best_split_idx,
            best_split_data['best_checkpoint'],
            args.gpu
        )
        if dr is not None:
            skew = float(stats.skew(dr))
            kurt = float(stats.kurtosis(dr, fisher=False))
            T = len(dr)
            _cached_daily_returns[best_split_idx] = dr
            print(f"    Daily returns: {T} days, skew={skew:.4f}, kurt={kurt:.4f}")
        else:
            print(f"    Re-evaluation failed, using normal assumption")

    print(f"\n{'─'*65}")
    print(f"  GLOBAL DSR (all {data['n_total_checkpoints']} checkpoints "
          f"across {data['n_splits_found']} splits)")
    print(f"{'─'*65}")

    global_dsr = deflated_sharpe_ratio(
        observed_sr=data['best_overall_sharpe'],
        sr_std=data['sharpe_std'],
        N=data['n_total_checkpoints'],
        T=T,
        skew=skew,
        kurtosis=kurt,
    )

    _print_dsr_results(global_dsr)

    results = {'global': global_dsr}

    # ── Per-split DSR ─────────────────────────────────────────────────
    if args.per_split:
        print(f"\n{'─'*65}")
        print(f"  PER-SPLIT DSR")
        print(f"{'─'*65}")

        split_results = {}
        for split_idx in sorted(data['per_split'].keys()):
            sp = data['per_split'][split_idx]
            sharpes = np.array(sp['sharpes'])
            n = sp['n_checkpoints']
            T_split = sp.get('test_days', 300) or 300
            sr_std = float(np.std(sharpes, ddof=1)) if n > 1 else 0.0

            split_skew, split_kurt = 0.0, 3.0
            if args.full:
                if split_idx in _cached_daily_returns:
                    dr = _cached_daily_returns[split_idx]
                else:
                    dr = _get_daily_returns_for_best(
                        results_dir, split_idx,
                        sp['best_checkpoint'],
                        args.gpu,
                    )
                if dr is not None:
                    split_skew = float(stats.skew(dr))
                    split_kurt = float(stats.kurtosis(dr, fisher=False))
                    T_split = len(dr)

            split_dsr = deflated_sharpe_ratio(
                observed_sr=sp['best_sharpe'],
                sr_std=sr_std,
                N=n,
                T=T_split,
                skew=split_skew,
                kurtosis=split_kurt,
            )
            split_results[split_idx] = split_dsr

            status = "✓" if split_dsr['dsr'] > 0.95 else "~" if split_dsr['dsr'] > 0.5 else "✗"
            moments = (f"  skew={split_skew:+.2f} kurt={split_kurt:.1f}"
                       if args.full else "")
            print(f"    Split {split_idx}: DSR={split_dsr['dsr']:.3f} "
                  f"[{status}]  SR={sp['best_sharpe']:.2f}  "
                  f"SR₀={split_dsr['target_sr']:.2f}  "
                  f"N={n}  T={T_split}{moments}  "
                  f"({sp['best_checkpoint']})")

        results['per_split'] = {str(k): v for k, v in split_results.items()}

        # Summary
        dsrs = [v['dsr'] for v in split_results.values()]
        print(f"\n    Mean DSR: {np.mean(dsrs):.3f}")
        print(f"    Splits passing (>0.95): "
              f"{sum(1 for d in dsrs if d > 0.95)}/{len(dsrs)}")
        print(f"    Splits marginal (0.5-0.95): "
              f"{sum(1 for d in dsrs if 0.5 < d <= 0.95)}/{len(dsrs)}")
        print(f"    Splits failing (<0.5): "
              f"{sum(1 for d in dsrs if d <= 0.5)}/{len(dsrs)}")

    # Save
    save_path = args.save or os.path.join(results_dir, "dsr_results.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")

    return 0


def _print_dsr_results(r: dict):
    """Pretty-print DSR results."""
    print(f"    Observed Sharpe    = {r['observed_sr']:.4f}")
    print(f"    Target Sharpe SR₀  = {r['target_sr']:.4f}  "
          f"(σ(SR)={r['sr_std']:.4f} × E[max({r['N']})]={r['e_max_N']:.4f})")
    print(f"    PSR(SR > 0)        = {r['psr_vs_zero']:.4f}")
    print(f"    DSR(SR > SR₀)      = {r['dsr']:.4f}")
    print(f"    Min Track Record   = {r['min_trl']:.0f} days "
          f"({'met ✓' if r['T'] >= r['min_trl'] else 'NOT met ✗'})")
    print(f"    Skew={r['skewness']:.3f}  Kurt={r['kurtosis']:.3f}  "
          f"T={r['T']}  N={r['N']}")
    print(f"    → {r['interpretation']}")


def _get_daily_returns_for_best(
    results_dir: str,
    split_idx: int,
    checkpoint_name: str,
    gpu_id: int,
) -> Optional[np.ndarray]:
    """Re-evaluate one checkpoint to get daily returns for skew/kurtosis.

    Reuses ``evaluate_single_checkpoint`` from eval_all_checkpoints.
    """
    try:
        import torch as th
        from elegantrl.envs.StockTradingEnv import StockTradingVecEnv
        from elegantrl.envs.vec_normalize import VecNormalize
        from cpcv_pipeline.config import DEFAULT_ENV_PARAMS, DEFAULT_ERL_PARAMS
        from cpcv_pipeline.function_train_test import (
            load_full_data, save_sliced_data, DATA_CACHE_DIR,
        )
        from cpcv_pipeline.eval_all_checkpoints import (
            load_dates, _split_contiguous_segments, evaluate_single_checkpoint,
        )

        split_dir = os.path.join(results_dir, f"split_{split_idx}")
        ckpt_path = os.path.join(split_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            return None

        with open(os.path.join(split_dir, "split_meta.json")) as f:
            meta = json.load(f)
        test_indices = np.array(meta['test_indices'])

        close_ary, tech_ary = load_full_data()
        all_dates = load_dates()

        initial_amount = DEFAULT_ENV_PARAMS.get('initial_amount', 1e6)
        gamma = DEFAULT_ERL_PARAMS.get('gamma', 0.99)
        device = f'cuda:{gpu_id}' if gpu_id >= 0 and th.cuda.is_available() else 'cpu'

        vec_norm_path = os.path.join(split_dir, 'vec_normalize.pt')
        has_vec_norm = os.path.exists(vec_norm_path)

        # Build segment envs (same as evaluate_split)
        segments = _split_contiguous_segments(test_indices)
        segment_envs = []
        tmp_npzs = []

        for seg_idx, seg_indices in enumerate(segments):
            num_days = len(seg_indices)
            seg_npz = os.path.join(DATA_CACHE_DIR, "cpcv_splits",
                                   f"_dsr_eval_seg{seg_idx}_{os.getpid()}.npz")
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

        # Evaluate checkpoint
        result = evaluate_single_checkpoint(
            checkpoint_path=ckpt_path,
            segment_envs=segment_envs,
            initial_amount=initial_amount,
            device=device,
        )

        # Cleanup
        for p in tmp_npzs:
            if os.path.exists(p):
                os.remove(p)

        if result is None:
            return None

        # Derive daily returns from account_values
        av = result['account_values']
        return np.diff(av) / (av[:-1] + 1e-9)

    except Exception as e:
        print(f"    Re-evaluation error: {e}")
        return None


if __name__ == "__main__":
    sys.exit(main())
