#!/usr/bin/env python3
"""
Probability of Backtest Overfitting (PBO) Analysis.

This script performs PBO analysis on CPCV training results following
Lopez de Prado (2018) and the FinRL_Crypto implementation.

Two modes:
  1. Single-config mode: Analyze OOS returns from CPCV backtest paths
     (limited — only N_PATHS columns in M).
  2. HPO mode: Build matrix M from Hydra/Hypersweeper HPO trials, each trained
     across all CPCV splits. Each column = one HPO trial's OOS returns.

Usage:
    # Single config (from CPCV results directory)
    python -m cpcv_pipeline.run_pbo --results-dir train_results/20250101_CPCV_PPO_N5K2_seed1943

    # HPO mode (from Optuna study pickle)
    python -m cpcv_pipeline.run_pbo --study-pkl train_results/study.pkl

    # DSR analysis only
    python -m cpcv_pipeline.run_pbo --results-dir ... --dsr-only
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import local PBO library
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lopez_de_prado_analysis", "pypbo"))

from cpcv_pipeline.config import N_GROUPS, K_TEST_GROUPS, N_PATHS, N_SPLITS
from cpcv_pipeline.function_CPCV import back_test_paths_generator


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions for PBO
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_metric(returns: np.ndarray) -> np.ndarray:
    """
    PBO metric function: compute Sharpe ratio for each column.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N) or (T,)
        Daily returns for N strategies over T time steps.

    Returns
    -------
    sharpe : np.ndarray, shape (N,) or scalar
        Annualized Sharpe ratios.
    """
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    mean_r = np.mean(returns, axis=0)
    std_r = np.std(returns, axis=0, ddof=1)
    std_r[std_r < 1e-10] = 1e-10
    sharpe = mean_r / std_r * np.sqrt(252)
    if sharpe.shape[0] == 1:
        return float(sharpe[0])
    return sharpe


def sortino_metric(returns: np.ndarray) -> np.ndarray:
    """Sortino ratio metric for PBO."""
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    mean_r = np.mean(returns, axis=0)
    downside = np.where(returns < 0, returns, 0.0)
    downside_std = np.std(downside, axis=0, ddof=1)
    downside_std[downside_std < 1e-10] = 1e-10
    sortino = mean_r / downside_std * np.sqrt(252)
    if sortino.shape[0] == 1:
        return float(sortino[0])
    return sortino


# ─────────────────────────────────────────────────────────────────────────────
# Matrix M construction
# ─────────────────────────────────────────────────────────────────────────────

def build_matrix_M_from_paths(
    results_dir: str,
    total_days: int,
    n_groups: int = N_GROUPS,
    n_test_groups: int = K_TEST_GROUPS,
) -> Tuple[np.ndarray, dict]:
    """
    Build PBO matrix M from a single CPCV run's backtest paths.

    Each CPCV path provides a complete OOS return series.
    M = (T x n_paths) where each column is one path's daily returns.

    Returns
    -------
    M : np.ndarray, shape (T, n_paths)
    metadata : dict
    """
    # Load evaluation results per split
    is_test, paths, path_folds = back_test_paths_generator(
        total_days, n_groups, n_test_groups
    )
    n_paths = paths.shape[1]

    # Load account values per split from results
    split_results = {}
    for split_dir in sorted(os.listdir(results_dir)):
        if not split_dir.startswith("split_"):
            continue
        split_idx = int(split_dir.split("_")[1])
        meta_path = os.path.join(results_dir, split_dir, "split_meta.json")
        eval_path = os.path.join(results_dir, split_dir, "eval_results.json")

        if os.path.exists(eval_path):
            with open(eval_path) as f:
                split_results[split_idx] = json.load(f)
        elif os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            split_results[split_idx] = meta

    if not split_results:
        raise FileNotFoundError(
            f"No split results found in {results_dir}. "
            f"Run evaluate_splits.py first."
        )

    # For each path, assemble OOS returns from the contributing splits
    M_cols = []
    for p in range(n_paths):
        path_returns = np.zeros(total_days)
        for t in range(total_days):
            split_idx = int(paths[t, p])
            # The daily return for day t comes from split split_idx's test eval
            # This requires per-day returns stored in eval_results
            if split_idx in split_results:
                eval_data = split_results[split_idx]
                if 'daily_returns' in eval_data:
                    # Map back to the position in the test set
                    test_idx = np.array(eval_data.get('test_indices', []))
                    pos = np.searchsorted(test_idx, t)
                    if pos < len(eval_data['daily_returns']):
                        path_returns[t] = eval_data['daily_returns'][pos]
        M_cols.append(path_returns)

    M = np.column_stack(M_cols)
    metadata = {
        'n_paths': n_paths,
        'total_days': total_days,
        'n_groups': n_groups,
        'source': 'cpcv_paths',
    }
    return M, metadata


def build_matrix_M_from_hpo_trials(
    pbo_returns_dir: str,
) -> Tuple[np.ndarray, dict]:
    """
    Build PBO matrix M from Hydra/Hypersweeper HPO trial results.

    Each HPO trial trained on all CPCV splits and saved per-split OOS
    daily returns as .npy files in pbo_returns_dir:
        trial_<id>_split_<idx>.npy

    We average OOS returns across splits for each trial → one column
    per trial.  M = (T x N_trials).
    """
    import re

    ret_files = sorted([
        f for f in os.listdir(pbo_returns_dir) if f.endswith('.npy')
    ])
    if not ret_files:
        raise ValueError(f"No .npy return files found in {pbo_returns_dir}")

    # Group by trial id
    trial_splits = {}  # trial_id -> [array, ...]
    pat = re.compile(r'trial_(.+?)_split_(\d+)\.npy')
    for fname in ret_files:
        m = pat.match(fname)
        if m:
            tid = m.group(1)
            arr = np.load(os.path.join(pbo_returns_dir, fname))
            trial_splits.setdefault(tid, []).append(arr)

    n_trials = len(trial_splits)
    print(f"  Loaded {n_trials} HPO trials from {pbo_returns_dir}")

    M_cols = []
    for tid in sorted(trial_splits.keys()):
        rets_list = trial_splits[tid]
        rets_equalized = _equalize_array_lengths(rets_list)
        avg_rets = np.mean(rets_equalized, axis=1)  # (T,)
        M_cols.append(avg_rets)

    if not M_cols:
        raise ValueError("No valid trial data found")

    M = np.column_stack(M_cols)
    metadata = {
        'n_trials': len(M_cols),
        'total_rows': M.shape[0],
        'source': 'hypersweeper_trials',
    }
    return M, metadata


def build_matrix_M_from_json_results(
    results_dir: str,
) -> Tuple[np.ndarray, dict]:
    """
    Build PBO matrix M from JSON result files saved by our pipeline.

    Each subdirectory (e.g., trial_0/, trial_1/) contains per-split eval
    results with daily returns. This is the non-Hypersweeper path.
    """
    trial_dirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
        and d.startswith("trial_")
    ])

    if not trial_dirs:
        # Fall back to single-run path mode
        return None, None

    M_cols = []
    for trial_dir in trial_dirs:
        trial_path = os.path.join(results_dir, trial_dir, "oos_returns.npy")
        if os.path.exists(trial_path):
            rets = np.load(trial_path)
            M_cols.append(rets)

    if not M_cols:
        return None, None

    M = np.column_stack(M_cols)
    metadata = {
        'n_trials': len(M_cols),
        'total_rows': M.shape[0],
        'source': 'json_results',
    }
    return M, metadata


def _equalize_array_lengths(arrays: list) -> np.ndarray:
    """
    Pad shorter arrays with their last element to match the longest.

    Returns (max_len, n_arrays) array.
    """
    max_len = max(len(a) for a in arrays)
    result = np.empty((max_len, len(arrays)))
    for i, a in enumerate(arrays):
        a = np.array(a)
        padded = np.pad(a, (0, max_len - len(a)), mode='edge')
        result[:, i] = padded
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DSR / PSR Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_dsr_analysis(
    sharpe_ratios: np.ndarray,
    daily_returns: np.ndarray,
    n_trials: int,
) -> dict:
    """
    Compute Deflated Sharpe Ratio for the best strategy.

    Parameters
    ----------
    sharpe_ratios : np.ndarray
        Sharpe ratios from all trials/paths.
    daily_returns : np.ndarray
        Daily returns of the best strategy.
    n_trials : int
        Total number of trials/paths tested.

    Returns
    -------
    results : dict
    """
    from scipy import stats

    best_idx = np.argmax(sharpe_ratios)
    best_sharpe = sharpe_ratios[best_idx]
    T = len(daily_returns)
    skew = float(stats.skew(daily_returns))
    kurtosis = float(stats.kurtosis(daily_returns, fisher=False))  # excess kurtosis

    # Expected maximum Sharpe under null (all configs iid)
    sharpe_std = np.std(sharpe_ratios, ddof=1)

    # E[max] of N iid standard normals
    N = n_trials
    if N >= 5:
        from scipy.special import erfinv
        e_max = ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / N)
                 + np.euler_gamma * stats.norm.ppf(1 - np.exp(-1) / N))
    else:
        e_max = 1.0  # conservative fallback

    target_sharpe = sharpe_std * e_max

    # PSR
    psr_denom = np.sqrt(
        1.0 - skew * best_sharpe
        + best_sharpe**2 * (kurtosis - 1) / 4.0
    )
    if psr_denom > 0:
        psr_stat = ((best_sharpe - target_sharpe) * np.sqrt(T - 1)) / psr_denom
        dsr_value = float(stats.norm.cdf(psr_stat))
    else:
        dsr_value = 0.5

    return {
        'best_sharpe': float(best_sharpe),
        'target_sharpe': float(target_sharpe),
        'expected_max_sharpe': float(e_max),
        'dsr': dsr_value,
        'psr_stat': float(psr_stat) if psr_denom > 0 else None,
        'n_trials': N,
        'T': T,
        'skewness': skew,
        'kurtosis': kurtosis,
        'sharpe_std': float(sharpe_std),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main PBO analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_pbo_analysis(
    M: np.ndarray,
    S: int = 14,
    threshold: float = 0.0,
    name: str = "CPCV",
    plot: bool = False,
) -> dict:
    """
    Run full PBO analysis on matrix M.

    Parameters
    ----------
    M : np.ndarray, shape (T, N)
        Returns matrix. T = time steps, N = strategies/trials.
    S : int
        Number of CSCV chunks (must be even). Paper suggests S=16.
    threshold : float
        OOS loss threshold (0 for Sharpe-based analysis).
    name : str
        Experiment name for plots.
    plot : bool
        Whether to generate plots.

    Returns
    -------
    results : dict
        PBO value, prob_oos_loss, logits stats, DSR results.
    """
    T, N = M.shape
    print(f"\n{'='*60}")
    print(f"PBO Analysis: {name}")
    print(f"{'='*60}")
    print(f"  Matrix M: {T} time steps x {N} trials/paths")
    print(f"  CSCV chunks: S={S}")

    if N < 3:
        print(f"  WARNING: Only {N} columns in M. PBO needs N >= 3 for meaningful results.")
        print(f"  Consider running HPO with more trials or increasing N_GROUPS.")

    # Adjust S if T is too small
    while T // S < 2 and S > 2:
        S -= 2
    if S < 2:
        print(f"  ERROR: Not enough data points (T={T}) for PBO analysis.")
        return {'error': 'insufficient_data'}

    print(f"  Adjusted S={S} (chunk size={T//S})")

    try:
        from pypbo.pbo import pbo as pbo_func
        pbo_source = "pypbo"
    except ImportError:
        try:
            sys.path.insert(0, os.path.join(
                PROJECT_ROOT, "lopez_de_prado_analysis", "pypbo"
            ))
            from pypbo.pbo import pbo as pbo_func
            pbo_source = "pypbo_local"
        except ImportError:
            print("  ERROR: pypbo not found. Install or check path.")
            return {'error': 'pypbo_not_found'}

    print(f"  Using {pbo_source}")

    # Run PBO
    pbo_result = pbo_func(
        M=M,
        S=S,
        metric_func=sharpe_metric,
        threshold=threshold,
        n_jobs=1,
        verbose=False,
        plot=False,
        hist=False,
    )

    pbo_value = pbo_result.pbo
    prob_oos_loss = pbo_result.prob_oos_loss
    logits = np.array(pbo_result.logits)

    print(f"\n  Results:")
    print(f"    PBO = {pbo_value*100:.1f}%")
    print(f"    Prob(OOS loss) = {prob_oos_loss*100:.1f}%")
    print(f"    Logits: min={min(logits):.3f}, max={max(logits):.3f}, "
          f"mean={np.mean(logits):.3f}")

    # Linear model (IS vs OOS performance degradation)
    lm = pbo_result.linear_model
    print(f"    Perf. degradation: slope={lm.slope:.4f}, "
          f"R²={lm.rvalue**2:.4f}, p={lm.pvalue:.4e}")

    # Per-column Sharpe ratios
    sharpe_cols = sharpe_metric(M)
    if isinstance(sharpe_cols, (int, float)):
        sharpe_cols = np.array([sharpe_cols])

    # DSR
    best_col = np.argmax(sharpe_cols)
    dsr_results = compute_dsr_analysis(
        sharpe_ratios=sharpe_cols,
        daily_returns=M[:, best_col],
        n_trials=N,
    )

    print(f"\n  DSR Analysis:")
    print(f"    Best Sharpe = {dsr_results['best_sharpe']:.4f}")
    print(f"    Target Sharpe (E[max]) = {dsr_results['target_sharpe']:.4f}")
    print(f"    DSR = {dsr_results['dsr']:.4f}")

    results = {
        'pbo': float(pbo_value),
        'pbo_pct': float(pbo_value * 100),
        'prob_oos_loss': float(prob_oos_loss),
        'logits_mean': float(np.mean(logits)),
        'logits_std': float(np.std(logits)),
        'logits_min': float(min(logits)),
        'logits_max': float(max(logits)),
        'perf_degradation_slope': float(lm.slope),
        'perf_degradation_r2': float(lm.rvalue**2),
        'perf_degradation_p': float(lm.pvalue),
        'sharpe_ratios': sharpe_cols.tolist(),
        'dsr': dsr_results,
        'n_trials': N,
        'T': T,
        'S': S,
    }

    if plot:
        try:
            from pypbo.pbo import plot_pbo
            plot_pbo(pbo_result, hist=False)
        except Exception as e:
            print(f"  Plot failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="PBO Analysis for CPCV Pipeline"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to CPCV results directory"
    )
    parser.add_argument(
        "--pbo-returns-dir", type=str, default=None,
        help="Path to pbo_returns/ directory from HPO sweep"
    )
    parser.add_argument(
        "--S", type=int, default=14,
        help="Number of CSCV chunks (must be even, default 14)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="OOS loss threshold (default 0 for Sharpe)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate PBO plots"
    )
    parser.add_argument(
        "--dsr-only", action="store_true",
        help="Only compute DSR (skip full PBO)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save results to JSON file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    M = None
    metadata = None

    # Mode 1: HPO trial returns
    if args.pbo_returns_dir:
        M, metadata = build_matrix_M_from_hpo_trials(args.pbo_returns_dir)

    # Mode 2: Results directory
    elif args.results_dir:
        # Try JSON trial results first
        M, metadata = build_matrix_M_from_json_results(args.results_dir)

        if M is None:
            # Try single-run path mode
            # Need total_days from config or data
            config_path = os.path.join(args.results_dir, "run_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                total_days = config.get('total_days', 753)
            else:
                total_days = 753
            M, metadata = build_matrix_M_from_paths(
                args.results_dir, total_days
            )
    else:
        print("ERROR: Provide --results-dir or --pbo-returns-dir")
        return 1

    if M is None:
        print("ERROR: Could not build matrix M from provided data.")
        return 1

    print(f"\nMatrix M shape: {M.shape}")
    print(f"Source: {metadata.get('source', 'unknown')}")

    if args.dsr_only:
        sharpes = sharpe_metric(M)
        if isinstance(sharpes, (int, float)):
            sharpes = np.array([sharpes])
        best_col = np.argmax(sharpes)
        dsr_results = compute_dsr_analysis(sharpes, M[:, best_col], M.shape[1])
        print(f"\nDSR Analysis:")
        for k, v in dsr_results.items():
            print(f"  {k}: {v}")
        results = {'dsr': dsr_results, 'metadata': metadata}
    else:
        results = run_pbo_analysis(
            M=M,
            S=args.S,
            threshold=args.threshold,
            name=metadata.get('source', 'CPCV'),
            plot=args.plot,
        )
        results['metadata'] = metadata

    if args.save:
        with open(args.save, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
