"""
Adaptive CPCV — External Feature Comparison for Wyckoff NQ Range Bars.

Compares multiple candidate features for boundary-shifting in
AdaptiveCombPurgedKFoldCV on the Wyckoff NQ dataset.

Candidates:
  1. Rolling Volatility (200-bar)  — std of log returns
  2. Rolling Drawdown   (200-bar)  — peak-to-trough drawdown
  3. WaveER             (from tech) — Wyckoff effort-result ratio
  4. ER_Ratio           (from tech) — effort ratio
  5. VolumeFilter       (from tech) — volume regime
  6. cvd_slope          (from tech) — CVD momentum
  7. Rolling Sharpe     (200-bar)  — risk-adjusted return

Metrics:
  - AvgGap%:  mean |train_return − test_return| across splits (lower = better)
  - MaxGap%:  worst-case regime mismatch
  - MM:       strict regime mismatches (train<−X% & test>+Y%, or vice versa)
  - Group size range (how unequal the groups become)

Usage:
    python -m wyckoff_rl.acpcv_feature_comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cpcv_pipeline.function_CPCV import (
    CombPurgedKFoldCV,
    AdaptiveCombPurgedKFoldCV,
)
from wyckoff_rl.config import (
    WYCKOFF_NPZ_PATH, N_GROUPS, K_TEST_GROUPS, EMBARGO_BARS,
)


# ═════════════════════════════════════════════════════════════════════════════
# Feature computation functions (single-instrument NQ)
# ═════════════════════════════════════════════════════════════════════════════

def _rolling_volatility(close: np.ndarray, window: int = 200) -> np.ndarray:
    """Rolling std of log returns."""
    prices = close.ravel()
    log_ret = np.diff(np.log(prices + 1e-12))
    log_ret = np.concatenate([[0.0], log_ret])
    vol = np.full_like(log_ret, np.nan)
    for i in range(window, len(log_ret)):
        vol[i] = log_ret[i - window:i].std()
    vol[:window] = vol[window] if window < len(vol) else 0.0
    return vol


def _rolling_drawdown(close: np.ndarray, window: int = 200) -> np.ndarray:
    """Rolling max drawdown (always <= 0)."""
    prices = close.ravel()
    dd = np.zeros(len(prices))
    for i in range(len(prices)):
        start = max(0, i - window + 1)
        peak = np.max(prices[start:i + 1])
        dd[i] = (prices[i] - peak) / (peak + 1e-9)
    return dd


def _rolling_sharpe(close: np.ndarray, window: int = 200) -> np.ndarray:
    """Rolling Sharpe ratio (mean/std of log returns)."""
    prices = close.ravel()
    log_ret = np.diff(np.log(prices + 1e-12))
    log_ret = np.concatenate([[0.0], log_ret])
    sharpe = np.full_like(log_ret, np.nan)
    for i in range(window, len(log_ret)):
        w = log_ret[i - window:i]
        s = w.std()
        sharpe[i] = w.mean() / s if s > 1e-12 else 0.0
    sharpe[:window] = sharpe[window] if window < len(sharpe) else 0.0
    return sharpe


def _extract_tech_feature(tech_ary: np.ndarray, feature_names: np.ndarray,
                          name: str, smooth_window: int = 50) -> np.ndarray:
    """Extract a named feature from tech_ary and smooth it for ACPCV use.

    Many Wyckoff features are sparse/binary (e.g. Spring, Upthrust).
    Smoothing with a rolling mean makes them usable as boundary-shift signals.
    """
    idx = np.where(feature_names == name)[0]
    if len(idx) == 0:
        raise ValueError(f"Feature '{name}' not found in: {list(feature_names)}")
    raw = tech_ary[:, idx[0]].astype(np.float64)

    # For continuous features, light smoothing; for sparse, heavier
    if smooth_window > 1 and raw.std() > 0:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(raw, kernel, mode='same')
        return smoothed
    return raw


# ═════════════════════════════════════════════════════════════════════════════
# Regime analysis helpers
# ═════════════════════════════════════════════════════════════════════════════

def _contiguous_segments(idx: np.ndarray):
    """Split sorted index array into contiguous segments."""
    if len(idx) == 0:
        return []
    segments, start = [], 0
    for j in range(1, len(idx)):
        if idx[j] != idx[j - 1] + 1:
            segments.append(idx[start:j])
            start = j
    segments.append(idx[start:])
    return segments


def _bh_return_segments(close: np.ndarray, idx: np.ndarray) -> float:
    """Buy-hold return over (possibly non-contiguous) index range."""
    prices = close.ravel()
    segs = _contiguous_segments(idx)
    if not segs:
        return 0.0
    total_len = sum(len(s) for s in segs)
    weighted = sum(
        (prices[s[-1]] / prices[s[0]] - 1) * 100 * len(s)
        for s in segs
    ) / total_len
    return weighted


def _count_mismatches(close: np.ndarray, splits: list,
                      low_thresh: float = -2.0, high_thresh: float = 2.0) -> int:
    """Count regime mismatches: train in opposite regime from test.

    Uses ±2% thresholds (tighter than the Alpaca ±15% since NQ is one instrument
    and range bars span a shorter price range per fold).
    """
    n_mm = 0
    for tr, tt in splits:
        tr_ret = _bh_return_segments(close, tr)
        tt_ret = _bh_return_segments(close, tt)
        if (tr_ret < low_thresh and tt_ret > high_thresh) or \
           (tr_ret > high_thresh and tt_ret < low_thresh):
            n_mm += 1
    return n_mm


def _regime_gaps(close: np.ndarray, splits: list) -> list:
    """Compute per-split |train_return − test_return| gaps."""
    gaps = []
    for tr, tt in splits:
        tr_ret = _bh_return_segments(close, tr)
        tt_ret = _bh_return_segments(close, tt)
        gaps.append(abs(tr_ret - tt_ret))
    return gaps


def _split_size_stats(cv_obj, total_samples: int) -> dict:
    """Get fold group size statistics."""
    bounds = cv_obj.get_fold_bounds(total_samples)
    sizes = [e - s for s, e in bounds]
    return {
        'sizes': sizes,
        'range': max(sizes) - min(sizes),
        'min': min(sizes),
        'max': max(sizes),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main comparison
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    data = np.load(WYCKOFF_NPZ_PATH, allow_pickle=True)
    close_ary = data['close_ary']
    tech_ary = data['tech_ary']
    feature_names = data['feature_names']

    T = close_ary.shape[0]
    print(f"Wyckoff NQ data: {T:,} bars, {tech_ary.shape[1]} features")
    print(f"Close range: {close_ary.min():.1f} – {close_ary.max():.1f}")
    print(f"CPCV: N={N_GROUPS}, K={K_TEST_GROUPS}, embargo={EMBARGO_BARS} bars")
    print()

    # ── Build candidate features ──────────────────────────────────────────
    WINDOW = 200  # range bars (≈ a few hours of NQ trading)

    candidates = {}
    candidates['Rolling Vol (200)'] = _rolling_volatility(close_ary, WINDOW)
    candidates['Rolling DD (200)'] = _rolling_drawdown(close_ary, WINDOW)
    candidates['Rolling Sharpe (200)'] = _rolling_sharpe(close_ary, WINDOW)
    candidates['WaveER (smooth 50)'] = _extract_tech_feature(
        tech_ary, feature_names, 'WaveER', smooth_window=50)
    candidates['ER_Ratio (smooth 50)'] = _extract_tech_feature(
        tech_ary, feature_names, 'ER_Ratio', smooth_window=50)
    candidates['VolumeFilter (smooth 100)'] = _extract_tech_feature(
        tech_ary, feature_names, 'VolumeFilter', smooth_window=100)
    candidates['cvd_slope (smooth 100)'] = _extract_tech_feature(
        tech_ary, feature_names, 'cvd_slope', smooth_window=100)
    candidates['delta_norm (smooth 100)'] = _extract_tech_feature(
        tech_ary, feature_names, 'delta_norm', smooth_window=100)
    candidates['Rolling DD (500)'] = _rolling_drawdown(close_ary, 500)
    candidates['Rolling Vol (500)'] = _rolling_volatility(close_ary, 500)

    # ── Feature stats ─────────────────────────────────────────────────────
    print(f"{'Feature':<28} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("─" * 72)
    for name, feat in candidates.items():
        print(f"{name:<28} {feat.min():>10.6f} {feat.max():>10.6f} "
              f"{feat.mean():>10.6f} {feat.std():>10.6f}")
    print()

    # ── Standard CPCV baseline ────────────────────────────────────────────
    std_cv = CombPurgedKFoldCV(
        n_splits=N_GROUPS, n_test_splits=K_TEST_GROUPS,
        embargo_days=EMBARGO_BARS,
    )
    std_splits = list(std_cv.split(T))
    std_gaps = _regime_gaps(close_ary, std_splits)
    std_mm = _count_mismatches(close_ary, std_splits)
    std_sizes = _split_size_stats(std_cv, T)

    # ── Evaluate each candidate ───────────────────────────────────────────
    results = []

    # Add standard baseline
    results.append({
        'Feature': 'Standard CPCV (no adapt.)',
        'AvgGap%': np.mean(std_gaps),
        'MaxGap%': max(std_gaps),
        'MM': std_mm,
        'Grp Range': std_sizes['range'],
        'Grp Min': std_sizes['min'],
        'Grp Max': std_sizes['max'],
        'Sizes': std_sizes['sizes'],
    })

    for name, feat in candidates.items():
        acv = AdaptiveCombPurgedKFoldCV(
            n_splits=N_GROUPS,
            n_test_splits=K_TEST_GROUPS,
            embargo_days=EMBARGO_BARS,
            external_feature=feat,
            n_subsplits=3,
            lower_quantile=0.25,
            upper_quantile=0.75,
        )
        splits = list(acv.split(T))
        gaps = _regime_gaps(close_ary, splits)
        mm = _count_mismatches(close_ary, splits)
        sizes = _split_size_stats(acv, T)

        results.append({
            'Feature': name,
            'AvgGap%': np.mean(gaps),
            'MaxGap%': max(gaps),
            'MM': mm,
            'Grp Range': sizes['range'],
            'Grp Min': sizes['min'],
            'Grp Max': sizes['max'],
            'Sizes': sizes['sizes'],
        })

    # ── Results table ─────────────────────────────────────────────────────
    # Sort by AvgGap (lower is better)
    results.sort(key=lambda r: r['AvgGap%'])

    best_avg = results[0]['AvgGap%']
    std_avg = next(r['AvgGap%'] for r in results
                   if r['Feature'].startswith('Standard'))

    print(f"{'Feature':<30} {'AvgGap%':>8} {'MaxGap%':>8} {'MM':>4} "
          f"{'GrpRange':>8} {'GrpSizes':>30}")
    print("─" * 95)
    for r in results:
        better = "  ★" if r['AvgGap%'] < std_avg and \
                 r['Feature'] != 'Standard CPCV (no adapt.)' else ""
        best_mark = "  ← BEST" if r['AvgGap%'] == best_avg else ""
        sizes_str = ", ".join(f"{s:,}" for s in r['Sizes'])
        print(f"{r['Feature']:<30} {r['AvgGap%']:>8.3f} {r['MaxGap%']:>8.3f} "
              f"{r['MM']:>4} {r['Grp Range']:>8,} "
              f"[{sizes_str}]{better}{best_mark}")

    print()
    print(f"MM = regime mismatches (train & test in opposite regimes)")
    print(f"GrpRange = max group size − min group size (0 = equal groups)")
    print(f"★ = better than standard CPCV")

    # ── Per-split details for top 3 ──────────────────────────────────────
    print(f"\n{'='*95}")
    print("Per-split regime details (top 3 + standard)")
    print(f"{'='*95}")

    show_features = ['Standard CPCV (no adapt.)'] + \
                    [r['Feature'] for r in results[:3]
                     if r['Feature'] != 'Standard CPCV (no adapt.)']
    # Deduplicate while preserving order
    seen = set()
    show_features = [f for f in show_features if not (f in seen or seen.add(f))]

    for feat_name in show_features:
        if feat_name == 'Standard CPCV (no adapt.)':
            cv_obj = std_cv
        else:
            feat = candidates[feat_name]
            cv_obj = AdaptiveCombPurgedKFoldCV(
                n_splits=N_GROUPS, n_test_splits=K_TEST_GROUPS,
                embargo_days=EMBARGO_BARS,
                external_feature=feat, n_subsplits=3,
                lower_quantile=0.25, upper_quantile=0.75,
            )
        splits = list(cv_obj.split(T))

        print(f"\n── {feat_name} ──")
        print(f"  {'Split':>5}  {'Train':>8}  {'Test':>8}  "
              f"{'TrainRet%':>10}  {'TestRet%':>10}  {'Gap%':>8}  {'Flag':>5}")
        for i, (tr, tt) in enumerate(splits):
            tr_ret = _bh_return_segments(close_ary, tr)
            tt_ret = _bh_return_segments(close_ary, tt)
            gap = abs(tr_ret - tt_ret)
            flag = "⚠" if (tr_ret < -2 and tt_ret > 2) or \
                          (tr_ret > 2 and tt_ret < -2) else ""
            print(f"  {i:>5}  {len(tr):>8,}  {len(tt):>8,}  "
                  f"{tr_ret:>+10.3f}  {tt_ret:>+10.3f}  {gap:>8.3f}  {flag:>5}")

    # ── Recommendation ────────────────────────────────────────────────────
    best = results[0]
    print(f"\n{'='*95}")
    print(f"RECOMMENDATION: {best['Feature']}")
    print(f"  AvgGap={best['AvgGap%']:.3f}% (std={std_avg:.3f}%), "
          f"MaxGap={best['MaxGap%']:.3f}%, MM={best['MM']}")
    if best['Feature'] != 'Standard CPCV (no adapt.)':
        improvement = (std_avg - best['AvgGap%']) / std_avg * 100
        print(f"  {improvement:+.1f}% improvement over standard CPCV")
    print(f"{'='*95}")

    return results


if __name__ == '__main__':
    main()
