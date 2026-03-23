#!/usr/bin/env python3
"""
Pre-training data validation for Wyckoff NPZ files.

Checks for issues that would cause the RL network to choke or train poorly:
  1. NaN / Inf values
  2. Constant or near-constant columns (zero variance)
  3. Extreme outliers (values > 10 std from mean)
  4. Feature scale mismatches (some features 1e6, others 1e-6)
  5. High correlation / redundancy between selected features
  6. Shape consistency with training config
  7. Date continuity (no huge gaps)
  8. Close price sanity (monotonic-ish, no zeros)

Usage:
    python -m wyckoff_rl.tools.validate_npz path/to/file.npz
    python -m wyckoff_rl.tools.validate_npz  # uses default WYCKOFF_NPZ_PATH
"""

import sys
import os
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def validate_npz(npz_path: str, feature_indices: list[int] = None) -> dict:
    """Run all validation checks on a Wyckoff NPZ file.

    Returns dict with 'passed': bool and 'issues': list[str].
    """
    issues = []
    warnings = []

    # ── Load ─────────────────────────────────────────────────────────────
    if not os.path.exists(npz_path):
        return {"passed": False, "issues": [f"File not found: {npz_path}"], "warnings": []}

    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"NPZ: {npz_path}")
    print(f"Keys: {keys}")

    required = ["close_ary", "tech_ary"]
    for k in required:
        if k not in keys:
            issues.append(f"MISSING required key: {k}")
    if issues:
        return {"passed": False, "issues": issues, "warnings": []}

    close_ary = data["close_ary"]
    tech_ary = data["tech_ary"]
    feature_names = list(data["feature_names"]) if "feature_names" in data else None
    dates_ary = data["dates_ary"] if "dates_ary" in data else None

    n_bars, n_feat = tech_ary.shape
    print(f"Bars: {n_bars:,}, Full features: {n_feat}, Close: {close_ary.shape}")

    # Determine which features training actually uses
    if feature_indices is None:
        from wyckoff_rl.feature_config import SELECTED_INDICES
        feature_indices = SELECTED_INDICES

    selected = tech_ary[:, feature_indices]
    n_sel = len(feature_indices)
    sel_names = [feature_names[i] for i in feature_indices] if feature_names else [f"feat_{i}" for i in feature_indices]
    print(f"Selected features: {n_sel} (training subset)\n")

    # ── 1. NaN / Inf ────────────────────────────────────────────────────
    print("=" * 60)
    print("1. NaN / Inf Check")
    nan_count = np.isnan(selected).sum()
    inf_count = np.isinf(selected).sum()
    close_nan = np.isnan(close_ary).sum()
    if nan_count > 0:
        issues.append(f"NaN values in tech_ary: {nan_count}")
        # Which columns?
        for j, name in enumerate(sel_names):
            col_nan = np.isnan(selected[:, j]).sum()
            if col_nan > 0:
                issues.append(f"  {name}: {col_nan} NaN ({col_nan/n_bars*100:.1f}%)")
    if inf_count > 0:
        issues.append(f"Inf values in tech_ary: {inf_count}")
    if close_nan > 0:
        issues.append(f"NaN in close_ary: {close_nan}")
    if nan_count == 0 and inf_count == 0 and close_nan == 0:
        print("  PASS: No NaN or Inf values")
    else:
        for i in issues:
            print(f"  FAIL: {i}")

    # ── 2. Constant / Near-constant columns ─────────────────────────────
    print("\n" + "=" * 60)
    print("2. Constant / Near-Constant Columns")
    for j, name in enumerate(sel_names):
        col = selected[:, j]
        std = np.nanstd(col)
        n_unique = len(np.unique(col[~np.isnan(col)]))
        pct_zero = (col == 0).sum() / n_bars * 100
        if std == 0:
            issues.append(f"CONSTANT column: {name} (all same value: {col[0]:.4f})")
        elif n_unique <= 2:
            warnings.append(f"Binary column: {name} ({n_unique} unique values, {pct_zero:.0f}% zero)")
        elif std < 1e-7:
            issues.append(f"Near-constant: {name} (std={std:.2e})")
        elif pct_zero > 99:
            warnings.append(f"Sparse column: {name} ({pct_zero:.1f}% zero)")

    const_issues = [i for i in issues if "CONSTANT" in i or "Near-constant" in i]
    if not const_issues:
        print("  PASS: No constant columns")
    else:
        for i in const_issues:
            print(f"  FAIL: {i}")
    sparse_warns = [w for w in warnings if "Binary" in w or "Sparse" in w]
    for w in sparse_warns:
        print(f"  WARN: {w}")

    # ── 3. Scale / Range Analysis ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Feature Scale Analysis")
    print(f"  {'Feature':<28s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s} {'|Max|':>10s}")
    print("  " + "-" * 78)

    scales = []
    for j, name in enumerate(sel_names):
        col = selected[:, j]
        mean, std, cmin, cmax = np.nanmean(col), np.nanstd(col), np.nanmin(col), np.nanmax(col)
        absmax = max(abs(cmin), abs(cmax))
        scales.append(absmax)
        print(f"  {name:<28s} {mean:10.4f} {std:10.4f} {cmin:10.4f} {cmax:10.4f} {absmax:10.4f}")

    scales = np.array(scales)
    nonzero_scales = scales[scales > 0]
    if len(nonzero_scales) > 1:
        scale_ratio = nonzero_scales.max() / nonzero_scales.min()
        if scale_ratio > 1000:
            issues.append(f"Extreme scale mismatch: {scale_ratio:.0f}x between features")
            print(f"\n  FAIL: Scale ratio = {scale_ratio:.0f}x (max/min absolute range)")
        elif scale_ratio > 100:
            warnings.append(f"Large scale mismatch: {scale_ratio:.0f}x — normalization recommended")
            print(f"\n  WARN: Scale ratio = {scale_ratio:.0f}x")
        else:
            print(f"\n  PASS: Scale ratio = {scale_ratio:.1f}x (reasonable)")

    # ── 4. Outlier Check (> 10 std) ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. Extreme Outlier Check (|z| > 10)")
    outlier_cols = []
    for j, name in enumerate(sel_names):
        col = selected[:, j]
        std = np.nanstd(col)
        if std > 0:
            z = np.abs((col - np.nanmean(col)) / std)
            n_extreme = (z > 10).sum()
            if n_extreme > 0:
                outlier_cols.append((name, n_extreme, z.max()))
                warnings.append(f"Outliers in {name}: {n_extreme} bars with |z|>10 (max z={z.max():.1f})")

    if not outlier_cols:
        print("  PASS: No extreme outliers")
    else:
        for name, n, maxz in outlier_cols:
            print(f"  WARN: {name}: {n} bars with |z|>10 (max z={maxz:.1f})")

    # ── 5. Correlation Check (selected features) ───────────────────────
    print("\n" + "=" * 60)
    print("5. High Correlation Check (|r| > 0.95)")
    valid = selected[~np.isnan(selected).any(axis=1)]
    if len(valid) > 100:
        corr = np.corrcoef(valid.T)
        high_corr = []
        for i in range(n_sel):
            for j2 in range(i + 1, n_sel):
                r = abs(corr[i, j2])
                if r > 0.95:
                    high_corr.append((sel_names[i], sel_names[j2], r))
        if not high_corr:
            print("  PASS: No highly correlated feature pairs")
        else:
            for n1, n2, r in sorted(high_corr, key=lambda x: -x[2]):
                warnings.append(f"High correlation: {n1} ↔ {n2} (r={r:.3f})")
                print(f"  WARN: {n1} ↔ {n2}: r={r:.3f}")

    # ── 6. Shape Consistency ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. Shape Consistency")
    if close_ary.shape[0] != n_bars:
        issues.append(f"close_ary rows ({close_ary.shape[0]}) != tech_ary rows ({n_bars})")
    if close_ary.shape[1] != 1:
        issues.append(f"close_ary should be (n,1), got {close_ary.shape}")
    if dates_ary is not None and len(dates_ary) != n_bars:
        issues.append(f"dates_ary length ({len(dates_ary)}) != tech_ary rows ({n_bars})")

    from wyckoff_rl.feature_config import WINDOW_SIZE
    if n_bars < WINDOW_SIZE + 50:
        issues.append(f"Too few bars ({n_bars}) for window_size={WINDOW_SIZE} + training")
    else:
        n_windows = n_bars - WINDOW_SIZE + 1
        print(f"  PASS: {n_bars:,} bars → {n_windows:,} sliding windows (window={WINDOW_SIZE})")

    shape_issues = [i for i in issues if "rows" in i or "shape" in i or "Too few" in i]
    if not shape_issues:
        print(f"  PASS: Shapes consistent")
    for i in shape_issues:
        print(f"  FAIL: {i}")

    # ── 7. Close Price Sanity ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. Close Price Sanity")
    closes = close_ary.flatten()
    if (closes <= 0).any():
        issues.append(f"Zero/negative close prices: {(closes <= 0).sum()}")
    close_range = closes.max() - closes.min()
    close_pct = close_range / closes.mean() * 100
    print(f"  Range: {closes.min():.2f} - {closes.max():.2f} ({close_range:.0f} pts, {close_pct:.1f}%)")
    if close_pct > 50:
        warnings.append(f"Large price range: {close_pct:.1f}% — consider if this causes reward scaling issues")
        print(f"  WARN: Large price range ({close_pct:.1f}%)")
    else:
        print(f"  PASS: Price range reasonable")

    # ── 8. Date Continuity ──────────────────────────────────────────────
    if dates_ary is not None:
        print("\n" + "=" * 60)
        print("8. Date Continuity")
        try:
            import pandas as pd
            dates = pd.to_datetime(dates_ary)
            gaps = dates.diff()
            max_gap = gaps.max()
            median_gap = gaps.median()
            print(f"  Date range: {dates[0]} → {dates[-1]}")
            print(f"  Median gap: {median_gap}, Max gap: {max_gap}")
            if max_gap > pd.Timedelta(days=5):
                warnings.append(f"Large date gap: {max_gap}")
                print(f"  WARN: Max gap > 5 days ({max_gap})")
            else:
                print(f"  PASS: No unusual gaps")
        except Exception as e:
            warnings.append(f"Could not parse dates: {e}")
            print(f"  SKIP: Could not parse dates")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = len(issues) == 0
    status = "PASS" if passed else "FAIL"
    print(f"\nRESULT: {status}")
    print(f"  Issues:   {len(issues)}")
    print(f"  Warnings: {len(warnings)}")
    if issues:
        print("\n  ISSUES (must fix):")
        for i in issues:
            print(f"    ✗ {i}")
    if warnings:
        print("\n  WARNINGS (review):")
        for w in warnings:
            print(f"    ⚠ {w}")

    return {"passed": passed, "issues": issues, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description="Validate Wyckoff NPZ for RL training")
    parser.add_argument("npz", nargs="?", default=None, help="Path to NPZ file")
    args = parser.parse_args()

    if args.npz is None:
        from wyckoff_rl.config import WYCKOFF_NPZ_PATH
        args.npz = WYCKOFF_NPZ_PATH

    result = validate_npz(args.npz)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
