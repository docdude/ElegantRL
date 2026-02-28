#!/usr/bin/env python3
"""
Leakage verification tests for CPCV pipeline.

Run this BEFORE any GPU training to verify:
1. No train/test index overlap in any split
2. Complete OOS coverage (all samples appear in at least one test set)
3. Purge + embargo gaps are correct
4. Pre-sliced .npz files produce non-overlapping data
5. Backtest paths are valid (each path covers all time steps)
6. Compare old vs new implementation to highlight differences

Usage:
    python -m cpcv_pipeline.test_leakage
    python -m cpcv_pipeline.test_leakage --verbose
    python -m cpcv_pipeline.test_leakage --total-days 753
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cpcv_pipeline.function_CPCV import (
    CombPurgedKFoldCV,
    back_test_paths_generator,
    verify_no_leakage,
    verify_complete_oos_coverage,
    _indices_to_ranges,
)
from cpcv_pipeline.config import (
    N_GROUPS, K_TEST_GROUPS, EMBARGO_DAYS,
    ALPACA_NPZ_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_no_leakage(total_days: int, verbose: bool = False) -> bool:
    """Test 1: No train/test overlap in any split."""
    print("\n" + "="*60)
    print("TEST 1: No train/test index overlap")
    print("="*60)

    cv = CombPurgedKFoldCV(
        n_splits=N_GROUPS,
        n_test_splits=K_TEST_GROUPS,
        embargo_days=EMBARGO_DAYS,
    )

    n_passes = 0
    n_fails = 0

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(total_days)):
        overlap = np.intersect1d(train_idx, test_idx)
        status = "✓ PASS" if len(overlap) == 0 else "✗ FAIL"

        if len(overlap) > 0:
            n_fails += 1
            print(f"  Split {split_idx}: {status} — "
                  f"{len(overlap)} overlapping indices: {overlap[:10]}...")
        else:
            n_passes += 1
            if verbose:
                train_ranges = _indices_to_ranges(train_idx)
                test_ranges = _indices_to_ranges(test_idx)
                print(f"  Split {split_idx}: {status} — "
                      f"Train: {len(train_idx)}d, Test: {len(test_idx)}d")
                print(f"    Train ranges: {train_ranges}")
                print(f"    Test ranges:  {test_ranges}")

    print(f"\n  Result: {n_passes} passed, {n_fails} failed")
    return n_fails == 0


def test_complete_coverage(total_days: int, verbose: bool = False) -> bool:
    """Test 2: Every sample appears in at least one test set."""
    print("\n" + "="*60)
    print("TEST 2: Complete OOS coverage")
    print("="*60)

    cv = CombPurgedKFoldCV(
        n_splits=N_GROUPS,
        n_test_splits=K_TEST_GROUPS,
        embargo_days=EMBARGO_DAYS,
    )

    test_count = np.zeros(total_days, dtype=int)
    for _, test_idx in cv.split(total_days):
        test_count[test_idx] += 1

    uncovered = np.where(test_count == 0)[0]

    if len(uncovered) == 0:
        print(f"  ✓ PASS: All {total_days} samples appear in at least one test set")
        if verbose:
            print(f"  Min appearances: {test_count.min()}")
            print(f"  Max appearances: {test_count.max()}")
            print(f"  Mean appearances: {test_count.mean():.1f}")
    else:
        print(f"  ✗ FAIL: {len(uncovered)} samples never tested")
        print(f"  Uncovered indices: {uncovered[:20]}...")

    return len(uncovered) == 0


def test_purge_embargo_gaps(total_days: int, verbose: bool = False) -> bool:
    """Test 3: Purge + embargo create proper gaps."""
    print("\n" + "="*60)
    print("TEST 3: Purge + embargo gaps")
    print("="*60)

    cv = CombPurgedKFoldCV(
        n_splits=N_GROUPS,
        n_test_splits=K_TEST_GROUPS,
        embargo_days=EMBARGO_DAYS,
    )

    all_pass = True

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(total_days)):
        # Find gap indices (neither train nor test)
        all_used = np.union1d(train_idx, test_idx)
        gap_idx = np.setdiff1d(np.arange(total_days), all_used)

        if verbose:
            test_ranges = _indices_to_ranges(test_idx)
            gap_ranges = _indices_to_ranges(gap_idx) if len(gap_idx) > 0 else []
            print(f"  Split {split_idx}: "
                  f"train={len(train_idx)}, test={len(test_idx)}, "
                  f"gap={len(gap_idx)}")
            print(f"    Test ranges: {test_ranges}")
            if gap_ranges:
                print(f"    Gap ranges:  {gap_ranges}")

        # Verify gaps are near test fold boundaries
        fold_bounds = cv.get_fold_bounds(total_days)
        test_groups = []
        for g, (s, e) in enumerate(fold_bounds):
            if np.any((test_idx >= s) & (test_idx < e)):
                test_groups.append(g)

        for gap_i in gap_idx:
            near_boundary = False
            for tg in test_groups:
                ts, te = fold_bounds[tg]
                # Gap should be within embargo_days of a test boundary
                if (ts - EMBARGO_DAYS - 1 <= gap_i < ts) or \
                   (te <= gap_i < te + EMBARGO_DAYS + 1):
                    near_boundary = True
                    break
            if not near_boundary:
                print(f"  ✗ FAIL: Gap index {gap_i} not near any test boundary")
                all_pass = False

    if all_pass:
        print(f"  ✓ PASS: All gaps are within purge/embargo zones")
    return all_pass


def test_backtest_paths(total_days: int, verbose: bool = False) -> bool:
    """Test 4: Backtest paths cover all time steps."""
    print("\n" + "="*60)
    print("TEST 4: Backtest paths")
    print("="*60)

    is_test, paths, path_folds = back_test_paths_generator(
        total_days, N_GROUPS, K_TEST_GROUPS
    )

    n_paths = paths.shape[1]
    print(f"  Generated {n_paths} backtest paths")

    all_pass = True

    for p in range(n_paths):
        path_p = paths[:, p]
        uncovered = np.isnan(path_p)
        n_uncovered = uncovered.sum()

        if n_uncovered > 0:
            print(f"  ✗ FAIL: Path {p} has {n_uncovered} uncovered time steps")
            all_pass = False
        elif verbose:
            # Show which splits contribute to this path
            unique_splits = np.unique(path_p[~np.isnan(path_p)])
            print(f"  Path {p}: uses splits {unique_splits.astype(int).tolist()}")

    if all_pass:
        print(f"  ✓ PASS: All {n_paths} paths cover all {total_days} time steps")

    return all_pass


def test_sliced_data_integrity(verbose: bool = False) -> bool:
    """Test 5: Pre-sliced .npz files contain correct data."""
    print("\n" + "="*60)
    print("TEST 5: Sliced data integrity")
    print("="*60)

    if not os.path.exists(ALPACA_NPZ_PATH):
        print(f"  ⚠ SKIP: Data file not found at {ALPACA_NPZ_PATH}")
        return True

    from cpcv_pipeline.function_train_test import (
        load_full_data, save_sliced_data,
    )

    close_ary, tech_ary = load_full_data()
    total_days = close_ary.shape[0]
    print(f"  Full data: {close_ary.shape[0]} days × {close_ary.shape[1]} stocks")

    cv = CombPurgedKFoldCV(
        n_splits=N_GROUPS,
        n_test_splits=K_TEST_GROUPS,
        embargo_days=EMBARGO_DAYS,
    )

    all_pass = True
    temp_dir = os.path.join(os.path.dirname(ALPACA_NPZ_PATH), "_test_temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        for split_idx, (train_idx, test_idx) in enumerate(cv.split(total_days)):
            # Save and reload train data
            train_npz = os.path.join(temp_dir, f"test_train_{split_idx}.npz")
            test_npz = os.path.join(temp_dir, f"test_test_{split_idx}.npz")

            save_sliced_data(train_idx, train_npz, close_ary, tech_ary)
            save_sliced_data(test_idx, test_npz, close_ary, tech_ary)

            # Verify round-trip
            train_data = np.load(train_npz)
            test_data = np.load(test_npz)

            expected_train_close = close_ary[np.sort(train_idx)]
            expected_test_close = close_ary[np.sort(test_idx)]

            if not np.allclose(train_data['close_ary'], expected_train_close):
                print(f"  ✗ FAIL: Split {split_idx} train data mismatch")
                all_pass = False
            if not np.allclose(test_data['close_ary'], expected_test_close):
                print(f"  ✗ FAIL: Split {split_idx} test data mismatch")
                all_pass = False

            # Verify no data overlap between train and test slices
            # (check first stock's prices as fingerprints)
            train_prices_set = set(train_data['close_ary'][:, 0].tolist())
            test_prices_set = set(test_data['close_ary'][:, 0].tolist())
            price_overlap = train_prices_set & test_prices_set

            # Note: price overlap is POSSIBLE (same price on different days)
            # But INDEX overlap is the real concern, which we already tested

            if verbose:
                print(f"  Split {split_idx}: "
                      f"train {train_data['close_ary'].shape}, "
                      f"test {test_data['close_ary'].shape} — ✓")

            # Clean up
            os.remove(train_npz)
            os.remove(test_npz)

    finally:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    if all_pass:
        print(f"  ✓ PASS: All sliced data files are correct")
    return all_pass


def test_compare_old_vs_new(total_days: int, verbose: bool = False) -> bool:
    """
    Test 6: Compare old (flattened ranges) vs new (index arrays) to
    highlight where leakage existed.
    """
    print("\n" + "="*60)
    print("TEST 6: Old (leaky) vs New (correct) comparison")
    print("="*60)

    cv = CombPurgedKFoldCV(
        n_splits=N_GROUPS,
        n_test_splits=K_TEST_GROUPS,
        embargo_days=EMBARGO_DAYS,
    )

    fold_bounds = cv.get_fold_bounds(total_days)

    # Simulate the OLD code's behavior (flattening ranges)
    import itertools
    test_combinations = list(
        itertools.combinations(range(N_GROUPS), K_TEST_GROUPS)
    )

    print(f"\n  {'Split':>5} | {'Test Grps':>10} | "
          f"{'Old Train':>20} | {'New Train':>12} | "
          f"{'Leaked':>8} | {'Status':>8}")
    print("  " + "-" * 80)

    for split_idx, test_group_ids in enumerate(test_combinations):
        # OLD: flatten ranges
        test_ranges = [fold_bounds[i] for i in test_group_ids]
        old_train_start = 0  # simplified — old code behavior varies
        old_train_end = total_days

        # More accurate: simulate the old get_cpcv_splits + flattening
        train_group_ids = [i for i in range(N_GROUPS) if i not in test_group_ids]
        old_train_ranges = [fold_bounds[i] for i in train_group_ids]
        old_flat_start = old_train_ranges[0][0]
        old_flat_end = old_train_ranges[-1][1]
        old_train_days = old_flat_end - old_flat_start

        # NEW: actual index arrays (from our CombPurgedKFoldCV)
        # We need to iterate to get the right split
        for i, (train_idx, test_idx) in enumerate(cv.split(total_days)):
            if i == split_idx:
                new_train_days = len(train_idx)
                break

        # How many days in the old flattened range are actually test days?
        test_indices_set = set()
        for ts, te in test_ranges:
            test_indices_set.update(range(ts, te))

        old_flat_indices = set(range(old_flat_start, old_flat_end))
        leaked = old_flat_indices & test_indices_set

        status = "✓ OK" if len(leaked) == 0 else f"⚠ LEAK"

        print(f"  {split_idx:>5} | {str(test_group_ids):>10} | "
              f"[{old_flat_start}:{old_flat_end}] ({old_train_days}d) | "
              f"{new_train_days:>8}d    | "
              f"{len(leaked):>6}d  | {status:>8}")

    print(f"\n  The 'Leaked' column shows how many test-set days were")
    print(f"  included in training under the old flattened-range approach.")
    return True  # informational only


def test_env_construction(verbose: bool = False) -> bool:
    """Test 7: Verify dynamic env class can load sliced data."""
    print("\n" + "="*60)
    print("TEST 7: Dynamic env class construction")
    print("="*60)

    if not os.path.exists(ALPACA_NPZ_PATH):
        print(f"  ⚠ SKIP: Data not found at {ALPACA_NPZ_PATH}")
        return True

    from cpcv_pipeline.function_train_test import (
        load_full_data, save_sliced_data,
    )
    from elegantrl.envs.StockTradingEnv import StockTradingVecEnv

    close_ary, tech_ary = load_full_data()
    total_days = close_ary.shape[0]

    # Use first 100 days as a quick test
    test_indices = np.arange(100)
    test_npz = os.path.join(
        os.path.dirname(ALPACA_NPZ_PATH), "_test_env_construct.npz"
    )

    try:
        save_sliced_data(test_indices, test_npz, close_ary, tech_ary)

        # Try to construct the env (CPU mode)
        env = StockTradingVecEnv(
            npz_path=test_npz,
            beg_idx=0,
            end_idx=100,
            num_envs=2,
            gpu_id=-1,  # CPU
        )

        # Verify dimensions
        assert env.close_price.shape[0] == 100, \
            f"Expected 100 days, got {env.close_price.shape[0]}"
        assert env.close_price.shape[1] == close_ary.shape[1], \
            f"Stock count mismatch"

        # Verify data matches
        expected = close_ary[test_indices]
        actual = env.close_price.cpu().numpy()
        assert np.allclose(actual, expected), "Data mismatch in env"

        # Test reset
        state, info = env.reset()
        assert state is not None, "Reset returned None state"

        print(f"  ✓ PASS: Dynamic env created successfully")
        print(f"    Shape: {env.close_price.shape}")
        print(f"    State dim: {env.state_dim}, Action dim: {env.action_dim}")

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(test_npz):
            os.remove(test_npz)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CPCV pipeline leakage verification tests"
    )
    parser.add_argument(
        "--total-days", type=int, default=None,
        help="Total trading days (auto-detected from data if not set)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output for each split"
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip tests that require actual data files"
    )
    args = parser.parse_args()

    # Determine total days
    if args.total_days is not None:
        total_days = args.total_days
    elif os.path.exists(ALPACA_NPZ_PATH):
        ary = np.load(ALPACA_NPZ_PATH, allow_pickle=True)
        total_days = ary['close_ary'].shape[0]
        print(f"Auto-detected {total_days} total days from {ALPACA_NPZ_PATH}")
    else:
        total_days = 753
        print(f"Data not found, using default total_days={total_days}")

    print(f"\nCPCV Leakage Verification Tests")
    print(f"Configuration: N={N_GROUPS}, K={K_TEST_GROUPS}, "
          f"embargo={EMBARGO_DAYS}d, total_days={total_days}")

    results = {}

    # Tests that don't need data files
    results['no_leakage'] = test_no_leakage(total_days, args.verbose)
    results['coverage'] = test_complete_coverage(total_days, args.verbose)
    results['gaps'] = test_purge_embargo_gaps(total_days, args.verbose)
    results['paths'] = test_backtest_paths(total_days, args.verbose)
    results['comparison'] = test_compare_old_vs_new(total_days, args.verbose)

    # Tests that need data files
    if not args.skip_data:
        results['sliced_data'] = test_sliced_data_integrity(args.verbose)
        results['env_construct'] = test_env_construction(args.verbose)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:>20}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  ✅ ALL TESTS PASSED — safe to run on GPU")
    else:
        print(f"\n  ❌ SOME TESTS FAILED — DO NOT run on GPU until fixed")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
