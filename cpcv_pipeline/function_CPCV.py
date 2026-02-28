"""
Combinatorial Purged K-Fold Cross-Validation (CPCV) for DRL.

Based on:
- Lopez de Prado (2018) "Advances in Financial Machine Learning"
- AI4Finance-Foundation/FinRL_Crypto/function_CPCV.py
- Berend & Bruce Yang, "Combinatorial PurgedKFold CV for Deep RL"
- RiskLabAI: Adaptive CPCV and Bagged CPCV extensions

This module provides:
1. CombPurgedKFoldCV: generates train/test index arrays with purging + embargo
2. AdaptiveCombPurgedKFoldCV: shifts group boundaries based on an external
   feature (e.g. volatility) to avoid splitting at regime transitions
3. BaggedCombPurgedKFoldCV: trains multiple seeds per split and ensembles
4. back_test_paths_generator: assembles C(N,K) splits into φ backtest paths
5. Verification utilities to detect data leakage

Key difference from our old get_cpcv_splits():
- Returns ACTUAL INDEX ARRAYS (np.ndarray), NOT (start, end) range tuples
- Train indices are the complement of test indices, with purging/embargo applied
- Disjoint test groups produce disjoint train index arrays (no flattening)
"""

import itertools as itt
import numbers
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Core CPCV class (adapted from FinRL_Crypto/function_CPCV.py)
# ─────────────────────────────────────────────────────────────────────────────

class CombPurgedKFoldCV:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Splits `total_samples` into `n_splits` groups. For each combination of
    `n_test_splits` groups chosen as the test set, yields (train_indices,
    test_indices) with purging and embargo applied to avoid leakage.

    Parameters
    ----------
    n_splits : int
        Number of groups to divide data into (N).
    n_test_splits : int
        Number of groups used as test set per split (K).
    embargo_days : int
        Number of days to embargo after each test fold boundary.
        Train samples within `embargo_days` after a test fold are removed.

    Properties
    ----------
    n_combinations : int
        C(N, K) = total number of train/test splits.
    n_paths : int
        Number of unique backtest paths = C(N,K) * K / N.
    """

    def __init__(self, n_splits: int = 5, n_test_splits: int = 2,
                 embargo_days: int = 7):
        if not isinstance(n_splits, numbers.Integral) or n_splits < 2:
            raise ValueError(f"n_splits must be integer >= 2, got {n_splits}")
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"n_test_splits must be integer, got {n_test_splits}")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError(
                f"n_test_splits must be in [1, n_splits-1], got {n_test_splits}"
            )
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")

        self.n_splits = int(n_splits)
        self.n_test_splits = int(n_test_splits)
        self.embargo_days = int(embargo_days)

    @property
    def n_combinations(self) -> int:
        """Total number of C(N, K) splits."""
        return len(list(itt.combinations(range(self.n_splits), self.n_test_splits)))

    @property
    def n_paths(self) -> int:
        """Number of unique backtest paths."""
        return self.n_combinations * self.n_test_splits // self.n_splits

    def split(self, total_samples: int
              ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, test_indices) for each split.

        Parameters
        ----------
        total_samples : int
            Total number of time steps (days) in the dataset.

        Yields
        ------
        train_indices : np.ndarray of int
            Indices for training (with purging + embargo applied).
        test_indices : np.ndarray of int
            Indices for testing.
        """
        all_indices = np.arange(total_samples)

        # Compute fold boundaries (equal-sized groups)
        fold_bounds = []
        fold_size = total_samples // self.n_splits
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else total_samples
            fold_bounds.append((start, end))

        # Generate all C(N, K) combinations of test folds
        test_combinations = list(
            itt.combinations(range(self.n_splits), self.n_test_splits)
        )
        # Reverse so first split has test folds at end (convention)
        test_combinations.reverse()

        for test_fold_ids in test_combinations:
            # Build test indices
            test_indices = np.concatenate([
                all_indices[fold_bounds[i][0]:fold_bounds[i][1]]
                for i in test_fold_ids
            ])

            # Build train indices = complement of test indices
            train_indices = np.setdiff1d(all_indices, test_indices)

            # Apply purging + embargo for each test fold
            for fold_id in test_fold_ids:
                test_start, test_end = fold_bounds[fold_id]

                # Purge: remove train samples immediately BEFORE test fold
                # (whose evaluation time might overlap with test prediction)
                # For DRL daily data, purge = remove train samples adjacent to
                # the test fold start (conservative: 1 day)
                if test_start > 0:
                    purge_start = max(0, test_start - 1)
                    purge_indices = all_indices[purge_start:test_start]
                    train_indices = np.setdiff1d(train_indices, purge_indices)

                # Embargo: remove train samples within embargo_days AFTER
                # the test fold end
                if test_end < total_samples and self.embargo_days > 0:
                    embargo_end = min(total_samples, test_end + self.embargo_days)
                    embargo_indices = all_indices[test_end:embargo_end]
                    train_indices = np.setdiff1d(train_indices, embargo_indices)

            yield train_indices, test_indices

    def get_fold_bounds(self, total_samples: int) -> List[Tuple[int, int]]:
        """Return the (start, end) boundaries for each fold group."""
        fold_size = total_samples // self.n_splits
        bounds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else total_samples
            bounds.append((start, end))
        return bounds

    def describe_splits(self, total_samples: int) -> str:
        """Print a human-readable description of all splits."""
        lines = []
        fold_bounds = self.get_fold_bounds(total_samples)
        lines.append(f"CPCV Configuration: N={self.n_splits}, K={self.n_test_splits}, "
                     f"embargo={self.embargo_days}d")
        lines.append(f"Total samples: {total_samples}")
        lines.append(f"Fold size: {total_samples // self.n_splits}")
        lines.append(f"Number of splits: {self.n_combinations}")
        lines.append(f"Number of paths: {self.n_paths}")
        lines.append(f"\nFold boundaries:")
        for i, (s, e) in enumerate(fold_bounds):
            lines.append(f"  Group {i}: [{s}:{e}] ({e - s} days)")
        lines.append(f"\nSplit details:")
        for split_idx, (train_idx, test_idx) in enumerate(self.split(total_samples)):
            test_ranges = _indices_to_ranges(test_idx)
            train_ranges = _indices_to_ranges(train_idx)
            lines.append(f"\n  Split {split_idx}:")
            lines.append(f"    Test groups: {_identify_groups(test_idx, fold_bounds)}")
            lines.append(f"    Test ranges:  {test_ranges}  ({len(test_idx)} days)")
            lines.append(f"    Train ranges: {train_ranges}  ({len(train_idx)} days)")
            # Check for overlap
            overlap = np.intersect1d(train_idx, test_idx)
            if len(overlap) > 0:
                lines.append(f"    ⚠️  LEAKAGE: {len(overlap)} overlapping indices!")
            else:
                lines.append(f"    ✓ No leakage")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive CPCV (A-CPCV)
# Based on RiskLabAI: shifts group boundaries based on an external feature
# (e.g. rolling volatility) so that splits avoid regime transitions.
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveCombPurgedKFoldCV(CombPurgedKFoldCV):
    """
    Adaptive Combinatorial Purged K-Fold Cross-Validation.

    Extends standard CPCV by making group boundaries *adaptive* —
    they shift based on an external feature (e.g. rolling volatility,
    market regime indicator) at the boundary points.

    Algorithm:
    1. Create ``n_splits * n_subsplits`` fine-grained sub-intervals.
    2. Initial group boundaries fall every ``n_subsplits`` sub-intervals
       (same as standard CPCV equal-size groups).
    3. At each boundary, check the external feature value:
       - If feature < lower_quantile → shift boundary RIGHT (+1 subsplit)
       - If feature > upper_quantile → shift boundary LEFT  (-1 subsplit)
       This moves boundaries *away* from extreme feature values, keeping
       similar-regime days together within the same fold group.
    4. Result: unequal-size groups that adapt to market conditions.

    Everything else (combinatorial enumeration, purging, embargo, path
    construction) remains unchanged from standard CPCV.

    Parameters
    ----------
    n_splits : int
        Number of groups (N).
    n_test_splits : int
        Number of test groups per split (K).
    embargo_days : int
        Embargo days after each test fold boundary.
    external_feature : np.ndarray
        1-D array of length ``total_samples``, e.g. rolling volatility.
        Used to decide boundary shifts.
    n_subsplits : int, default 3
        Sub-intervals per group.  Higher = finer boundary adjustments.
    lower_quantile : float, default 0.25
        Feature below this percentile → shift boundary.
    upper_quantile : float, default 0.75
        Feature above this percentile → shift boundary.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_days: int = 7,
        external_feature: np.ndarray = None,
        n_subsplits: int = 3,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75,
    ):
        super().__init__(n_splits, n_test_splits, embargo_days)
        if external_feature is None:
            raise ValueError("external_feature must be provided for A-CPCV")
        self.external_feature = np.asarray(external_feature)
        self.n_subsplits = n_subsplits
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    # -- override fold boundary computation --------------------------------

    def get_fold_bounds(self, total_samples: int) -> List[Tuple[int, int]]:
        """Compute adaptive fold boundaries using the external feature.

        Returns list of (start, end) tuples — groups may have unequal sizes.
        """
        if len(self.external_feature) != total_samples:
            raise ValueError(
                f"external_feature length ({len(self.external_feature)}) "
                f"!= total_samples ({total_samples})"
            )

        feat = self.external_feature
        n_total_sub = self.n_splits * self.n_subsplits

        # Fine-grained subsplit boundaries
        sub_size = total_samples / n_total_sub
        sub_starts = [int(round(i * sub_size)) for i in range(n_total_sub)]

        # Quantile thresholds on the full feature
        lower_thresh = np.percentile(feat, self.lower_quantile * 100)
        upper_thresh = np.percentile(feat, self.upper_quantile * 100)

        # Initial group boundaries: every n_subsplits sub-intervals
        # border_sub_indices[i] is the subsplit index that starts group i+1
        border_subs = list(range(self.n_subsplits, n_total_sub, self.n_subsplits))

        # Shift each border based on feature value at that boundary
        adjusted = []
        for b in border_subs:
            if b >= len(sub_starts):
                adjusted.append(b)
                continue
            fval = feat[sub_starts[b]] if sub_starts[b] < total_samples else feat[-1]
            shift = 0
            if fval < lower_thresh:
                shift = -1
            elif fval > upper_thresh:
                shift = +1
            new_b = max(1, min(n_total_sub - 1, b - shift))
            adjusted.append(new_b)

        # Convert subsplit indices to sample indices
        split_points = sorted(set(sub_starts[b] for b in adjusted
                                  if b < len(sub_starts)))

        # Build fold bounds
        bounds = []
        prev = 0
        for sp in split_points:
            if sp > prev:
                bounds.append((prev, sp))
                prev = sp
        bounds.append((prev, total_samples))

        # If we have more groups than n_splits (numerical edge), merge smallest
        while len(bounds) > self.n_splits:
            # merge the two smallest adjacent groups
            sizes = [e - s for s, e in bounds]
            min_idx = int(np.argmin(sizes))
            if min_idx == 0:
                bounds[1] = (bounds[0][0], bounds[1][1])
                bounds.pop(0)
            else:
                bounds[min_idx - 1] = (bounds[min_idx - 1][0], bounds[min_idx][1])
                bounds.pop(min_idx)

        # If we have fewer, split the largest
        while len(bounds) < self.n_splits:
            sizes = [e - s for s, e in bounds]
            max_idx = int(np.argmax(sizes))
            s, e = bounds[max_idx]
            mid = (s + e) // 2
            bounds[max_idx] = (s, mid)
            bounds.insert(max_idx + 1, (mid, e))

        return bounds

    def split(self, total_samples: int
              ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) with adaptive fold boundaries."""
        all_indices = np.arange(total_samples)
        fold_bounds = self.get_fold_bounds(total_samples)

        test_combinations = list(
            itt.combinations(range(self.n_splits), self.n_test_splits)
        )
        test_combinations.reverse()

        for test_fold_ids in test_combinations:
            test_indices = np.concatenate([
                all_indices[fold_bounds[i][0]:fold_bounds[i][1]]
                for i in test_fold_ids
            ])
            train_indices = np.setdiff1d(all_indices, test_indices)

            for fold_id in test_fold_ids:
                test_start, test_end = fold_bounds[fold_id]
                if test_start > 0:
                    purge_start = max(0, test_start - 1)
                    purge_indices = all_indices[purge_start:test_start]
                    train_indices = np.setdiff1d(train_indices, purge_indices)
                if test_end < total_samples and self.embargo_days > 0:
                    embargo_end = min(total_samples, test_end + self.embargo_days)
                    embargo_indices = all_indices[test_end:embargo_end]
                    train_indices = np.setdiff1d(train_indices, embargo_indices)

            yield train_indices, test_indices

    def describe_splits(self, total_samples: int) -> str:
        """Print A-CPCV split description with adaptive group sizes."""
        lines = []
        fold_bounds = self.get_fold_bounds(total_samples)
        lines.append(f"Adaptive CPCV: N={self.n_splits}, K={self.n_test_splits}, "
                     f"embargo={self.embargo_days}d, "
                     f"subsplits={self.n_subsplits}, "
                     f"quantiles=[{self.lower_quantile}, {self.upper_quantile}]")
        lines.append(f"Total samples: {total_samples}")
        lines.append(f"Number of splits: {self.n_combinations}")
        lines.append(f"Number of paths: {self.n_paths}")
        lines.append(f"\nAdaptive fold boundaries:")
        eq_size = total_samples // self.n_splits
        for i, (s, e) in enumerate(fold_bounds):
            diff = (e - s) - eq_size
            sign = "+" if diff >= 0 else ""
            lines.append(f"  Group {i}: [{s}:{e}] ({e - s} days, "
                         f"{sign}{diff} vs equal)")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Bagged CPCV (B-CPCV)
# Based on RiskLabAI: trains multiple agents per split with different seeds,
# then ensembles. For DRL, "bagging" = multi-seed training per split.
# ─────────────────────────────────────────────────────────────────────────────

class BaggedCombPurgedKFoldCV(CombPurgedKFoldCV):
    """
    Bagged Combinatorial Purged K-Fold Cross-Validation for DRL.

    Same splits as standard CPCV, but each split trains ``n_bags`` agents
    with different random seeds.  At evaluation time, actions from all
    bags are averaged (continuous) or majority-voted (discrete).

    Parameters
    ----------
    n_splits : int
        Number of groups (N).
    n_test_splits : int
        Number of test groups per split (K).
    embargo_days : int
        Embargo days after each test fold boundary.
    n_bags : int, default 5
        Number of bagged agents per split.
    base_seed : int, default 1943
        Base random seed.  Bag ``b`` uses seed ``base_seed + b``.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_days: int = 7,
        n_bags: int = 5,
        base_seed: int = 1943,
    ):
        super().__init__(n_splits, n_test_splits, embargo_days)
        if n_bags < 2:
            raise ValueError(f"n_bags must be >= 2, got {n_bags}")
        self.n_bags = n_bags
        self.base_seed = base_seed

    def bag_seeds(self) -> List[int]:
        """Return the list of seeds for each bag."""
        return [self.base_seed + b for b in range(self.n_bags)]

    def describe_splits(self, total_samples: int) -> str:
        """Print B-CPCV description."""
        base = super().describe_splits(total_samples)
        lines = [
            f"\nBagged CPCV: {self.n_bags} bags per split "
            f"(seeds: {self.bag_seeds()})",
            f"Total training runs: {self.n_combinations} splits × "
            f"{self.n_bags} bags = {self.n_combinations * self.n_bags}",
            "",
            base,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest path generator
# ─────────────────────────────────────────────────────────────────────────────

def back_test_paths_generator(
    total_samples: int,
    n_groups: int,
    n_test_groups: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate backtest path assignments from CPCV splits.

    Each backtest path covers ALL time steps, assembled from test folds of
    different splits. This allows constructing φ = C(N,K)*K/N independent
    OOS backtest paths.

    Parameters
    ----------
    total_samples : int
        Total number of time steps.
    n_groups : int
        Number of fold groups (N).
    n_test_groups : int
        Number of test groups per split (K).

    Returns
    -------
    is_test : np.ndarray, shape (total_samples, C(N,K))
        Boolean array. is_test[t, s] = True if sample t is in the test set
        of split s.
    paths : np.ndarray, shape (total_samples, n_paths)
        paths[t, p] = split index from which sample t's prediction comes
        in backtest path p.
    path_folds : np.ndarray, shape (n_groups, n_paths)
        path_folds[g, p] = which split provides the prediction for
        group g in path p.
    """
    # Assign each sample to a group
    group_num = np.arange(total_samples) // (total_samples // n_groups)
    group_num[group_num >= n_groups] = n_groups - 1

    # All C(N, K) test group combinations
    test_groups = np.array(
        list(itt.combinations(np.arange(n_groups), n_test_groups))
    )
    C_nk = len(test_groups)
    n_paths = C_nk * n_test_groups // n_groups

    # Track which groups are test groups in each split
    is_test_group = np.full((n_groups, C_nk), fill_value=False)
    is_test = np.full((total_samples, C_nk), fill_value=False)

    for s, pair in enumerate(test_groups):
        for g in pair:
            is_test_group[g, s] = True
        mask = np.isin(group_num, pair)
        is_test[mask, s] = True

    # Build paths: for each path, assign each group to a split
    path_folds = np.full((n_groups, n_paths), fill_value=np.nan)
    is_test_group_remaining = is_test_group.copy()

    for p in range(n_paths):
        for g in range(n_groups):
            # Find first remaining split where this group is a test group
            s_idx = is_test_group_remaining[g, :].argmax().astype(int)
            if not is_test_group_remaining[g, s_idx]:
                raise ValueError(
                    f"No remaining split for group {g} in path {p}"
                )
            path_folds[g, p] = s_idx
            is_test_group_remaining[g, s_idx] = False

    # Build full path assignment matrix
    paths = np.full((total_samples, n_paths), fill_value=np.nan)
    for p in range(n_paths):
        for g in range(n_groups):
            mask = (group_num == g)
            paths[mask, p] = int(path_folds[g, p])

    return is_test, paths, path_folds


# ─────────────────────────────────────────────────────────────────────────────
# Leakage verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_no_leakage(cv: CombPurgedKFoldCV, total_samples: int) -> bool:
    """
    Verify that no train/test indices overlap for any split.

    Returns True if ALL splits are clean. Raises AssertionError on leakage.
    """
    for split_idx, (train_idx, test_idx) in enumerate(cv.split(total_samples)):
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0, (
            f"Leakage in split {split_idx}: {len(overlap)} overlapping indices"
        )
        # Also verify no index is unaccounted for (except purge/embargo gaps)
        combined = np.union1d(train_idx, test_idx)
        gap_indices = np.setdiff1d(np.arange(total_samples), combined)
        # Gap indices should be small (purge + embargo only)
        max_gap = cv.embargo_days * cv.n_test_splits + cv.n_test_splits
        assert len(gap_indices) <= max_gap * 2, (
            f"Split {split_idx}: too many gap indices ({len(gap_indices)}), "
            f"expected <= {max_gap * 2}"
        )
    return True


def verify_complete_oos_coverage(
    cv: CombPurgedKFoldCV,
    total_samples: int,
) -> bool:
    """
    Verify that the union of all test sets covers all samples.

    In CPCV, every sample appears in exactly K * C(N-1, K-1) / C(N, K)
    fraction of test sets. Each sample should appear in at least one test set.
    """
    all_test = np.zeros(total_samples, dtype=int)
    for _, test_idx in cv.split(total_samples):
        all_test[test_idx] += 1
    uncovered = np.where(all_test == 0)[0]
    assert len(uncovered) == 0, (
        f"{len(uncovered)} samples never appear in any test set: "
        f"indices {uncovered[:10]}..."
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _indices_to_ranges(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a sorted array of indices to a list of (start, end_exclusive) ranges."""
    if len(indices) == 0:
        return []
    indices = np.sort(indices)
    breaks = np.where(np.diff(indices) > 1)[0]
    ranges = []
    start = indices[0]
    for b in breaks:
        ranges.append((int(start), int(indices[b]) + 1))
        start = indices[b + 1]
    ranges.append((int(start), int(indices[-1]) + 1))
    return ranges


def format_segments(indices: np.ndarray) -> str:
    """Format index array as inclusive segments, e.g. '[0..298] + [457..598]'."""
    if len(indices) == 0:
        return '[]'
    indices = np.sort(indices)
    segs = []
    seg_start = int(indices[0])
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            segs.append(f'[{seg_start}..{int(indices[i - 1])}]')
            seg_start = int(indices[i])
    segs.append(f'[{seg_start}..{int(indices[-1])}]')
    return ' + '.join(segs)


def _identify_groups(
    indices: np.ndarray,
    fold_bounds: List[Tuple[int, int]],
) -> List[int]:
    """Identify which fold groups a set of indices belongs to."""
    groups = []
    for g, (start, end) in enumerate(fold_bounds):
        if np.any((indices >= start) & (indices < end)):
            groups.append(g)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# External feature computation for Adaptive CPCV
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_CHOICES = ['drawdown', 'volatility', 'vix', 'turbulence', 'ichimoku', 'rsi']

# Recommended window per feature (used when caller doesn't specify).
# Only drawdown and volatility use a rolling window; others ignore it.
FEATURE_DEFAULT_WINDOWS = {
    'drawdown':   63,   # quarterly rolling max drawdown
    'volatility': 21,   # monthly rolling std of log returns
    'vix':        None, # direct from tech_ary, no window
    'turbulence': None,
    'ichimoku':   None, # fixed 26-period Kijun-sen
    'rsi':        None,
}


def compute_external_feature(
    close_ary: np.ndarray,
    feature_name: str = 'drawdown',
    window: int | None = None,
    tech_ary: np.ndarray | None = None,
) -> np.ndarray:
    """Compute an external feature for Adaptive CPCV boundary adjustment.

    Parameters
    ----------
    close_ary : ndarray, shape (T, n_stocks)
    feature_name : str
        'volatility'  — rolling portfolio volatility (log-return std)
        'drawdown'    — rolling max drawdown (63d window, 0-mismatch winner)
        'vix'         — VIX level from tech_ary[:, -2]
        'turbulence'  — turbulence index from tech_ary[:, -1]
        'ichimoku'    — equal-weight portfolio price minus Kijun-sen (26d)
        'rsi'         — average RSI-30 across all stocks from tech_ary
    window : int or None
        Rolling window in days (used by volatility and drawdown).
        If None, uses FEATURE_DEFAULT_WINDOWS for the given feature.
    tech_ary : ndarray, shape (T, n_tech), optional
        Technical feature array. Required for 'vix', 'turbulence', and 'rsi'.

    Returns
    -------
    feature : ndarray, shape (T,)
    """
    if window is None:
        window = FEATURE_DEFAULT_WINDOWS.get(feature_name, 63)
    if feature_name == 'volatility':
        log_returns = np.diff(np.log(close_ary + 1e-9), axis=0)
        portfolio_returns = log_returns.mean(axis=1)
        vol = np.full(len(portfolio_returns), np.nan)
        for i in range(window - 1, len(portfolio_returns)):
            vol[i] = np.std(portfolio_returns[i - window + 1:i + 1])
        first_valid = vol[window - 1]
        vol[:window - 1] = first_valid
        # Prepend one value so length matches close_ary (T rows)
        feature = np.concatenate([[vol[0]], vol])
    elif feature_name == 'drawdown':
        # Equal-weight portfolio price, rolling max drawdown
        portfolio_price = close_ary.mean(axis=1)  # (T,)
        dd = np.zeros(len(portfolio_price))
        for i in range(len(portfolio_price)):
            start = max(0, i - window + 1)
            peak = np.max(portfolio_price[start:i + 1])
            dd[i] = (portfolio_price[i] - peak) / (peak + 1e-9)
        feature = dd
    elif feature_name == 'vix':
        if tech_ary is None:
            raise ValueError("tech_ary required for feature='vix'")
        feature = tech_ary[:, -2].copy()
    elif feature_name == 'turbulence':
        if tech_ary is None:
            raise ValueError("tech_ary required for feature='turbulence'")
        feature = tech_ary[:, -1].copy()
    elif feature_name == 'ichimoku':
        # Equal-weight portfolio price, then Kijun-sen (26-period mid)
        portfolio_price = close_ary.mean(axis=1)
        kijun_period = 26
        kijun = np.full(len(portfolio_price), np.nan)
        for i in range(kijun_period - 1, len(portfolio_price)):
            w = portfolio_price[i - kijun_period + 1:i + 1]
            kijun[i] = (w.max() + w.min()) / 2
        kijun[:kijun_period - 1] = kijun[kijun_period - 1]
        feature = portfolio_price - kijun
    elif feature_name == 'rsi':
        # Average RSI-30 across all stocks
        # tech_ary layout: 8 indicators per stock, then 2 market-wide (VIX, turb)
        if tech_ary is None:
            raise ValueError("tech_ary required for feature='rsi'")
        n_market = 2  # VIX, turbulence
        n_stocks = close_ary.shape[1]
        n_tech_per_stock = (tech_ary.shape[1] - n_market) // n_stocks  # 8
        rsi_offset = 3  # MACD=0, Boll_UB=1, Boll_LB=2, RSI-30=3
        rsi_cols = [rsi_offset + i * n_tech_per_stock for i in range(n_stocks)]
        feature = tech_ary[:, rsi_cols].mean(axis=1)
    else:
        raise ValueError(
            f"Unknown feature '{feature_name}'. "
            f"Choose from: {FEATURE_CHOICES}"
        )
    return feature
