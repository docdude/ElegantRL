"""
Phase 4B: Bet Sizing

Maps meta-label probabilities to position sizes using RiskLabAI's
bet sizing functions. Supports:

1. Normal CDF sizing (de Prado): size = side × (2Φ(p) - 1)
2. Kelly criterion: f = (p·b - q) / b
3. Discretized tiers: skip / half / full

Input: meta-label probabilities + signal sides
Output: position sizes for each signal
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def probability_to_bet_size(
    probabilities: np.ndarray,
    sides: np.ndarray,
    method: str = "cdf",
    reward_risk_ratio: float = 1.5,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert meta-label probabilities to position sizes.

    Parameters
    ----------
    probabilities : np.ndarray
        P(success) from meta-label model, range [0, 1].
    sides : np.ndarray
        Signal direction: +1 (long) or -1 (short).
    method : str
        "cdf"     — Normal CDF transform (de Prado): size = side × (2Φ(p) - 1)
        "kelly"   — Kelly criterion: f = (p·b - q) / b
        "tiered"  — Discrete tiers: 0 / 0.5 / 1.0
    reward_risk_ratio : float
        For Kelly: b = reward/risk ratio (TP_mult / SL_mult).
    threshold : float
        Minimum probability to take a trade.

    Returns
    -------
    sizes : np.ndarray, range [-1, 1]
        Positive = long, negative = short, 0 = skip.
    """
    sizes = np.zeros_like(probabilities)
    active = probabilities >= threshold

    if method == "cdf":
        from RiskLabAI.backtest.bet_sizing import probability_bet_size
        sizes[active] = probability_bet_size(
            probabilities[active], sides[active]
        )

    elif method == "kelly":
        p = probabilities[active]
        q = 1.0 - p
        b = reward_risk_ratio
        kelly_f = (p * b - q) / b
        kelly_f = np.clip(kelly_f, 0, 1)  # no negative sizing
        sizes[active] = sides[active] * kelly_f

    elif method == "tiered":
        # Discrete tiers
        for i in np.where(active)[0]:
            p = probabilities[i]
            if p >= 0.7:
                sizes[i] = sides[i] * 1.0    # full size
            elif p >= 0.55:
                sizes[i] = sides[i] * 0.5    # half size
            # else: skip (stays 0)

    else:
        raise ValueError(f"Unknown method: {method}")

    n_active = active.sum()
    n_skip = (~active).sum()
    logger.info(
        f"Bet sizing ({method}): {n_active} active, {n_skip} skipped, "
        f"avg |size|={np.abs(sizes[active]).mean():.3f}" if n_active > 0 else
        f"Bet sizing ({method}): 0 active, {n_skip} skipped"
    )
    return sizes


def compute_average_active_bet_size(
    signals_df: pd.DataFrame,
    sizes: np.ndarray,
    n_total_bars: int,
) -> np.ndarray:
    """
    Compute the time-series of average active bet size.

    At each bar, averages the bet sizes of all currently active signals
    (between entry and exit). Uses RiskLabAI's Numba-optimized function.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Must have bar_idx and exit_bar columns.
    sizes : np.ndarray
        Bet sizes from probability_to_bet_size().
    n_total_bars : int
        Total bars in the dataset.

    Returns
    -------
    avg_sizes : np.ndarray, shape (n_total_bars,)
    """
    from RiskLabAI.backtest.bet_sizing import average_bet_sizes

    bar_indices = np.arange(n_total_bars, dtype=np.int64)
    start_dates = signals_df["bar_idx"].values.astype(np.int64)
    end_dates = signals_df["exit_bar"].values.astype(np.int64)

    avg_sizes = average_bet_sizes(bar_indices, start_dates, end_dates, sizes)
    return avg_sizes
