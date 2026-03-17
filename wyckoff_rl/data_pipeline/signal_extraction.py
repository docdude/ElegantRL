"""
Phase 4B: Wyckoff Signal Extraction & Triple Barrier Labeling

1. Extract Wyckoff signal events (Spring, Upthrust, SC, BC) with feature snapshots
2. Apply triple barrier labeling using RiskLabAI
3. Generate sample weights using RiskLabAI uniqueness/concurrency

Input: analyzed DataFrame from Phase 1 (parquet)
Output: events DataFrame with labels + feature snapshots for meta-labeling
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

from .config import (
    WYCKOFF_FEATURE_COLUMNS,
    TRIPLE_BARRIER_DEFAULTS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Wyckoff Signal Event Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_wyckoff_signals(
    df: pd.DataFrame,
    min_volume_strength: int = 0,
) -> pd.DataFrame:
    """
    Extract Wyckoff signal events with feature snapshots.

    Looks for Spring, Upthrust, SellingClimax, BuyingClimax events.
    For each event, captures all Wyckoff features at that bar.

    Parameters
    ----------
    df : pd.DataFrame
        Analyzed DataFrame from Phase 1 (must have Wyckoff columns).
    min_volume_strength : int
        Minimum VolumeStrength to include signal (0 = all signals).

    Returns
    -------
    signals_df : pd.DataFrame
        One row per signal event with columns:
        - bar_idx: index into the original DataFrame
        - event_type: "Spring", "Upthrust", "SellingClimax", "BuyingClimax"
        - side: +1 (long signal) or -1 (short signal)
        - close: price at signal bar
        - All WYCKOFF_FEATURE_COLUMNS as feature snapshots
    """
    events = []

    event_map = {
        "Spring":        +1,  # long signal
        "Upthrust":      -1,  # short signal
        "SellingClimax": +1,  # long signal (exhaustion of selling)
        "BuyingClimax":  -1,  # short signal (exhaustion of buying)
    }

    for event_col, side in event_map.items():
        if event_col not in df.columns:
            continue
        mask = df[event_col] == 1
        if min_volume_strength > 0 and "VolumeStrength" in df.columns:
            mask = mask & (df["VolumeStrength"] >= min_volume_strength)

        event_bars = df[mask]
        for idx, row in event_bars.iterrows():
            event = {
                "bar_idx": idx if isinstance(idx, int) else df.index.get_loc(idx),
                "event_type": event_col,
                "side": side,
                "close": row["close"],
            }
            # Capture feature snapshot
            for col in WYCKOFF_FEATURE_COLUMNS:
                if col in row.index:
                    event[col] = row[col]
                else:
                    event[col] = 0.0
            events.append(event)

    signals_df = pd.DataFrame(events)
    if len(signals_df) > 0:
        signals_df = signals_df.sort_values("bar_idx").reset_index(drop=True)

    logger.info(
        f"Extracted {len(signals_df)} signals: "
        + ", ".join(
            f"{et}={len(signals_df[signals_df['event_type']==et])}"
            for et in event_map
            if et in signals_df.get("event_type", pd.Series()).values
        )
    )
    return signals_df


# ─────────────────────────────────────────────────────────────────────────────
# ATR Calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute Average True Range."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(window=period, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Triple Barrier Labeling
# ─────────────────────────────────────────────────────────────────────────────

def apply_triple_barrier(
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    pt_multiplier: float = None,
    sl_multiplier: float = None,
    vertical_bars: int = None,
    atr_period: int = None,
) -> pd.DataFrame:
    """
    Apply triple barrier labeling to Wyckoff signal events.

    For each signal, defines:
    - Upper barrier: close + pt_multiplier × ATR (take profit)
    - Lower barrier: close - sl_multiplier × ATR (stop loss)
    - Vertical barrier: max holding period in bars
    Side-adjusted: for short signals, barriers are flipped.

    Parameters
    ----------
    df : pd.DataFrame
        Full analyzed DataFrame with close/high/low prices.
    signals_df : pd.DataFrame
        Signal events from extract_wyckoff_signals().
    pt_multiplier, sl_multiplier : float
        Barrier widths as multiples of ATR.
    vertical_bars : int
        Maximum holding period.
    atr_period : int
        ATR lookback period.

    Returns
    -------
    labeled_df : pd.DataFrame
        signals_df with additional columns:
        - label: +1 (TP hit first) or 0 (SL/vertical hit first)
        - ret: log return at exit
        - exit_bar: index of exit bar
        - exit_type: "tp", "sl", or "vertical"
    """
    defaults = TRIPLE_BARRIER_DEFAULTS
    pt_multiplier = pt_multiplier or defaults["pt_multiplier"]
    sl_multiplier = sl_multiplier or defaults["sl_multiplier"]
    vertical_bars = vertical_bars or defaults["vertical_bars"]
    atr_period = atr_period or defaults["atr_period"]

    atr = compute_atr(df, period=atr_period)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n_bars = len(df)

    labels = []
    returns = []
    exit_bars = []
    exit_types = []

    for _, signal in signals_df.iterrows():
        entry_idx = int(signal["bar_idx"])
        entry_price = closes[entry_idx]
        side = signal["side"]
        barrier_width = atr.iloc[entry_idx]

        # Set barriers (side-adjusted)
        if side > 0:  # long
            tp_price = entry_price + pt_multiplier * barrier_width
            sl_price = entry_price - sl_multiplier * barrier_width
        else:  # short
            tp_price = entry_price - pt_multiplier * barrier_width
            sl_price = entry_price + sl_multiplier * barrier_width

        max_bar = min(entry_idx + vertical_bars, n_bars - 1)
        exit_bar = max_bar
        exit_type = "vertical"

        # Scan forward for barrier touch
        for j in range(entry_idx + 1, max_bar + 1):
            if side > 0:  # long
                if highs[j] >= tp_price:
                    exit_bar = j
                    exit_type = "tp"
                    break
                if lows[j] <= sl_price:
                    exit_bar = j
                    exit_type = "sl"
                    break
            else:  # short
                if lows[j] <= tp_price:
                    exit_bar = j
                    exit_type = "tp"
                    break
                if highs[j] >= sl_price:
                    exit_bar = j
                    exit_type = "sl"
                    break

        # Log return
        exit_price = closes[exit_bar]
        ret = side * np.log(exit_price / entry_price)

        # Label: 1 if TP hit first, 0 otherwise
        label = 1 if exit_type == "tp" else 0

        labels.append(label)
        returns.append(ret)
        exit_bars.append(exit_bar)
        exit_types.append(exit_type)

    labeled_df = signals_df.copy()
    labeled_df["label"] = labels
    labeled_df["ret"] = returns
    labeled_df["exit_bar"] = exit_bars
    labeled_df["exit_type"] = exit_types

    n_tp = sum(1 for t in exit_types if t == "tp")
    n_sl = sum(1 for t in exit_types if t == "sl")
    n_vert = sum(1 for t in exit_types if t == "vertical")
    logger.info(
        f"Triple barrier: {len(labeled_df)} events → "
        f"TP={n_tp}, SL={n_sl}, Vertical={n_vert}, "
        f"Win rate={sum(labels)/len(labels)*100:.1f}%"
    )
    return labeled_df


# ─────────────────────────────────────────────────────────────────────────────
# Sample Weights (uniqueness / concurrency)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(
    labeled_df: pd.DataFrame,
    n_total_bars: int,
) -> np.ndarray:
    """
    Compute sample weights based on event uniqueness (concurrency).

    For each event, weight = mean(1/c_t) over its active bars,
    where c_t = number of concurrent events at bar t.
    Memory-efficient: uses a concurrency array instead of dense indicator.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        From apply_triple_barrier(), needs bar_idx and exit_bar columns.
    n_total_bars : int
        Total number of bars in the dataset.

    Returns
    -------
    weights : np.ndarray, shape (n_events,)
    """
    starts = labeled_df["bar_idx"].values.astype(int)
    ends = labeled_df["exit_bar"].values.astype(int)
    n_events = len(labeled_df)

    # Step 1: compute concurrency at each bar (how many events overlap)
    concurrency = np.zeros(n_total_bars, dtype=np.int32)
    for i in range(n_events):
        concurrency[starts[i]:ends[i] + 1] += 1

    # Step 2: average uniqueness per event = mean(1/c_t) over active bars
    weights = np.zeros(n_events, dtype=np.float32)
    for i in range(n_events):
        s, e = starts[i], ends[i]
        c_slice = concurrency[s:e + 1].astype(np.float32)
        c_slice[c_slice == 0] = 1.0  # avoid division by zero
        weights[i] = np.mean(1.0 / c_slice)

    # Normalize to sum to N
    if weights.sum() > 0:
        weights *= n_events / weights.sum()

    logger.info(
        f"Sample weights: min={weights.min():.3f}, max={weights.max():.3f}, "
        f"mean={weights.mean():.3f}"
    )
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Full Phase 4B Signal Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_signal_pipeline(
    df: pd.DataFrame,
    min_volume_strength: int = 0,
    pt_multiplier: float = None,
    sl_multiplier: float = None,
    vertical_bars: int = None,
) -> dict:
    """
    Run the full signal extraction + triple barrier pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Analyzed DataFrame from Phase 1.

    Returns
    -------
    dict with signals, labels, weights, and metadata
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Extract signals
    signals = extract_wyckoff_signals(df, min_volume_strength=min_volume_strength)
    if len(signals) == 0:
        logger.warning("No Wyckoff signals found!")
        return {"signals": signals, "n_signals": 0}

    # 2. Apply triple barrier
    labeled = apply_triple_barrier(
        df, signals,
        pt_multiplier=pt_multiplier,
        sl_multiplier=sl_multiplier,
        vertical_bars=vertical_bars,
    )

    # 3. Compute sample weights
    weights = compute_sample_weights(labeled, n_total_bars=len(df))

    return {
        "signals": signals,
        "labeled": labeled,
        "weights": weights,
        "n_signals": len(signals),
        "win_rate": labeled["label"].mean(),
        "feature_columns": WYCKOFF_FEATURE_COLUMNS,
    }
