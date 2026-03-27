"""
Structural labeling pipeline for supervised pre-training of transformer heads.

Derives phase and event labels from RAW price/volume structure, NOT from
pre-computed feature scores. This avoids the circular problem where the
transformer would just learn to replicate the feature pipeline's output.

Data sources:
  - Parquet file: raw OHLCV + delta + CVD per range bar
  - Wave segmentation: _segment_waves_nolag() on raw bars
  - Close prices: forward excursion

Labels produced:
  - phase (n_bars,) int64: Wyckoff phase from wave geometry
  - events (n_bars, N_EVENTS) float32: structural event detection
  - excursion (n_bars, 2) float32: forward favorable/adverse move

Usage:
    labels = generate_structural_labels(
        parquet_path='pipeline_output/wyckoff_us30_100pt_bars.parquet'
    )
"""

import numpy as np
import pandas as pd
from .config import N_PHASES, N_EVENTS


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def generate_structural_labels(
    parquet_path: str,
    npz_path: str | None = None,
    excursion_horizon: int = 20,
    range_lookback: int = 60,
    range_min_tests: int = 3,
    climax_vol_sigma: float = 2.0,
    spring_reclaim_bars: int = 5,
) -> dict:
    """
    Generate structural Wyckoff labels from raw OHLCV data.

    Parameters
    ----------
    parquet_path : str
        Path to parquet with raw OHLCV + delta + CVD bars.
    npz_path : str, optional
        If provided, verifies alignment (same close prices).
    excursion_horizon : int
        Forward bars for favorable/adverse excursion.
    range_lookback : int
        Bars to look back for range detection.
    range_min_tests : int
        Min touches of support/resistance to confirm range.
    climax_vol_sigma : float
        Volume z-score threshold for climax detection.
    spring_reclaim_bars : int
        Max bars to reclaim range boundary after break.

    Returns
    -------
    dict with 'phase', 'events', 'excursion'
    """
    df = pd.read_parquet(parquet_path)
    n = len(df)

    # Verify alignment if NPZ provided
    if npz_path is not None:
        npz_data = np.load(npz_path, allow_pickle=True)
        npz_close = npz_data['close_ary'].flatten()
        parquet_close = df['close'].values
        if len(npz_close) != n:
            print(f"WARNING: NPZ has {len(npz_close)} bars, parquet has {n}")
        else:
            max_diff = np.abs(npz_close - parquet_close).max()
            if max_diff > 0.01:
                print(f"WARNING: NPZ/parquet close prices differ by up to {max_diff:.4f}")

    # Wave segmentation from raw OHLCV
    waves, lib_df = _get_wave_segments(df)

    # Structural range detection
    ranges = _detect_ranges(df, lookback=range_lookback, min_tests=range_min_tests)

    # Phase labels from wave geometry
    phase_labels = _label_phases_structural(df, waves, ranges)

    # Event labels from price structure
    event_labels = _label_events_structural(
        df, waves, lib_df, ranges,
        climax_vol_sigma=climax_vol_sigma,
        spring_reclaim_bars=spring_reclaim_bars,
    )

    # Forward excursion (this stays — it's from actual future prices)
    excursion = _compute_excursion(df['close'].values, excursion_horizon)

    return {
        'phase': phase_labels,
        'events': event_labels,
        'excursion': excursion,
    }


# Backwards compat alias
generate_weak_labels = generate_structural_labels


# ═══════════════════════════════════════════════════════════════════════
# Wave Segmentation
# ═══════════════════════════════════════════════════════════════════════

def _get_wave_segments(df: pd.DataFrame):
    """Run structural wave segmentation on raw OHLCV."""
    try:
        from wyckoff_effort.pipeline.wyckoff_features import _segment_waves_nolag
        return _segment_waves_nolag(df)
    except ImportError:
        return _simple_zigzag_waves(df), None


def _simple_zigzag_waves(df: pd.DataFrame, pct_reversal: float = 0.005) -> dict:
    """Fallback zigzag when srl-python-indicators not available."""
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    n = len(close)

    wave_dir = np.ones(n)
    wave_id = np.zeros(n, dtype=np.int32)

    direction = 1
    last_pivot = close[0]
    current_wave = 0

    for i in range(1, n):
        if direction == 1:
            if close[i] > last_pivot:
                last_pivot = close[i]
            elif (last_pivot - close[i]) / last_pivot > pct_reversal:
                direction = -1
                current_wave += 1
                last_pivot = close[i]
        else:
            if close[i] < last_pivot:
                last_pivot = close[i]
            elif (close[i] - last_pivot) / last_pivot > pct_reversal:
                direction = 1
                current_wave += 1
                last_pivot = close[i]

        wave_dir[i] = direction
        wave_id[i] = current_wave

    # Per-wave running extrema and volume
    wave_high = np.empty(n)
    wave_low = np.empty(n)
    wave_vol = np.empty(n)
    wave_delta = np.zeros(n)
    seg_start = 0
    for i in range(n):
        if i > 0 and wave_id[i] != wave_id[i - 1]:
            seg_start = i
        wave_high[i] = high[seg_start:i + 1].max()
        wave_low[i] = low[seg_start:i + 1].min()
        wave_vol[i] = volume[seg_start:i + 1].sum()

    return {
        'wave_dir': wave_dir,
        'wave_id': wave_id,
        'wave_high': wave_high,
        'wave_low': wave_low,
        'wave_vol': wave_vol,
        'wave_delta': wave_delta,
    }


# ═══════════════════════════════════════════════════════════════════════
# Range Detection (structural)
# ═══════════════════════════════════════════════════════════════════════

def _detect_ranges(
    df: pd.DataFrame, lookback: int = 60, min_tests: int = 3,
    touch_pct: float = 0.10,
) -> np.ndarray:
    """
    Detect consolidation ranges from raw price action.

    Uses percentile-based range boundaries from a PRIOR window
    (excluding current bar), so breaks below/above are possible.

    Support = 10th percentile of lows in lookback window.
    Resistance = 90th percentile of highs in lookback window.
    A "test" = bar extreme within touch_pct of range width of the boundary.

    Requires tests to be SPREAD OUT (not all contiguous), which
    distinguishes actual oscillation from a one-off dip/spike.

    Returns (n_bars, 4): [range_low, range_high, support_tests, resistance_tests]
    """
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    n = len(df)

    ranges = np.zeros((n, 4), dtype=np.float64)

    for i in range(lookback + 1, n):
        w_high = high[i - lookback:i]
        w_low = low[i - lookback:i]

        range_low = np.percentile(w_low, 10)
        range_high = np.percentile(w_high, 90)
        range_width = range_high - range_low

        if range_width < 1e-6:
            continue
        if range_width / close[i] > 0.04:
            continue

        tol = range_width * touch_pct

        # Find bars that test support/resistance
        s_mask = np.abs(w_low - range_low) < tol
        r_mask = np.abs(w_high - range_high) < tol
        support_tests = s_mask.sum()
        resistance_tests = r_mask.sum()

        if support_tests < min_tests or resistance_tests < min_tests:
            continue

        # Require tests to be spread across the window (not contiguous).
        # Count distinct "test clusters" — groups of tests separated by gaps.
        s_clusters = _count_clusters(s_mask)
        r_clusters = _count_clusters(r_mask)

        if s_clusters >= 2 and r_clusters >= 2:
            ranges[i] = [range_low, range_high, support_tests, resistance_tests]

    return ranges


def _count_clusters(mask: np.ndarray) -> int:
    """Count contiguous groups of True in a boolean array."""
    if not mask.any():
        return 0
    # Each transition from False→True starts a new cluster
    return 1 + np.sum(mask[1:] & ~mask[:-1])


# ═══════════════════════════════════════════════════════════════════════
# Phase Labels (from wave geometry, NOT from feature scores)
# ═══════════════════════════════════════════════════════════════════════

def _label_phases_structural(
    df: pd.DataFrame, waves: dict, ranges: np.ndarray,
    trend_window: int = 5,
) -> np.ndarray:
    """
    Label Wyckoff phases from swing high / swing low progression.

    Uses separate swing high and swing low sequences (not interleaved).
    - Markup (3): Recent swing highs mostly HH AND swing lows mostly HL.
    - Markdown (6): Recent swing highs mostly LH AND swing lows mostly LL.
    - Accumulation (1): Narrowing swings, HL bias, after decline.
    - Distribution (4): Narrowing swings, LH bias, after advance.
    - Re-accumulation (2): Range within uptrend context.
    - Re-distribution (5): Range within downtrend context.
    """
    n = len(df)
    labels = np.zeros(n, dtype=np.int64)

    wave_id = waves['wave_id']
    wave_dir = waves['wave_dir']
    wave_high = waves['wave_high']
    wave_low = waves['wave_low']

    sh_prices, sl_prices, sh_bars, sl_bars = _extract_pivots(
        wave_id, wave_dir, wave_high, wave_low)

    if len(sh_prices) < trend_window or len(sl_prices) < trend_window:
        return labels

    sh_idx = 0  # pointer into swing highs
    sl_idx = 0  # pointer into swing lows

    for i in range(n):
        while sh_idx < len(sh_bars) and sh_bars[sh_idx] <= i:
            sh_idx += 1
        while sl_idx < len(sl_bars) and sl_bars[sl_idx] <= i:
            sl_idx += 1

        if sh_idx < trend_window or sl_idx < trend_window:
            continue

        rh = sh_prices[sh_idx - trend_window:sh_idx]
        rl = sl_prices[sl_idx - trend_window:sl_idx]
        tw = trend_window - 1

        hh = sum(1 for j in range(1, len(rh)) if rh[j] > rh[j - 1])
        lh = sum(1 for j in range(1, len(rh)) if rh[j] < rh[j - 1])
        hl = sum(1 for j in range(1, len(rl)) if rl[j] > rl[j - 1])
        ll = sum(1 for j in range(1, len(rl)) if rl[j] < rl[j - 1])

        has_range = ranges[i, 2] > 0

        # Markup: strongly trending HH + HL (>= 75%)
        if hh >= tw * 0.75 and hl >= tw * 0.75:
            labels[i] = 3

        # Markdown: strongly trending LH + LL (>= 75%)
        elif ll >= tw * 0.75 and lh >= tw * 0.75:
            labels[i] = 6

        # Accumulation: higher-low bias (bottom-building, not yet markup)
        elif hl > ll and hh < tw * 0.75:
            labels[i] = 1

        # Distribution: lower-high bias (top-building, not yet markdown)
        elif lh > hh and ll < tw * 0.75:
            labels[i] = 4

        # Re-accumulation: range within uptrend (weaker trend + range)
        elif has_range and hl >= ll:
            labels[i] = 2

        # Re-distribution: range within downtrend
        elif has_range and lh >= hh:
            labels[i] = 5

    return labels


def _extract_pivots(wave_id, wave_dir, wave_high, wave_low):
    """
    Extract alternating swing highs and swing lows from wave segments.

    Filters out trivial waves (< 3 bars) to avoid noisy zigzag pivots.
    Returns separate swing high and swing low sequences with their bar indices,
    suitable for comparing HH/HL/LH/LL patterns.
    """
    n = len(wave_id)
    # Collect raw wave segments
    raw_waves = []  # (end_bar, is_up, wave_high, wave_low, wave_len)
    seg_start = 0
    prev_wave = wave_id[0]
    for i in range(1, n):
        if wave_id[i] != prev_wave:
            wave_len = i - seg_start
            raw_waves.append((i - 1, wave_dir[i - 1] > 0,
                              wave_high[i - 1], wave_low[i - 1], wave_len))
            seg_start = i
        prev_wave = wave_id[i]
    raw_waves.append((n - 1, wave_dir[n - 1] > 0,
                      wave_high[n - 1], wave_low[n - 1], n - seg_start))

    # Filter: merge trivial waves (< 3 bars) into neighbors
    min_wave_bars = 3
    significant = []
    for w in raw_waves:
        if w[4] >= min_wave_bars:
            significant.append(w)
        elif significant:
            prev = significant[-1]
            significant[-1] = (w[0], prev[1], max(prev[2], w[2]),
                               min(prev[3], w[3]), prev[4] + w[4])

    # Extract swing highs (from up-waves) and swing lows (from down-waves)
    swing_highs = []     # (bar_idx, price)
    swing_lows = []      # (bar_idx, price)
    all_pivots_bars = [] # combined chronological bar indices

    for bar_idx, is_up, wh, wl, wlen in significant:
        if is_up:
            swing_highs.append((bar_idx, wh))
        else:
            swing_lows.append((bar_idx, wl))
        all_pivots_bars.append(bar_idx)

    sh_bars = np.array([s[0] for s in swing_highs], dtype=np.int64) if swing_highs else np.array([], dtype=np.int64)
    sh_prices = np.array([s[1] for s in swing_highs]) if swing_highs else np.array([])
    sl_bars = np.array([s[0] for s in swing_lows], dtype=np.int64) if swing_lows else np.array([], dtype=np.int64)
    sl_prices = np.array([s[1] for s in swing_lows]) if swing_lows else np.array([])

    return sh_prices, sl_prices, sh_bars, sl_bars


# ═══════════════════════════════════════════════════════════════════════
# Event Labels (from price structure, NOT from feature scores)
# ═══════════════════════════════════════════════════════════════════════

def _label_events_structural(
    df: pd.DataFrame, waves: dict, lib_df,
    ranges: np.ndarray,
    climax_vol_sigma: float = 2.0,
    spring_reclaim_bars: int = 5,
) -> np.ndarray:
    """
    Detect Wyckoff events from raw price/volume structure.

    Spring (1):  Close breaks below range support, reclaims within N bars.
    Upthrust (2):  Close breaks above range resistance, falls back.
    Selling Climax (3):  Volume spike + wide down bar + reversal.
    Buying Climax (4):  Volume spike + wide up bar + reversal.
    Test (5):  Return to S/R on volume < 60% of recent avg.
    SOS (6):  Up-wave with expanding volume vs prior up-waves.
    SOW (7):  Down-wave with expanding volume vs prior down-waves.
    No Demand (8):  Up-wave with declining volume.
    No Supply (9):  Down-wave with declining volume.
    Effort > Result (10):  High volume + small displacement.
    LPS (11):  Spring + subsequent test near support within 30 bars.
    LPSY (12):  Upthrust + subsequent test near resistance within 30 bars.
    """
    n = len(df)
    labels = np.zeros((n, N_EVENTS), dtype=np.float32)

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    open_ = df['open'].values.astype(np.float64)

    wave_dir = waves['wave_dir']
    wave_id = waves['wave_id']
    wave_vol = waves['wave_vol']

    # Rolling volume stats for climax detection
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(20, min_periods=5).mean().values
    vol_std20 = vol_series.rolling(20, min_periods=5).std().values

    # For range bars: use duration as "bar width" proxy instead of displacement
    # (range bar displacement is always ~range_size by definition)
    has_duration = 'duration_seconds' in df.columns
    if has_duration:
        duration = df['duration_seconds'].values.astype(np.float64)
        dur_ma20 = pd.Series(duration).rolling(20, min_periods=5).mean().values
    else:
        duration = None
        dur_ma20 = None

    # Track completed wave volumes
    completed_up_vols = []
    completed_down_vols = []
    prev_wave_id = -1

    # Track event locations for LPS/LPSY derivation
    spring_bars = []
    upthrust_bars = []

    for i in range(1, n):
        # -- Wave completion tracking --
        if wave_id[i] != prev_wave_id and prev_wave_id >= 0:
            if wave_dir[i - 1] > 0:
                completed_up_vols.append(wave_vol[i - 1])
            else:
                completed_down_vols.append(wave_vol[i - 1])
        prev_wave_id = wave_id[i]

        range_low = ranges[i, 0]
        range_high = ranges[i, 1]
        has_range = ranges[i, 2] > 0

        # -- Spring: break below support + reclaim --
        if has_range and range_low > 0:
            if low[i] < range_low and close[i] > range_low:
                labels[i, 1] = 1.0
                spring_bars.append(i)
            elif i >= spring_reclaim_bars:
                for j in range(1, min(spring_reclaim_bars + 1, i)):
                    if low[i - j] < range_low and close[i] > range_low:
                        labels[i, 1] = 0.8
                        if not spring_bars or spring_bars[-1] != i:
                            spring_bars.append(i)
                        break

        # -- Upthrust: break above resistance + fall back --
        if has_range and range_high > 0:
            if high[i] > range_high and close[i] < range_high:
                labels[i, 2] = 1.0
                upthrust_bars.append(i)
            elif i >= spring_reclaim_bars:
                for j in range(1, min(spring_reclaim_bars + 1, i)):
                    if high[i - j] > range_high and close[i] < range_high:
                        labels[i, 2] = 0.8
                        if not upthrust_bars or upthrust_bars[-1] != i:
                            upthrust_bars.append(i)
                        break

        # -- Selling Climax: volume spike + fast bar (range bar) + reversal --
        if (vol_ma20[i] > 0 and vol_std20[i] > 0
                and volume[i] > vol_ma20[i] + climax_vol_sigma * vol_std20[i]
                and close[i] < open_[i]):
            # For range bars: "fast" = short duration (completed quickly)
            fast_bar = (duration is not None and dur_ma20[i] > 0
                        and duration[i] < dur_ma20[i] * 0.5)
            # Reversal: next bar closes higher OR strong rejection wick
            reversal = (i < n - 1 and close[i + 1] > close[i])
            if fast_bar or reversal:
                labels[i, 3] = 1.0

        # -- Buying Climax: volume spike + fast bar (range bar) + reversal --
        if (vol_ma20[i] > 0 and vol_std20[i] > 0
                and volume[i] > vol_ma20[i] + climax_vol_sigma * vol_std20[i]
                and close[i] > open_[i]):
            fast_bar = (duration is not None and dur_ma20[i] > 0
                        and duration[i] < dur_ma20[i] * 0.5)
            reversal = (i < n - 1 and close[i + 1] < close[i])
            if fast_bar or reversal:
                labels[i, 4] = 1.0

        # -- Test: return to S/R on low volume --
        if has_range and vol_ma20[i] > 0:
            rng_w = range_high - range_low
            near_support = abs(low[i] - range_low) < rng_w * 0.15
            near_resist = abs(high[i] - range_high) < rng_w * 0.15
            low_vol = volume[i] < vol_ma20[i] * 0.6
            if (near_support or near_resist) and low_vol:
                labels[i, 5] = 1.0

        # -- SOS: up-wave with expanding volume --
        if len(completed_up_vols) >= 2 and wave_dir[i] > 0:
            ref = np.mean(completed_up_vols[-4:])
            if wave_vol[i] > ref * 1.3:
                labels[i, 6] = 1.0

        # -- SOW: down-wave with expanding volume --
        if len(completed_down_vols) >= 2 and wave_dir[i] < 0:
            ref = np.mean(completed_down_vols[-4:])
            if wave_vol[i] > ref * 1.3:
                labels[i, 7] = 1.0

        # -- No Demand: up-wave with declining volume --
        if len(completed_up_vols) >= 2 and wave_dir[i] > 0:
            ref = np.mean(completed_up_vols[-4:])
            if wave_vol[i] < ref * 0.5:
                labels[i, 8] = 1.0

        # -- No Supply: down-wave with declining volume --
        if len(completed_down_vols) >= 2 and wave_dir[i] < 0:
            ref = np.mean(completed_down_vols[-4:])
            if wave_vol[i] < ref * 0.5:
                labels[i, 9] = 1.0

        # -- Effort > Result: high volume + slow completion (range bars) --
        if vol_ma20[i] > 0:
            high_effort = volume[i] > vol_ma20[i] * 1.3
            if duration is not None and dur_ma20[i] > 0:
                # Range bar: lots of volume but took a long time (indecision)
                slow_result = duration[i] > dur_ma20[i] * 1.5
            else:
                slow_result = False
            if high_effort and slow_result:
                labels[i, 10] = 1.0

        # -- LPS: spring within 30 bars + test near support --
        if has_range and spring_bars:
            if 0 < (i - spring_bars[-1]) <= 30:
                rng_w = range_high - range_low
                if (abs(low[i] - range_low) < rng_w * 0.2
                        and vol_ma20[i] > 0
                        and volume[i] < vol_ma20[i] * 0.6):
                    labels[i, 11] = 1.0

        # -- LPSY: upthrust within 30 bars + test near resistance --
        if has_range and upthrust_bars:
            if 0 < (i - upthrust_bars[-1]) <= 30:
                rng_w = range_high - range_low
                if (abs(high[i] - range_high) < rng_w * 0.2
                        and vol_ma20[i] > 0
                        and volume[i] < vol_ma20[i] * 0.6):
                    labels[i, 12] = 1.0

    # "none" where no events active
    has_event = labels[:, 1:].max(axis=1) > 0
    labels[~has_event, 0] = 1.0

    return labels


def _compute_excursion(
    close: np.ndarray, horizon: int
) -> np.ndarray:
    """
    Compute forward-looking favorable/adverse excursion.

    For each bar, look ahead `horizon` bars and compute:
        favorable = max price move in favorable direction / ATR
        adverse = max price move in adverse direction / ATR

    Since we don't know the trade direction, we compute both up/down
    and let the head learn the directional component.

    Returns (n_bars, 2) float32 — [max_up, max_down] normalized.
    """
    n = len(close)
    excursion = np.zeros((n, 2), dtype=np.float32)

    # Local ATR estimate (20-bar rolling range)
    for i in range(n):
        end = min(i + horizon, n)
        if end <= i + 1:
            continue
        future = close[i + 1:end]
        max_up = (future.max() - close[i])
        max_down = (close[i] - future.min())

        # Normalize by local volatility (20-bar std) to make scale-free
        local_start = max(0, i - 20)
        local_std = close[local_start:i + 1].std()
        if local_std > 1e-6:
            excursion[i, 0] = max_up / local_std
            excursion[i, 1] = max_down / local_std

    return excursion


def save_labels(labels: dict, output_path: str):
    """Save generated labels to NPZ."""
    np.savez_compressed(
        output_path,
        phase=labels['phase'],
        events=labels['events'],
        excursion=labels['excursion'],
    )


def load_labels(label_path: str) -> dict:
    """Load pre-computed labels from NPZ."""
    data = np.load(label_path, allow_pickle=True)
    return {
        'phase': data['phase'],
        'events': data['events'],
        'excursion': data['excursion'],
    }
