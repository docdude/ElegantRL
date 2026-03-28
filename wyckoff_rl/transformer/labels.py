"""
Structural labeling pipeline for supervised pre-training of transformer heads.

Derives regime and event labels from RAW price/volume structure, NOT from
pre-computed feature scores. This avoids the circular problem where the
transformer would just learn to replicate the feature pipeline's output.

v1 simplified ontology:
  - regime (n_bars,) int64: 0=balance, 1=uptrend, 2=downtrend
  - events (n_bars, 4) float32: spring_like, upthrust_like, absorption_like, exhaustion_like

Data sources:
  - Parquet file: raw OHLCV + delta + CVD per range bar
  - Wave segmentation: _segment_waves_nolag() on raw bars
  - Close prices: forward excursion (optional, not in v1)

Usage:
    labels = generate_structural_labels(
        parquet_path='pipeline_output/wyckoff_us30_100pt_bars.parquet'
    )
"""

import numpy as np
import pandas as pd
from .config import N_REGIMES, N_EVENTS


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
    Generate simplified structural labels (v1) from raw OHLCV data.

    v1 ontology:
      - regime (n_bars,) int64: 0=balance, 1=uptrend, 2=downtrend
      - events (n_bars, 4) float32: [spring_like, upthrust_like, absorption_like, exhaustion_like]

    Parameters
    ----------
    parquet_path : str
        Path to parquet with raw OHLCV + delta + CVD bars.
    npz_path : str, optional
        If provided, verifies alignment (same close prices).
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
    dict with 'phase' (regime labels), 'events'
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

    # Regime labels (3-class: balance/uptrend/downtrend)
    regime_labels = _label_regime(df, waves, ranges)

    # Event labels (4 binary channels)
    event_labels = _label_events_v1(
        df, waves, ranges,
        climax_vol_sigma=climax_vol_sigma,
        spring_reclaim_bars=spring_reclaim_bars,
    )

    return {
        'phase': regime_labels,   # key kept as 'phase' for checkpoint compat
        'events': event_labels,
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
    df: pd.DataFrame, lookback: int = 60, min_tests: int = 4,
    touch_pct: float = 0.08,
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
        # Reject ranges wider than 2% of price (real consolidation is tight)
        if range_width / close[i] > 0.02:
            continue

        # Reject if price has moved directionally across most of the range
        # (trending through a window is not consolidation)
        net_move = abs(close[i] - close[max(0, i - lookback)])
        if net_move > range_width * 0.6:
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

        if s_clusters >= 3 and r_clusters >= 3:
            ranges[i] = [range_low, range_high, support_tests, resistance_tests]

    return ranges


def _count_clusters(mask: np.ndarray) -> int:
    """Count contiguous groups of True in a boolean array."""
    if not mask.any():
        return 0
    # Each transition from False→True starts a new cluster
    return 1 + np.sum(mask[1:] & ~mask[:-1])


# ═══════════════════════════════════════════════════════════════════════
# Regime Labels (v1: balance / uptrend / downtrend)
# ═══════════════════════════════════════════════════════════════════════

def _label_regime(
    df: pd.DataFrame, waves: dict, ranges: np.ndarray,
    trend_window: int = 5,
) -> np.ndarray:
    """
    Classify market regime from swing progression.

    0 = balance  (range-bound, unclear, accumulation, distribution)
    1 = uptrend  (HH + HL dominant)
    2 = downtrend (LH + LL dominant)
    """
    n = len(df)
    labels = np.zeros(n, dtype=np.int64)  # default: balance

    wave_id = waves['wave_id']
    wave_dir = waves['wave_dir']
    wave_high = waves['wave_high']
    wave_low = waves['wave_low']

    sh_prices, sl_prices, sh_bars, sl_bars = _extract_pivots(
        wave_id, wave_dir, wave_high, wave_low)

    if len(sh_prices) < trend_window or len(sl_prices) < trend_window:
        return labels

    sh_idx = 0
    sl_idx = 0

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

        # Uptrend: strongly trending HH + HL (>= 75%)
        if hh >= tw * 0.75 and hl >= tw * 0.75:
            labels[i] = 1

        # Downtrend: strongly trending LH + LL (>= 75%)
        elif ll >= tw * 0.75 and lh >= tw * 0.75:
            labels[i] = 2

        # Everything else stays 0 (balance)

    return labels


def _extract_pivots(wave_id, wave_dir, wave_high, wave_low):
    """
    Extract alternating swing highs and swing lows from wave segments.

    Filters out trivial waves (< 3 bars) to avoid noisy zigzag pivots.
    Returns separate swing high and swing low sequences with their bar indices.
    """
    n = len(wave_id)
    raw_waves = []
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

    min_wave_bars = 3
    significant = []
    for w in raw_waves:
        if w[4] >= min_wave_bars:
            significant.append(w)
        elif significant:
            prev = significant[-1]
            significant[-1] = (w[0], prev[1], max(prev[2], w[2]),
                               min(prev[3], w[3]), prev[4] + w[4])

    swing_highs = []
    swing_lows = []
    for bar_idx, is_up, wh, wl, wlen in significant:
        if is_up:
            swing_highs.append((bar_idx, wh))
        else:
            swing_lows.append((bar_idx, wl))

    sh_bars = np.array([s[0] for s in swing_highs], dtype=np.int64) if swing_highs else np.array([], dtype=np.int64)
    sh_prices = np.array([s[1] for s in swing_highs]) if swing_highs else np.array([])
    sl_bars = np.array([s[0] for s in swing_lows], dtype=np.int64) if swing_lows else np.array([], dtype=np.int64)
    sl_prices = np.array([s[1] for s in swing_lows]) if swing_lows else np.array([])

    return sh_prices, sl_prices, sh_bars, sl_bars


# ═══════════════════════════════════════════════════════════════════════
# Event Labels v1 (4 channels: spring_like, upthrust_like, absorption_like, exhaustion_like)
# ═══════════════════════════════════════════════════════════════════════

def _label_events_v1(
    df: pd.DataFrame, waves: dict,
    ranges: np.ndarray,
    climax_vol_sigma: float = 2.0,
    spring_reclaim_bars: int = 5,
) -> np.ndarray:
    """
    Detect simplified Wyckoff events from raw price/volume structure.

    Channel 0 (spring_like): Break below support + reclaim (spring, LPS, test of support)
    Channel 1 (upthrust_like): Break above resistance + fall back (UT, LPSY, test of resistance)
    Channel 2 (absorption_like): High effort, little result — effort > result, no demand/supply
    Channel 3 (exhaustion_like): Climax volume + reversal — SC/BC terminal moves
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

    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(20, min_periods=5).mean().values
    vol_std20 = vol_series.rolling(20, min_periods=5).std().values

    spread_all = high - low
    spread_ma20 = pd.Series(spread_all).rolling(20, min_periods=5).mean().values

    # Duration stats for range bars (exhaustion = fast bar completion)
    has_duration = 'duration_seconds' in df.columns
    if has_duration:
        duration = df['duration_seconds'].values.astype(np.float64)
        dur_ma20 = pd.Series(duration).rolling(20, min_periods=5).mean().values
    else:
        duration = None
        dur_ma20 = None

    completed_up_vols = []
    completed_down_vols = []
    prev_wave_id = -1

    spring_bars = []
    upthrust_bars = []

    for i in range(1, n):
        # Wave completion tracking
        if wave_id[i] != prev_wave_id and prev_wave_id >= 0:
            if wave_dir[i - 1] > 0:
                completed_up_vols.append(wave_vol[i - 1])
            else:
                completed_down_vols.append(wave_vol[i - 1])
        prev_wave_id = wave_id[i]

        range_low = ranges[i, 0]
        range_high = ranges[i, 1]
        has_range = ranges[i, 2] > 0

        # ── Channel 0: spring_like ──
        # Spring: break below support + reclaim (same bar)
        if has_range and range_low > 0:
            if low[i] < range_low and close[i] > range_low:
                labels[i, 0] = 1.0
                spring_bars.append(i)
            elif i >= spring_reclaim_bars:
                # Reclaim within a few bars (weaker signal)
                for j in range(1, min(spring_reclaim_bars + 1, i)):
                    if low[i - j] < range_low and close[i] > range_low:
                        labels[i, 0] = 0.8
                        if not spring_bars or spring_bars[-1] != i:
                            spring_bars.append(i)
                        break

        # LPS: spring within 15 bars + test near support on low volume
        # (tighter window — real LPS happens soon after spring)
        if has_range and spring_bars:
            if 0 < (i - spring_bars[-1]) <= 15:
                rng_w = range_high - range_low
                if (rng_w > 0 and abs(low[i] - range_low) < rng_w * 0.10
                        and vol_ma20[i] > 0
                        and volume[i] < vol_ma20[i] * 0.5):
                    labels[i, 0] = max(labels[i, 0], 0.7)

        # ── Channel 1: upthrust_like ──
        # Upthrust: break above resistance + fall back (same bar)
        if has_range and range_high > 0:
            if high[i] > range_high and close[i] < range_high:
                labels[i, 1] = 1.0
                upthrust_bars.append(i)
            elif i >= spring_reclaim_bars:
                # Fail back within a few bars (weaker signal)
                for j in range(1, min(spring_reclaim_bars + 1, i)):
                    if high[i - j] > range_high and close[i] < range_high:
                        labels[i, 1] = 0.8
                        if not upthrust_bars or upthrust_bars[-1] != i:
                            upthrust_bars.append(i)
                        break

        # LPSY: upthrust within 15 bars + test near resistance on low volume
        if has_range and upthrust_bars:
            if 0 < (i - upthrust_bars[-1]) <= 15:
                rng_w = range_high - range_low
                if (rng_w > 0 and abs(high[i] - range_high) < rng_w * 0.10
                        and vol_ma20[i] > 0
                        and volume[i] < vol_ma20[i] * 0.5):
                    labels[i, 1] = max(labels[i, 1], 0.7)

        # ── Channel 2: absorption_like ──
        # Effort > Result: high volume + small body (price didn't move)
        # Both conditions required: significant volume AND weak price response
        if vol_ma20[i] > 0 and vol_std20[i] > 0:
            spread_i = high[i] - low[i]
            body_i = abs(close[i] - open_[i])

            # Volume must be genuinely elevated (> 1σ above mean)
            high_effort = volume[i] > vol_ma20[i] + vol_std20[i]
            # Body must be small relative to spread (indecision)
            weak_result = body_i < 0.35 * spread_i if spread_i > 0 else False
            # Spread must be meaningful (not a tiny doji on no activity)
            meaningful_bar = spread_i >= 0.8 * spread_ma20[i] if spread_ma20[i] > 0 else True

            if high_effort and weak_result and meaningful_bar:
                labels[i, 2] = 1.0

        # ── Channel 3: exhaustion_like ──
        # Climax: extreme volume + directional body + fast completion
        # On range bars, spread is capped by construction, so "wide bar" is
        # nearly impossible. Instead, exhaustion = the bar fills its fixed range
        # FAST with high volume and strong directional close.
        # No future-bar lookahead.
        if vol_ma20[i] > 0 and vol_std20[i] > 0:
            climax_vol = volume[i] > vol_ma20[i] + climax_vol_sigma * vol_std20[i]
            spread_i = high[i] - low[i]
            body_i = abs(close[i] - open_[i])

            # Directional body: close-open covers >50% of the bar range
            big_body = body_i > 0.5 * spread_i if spread_i > 0 else False

            # Fast completion: bar filled in less than half the recent average time
            fast_bar = (has_duration and dur_ma20 is not None
                        and dur_ma20[i] > 0
                        and duration[i] < dur_ma20[i] * 0.5)

            if climax_vol and big_body and fast_bar:
                labels[i, 3] = 1.0

    return labels


# ═══════════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════════

def save_labels(labels: dict, output_path: str):
    """Save generated labels to NPZ."""
    np.savez_compressed(
        output_path,
        phase=labels['phase'],
        events=labels['events'],
    )


def load_labels(label_path: str) -> dict:
    """Load pre-computed labels from NPZ."""
    data = np.load(label_path, allow_pickle=True)
    return {
        'phase': data['phase'],
        'events': data['events'],
    }
