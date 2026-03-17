"""
Weis Wave utility functions — ZigZag logic and wave comparison.

Mirrors the structure of srlcarlg's models_utils/ww_utils.py.
Contains the core wave segmentation algorithm and wave-to-wave
comparison logic (yellow bar detection, directional comparisons).
"""

import numpy as np
import pandas as pd

from .ww_models import (Direction, ZigZagMode, ZigZagInit, WavesInit,
                        YellowWaves, FilterType, FilterRatio, StrengthFilter)
from .custom_mas import get_ma, get_stddev, rolling_percentile, l1norm


# ─── ZigZag wave segmentation ─────────────────────────────────────────────

def segment_waves(df, reversal_pct=0.5, zigzag_init=None, waves_init=None):
    """
    Segment price into up-waves and down-waves (Weis Wave style).

    A wave reverses when price retraces by >= reversal_pct% from the
    current wave extreme (ZigZagMode.Percentage).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with at least 'close', 'high', 'low', 'volume', 'delta'.
    reversal_pct : float
        Reversal threshold as percentage (default 0.5 = 0.5%).
    zigzag_init : ZigZagInit, optional
        ZigZag configuration. If None, uses Percentage mode with reversal_pct.
    waves_init : WavesInit, optional
        Waves configuration. If None, uses defaults.

    Returns
    -------
    df with added columns:
        WaveDir        :  +1 (up wave) / −1 (down wave)
        WaveID         :  integer wave counter
        WaveVolume     :  cumulative volume within the current wave
        WaveDelta      :  cumulative delta within the current wave
        WaveHigh       :  highest price in the current wave
        WaveLow        :  lowest price in the current wave
        WavePriceDisp  :  price displacement of the current wave
        Trendline      :  ZigZag trendline connecting wave extremes
        TurningPoint   :  price at wave reversal pivots
        WaveTime       :  duration of current wave in seconds
    """
    n = len(df)
    if n == 0:
        return df

    # Initialize configs
    if zigzag_init is None:
        zigzag_init = ZigZagInit(ZigZagMode.Percentage, pct_value=reversal_pct)
    else:
        zigzag_init.reset()
    if waves_init is None:
        waves_init = WavesInit()
    else:
        waves_init.reset()

    close = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    volume = df['volume'].values
    delta = df['delta'].values
    has_datetime = hasattr(df.index, 'dtype') and np.issubdtype(df.index.dtype, np.datetime64)

    # Output arrays
    wave_dir = np.zeros(n, dtype=np.int32)
    wave_id = np.zeros(n, dtype=np.int32)
    wave_vol = np.zeros(n, dtype=np.int64)
    wave_delta = np.zeros(n, dtype=np.int64)
    wave_high = np.zeros(n, dtype=np.float64)
    wave_low = np.zeros(n, dtype=np.float64)
    trendline = np.full(n, np.nan)
    turning_point = np.full(n, np.nan)

    threshold = zigzag_init.pct_value / 100.0

    # Initialize first bar
    current_dir = 1  # assume up
    current_wave = 0
    w_high = close[0]
    w_low = close[0]
    w_vol = volume[0]
    w_delt = delta[0]
    extremum_idx = 0
    extremum_price = w_high

    wave_dir[0] = current_dir
    wave_id[0] = current_wave
    wave_vol[0] = w_vol
    wave_delta[0] = w_delt
    wave_high[0] = w_high
    wave_low[0] = w_low
    trendline[0] = close[0]

    wave_start_idx = 0

    for i in range(1, n):
        price = close[i]

        if current_dir == 1:  # up wave
            if price > w_high:
                w_high = price
                # Move extremum
                trendline[extremum_idx] = np.nan
                extremum_idx = i
                extremum_price = w_high
                trendline[extremum_idx] = extremum_price
            # Check reversal
            if w_high > 0 and _zigzag_reversed(zigzag_init, w_high, price, Direction.UP):
                # Lock the extremum as turning point
                turning_point[extremum_idx] = high_arr[extremum_idx]
                # New down wave
                current_dir = -1
                current_wave += 1
                wave_start_idx = i
                w_high = price
                w_low = price
                w_vol = volume[i]
                w_delt = delta[i]
                extremum_idx = i
                extremum_price = w_low
                trendline[extremum_idx] = extremum_price
            else:
                w_vol += volume[i]
                w_delt += delta[i]
        else:  # down wave
            if price < w_low:
                w_low = price
                trendline[extremum_idx] = np.nan
                extremum_idx = i
                extremum_price = w_low
                trendline[extremum_idx] = extremum_price
            # Check reversal
            if w_low > 0 and _zigzag_reversed(zigzag_init, w_low, price, Direction.DOWN):
                turning_point[extremum_idx] = low_arr[extremum_idx]
                # New up wave
                current_dir = 1
                current_wave += 1
                wave_start_idx = i
                w_high = price
                w_low = price
                w_vol = volume[i]
                w_delt = delta[i]
                extremum_idx = i
                extremum_price = w_high
                trendline[extremum_idx] = extremum_price
            else:
                w_vol += volume[i]
                w_delt += delta[i]

        wave_dir[i] = current_dir
        wave_id[i] = current_wave
        wave_vol[i] = w_vol
        wave_delta[i] = w_delt
        wave_high[i] = w_high
        wave_low[i] = w_low

    df['WaveDir'] = wave_dir
    df['WaveID'] = wave_id
    df['WaveVolume'] = wave_vol
    df['WaveDelta'] = wave_delta
    df['WaveHigh'] = wave_high
    df['WaveLow'] = wave_low
    df['WavePriceDisp'] = np.where(
        wave_dir == 1,
        wave_high - wave_low,
        wave_low - wave_high
    )
    df['Trendline'] = trendline
    df['TurningPoint'] = turning_point

    # ─── Wave time tracking ───
    df['WaveTime'] = 0.0
    if has_datetime:
        timestamps = df.index.values.astype('datetime64[s]').astype(np.float64)
        for wid in range(current_wave + 1):
            mask = wave_id == wid
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                wave_dur = timestamps[idxs[-1]] - timestamps[idxs[0]]
                df.iloc[idxs, df.columns.get_loc('WaveTime')] = wave_dur

    # ─── Wave-to-wave comparison (Weis yellow bars) ───
    compute_wave_comparison(df, wave_id, wave_dir, wave_vol, current_wave,
                            waves_init=waves_init)

    return df


# ─── ZigZag reversal check ────────────────────────────────────────────────

def _zigzag_reversed(zz, extremum, price, direction):
    """Check if price has reversed from extremum by enough to trigger a new wave."""
    if zz.mode == ZigZagMode.Percentage:
        if direction == Direction.UP:
            return (extremum - price) / extremum >= (zz.pct_value / 100.0)
        else:
            return (price - extremum) / extremum >= (zz.pct_value / 100.0)
    elif zz.mode == ZigZagMode.Points:
        return abs(price - extremum) >= zz.points_value
    return False


# ─── Wave-to-wave comparison ──────────────────────────────────────────────

def compute_wave_comparison(df, wave_id, wave_dir, wave_vol, max_wave,
                            waves_init=None, large_ratio=None):
    """
    Compare each wave's volume and effort/result against prior 4 waves.
    Flags anomalously large waves (Weis "yellow bars").

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'WavePriceDisp' column already set.
    wave_id, wave_dir, wave_vol : np.ndarray
        Wave arrays from segment_waves().
    max_wave : int
        Maximum wave ID.
    waves_init : WavesInit, optional
        Configuration for yellow wave detection.
    large_ratio : float, optional
        Override for large_wave_ratio (default from waves_init or 1.5).

    Columns added:
        WaveEndVol    : final volume for each completed wave (NaN mid-wave)
        WaveEndER     : final effort/result for each completed wave
        LargeWave     : 1 if wave vol > avg(prior 4) × large_ratio
        LargeER       : 1 if E/R > avg(prior 4) × large_ratio
        WaveVsSame    : +1/-1 vs prior same-direction wave volume
        WaveVsPrev    : +1/-1 vs prior opposite-direction wave volume
        ERVsSame      : +1/-1 vs prior same-direction E/R
        ERVsPrev      : +1/-1 vs prior opposite-direction E/R
    """
    if waves_init is None:
        waves_init = WavesInit()
    if large_ratio is None:
        large_ratio = waves_init.large_wave_ratio

    n = len(df)
    wave_end_vol = np.full(n, np.nan)
    wave_end_er = np.full(n, np.nan)
    large_wave = np.zeros(n, dtype=np.int32)
    large_er = np.zeros(n, dtype=np.int32)
    wave_vs_same = np.full(n, np.nan)
    wave_vs_prev = np.full(n, np.nan)
    er_vs_same = np.full(n, np.nan)
    er_vs_prev = np.full(n, np.nan)

    # Collect per-wave final stats
    wave_stats = []  # list of (direction, volume, er, last_bar_idx)
    for wid in range(max_wave + 1):
        mask = wave_id == wid
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        last_idx = idxs[-1]
        w_vol = wave_vol[last_idx]
        w_dir = wave_dir[last_idx]
        w_price_disp = abs(df['WavePriceDisp'].iloc[last_idx])
        w_er = w_vol / w_price_disp if w_price_disp > 0 else 0.0
        wave_stats.append((w_dir, w_vol, w_er, last_idx))

    # Sliding windows
    prev_vols = list(waves_init.prev_waves_volume)
    prev_ers = list(waves_init.prev_waves_er)
    prev_up = waves_init.prev_wave_up
    prev_down = waves_init.prev_wave_down

    for stat_idx, (w_dir, w_vol, w_er, last_idx) in enumerate(wave_stats):
        wave_end_vol[last_idx] = w_vol
        wave_end_er[last_idx] = round(w_er, 1)

        # Large wave detection: compare vs average of prior 4
        if not any(v == 0.0 for v in prev_vols):
            avg_vol = sum(prev_vols) / 4.0
            if w_vol > avg_vol * large_ratio:
                large_wave[last_idx] = 1
        if not any(v == 0.0 for v in prev_ers):
            avg_er = sum(prev_ers) / 4.0
            if w_er > avg_er * large_ratio:
                large_er[last_idx] = 1

        # Wave vs same direction / vs previous direction
        if w_dir == 1:
            wave_vs_same[last_idx] = 1 if w_vol > prev_up[0] else -1
            wave_vs_prev[last_idx] = 1 if w_vol > prev_down[0] else -1
            er_vs_same[last_idx] = 1 if w_er > prev_up[1] else -1
            er_vs_prev[last_idx] = 1 if w_er > prev_down[1] else -1
            prev_up = (float(w_vol), w_er)
        else:
            wave_vs_same[last_idx] = 1 if w_vol > prev_down[0] else -1
            wave_vs_prev[last_idx] = 1 if w_vol > prev_up[0] else -1
            er_vs_same[last_idx] = 1 if w_er > prev_down[1] else -1
            er_vs_prev[last_idx] = 1 if w_er > prev_up[1] else -1
            prev_down = (float(w_vol), w_er)

        # Shift sliding window (push value based on yellow_waves config)
        _push_wave = w_vol
        _push_er = w_er
        if waves_init.yellow_waves == YellowWaves.UsePrev_SameWave:
            if w_dir == 1:
                _push_wave = prev_up[0]
                _push_er = prev_up[1]
            else:
                _push_wave = prev_down[0]
                _push_er = prev_down[1]
        elif waves_init.yellow_waves == YellowWaves.UsePrev_InvertWave:
            if w_dir == 1:
                _push_wave = prev_down[0]
                _push_er = prev_down[1]
            else:
                _push_wave = prev_up[0]
                _push_er = prev_up[1]

        prev_vols = [prev_vols[1], prev_vols[2], prev_vols[3], float(_push_wave)]
        prev_ers = [prev_ers[1], prev_ers[2], prev_ers[3], _push_er]

    # Save state back for potential continuation
    waves_init.prev_waves_volume = prev_vols
    waves_init.prev_waves_er = prev_ers
    waves_init.prev_wave_up = prev_up
    waves_init.prev_wave_down = prev_down

    df['WaveEndVol'] = wave_end_vol
    df['WaveEndER'] = wave_end_er
    df['LargeWave'] = large_wave
    df['LargeER'] = large_er
    df['WaveVsSame'] = wave_vs_same
    df['WaveVsPrev'] = wave_vs_prev
    df['ERVsSame'] = er_vs_same
    df['ERVsPrev'] = er_vs_prev


# ─── Bar-level Wyckoff strength analysis ──────────────────────────────────

def compute_bar_strength(df, strength_filter=None):
    """
    Classify each bar's volume and time into 5-tier strength scores.

    Mirrors srlcarlg's ``wyckoff_analysis()`` method.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'volume' column and a DatetimeIndex.
    strength_filter : StrengthFilter, optional
        Configuration.  Uses Normalized_Emphasized with Percentage defaults
        if None.

    Columns added:
        BarTime          : bar duration as seconds (float)
        VolumeFilter     : normalised volume metric
        TimeFilter       : normalised time metric
        VolumeStrength   : 0 (lowest) → 4 (ultra)
        TimeStrength     : 0 (lowest) → 4 (ultra)
    """
    if strength_filter is None:
        strength_filter = StrengthFilter()
    f = strength_filter

    n = len(df)
    if n < 2:
        for col in ('BarTime', 'VolumeFilter', 'TimeFilter',
                     'VolumeStrength', 'TimeStrength'):
            df[col] = 0
        return df

    # ── Bar time (seconds) ──
    if hasattr(df.index, 'dtype') and np.issubdtype(df.index.dtype, np.datetime64):
        ts = df.index.values.astype('datetime64[ms]').astype(np.float64)
        bar_time_ms = np.full(n, np.nan, dtype=np.float64)
        if f.is_open_time:
            bar_time_ms[:-1] = np.diff(ts)
            bar_time_ms[-1] = bar_time_ms[-2] if n > 1 else 0
        else:
            bar_time_ms[1:] = np.diff(ts)
            bar_time_ms[0] = bar_time_ms[1] if n > 1 else 0
    else:
        bar_time_ms = np.ones(n, dtype=np.float64)

    df['BarTime'] = bar_time_ms / 1000.0  # seconds

    # ── Compute filters ──
    vol = df['volume'].values.astype(np.float64)
    time_ms = bar_time_ms.copy()

    if f.filter_type in (FilterType.MA, FilterType.StdDev, FilterType.Both):
        vol_ma = get_ma(vol, f.ma_type, f.ma_period)
        time_ma = get_ma(time_ms, f.ma_type, f.ma_period)
        vol_filter = vol / np.where(vol_ma == 0, 1, vol_ma)
        time_filter = time_ms / np.where(time_ma == 0, 1, time_ma)

        if f.filter_type in (FilterType.StdDev, FilterType.Both):
            vol_sd = get_stddev(vol, vol_ma, f.ma_period)
            time_sd = get_stddev(time_ms, time_ma, f.ma_period)
            if f.filter_type == FilterType.StdDev:
                vol_filter = vol / np.where(vol_sd == 0, 1, vol_sd)
                time_filter = time_ms / np.where(time_sd == 0, 1, time_sd)
            else:  # Both => z-score
                vol_filter = (vol - vol_ma) / np.where(vol_sd == 0, 1, vol_sd)
                time_filter = (time_ms - time_ma) / np.where(time_sd == 0, 1, time_sd)

    elif f.filter_type == FilterType.Normalized_Emphasized:
        vol_avg = pd.Series(vol).rolling(f.n_period, min_periods=1).mean().values
        vol_norm = vol / np.where(vol_avg == 0, 1, vol_avg)
        vol_pct = (vol_norm * 100) - 100
        vol_filter = vol_pct * f.n_multiplier

        time_avg = pd.Series(time_ms).rolling(f.n_period, min_periods=1).mean().values
        time_norm = time_ms / np.where(time_avg == 0, 1, time_avg)
        time_pct = (time_norm * 100) - 100
        time_filter = time_pct * f.n_multiplier

    elif f.filter_type == FilterType.L1Norm:
        vol_filter = pd.Series(vol).rolling(f.ma_period, min_periods=1).apply(
            l1norm, raw=True).values
        time_filter = pd.Series(time_ms).rolling(f.ma_period, min_periods=1).apply(
            l1norm, raw=True).values
    else:
        vol_filter = np.ones(n)
        time_filter = np.ones(n)

    vol_filter = np.abs(vol_filter)
    time_filter = np.abs(time_filter)
    vol_filter = np.round(vol_filter, 2)
    time_filter = np.round(time_filter, 2)

    # Percentile transform if requested
    if f.filter_ratio == FilterRatio.Percentage and \
       f.filter_type != FilterType.Normalized_Emphasized:
        vol_filter = pd.Series(vol_filter).rolling(
            f.n_period, min_periods=1).apply(rolling_percentile, raw=True).values
        vol_filter = np.round(vol_filter, 1)
        time_filter = pd.Series(time_filter).rolling(
            f.n_period, min_periods=1).apply(rolling_percentile, raw=True).values
        time_filter = np.round(time_filter, 1)

    df['VolumeFilter'] = vol_filter
    df['TimeFilter'] = time_filter

    # ── Classify into 5 strength tiers ──
    df['VolumeStrength'] = _classify_strength(vol_filter, f)
    df['TimeStrength'] = _classify_strength(time_filter, f)

    return df


def _classify_strength(values, f):
    """Map filter values → 0–4 strength tier."""
    if f.filter_type == FilterType.Normalized_Emphasized:
        return np.where(values < f.lowest_pct, 0,
               np.where(values < f.low_pct, 1,
               np.where(values < f.average_pct, 2,
               np.where(values < f.high_pct, 3,
               np.where(values >= f.ultra_pct, 4, 4)))))
    elif f.filter_ratio == FilterRatio.Percentage:
        return np.where(values < f.lowest_pctile, 0,
               np.where(values < f.low_pctile, 1,
               np.where(values < f.average_pctile, 2,
               np.where(values < f.high_pctile, 3,
               np.where(values >= f.ultra_pctile, 4, 4)))))
    else:  # Fixed
        return np.where(values < f.lowest, 0,
               np.where(values < f.low, 1,
               np.where(values < f.average, 2,
               np.where(values < f.high, 3,
               np.where(values >= f.ultra, 4, 4)))))
