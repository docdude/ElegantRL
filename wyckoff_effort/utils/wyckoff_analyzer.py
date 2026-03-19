"""
Wyckoff Effort-vs-Result Analyzer — inspired by Weis Wave and DeepCharts Deep-M.

Computes from tick-level Bid/Ask data (SCID or Alpaca):
  1. Volume Delta  (aggressive buy − aggressive sell)
  2. Cumulative Volume Delta (CVD)
  3. Weis-style wave segmentation
  4. Per-wave effort vs result scoring
  5. Absorption detection
  6. Spring / Upthrust / Selling Climax / Buying Climax events
  7. Supply–demand phase classification

All outputs are numeric columns suitable for feeding into an RL state.
"""

import numpy as np
import pandas as pd
import logging

from .models_utils.ww_models import ZigZagInit, ZigZagMode, WavesInit, StrengthFilter
from .models_utils.ww_utils import segment_waves, compute_wave_comparison, compute_bar_strength

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  1.  DELTA & CVD (already in SCID data, but also for Alpaca-estimated data)
# ---------------------------------------------------------------------------

def compute_delta_from_ohlcv(df):
    """
    Estimate delta when only OHLCV is available (no bid/ask split).
    Uses bar internals:  delta ≈ volume * (2 * (close − low) / (high − low) − 1)
    """
    hl_range = (df['high'] - df['low']).replace(0, np.nan)
    ratio = 2.0 * (df['close'] - df['low']) / hl_range - 1.0
    df['delta'] = (df['volume'] * ratio).fillna(0).astype(np.int64)
    df['cvd'] = df['delta'].cumsum()
    return df


def ensure_delta(df):
    """Make sure delta and cvd columns exist — use real if available, else estimate."""
    if 'delta' not in df.columns:
        if 'ask_volume' in df.columns and 'bid_volume' in df.columns:
            df['delta'] = df['ask_volume'].astype(np.int64) - df['bid_volume'].astype(np.int64)
        else:
            hl_range = (df['high'] - df['low']).replace(0, np.nan)
            ratio = 2.0 * (df['close'] - df['low']) / hl_range - 1.0
            df['delta'] = (df['volume'] * ratio).fillna(0).astype(np.int64)
    if 'cvd' not in df.columns:
        df['cvd'] = df['delta'].cumsum()
    return df


# ---------------------------------------------------------------------------
#  2.  WEIS WAVE SEGMENTATION  (delegated to models_utils.ww_utils)
# ---------------------------------------------------------------------------

# segment_waves and compute_wave_comparison are imported from
# utils.models_utils.ww_utils — see imports at top of file.
#
# Re-export for backward compatibility:
#   from utils.wyckoff_analyzer import segment_waves  # still works


# ---------------------------------------------------------------------------
#  3.  EFFORT vs RESULT  (per bar and per wave)
# ---------------------------------------------------------------------------

def compute_effort_result(df, lookback=20):
    """
    Compute effort-vs-result metrics.

    Bar-level:
        EffortResult  = |price_change| / volume  (result per unit effort)
                        High value = ease of movement  (low effort, high result)
                        Low value  = absorption         (high effort, low result)

    Normalized:
        ER_Ratio      = current EffortResult / rolling mean EffortResult
                        < 0.5 → absorption zone  (high effort, low result)
                        > 2.0 → ease of movement (low effort, high result)

    Wave-level (computed per WaveID):
        WaveER        = wave price displacement / wave volume
    """
    volume = df['volume'].replace(0, np.nan)

    price_change = df['close'].diff().abs()
    df['EffortResult'] = (price_change / volume).fillna(0)

    rolling_mean = df['EffortResult'].rolling(window=lookback, min_periods=1).mean()
    df['ER_Ratio'] = (df['EffortResult'] / rolling_mean.replace(0, np.nan)).fillna(1.0)

    # Wave-level effort vs result
    if 'WaveID' in df.columns and 'WavePriceDisp' in df.columns:
        wave_vol = df['WaveVolume'].replace(0, np.nan)
        df['WaveER'] = (df['WavePriceDisp'].abs() / wave_vol).fillna(0)
    else:
        df['WaveER'] = 0.0

    return df


# ---------------------------------------------------------------------------
#  4.  ABSORPTION DETECTION
# ---------------------------------------------------------------------------

def detect_absorption(df, vol_threshold=1.5, er_threshold=0.5, lookback=20):
    """
    Detect absorption: high volume + small price change = institutional
    accumulation or distribution.

    Signals:
        Absorption     : 1 = detected, 0 = not
        AbsorptionType : +1 = bullish absorption (buying), −1 = bearish (selling), 0 = none

    Logic:
        - Volume >= vol_threshold × rolling average volume
        - ER_Ratio <= er_threshold (high effort, low result)
        - Delta sign determines bullish vs bearish
    """
    avg_vol = df['volume'].rolling(window=lookback, min_periods=1).mean()
    vol_ratio = df['volume'] / avg_vol.replace(0, np.nan)

    high_volume = vol_ratio >= vol_threshold

    if 'ER_Ratio' not in df.columns:
        df = compute_effort_result(df, lookback)

    low_result = df['ER_Ratio'] <= er_threshold

    df['Absorption'] = (high_volume & low_result).astype(int)

    # Classify direction by delta
    df['AbsorptionType'] = 0
    bullish = df['Absorption'] == 1
    df.loc[bullish & (df['delta'] > 0), 'AbsorptionType'] = 1   # buying absorption
    df.loc[bullish & (df['delta'] < 0), 'AbsorptionType'] = -1  # selling absorption

    return df


# ---------------------------------------------------------------------------
#  5.  WYCKOFF EVENT DETECTION (Spring, Upthrust, SC, BC)
# ---------------------------------------------------------------------------

def detect_wyckoff_events(df, swing_lookback=20, vol_multiplier=2.0):
    """
    Detect classic Wyckoff events using delta-confirmed logic.

    Events detected:
        SellingClimax (SC)  : Sharp drop + extreme volume + positive delta divergence
        BuyingClimax  (BC)  : Sharp rise + extreme volume + negative delta divergence
        Spring              : Price breaks below recent swing low then recovers,
                              with delta turning positive (buyers step in)
        Upthrust (UT)       : Price breaks above recent swing high then reverses,
                              with delta turning negative (sellers step in)

    Returns df with binary columns for each event.
    """
    n = len(df)

    df['SellingClimax'] = 0
    df['BuyingClimax'] = 0
    df['Spring'] = 0
    df['Upthrust'] = 0

    if n < swing_lookback + 2:
        return df

    avg_vol = df['volume'].rolling(window=swing_lookback, min_periods=1).mean()
    swing_low = df['low'].rolling(window=swing_lookback, min_periods=1).min()
    swing_high = df['high'].rolling(window=swing_lookback, min_periods=1).max()

    close = df['close'].values
    low = df['low'].values
    high = df['high'].values
    opn = df['open'].values
    volume = df['volume'].values
    delta = df['delta'].values
    avg_v = avg_vol.values
    sw_low = swing_low.values
    sw_high = swing_high.values

    for i in range(swing_lookback + 1, n):
        vol_spike = volume[i] >= vol_multiplier * avg_v[i]
        bar_range = high[i] - low[i]
        if bar_range == 0:
            continue

        # --- Selling Climax ---
        # Large down bar on extreme volume, close near the low,
        # BUT delta is positive (buyers absorbing the selling)
        price_drop = close[i] < close[i - 1]
        close_near_low = (close[i] - low[i]) / bar_range < 0.3
        if vol_spike and price_drop and close_near_low and delta[i] > 0:
            df.iloc[i, df.columns.get_loc('SellingClimax')] = 1

        # --- Buying Climax ---
        # Large up bar on extreme volume, close near the high,
        # BUT delta is negative (sellers absorbing the buying)
        price_rise = close[i] > close[i - 1]
        close_near_high = (high[i] - close[i]) / bar_range < 0.3
        if vol_spike and price_rise and close_near_high and delta[i] < 0:
            df.iloc[i, df.columns.get_loc('BuyingClimax')] = 1

        # --- Spring ---
        # Low penetrates below recent swing low, then closes back above it
        # Delta positive on that bar or next (buyers defending)
        if low[i] < sw_low[i - 1] and close[i] > sw_low[i - 1] and delta[i] > 0:
            df.iloc[i, df.columns.get_loc('Spring')] = 1

        # --- Upthrust ---
        # High penetrates above recent swing high, then closes back below it
        # Delta negative on that bar (sellers defending)
        if high[i] > sw_high[i - 1] and close[i] < sw_high[i - 1] and delta[i] < 0:
            df.iloc[i, df.columns.get_loc('Upthrust')] = 1

    return df


# ---------------------------------------------------------------------------
#  6.  SUPPLY / DEMAND PHASE CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_phase(df, phase_lookback=50):
    """
    Classify each bar into a Wyckoff phase based on rolling metrics.

    Phases (encoded as integers for RL):
        0 = Unknown / Ranging
        1 = Accumulation  (price flat/declining + bullish absorption + CVD rising)
        2 = Markup         (price rising + positive delta + expanding waves)
        3 = Distribution   (price flat/rising + bearish absorption + CVD falling)
        4 = Markdown       (price falling + negative delta + expanding down waves)
    """
    n = len(df)
    df['Phase'] = 0

    if n < phase_lookback:
        return df

    close = df['close'].values
    cvd = df['cvd'].values
    absorption_type = df.get('AbsorptionType', pd.Series(np.zeros(n))).values

    price_slope = pd.Series(close).rolling(window=phase_lookback, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
    ).values

    cvd_slope = pd.Series(cvd).rolling(window=phase_lookback, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
    ).values

    # Normalize slopes relative to price level
    price_level = pd.Series(close).rolling(window=phase_lookback, min_periods=1).mean().values
    norm_price_slope = np.divide(price_slope, np.where(price_level != 0, price_level, 1))

    # Rolling absorption counts
    abs_window = min(phase_lookback, n)
    if 'AbsorptionType' in df.columns:
        bullish_abs = (df['AbsorptionType'] == 1).rolling(window=abs_window, min_periods=1).sum().values
        bearish_abs = (df['AbsorptionType'] == -1).rolling(window=abs_window, min_periods=1).sum().values
    else:
        bullish_abs = np.zeros(n)
        bearish_abs = np.zeros(n)

    for i in range(phase_lookback, n):
        ps = norm_price_slope[i]
        cs = cvd_slope[i]
        ba = bullish_abs[i]
        sa = bearish_abs[i]

        # Accumulation: price flat/down, CVD rising, bullish absorption
        if ps < 0.0002 and cs > 0 and ba > sa:
            df.iloc[i, df.columns.get_loc('Phase')] = 1

        # Markup: price clearly rising, CVD rising
        elif ps > 0.0005 and cs > 0:
            df.iloc[i, df.columns.get_loc('Phase')] = 2

        # Distribution: price flat/up, CVD falling, bearish absorption
        elif ps > -0.0002 and cs < 0 and sa > ba:
            df.iloc[i, df.columns.get_loc('Phase')] = 3

        # Markdown: price clearly falling, CVD falling
        elif ps < -0.0005 and cs < 0:
            df.iloc[i, df.columns.get_loc('Phase')] = 4

    return df


# ---------------------------------------------------------------------------
#  7.  RULE-BASED SIGNAL GENERATOR
# ---------------------------------------------------------------------------

def generate_signals(df, effort_score=None, cooldown=5, confirmation_bars=2):
    """
    Generate long/short/flat trade signals by combining Wyckoff events,
    effort zones, and phase classification.

    Signal logic (inspired by DeepCharts Deep-M):

      LONG ENTRY:
        - Trigger: Spring OR Selling Climax
        - Confirm: effort_score > 0 (green zone / ask dominance) for
          `confirmation_bars` of the last 3 bars
        - Context: Phase is Accumulation (1) or Unknown (0)
        - Exit: Upthrust, BuyingClimax, phase flips to Distribution (3),
                or effort turns purple for `confirmation_bars` consecutive bars

      SHORT ENTRY:
        - Trigger: Upthrust OR Buying Climax
        - Confirm: effort_score < 0 (purple zone / bid dominance) for
          `confirmation_bars` of the last 3 bars
        - Context: Phase is Distribution (3) or Unknown (0)
        - Exit: Spring, SellingClimax, phase flips to Accumulation (1),
                or effort turns green for `confirmation_bars` consecutive bars

    Args:
        df: DataFrame from analyze_wyckoff() — must have Spring, Upthrust,
            SellingClimax, BuyingClimax, Phase, Delta columns.
        effort_score: Optional numpy array from compute_effort_zones().
            If None, a basic effort proxy is computed from Delta + CVD.
        cooldown: Minimum bars between signal changes to avoid whipsaws.
        confirmation_bars: How many of the last 3 bars must confirm zone direction.

    Returns:
        df with new columns:
            Signal       :  +1 (long) / -1 (short) / 0 (flat)
            SignalEvent  :  string label for the triggering event (for plotting)
            EffortScore  :  the effort score used for zone classification
    """
    n = len(df)
    signal = np.zeros(n, dtype=np.int32)
    signal_event = [''] * n

    # Compute effort score if not provided
    if effort_score is None:
        effort_score = _compute_basic_effort(df)

    df['EffortScore'] = effort_score

    # Extract arrays for speed
    spring = df['Spring'].values if 'Spring' in df.columns else np.zeros(n)
    upthrust = df['Upthrust'].values if 'Upthrust' in df.columns else np.zeros(n)
    sc = df['SellingClimax'].values if 'SellingClimax' in df.columns else np.zeros(n)
    bc = df['BuyingClimax'].values if 'BuyingClimax' in df.columns else np.zeros(n)
    phase = df['Phase'].values if 'Phase' in df.columns else np.zeros(n)
    absorption = df['AbsorptionType'].values if 'AbsorptionType' in df.columns else np.zeros(n)

    current_signal = 0  # flat
    bars_since_change = 0

    for i in range(n):
        bars_since_change += 1

        # Count recent green/purple effort bars
        lookback_start = max(0, i - 2)  # last 3 bars including current
        recent_effort = effort_score[lookback_start:i + 1]
        green_count = np.sum(recent_effort > 0)
        purple_count = np.sum(recent_effort < 0)

        # ─── ENTRY LOGIC ───
        if current_signal == 0 and bars_since_change >= cooldown:
            # Long entry: Spring or SC + green effort confirmation + accumulation context
            if (spring[i] or sc[i]) and green_count >= confirmation_bars and phase[i] in (0, 1):
                current_signal = 1
                bars_since_change = 0
                signal_event[i] = 'Spring' if spring[i] else 'SC→Long'

            # Short entry: Upthrust or BC + purple effort confirmation + distribution context
            elif (upthrust[i] or bc[i]) and purple_count >= confirmation_bars and phase[i] in (0, 3):
                current_signal = -1
                bars_since_change = 0
                signal_event[i] = 'Upthrust' if upthrust[i] else 'BC→Short'

        # ─── EXIT / REVERSAL LOGIC ───
        elif current_signal == 1 and bars_since_change >= cooldown:
            # Exit long on: Upthrust, BC, phase→Distribution, or sustained purple effort
            if upthrust[i]:
                current_signal = -1  # reverse to short
                bars_since_change = 0
                signal_event[i] = 'UT→Short'
            elif bc[i]:
                current_signal = 0  # exit to flat (climax = exhaustion)
                bars_since_change = 0
                signal_event[i] = 'BC→Flat'
            elif phase[i] == 3 and purple_count >= confirmation_bars:
                current_signal = 0
                bars_since_change = 0
                signal_event[i] = 'Dist→Flat'
            elif purple_count >= 3:  # all 3 bars purple — effort exhausted
                current_signal = 0
                bars_since_change = 0
                signal_event[i] = 'Effort→Flat'

        elif current_signal == -1 and bars_since_change >= cooldown:
            # Exit short on: Spring, SC, phase→Accumulation, or sustained green effort
            if spring[i]:
                current_signal = 1  # reverse to long
                bars_since_change = 0
                signal_event[i] = 'Spr→Long'
            elif sc[i]:
                current_signal = 0  # exit to flat (climax = exhaustion)
                bars_since_change = 0
                signal_event[i] = 'SC→Flat'
            elif phase[i] == 1 and green_count >= confirmation_bars:
                current_signal = 0
                bars_since_change = 0
                signal_event[i] = 'Accum→Flat'
            elif green_count >= 3:  # all 3 bars green — effort exhausted
                current_signal = 0
                bars_since_change = 0
                signal_event[i] = 'Effort→Flat'

        signal[i] = current_signal

    df['Signal'] = signal
    df['SignalEvent'] = signal_event

    # Summary stats
    longs = np.sum(np.diff(signal) > 0)   # transitions to long
    shorts = np.sum(np.diff(signal) < 0)  # transitions to short
    logger.info(f"Signals: {longs} long entries, {shorts} short entries")

    return df


def _compute_basic_effort(df, lookback=14):
    """Basic effort score from delta + cvd when plotter zones aren't available."""
    delta = df['delta'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    cvd = np.cumsum(delta)
    cvd_series = pd.Series(cvd)
    cvd_fast = cvd_series.ewm(span=lookback, adjust=False).mean()
    cvd_slow = cvd_series.ewm(span=lookback * 3, adjust=False).mean()
    cvd_slope = (cvd_fast - cvd_slow).values

    avg_vol = pd.Series(volume).rolling(window=lookback, min_periods=1).mean().values
    delta_ratio = np.where(avg_vol > 0, delta / avg_vol, 0.0)

    scale = np.nanmean(np.abs(cvd_slope[lookback:])) + 1e-9
    cvd_norm = np.tanh(cvd_slope / scale)
    delta_norm = np.clip(delta_ratio, -1, 1)

    return 0.6 * cvd_norm + 0.4 * delta_norm


# ---------------------------------------------------------------------------
#  8.  MASTER ANALYSIS PIPELINE
# ---------------------------------------------------------------------------

def analyze_wyckoff(df, reversal_pct=0.5, er_lookback=20, vol_threshold=1.5,
                    er_threshold=0.5, swing_lookback=20, vol_multiplier=2.0,
                    phase_lookback=50, strength_filter=None):
    """
    Run the full Wyckoff effort-vs-result analysis pipeline.

    Input: DataFrame with at minimum OHLCV (or OHLCV + BidVolume/AskVolume).
    Output: Same DataFrame with all Wyckoff indicator columns added.

    New columns added:
        Delta, CVD,
        WaveDir, WaveID, WaveVolume, WaveDelta, WaveHigh, WaveLow, WavePriceDisp,
        EffortResult, ER_Ratio, WaveER,
        Absorption, AbsorptionType,
        SellingClimax, BuyingClimax, Spring, Upthrust,
        Phase,
        Signal, SignalEvent, EffortScore,
        BarTime, VolumeFilter, TimeFilter, VolumeStrength, TimeStrength
    """
    logger.info(f"Running Wyckoff analysis on {len(df)} bars")

    df = ensure_delta(df)
    df = segment_waves(df, reversal_pct=reversal_pct)
    df = compute_effort_result(df, lookback=er_lookback)
    df = detect_absorption(df, vol_threshold=vol_threshold,
                           er_threshold=er_threshold, lookback=er_lookback)
    df = detect_wyckoff_events(df, swing_lookback=swing_lookback,
                               vol_multiplier=vol_multiplier)
    df = classify_phase(df, phase_lookback=phase_lookback)
    df = generate_signals(df)
    df = compute_bar_strength(df, strength_filter=strength_filter)

    logger.info(
        f"Analysis complete — "
        f"Springs: {df['Spring'].sum()}, "
        f"Upthrusts: {df['Upthrust'].sum()}, "
        f"SC: {df['SellingClimax'].sum()}, "
        f"BC: {df['BuyingClimax'].sum()}, "
        f"Absorption bars: {df['Absorption'].sum()}"
    )

    return df


# ---------------------------------------------------------------------------
#  8.  RL STATE VECTOR EXTRACTION
# ---------------------------------------------------------------------------

def get_rl_state_vector(df, idx):
    """
    Extract a fixed-size numeric state vector from bar `idx` for the RL agent.

    Returns a numpy array of 12 features:
        [0]  price_bin          — discretized price (0–99)
        [1]  delta_norm         — bar delta / avg volume (−1 to +1 range)
        [2]  cvd_slope_norm     — CVD trend direction
        [3]  er_ratio           — effort/result ratio (absorption < 0.5, ease > 2)
        [4]  wave_dir           — current wave direction (+1/−1)
        [5]  wave_er            — wave-level effort/result
        [6]  absorption         — absorption detected (0/1)
        [7]  absorption_type    — +1 bullish / −1 bearish / 0 none
        [8]  spring             — spring event (0/1)
        [9]  upthrust           — upthrust event (0/1)
        [10] selling_climax     — selling climax (0/1)
        [11] phase              — market phase (0–4)
    """
    row = df.iloc[idx]

    # Price bin (0–99)
    prices = df['close'].values
    min_p, max_p = prices.min(), prices.max()
    p_range = max_p - min_p if max_p != min_p else 1.0
    price_bin = int(99 * (row['close'] - min_p) / p_range)
    price_bin = max(0, min(99, price_bin))

    # Normalized delta
    avg_vol = df['volume'].rolling(window=20, min_periods=1).mean().iloc[idx]
    delta_norm = row['delta'] / avg_vol if avg_vol > 0 else 0.0
    delta_norm = max(-1.0, min(1.0, delta_norm))

    # CVD slope (normalized)
    lookback = min(20, idx + 1)
    if lookback > 1:
        cvd_window = df['cvd'].iloc[max(0, idx - lookback + 1):idx + 1].values
        cvd_slope = np.polyfit(range(len(cvd_window)), cvd_window, 1)[0]
        cvd_slope_norm = np.tanh(cvd_slope / (avg_vol if avg_vol > 0 else 1.0))
    else:
        cvd_slope_norm = 0.0

    return np.array([
        price_bin,
        delta_norm,
        cvd_slope_norm,
        row.get('ER_Ratio', 1.0),
        row.get('WaveDir', 0),
        row.get('WaveER', 0.0),
        row.get('Absorption', 0),
        row.get('AbsorptionType', 0),
        row.get('Spring', 0),
        row.get('Upthrust', 0),
        row.get('SellingClimax', 0),
        row.get('Phase', 0),
    ], dtype=np.float32)


# Number of features in the RL state vector (excluding position, which the env adds)
RL_STATE_SIZE = 12
