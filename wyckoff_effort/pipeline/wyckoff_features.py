"""
Wyckoff-Weis Wave Feature Engineering — 65 features across 5 blocks.

Block 1: Bar Microstructure       (~15 features)
Block 2: Weis Wave Analysis       (~20 features)
Block 3: Wyckoff Event Evidence   (~15 features)
Block 4: Range / Phase / Context  (~10 features)
Block 5: Agent State / Execution  (~5 features, added at env level)

Input:  Raw analyzed DataFrame from analyze_wyckoff() with 40-point NQ range bars
Output: tech_ary (n_bars, ~60) for NPZ, feature_names list

Design principles:
  - Soft continuous scores (0-1) for binary events
  - No PCA, no z-score destruction of categoricals
  - Per-type normalization (ratios pass through, returns use tanh, counts use decay)
  - Weis Wave comparison is the analytical heart
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EPSILON = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a, b, fill: float = 0.0):
    """Element-wise division, returning fill where b is zero/nan."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > EPSILON, a / b, fill)
    return np.nan_to_num(result, nan=fill, posinf=fill, neginf=fill)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean using cumsum trick. Handles edges with expanding window."""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    cumsum = np.cumsum(arr)
    for i in range(n):
        w = min(window, i + 1)
        if i - w >= 0:
            out[i] = (cumsum[i] - cumsum[i - w]) / w
        else:
            out[i] = cumsum[i] / (i + 1)
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation."""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        w = min(window, i + 1)
        segment = arr[max(0, i - w + 1): i + 1]
        out[i] = np.std(segment) if len(segment) > 1 else 0.0
    return out


def _linreg_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling linear regression slope, normalized by window scale."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        w = min(window, i + 1)
        y = arr[max(0, i - w + 1): i + 1]
        if len(y) < 2:
            continue
        x = np.arange(len(y), dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom > EPSILON:
            out[i] = ((x - x_mean) * (y - y_mean)).sum() / denom
    return out


def _decay(bars_since: np.ndarray, scale: float = 20.0) -> np.ndarray:
    """Convert bars-since-event to 0-1 decay: 1/(1 + x/scale)."""
    return 1.0 / (1.0 + bars_since / scale)


def _bars_since_condition(mask: np.ndarray, max_bars: int = 500) -> np.ndarray:
    """Count bars since last True in boolean mask."""
    n = len(mask)
    out = np.full(n, max_bars, dtype=np.float64)
    last_true = -max_bars
    for i in range(n):
        if mask[i]:
            last_true = i
        out[i] = i - last_true
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Block 1: Bar Microstructure (~15 features)
# ─────────────────────────────────────────────────────────────────────────────

def compute_block1_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-bar microstructure features from OHLCV + delta + duration.

    Returns DataFrame with feature columns appended.
    """
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)

    # Delta: use raw bid/ask if available, else use precomputed
    if "delta" in df.columns:
        delta = df["delta"].values.astype(np.float64)
    elif "ask_volume" in df.columns and "bid_volume" in df.columns:
        delta = df["ask_volume"].values - df["bid_volume"].values
    else:
        delta = np.zeros(len(df), dtype=np.float64)

    # CVD
    cvd = np.cumsum(delta)

    # Duration
    if "duration_seconds" in df.columns:
        dur = df["duration_seconds"].values.astype(np.float64)
    elif "num_trades" in df.columns:
        dur = _safe_div(vol, df["num_trades"].values.astype(np.float64), fill=1.0)
    else:
        dur = np.ones(len(df), dtype=np.float64)

    bar_range = h - l
    bar_range_safe = np.maximum(bar_range, EPSILON)

    # 1. body_ratio: signed bar conviction [-1, +1]
    body_ratio = _safe_div(c - o, bar_range_safe)

    # 2. upper_wick_ratio [0, 1]
    upper_wick = h - np.maximum(o, c)
    upper_wick_ratio = _safe_div(upper_wick, bar_range_safe)

    # 3. lower_wick_ratio [0, 1]
    lower_wick = np.minimum(o, c) - l
    lower_wick_ratio = _safe_div(lower_wick, bar_range_safe)

    # 4. close_location [0, 1]
    close_location = _safe_div(c - l, bar_range_safe)

    # 5. delta_ratio [-1, +1]
    delta_ratio = np.clip(_safe_div(delta, np.maximum(vol, EPSILON)), -1.0, 1.0)

    # 6. vol_vs_ma20
    vol_ma20 = _rolling_mean(vol, 20)
    vol_vs_ma20 = np.clip(_safe_div(vol, vol_ma20), 0, 5.0)

    # 7. vol_vs_ma50
    vol_ma50 = _rolling_mean(vol, 50)
    vol_vs_ma50 = np.clip(_safe_div(vol, vol_ma50), 0, 5.0)

    # 8. er_ratio: |price_change| / volume, normalized by rolling mean
    price_change = np.abs(np.diff(c, prepend=c[0]))
    effort_result = _safe_div(price_change, np.maximum(vol, EPSILON))
    er_mean = _rolling_mean(effort_result, 20)
    er_ratio = np.clip(_safe_div(effort_result, er_mean), 0, 5.0)

    # 9. duration_norm: bar pace relative to average
    dur_ma20 = _rolling_mean(dur, 20)
    duration_norm = np.clip(_safe_div(dur, dur_ma20), 0, 5.0)

    # 10. cvd_slope_fast: LinReg(CVD, 10), tanh-bounded
    cvd_slope_fast = np.tanh(_safe_div(_linreg_slope(cvd, 10), vol_ma20))

    # 11. cvd_slope_slow: LinReg(CVD, 30), tanh-bounded
    cvd_slope_slow = np.tanh(_safe_div(_linreg_slope(cvd, 30), vol_ma20))

    # 12. cvd_divergence: price slope vs cvd slope disagreement
    price_slope = _linreg_slope(c, 20)
    cvd_slope_20 = _linreg_slope(cvd, 20)
    # Normalize each to [-1, 1] via tanh, then take difference
    price_dir = np.tanh(price_slope / (np.std(price_change) + EPSILON))
    cvd_dir = np.tanh(_safe_div(cvd_slope_20, vol_ma20))
    cvd_divergence = np.clip(price_dir - cvd_dir, -2.0, 2.0)

    # 13. return_1
    return_1 = np.tanh(_safe_div(np.diff(c, prepend=c[0]), c) * 100)

    # 14. return_5
    ret5 = np.zeros_like(c)
    ret5[5:] = (c[5:] - c[:-5]) / (c[:-5] + EPSILON)
    return_5 = np.tanh(ret5 * 50)

    # 15. volatility_20: rolling std of returns
    returns_raw = _safe_div(np.diff(c, prepend=c[0]), c)
    volatility_20 = _rolling_std(returns_raw, 20)
    # Normalize: tanh so it's bounded
    vol_20_norm = np.tanh(volatility_20 * 200)

    result = pd.DataFrame({
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "close_location": close_location,
        "delta_ratio": delta_ratio,
        "vol_vs_ma20": vol_vs_ma20,
        "vol_vs_ma50": vol_vs_ma50,
        "er_ratio": er_ratio,
        "duration_norm": duration_norm,
        "cvd_slope_fast": cvd_slope_fast,
        "cvd_slope_slow": cvd_slope_slow,
        "cvd_divergence": cvd_divergence,
        "return_1": return_1,
        "return_5": return_5,
        "volatility_20": vol_20_norm,
    }, index=df.index)

    logger.info(f"Block 1 (Microstructure): {result.shape[1]} features computed")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Block 2: Weis Wave Analysis (~20 features)
# ─────────────────────────────────────────────────────────────────────────────

def _segment_waves(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    delta: np.ndarray,
    reversal_points: float = 40.0,
) -> dict:
    """
    Weis Wave ZigZag segmentation (points-based reversal for 40pt NQ bars).

    Returns dict of arrays: wave_dir, wave_id, wave_high, wave_low,
                            wave_vol, wave_delta, wave_start_idx
    """
    n = len(close)
    wave_dir = np.ones(n, dtype=np.float64)
    wave_id = np.zeros(n, dtype=np.int32)
    wave_high = np.zeros(n, dtype=np.float64)
    wave_low = np.zeros(n, dtype=np.float64)
    wave_vol = np.zeros(n, dtype=np.float64)
    wave_delta = np.zeros(n, dtype=np.float64)

    cur_dir = 1
    cur_id = 0
    w_high = close[0]
    w_low = close[0]
    w_vol = volume[0]
    w_delta = delta[0]
    extremum_price = close[0]

    for i in range(n):
        if i == 0:
            wave_dir[i] = cur_dir
            wave_id[i] = cur_id
            wave_high[i] = w_high
            wave_low[i] = w_low
            wave_vol[i] = w_vol
            wave_delta[i] = w_delta
            continue

        reversed_flag = False

        if cur_dir == 1:  # up wave
            if close[i] > w_high:
                w_high = close[i]
                extremum_price = w_high
            # Check reversal: price dropped reversal_points from high
            if extremum_price - close[i] >= reversal_points:
                reversed_flag = True
        else:  # down wave
            if close[i] < w_low:
                w_low = close[i]
                extremum_price = w_low
            if close[i] - extremum_price >= reversal_points:
                reversed_flag = True

        if reversed_flag:
            cur_dir = -cur_dir
            cur_id += 1
            w_high = close[i]
            w_low = close[i]
            w_vol = volume[i]
            w_delta = delta[i]
            extremum_price = close[i]
        else:
            w_vol += volume[i]
            w_delta += delta[i]

        wave_dir[i] = cur_dir
        wave_id[i] = cur_id
        wave_high[i] = w_high
        wave_low[i] = w_low
        wave_vol[i] = w_vol
        wave_delta[i] = w_delta

    return {
        "wave_dir": wave_dir,
        "wave_id": wave_id,
        "wave_high": wave_high,
        "wave_low": wave_low,
        "wave_vol": wave_vol,
        "wave_delta": wave_delta,
    }


def _compute_completed_wave_stats(wave_id, wave_dir, wave_vol,
                                   wave_delta, wave_high, wave_low):
    """
    Build per-wave summary at wave completion boundaries.

    Returns arrays sized n_bars with wave stats populated at each bar
    from the sliding window of completed waves.
    """
    n = len(wave_id)

    # Storage for last 4 completed waves
    completed_vols = []
    completed_disps = []
    completed_ers = []
    completed_deltas = []
    completed_dirs = []

    # Per-bar outputs: rolling wave comparison features
    wave_vol_vs_same = np.ones(n)
    wave_vol_vs_prev = np.ones(n)
    wave_disp_vs_same = np.ones(n)
    wave_disp_vs_prev = np.ones(n)
    wave_er_vs_same = np.ones(n)
    wave_er_vs_prev = np.ones(n)
    wave_delta_vs_same = np.zeros(n)
    wave_vol_vs_avg4 = np.ones(n)
    yellow_bar = np.zeros(n)
    large_wave_score = np.ones(n)

    last_id = wave_id[0]
    prev_up_vol = 0.0
    prev_up_disp = 0.0
    prev_up_er = 0.0
    prev_up_delta = 0.0
    prev_down_vol = 0.0
    prev_down_disp = 0.0
    prev_down_er = 0.0
    prev_down_delta = 0.0
    prev_wave_vol = 0.0
    prev_wave_disp = 0.0
    prev_wave_er = 0.0

    for i in range(1, n):
        cur_id = wave_id[i]

        # Wave just completed at i-1
        if cur_id != last_id:
            j = i - 1
            wv = wave_vol[j]
            wd = wave_dir[j]
            wh = wave_high[j]
            wl = wave_low[j]
            disp = abs(wh - wl)
            er = wv / max(disp, EPSILON)
            w_delt = wave_delta[j] / max(wv, EPSILON)

            completed_vols.append(wv)
            completed_disps.append(disp)
            completed_ers.append(er)
            completed_deltas.append(w_delt)
            completed_dirs.append(wd)

            # Keep sliding window of last 8 waves
            if len(completed_vols) > 8:
                completed_vols.pop(0)
                completed_disps.pop(0)
                completed_ers.pop(0)
                completed_deltas.pop(0)
                completed_dirs.pop(0)

            # Update same-direction trackers
            if wd == 1:
                prev_up_vol, prev_up_disp, prev_up_er, prev_up_delta = wv, disp, er, w_delt
            else:
                prev_down_vol, prev_down_disp, prev_down_er, prev_down_delta = wv, disp, er, w_delt

            prev_wave_vol = wv
            prev_wave_disp = disp
            prev_wave_er = er

        last_id = cur_id

        # Current wave state
        cur_vol = wave_vol[i]
        cur_dir = wave_dir[i]
        cur_disp = abs(wave_high[i] - wave_low[i])
        cur_er = cur_vol / max(cur_disp, EPSILON)
        cur_delta_r = wave_delta[i] / max(cur_vol, EPSILON)

        # vs same direction
        if cur_dir == 1 and prev_up_vol > EPSILON:
            wave_vol_vs_same[i] = cur_vol / prev_up_vol
            wave_disp_vs_same[i] = cur_disp / max(prev_up_disp, EPSILON)
            wave_er_vs_same[i] = cur_er / max(prev_up_er, EPSILON)
            wave_delta_vs_same[i] = cur_delta_r - prev_up_delta
        elif cur_dir == -1 and prev_down_vol > EPSILON:
            wave_vol_vs_same[i] = cur_vol / prev_down_vol
            wave_disp_vs_same[i] = cur_disp / max(prev_down_disp, EPSILON)
            wave_er_vs_same[i] = cur_er / max(prev_down_er, EPSILON)
            wave_delta_vs_same[i] = cur_delta_r - prev_down_delta

        # vs immediately previous wave
        if prev_wave_vol > EPSILON:
            wave_vol_vs_prev[i] = cur_vol / prev_wave_vol
            wave_disp_vs_prev[i] = cur_disp / max(prev_wave_disp, EPSILON)
            wave_er_vs_prev[i] = cur_er / max(prev_wave_er, EPSILON)

        # vs average of last 4 completed waves
        if len(completed_vols) >= 4:
            avg4 = np.mean(completed_vols[-4:])
            wave_vol_vs_avg4[i] = cur_vol / max(avg4, EPSILON)
            large_wave_score[i] = cur_vol / max(avg4, EPSILON)

        # Yellow bar: cumulative vol exceeds last same-dir total
        if cur_dir == 1 and prev_up_vol > EPSILON:
            yellow_bar[i] = 1.0 if cur_vol > prev_up_vol else cur_vol / prev_up_vol
        elif cur_dir == -1 and prev_down_vol > EPSILON:
            yellow_bar[i] = 1.0 if cur_vol > prev_down_vol else cur_vol / prev_down_vol

    return {
        "wave_vol_vs_same": wave_vol_vs_same,
        "wave_vol_vs_prev": wave_vol_vs_prev,
        "wave_disp_vs_same": wave_disp_vs_same,
        "wave_disp_vs_prev": wave_disp_vs_prev,
        "wave_er_vs_same": wave_er_vs_same,
        "wave_er_vs_prev": wave_er_vs_prev,
        "wave_delta_vs_same": wave_delta_vs_same,
        "wave_vol_vs_avg4": wave_vol_vs_avg4,
        "yellow_bar": yellow_bar,
        "large_wave_score": large_wave_score,
    }


def _compute_supply_demand_balance(wave_id, wave_dir, wave_vol, wave_high, wave_low):
    """
    Multi-wave supply/demand scoring.

    Computes demand_score (up-wave vs down-wave power) and
    wave volume/displacement trends for each direction.
    """
    n = len(wave_id)

    # Collect completed waves into up/down buckets
    up_vols = []
    down_vols = []
    up_disps = []
    down_disps = []

    demand_score = np.ones(n)
    supply_score = np.ones(n)
    wave_vol_trend_up = np.zeros(n)
    wave_vol_trend_down = np.zeros(n)
    wave_shortening_up = np.zeros(n)
    wave_shortening_down = np.zeros(n)

    last_id = wave_id[0]

    for i in range(1, n):
        cur_id = wave_id[i]

        if cur_id != last_id:
            j = i - 1
            wv = wave_vol[j]
            wd = wave_dir[j]
            disp = abs(wave_high[j] - wave_low[j])
            if wd == 1:
                up_vols.append(wv)
                up_disps.append(disp)
                if len(up_vols) > 6:
                    up_vols.pop(0)
                    up_disps.pop(0)
            else:
                down_vols.append(wv)
                down_disps.append(disp)
                if len(down_vols) > 6:
                    down_vols.pop(0)
                    down_disps.pop(0)

        last_id = cur_id

        # demand/supply balance from last 3 each direction
        if len(up_vols) >= 3 and len(down_vols) >= 3:
            avg_up = np.mean(up_vols[-3:])
            avg_dn = np.mean(down_vols[-3:])
            demand_score[i] = avg_up / max(avg_dn, EPSILON)
            supply_score[i] = avg_dn / max(avg_up, EPSILON)

        # Volume trend (slope of last 4 same-dir waves)
        if len(up_vols) >= 4:
            x = np.arange(4, dtype=np.float64)
            y = np.array(up_vols[-4:], dtype=np.float64)
            y_mean = y.mean()
            wave_vol_trend_up[i] = np.tanh(_safe_div(
                ((x - x.mean()) * (y - y_mean)).sum(),
                ((x - x.mean()) ** 2).sum() + EPSILON
            ) / max(y_mean, EPSILON))

        if len(down_vols) >= 4:
            x = np.arange(4, dtype=np.float64)
            y = np.array(down_vols[-4:], dtype=np.float64)
            y_mean = y.mean()
            wave_vol_trend_down[i] = np.tanh(_safe_div(
                ((x - x.mean()) * (y - y_mean)).sum(),
                ((x - x.mean()) ** 2).sum() + EPSILON
            ) / max(y_mean, EPSILON))

        # Displacement shortening trend
        if len(up_disps) >= 4:
            x = np.arange(4, dtype=np.float64)
            y = np.array(up_disps[-4:], dtype=np.float64)
            y_mean = y.mean()
            wave_shortening_up[i] = np.tanh(_safe_div(
                ((x - x.mean()) * (y - y_mean)).sum(),
                ((x - x.mean()) ** 2).sum() + EPSILON
            ) / max(y_mean, EPSILON))

        if len(down_disps) >= 4:
            x = np.arange(4, dtype=np.float64)
            y = np.array(down_disps[-4:], dtype=np.float64)
            y_mean = y.mean()
            wave_shortening_down[i] = np.tanh(_safe_div(
                ((x - x.mean()) * (y - y_mean)).sum(),
                ((x - x.mean()) ** 2).sum() + EPSILON
            ) / max(y_mean, EPSILON))

    return {
        "demand_score_3wave": np.clip(demand_score, 0, 5.0),
        "supply_score_3wave": np.clip(supply_score, 0, 5.0),
        "wave_vol_trend_up": wave_vol_trend_up,
        "wave_vol_trend_down": wave_vol_trend_down,
        "wave_shortening_up": wave_shortening_up,
        "wave_shortening_down": wave_shortening_down,
    }


def compute_block2_weis_wave(df: pd.DataFrame, reversal_points: float = 40.0) -> pd.DataFrame:
    """
    Compute Weis Wave features: wave segmentation + wave-to-wave comparison.

    Parameters
    ----------
    reversal_points : float
        ZigZag reversal threshold in price points (40 for NQ 40pt range bars).
    """
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)
    delta = df.get("delta", pd.Series(np.zeros(len(df)))).values.astype(np.float64)

    # Segment waves
    waves = _segment_waves(c, h, l, vol, delta, reversal_points)

    # Current wave features
    wave_disp = np.abs(waves["wave_high"] - waves["wave_low"])
    avg_wave_vol = _rolling_mean(waves["wave_vol"], 50)
    avg_wave_len_approx = 20.0  # approximate bars per wave for progress

    # Wave displacement in bars (count bars in current wave)
    wave_bars = np.zeros(len(c))
    last_id = waves["wave_id"][0]
    count = 0
    for i in range(len(c)):
        if waves["wave_id"][i] != last_id:
            count = 0
            last_id = waves["wave_id"][i]
        count += 1
        wave_bars[i] = count

    # Wave-to-wave comparisons
    comp = _compute_completed_wave_stats(
        waves["wave_id"], waves["wave_dir"], waves["wave_vol"],
        waves["wave_delta"], waves["wave_high"], waves["wave_low"],
    )

    # Supply/demand balance
    sd = _compute_supply_demand_balance(
        waves["wave_id"], waves["wave_dir"], waves["wave_vol"],
        waves["wave_high"], waves["wave_low"],
    )

    # ATR for displacement normalization
    atr = _rolling_mean(np.abs(np.diff(c, prepend=c[0])), 20) * 20.0
    atr = np.maximum(atr, EPSILON)

    result = pd.DataFrame({
        # Current wave state (5)
        "wave_direction": waves["wave_dir"],
        "wave_progress": np.clip(wave_bars / avg_wave_len_approx, 0, 5.0),
        "wave_displacement_norm": np.clip(wave_disp / atr, 0, 5.0),
        "wave_vol_cumulative_norm": np.clip(_safe_div(waves["wave_vol"], avg_wave_vol), 0, 5.0),
        "wave_delta_ratio": np.clip(_safe_div(waves["wave_delta"], np.maximum(waves["wave_vol"], EPSILON)), -1, 1),
        # Wave comparison (7)
        "wave_vol_vs_same": np.clip(comp["wave_vol_vs_same"], 0, 5.0),
        "wave_vol_vs_prev": np.clip(comp["wave_vol_vs_prev"], 0, 5.0),
        "wave_disp_vs_same": np.clip(comp["wave_disp_vs_same"], 0, 5.0),
        "wave_disp_vs_prev": np.clip(comp["wave_disp_vs_prev"], 0, 5.0),
        "wave_er_vs_same": np.clip(comp["wave_er_vs_same"], 0, 5.0),
        "wave_er_vs_prev": np.clip(comp["wave_er_vs_prev"], 0, 5.0),
        "wave_delta_vs_same": np.clip(comp["wave_delta_vs_same"], -2, 2),
        # Multi-wave scoring (8)
        "demand_score_3wave": sd["demand_score_3wave"],
        "supply_score_3wave": sd["supply_score_3wave"],
        "wave_vol_trend_up": sd["wave_vol_trend_up"],
        "wave_vol_trend_down": sd["wave_vol_trend_down"],
        "wave_shortening_up": sd["wave_shortening_up"],
        "wave_shortening_down": sd["wave_shortening_down"],
        "yellow_bar": comp["yellow_bar"],
        "large_wave_score": np.clip(comp["large_wave_score"], 0, 5.0),
    }, index=df.index)

    logger.info(f"Block 2 (Weis Wave): {result.shape[1]} features computed, "
                f"reversal={reversal_points}pts")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Block 3: Wyckoff Event Evidence (~15 features)
# ─────────────────────────────────────────────────────────────────────────────

def compute_block3_events(df: pd.DataFrame, b2: pd.DataFrame,
                          swing_lookback: int = 20) -> pd.DataFrame:
    """
    Compute continuous Wyckoff event scores using both bar-level
    and wave-level data.

    Parameters
    ----------
    df : pd.DataFrame — original OHLCV data
    b2 : pd.DataFrame — Block 2 Weis Wave features (for wave context)
    swing_lookback : int — bars to compute swing high/low
    """
    n = len(df)
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)
    delta = df.get("delta", pd.Series(np.zeros(n))).values.astype(np.float64)

    bar_range = np.maximum(h - l, EPSILON)
    vol_ma = _rolling_mean(vol, swing_lookback)
    vol_ratio = _safe_div(vol, vol_ma)

    # Swing high/low
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    for i in range(n):
        start = max(0, i - swing_lookback)
        swing_high[i] = h[start:i + 1].max() if i > 0 else h[0]
        swing_low[i] = l[start:i + 1].min() if i > 0 else l[0]

    # Wave decay features from block 2
    wave_vol_same = b2["wave_vol_vs_same"].values if "wave_vol_vs_same" in b2 else np.ones(n)

    # Close location within bar
    close_near_low = _safe_div(c - l, bar_range)   # 0 = at low, 1 = at high
    close_near_high = 1.0 - close_near_low

    # Delta direction strength
    delta_ratio = np.clip(_safe_div(delta, np.maximum(vol, EPSILON)), -1, 1)

    # ---- Primary event scores (soft 0-1) ----

    # SPRING: low < swing_low_prev, close recovers, delta positive, wave exhaustion
    prev_swing_low = np.roll(swing_low, 1)
    prev_swing_low[0] = swing_low[0]
    spring_penetration = np.clip(_safe_div(prev_swing_low - l, bar_range), 0, 1)
    spring_recovery = np.clip(close_near_low, 0, 1)  # close recovery above bar low
    spring_delta = np.clip(delta_ratio, 0, 1)  # positive delta = buying
    # Wave exhaustion: previous down-waves declining in vol (< 1 = exhaustion)
    wave_exhaust_down = np.clip(1.0 - np.minimum(wave_vol_same, 1.0), 0, 1)
    spring_score = np.clip(
        spring_penetration * spring_recovery * (0.5 + 0.5 * spring_delta), 0, 1
    )

    # UPTHRUST: high > swing_high_prev, close rejects, delta negative
    prev_swing_high = np.roll(swing_high, 1)
    prev_swing_high[0] = swing_high[0]
    ut_penetration = np.clip(_safe_div(h - prev_swing_high, bar_range), 0, 1)
    ut_rejection = np.clip(close_near_high, 0, 1)  # close near high = no rejection
    ut_rejection = 1.0 - ut_rejection  # invert: close near low = strong rejection
    ut_delta = np.clip(-delta_ratio, 0, 1)  # negative delta = selling
    upthrust_score = np.clip(
        ut_penetration * ut_rejection * (0.5 + 0.5 * ut_delta), 0, 1
    )

    # SELLING CLIMAX: vol spike + close near low + delta positive (absorption)
    vol_spike = np.clip((vol_ratio - 1.0) / 2.0, 0, 1)  # 3× avg → 1.0
    sc_close_low = np.clip(1.0 - close_near_low, 0, 1)  # close at low end
    sc_delta = np.clip(delta_ratio, 0, 1)
    sc_score = np.clip(vol_spike * sc_close_low * (0.5 + 0.5 * sc_delta), 0, 1)

    # BUYING CLIMAX: vol spike + close near high + delta negative (absorption)
    bc_close_high = np.clip(close_near_low, 0, 1)  # close at high end
    bc_delta = np.clip(-delta_ratio, 0, 1)
    bc_score = np.clip(vol_spike * bc_close_high * (0.5 + 0.5 * bc_delta), 0, 1)

    # ABSORPTION: high volume + tiny price change
    er_bar = _safe_div(np.abs(np.diff(c, prepend=c[0])), np.maximum(vol, EPSILON))
    er_mean = _rolling_mean(er_bar, 20)
    er_ratio_bar = _safe_div(er_bar, er_mean)
    # Absorption: vol high + ER low
    abs_vol_component = np.clip((vol_ratio - 1.0) / 0.5, 0, 1)  # 1.5× → 1.0
    abs_er_component = np.clip(1.0 - er_ratio_bar / 0.5, 0, 1)  # ER<0.5 → 1.0
    absorption_score = np.clip(abs_vol_component * abs_er_component, 0, 1)
    absorption_direction = np.where(
        absorption_score > 0.3,
        np.sign(delta),
        0.0
    )

    # STOPPING ACTION (wave level): large wave vol + small displacement
    large_wave = b2["large_wave_score"].values if "large_wave_score" in b2 else np.ones(n)
    wave_disp = b2["wave_displacement_norm"].values if "wave_displacement_norm" in b2 else np.ones(n)
    # High effort (>1.5× avg vol) + small result (<0.5× avg displacement)
    stopping_vol = np.clip((large_wave - 1.0) / 1.0, 0, 1)
    stopping_disp = np.clip(1.0 - wave_disp / 1.0, 0, 1)
    stopping_action_score = np.clip(stopping_vol * stopping_disp, 0, 1)

    # ---- Event-relative temporal features ----
    bars_since_spring = _bars_since_condition(spring_score > 0.3)
    bars_since_upthrust = _bars_since_condition(upthrust_score > 0.3)
    bars_since_climax = _bars_since_condition(
        (sc_score > 0.3) | (bc_score > 0.3)
    )

    # Cumulative absorption (last 10 bars)
    cumulative_absorption = np.zeros(n)
    for i in range(n):
        start = max(0, i - 9)
        cumulative_absorption[i] = absorption_score[start:i + 1].sum() / 10.0

    # Event sequence score (simple schematic progression)
    # Bull: SC → spring → SOS-like (breakout with volume)
    # Bear: BC → upthrust → SOW-like
    event_seq_bull = (
        _decay(bars_since_climax, 100) * 0.3  # climax happened recently
        + _decay(_bars_since_condition(spring_score > 0.3), 50) * 0.4  # spring
        + np.clip(b2["demand_score_3wave"].values - 1.0, 0, 1) * 0.3  # demand building
    )
    event_seq_bear = (
        _decay(bars_since_climax, 100) * 0.3
        + _decay(_bars_since_condition(upthrust_score > 0.3), 50) * 0.4
        + np.clip(b2["supply_score_3wave"].values - 1.0, 0, 1) * 0.3
    )

    result = pd.DataFrame({
        "spring_score": spring_score,
        "upthrust_score": upthrust_score,
        "sc_score": sc_score,
        "bc_score": bc_score,
        "absorption_score": absorption_score,
        "absorption_direction": absorption_direction,
        "stopping_action_score": stopping_action_score,
        "bars_since_spring": _decay(bars_since_spring),
        "bars_since_upthrust": _decay(bars_since_upthrust),
        "bars_since_climax": _decay(bars_since_climax),
        "cumulative_absorption": cumulative_absorption,
        "event_sequence_bull": np.clip(event_seq_bull, 0, 1),
        "event_sequence_bear": np.clip(event_seq_bear, 0, 1),
    }, index=df.index)

    logger.info(f"Block 3 (Events): {result.shape[1]} features computed")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Block 4: Range / Phase / Context (~10 features)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_trading_range(close, high, low, lookback: int = 50, breakout_mult: float = 1.5):
    """
    Adaptive trading range detection.

    A range is the high/low of the last `lookback` bars.
    Resets when close breaks cleanly beyond range by breakout_mult × ATR.
    """
    n = len(close)
    range_high = np.zeros(n)
    range_low = np.zeros(n)
    bars_in_range = np.zeros(n)
    support_tests = np.zeros(n)
    resistance_tests = np.zeros(n)

    atr = _rolling_mean(np.abs(np.diff(close, prepend=close[0])), 20)
    # Use expanding window up to lookback
    for i in range(n):
        start = max(0, i - lookback + 1)
        range_high[i] = high[start:i + 1].max()
        range_low[i] = low[start:i + 1].min()

    # Count support/resistance tests
    rng = range_high - range_low
    test_zone = 0.15  # within 15% of boundary = "test"

    s_count = 0
    r_count = 0
    bar_count = 0
    prev_rh = range_high[0]
    prev_rl = range_low[0]

    for i in range(n):
        # Detect range reset (significant breakout)
        if i > 0:
            a = max(atr[i], EPSILON)
            if close[i] > prev_rh + breakout_mult * a or close[i] < prev_rl - breakout_mult * a:
                s_count = 0
                r_count = 0
                bar_count = 0

        bar_count += 1
        rng_i = range_high[i] - range_low[i]

        # Support test: price touches lower 15% of range
        if rng_i > EPSILON and (close[i] - range_low[i]) / rng_i < test_zone:
            s_count += 1
        # Resistance test: price touches upper 15% of range
        if rng_i > EPSILON and (range_high[i] - close[i]) / rng_i < test_zone:
            r_count += 1

        bars_in_range[i] = bar_count
        support_tests[i] = s_count
        resistance_tests[i] = r_count
        prev_rh = range_high[i]
        prev_rl = range_low[i]

    return range_high, range_low, bars_in_range, support_tests, resistance_tests


def compute_block4_context(df: pd.DataFrame, b2: pd.DataFrame,
                           phase_lookback: int = 50) -> pd.DataFrame:
    """
    Compute trading range structure, phase likelihood, and multi-timeframe context.
    """
    n = len(df)
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)
    delta = df.get("delta", pd.Series(np.zeros(n))).values.astype(np.float64)
    cvd = np.cumsum(delta)

    # Trading range detection
    rh, rl, bars_rng, s_tests, r_tests = _detect_trading_range(c, h, l)

    # Percent in range
    rng = rh - rl
    pct_in_range = np.clip(_safe_div(c - rl, np.maximum(rng, EPSILON)), 0, 1)

    # Range width normalized by ATR
    atr_50 = _rolling_mean(np.abs(np.diff(c, prepend=c[0])), 50) * 50
    range_width_norm = np.clip(_safe_div(rng, np.maximum(atr_50, EPSILON)), 0, 5)

    # ---- Phase likelihood scores ----
    price_slope = _linreg_slope(c, phase_lookback)
    cvd_slope = _linreg_slope(cvd, phase_lookback)
    vol_ma = _rolling_mean(vol, phase_lookback)

    # Normalize slopes
    price_level = _rolling_mean(c, phase_lookback)
    norm_price_slope = _safe_div(price_slope, price_level)
    norm_cvd_slope = np.tanh(_safe_div(cvd_slope, vol_ma))

    # Wave trends from block 2
    wv_trend_up = b2["wave_vol_trend_up"].values
    wv_trend_down = b2["wave_vol_trend_down"].values
    demand = b2["demand_score_3wave"].values
    supply = b2["supply_score_3wave"].values

    # Phase scores (continuous 0-1)
    # Accumulation: flat price, rising CVD, demand > supply
    phase_accum = np.clip(
        (1.0 - np.clip(np.abs(norm_price_slope) * 2000, 0, 1)) * 0.4  # flat price
        + np.clip(norm_cvd_slope, 0, 1) * 0.3  # rising CVD
        + np.clip((demand - 1.0), 0, 1) * 0.3,  # demand winning
        0, 1
    )

    # Markup: rising price, rising CVD, up-waves getting bigger
    phase_markup = np.clip(
        np.clip(norm_price_slope * 2000, 0, 1) * 0.4  # rising price
        + np.clip(norm_cvd_slope, 0, 1) * 0.3  # rising CVD
        + np.clip(wv_trend_up, 0, 1) * 0.3,  # up-wave vol increasing
        0, 1
    )

    # Distribution: flat price, falling CVD, supply > demand
    phase_distrib = np.clip(
        (1.0 - np.clip(np.abs(norm_price_slope) * 2000, 0, 1)) * 0.4
        + np.clip(-norm_cvd_slope, 0, 1) * 0.3
        + np.clip((supply - 1.0), 0, 1) * 0.3,
        0, 1
    )

    # Markdown: falling price, falling CVD, down-waves getting bigger
    phase_markdown = np.clip(
        np.clip(-norm_price_slope * 2000, 0, 1) * 0.4
        + np.clip(-norm_cvd_slope, 0, 1) * 0.3
        + np.clip(wv_trend_down, 0, 1) * 0.3,
        0, 1
    )

    # Higher timeframe trend (4× structure)
    trend_4x = np.tanh(_linreg_slope(c, 80) / (np.std(c) / 80 + EPSILON))

    result = pd.DataFrame({
        "pct_in_range": pct_in_range,
        "range_width_norm": range_width_norm,
        "bars_in_range": _decay(bars_rng, 100),
        "support_test_count": np.clip(s_tests / 5.0, 0, 1),
        "resistance_test_count": np.clip(r_tests / 5.0, 0, 1),
        "phase_accum_score": phase_accum,
        "phase_markup_score": phase_markup,
        "phase_distrib_score": phase_distrib,
        "phase_markdown_score": phase_markdown,
        "trend_4x": trend_4x,
    }, index=df.index)

    logger.info(f"Block 4 (Context): {result.shape[1]} features computed")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Master Feature Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_all_features(
    df: pd.DataFrame,
    reversal_points: float = 40.0,
    swing_lookback: int = 20,
    phase_lookback: int = 50,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Build all feature blocks from raw analyzed bar data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + delta + duration data from range bars.
    reversal_points : float
        ZigZag reversal in price points (40 for NQ 40pt range bars).
    swing_lookback : int
        Bars for swing high/low computation.
    phase_lookback : int
        Bars for phase likelihood lookback.

    Returns
    -------
    tech_ary : np.ndarray, shape (n_bars, n_features)
    feature_names : list of str
    features_df : pd.DataFrame with all feature columns
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Block 1: Bar microstructure
    b1 = compute_block1_microstructure(df)

    # Block 2: Weis Wave analysis
    b2 = compute_block2_weis_wave(df, reversal_points=reversal_points)

    # Block 3: Wyckoff event evidence (uses wave data from B2)
    b3 = compute_block3_events(df, b2, swing_lookback=swing_lookback)

    # Block 4: Range/phase/context (uses wave data from B2)
    b4 = compute_block4_context(df, b2, phase_lookback=phase_lookback)

    # Concatenate all blocks
    features_df = pd.concat([b1, b2, b3, b4], axis=1)
    feature_names = features_df.columns.tolist()

    # Convert to numpy, clean NaN/inf
    tech_ary = features_df.values.astype(np.float32)
    tech_ary = np.nan_to_num(tech_ary, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        f"Total features: {len(feature_names)} across 4 blocks "
        f"(B1={b1.shape[1]}, B2={b2.shape[1]}, B3={b3.shape[1]}, B4={b4.shape[1]})")
    return tech_ary, feature_names, features_df


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance Evaluation (no PCA — MDA/SFI + forward return correlation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_forward_returns(close: np.ndarray, horizons: list = None) -> dict:
    """
    Compute forward returns at multiple horizons for feature evaluation.

    Returns dict of {f"fwd_{h}bar": np.ndarray} for each horizon.
    """
    horizons = horizons or [1, 5, 10, 20]
    result = {}
    for h in horizons:
        fwd = np.zeros(len(close), dtype=np.float64)
        if h < len(close):
            fwd[:-h] = (close[h:] - close[:-h]) / (close[:-h] + EPSILON)
        result[f"fwd_{h}bar"] = fwd
    return result


def evaluate_feature_importance(
    tech_ary: np.ndarray,
    feature_names: list,
    close: np.ndarray,
    method: str = "all",
) -> dict:
    """
    Evaluate feature importance using multiple methods.

    Methods:
    1. Forward return correlation: Spearman rank correlation of each feature
       with forward returns at 1/5/10/20 bar horizons.
    2. MDA (Mean Decrease Accuracy): Permutation-based with RF classifier
       trained on sign(fwd_5bar) as target.
    3. SFI (Single Feature Importance): Each feature alone predicts the target.
       Identifies features with standalone predictive power vs features that
       only contribute in combination.

    No PCA — features are evaluated in their original space.
    """
    from scipy.stats import spearmanr

    results = {}

    # Forward returns
    fwd = compute_forward_returns(close)

    # ---- 1. Spearman correlation with forward returns ----
    if method in ("correlation", "all"):
        corr_data = {}
        for horizon, fwd_ret in fwd.items():
            valid = ~np.isnan(fwd_ret) & (fwd_ret != 0)
            corrs = []
            for j in range(tech_ary.shape[1]):
                feat = tech_ary[valid, j]
                ret = fwd_ret[valid]
                if np.std(feat) < EPSILON:
                    corrs.append(0.0)
                else:
                    rho, _ = spearmanr(feat, ret)
                    corrs.append(rho if not np.isnan(rho) else 0.0)
            corr_data[horizon] = corrs

        corr_df = pd.DataFrame(corr_data, index=feature_names)
        corr_df["abs_mean_corr"] = corr_df.abs().mean(axis=1)
        corr_df = corr_df.sort_values("abs_mean_corr", ascending=False)
        results["correlation"] = corr_df
        logger.info(f"Correlation top 10:\n{corr_df['abs_mean_corr'].head(10)}")

    # ---- 2. MDA (permutation importance) ----
    if method in ("mda", "all"):
        try:
            from .feature_selection import compute_feature_importance_mda
            # Target: sign of 5-bar forward return
            target = np.sign(fwd["fwd_5bar"])
            # Remove bars where fwd return is 0 or NaN
            valid = (target != 0) & ~np.isnan(target)
            labels = ((target[valid] + 1) / 2).astype(int)  # 0 or 1
            mda_df = compute_feature_importance_mda(
                tech_ary[valid], labels, feature_names
            )
            results["mda"] = mda_df
        except Exception as e:
            logger.warning(f"MDA failed: {e}")

    # ---- 3. SFI (single feature importance) ----
    if method in ("sfi", "all"):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from RiskLabAI.features.feature_importance.feature_importance_sfi import (
                FeatureImportanceSFI,
            )

            target = np.sign(fwd["fwd_5bar"])
            valid = (target != 0) & ~np.isnan(target)
            labels = ((target[valid] + 1) / 2).astype(int)

            X = pd.DataFrame(tech_ary[valid], columns=feature_names)
            y = pd.Series(labels)

            clf = RandomForestClassifier(
                n_estimators=100, max_features=1,
                random_state=42, n_jobs=-1,
            )
            sfi = FeatureImportanceSFI(clf, n_splits=5, scoring="accuracy")
            sfi_df = sfi.compute(X, y)
            if "FeatureName" in sfi_df.columns:
                sfi_df = sfi_df.set_index("FeatureName")
            sfi_df = sfi_df.sort_values("Mean", ascending=False)
            results["sfi"] = sfi_df
            logger.info(f"SFI top 10:\n{sfi_df.head(10)}")
        except Exception as e:
            logger.warning(f"SFI failed: {e}")

    # ---- Summary: combine all methods into ranking ----
    if "correlation" in results:
        summary = pd.DataFrame(index=feature_names)
        summary["corr_rank"] = results["correlation"]["abs_mean_corr"].rank(ascending=False)
        if "mda" in results:
            summary["mda_rank"] = results["mda"]["Mean"].rank(ascending=False)
        if "sfi" in results:
            summary["sfi_rank"] = results["sfi"]["Mean"].rank(ascending=False)
        summary["avg_rank"] = summary.mean(axis=1)
        summary = summary.sort_values("avg_rank")
        results["summary"] = summary

    return results


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_pipeline(
    scid_path: str = None,
    bar_size: float = None,
    reversal_points: float = None,
    output_dir: str = None,
    run_importance: bool = True,
    mda_n_estimators: int = 200,
    mda_n_splits: int = 5,
    sfi_n_estimators: int = 100,
    sfi_n_splits: int = 5,
    subsample_importance: int = None,
) -> dict:
    """
    Run the complete new-feature pipeline: SCID → 40pt bars → features → NPZ.

    Optionally runs 3-method feature importance evaluation.

    Parameters
    ----------
    scid_path : str
        Path to SCID file. Defaults to config.DEFAULT_SCID_PATH.
    bar_size : float
        Range bar size in points. Defaults to config.RANGE_BAR_SIZE (40.0).
    reversal_points : float
        ZigZag reversal for Weis Wave. Defaults to config.REVERSAL_POINTS (200.0).
    output_dir : str
        Output directory. Defaults to config.OUTPUT_DIR.
    run_importance : bool
        If True, run correlation + MDA + SFI importance evaluation.
    mda_n_estimators : int
        Number of trees for MDA Random Forest.
    mda_n_splits : int
        CV folds for MDA.
    sfi_n_estimators : int
        Number of trees for SFI Random Forest.
    sfi_n_splits : int
        CV folds for SFI.
    subsample_importance : int or None
        If set, subsample to N bars for MDA/SFI (useful for large datasets).

    Returns
    -------
    dict with keys: tech_ary, feature_names, close_ary, bars_df, npz_path,
                    parquet_path, importance_results (if run_importance=True)
    """
    import os
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from ..utils.scid_parser import SCIDReader, resample_range_bars
    from .config import (
        DEFAULT_SCID_PATH, OUTPUT_DIR, RANGE_BAR_SIZE, REVERSAL_POINTS,
    )

    scid_path = scid_path or DEFAULT_SCID_PATH
    bar_size = bar_size or RANGE_BAR_SIZE
    reversal_points = reversal_points or REVERSAL_POINTS
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()

    # ── Step 1: Load SCID ticks ──────────────────────────────────────────
    logger.info(f"Loading SCID: {scid_path}")
    reader = SCIDReader(scid_path)
    ticks = reader.read()
    logger.info(f"Loaded {len(ticks):,} ticks")

    # ── Step 2: Build range bars ─────────────────────────────────────────
    logger.info(f"Building {bar_size}pt range bars...")
    bars_df = resample_range_bars(ticks, range_size=bar_size, tick_size=0.25)
    logger.info(f"Built {len(bars_df):,} range bars")

    # Save bars parquet
    bars_path = os.path.join(output_dir, f"wyckoff_nq_{int(bar_size)}pt_bars.parquet")
    bars_df.to_parquet(bars_path)
    logger.info(f"Saved bars: {bars_path}")

    # ── Step 3: Build features ───────────────────────────────────────────
    logger.info(f"Building features (reversal_points={reversal_points})...")
    tech_ary, feature_names, feat_df = build_all_features(
        bars_df, reversal_points=reversal_points,
    )
    logger.info(f"Feature matrix: {tech_ary.shape}")

    # ── Step 4: Save NPZ ────────────────────────────────────────────────
    close_ary = bars_df["close"].values.reshape(-1, 1).astype(np.float32)

    if isinstance(bars_df.index, pd.DatetimeIndex):
        dates_ary = bars_df.index.astype(str).values
    else:
        dates_ary = np.arange(len(bars_df)).astype(str)

    npz_path = os.path.join(output_dir, f"wyckoff_nq_{int(bar_size)}pt.npz")
    np.savez_compressed(
        npz_path,
        close_ary=close_ary,
        tech_ary=tech_ary,
        dates_ary=dates_ary,
        feature_names=np.array(feature_names),
    )
    size_mb = os.path.getsize(npz_path) / 1024 / 1024
    logger.info(f"Saved NPZ: {npz_path} ({size_mb:.1f} MB)")

    result = {
        "tech_ary": tech_ary,
        "feature_names": feature_names,
        "close_ary": close_ary,
        "bars_df": bars_df,
        "npz_path": npz_path,
        "parquet_path": bars_path,
        "n_bars": len(bars_df),
        "n_features": len(feature_names),
    }

    # ── Step 5: Feature importance (optional) ────────────────────────────
    if run_importance:
        logger.info("Running feature importance evaluation...")
        close = bars_df["close"].values

        importance = evaluate_feature_importance(
            tech_ary, feature_names, close,
            method="all",
        )

        # If dataset is large and subsample requested, re-run MDA/SFI
        if subsample_importance and len(bars_df) > subsample_importance:
            logger.info(f"Re-running MDA/SFI on {subsample_importance} subsample...")
            importance_sub = evaluate_feature_importance(
                tech_ary, feature_names, close,
                method="all",
            )
            importance = importance_sub

        # Save importance CSVs
        prefix = os.path.join(output_dir, f"feature_importance_{int(bar_size)}pt")
        if "correlation" in importance:
            importance["correlation"].to_csv(f"{prefix}_correlation.csv")
        if "mda" in importance:
            importance["mda"].to_csv(f"{prefix}_mda.csv")
        if "sfi" in importance:
            importance["sfi"].to_csv(f"{prefix}_sfi.csv")
        if "summary" in importance:
            importance["summary"].to_csv(f"{prefix}_summary.csv")
            logger.info(f"Feature importance saved to {prefix}_*.csv")

            # Log tiers
            summary = importance["summary"]
            tier1 = summary.head(15).index.tolist()
            tier4 = summary.tail(13).index.tolist()
            logger.info(f"Tier 1 (top 15): {tier1}")
            logger.info(f"Tier 4 (drop): {tier4}")

        result["importance"] = importance

    elapsed = time.time() - t_start
    logger.info(f"Pipeline complete: {result['n_bars']:,} bars, "
                f"{result['n_features']} features in {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wyckoff-Weis Wave Feature Pipeline")
    parser.add_argument("--scid", type=str, default=None, help="Path to SCID file")
    parser.add_argument("--bar-size", type=float, default=None, help="Range bar size in points")
    parser.add_argument("--reversal", type=float, default=None, help="ZigZag reversal points")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no-importance", action="store_true", help="Skip importance evaluation")
    args = parser.parse_args()

    result = run_feature_pipeline(
        scid_path=args.scid,
        bar_size=args.bar_size,
        reversal_points=args.reversal,
        output_dir=args.output_dir,
        run_importance=not args.no_importance,
    )
    print(f"\nPipeline complete:")
    print(f"  Bars:     {result['n_bars']:,}")
    print(f"  Features: {result['n_features']}")
    print(f"  NPZ:      {result['npz_path']}")
    print(f"  Parquet:  {result['parquet_path']}")
