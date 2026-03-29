"""
Structural labeling pipeline for supervised pre-training of transformer heads.

v2 design:
- Wave-first
- Stateful range objects
- Confirmation-based spring / upthrust labeling
- Confidence-weighted labels
- Phase labels use ignore index for ambiguous bars

Outputs:
    phase         : (n_bars,) int64
    phase_weight  : (n_bars,) float32
    events        : (n_bars, N_EVENTS) float32
    event_weight  : (n_bars, N_EVENTS) float32
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd

from .config_v2 import (
    N_EVENTS,
    REGIME_LABELS,
    EVENT_LABELS,
    PHASE_IGNORE_INDEX,
)

REGIME_TO_IDX = {v: k for k, v in REGIME_LABELS.items()}
EVENT_TO_IDX = {v: k for k, v in EVENT_LABELS.items()}


# ═══════════════════════════════════════════════════════════════════════
# Range state
# ═══════════════════════════════════════════════════════════════════════

class RangeSide(Enum):
    ACCUM = 1
    DISTR = -1


class RangeStatus(Enum):
    TENTATIVE_CLIMAX = auto()
    SEEK_TESTS = auto()
    MATURE = auto()
    POST_TERMINAL = auto()
    RESOLVED = auto()
    FAILED = auto()
    EXPIRED = auto()


@dataclass
class WyckoffLabelConfig:
    # Pivot detection mode: "swing" (confirmed H/L) or "nolag" (srl library)
    pivot_mode: str = "swing"
    swing_left: int = 4
    swing_right: int = 2

    trend_lookback_waves: int = 6
    climax_lookback_waves: int = 12

    climax_vol_pct: float = 0.85
    climax_disp_pct: float = 0.80

    ar_min_response_ratio: float = 0.50

    st_tolerance_frac: float = 0.12
    test_vol_max_ratio: float = 0.90

    spring_min_break_frac: float = 0.02
    spring_max_break_frac: float = 0.18
    spring_vol_max_ratio: float = 0.85

    confirm_waves: int = 2
    confirm_breakout_frac: float = 0.30
    hold_invalid_frac: float = 0.25

    breakout_min_frac: float = 0.12
    breakout_close_min_frac: float = 0.04
    breakout_hold_frac: float = 0.10
    breakout_followthrough_frac: float = 0.10
    breakout_vol_rank_min: float = 0.60
    breakout_disp_rank_min: float = 0.60
    post_terminal_breakout_window: int = 10
    mature_breakout_window: int = 12

    lps_tolerance_frac: float = 0.10
    lps_search_waves: int = 12
    lps_min_waves_after_breakout: int = 1
    lps_vol_max_ratio: float = 0.75
    lps_close_hold_frac: float = 0.15
    lps_max_retrace_frac: float = 0.45
    lps_max_disp_ratio: float = 0.80
    lps_resume_waves: int = 2
    lps_resume_frac: float = 0.08

    min_range_tests: int = 2
    mature_threshold: float = 0.55
    max_range_waves: int = 60

    phase_trend_window: int = 5
    min_event_confidence: float = 0.60
    min_breakout_confidence: float = 0.60

    # Threshold gates (previously hardcoded in function bodies)
    climax_tentative_gate: float = 0.62
    vol_rank_lookback: int = 8
    sos_follow_min: float = 0.40
    sow_follow_min: float = 0.35
    st_conf_gate: float = 0.55
    failed_terminal_gate: float = 0.35
    lps_resume_min: float = 0.40
    lps_conf_gate: float = 0.60
    weight_cap: float = 0.50
    weight_cap_events: tuple = field(default_factory=lambda: ("st_support", "st_resistance", "sos", "lpsy"))
    trend_strong_threshold: float = 0.75
    trend_gap_min: float = 0.15
    min_wave_bars: int = 3
    zigzag_pct_reversal: float = 0.005


@dataclass
class RangeObject:
    side: RangeSide
    climax_wave: int
    climax_bar: int
    climax_price: float
    climax_vol: float
    climax_high: float
    climax_low: float
    climax_disp: float
    climax_delta: float

    response_wave: int | None = None
    response_bar: int | None = None

    support: float = np.nan
    resistance: float = np.nan
    width: float = np.nan

    support_tests: list[int] = field(default_factory=list)
    resistance_tests: list[int] = field(default_factory=list)

    maturity: float = 0.0
    status: RangeStatus = RangeStatus.TENTATIVE_CLIMAX

    terminal_wave: int | None = None
    breakout_wave: int | None = None
    continuation_wave: int | None = None

    start_bar: int | None = None
    end_bar: int | None = None


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def generate_structural_labels(
    parquet_path: str,
    npz_path: str | None = None,
    label_config: WyckoffLabelConfig | None = None,
) -> dict:
    """
    Generate structural labels from raw OHLCV bars.

    Returns
    -------
    dict
        {
            "phase": (n,) int64,
            "phase_weight": (n,) float32,
            "events": (n, N_EVENTS) float32,
            "event_weight": (n, N_EVENTS) float32,
        }
    """
    cfg = label_config if isinstance(label_config, WyckoffLabelConfig) else WyckoffLabelConfig()

    df = pd.read_parquet(parquet_path).copy()
    n = len(df)

    if npz_path is not None:
        _verify_alignment(df, npz_path)

    waves, _ = _get_wave_segments(df, cfg)
    wave_df = build_wave_table(df, waves)

    phase = np.full(n, PHASE_IGNORE_INDEX, dtype=np.int64)
    phase_weight = np.zeros(n, dtype=np.float32)

    events = np.zeros((n, N_EVENTS), dtype=np.float32)
    event_weight = np.zeros((n, N_EVENTS), dtype=np.float32)

    # 1) Background trend labels: only markdown / markup / ignore
    bg_phase, bg_weight = _label_trend_background(df, waves, cfg)
    phase[:] = bg_phase
    phase_weight[:] = bg_weight

    # 2) Stateful range / event scan
    structures = _scan_range_structures(df, wave_df, cfg, events, event_weight)

    # 3) Post-process continuation events (LPS / LPSY)
    _post_label_continuation_events(wave_df, structures, cfg, events, event_weight)

    # 4) Paint accumulation / distribution over mature ranges
    _paint_range_phases(phase, phase_weight, wave_df, structures, cfg)

    # 5) Paint confirmed breakout continuation as markup / markdown
    _paint_post_breakout_phases(phase, phase_weight, wave_df, structures)

    # 6) Resolve bars with conflicting bull + bear events
    _deconflict_events(events, event_weight)

    # 7) Cap event_weight for low-alpha events (applied AFTER deconfliction
    #    so tie-breaking uses true confidence, not capped weight)
    for ename in cfg.weight_cap_events:
        eidx = EVENT_TO_IDX[ename]
        mask = events[:, eidx] > 0
        event_weight[mask, eidx] = np.minimum(event_weight[mask, eidx], cfg.weight_cap)

    return {
        "phase": phase,
        "phase_weight": phase_weight,
        "events": events,
        "event_weight": event_weight,
    }


# Backward-compat alias
generate_weak_labels = generate_structural_labels


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

# Bull events (accumulation side): SC, AR_rally, ST_support, spring, SoS, LPS, failed_spring
_BULL_COLS = np.array([EVENT_TO_IDX[e] for e in
    ("selling_climax", "automatic_rally", "st_support", "spring", "sos", "lps", "failed_spring")
    if e in EVENT_TO_IDX], dtype=np.intp)

# Bear events (distribution side): BC, AR_reaction, ST_resistance, upthrust, SoW, LPSY, failed_upthrust
_BEAR_COLS = np.array([EVENT_TO_IDX[e] for e in
    ("buying_climax", "automatic_reaction", "st_resistance", "upthrust", "sow", "lpsy", "failed_upthrust")
    if e in EVENT_TO_IDX], dtype=np.intp)


def _deconflict_events(events: np.ndarray, event_weight: np.ndarray):
    """On bars with both bull and bear events, keep the higher-confidence side."""
    for bar in range(events.shape[0]):
        bull_mask = events[bar, _BULL_COLS] > 0
        bear_mask = events[bar, _BEAR_COLS] > 0
        if not (bull_mask.any() and bear_mask.any()):
            continue

        bull_conf = event_weight[bar, _BULL_COLS[bull_mask]].max()
        bear_conf = event_weight[bar, _BEAR_COLS[bear_mask]].max()

        if bull_conf >= bear_conf:
            events[bar, _BEAR_COLS] = 0.0
            event_weight[bar, _BEAR_COLS] = 0.0
        else:
            events[bar, _BULL_COLS] = 0.0
            event_weight[bar, _BULL_COLS] = 0.0

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _percentile_rank(values, current: float) -> float:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return 0.50
    return float(np.mean(a <= current))


def _same_direction_history(wave_df: pd.DataFrame, i: int, direction: int, lookback: int) -> pd.DataFrame:
    hist = wave_df.iloc[max(0, i - lookback * 2):i]
    hist = hist[hist["direction"] == direction]
    if len(hist) > lookback:
        hist = hist.iloc[-lookback:]
    return hist


def _same_dir_rank(wave_df: pd.DataFrame, i: int, column: str, lookback: int) -> float:
    direction = int(wave_df["direction"].iat[i])
    hist = _same_direction_history(wave_df, i, direction, lookback)
    return _percentile_rank(hist[column].to_numpy(), float(wave_df[column].iat[i]))


def _trend_score(wave_df: pd.DataFrame, i: int, side: RangeSide, lookback: int) -> float:
    hist = wave_df.iloc[max(0, i - lookback):i]
    if len(hist) < max(4, lookback // 2):
        return 0.0

    if side == RangeSide.ACCUM:
        dir_frac = float((hist["direction"] == -1).mean())
        ll_frac = float((np.diff(hist["low"].to_numpy()) < 0).mean()) if len(hist) > 1 else 0.0
        net = float(hist["open"].iat[0] - hist["close"].iat[-1])
    else:
        dir_frac = float((hist["direction"] == 1).mean())
        ll_frac = float((np.diff(hist["high"].to_numpy()) > 0).mean()) if len(hist) > 1 else 0.0
        net = float(hist["close"].iat[-1] - hist["open"].iat[0])

    disp_scale = float(hist["displacement"].median()) + 1e-6
    net_score = _clip01(net / (disp_scale * max(len(hist) * 0.75, 1.0)))
    return _clip01(0.40 * dir_frac + 0.30 * ll_frac + 0.30 * net_score)


def _append_with_sep(values: list[int], new_value: int, min_sep: int = 2) -> bool:
    if not values:
        values.append(new_value)
        return True
    if new_value - values[-1] >= min_sep:
        values.append(new_value)
        return True
    return False


def _wave_end_bar(wave_df: pd.DataFrame, wave_idx: int | None) -> int | None:
    if wave_idx is None:
        return None
    return int(wave_df["end_idx"].iat[wave_idx])


def _paint_span(labels: np.ndarray, weights: np.ndarray, start: int, end: int, label: int, weight: float):
    if start is None or end is None:
        return
    start = max(0, int(start))
    end = min(len(labels) - 1, int(end))
    if end < start:
        return

    idx = np.arange(start, end + 1, dtype=np.int64)
    mask = weights[idx] <= weight
    idx = idx[mask]
    labels[idx] = label
    weights[idx] = np.float32(weight)


def _mark_event(events: np.ndarray, event_weight: np.ndarray, bar_idx: int, event_name: str, confidence: float, weight: float | None = None):
    if event_name not in EVENT_TO_IDX:
        return
    idx = EVENT_TO_IDX[event_name]
    bar_idx = int(np.clip(bar_idx, 0, len(events) - 1))
    events[bar_idx, idx] = 1.0
    w = weight if weight is not None else confidence
    event_weight[bar_idx, idx] = max(event_weight[bar_idx, idx], np.float32(w))


# ═══════════════════════════════════════════════════════════════════════
# Wave segmentation
# ═══════════════════════════════════════════════════════════════════════

def _get_wave_segments(df: pd.DataFrame, cfg: WyckoffLabelConfig | None = None):
    mode = cfg.pivot_mode if cfg is not None else "swing"
    if mode == "swing":
        swl = cfg.swing_left if cfg is not None else 8
        swr = cfg.swing_right if cfg is not None else 3
        return _swing_pivot_waves(df, swing_left=swl, swing_right=swr), None
    elif mode == "nolag":
        try:
            from wyckoff_effort.pipeline.wyckoff_features import _segment_waves_nolag
            return _segment_waves_nolag(df)
        except ImportError:
            pct = cfg.zigzag_pct_reversal if cfg is not None else 0.005
            return _simple_zigzag_waves(df, pct_reversal=pct), None
    else:
        pct = cfg.zigzag_pct_reversal if cfg is not None else 0.005
        return _simple_zigzag_waves(df, pct_reversal=pct), None


def _swing_pivot_waves(
    df: pd.DataFrame,
    swing_left: int = 8,
    swing_right: int = 3,
) -> dict:
    """Confirmed swing-pivot wave segmentation (similar to ta.pivothigh/low).

    A pivot high is confirmed when the high at bar *c* is the highest
    among bars [c - swing_left, c + swing_right].  Analogous for lows.
    This naturally filters micro-noise and produces meaningful structural
    waves — roughly equivalent to MAD135's swL=10/swR=5 on 5-min bars.
    """
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    n = len(df)

    # --- Pass 1: find confirmed pivot indices -----------------------
    pivot_idx: list[int] = []      # bar index of each pivot
    pivot_type: list[int] = []     # +1 = swing high, -1 = swing low

    for c in range(swing_left, n - swing_right):
        is_hi = True
        for j in range(c - swing_left, c + swing_right + 1):
            if j != c and high[j] >= high[c]:
                is_hi = False
                break
        is_lo = True
        for j in range(c - swing_left, c + swing_right + 1):
            if j != c and low[j] <= low[c]:
                is_lo = False
                break
        if is_hi and is_lo:
            # Ambiguous — pick whichever extreme is more prominent
            hi_range = high[c] - min(high[c - swing_left:c + swing_right + 1])
            lo_range = max(low[c - swing_left:c + swing_right + 1]) - low[c]
            if hi_range >= lo_range:
                is_lo = False
            else:
                is_hi = False
        if is_hi:
            # Avoid consecutive same-type pivots — keep the higher one
            if pivot_type and pivot_type[-1] == 1:
                if high[c] > high[pivot_idx[-1]]:
                    pivot_idx[-1] = c
                continue
            pivot_idx.append(c)
            pivot_type.append(1)
        elif is_lo:
            if pivot_type and pivot_type[-1] == -1:
                if low[c] < low[pivot_idx[-1]]:
                    pivot_idx[-1] = c
                continue
            pivot_idx.append(c)
            pivot_type.append(-1)

    # --- Pass 2: assign wave_id and wave_dir per bar ----------------
    wave_dir = np.ones(n, dtype=np.int8)
    wave_id = np.zeros(n, dtype=np.int32)

    if len(pivot_idx) < 2:
        # Not enough pivots — return a single wave
        return {
            "wave_dir": wave_dir,
            "wave_id": wave_id,
            "wave_high": high.copy(),
            "wave_low": low.copy(),
            "wave_vol": volume.copy(),
            "wave_delta": np.zeros(n, dtype=np.float64),
        }

    current_wave = 0
    # Before first pivot: direction from first pivot type
    first_dir = -1 if pivot_type[0] == 1 else 1  # approaching a high → up wave
    for b in range(pivot_idx[0] + 1):
        wave_dir[b] = first_dir
        wave_id[b] = current_wave

    # Between consecutive pivots
    for p in range(len(pivot_idx) - 1):
        current_wave += 1
        seg_dir = 1 if pivot_type[p] == -1 else -1  # from low→high = up wave
        for b in range(pivot_idx[p] + 1, pivot_idx[p + 1] + 1):
            wave_dir[b] = seg_dir
            wave_id[b] = current_wave

    # After last pivot
    current_wave += 1
    last_dir = 1 if pivot_type[-1] == -1 else -1
    for b in range(pivot_idx[-1] + 1, n):
        wave_dir[b] = last_dir
        wave_id[b] = current_wave

    # --- Pass 3: running wave aggregates ----------------------------
    wave_high = np.empty(n, dtype=np.float64)
    wave_low = np.empty(n, dtype=np.float64)
    wave_vol = np.empty(n, dtype=np.float64)
    wave_delta = np.zeros(n, dtype=np.float64)

    seg_start = 0
    for i in range(n):
        if i > 0 and wave_id[i] != wave_id[i - 1]:
            seg_start = i
        wave_high[i] = high[seg_start:i + 1].max()
        wave_low[i] = low[seg_start:i + 1].min()
        wave_vol[i] = volume[seg_start:i + 1].sum()

    return {
        "wave_dir": wave_dir,
        "wave_id": wave_id,
        "wave_high": wave_high,
        "wave_low": wave_low,
        "wave_vol": wave_vol,
        "wave_delta": wave_delta,
    }


def _simple_zigzag_waves(df: pd.DataFrame, pct_reversal: float = 0.005) -> dict:
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    n = len(close)

    wave_dir = np.ones(n, dtype=np.int8)
    wave_id = np.zeros(n, dtype=np.int32)

    direction = 1
    last_pivot = close[0]
    current_wave = 0

    for i in range(1, n):
        if direction == 1:
            if close[i] > last_pivot:
                last_pivot = close[i]
            elif (last_pivot - close[i]) / max(last_pivot, 1e-6) > pct_reversal:
                direction = -1
                current_wave += 1
                last_pivot = close[i]
        else:
            if close[i] < last_pivot:
                last_pivot = close[i]
            elif (close[i] - last_pivot) / max(last_pivot, 1e-6) > pct_reversal:
                direction = 1
                current_wave += 1
                last_pivot = close[i]

        wave_dir[i] = direction
        wave_id[i] = current_wave

    wave_high = np.empty(n, dtype=np.float64)
    wave_low = np.empty(n, dtype=np.float64)
    wave_vol = np.empty(n, dtype=np.float64)
    wave_delta = np.zeros(n, dtype=np.float64)

    seg_start = 0
    for i in range(n):
        if i > 0 and wave_id[i] != wave_id[i - 1]:
            seg_start = i
        wave_high[i] = high[seg_start:i + 1].max()
        wave_low[i] = low[seg_start:i + 1].min()
        wave_vol[i] = volume[seg_start:i + 1].sum()

    return {
        "wave_dir": wave_dir,
        "wave_id": wave_id,
        "wave_high": wave_high,
        "wave_low": wave_low,
        "wave_vol": wave_vol,
        "wave_delta": wave_delta,
    }


def build_wave_table(df: pd.DataFrame, waves: dict) -> pd.DataFrame:
    wave_id = np.asarray(waves["wave_id"], dtype=np.int64)
    wave_dir = np.asarray(waves["wave_dir"], dtype=np.int64)

    rows = []
    seg_start = 0

    for i in range(1, len(df) + 1):
        if i == len(df) or wave_id[i] != wave_id[seg_start]:
            seg = df.iloc[seg_start:i]

            direction = 1 if int(wave_dir[i - 1]) > 0 else -1
            open_ = float(seg["open"].iat[0])
            close_ = float(seg["close"].iat[-1])
            low_ = float(seg["low"].min())
            high_ = float(seg["high"].max())
            vol_ = float(seg["volume"].sum())
            delta_ = float(seg["delta"].sum()) if "delta" in seg.columns else 0.0
            dur_ = float(seg["duration_seconds"].sum()) if "duration_seconds" in seg.columns else 0.0
            disp_ = abs(close_ - open_)

            rows.append({
                "wave_idx": len(rows),
                "wave_id": int(wave_id[seg_start]),
                "start_idx": int(seg_start),
                "end_idx": int(i - 1),
                "direction": int(direction),
                "open": open_,
                "close": close_,
                "low": low_,
                "high": high_,
                "volume": vol_,
                "delta": delta_,
                "duration": dur_,
                "displacement": disp_,
                "effort_result": vol_ / max(disp_, 1e-6),
            })
            seg_start = i

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Structural scan
# ═══════════════════════════════════════════════════════════════════════

def _climax_conf(wave_df: pd.DataFrame, i: int, side: RangeSide, cfg: WyckoffLabelConfig) -> float:
    trend = _trend_score(wave_df, i, side, cfg.trend_lookback_waves)
    vol_rank = _same_dir_rank(wave_df, i, "volume", cfg.climax_lookback_waves)
    disp_rank = _same_dir_rank(wave_df, i, "displacement", cfg.climax_lookback_waves)
    return _clip01(0.35 * trend + 0.40 * vol_rank + 0.25 * disp_rank)


def _is_tentative_sc(wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> bool:
    if int(wave_df["direction"].iat[i]) != -1 or i < 4:
        return False
    conf = _climax_conf(wave_df, i, RangeSide.ACCUM, cfg)
    vol_rank = _same_dir_rank(wave_df, i, "volume", cfg.climax_lookback_waves)
    disp_rank = _same_dir_rank(wave_df, i, "displacement", cfg.climax_lookback_waves)
    return conf >= cfg.climax_tentative_gate and (vol_rank >= cfg.climax_vol_pct or disp_rank >= cfg.climax_disp_pct)


def _is_tentative_bc(wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> bool:
    if int(wave_df["direction"].iat[i]) != 1 or i < 4:
        return False
    conf = _climax_conf(wave_df, i, RangeSide.DISTR, cfg)
    vol_rank = _same_dir_rank(wave_df, i, "volume", cfg.climax_lookback_waves)
    disp_rank = _same_dir_rank(wave_df, i, "displacement", cfg.climax_lookback_waves)
    return conf >= cfg.climax_tentative_gate and (vol_rank >= cfg.climax_vol_pct or disp_rank >= cfg.climax_disp_pct)


def _response_ratio(ro: RangeObject, w: pd.Series) -> float:
    wave_retrace = float(w["displacement"]) / max(ro.climax_disp, 1e-6)
    bar_retrace = (
        (float(w["high"]) - ro.climax_low) / max(ro.climax_high - ro.climax_low, 1e-6)
        if ro.side == RangeSide.ACCUM
        else (ro.climax_high - float(w["low"])) / max(ro.climax_high - ro.climax_low, 1e-6)
    )
    return max(wave_retrace, bar_retrace)


def _automatic_response_conf(wave_df: pd.DataFrame, i: int, ro: RangeObject, cfg: WyckoffLabelConfig) -> float:
    resp = _clip01(_response_ratio(ro, wave_df.iloc[i]))
    vol_rank = _same_dir_rank(wave_df, i, "volume", cfg.vol_rank_lookback)
    return _clip01(0.65 * resp + 0.35 * vol_rank)


def _is_automatic_rally(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> bool:
    if i <= ro.climax_wave:
        return False
    w = wave_df.iloc[i]
    return int(w["direction"]) == 1 and _response_ratio(ro, w) >= cfg.ar_min_response_ratio


def _is_automatic_reaction(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> bool:
    if i <= ro.climax_wave:
        return False
    w = wave_df.iloc[i]
    return int(w["direction"]) == -1 and _response_ratio(ro, w) >= cfg.ar_min_response_ratio


def _touch_boundaries(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> tuple[bool, bool]:
    if not np.isfinite(ro.width) or ro.width <= 0:
        return False, False

    w = wave_df.iloc[i]
    tol = ro.width * cfg.st_tolerance_frac

    support_touch = (
        int(w["direction"]) == -1 and
        (ro.support - ro.width * cfg.spring_max_break_frac) <= float(w["low"]) <= (ro.support + tol)
    )
    resistance_touch = (
        int(w["direction"]) == 1 and
        (ro.resistance - tol) <= float(w["high"]) <= (ro.resistance + ro.width * cfg.spring_max_break_frac)
    )
    return support_touch, resistance_touch


def _range_maturity(ro: RangeObject, current_wave: int, cfg: WyckoffLabelConfig) -> float:
    if ro.response_wave is None:
        return 0.0
    s_score = min(len(ro.support_tests), cfg.min_range_tests) / max(cfg.min_range_tests, 1)
    r_score = min(len(ro.resistance_tests), cfg.min_range_tests) / max(cfg.min_range_tests, 1)
    age_score = _clip01((current_wave - ro.response_wave + 1) / 4.0)
    return _clip01(0.40 * s_score + 0.40 * r_score + 0.20 * age_score)


def _support_test_conf(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> float | None:
    if not np.isfinite(ro.width) or ro.width <= 0:
        return None
    w = wave_df.iloc[i]
    if int(w["direction"]) != -1:
        return None

    dist = abs(float(w["low"]) - ro.support) / ro.width
    break_depth = (ro.support - float(w["low"])) / ro.width
    within = (dist <= cfg.st_tolerance_frac) or (0.0 <= break_depth <= cfg.spring_min_break_frac)
    if not within:
        return None

    vol_ratio = float(w["volume"]) / max(ro.climax_vol, 1e-6)
    if vol_ratio > cfg.test_vol_max_ratio:
        return None

    prox_score = _clip01(1.0 - dist / max(cfg.st_tolerance_frac, 1e-6))
    vol_score = _clip01(1.0 - vol_ratio / max(cfg.test_vol_max_ratio, 1e-6))
    reject_score = _clip01((float(w["close"]) - float(w["low"])) / max(float(w["high"]) - float(w["low"]), 1e-6))
    return _clip01(0.35 * prox_score + 0.35 * vol_score + 0.20 * reject_score + 0.10 * ro.maturity)


def _resistance_test_conf(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> float | None:
    if not np.isfinite(ro.width) or ro.width <= 0:
        return None
    w = wave_df.iloc[i]
    if int(w["direction"]) != 1:
        return None

    dist = abs(float(w["high"]) - ro.resistance) / ro.width
    break_depth = (float(w["high"]) - ro.resistance) / ro.width
    within = (dist <= cfg.st_tolerance_frac) or (0.0 <= break_depth <= cfg.spring_min_break_frac)
    if not within:
        return None

    vol_ratio = float(w["volume"]) / max(ro.climax_vol, 1e-6)
    if vol_ratio > cfg.test_vol_max_ratio:
        return None

    prox_score = _clip01(1.0 - dist / max(cfg.st_tolerance_frac, 1e-6))
    vol_score = _clip01(1.0 - vol_ratio / max(cfg.test_vol_max_ratio, 1e-6))
    reject_score = _clip01((float(w["high"]) - float(w["close"])) / max(float(w["high"]) - float(w["low"]), 1e-6))
    return _clip01(0.35 * prox_score + 0.35 * vol_score + 0.20 * reject_score + 0.10 * ro.maturity)


def _confirm_spring(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> float:
    nxt = wave_df.iloc[i + 1:i + 1 + cfg.confirm_waves]
    if len(nxt) == 0 or not np.isfinite(ro.width) or ro.width <= 0:
        return 0.0

    held_support = float(nxt["low"].min() > ro.support - ro.width * cfg.hold_invalid_frac)
    base = max(ro.support, float(wave_df["close"].iat[i]))
    progress = (float(nxt["high"].max()) - base) / ro.width
    progress_score = _clip01(progress / max(cfg.confirm_breakout_frac, 1e-6))
    return _clip01(0.50 * held_support + 0.50 * progress_score)


def _confirm_upthrust(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> float:
    nxt = wave_df.iloc[i + 1:i + 1 + cfg.confirm_waves]
    if len(nxt) == 0 or not np.isfinite(ro.width) or ro.width <= 0:
        return 0.0

    held_resistance = float(nxt["high"].max() < ro.resistance + ro.width * cfg.hold_invalid_frac)
    base = min(ro.resistance, float(wave_df["close"].iat[i]))
    progress = (base - float(nxt["low"].min())) / ro.width
    progress_score = _clip01(progress / max(cfg.confirm_breakout_frac, 1e-6))
    return _clip01(0.50 * held_resistance + 0.50 * progress_score)

def _wave_close_location(w: pd.Series, bullish: bool) -> float:
    span = max(float(w["high"]) - float(w["low"]), 1e-6)
    if bullish:
        return _clip01((float(w["close"]) - float(w["low"])) / span)
    return _clip01((float(w["high"]) - float(w["close"])) / span)


def _breakout_followthrough_conf(
    ro: RangeObject,
    wave_df: pd.DataFrame,
    i: int,
    cfg: WyckoffLabelConfig,
    bullish: bool,
) -> float:
    nxt = wave_df.iloc[i + 1:i + 1 + cfg.confirm_waves]
    if len(nxt) == 0 or not np.isfinite(ro.width) or ro.width <= 0:
        return 0.0

    if bullish:
        hold_level = ro.resistance - ro.width * cfg.breakout_hold_frac
        held = float(nxt["low"].min() >= hold_level)
        base = max(ro.resistance, float(wave_df["close"].iat[i]))
        progress = (float(nxt["high"].max()) - base) / ro.width
        dir_frac = float((nxt["direction"] == 1).mean())
    else:
        hold_level = ro.support + ro.width * cfg.breakout_hold_frac
        held = float(nxt["high"].max() <= hold_level)
        base = min(ro.support, float(wave_df["close"].iat[i]))
        progress = (base - float(nxt["low"].min())) / ro.width
        dir_frac = float((nxt["direction"] == -1).mean())

    progress_score = _clip01(progress / max(cfg.breakout_followthrough_frac, 1e-6))
    return _clip01(0.45 * held + 0.35 * progress_score + 0.20 * dir_frac)


def _resume_after_pullback_conf(
    ro: RangeObject,
    wave_df: pd.DataFrame,
    i: int,
    cfg: WyckoffLabelConfig,
    bullish: bool,
) -> float:
    nxt = wave_df.iloc[i + 1:i + 1 + cfg.lps_resume_waves]
    if len(nxt) == 0 or not np.isfinite(ro.width) or ro.width <= 0:
        return 0.0

    if bullish:
        hold_level = ro.resistance - ro.width * cfg.breakout_hold_frac
        held = float(nxt["low"].min() >= hold_level)
        base = max(float(wave_df["high"].iat[i]), ro.resistance)
        progress = (float(nxt["high"].max()) - base) / ro.width
        dir_frac = float((nxt["direction"] == 1).mean())
    else:
        hold_level = ro.support + ro.width * cfg.breakout_hold_frac
        held = float(nxt["high"].max() <= hold_level)
        base = min(float(wave_df["low"].iat[i]), ro.support)
        progress = (base - float(nxt["low"].min())) / ro.width
        dir_frac = float((nxt["direction"] == -1).mean())

    progress_score = _clip01(progress / max(cfg.lps_resume_frac, 1e-6))
    return _clip01(0.40 * held + 0.40 * progress_score + 0.20 * dir_frac)

def _spring_conf(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> float | None:
    if ro.maturity < cfg.mature_threshold or not np.isfinite(ro.width) or ro.width <= 0:
        return None

    w = wave_df.iloc[i]
    if int(w["direction"]) != -1:
        return None

    break_depth = (ro.support - float(w["low"])) / ro.width
    if not (cfg.spring_min_break_frac <= break_depth <= cfg.spring_max_break_frac):
        return None

    reclaimed = float(w["close"]) > ro.support
    next_reclaim = (
        i + 1 < len(wave_df) and
        int(wave_df["direction"].iat[i + 1]) == 1 and
        float(wave_df["high"].iat[i + 1]) > ro.support
    )
    if not (reclaimed or next_reclaim):
        return None

    vol_ratio = float(w["volume"]) / max(ro.climax_vol, 1e-6)
    if vol_ratio > cfg.spring_vol_max_ratio:
        return None

    prev_idx = None
    if len(ro.support_tests) >= 2 and ro.support_tests[-1] == i:
        prev_idx = ro.support_tests[-2]
    elif len(ro.support_tests) >= 1:
        prev_idx = ro.support_tests[-1]

    delta_div_score = 0.50
    if prev_idx is not None:
        prev_delta = abs(float(wave_df["delta"].iat[prev_idx]))
        curr_delta = abs(float(w["delta"]))
        if prev_delta > 1e-6:
            delta_div_score = _clip01(0.50 + 0.50 * (prev_delta - curr_delta) / prev_delta)

    center = 0.5 * (cfg.spring_min_break_frac + cfg.spring_max_break_frac)
    halfspan = 0.5 * (cfg.spring_max_break_frac - cfg.spring_min_break_frac) + 1e-6
    break_depth_score = _clip01(1.0 - abs(break_depth - center) / halfspan)

    reclaim_score = 1.0 if reclaimed else 0.70
    vol_contract_score = _clip01(1.0 - vol_ratio / max(cfg.spring_vol_max_ratio, 1e-6))
    confirm_score = _confirm_spring(ro, wave_df, i, cfg)

    return _clip01(
        0.15 * ro.maturity +
        0.20 * break_depth_score +
        0.20 * reclaim_score +
        0.15 * vol_contract_score +
        0.10 * delta_div_score +
        0.20 * confirm_score
    )


def _upthrust_conf(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> float | None:
    if ro.maturity < cfg.mature_threshold or not np.isfinite(ro.width) or ro.width <= 0:
        return None

    w = wave_df.iloc[i]
    if int(w["direction"]) != 1:
        return None

    break_depth = (float(w["high"]) - ro.resistance) / ro.width
    if not (cfg.spring_min_break_frac <= break_depth <= cfg.spring_max_break_frac):
        return None

    rejected = float(w["close"]) < ro.resistance
    next_reject = (
        i + 1 < len(wave_df) and
        int(wave_df["direction"].iat[i + 1]) == -1 and
        float(wave_df["low"].iat[i + 1]) < ro.resistance
    )
    if not (rejected or next_reject):
        return None

    vol_ratio = float(w["volume"]) / max(ro.climax_vol, 1e-6)
    if vol_ratio > cfg.spring_vol_max_ratio:
        return None

    prev_idx = None
    if len(ro.resistance_tests) >= 2 and ro.resistance_tests[-1] == i:
        prev_idx = ro.resistance_tests[-2]
    elif len(ro.resistance_tests) >= 1:
        prev_idx = ro.resistance_tests[-1]

    delta_div_score = 0.50
    if prev_idx is not None:
        prev_delta = abs(float(wave_df["delta"].iat[prev_idx]))
        curr_delta = abs(float(w["delta"]))
        if prev_delta > 1e-6:
            delta_div_score = _clip01(0.50 + 0.50 * (prev_delta - curr_delta) / prev_delta)

    center = 0.5 * (cfg.spring_min_break_frac + cfg.spring_max_break_frac)
    halfspan = 0.5 * (cfg.spring_max_break_frac - cfg.spring_min_break_frac) + 1e-6
    break_depth_score = _clip01(1.0 - abs(break_depth - center) / halfspan)

    reject_score = 1.0 if rejected else 0.70
    vol_contract_score = _clip01(1.0 - vol_ratio / max(cfg.spring_vol_max_ratio, 1e-6))
    confirm_score = _confirm_upthrust(ro, wave_df, i, cfg)

    return _clip01(
        0.15 * ro.maturity +
        0.20 * break_depth_score +
        0.20 * reject_score +
        0.15 * vol_contract_score +
        0.10 * delta_div_score +
        0.20 * confirm_score
    )

def _sos_conf(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> float | None:
    if not np.isfinite(ro.width) or ro.width <= 0:
        return None
    # Allow breakout from POST_TERMINAL or MATURE ranges
    if ro.terminal_wave is not None:
        anchor = ro.terminal_wave
        window = cfg.post_terminal_breakout_window
    elif ro.response_wave is not None:
        anchor = ro.response_wave
        window = cfg.mature_breakout_window
    else:
        return None
    if i <= anchor:
        return None
    if i > anchor + window:
        return None

    w = wave_df.iloc[i]
    if int(w["direction"]) != 1:
        return None

    breakout = (float(w["high"]) - ro.resistance) / ro.width
    close_through = (float(w["close"]) - ro.resistance) / ro.width
    if breakout < cfg.breakout_min_frac:
        return None
    if close_through < cfg.breakout_close_min_frac:
        return None

    close_loc = _wave_close_location(w, bullish=True)
    vol_rank = _same_dir_rank(wave_df, i, "volume", cfg.vol_rank_lookback)
    disp_rank = _same_dir_rank(wave_df, i, "displacement", cfg.vol_rank_lookback)

    if vol_rank < cfg.breakout_vol_rank_min and disp_rank < cfg.breakout_disp_rank_min:
        return None

    follow_score = _breakout_followthrough_conf(ro, wave_df, i, cfg, bullish=True)
    if follow_score < cfg.sos_follow_min:
        return None

    break_score = _clip01(breakout / max(cfg.breakout_min_frac, 1e-6))
    close_score = _clip01(close_through / max(cfg.breakout_close_min_frac, 1e-6))

    return _clip01(
        0.20 * break_score +
        0.25 * close_score +
        0.15 * close_loc +
        0.15 * vol_rank +
        0.10 * disp_rank +
        0.15 * follow_score
    )


def _sow_conf(ro: RangeObject, wave_df: pd.DataFrame, i: int, cfg: WyckoffLabelConfig) -> float | None:
    if not np.isfinite(ro.width) or ro.width <= 0:
        return None
    # Allow breakout from POST_TERMINAL or MATURE ranges
    if ro.terminal_wave is not None:
        anchor = ro.terminal_wave
        window = cfg.post_terminal_breakout_window
    elif ro.response_wave is not None:
        anchor = ro.response_wave
        window = cfg.mature_breakout_window
    else:
        return None
    if i <= anchor:
        return None
    if i > anchor + window:
        return None

    w = wave_df.iloc[i]
    if int(w["direction"]) != -1:
        return None

    breakout = (ro.support - float(w["low"])) / ro.width
    close_through = (ro.support - float(w["close"])) / ro.width
    if breakout < cfg.breakout_min_frac:
        return None
    if close_through < cfg.breakout_close_min_frac:
        return None

    close_loc = _wave_close_location(w, bullish=False)
    vol_rank = _same_dir_rank(wave_df, i, "volume", cfg.vol_rank_lookback)
    disp_rank = _same_dir_rank(wave_df, i, "displacement", cfg.vol_rank_lookback)

    if vol_rank < cfg.breakout_vol_rank_min and disp_rank < cfg.breakout_disp_rank_min:
        return None

    follow_score = _breakout_followthrough_conf(ro, wave_df, i, cfg, bullish=False)
    if follow_score < cfg.sow_follow_min:
        return None

    break_score = _clip01(breakout / max(cfg.breakout_min_frac, 1e-6))
    close_score = _clip01(close_through / max(cfg.breakout_close_min_frac, 1e-6))

    return _clip01(
        0.20 * break_score +
        0.25 * close_score +
        0.15 * close_loc +
        0.15 * vol_rank +
        0.10 * disp_rank +
        0.15 * follow_score
    )


def _has_live_side(active_ranges: list[RangeObject], side: RangeSide) -> bool:
    live = {
        RangeStatus.TENTATIVE_CLIMAX,
        RangeStatus.SEEK_TESTS,
        RangeStatus.MATURE,
        RangeStatus.POST_TERMINAL,
    }
    return any(ro.side == side and ro.status in live for ro in active_ranges)

def _can_spawn_range(active_ranges: list[RangeObject], side: RangeSide) -> bool:
    if _has_live_side(active_ranges, side):
        return False

    opposite = RangeSide.DISTR if side == RangeSide.ACCUM else RangeSide.ACCUM
    blocking_status = {
        RangeStatus.MATURE,
        RangeStatus.POST_TERMINAL,
    }

    return not any(
        ro.side == opposite and ro.status in blocking_status
        for ro in active_ranges
    )

def _scan_range_structures(
    df: pd.DataFrame,
    wave_df: pd.DataFrame,
    cfg: WyckoffLabelConfig,
    events: np.ndarray,
    event_weight: np.ndarray,
) -> list[RangeObject]:
    active_ranges: list[RangeObject] = []
    finished_ranges: list[RangeObject] = []

    for i in range(len(wave_df)):
        w = wave_df.iloc[i]

        updated: list[RangeObject] = []

        for ro in active_ranges:
            if i < ro.climax_wave:
                updated.append(ro)
                continue

            if i - ro.climax_wave > cfg.max_range_waves:
                # POST_TERMINAL / MATURE ranges get extra time for SOS/SOW breakout
                if ro.status == RangeStatus.POST_TERMINAL and ro.terminal_wave is not None and i - ro.terminal_wave <= cfg.post_terminal_breakout_window:
                    pass  # don't expire yet
                elif ro.status == RangeStatus.MATURE and ro.response_wave is not None and i - ro.response_wave <= cfg.mature_breakout_window:
                    pass  # give MATURE ranges a breakout window too
                else:
                    ro.status = RangeStatus.EXPIRED
                    ro.end_bar = int(w["end_idx"])
                    finished_ranges.append(ro)
                    continue

            # 1) Need automatic response before we trust the climax
            if ro.status == RangeStatus.TENTATIVE_CLIMAX:
                if ro.side == RangeSide.ACCUM and _is_automatic_rally(ro, wave_df, i, cfg):
                    ro.response_wave = i
                    ro.response_bar = int(w["end_idx"])
                    ro.support = ro.climax_low
                    ro.resistance = float(w["high"])
                    ro.width = ro.resistance - ro.support
                    ro.status = RangeStatus.SEEK_TESTS

                    _mark_event(events, event_weight, ro.climax_bar, "selling_climax", _climax_conf(wave_df, ro.climax_wave, RangeSide.ACCUM, cfg))
                    _mark_event(events, event_weight, ro.response_bar, "automatic_rally", _automatic_response_conf(wave_df, i, ro, cfg))

                    updated.append(ro)
                    continue

                if ro.side == RangeSide.DISTR and _is_automatic_reaction(ro, wave_df, i, cfg):
                    ro.response_wave = i
                    ro.response_bar = int(w["end_idx"])
                    ro.resistance = ro.climax_high
                    ro.support = float(w["low"])
                    ro.width = ro.resistance - ro.support
                    ro.status = RangeStatus.SEEK_TESTS

                    _mark_event(events, event_weight, ro.climax_bar, "buying_climax", _climax_conf(wave_df, ro.climax_wave, RangeSide.DISTR, cfg))
                    _mark_event(events, event_weight, ro.response_bar, "automatic_reaction", _automatic_response_conf(wave_df, i, ro, cfg))

                    updated.append(ro)
                    continue

                updated.append(ro)
                continue

            # 2) Update touch counts and maturity
            if ro.response_wave is not None and i > ro.response_wave:
                support_touch, resistance_touch = _touch_boundaries(ro, wave_df, i, cfg)
                added_support = support_touch and _append_with_sep(ro.support_tests, i, min_sep=2)
                added_resistance = resistance_touch and _append_with_sep(ro.resistance_tests, i, min_sep=2)

                ro.maturity = _range_maturity(ro, i, cfg)

                if ro.side == RangeSide.ACCUM and added_support:
                    conf = _support_test_conf(ro, wave_df, i, cfg)
                    if conf is not None and conf >= cfg.st_conf_gate:
                        _mark_event(events, event_weight, int(w["end_idx"]), "st_support", conf)

                if ro.side == RangeSide.DISTR and added_resistance:
                    conf = _resistance_test_conf(ro, wave_df, i, cfg)
                    if conf is not None and conf >= cfg.st_conf_gate:
                        _mark_event(events, event_weight, int(w["end_idx"]), "st_resistance", conf)

                if ro.status == RangeStatus.SEEK_TESTS and ro.maturity >= cfg.mature_threshold:
                    ro.status = RangeStatus.MATURE

            # 3) Spring / Upthrust
            if ro.status == RangeStatus.MATURE:
                if ro.side == RangeSide.ACCUM:
                    conf = _spring_conf(ro, wave_df, i, cfg)
                    if conf is not None:
                        ro.terminal_wave = i
                        if conf >= cfg.min_event_confidence:
                            ro.status = RangeStatus.POST_TERMINAL
                            _mark_event(events, event_weight, int(w["end_idx"]), "spring", conf)
                            updated.append(ro)
                            continue
                        if conf >= cfg.failed_terminal_gate:
                            ro.status = RangeStatus.FAILED
                            ro.end_bar = int(w["end_idx"])
                            _mark_event(events, event_weight, int(w["end_idx"]), "failed_spring", conf)
                            finished_ranges.append(ro)
                            continue

                if ro.side == RangeSide.DISTR:
                    conf = _upthrust_conf(ro, wave_df, i, cfg)
                    if conf is not None:
                        ro.terminal_wave = i
                        if conf >= cfg.min_event_confidence:
                            ro.status = RangeStatus.POST_TERMINAL
                            _mark_event(events, event_weight, int(w["end_idx"]), "upthrust", conf)
                            updated.append(ro)
                            continue
                        if conf >= cfg.failed_terminal_gate:
                            ro.status = RangeStatus.FAILED
                            ro.end_bar = int(w["end_idx"])
                            _mark_event(events, event_weight, int(w["end_idx"]), "failed_upthrust", conf)
                            finished_ranges.append(ro)
                            continue

            # 4) SOS / SOW
            # SOS can fire from POST_TERMINAL or MATURE (matching SOW behavior)
            if ro.status in (RangeStatus.POST_TERMINAL, RangeStatus.MATURE):
                if ro.side == RangeSide.ACCUM:
                    conf = _sos_conf(ro, wave_df, i, cfg)
                    if conf is not None and conf >= cfg.min_breakout_confidence:
                        ro.breakout_wave = i
                        ro.status = RangeStatus.RESOLVED
                        ro.end_bar = int(w["end_idx"])
                        _mark_event(events, event_weight, int(w["end_idx"]), "sos", conf)
                        finished_ranges.append(ro)
                        continue

            if ro.status in (RangeStatus.POST_TERMINAL, RangeStatus.MATURE):
                if ro.side == RangeSide.DISTR:
                    conf = _sow_conf(ro, wave_df, i, cfg)
                    if conf is not None and conf >= cfg.min_breakout_confidence:
                        ro.breakout_wave = i
                        ro.status = RangeStatus.RESOLVED
                        ro.end_bar = int(w["end_idx"])
                        _mark_event(events, event_weight, int(w["end_idx"]), "sow", conf)
                        finished_ranges.append(ro)
                        continue

            updated.append(ro)

        active_ranges = updated

        # Spawn AFTER processing existing ranges
        if _is_tentative_sc(wave_df, i, cfg) and _can_spawn_range(active_ranges, RangeSide.ACCUM):
            active_ranges.append(
                RangeObject(
                    side=RangeSide.ACCUM,
                    climax_wave=i,
                    climax_bar=int(w["end_idx"]),
                    climax_price=float(w["low"]),
                    climax_vol=float(w["volume"]),
                    climax_high=float(w["high"]),
                    climax_low=float(w["low"]),
                    climax_disp=float(w["displacement"]),
                    climax_delta=float(w["delta"]),
                    start_bar=int(w["start_idx"]),
                )
            )

        if _is_tentative_bc(wave_df, i, cfg) and _can_spawn_range(active_ranges, RangeSide.DISTR):
            active_ranges.append(
                RangeObject(
                    side=RangeSide.DISTR,
                    climax_wave=i,
                    climax_bar=int(w["end_idx"]),
                    climax_price=float(w["high"]),
                    climax_vol=float(w["volume"]),
                    climax_high=float(w["high"]),
                    climax_low=float(w["low"]),
                    climax_disp=float(w["displacement"]),
                    climax_delta=float(w["delta"]),
                    start_bar=int(w["start_idx"]),
                )
            )

    last_bar = int(df.index.size - 1)
    for ro in active_ranges:
        if ro.end_bar is None:
            ro.end_bar = last_bar
        finished_ranges.append(ro)

    return finished_ranges

def _post_label_continuation_events(
    wave_df: pd.DataFrame,
    structures: list[RangeObject],
    cfg: WyckoffLabelConfig,
    events: np.ndarray,
    event_weight: np.ndarray,
):
    for ro in structures:
        if ro.breakout_wave is None or not np.isfinite(ro.width) or ro.width <= 0:
            continue

        breakout_w = wave_df.iloc[ro.breakout_wave]
        breakout_vol = float(breakout_w["volume"])
        breakout_disp = float(breakout_w["displacement"]) + 1e-6

        search_start = ro.breakout_wave + cfg.lps_min_waves_after_breakout
        search_end = min(len(wave_df), ro.breakout_wave + 1 + cfg.lps_search_waves)

        if ro.side == RangeSide.ACCUM:
            breakout_high = float(breakout_w["high"])
            hold_level = ro.resistance - ro.width * cfg.lps_tolerance_frac
            close_hold_level = ro.resistance - ro.width * cfg.lps_close_hold_frac

            for j in range(search_start, search_end):
                w = wave_df.iloc[j]
                if int(w["direction"]) != -1:
                    continue

                low = float(w["low"])
                close = float(w["close"])

                if low < hold_level:
                    continue
                if close < close_hold_level:
                    continue

                retrace = (breakout_high - low) / ro.width
                if retrace <= 0.0 or retrace > cfg.lps_max_retrace_frac:
                    continue

                vol_ratio = float(w["volume"]) / max(breakout_vol, 1e-6)
                if vol_ratio > cfg.lps_vol_max_ratio:
                    continue

                disp_ratio = float(w["displacement"]) / breakout_disp
                if disp_ratio > cfg.lps_max_disp_ratio:
                    continue

                resume_score = _resume_after_pullback_conf(ro, wave_df, j, cfg, bullish=True)
                if resume_score < cfg.lps_resume_min:
                    continue

                hold_score = _clip01((low - hold_level) / max(ro.width * cfg.lps_tolerance_frac, 1e-6))
                close_hold_score = _clip01((close - close_hold_level) / max(ro.width * cfg.lps_close_hold_frac, 1e-6))
                close_loc_score = _wave_close_location(w, bullish=True)
                vol_score = _clip01(1.0 - vol_ratio / max(cfg.lps_vol_max_ratio, 1e-6))
                shallow_score = _clip01(1.0 - retrace / max(cfg.lps_max_retrace_frac, 1e-6))
                disp_score = _clip01(1.0 - disp_ratio / max(cfg.lps_max_disp_ratio, 1e-6))

                conf = _clip01(
                    0.18 * hold_score +
                    0.14 * close_hold_score +
                    0.12 * close_loc_score +
                    0.16 * vol_score +
                    0.10 * shallow_score +
                    0.10 * disp_score +
                    0.20 * resume_score
                )

                if conf >= cfg.lps_conf_gate:
                    ro.continuation_wave = j
                    _mark_event(events, event_weight, int(w["end_idx"]), "lps", conf)
                    break

        if ro.side == RangeSide.DISTR:
            breakout_low = float(breakout_w["low"])
            fail_level = ro.support + ro.width * cfg.lps_tolerance_frac
            close_fail_level = ro.support + ro.width * cfg.lps_close_hold_frac

            for j in range(search_start, search_end):
                w = wave_df.iloc[j]
                if int(w["direction"]) != 1:
                    continue

                high = float(w["high"])
                close = float(w["close"])

                if high > fail_level:
                    continue
                if close > close_fail_level:
                    continue

                retrace = (high - breakout_low) / ro.width
                if retrace <= 0.0 or retrace > cfg.lps_max_retrace_frac:
                    continue

                vol_ratio = float(w["volume"]) / max(breakout_vol, 1e-6)
                if vol_ratio > cfg.lps_vol_max_ratio:
                    continue

                disp_ratio = float(w["displacement"]) / breakout_disp
                if disp_ratio > cfg.lps_max_disp_ratio:
                    continue

                resume_score = _resume_after_pullback_conf(ro, wave_df, j, cfg, bullish=False)
                if resume_score < cfg.lps_resume_min:
                    continue

                hold_score = _clip01((fail_level - high) / max(ro.width * cfg.lps_tolerance_frac, 1e-6))
                close_hold_score = _clip01((close_fail_level - close) / max(ro.width * cfg.lps_close_hold_frac, 1e-6))
                close_loc_score = _wave_close_location(w, bullish=False)
                vol_score = _clip01(1.0 - vol_ratio / max(cfg.lps_vol_max_ratio, 1e-6))
                shallow_score = _clip01(1.0 - retrace / max(cfg.lps_max_retrace_frac, 1e-6))
                disp_score = _clip01(1.0 - disp_ratio / max(cfg.lps_max_disp_ratio, 1e-6))

                conf = _clip01(
                    0.18 * hold_score +
                    0.14 * close_hold_score +
                    0.12 * close_loc_score +
                    0.16 * vol_score +
                    0.10 * shallow_score +
                    0.10 * disp_score +
                    0.20 * resume_score
                )

                if conf >= cfg.lps_conf_gate:
                    ro.continuation_wave = j
                    _mark_event(events, event_weight, int(w["end_idx"]), "lpsy", conf)
                    break


# ═══════════════════════════════════════════════════════════════════════
# Phase painting
# ═══════════════════════════════════════════════════════════════════════

def _paint_range_phases(
    phase: np.ndarray,
    phase_weight: np.ndarray,
    wave_df: pd.DataFrame,
    structures: list[RangeObject],
    cfg: WyckoffLabelConfig,
):
    accum_idx = REGIME_TO_IDX["accumulation"]
    distrib_idx = REGIME_TO_IDX["distribution"]

    for ro in structures:
        if ro.response_bar is None or ro.maturity < cfg.mature_threshold:
            continue

        end_bar = ro.end_bar
        if ro.breakout_wave is not None:
            end_bar = _wave_end_bar(wave_df, ro.breakout_wave)
        elif ro.terminal_wave is not None:
            end_bar = _wave_end_bar(wave_df, ro.terminal_wave)

        label = accum_idx if ro.side == RangeSide.ACCUM else distrib_idx
        weight = max(0.65, min(0.95, 0.55 + 0.35 * ro.maturity))
        _paint_span(phase, phase_weight, ro.response_bar, end_bar, label, weight)


def _paint_post_breakout_phases(
    phase: np.ndarray,
    phase_weight: np.ndarray,
    wave_df: pd.DataFrame,
    structures: list[RangeObject],
):
    markup_idx = REGIME_TO_IDX["markup"]
    markdown_idx = REGIME_TO_IDX["markdown"]

    for ro in structures:
        if ro.breakout_wave is None:
            continue

        start = _wave_end_bar(wave_df, ro.breakout_wave)
        end = _wave_end_bar(wave_df, ro.continuation_wave) if ro.continuation_wave is not None else ro.end_bar
        label = markup_idx if ro.side == RangeSide.ACCUM else markdown_idx
        _paint_span(phase, phase_weight, start, end, label, 0.85)


def _label_trend_background(
    df: pd.DataFrame,
    waves: dict,
    cfg: WyckoffLabelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Background trend labels:
    - markdown
    - markup
    - ignore otherwise

    Range phases later override this.
    """
    n = len(df)
    labels = np.full(n, PHASE_IGNORE_INDEX, dtype=np.int64)
    weights = np.zeros(n, dtype=np.float32)

    wave_id = waves["wave_id"]
    wave_dir = waves["wave_dir"]
    wave_high = waves["wave_high"]
    wave_low = waves["wave_low"]

    sh_prices, sl_prices, sh_bars, sl_bars = _extract_pivots(wave_id, wave_dir, wave_high, wave_low, cfg.min_wave_bars)

    trend_window = cfg.phase_trend_window
    if len(sh_prices) < trend_window or len(sl_prices) < trend_window:
        return labels, weights

    sh_idx = 0
    sl_idx = 0
    tw = trend_window - 1

    for i in range(n):
        while sh_idx < len(sh_bars) and sh_bars[sh_idx] <= i:
            sh_idx += 1
        while sl_idx < len(sl_bars) and sl_bars[sl_idx] <= i:
            sl_idx += 1

        if sh_idx < trend_window or sl_idx < trend_window:
            continue

        rh = sh_prices[sh_idx - trend_window:sh_idx]
        rl = sl_prices[sl_idx - trend_window:sl_idx]

        hh = sum(1 for j in range(1, len(rh)) if rh[j] > rh[j - 1])
        lh = sum(1 for j in range(1, len(rh)) if rh[j] < rh[j - 1])
        hl = sum(1 for j in range(1, len(rl)) if rl[j] > rl[j - 1])
        ll = sum(1 for j in range(1, len(rl)) if rl[j] < rl[j - 1])

        up_score = 0.50 * (hh / max(tw, 1)) + 0.50 * (hl / max(tw, 1))
        dn_score = 0.50 * (ll / max(tw, 1)) + 0.50 * (lh / max(tw, 1))

        if up_score >= cfg.trend_strong_threshold and up_score > dn_score + cfg.trend_gap_min:
            labels[i] = REGIME_TO_IDX["markup"]
            weights[i] = np.float32(up_score)

        elif dn_score >= cfg.trend_strong_threshold and dn_score > up_score + cfg.trend_gap_min:
            labels[i] = REGIME_TO_IDX["markdown"]
            weights[i] = np.float32(dn_score)

    return labels, weights


def _extract_pivots(wave_id, wave_dir, wave_high, wave_low, min_wave_bars: int = 3):
    n = len(wave_id)
    raw_waves = []
    seg_start = 0
    prev_wave = wave_id[0]

    for i in range(1, n):
        if wave_id[i] != prev_wave:
            wave_len = i - seg_start
            raw_waves.append((i - 1, wave_dir[i - 1] > 0, wave_high[i - 1], wave_low[i - 1], wave_len))
            seg_start = i
        prev_wave = wave_id[i]

    raw_waves.append((n - 1, wave_dir[n - 1] > 0, wave_high[n - 1], wave_low[n - 1], n - seg_start))

    significant = []
    for w in raw_waves:
        if w[4] >= min_wave_bars:
            significant.append(w)
        elif significant:
            prev = significant[-1]
            significant[-1] = (w[0], prev[1], max(prev[2], w[2]), min(prev[3], w[3]), prev[4] + w[4])

    swing_highs = []
    swing_lows = []
    for bar_idx, is_up, wh, wl, _ in significant:
        if is_up:
            swing_highs.append((bar_idx, wh))
        else:
            swing_lows.append((bar_idx, wl))

    sh_bars = np.array([s[0] for s in swing_highs], dtype=np.int64) if swing_highs else np.array([], dtype=np.int64)
    sh_prices = np.array([s[1] for s in swing_highs], dtype=np.float64) if swing_highs else np.array([], dtype=np.float64)
    sl_bars = np.array([s[0] for s in swing_lows], dtype=np.int64) if swing_lows else np.array([], dtype=np.int64)
    sl_prices = np.array([s[1] for s in swing_lows], dtype=np.float64) if swing_lows else np.array([], dtype=np.float64)

    return sh_prices, sl_prices, sh_bars, sl_bars


# ═══════════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════════

def save_labels(labels: dict, output_path: str):
    np.savez_compressed(
        output_path,
        phase=labels["phase"],
        phase_weight=labels["phase_weight"],
        events=labels["events"],
        event_weight=labels["event_weight"],
    )


def load_labels(label_path: str) -> dict:
    data = np.load(label_path, allow_pickle=True)

    phase = data["phase"]
    events = data["events"]

    if "phase_weight" in data.files:
        phase_weight = data["phase_weight"].astype(np.float32)
    else:
        phase_weight = np.where(phase != PHASE_IGNORE_INDEX, 1.0, 0.0).astype(np.float32)

    if "event_weight" in data.files:
        event_weight = data["event_weight"].astype(np.float32)
    else:
        event_weight = (events > 0).astype(np.float32)

    return {
        "phase": phase,
        "phase_weight": phase_weight,
        "events": events,
        "event_weight": event_weight,
    }


# ═══════════════════════════════════════════════════════════════════════
# Misc
# ═══════════════════════════════════════════════════════════════════════

def _verify_alignment(df: pd.DataFrame, npz_path: str):
    npz_data = np.load(npz_path, allow_pickle=True)
    npz_close = npz_data["close_ary"].flatten()
    parquet_close = df["close"].values

    if len(npz_close) != len(parquet_close):
        print(f"WARNING: NPZ has {len(npz_close)} bars, parquet has {len(parquet_close)}")
        return

    max_diff = np.abs(npz_close - parquet_close).max()
    if max_diff > 0.01:
        print(f"WARNING: NPZ/parquet close prices differ by up to {max_diff:.4f}")
