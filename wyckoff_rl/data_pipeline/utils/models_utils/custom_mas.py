"""
Custom Moving Averages — vectorised (numpy) implementations.

Adapted from srlcarlg/srl-python-indicators/custom_mas.py (Apache-2.0).
Provides multiple MA types that feed the Wyckoff strength filter.
"""

import numpy as np
from enum import Enum


class MAType(Enum):
    Simple = 1
    Exponential = 2
    Weighted = 3
    Triangular = 4
    Hull = 5
    VIDYA = 6             # Variable Index Dynamic Average (CMO-based)
    WilderSmoothing = 7
    KaufmanAdaptive = 8


# ──────────────────────────────────────────────────────────────────────────
#  Vectorised MA implementations (numpy)
# ──────────────────────────────────────────────────────────────────────────

def _sma_np(src: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    out = np.full_like(src, np.nan, dtype=np.float64)
    cumsum = np.cumsum(src)
    out[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0], cumsum[:-period]))) / period
    return out


def _ema_np(src: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(src, dtype=np.float64)
    out[0] = src[0]
    for i in range(1, len(src)):
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def _wma_np(src: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = weights.sum()
    out = np.full_like(src, np.nan, dtype=np.float64)
    for i in range(period - 1, len(src)):
        out[i] = np.dot(src[i - period + 1:i + 1], weights) / denom
    return out


def _tma_np(src: np.ndarray, period: int) -> np.ndarray:
    """Triangular Moving Average (SMA of SMA)."""
    sma1 = _sma_np(src, period)
    return _sma_np(np.nan_to_num(sma1, nan=0.0), period)


def _hull_np(src: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(1, period // 2)
    sqrt_p = max(1, int(np.sqrt(period)))
    wma_half = _wma_np(src, half)
    wma_full = _wma_np(src, period)
    diff = 2 * np.nan_to_num(wma_half) - np.nan_to_num(wma_full)
    return _wma_np(diff, sqrt_p)


def _vidya_np(src: np.ndarray, period: int) -> np.ndarray:
    """Variable Index Dynamic Average (Chande Momentum Oscillator based)."""
    out = np.empty_like(src, dtype=np.float64)
    out[0] = src[0]
    for i in range(1, len(src)):
        start = max(0, i - period + 1)
        window = src[start:i + 1]
        changes = np.diff(window)
        up = np.sum(changes[changes > 0])
        down = -np.sum(changes[changes < 0])
        total = up + down
        cmo = abs(up - down) / total if total != 0 else 0
        alpha = 2.0 / (period + 1) * cmo
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def _wilder_np(src: np.ndarray, period: int) -> np.ndarray:
    """Wilder Smoothing (RMA): alpha = 1/period."""
    alpha = 1.0 / period
    out = np.empty_like(src, dtype=np.float64)
    out[0] = src[0]
    for i in range(1, len(src)):
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def _kama_np(src: np.ndarray, period: int) -> np.ndarray:
    """Kaufman Adaptive Moving Average."""
    fast_sc = 2.0 / (2 + 1)
    slow_sc = 2.0 / (30 + 1)
    out = np.empty_like(src, dtype=np.float64)
    out[0] = src[0]
    for i in range(1, len(src)):
        start = max(0, i - period)
        direction = abs(src[i] - src[start])
        volatility = np.sum(np.abs(np.diff(src[start:i + 1])))
        er = direction / volatility if volatility != 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (src[i] - out[i - 1])
    return out


# ──────────────────────────────────────────────────────────────────────────
#  Dispatcher
# ──────────────────────────────────────────────────────────────────────────

_MA_MAP = {
    MAType.Simple: _sma_np,
    MAType.Exponential: _ema_np,
    MAType.Weighted: _wma_np,
    MAType.Triangular: _tma_np,
    MAType.Hull: _hull_np,
    MAType.VIDYA: _vidya_np,
    MAType.WilderSmoothing: _wilder_np,
    MAType.KaufmanAdaptive: _kama_np,
}


def get_ma(src: np.ndarray, ma_type: MAType, period: int) -> np.ndarray:
    """Compute a moving average of the given type."""
    fn = _MA_MAP.get(ma_type)
    if fn is None:
        raise ValueError(f"Unknown MA type: {ma_type}")
    return fn(src, period)


def get_stddev(src: np.ndarray, ma: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation around a pre-computed MA."""
    out = np.full_like(src, np.nan, dtype=np.float64)
    for i in range(period - 1, len(src)):
        window = src[i - period + 1:i + 1]
        ma_window = ma[i - period + 1:i + 1]
        valid = ~np.isnan(ma_window)
        if valid.any():
            out[i] = np.std(window[valid] - ma_window[valid]) + 1e-10
        else:
            out[i] = 1e-10
    return out


# ──────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ──────────────────────────────────────────────────────────────────────────

def rolling_percentile(a):
    """Percentile rank of the last element inside a rolling window."""
    last = a[-1]
    return np.mean(a <= last) * 100


def l1norm(window_values):
    """L1-normalized value: last_value / sum(abs(window))."""
    denom = np.abs(window_values).sum()
    return window_values[-1] / denom if denom != 0 else 1
