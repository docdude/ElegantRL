"""
Deep-M Effort NQ — Consolidated Working Script
=================================================
Reads Sierra Chart .scid files and outputs a compilable C++ study.

Methodology (from DeepCharts documentation & Wyckoff theory):
  - 40 Range bars on NQ normalize time out of the equation
  - "Effort" = volume/delta/time needed to complete each range bar
  - Purple zones = path of least resistance is DOWN (bearish pressure)
  - Green zones  = path of least resistance is UP   (bullish pressure)
  - Built-in moving average adds confluence and confirms directional bias

Key indicator outputs:
  1. Colored zones on the price chart (green=bullish, purple=bearish)
  2. A dynamic moving average that changes color above/below price
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from pathlib import Path
import warnings
import time as time_module

from wyckoff_effort.utils.scid_parser import (
    SCIDReader, SierraChartDataLocator,
    resample_range_bars, resample_ticks, load_nq_data,
)

warnings.filterwarnings("ignore")

# Use non-interactive backend if no display
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")


# =============================================================================
# Configuration & Types
# =============================================================================

@dataclass
class DeepMEffortConfig:
    """
    Parameters for Deep-M Effort indicator.

    Based on DeepCharts documentation:
    - Designed for NQ on 40 Range bars
    - Highlights path of least resistance
    - Green = bullish pressure, Purple = bearish pressure
    - Built-in EMA for confluence
    """
    range_size: float = 40.0          # Range bar size in ticks
    tick_size: float = 0.25           # NQ tick size ($5 per tick)
    ema_period: int = 20              # EMA for effort smoothing / price MA
    zone_std_mult: float = 1.5        # Std dev multiplier for zone thresholds
    delta_ema_period: int = 14        # EMA for delta effort
    time_ema_period: int = 14         # EMA for time effort
    absorption_threshold: float = 1.5 # Volume/EMA ratio for absorption
    vacuum_threshold: float = 0.6     # Volume/EMA ratio for vacuum
    delta_filter: float = 0.20        # Max |delta%| for absorption
    speed_filter: float = 1.1         # Min speed ratio for vacuum
    abs_continuation: float = 0.6     # Absorption cluster continuation factor
    vac_continuation: float = 1.3     # Vacuum cluster continuation factor
    min_zone_bars: int = 2            # Minimum bars in cluster to form zone
    zone_extension: int = 15          # Bars to extend zone forward
    max_zone_age: int = 200           # Max bars before zone expires
    max_cluster_bars: int = 40        # Max bars per zone cluster


class ZoneType(Enum):
    """
    Zone types matching DeepCharts colors:
    - Green (bullish)  = path of least resistance UP
    - Purple (bearish) = path of least resistance DOWN
    """
    BULLISH = "bullish"    # Green zones
    BEARISH = "bearish"    # Purple zones


@dataclass
class EffortZone:
    zone_type: ZoneType
    price_high: float
    price_low: float
    bar_index_start: int
    bar_index_end: int
    strength: float
    volume_effort: float
    delta_effort: float
    active: bool = True


@dataclass
class RangeBar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    delta: float
    duration_seconds: float
    tick_count: int
    bar_index: int
    timestamp: Optional[pd.Timestamp] = None


# =============================================================================
# Range Bar Builder
# =============================================================================

class RangeBarBuilder:
    def __init__(self, config: DeepMEffortConfig):
        self.config = config
        self.range_points = config.range_size * config.tick_size

    def from_synthetic(self, n_bars: int = 500, seed: int = 42) -> List[RangeBar]:
        """Generate synthetic range bars for testing."""
        rng = np.random.RandomState(seed)
        bars = []
        price = 20000.0

        for i in range(n_bars):
            trend_bias = 0.1 * np.sin(i / 80)
            move = (rng.randn() * 0.4 + trend_bias) * self.range_points
            bar_open = price
            if move >= 0:
                bar_high = bar_open + self.range_points
                bar_low = bar_open
                bar_close = bar_high
            else:
                bar_low = bar_open - self.range_points
                bar_high = bar_open
                bar_close = bar_low

            base_vol = 800 + 400 * np.abs(np.sin(i / 30))
            volume = base_vol + rng.exponential(200)
            direction = 1 if bar_close > bar_open else -1
            delta = direction * volume * (0.1 + 0.3 * rng.rand()) + rng.randn() * 100
            duration = max(5, 60 / (volume / 500) + rng.randn() * 10)

            # Inject absorption events
            if rng.rand() < 0.08:
                volume *= 3.0
                delta *= 0.2
                duration *= 0.5

            bars.append(RangeBar(
                open=round(bar_open, 2), high=round(bar_high, 2),
                low=round(bar_low, 2), close=round(bar_close, 2),
                volume=round(volume), delta=round(delta),
                duration_seconds=round(duration, 1),
                tick_count=int(volume / 10), bar_index=i
            ))
            price = bar_close
        return bars


def dataframe_to_rangebars(df: pd.DataFrame) -> List[RangeBar]:
    """Convert a range bar DataFrame (from scid_parser.resample_range_bars) to List[RangeBar]."""
    bars = []
    for i, (ts, row) in enumerate(df.iterrows()):
        bars.append(RangeBar(
            open=round(row["open"], 2),
            high=round(row["high"], 2),
            low=round(row["low"], 2),
            close=round(row["close"], 2),
            volume=row["volume"],
            delta=row["delta"],
            duration_seconds=row.get("duration_seconds", 1.0),
            tick_count=int(row.get("num_trades", 0)),
            bar_index=i,
            timestamp=ts,
        ))
    return bars


# =============================================================================
# EMA & Statistics — causal (no future leak)
# =============================================================================

def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (causal, no lookahead)."""
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    return result


def rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation (causal)."""
    result = np.full_like(data, np.nan, dtype=float)
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1:i + 1])
    return result


def rolling_normalize(data: np.ndarray, period: int) -> np.ndarray:
    """
    Rolling min-max normalization over a lookback window.
    Unlike global normalize_series, this is causal — each value is normalized
    against only the past 'period' values, preventing future information leak.
    """
    result = np.full_like(data, 0.0, dtype=float)
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        mn = np.nanmin(window)
        mx = np.nanmax(window)
        if mx - mn > 1e-10:
            result[i] = (data[i] - mn) / (mx - mn)
        else:
            result[i] = 0.5
    return result


# =============================================================================
# Deep-M Effort Engine
# =============================================================================

class DeepMEffortEngine:
    """
    Core computation engine for the Deep-M Effort indicator.

    Effort is decomposed into three components:
      1. Volume Effort — contracts per range bar vs EMA
      2. Delta Effort  — net directional pressure (signed & abs)
      3. Time Effort   — speed of bar completion (1 / duration)

    These produce a composite Effort Index. Zones are detected where
    effort is anomalous relative to the EMA baseline.

    Zone classification (matching DeepCharts):
      - BULLISH (green)  = path of least resistance UP
        Triggered by: absorption at support (selling absorbed by buyers)
                  or: vacuum moves up (low effort = no resistance)
      - BEARISH (purple) = path of least resistance DOWN
        Triggered by: absorption at resistance (buying absorbed by sellers)
                  or: vacuum moves down
    """

    def __init__(self, config: DeepMEffortConfig):
        self.config = config
        self.zones: List[EffortZone] = []

    def compute(self, bars: List[RangeBar]) -> pd.DataFrame:
        n = len(bars)
        print(f"[Effort Engine] Computing metrics for {n:,} range bars...")
        t0 = time_module.time()

        opens = np.array([b.open for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        closes = np.array([b.close for b in bars])
        volumes = np.array([b.volume for b in bars], dtype=float)
        deltas = np.array([b.delta for b in bars], dtype=float)
        durations = np.array([b.duration_seconds for b in bars], dtype=float)
        timestamps = [b.timestamp for b in bars]

        direction = np.where(closes >= opens, 1.0, -1.0)

        # Volume Effort: ratio of current volume to its EMA
        vol_ema_arr = ema(volumes, self.config.ema_period)
        vol_ratio = np.where(vol_ema_arr > 0, volumes / vol_ema_arr, 1.0)

        # Delta Effort: signed delta as fraction of volume
        abs_delta = np.abs(deltas)
        delta_ema_arr = ema(abs_delta, self.config.delta_ema_period)
        delta_pct = np.where(volumes > 0, deltas / volumes, 0.0)
        delta_pct_ema = ema(delta_pct, self.config.delta_ema_period)
        # Delta divergence: high volume but weak net direction
        delta_divergence = np.where(
            abs_delta > 0,
            vol_ratio / (np.abs(delta_pct) + 0.01),
            0.0
        )

        # Time Effort (Speed): inverse of duration
        speed = np.where(durations > 0, 1.0 / durations, 0.0)
        speed_ema = ema(speed, self.config.time_ema_period)
        speed_ratio = np.where(speed_ema > 0, speed / speed_ema, 1.0)

        # Composite Effort Index — using rolling normalization (causal)
        norm_period = self.config.ema_period * 5  # 100-bar lookback
        effort_index = (
            0.50 * rolling_normalize(vol_ratio, norm_period) +
            0.25 * rolling_normalize(delta_divergence, norm_period) +
            0.25 * rolling_normalize(speed_ratio, norm_period)
        )
        effort_ema_arr = ema(effort_index, self.config.ema_period)
        effort_std = rolling_std(effort_index, self.config.ema_period)
        effort_upper = effort_ema_arr + self.config.zone_std_mult * effort_std
        effort_lower = effort_ema_arr - self.config.zone_std_mult * effort_std

        # Absorption Score: high volume + weak delta + slow completion
        absorption_score = vol_ratio * (1.0 - np.abs(delta_pct))
        # Weight by inverse speed ratio (slow bars get higher score)
        absorption_score *= np.where(speed_ratio > 0,
                                     1.0 / (speed_ratio + 0.1), 1.0)
        absorption_ema_arr = ema(absorption_score, self.config.ema_period)

        # Price EMA (the "built-in moving average" from DeepCharts)
        price_ema_arr = ema(closes, self.config.ema_period)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "delta": deltas, "duration": durations,
            "direction": direction,
            "vol_ema": vol_ema_arr, "vol_ratio": vol_ratio,
            "abs_delta": abs_delta, "delta_ema": delta_ema_arr,
            "delta_pct": delta_pct, "delta_pct_ema": delta_pct_ema,
            "delta_divergence": delta_divergence,
            "speed": speed, "speed_ema": speed_ema, "speed_ratio": speed_ratio,
            "effort_index": effort_index,
            "effort_ema": effort_ema_arr,
            "effort_upper": effort_upper,
            "effort_lower": effort_lower,
            "absorption_score": absorption_score,
            "absorption_ema": absorption_ema_arr,
            "price_ema": price_ema_arr,
        })

        elapsed = time_module.time() - t0
        print(f"[Effort Engine] Metrics computed in {elapsed:.1f}s")

        print("[Effort Engine] Detecting zones...")
        self.zones = self._detect_zones(df, bars)
        print(f"[Effort Engine] Found {len(self.zones)} zones")
        return df

    def _detect_zones(self, df: pd.DataFrame,
                      bars: List[RangeBar]) -> List[EffortZone]:
        """
        Zone detection logic:

        ABSORPTION zones — high volume, weak delta, slow bars
          These represent institutional order absorption (Wyckoff).
          - If absorbing at lows (bearish bars absorbed) -> support -> BULLISH
          - If absorbing at highs (bullish bars absorbed) -> resistance -> BEARISH

        VACUUM zones — low volume, fast bars
          Price moves through these areas with no resistance.
          - Moving up -> BULLISH (path of least resistance is up)
          - Moving down -> BEARISH (path of least resistance is down)
        """
        zones = []
        n = len(df)
        cfg = self.config
        i = cfg.ema_period

        while i < n:
            vol_r = df.iloc[i]["vol_ratio"]
            delta_p = df.iloc[i]["delta_pct"]
            speed_r = df.iloc[i]["speed_ratio"]

            # --- Absorption Zone Detection ---
            if vol_r >= cfg.absorption_threshold and abs(delta_p) < cfg.delta_filter:
                j = i + 1
                while (j < n and
                       j - i < cfg.max_cluster_bars and
                       df.iloc[j]["vol_ratio"] >= cfg.absorption_threshold * cfg.abs_continuation):
                    j += 1

                if j - i >= cfg.min_zone_bars:
                    cluster = bars[i:j]
                    zh = max(b.high for b in cluster) + cfg.tick_size * 4
                    zl = min(b.low for b in cluster) - cfg.tick_size * 4

                    avg_dir = np.mean(df.iloc[i:j]["direction"].values)
                    # Bearish bars being absorbed at lows = buying support = BULLISH
                    # Bullish bars being absorbed at highs = selling resistance = BEARISH
                    if avg_dir < 0:
                        z_type = ZoneType.BULLISH   # support absorption
                    else:
                        z_type = ZoneType.BEARISH   # resistance absorption

                    strength = min(1.0,
                                   np.mean(df.iloc[i:j]["vol_ratio"].values) /
                                   (cfg.absorption_threshold * 2))

                    zones.append(EffortZone(
                        zone_type=z_type,
                        price_high=zh, price_low=zl,
                        bar_index_start=i, bar_index_end=j - 1,
                        strength=strength,
                        volume_effort=float(np.sum(df.iloc[i:j]["volume"].values)),
                        delta_effort=float(np.mean(df.iloc[i:j]["delta"].values)),
                    ))
                    i = j
                    continue

            # --- Vacuum Zone Detection ---
            if vol_r <= cfg.vacuum_threshold and speed_r > cfg.speed_filter:
                j = i + 1
                while (j < n and
                       j - i < cfg.max_cluster_bars and
                       df.iloc[j]["vol_ratio"] <= cfg.vacuum_threshold * cfg.vac_continuation):
                    j += 1

                if j - i >= cfg.min_zone_bars:
                    cluster = bars[i:j]
                    zh = max(b.high for b in cluster)
                    zl = min(b.low for b in cluster)
                    avg_dir = np.mean(df.iloc[i:j]["direction"].values)

                    # Vacuum in direction of move = continuation
                    z_type = (ZoneType.BULLISH if avg_dir > 0
                              else ZoneType.BEARISH)
                    strength = min(1.0,
                                   1.0 - np.mean(df.iloc[i:j]["vol_ratio"].values))

                    zones.append(EffortZone(
                        zone_type=z_type,
                        price_high=zh, price_low=zl,
                        bar_index_start=i, bar_index_end=j - 1,
                        strength=strength,
                        volume_effort=float(np.sum(df.iloc[i:j]["volume"].values)),
                        delta_effort=float(np.mean(df.iloc[i:j]["delta"].values)),
                    ))
                    i = j
                    continue

            i += 1
        return zones


# =============================================================================
# Signal Generator
# =============================================================================

class SignalGenerator:
    """
    Generates trading signals from zones + price action + EMA confluence.

    Rules:
      1. LONG when price is in a BULLISH zone AND delta confirms
         (delta_pct > delta_pct_ema) AND price > price_ema
      2. SHORT when price is in a BEARISH zone AND delta confirms
         (delta_pct < delta_pct_ema) AND price < price_ema
      3. EMA confluence: the built-in MA adds directional confirmation
    """

    @staticmethod
    def generate(df: pd.DataFrame,
                 zones: List[EffortZone]) -> pd.DataFrame:
        n = len(df)
        signals = pd.DataFrame({
            "signal": np.zeros(n, dtype=int),
            "signal_type": [""] * n,
            "zone_strength": np.zeros(n),
        })

        for zone in zones:
            if not zone.active:
                continue
            start = zone.bar_index_start
            end = min(zone.bar_index_end + 1, n)

            if zone.zone_type == ZoneType.BULLISH:
                for k in range(start, end):
                    row = df.iloc[k]
                    # Delta turning positive + price above EMA = confluence
                    delta_ok = row["delta_pct"] > row["delta_pct_ema"]
                    ema_ok = (np.isnan(row["price_ema"]) or
                              row["close"] >= row["price_ema"])
                    if delta_ok and ema_ok:
                        signals.at[k, "signal"] = 1
                        signals.at[k, "signal_type"] = "BULLISH_ZONE"
                        signals.at[k, "zone_strength"] = zone.strength
                        break

            elif zone.zone_type == ZoneType.BEARISH:
                for k in range(start, end):
                    row = df.iloc[k]
                    delta_ok = row["delta_pct"] < row["delta_pct_ema"]
                    ema_ok = (np.isnan(row["price_ema"]) or
                              row["close"] <= row["price_ema"])
                    if delta_ok and ema_ok:
                        signals.at[k, "signal"] = -1
                        signals.at[k, "signal_type"] = "BEARISH_ZONE"
                        signals.at[k, "zone_strength"] = zone.strength
                        break

        return signals


# =============================================================================
# Chart
# =============================================================================

class DeepMEffortChart:
    """
    Interactive Plotly chart matching DeepCharts visual style:
      - Green zones on price chart (bullish)
      - Purple zones on price chart (bearish)
      - Price EMA that changes color above/below price
      - 4 panels: Price, Volume, Effort, Delta/Absorption
    """

    ZONE_COLORS = {
        ZoneType.BULLISH:  ("rgba(76, 175, 80, 0.25)",  "rgba(76, 175, 80, 0.5)"),
        ZoneType.BEARISH:  ("rgba(156, 39, 176, 0.25)", "rgba(156, 39, 176, 0.5)"),
    }

    def __init__(self, df, zones, signals, config):
        self.df = df
        self.zones = zones
        self.signals = signals
        self.config = config

    def plot(self, start=0, end=None, figsize=(22, 16)):
        if end is None:
            end = len(self.df)
        df = self.df.iloc[start:end].copy().reset_index(drop=True)
        n_bars = len(df)
        x = np.arange(n_bars)

        # Use timestamps for x-axis if available
        has_ts = "timestamp" in df.columns and df["timestamp"].notna().any()
        if has_ts:
            x_vals = df["timestamp"]
        else:
            x_vals = x

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.18, 0.18, 0.19],
            vertical_spacing=0.03,
            subplot_titles=["", "Volume", "Effort Index", "Delta / Absorption"],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": True}]],
        )

        # Compute bar width for timestamp x-axis (in milliseconds for Plotly)
        if has_ts and n_bars > 1:
            median_dt = df["timestamp"].diff().dropna().median()
            bar_width = median_dt.total_seconds() * 1000 * 0.8
        else:
            bar_width = None

        # --- Row 1: Price (Candlestick + EMA + Zones + Signals) ---
        fig.add_trace(go.Candlestick(
            x=x_vals, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
            increasing_fillcolor="#26A69A", decreasing_fillcolor="#EF5350",
            name="Price", showlegend=False,
        ), row=1, col=1)

        # Price EMA with slope-based coloring
        price_ema_vals = df["price_ema"].values
        ema_above_x, ema_above_y = [], []
        ema_below_x, ema_below_y = [], []
        for i in range(n_bars):
            if np.isnan(price_ema_vals[i]):
                continue
            xv = x_vals.iloc[i] if has_ts else i
            if df.iloc[i]["close"] >= price_ema_vals[i]:
                ema_above_x.append(xv)
                ema_above_y.append(price_ema_vals[i])
                if ema_below_x:
                    # bridge the gap
                    ema_below_x.append(xv)
                    ema_below_y.append(price_ema_vals[i])
            else:
                ema_below_x.append(xv)
                ema_below_y.append(price_ema_vals[i])
                if ema_above_x:
                    ema_above_x.append(xv)
                    ema_above_y.append(price_ema_vals[i])

        if ema_above_x:
            fig.add_trace(go.Scatter(
                x=ema_above_x, y=ema_above_y, mode="lines",
                line=dict(color="#4CAF50", width=2),
                name="EMA (above)", showlegend=False,
                connectgaps=False,
            ), row=1, col=1)
        if ema_below_x:
            fig.add_trace(go.Scatter(
                x=ema_below_x, y=ema_below_y, mode="lines",
                line=dict(color="#9C27B0", width=2),
                name="EMA (below)", showlegend=False,
                connectgaps=False,
            ), row=1, col=1)

        # Zones as rectangles
        for zone in self.zones:
            if zone.bar_index_end < start or zone.bar_index_start >= end:
                continue
            if zone.bar_index_end < start:
                continue
            fill_color, line_color = self.ZONE_COLORS.get(
                zone.zone_type, ("rgba(128,128,128,0.1)", "rgba(128,128,128,0.3)"))
            # Increase opacity with strength
            xs = max(zone.bar_index_start - start, 0)
            xe = min(n_bars - 1,
                     zone.bar_index_end - start + self.config.zone_extension)
            if has_ts:
                x0 = x_vals.iloc[xs]
                x1 = x_vals.iloc[min(xe, n_bars - 1)]
            else:
                x0, x1 = xs, xe
            fig.add_shape(
                type="rect", x0=x0, x1=x1,
                y0=zone.price_low, y1=zone.price_high,
                fillcolor=fill_color, line=dict(color=line_color, width=1, dash="dot"),
                row=1, col=1,
            )

        # Signals
        sigs = self.signals.iloc[start:start + n_bars].reset_index(drop=True)
        tick_off = self.config.range_size * self.config.tick_size * 0.5
        long_x, long_y = [], []
        short_x, short_y = [], []
        for i in range(len(sigs)):
            s = sigs.iloc[i]
            xv = x_vals.iloc[i] if has_ts else i
            if s["signal"] == 1:
                long_x.append(xv)
                long_y.append(df.iloc[i]["low"] - tick_off)
            elif s["signal"] == -1:
                short_x.append(xv)
                short_y.append(df.iloc[i]["high"] + tick_off)
        if long_x:
            fig.add_trace(go.Scatter(
                x=long_x, y=long_y, mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#00E676"),
                name="Long Signal", showlegend=True,
            ), row=1, col=1)
        if short_x:
            fig.add_trace(go.Scatter(
                x=short_x, y=short_y, mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#FF1744"),
                name="Short Signal", showlegend=True,
            ), row=1, col=1)

        # --- Row 2: Volume ---
        vol_colors = ["#26A69A" if d > 0 else "#EF5350"
                       for d in df["direction"]]
        bar_kw = dict(width=bar_width) if bar_width is not None else {}
        fig.add_trace(go.Bar(
            x=x_vals, y=df["volume"], marker_color=vol_colors,
            opacity=0.7, name="Volume", showlegend=False, **bar_kw,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=x_vals, y=df["vol_ema"], mode="lines",
            line=dict(color="#FFD700", width=1),
            name="Vol EMA", showlegend=False,
        ), row=2, col=1)

        # --- Row 3: Effort Index ---
        fig.add_trace(go.Scatter(
            x=x_vals, y=df["effort_upper"], mode="lines",
            line=dict(color="rgba(123,31,162,0.3)", width=0),
            showlegend=False, name="Effort Upper",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=x_vals, y=df["effort_lower"], mode="lines",
            line=dict(color="rgba(123,31,162,0.3)", width=0),
            fill="tonexty", fillcolor="rgba(123,31,162,0.12)",
            showlegend=False, name="Effort Lower",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=x_vals, y=df["effort_index"], mode="lines",
            line=dict(color="#00BCD4", width=1.5),
            name="Effort Index", showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=x_vals, y=df["effort_ema"], mode="lines",
            line=dict(color="#FFD700", width=1),
            name="Effort EMA", showlegend=False,
        ), row=3, col=1)

        # --- Row 4: Delta + Absorption (dual y-axis) ---
        delta_colors = ["#26A69A" if d > 0 else "#EF5350"
                         for d in df["delta"]]
        fig.add_trace(go.Bar(
            x=x_vals, y=df["delta"], marker_color=delta_colors,
            opacity=0.6, name="Delta", showlegend=False, **bar_kw,
        ), row=4, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=x_vals, y=df["absorption_score"], mode="lines",
            line=dict(color="#AB47BC", width=1),
            name="Absorption", showlegend=False,
        ), row=4, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(
            x=x_vals, y=df["absorption_ema"], mode="lines",
            line=dict(color="#FFD700", width=0.8),
            name="Abs EMA", showlegend=False,
        ), row=4, col=1, secondary_y=True)

        # --- Layout ---
        title = "Deep-M Effort NQ — Range 40"
        if has_ts:
            ts0 = df["timestamp"].dropna().iloc[0]
            ts1 = df["timestamp"].dropna().iloc[-1]
            title += f" | {ts0} → {ts1}"

        fig.update_layout(
            template="plotly_dark",
            title=dict(text=title, font=dict(size=16)),
            height=900,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )

        # Style axes
        for i in range(1, 5):
            fig.update_xaxes(
                gridcolor="#333355", showgrid=True, row=i, col=1,
            )
            fig.update_yaxes(
                gridcolor="#333355", showgrid=True, row=i, col=1,
            )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")
        fig.update_yaxes(title_text="Effort", row=3, col=1)
        fig.update_yaxes(title_text="Delta", row=4, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Absorption", row=4, col=1,
                         secondary_y=True, showgrid=False,
                         color="#AB47BC", tickfont=dict(color="#AB47BC"))

        # Save interactive HTML and static PNG
        html_file = "deep_m_effort_nq.html"
        fig.write_html(html_file)
        print(f"[Interactive chart saved: {html_file}]")

        try:
            png_file = "deep_m_effort_nq.png"
            fig.write_image(png_file, width=1800, height=1000, scale=2)
            print(f"[Static chart saved: {png_file}]")
        except Exception as e:
            print(f"[PNG export skipped — install kaleido: pip install kaleido] ({e})")


# =============================================================================
# Performance Summary
# =============================================================================

def performance_summary(df, signals, config):
    """Simple backtest using zone signals with trailing stop exit."""
    trades = []
    position = 0
    entry_price = entry_bar = 0
    best_price = 0.0

    # Use a trailing stop of 2x range bar size in points
    range_points = config.range_size * config.tick_size
    trail_stop_pts = range_points * 2.0
    # Take profit at 4x range bar size
    take_profit_pts = range_points * 4.0

    for i in range(len(signals)):
        sig = signals.iloc[i]["signal"]
        close = df.iloc[i]["close"]

        # Entry
        if sig != 0 and position == 0:
            position = int(sig)
            entry_price = close
            entry_bar = i
            best_price = close
            continue

        # Manage open position
        if position != 0:
            if position == 1:
                best_price = max(best_price, close)
                trail_hit = close <= best_price - trail_stop_pts
                tp_hit = close >= entry_price + take_profit_pts
            else:
                best_price = min(best_price, close)
                trail_hit = close >= best_price + trail_stop_pts
                tp_hit = close <= entry_price - take_profit_pts

            if trail_hit or tp_hit or (i - entry_bar >= 50):
                pnl = (close - entry_price) * position
                trades.append({
                    "entry_bar": entry_bar, "exit_bar": i,
                    "direction": "LONG" if position > 0 else "SHORT",
                    "entry": entry_price, "exit": close,
                    "pnl_points": round(pnl, 2),
                    "pnl_ticks": round(pnl / config.tick_size, 1),
                    "exit_reason": ("TP" if tp_hit else
                                    "Trail" if trail_hit else "MaxBars"),
                    "signal_type": signals.iloc[entry_bar]["signal_type"],
                })
                position = 0

    if not trades:
        print("No trades generated.")
        return pd.DataFrame()

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl_points"] > 0]
    print(f"\n{'='*60}")
    print(f"  DEEP-M EFFORT NQ — PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Trades:   {len(tdf)}")
    print(f"  Winners:        {len(wins)} ({100*len(wins)/len(tdf):.1f}%)")
    print(f"  Total PnL:      {tdf['pnl_points'].sum():.2f} pts "
          f"({tdf['pnl_ticks'].sum():.0f} ticks)")
    print(f"  Avg PnL/Trade:  {tdf['pnl_points'].mean():.2f} pts")
    if len(wins) > 0:
        print(f"  Max Win:        {tdf['pnl_points'].max():.2f} pts")
    print(f"  Max Loss:       {tdf['pnl_points'].min():.2f} pts")
    gl = abs(tdf[tdf["pnl_points"] < 0]["pnl_points"].sum())
    pf = wins["pnl_points"].sum() / gl if gl > 0 else float("inf")
    print(f"  Profit Factor:  {pf:.2f}")
    print(f"{'='*60}")
    print("\nBy Signal Type:")
    print(tdf.groupby("signal_type")["pnl_points"].agg(
        ["count", "sum", "mean"]).round(2))
    print("\nBy Exit Reason:")
    print(tdf.groupby("exit_reason")["pnl_points"].agg(
        ["count", "sum", "mean"]).round(2))
    return tdf


# =============================================================================
# Main
# =============================================================================

def main():
    config = DeepMEffortConfig(
        range_size=40, tick_size=0.25, ema_period=20,
        zone_std_mult=1.5, absorption_threshold=1.5, vacuum_threshold=0.6,
    )

    # --- Find SCID file ---
    SCID_PATH = None
    found = SierraChartDataLocator.find_scid_file("NQZ25-CME")
    if found:
        SCID_PATH = str(found)
        print(f"[Auto-detected] {SCID_PATH}")

    use_real = SCID_PATH is not None and Path(SCID_PATH).exists()

    if use_real:
        print(f"\n{'='*60}")
        print(f"  LOADING: {SCID_PATH}")
        print(f"  Size: {Path(SCID_PATH).stat().st_size / (1024**2):.1f} MB")
        print(f"{'='*60}\n")

        reader = SCIDReader(SCID_PATH)
        try:
            info = reader.info()
            print(f"  First record: {info['first_date']}")
            print(f"  Last record:  {info['last_date']}")
            print(f"  First close:  {info['first_close']:.2f}")
            print(f"  Last close:   {info['last_close']:.2f}\n")
        except Exception as e:
            print(f"  (Could not read file info: {e})")

        scid_df = reader.read(
            start_date=None,
            end_date=None,
            max_records=5_000_000,
            trading_hours_only=True,
        )

        if scid_df.empty:
            print("[WARNING] No data loaded. Falling back to synthetic.")
            use_real = False
        else:
            range_df = resample_range_bars(
                scid_df,
                range_size=config.range_size * config.tick_size,
                tick_size=config.tick_size,
            )
            if range_df.empty:
                print("[WARNING] No range bars built. Falling back to synthetic.")
                use_real = False
            else:
                bars = dataframe_to_rangebars(range_df)

    if not use_real:
        print("\n[Using synthetic data for demonstration]")
        avail = SierraChartDataLocator.list_all_scid()
        if avail:
            print("Available .scid files:")
            for f in avail[:10]:
                print(f"  {f.name:40s} ({f.stat().st_size/(1024**2):.1f} MB)")
        builder = RangeBarBuilder(config)
        bars = builder.from_synthetic(n_bars=500, seed=42)

    print(f"\nTotal range bars: {len(bars):,}")

    # Compute effort metrics
    engine = DeepMEffortEngine(config)
    df = engine.compute(bars)
    zones = engine.zones

    print(f"\nDetected {len(zones)} effort zones:")
    for z in zones[:20]:
        ts = ""
        if (z.bar_index_start < len(bars) and
                bars[z.bar_index_start].timestamp):
            ts = f" @ {bars[z.bar_index_start].timestamp}"
        color = "GREEN" if z.zone_type == ZoneType.BULLISH else "PURPLE"
        print(f"  {color:8s} [{z.price_low:.2f} - {z.price_high:.2f}] "
              f"str={z.strength:.2f}{ts}")
    if len(zones) > 20:
        print(f"  ... and {len(zones) - 20} more zones")

    # Signals
    signals = SignalGenerator.generate(df, zones)
    n_long = (signals["signal"] == 1).sum()
    n_short = (signals["signal"] == -1).sum()
    print(f"\nSignals: {n_long} long, {n_short} short")

    # Performance
    trades_df = performance_summary(df, signals, config)

    # Chart — filter to specific time window for comparison
    # 12/18/2025 morning session: 7:30 AM - 11:30 AM MT (4 hours)
    chart_start_ts = pd.Timestamp("2025-12-18 07:30:00")
    chart_end_ts = pd.Timestamp("2025-12-18 11:30:00")

    chart_start_idx = None
    chart_end_idx = None
    for i in range(len(df)):
        ts = df.iloc[i]["timestamp"]
        if pd.notna(ts):
            if chart_start_idx is None and ts >= chart_start_ts:
                chart_start_idx = i
            if ts <= chart_end_ts:
                chart_end_idx = i

    if chart_start_idx is not None and chart_end_idx is not None:
        # Skip gap-fill bars at session open (synthetic bars from overnight gap,
        # all sharing the same sub-second timestamp)
        dw = df.iloc[chart_start_idx:chart_end_idx + 1]
        first_ts = dw.iloc[0]["timestamp"]
        gap_bars = (dw["timestamp"] == first_ts).sum()
        if gap_bars > 3:
            print(f"\n[Chart] Skipping {gap_bars} gap-fill bars at open "
                  f"({dw.iloc[0]['open']:.2f} -> {dw.iloc[gap_bars-1]['close']:.2f})")
            chart_start_idx += gap_bars

        print(f"\n[Chart] Plotting bars {chart_start_idx}-{chart_end_idx} "
              f"({chart_end_idx - chart_start_idx + 1} bars) "
              f"for {chart_start_ts} -> {chart_end_ts}")
        chart = DeepMEffortChart(df, zones, signals, config)
        chart.plot(start=chart_start_idx, end=chart_end_idx + 1)
    else:
        print(f"\n[Chart] Could not find bars for {chart_start_ts} -> "
              f"{chart_end_ts}, plotting last 300 bars")
        n_chart = min(len(df), 300)
        chart = DeepMEffortChart(df, zones, signals, config)
        chart.plot(start=max(0, len(df) - n_chart), end=len(df))


if __name__ == "__main__":
    main()
