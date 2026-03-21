"""
Deep-M Effort NQ — Indicator Recreation
=========================================
Mimics the DeepCharts Deep-M Effort (NQ) indicator logic.

Methodology & Rationale:
-------------------------
The indicator is rooted in Wyckoff Effort vs. Result analysis applied to
range bars. "Effort" = volume/delta/time required to move price through a
fixed range. When effort is disproportionate to result, it signals
absorption (institutional activity) and potential reversal zones.

Key concepts captured:
  1. Range Bars (40-tick for NQ) normalize price action.
  2. Volume Effort  — total contracts to complete one range bar.
  3. Delta Effort   — net buy vs sell pressure per bar.
  4. Time Effort    — seconds to complete each range bar.
  5. EMA baseline   — smoothed effort for dynamic zone generation.
  6. Zones          — areas where effort diverges from the norm.

Trading Practice:
  • HIGH effort + small result → absorption / accumulation / distribution
  • LOW  effort + full range   → path of least resistance (continuation)
  • Effort DIVERGENCE from EMA → institutional footprint, zone boundary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DeepMEffortConfig:
    """All tunable parameters for the Deep-M Effort indicator."""
    range_size: float = 40.0          # Range bar size in ticks
    tick_size: float = 0.25           # NQ tick size
    ema_period: int = 20              # EMA period for effort smoothing
    zone_std_mult: float = 1.5        # Std dev multiplier for zone thresholds
    delta_ema_period: int = 14        # EMA for delta effort
    time_ema_period: int = 14         # EMA for time effort
    volume_ma_period: int = 20        # MA period for volume normalization
    zone_lookback: int = 50           # Bars to look back for zone detection
    absorption_threshold: float = 1.5 # Volume/EMA ratio for absorption
    vacuum_threshold: float = 0.6     # Volume/EMA ratio for low-effort moves
    delta_filter: float = 0.20        # Max |delta%| for absorption
    speed_filter: float = 1.1         # Min speed ratio for vacuum
    abs_continuation: float = 0.6     # Absorption cluster continuation factor
    vac_continuation: float = 1.3     # Vacuum cluster continuation factor
    min_zone_bars: int = 2            # Min bars in cluster to form zone
    min_zone_width: float = 10.0      # Minimum zone width in ticks
    max_zone_age: int = 200           # Max bars before zone expires


class ZoneType(Enum):
    ABSORPTION_SUPPORT = "absorption_support"
    ABSORPTION_RESISTANCE = "absorption_resistance"
    VACUUM_BULLISH = "vacuum_bullish"
    VACUUM_BEARISH = "vacuum_bearish"


@dataclass
class EffortZone:
    """A detected effort zone with price boundaries and metadata."""
    zone_type: ZoneType
    price_high: float
    price_low: float
    bar_index_start: int
    bar_index_end: int
    strength: float            # Normalized zone strength 0–1
    volume_effort: float
    delta_effort: float
    active: bool = True
    touches: int = 0


@dataclass
class RangeBar:
    """A single range bar built from tick/OHLCV data."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    delta: float               # ask_volume - bid_volume
    duration_seconds: float    # Time to complete bar
    tick_count: int
    bar_index: int
    timestamp: Optional[pd.Timestamp] = None


# =============================================================================
# Range Bar Builder
# =============================================================================

class RangeBarBuilder:
    """Constructs range bars from tick-level or OHLCV data."""

    def __init__(self, config: DeepMEffortConfig):
        self.config = config
        self.range_points = config.range_size * config.tick_size  # 40 * 0.25 = 10 pts

    def from_ohlcv(self, df: pd.DataFrame) -> List[RangeBar]:
        """
        Build range bars from minute/second OHLCV data.
        Expects columns: open, high, low, close, volume
        Optionally: ask_volume, bid_volume (for delta)
        """
        bars = []
        current_open = df.iloc[0]["open"]
        current_high = current_open
        current_low = current_open
        cum_volume = 0.0
        cum_delta = 0.0
        cum_ticks = 0
        start_time = df.index[0] if isinstance(df.index, pd.DatetimeIndex) else None
        bar_idx = 0

        for i in range(len(df)):
            row = df.iloc[i]
            price = row["close"]
            vol = row.get("volume", 0)
            ask_vol = row.get("ask_volume", vol * 0.5)
            bid_vol = row.get("bid_volume", vol * 0.5)

            current_high = max(current_high, row["high"])
            current_low = min(current_low, row["low"])
            cum_volume += vol
            cum_delta += (ask_vol - bid_vol)
            cum_ticks += 1

            while (current_high - current_low) >= self.range_points:
                if price >= current_open:
                    bar_close = current_low + self.range_points
                    bar_high = bar_close
                    bar_low = current_low
                else:
                    bar_close = current_high - self.range_points
                    bar_low = bar_close
                    bar_high = current_high

                end_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
                duration = 0.0
                if start_time and end_time:
                    duration = (end_time - start_time).total_seconds()

                bars.append(RangeBar(
                    open=current_open,
                    high=bar_high,
                    low=bar_low,
                    close=bar_close,
                    volume=max(cum_volume, 1),
                    delta=cum_delta,
                    duration_seconds=max(duration, 1),
                    tick_count=cum_ticks,
                    bar_index=bar_idx,
                    timestamp=end_time
                ))

                bar_idx += 1
                current_open = bar_close
                current_high = max(price, bar_close)
                current_low = min(price, bar_close)
                cum_volume = 0
                cum_delta = 0
                cum_ticks = 0
                start_time = end_time

        return bars

    def from_synthetic(self, n_bars: int = 500, seed: int = 42) -> List[RangeBar]:
        """Generate synthetic range bars for demonstration/testing."""
        rng = np.random.RandomState(seed)
        bars = []
        price = 15000.0  # Starting NQ price

        for i in range(n_bars):
            trend_bias = 0.1 * np.sin(i / 80)
            move = (rng.randn() * 0.4 + trend_bias) * self.range_points
            bar_open = price
            bar_close = price + move

            if bar_close > bar_open:
                bar_high = bar_open + self.range_points
                bar_low = bar_open
                bar_close = bar_high
            else:
                bar_low = bar_open - self.range_points
                bar_high = bar_open
                bar_close = bar_low

            base_vol = 800 + 400 * np.abs(np.sin(i / 30))
            vol_noise = rng.exponential(200)
            volume = base_vol + vol_noise

            # Simulate delta: correlated with direction but noisy
            direction = 1 if bar_close > bar_open else -1
            delta = direction * volume * (0.1 + 0.3 * rng.rand()) + rng.randn() * 100

            # Simulate duration: inversely related to volume (more vol = faster fill)
            duration = max(5, 60 / (volume / 500) + rng.randn() * 10)

            # Inject absorption events (high vol, reversal coming)
            if rng.rand() < 0.08:
                volume *= 3.0
                delta *= 0.2  # High volume but weak delta = absorption
                duration *= 0.5

            bars.append(RangeBar(
                open=round(bar_open, 2),
                high=round(bar_high, 2),
                low=round(bar_low, 2),
                close=round(bar_close, 2),
                volume=round(volume),
                delta=round(delta),
                duration_seconds=round(duration, 1),
                tick_count=int(volume / 10),
                bar_index=i
            ))
            price = bar_close

        return bars


# =============================================================================
# EMA & Statistical Utilities
# =============================================================================

def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    return result


def rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    result = np.full_like(data, np.nan, dtype=float)
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1:i + 1])
    return result


def normalize_series(data: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = np.nanmin(data), np.nanmax(data)
    if mx - mn < 1e-10:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn)


# =============================================================================
# Deep-M Effort Engine
# =============================================================================

class DeepMEffortEngine:
    """
    Core computation engine for the Deep-M Effort indicator.

    Effort is decomposed into three components:
      1. Volume Effort  — raw contracts per range bar
      2. Delta Effort   — net directional pressure (abs + signed)
      3. Time Effort    — speed of bar completion (inverse of duration)

    These are combined into a composite Effort Index, smoothed with EMA,
    and compared against dynamic thresholds to produce zones.
    """

    def __init__(self, config: DeepMEffortConfig):
        self.config = config
        self.zones: List[EffortZone] = []

    def compute(self, bars: List[RangeBar]) -> pd.DataFrame:
        """Run the full indicator computation. Returns a DataFrame of all metrics."""
        n = len(bars)

        # Extract raw arrays
        opens = np.array([b.open for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        closes = np.array([b.close for b in bars])
        volumes = np.array([b.volume for b in bars], dtype=float)
        deltas = np.array([b.delta for b in bars], dtype=float)
        durations = np.array([b.duration_seconds for b in bars], dtype=float)

        # Direction: +1 bullish, -1 bearish
        direction = np.where(closes >= opens, 1.0, -1.0)

        # ----- Volume Effort -----
        vol_ema = ema(volumes, self.config.ema_period)
        vol_ratio = np.where(vol_ema > 0, volumes / vol_ema, 1.0)

        # ----- Delta Effort -----
        abs_delta = np.abs(deltas)
        delta_ema_line = ema(abs_delta, self.config.delta_ema_period)
        # Signed delta as % of volume
        delta_pct = np.where(volumes > 0, deltas / volumes, 0.0)
        delta_pct_ema = ema(delta_pct, self.config.delta_ema_period)
        # Delta divergence: high volume but weak delta
        delta_divergence = np.where(
            abs_delta > 0,
            vol_ratio / (np.abs(delta_pct) + 0.01),
            0.0
        )

        # ----- Time Effort (Speed) -----
        speed = np.where(durations > 0, 1.0 / durations, 0.0)
        speed_ema = ema(speed, self.config.time_ema_period)
        speed_ratio = np.where(speed_ema > 0, speed / speed_ema, 1.0)

        # ----- Composite Effort Index -----
        # Weighted combination: volume effort is primary
        effort_index = (
            0.50 * normalize_series(vol_ratio) +
            0.25 * normalize_series(delta_divergence) +
            0.25 * normalize_series(speed_ratio)
        )
        effort_ema = ema(effort_index, self.config.ema_period)
        effort_std = rolling_std(effort_index, self.config.ema_period)

        # ----- Upper/Lower Bands -----
        effort_upper = effort_ema + self.config.zone_std_mult * effort_std
        effort_lower = effort_ema - self.config.zone_std_mult * effort_std

        # ----- Absorption Score -----
        # High when: high volume + weak delta + slow bar
        absorption_score = vol_ratio * (1.0 - np.abs(delta_pct)) * (1.0 / (speed_ratio + 0.1))
        absorption_ema = ema(absorption_score, self.config.ema_period)

        # ----- Build DataFrame -----
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "delta": deltas, "duration": durations,
            "direction": direction,
            "vol_ema": vol_ema, "vol_ratio": vol_ratio,
            "abs_delta": abs_delta, "delta_ema": delta_ema_line,
            "delta_pct": delta_pct, "delta_pct_ema": delta_pct_ema,
            "delta_divergence": delta_divergence,
            "speed": speed, "speed_ema": speed_ema, "speed_ratio": speed_ratio,
            "effort_index": effort_index,
            "effort_ema": effort_ema,
            "effort_upper": effort_upper,
            "effort_lower": effort_lower,
            "absorption_score": absorption_score,
            "absorption_ema": absorption_ema,
        })

        # ----- Detect Zones -----
        self.zones = self._detect_zones(df, bars)
        return df

    def _detect_zones(self, df: pd.DataFrame, bars: List[RangeBar]) -> List[EffortZone]:
        """Detect effort-based zones from computed metrics."""
        zones = []
        n = len(df)
        cfg = self.config

        i = cfg.ema_period  # Start after EMA warmup
        while i < n:
            row = df.iloc[i]
            bar = bars[i]

            # --- Absorption Zone Detection ---
            if row["vol_ratio"] >= cfg.absorption_threshold and abs(row["delta_pct"]) < cfg.delta_filter:
                # Find extent of absorption cluster
                j = i + 1
                while j < n and df.iloc[j]["vol_ratio"] >= cfg.absorption_threshold * cfg.abs_continuation:
                    j += 1

                cluster_bars = bars[i:j]
                zone_high = max(b.high for b in cluster_bars) + cfg.tick_size * 4
                zone_low = min(b.low for b in cluster_bars) - cfg.tick_size * 4

                if (zone_high - zone_low) >= cfg.min_zone_width * cfg.tick_size:
                    avg_dir = np.mean(df.iloc[i:j]["direction"])
                    z_type = (ZoneType.ABSORPTION_SUPPORT if avg_dir < 0
                              else ZoneType.ABSORPTION_RESISTANCE)

                    strength = min(1.0, np.mean(df.iloc[i:j]["vol_ratio"]) / (cfg.absorption_threshold * 2))
                    avg_delta = float(np.mean(df.iloc[i:j]["delta"]))

                    zones.append(EffortZone(
                        zone_type=z_type,
                        price_high=zone_high,
                        price_low=zone_low,
                        bar_index_start=i,
                        bar_index_end=j - 1,
                        strength=strength,
                        volume_effort=float(np.sum(df.iloc[i:j]["volume"])),
                        delta_effort=avg_delta,
                    ))
                    i = j
                    continue

            # --- Vacuum Zone Detection (low effort moves) ---
            if row["vol_ratio"] <= cfg.vacuum_threshold and row["speed_ratio"] > cfg.speed_filter:
                j = i + 1
                while j < n and df.iloc[j]["vol_ratio"] <= cfg.vacuum_threshold * cfg.vac_continuation:
                    j += 1

                if j - i >= cfg.min_zone_bars:
                    cluster_bars = bars[i:j]
                    zone_high = max(b.high for b in cluster_bars)
                    zone_low = min(b.low for b in cluster_bars)

                    avg_dir = np.mean(df.iloc[i:j]["direction"])
                    z_type = (ZoneType.VACUUM_BULLISH if avg_dir > 0
                              else ZoneType.VACUUM_BEARISH)

                    strength = min(1.0, 1.0 - np.mean(df.iloc[i:j]["vol_ratio"]))

                    zones.append(EffortZone(
                        zone_type=z_type,
                        price_high=zone_high,
                        price_low=zone_low,
                        bar_index_start=i,
                        bar_index_end=j - 1,
                        strength=strength,
                        volume_effort=float(np.sum(df.iloc[i:j]["volume"])),
                        delta_effort=float(np.mean(df.iloc[i:j]["delta"])),
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
    Generates trading signals from effort zones and current price action.

    Trading Rules:
      1. LONG  when price enters an ABSORPTION_SUPPORT zone from above
         and delta starts turning positive.
      2. SHORT when price enters an ABSORPTION_RESISTANCE zone from below
         and delta starts turning negative.
      3. CONTINUATION LONG  in VACUUM_BULLISH zones (path of least resistance).
      4. CONTINUATION SHORT in VACUUM_BEARISH zones.
      5. EXIT when effort_index crosses back through effort_ema
         after being in an extreme zone.
    """

    @staticmethod
    def generate(df: pd.DataFrame, zones: List[EffortZone]) -> pd.DataFrame:
        n = len(df)
        signals = pd.DataFrame({
            "signal": np.zeros(n, dtype=int),          # +1 long, -1 short, 0 flat
            "signal_type": [""] * n,
            "zone_strength": np.zeros(n),
        })

        for zone in zones:
            if not zone.active:
                continue

            start = zone.bar_index_start
            end = min(zone.bar_index_end + 1, n)

            if zone.zone_type == ZoneType.ABSORPTION_SUPPORT:
                # Price entering support zone → potential long
                for k in range(start, end):
                    if df.iloc[k]["delta_pct"] > df.iloc[k]["delta_pct_ema"]:
                        signals.at[k, "signal"] = 1
                        signals.at[k, "signal_type"] = "ABSORPTION_LONG"
                        signals.at[k, "zone_strength"] = zone.strength
                        break

            elif zone.zone_type == ZoneType.ABSORPTION_RESISTANCE:
                for k in range(start, end):
                    if df.iloc[k]["delta_pct"] < df.iloc[k]["delta_pct_ema"]:
                        signals.at[k, "signal"] = -1
                        signals.at[k, "signal_type"] = "ABSORPTION_SHORT"
                        signals.at[k, "zone_strength"] = zone.strength
                        break

            elif zone.zone_type == ZoneType.VACUUM_BULLISH:
                mid = (start + end) // 2
                if mid < n:
                    signals.at[mid, "signal"] = 1
                    signals.at[mid, "signal_type"] = "VACUUM_CONTINUATION_LONG"
                    signals.at[mid, "zone_strength"] = zone.strength

            elif zone.zone_type == ZoneType.VACUUM_BEARISH:
                mid = (start + end) // 2
                if mid < n:
                    signals.at[mid, "signal"] = -1
                    signals.at[mid, "signal_type"] = "VACUUM_CONTINUATION_SHORT"
                    signals.at[mid, "zone_strength"] = zone.strength

        return signals


# =============================================================================
# Visualization
# =============================================================================

class DeepMEffortChart:
    """Multi-panel chart mimicking the DeepCharts visual style."""

    ZONE_COLORS = {
        ZoneType.ABSORPTION_SUPPORT: ("#2196F3", 0.20),       # Blue
        ZoneType.ABSORPTION_RESISTANCE: ("#F44336", 0.20),    # Red
        ZoneType.VACUUM_BULLISH: ("#4CAF50", 0.12),           # Green
        ZoneType.VACUUM_BEARISH: ("#FF9800", 0.12),           # Orange
    }

    def __init__(self, df: pd.DataFrame, zones: List[EffortZone],
                 signals: pd.DataFrame, config: DeepMEffortConfig):
        self.df = df
        self.zones = zones
        self.signals = signals
        self.config = config

    def plot(self, start: int = 0, end: Optional[int] = None,
             figsize: Tuple[int, int] = (22, 16)):
        if end is None:
            end = len(self.df)
        df = self.df.iloc[start:end].copy()
        df = df.reset_index(drop=True)
        x = np.arange(len(df))

        fig, axes = plt.subplots(4, 1, figsize=figsize, height_ratios=[4, 1.5, 1.2, 1.2],
                                 facecolor="#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333355")

        # ---- Panel 1: Price (Range Bars) + Zones + Signals ----
        ax_price = axes[0]
        self._draw_range_bars(ax_price, df, x)
        self._draw_zones(ax_price, start, end)
        self._draw_signals(ax_price, df, x, start)

        price_ema = ema(df["close"].values, self.config.ema_period)
        ax_price.plot(x, price_ema, color="#FFD700", linewidth=1.2, alpha=0.8, label="Price EMA")
        ax_price.set_ylabel("Price", color="white", fontsize=10)
        ax_price.set_title("Deep-M Effort NQ — Range 40 | Effort Analysis",
                           color="white", fontsize=14, fontweight="bold", pad=12)
        ax_price.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
                        edgecolor="#555", labelcolor="white")

        # ---- Panel 2: Volume + Volume EMA ----
        ax_vol = axes[1]
        colors_vol = np.where(df["direction"] > 0, "#26A69A", "#EF5350")
        ax_vol.bar(x, df["volume"], color=colors_vol, alpha=0.7, width=0.8)
        ax_vol.plot(x, df["vol_ema"], color="#FFD700", linewidth=1.0, label="Vol EMA")
        ax_vol.set_ylabel("Volume", color="white", fontsize=10)
        ax_vol.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
                      edgecolor="#555", labelcolor="white")

        # ---- Panel 3: Effort Index + EMA + Bands ----
        ax_effort = axes[2]
        ax_effort.fill_between(x, df["effort_upper"], df["effort_lower"],
                               color="#7B1FA2", alpha=0.15, label="Effort Bands")
        ax_effort.plot(x, df["effort_index"], color="#00BCD4", linewidth=1.0,
                       alpha=0.9, label="Effort Index")
        ax_effort.plot(x, df["effort_ema"], color="#FFD700", linewidth=1.0,
                       alpha=0.8, label="Effort EMA")
        ax_effort.axhline(y=0.5, color="#555", linestyle="--", linewidth=0.5)
        ax_effort.set_ylabel("Effort", color="white", fontsize=10)
        ax_effort.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e",
                         edgecolor="#555", labelcolor="white")

        # ---- Panel 4: Delta + Absorption Score ----
        ax_delta = axes[3]
        colors_delta = np.where(df["delta"] > 0, "#26A69A", "#EF5350")
        ax_delta.bar(x, df["delta"], color=colors_delta, alpha=0.6, width=0.8)
        ax_abs = ax_delta.twinx()
        ax_abs.plot(x, df["absorption_score"], color="#AB47BC", linewidth=1.0,
                    alpha=0.8, label="Absorption")
        ax_abs.plot(x, df["absorption_ema"], color="#FFD700", linewidth=0.8,
                    alpha=0.7, label="Absorption EMA")
        ax_delta.set_ylabel("Delta", color="white", fontsize=10)
        ax_abs.set_ylabel("Absorption", color="#AB47BC", fontsize=10)
        ax_abs.tick_params(colors="#AB47BC", labelsize=8)
        ax_delta.set_xlabel("Bar Index", color="white", fontsize=10)

        lines_abs, labels_abs = ax_abs.get_legend_handles_labels()
        ax_abs.legend(lines_abs, labels_abs, loc="upper right", fontsize=8,
                      facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

        plt.tight_layout()
        plt.savefig("deep_m_effort_nq.png", dpi=150, bbox_inches="tight",
                    facecolor="#1a1a2e")
        plt.show()
        print(f"[Chart saved: deep_m_effort_nq.png]")

    def _draw_range_bars(self, ax, df, x):
        for i in range(len(df)):
            o, h, l, c = df.iloc[i][["open", "high", "low", "close"]]
            color = "#26A69A" if c >= o else "#EF5350"
            ax.plot([x[i], x[i]], [l, h], color=color, linewidth=0.8)
            body_bottom = min(o, c)
            body_height = abs(c - o)
            rect = plt.Rectangle((x[i] - 0.35, body_bottom), 0.7,
                                 max(body_height, self.config.tick_size * 0.5),
                                 facecolor=color, edgecolor=color, linewidth=0.5)
            ax.add_patch(rect)

    def _draw_zones(self, ax, start, end):
        for zone in self.zones:
            if zone.bar_index_end < start or zone.bar_index_start > end:
                continue
            color, base_alpha = self.ZONE_COLORS.get(
                zone.zone_type, ("#888888", 0.1))
            alpha = base_alpha + zone.strength * 0.15

            x_start = max(zone.bar_index_start - start, 0)
            x_end = min(end - start, zone.bar_index_end - start + 30)  # Extend zone

            rect = plt.Rectangle(
                (x_start, zone.price_low),
                x_end - x_start,
                zone.price_high - zone.price_low,
                facecolor=color, alpha=alpha, edgecolor=color,
                linewidth=0.5, linestyle="--"
            )
            ax.add_patch(rect)

    def _draw_signals(self, ax, df, x, offset):
        sigs = self.signals.iloc[offset:offset + len(df)].reset_index(drop=True)
        for i in range(len(sigs)):
            sig = sigs.iloc[i]
            if sig["signal"] == 1:
                ax.scatter(x[i], df.iloc[i]["low"] - 3, marker="^",
                           color="#00E676", s=80, zorder=5, edgecolors="white",
                           linewidths=0.5)
            elif sig["signal"] == -1:
                ax.scatter(x[i], df.iloc[i]["high"] + 3, marker="v",
                           color="#FF1744", s=80, zorder=5, edgecolors="white",
                           linewidths=0.5)


# =============================================================================
# Sierra Chart C++ Study Export
# =============================================================================

def generate_sierra_chart_cpp() -> str:
    """
    Generate a Sierra Chart ACSF (Advanced Custom Study Function) C++ source
    that implements the Deep-M Effort indicator natively.
    """
    code = r"""
// ==========================================================================
// Deep-M Effort NQ — Sierra Chart Advanced Custom Study
// File: DeepMEffort_NQ.cpp
// Compile as a Sierra Chart DLL study.
// Chart Setup: NQ futures, Range 40 bar type, Volume at Price enabled.
// ==========================================================================

#include "SierraChart.h"

SCDLLName("Deep-M Effort NQ")

SCSFExport scsf_DeepMEffort_NQ(SCStudyInterfaceRef sc)
{
    // --- Subgraphs ---
    SCSubgraphRef EffortIndex   = sc.Subgraph[0];
    SCSubgraphRef EffortEMA     = sc.Subgraph[1];
    SCSubgraphRef EffortUpper   = sc.Subgraph[2];
    SCSubgraphRef EffortLower   = sc.Subgraph[3];
    SCSubgraphRef VolRatio      = sc.Subgraph[4];
    SCSubgraphRef AbsorptionSc  = sc.Subgraph[5];
    SCSubgraphRef ZoneSup       = sc.Subgraph[6];
    SCSubgraphRef ZoneRes       = sc.Subgraph[7];

    // --- Inputs ---
    SCInputRef EMAPeriod           = sc.Input[0];
    SCInputRef ZoneStdMult         = sc.Input[1];
    SCInputRef AbsorptionThreshold = sc.Input[2];
    SCInputRef VacuumThreshold     = sc.Input[3];

    if (sc.SetDefaults)
    {
        sc.GraphName = "Deep-M Effort NQ";
        sc.AutoLoop = 1;
        sc.GraphRegion = 1;  // Separate region below price
        sc.FreeDivisor = 10;

        EffortIndex.Name = "Effort Index";
        EffortIndex.DrawStyle = DRAWSTYLE_LINE;
        EffortIndex.PrimaryColor = RGB(0, 188, 212);
        EffortIndex.LineWidth = 2;

        EffortEMA.Name = "Effort EMA";
        EffortEMA.DrawStyle = DRAWSTYLE_LINE;
        EffortEMA.PrimaryColor = RGB(255, 215, 0);
        EffortEMA.LineWidth = 1;

        EffortUpper.Name = "Upper Band";
        EffortUpper.DrawStyle = DRAWSTYLE_LINE;
        EffortUpper.PrimaryColor = RGB(123, 31, 162);
        EffortUpper.LineWidth = 1;

        EffortLower.Name = "Lower Band";
        EffortLower.DrawStyle = DRAWSTYLE_LINE;
        EffortLower.PrimaryColor = RGB(123, 31, 162);
        EffortLower.LineWidth = 1;

        VolRatio.Name = "Volume Ratio";
        VolRatio.DrawStyle = DRAWSTYLE_IGNORE;

        AbsorptionSc.Name = "Absorption Score";
        AbsorptionSc.DrawStyle = DRAWSTYLE_LINE;
        AbsorptionSc.PrimaryColor = RGB(171, 71, 188);
        AbsorptionSc.LineWidth = 1;

        ZoneSup.Name = "Support Zone";
        ZoneSup.DrawStyle = DRAWSTYLE_COLOR_BAR_HLC_VALUE;
        ZoneSup.PrimaryColor = RGB(33, 150, 243);

        ZoneRes.Name = "Resistance Zone";
        ZoneRes.DrawStyle = DRAWSTYLE_COLOR_BAR_HLC_VALUE;
        ZoneRes.PrimaryColor = RGB(244, 67, 54);

        EMAPeriod.Name = "EMA Period";
        EMAPeriod.SetInt(20);
        EMAPeriod.SetIntLimits(5, 100);

        ZoneStdMult.Name = "Zone Std Multiplier";
        ZoneStdMult.SetFloat(1.5f);
        ZoneStdMult.SetFloatLimits(0.5f, 4.0f);

        AbsorptionThreshold.Name = "Absorption Threshold";
        AbsorptionThreshold.SetFloat(1.8f);
        AbsorptionThreshold.SetFloatLimits(1.0f, 5.0f);

        VacuumThreshold.Name = "Vacuum Threshold";
        VacuumThreshold.SetFloat(0.5f);
        VacuumThreshold.SetFloatLimits(0.1f, 1.0f);

        return;
    }

    int period = EMAPeriod.GetInt();
    int idx = sc.Index;
    if (idx < period) return;

    float volume = sc.Volume[idx];
    float askVol = sc.AskVolume[idx];
    float bidVol = sc.BidVolume[idx];
    float delta  = askVol - bidVol;
    float deltaPct = (volume > 0) ? delta / volume : 0.0f;

    // Volume EMA (stored in extra array)
    sc.ExponentialMovAvg(sc.Volume, sc.Subgraph[10], period);
    float volEma = sc.Subgraph[10][idx];
    float volRatio = (volEma > 0) ? volume / volEma : 1.0f;
    VolRatio[idx] = volRatio;

    // Duration approximation (using number of trades as proxy)
    float numTrades = (float)sc.NumberOfTrades[idx];
    float speed = (numTrades > 0) ? volume / numTrades : 1.0f;

    // Delta divergence
    float absDelta = fabs(delta);
    float deltaDivergence = (fabs(deltaPct) > 0.001f)
        ? volRatio / (fabs(deltaPct) + 0.01f) : 0.0f;

    // Store intermediate for normalization across the array
    sc.Subgraph[11][idx] = volRatio;
    sc.Subgraph[12][idx] = deltaDivergence;
    sc.Subgraph[13][idx] = speed;

    // Composite Effort Index (simplified — full normalization in extra pass)
    float effortRaw = 0.50f * volRatio + 0.25f * deltaDivergence * 0.1f
                    + 0.25f * speed * 0.01f;
    EffortIndex[idx] = effortRaw;

    // EMA of effort
    sc.ExponentialMovAvg(EffortIndex, EffortEMA, period);

    // Rolling Std for bands
    float sumSq = 0, sumVal = 0;
    for (int k = idx - period + 1; k <= idx; k++) {
        float v = EffortIndex[k];
        sumVal += v;
        sumSq += v * v;
    }
    float mean = sumVal / period;
    float variance = sumSq / period - mean * mean;
    float stdDev = (variance > 0) ? sqrtf(variance) : 0.0f;

    EffortUpper[idx] = EffortEMA[idx] + ZoneStdMult.GetFloat() * stdDev;
    EffortLower[idx] = EffortEMA[idx] - ZoneStdMult.GetFloat() * stdDev;

    // Absorption Score
    float absScore = volRatio * (1.0f - fabs(deltaPct));
    AbsorptionSc[idx] = absScore;

    // Zone marking on price chart via color bars
    if (volRatio >= AbsorptionThreshold.GetFloat() && fabs(deltaPct) < 0.15f) {
        if (sc.Close[idx] < sc.Open[idx])
            ZoneSup[idx] = sc.Low[idx];  // Potential support
        else
            ZoneRes[idx] = sc.High[idx]; // Potential resistance
    }

    // Drawing tool for persistent zones (use sc.UseTool for rectangles)
    if (volRatio >= AbsorptionThreshold.GetFloat() && fabs(deltaPct) < 0.15f) {
        s_UseTool tool;
        tool.Clear();
        tool.ChartNumber = sc.ChartNumber;
        tool.DrawingType = DRAWING_RECTANGLEHIGHLIGHT;
        tool.LineNumber = 70000 + idx;
        tool.BeginIndex = idx - 1;
        tool.EndIndex = idx + 30;
        tool.BeginValue = sc.Low[idx] - sc.TickSize * 4;
        tool.EndValue = sc.High[idx] + sc.TickSize * 4;
        tool.Color = (sc.Close[idx] < sc.Open[idx])
            ? RGB(33, 150, 243) : RGB(244, 67, 54);
        tool.TransparencyLevel = 80;
        tool.AddAsUserDrawnDrawing = 0;
        sc.UseTool(tool);
    }
}
"""
    return code


# =============================================================================
# Performance Summary
# =============================================================================

def performance_summary(df: pd.DataFrame, signals: pd.DataFrame,
                        config: DeepMEffortConfig) -> pd.DataFrame:
    """Simple backtest of signal performance."""
    trades = []
    position = 0
    entry_price = 0
    entry_bar = 0

    for i in range(len(signals)):
        sig = signals.iloc[i]["signal"]
        if sig != 0 and position == 0:
            position = int(sig)
            entry_price = df.iloc[i]["close"]
            entry_bar = i
        elif position != 0 and i - entry_bar >= 10:
            exit_price = df.iloc[i]["close"]
            pnl = (exit_price - entry_price) * position
            pnl_ticks = pnl / config.tick_size
            trades.append({
                "entry_bar": entry_bar, "exit_bar": i,
                "direction": "LONG" if position > 0 else "SHORT",
                "entry": entry_price, "exit": exit_price,
                "pnl_points": round(pnl, 2),
                "pnl_ticks": round(pnl_ticks, 1),
                "signal_type": signals.iloc[entry_bar]["signal_type"],
            })
            position = 0

    if not trades:
        print("No trades generated.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_points"] > 0]

    print("\n" + "=" * 60)
    print("  DEEP-M EFFORT NQ — PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  Total Trades:   {len(trades_df)}")
    print(f"  Winners:        {len(wins)} ({100*len(wins)/len(trades_df):.1f}%)")
    print(f"  Total PnL:      {trades_df['pnl_points'].sum():.2f} pts "
          f"({trades_df['pnl_ticks'].sum():.0f} ticks)")
    print(f"  Avg PnL/Trade:  {trades_df['pnl_points'].mean():.2f} pts")
    print(f"  Max Win:        {trades_df['pnl_points'].max():.2f} pts")
    print(f"  Max Loss:       {trades_df['pnl_points'].min():.2f} pts")
    print(f"  Profit Factor:  ", end="")
    gross_loss = abs(trades_df[trades_df["pnl_points"] < 0]["pnl_points"].sum())
    if gross_loss > 0:
        print(f"{wins['pnl_points'].sum() / gross_loss:.2f}")
    else:
        print("INF")
    print("=" * 60)
    print("\nBy Signal Type:")
    print(trades_df.groupby("signal_type")["pnl_points"].agg(["count", "sum", "mean"]).round(2))
    return trades_df


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║             DEEP-M EFFORT NQ — Indicator Engine                 ║
║                                                                  ║
║  Methodology: Wyckoff Effort vs Result on Range-40 Bars          ║
║  Components:  Volume Effort · Delta Effort · Time Effort         ║
║  Output:      Effort Zones, Absorption Areas, Trade Signals      ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # --- Configuration ---
    config = DeepMEffortConfig(
        range_size=40,
        tick_size=0.25,
        ema_period=20,
        zone_std_mult=1.5,
        absorption_threshold=1.5,
        vacuum_threshold=0.6,
    )

    # --- Build Range Bars (synthetic for demo; replace with real data) ---
    builder = RangeBarBuilder(config)
    bars = builder.from_synthetic(n_bars=500, seed=42)
    print(f"Built {len(bars)} range bars (range={config.range_size} ticks)")

    # --- Compute Effort Metrics ---
    engine = DeepMEffortEngine(config)
    df = engine.compute(bars)
    zones = engine.zones
    print(f"Detected {len(zones)} effort zones:")
    for z in zones:
        print(f"  {z.zone_type.value:30s}  "
              f"[{z.price_low:.2f} — {z.price_high:.2f}]  "
              f"strength={z.strength:.2f}  bars={z.bar_index_start}-{z.bar_index_end}")

    # --- Generate Signals ---
    signals = SignalGenerator.generate(df, zones)
    n_long = (signals["signal"] == 1).sum()
    n_short = (signals["signal"] == -1).sum()
    print(f"\nSignals: {n_long} long, {n_short} short")

    # --- Performance ---
    trades_df = performance_summary(df, signals, config)

    # --- Chart ---
    chart = DeepMEffortChart(df, zones, signals, config)
    chart.plot(start=50, end=350)

    # --- Export Sierra Chart C++ ---
    cpp_code = generate_sierra_chart_cpp()
    with open("DeepMEffort_NQ.cpp", "w") as f:
        f.write(cpp_code)
    print("\n[Sierra Chart C++ study exported: DeepMEffort_NQ.cpp]")

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  RATIONALE & TRADING METHODOLOGY                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. RANGE BARS (40 ticks) normalize time, letting us focus       ║
║     purely on price-volume dynamics.                             ║
║                                                                  ║
║  2. EFFORT = volume needed to complete each range bar.           ║
║     - High effort + small delta = ABSORPTION (institutions       ║
║       absorbing orders without moving price — Wyckoff).          ║
║     - Low effort + fast completion = VACUUM (no resistance,      ║
║       price moves freely — continuation expected).               ║
║                                                                  ║
║  3. The EMA of effort creates a dynamic baseline. Deviations     ║
║     above/below flag anomalous activity.                         ║
║                                                                  ║
║  4. ZONES are generated where clusters of extreme effort occur:  ║
║     - Blue zones  = support (buying absorption)                  ║
║     - Red zones   = resistance (selling absorption)              ║
║     - Green/Orange = vacuum continuation areas                   ║
║                                                                  ║
║  5. TRADE ENTRY: Enter when price revisits a zone and delta      ║
║     confirms direction. The zone's strength (volume intensity)   ║
║     determines conviction level.                                 ║
║                                                                  ║
║  6. This mimics what DeepCharts' Deep-M Effort does:             ║
║     analyzing the *effort* behind each range bar to identify     ║
║     where smart money is active, creating actionable zones       ║
║     on the NQ futures chart.                                     ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
