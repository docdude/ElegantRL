"""
Weis Wave & Wyckoff data models — enums and configuration classes.

Mirrors the structure of srlcarlg's models_utils/ww_models.py with
adaptations for our analyzer.
"""

from enum import Enum

from .custom_mas import MAType


# ─── Direction ────────────────────────────────────────────────────────────

class Direction(Enum):
    UP = 1
    DOWN = 2


# ─── ZigZag ──────────────────────────────────────────────────────────────

class ZigZagMode(Enum):
    """How the ZigZag identifies wave reversals."""
    Percentage = 1      # Reversal if price retraces >= pct_value %
    Points = 2          # Reversal if price retraces >= points_value


class ZigZagInit:
    """State holder for ZigZag trendline computation."""

    def __init__(self, mode: ZigZagMode = ZigZagMode.Percentage,
                 pct_value: float = 0.5,
                 points_value: float = 25.0):
        """
        Parameters
        ----------
        mode : ZigZagMode
            How reversals are detected.
        pct_value : float
            Reversal threshold as percentage (for ZigZagMode.Percentage).
        points_value : float
            Reversal threshold in absolute points (for ZigZagMode.Points).
        """
        self.mode = mode
        self.pct_value = pct_value
        self.points_value = points_value
        # Mutable state for the iteration loop
        self.extremum_index = 0
        self.extremum_price = 0.0
        self.direction = Direction.UP

    def reset(self):
        self.extremum_index = 0
        self.extremum_price = 0.0
        self.direction = Direction.UP


# ─── Waves ───────────────────────────────────────────────────────────────

class YellowWaves(Enum):
    """Which wave value to push into the sliding window for large-wave detection."""
    UseCurrent = 1          # Use current wave's value
    UsePrev_SameWave = 2    # Use previous same-direction wave's value
    UsePrev_InvertWave = 3  # Use previous opposite-direction wave's value


class WavesInit:
    """Configuration and running state for Weis wave segmentation."""

    def __init__(self,
                 yellow_waves: YellowWaves = YellowWaves.UseCurrent,
                 large_wave_ratio: float = 1.5,
                 is_open_time: bool = False):
        """
        Parameters
        ----------
        yellow_waves : YellowWaves
            Which wave value to push into the 4-wave sliding window.
        large_wave_ratio : float
            Multiplier vs avg of prior 4 waves to trigger yellow coloring.
        is_open_time : bool
            Whether the DataFrame index represents bar open time (True)
            or bar close time (False).
        """
        self.yellow_waves = yellow_waves
        self.large_wave_ratio = large_wave_ratio
        self.is_open_time = is_open_time
        # Running state
        self.prev_waves_volume = [0.0, 0.0, 0.0, 0.0]
        self.prev_waves_er = [0.0, 0.0, 0.0, 0.0]
        self.prev_wave_up = (0.0, 0.0)    # (vol, er) of last up wave
        self.prev_wave_down = (0.0, 0.0)  # (vol, er) of last down wave
        self.trend_start_index = 0

    def reset(self):
        self.prev_waves_volume = [0.0, 0.0, 0.0, 0.0]
        self.prev_waves_er = [0.0, 0.0, 0.0, 0.0]
        self.prev_wave_up = (0.0, 0.0)
        self.prev_wave_down = (0.0, 0.0)
        self.trend_start_index = 0


# ─── Strength Filter (Wyckoff bar-level analysis) ────────────────────────

class FilterType(Enum):
    """How bar volume/time is normalized before classification."""
    MA = 1                        # value / MA(value)
    StdDev = 2                    # value / StdDev(value)
    Both = 3                      # (value - MA) / StdDev  (z-score)
    Normalized_Emphasized = 4     # pct above/below rolling average × multiplier
    L1Norm = 5                    # value / sum(abs(window))


class FilterRatio(Enum):
    """Whether strength thresholds are fixed or percentile-based."""
    Fixed = 1
    Percentage = 2


class StrengthFilter:
    """Configuration for the Wyckoff volume/time strength filter.

    Classifies each bar into 5 tiers (0 = lowest → 4 = ultra)
    based on its volume and time relative to recent history.
    """

    def __init__(self,
                 filter_type: FilterType = FilterType.Normalized_Emphasized,
                 filter_ratio: FilterRatio = FilterRatio.Percentage,
                 ma_type: MAType = MAType.Exponential,
                 ma_period: int = 5,
                 n_period: int = 20,
                 n_multiplier: int = 1,
                 thresholds: tuple = (0.5, 1.2, 2.5, 3.5, 3.51),
                 percentage: tuple = (23.6, 38.2, 61.8, 100, 101),
                 pctile: tuple = (40, 70, 90, 97, 99),
                 is_open_time: bool = False):
        """
        Parameters
        ----------
        filter_type : FilterType
            How volume/time values are normalized.
        filter_ratio : FilterRatio
            Fixed thresholds vs percentile-based.
        ma_type : MAType
            MA type used by FilterType.MA / StdDev / Both.
        ma_period : int
            Period for MA / StdDev / L1Norm.
        n_period : int
            Rolling window for Normalized_Emphasized and FilterRatio.Percentage.
        n_multiplier : int
            Sensitivity multiplier for Normalized_Emphasized.
        thresholds : tuple
            (lowest, low, average, high, ultra) for FilterRatio.Fixed.
        percentage : tuple
            (lowest, low, average, high, ultra) for Normalized_Emphasized.
        pctile : tuple
            (lowest, low, average, high, ultra) for FilterRatio.Percentage.
        is_open_time : bool
            Whether index represents bar open time.
        """
        self.filter_type = filter_type
        self.filter_ratio = filter_ratio
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.is_open_time = is_open_time
        # Fixed thresholds
        self.lowest = thresholds[0]
        self.low = thresholds[1]
        self.average = thresholds[2]
        self.high = thresholds[3]
        self.ultra = thresholds[4]
        # Normalized_Emphasized percentage thresholds
        self.n_period = n_period
        self.n_multiplier = n_multiplier
        self.lowest_pct = percentage[0]
        self.low_pct = percentage[1]
        self.average_pct = percentage[2]
        self.high_pct = percentage[3]
        self.ultra_pct = percentage[4]
        # Percentile thresholds
        self.lowest_pctile = pctile[0]
        self.low_pctile = pctile[1]
        self.average_pctile = pctile[2]
        self.high_pctile = pctile[3]
        self.ultra_pctile = pctile[4]
