"""
Live Feature Engine — wraps wyckoff_features.py for incremental bar-by-bar use.

Strategy: maintain a rolling buffer of N recent bars (as a DataFrame).
On each new bar, append it, recompute features via build_all_features(),
and return the latest feature vector.

This guarantees exact parity with training data since it uses the same
code path. The buffer is large enough for all rolling windows (50 bars
for the longest lookback) plus some margin.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
from typing import Optional

# Add the pipeline directory to path so we can import wyckoff_features
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
_PIPELINE_DIRS = [
    os.path.join(_PROJECT_ROOT, "wyckoff_effort", "pipeline"),
    os.path.expanduser("~/wyckoff_effort/pipeline"),
]
for d in _PIPELINE_DIRS:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

from wyckoff_features import build_all_features  # noqa: E402
from wyckoff_features_legacy import build_all_features as build_all_features_legacy  # noqa: E402


# Legacy feature indices (36 of 61) — used by pre-audit models (--legacy)
LEGACY_TRAINING_INDICES = [
    0, 1, 4, 5, 8, 9, 10, 11, 13, 14,
    16, 17, 18, 19, 20, 21, 22, 23, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 41, 48, 49, 50,
]

# New training feature indices (49 of 72)
# Matches ENV_FEATURE_INDICES + Block 6 library Weis Wave features
TRAINING_FEATURE_INDICES = [
    # ── Bar microstructure (10) ──
    0,   # body_ratio
    1,   # upper_wick_ratio
    2,   # lower_wick_ratio
    4,   # delta_ratio
    5,   # vol_vs_ma20
    8,   # duration_norm
    9,   # cvd_slope_fast
    11,  # cvd_divergence
    13,  # return_5
    14,  # volatility_20
    # ── Weis Wave (15) ── includes wave_direction
    15,  # wave_direction
    16,  # wave_progress
    17,  # wave_displacement_norm
    18,  # wave_vol_cumulative_norm
    19,  # wave_delta_ratio
    21,  # wave_vol_vs_prev
    23,  # wave_disp_vs_prev
    27,  # demand_score_3wave
    28,  # supply_score_3wave
    29,  # wave_vol_trend_up
    30,  # wave_vol_trend_down
    34,  # large_wave_score
    58,  # wave_effort_result_raw
    59,  # wave_time_norm
    60,  # wave_velocity_norm
    # ── Wyckoff Events (6) ──
    35,  # spring_score
    36,  # upthrust_score
    37,  # sc_score
    38,  # bc_score
    39,  # absorption_score
    41,  # stopping_action_score
    # ── Range / Phase Context (7) ──
    48,  # pct_in_range
    49,  # range_width_norm
    50,  # bars_in_range
    53,  # phase_accum_score
    54,  # phase_markup_score
    55,  # phase_distrib_score
    56,  # phase_markdown_score
    # ── Block 6: Library Weis Wave (11) ──
    61,  # lib_volume_strength
    62,  # lib_wave_vs_same_dir
    63,  # lib_er_vs_same_dir
    64,  # lib_wave_vs_prev
    65,  # lib_er_vs_prev
    66,  # lib_large_wave
    67,  # lib_large_er
    68,  # lib_pivot_flag
    69,  # lib_exhaust_up
    70,  # lib_exhaust_down
    71,  # lib_in_range
]
N_TRAINING_FEATURES = len(TRAINING_FEATURE_INDICES)  # 49


class LiveFeatureEngine:
    """
    Maintains a buffer of range bars and computes Wyckoff features.

    Parameters
    ----------
    buffer_size : int
        Max bars to keep in buffer. Must be >= max rolling window used
        by feature computations (50 for phase_lookback) + margin.
    feature_indices : list[int]
        Indices into the 58-feature array to select for the model.
    reversal_points : float
        ZigZag reversal for Weis Wave (120 = 3x bar size for NQ 40pt).
    """

    def __init__(
        self,
        buffer_size: int = 200,
        feature_indices: Optional[list[int]] = None,
        reversal_points: float = 120.0,
        legacy: bool = False,
    ):
        self.buffer_size = buffer_size
        if feature_indices is not None:
            self.feature_indices = feature_indices
        elif legacy:
            self.feature_indices = LEGACY_TRAINING_INDICES
        else:
            self.feature_indices = TRAINING_FEATURE_INDICES
        self.reversal_points = reversal_points
        self.legacy = legacy
        self._bars: list[dict] = []

    def add_bar(self, bar) -> Optional[np.ndarray]:
        """
        Add a completed range bar and return the selected feature vector.

        Parameters
        ----------
        bar : RangeBar or dict
            Must have: open, high, low, close, volume, delta,
                       duration_seconds, num_trades, cvd.
            If a RangeBar dataclass, attributes are read directly.

        Returns
        -------
        features : np.ndarray, shape (n_selected_features,)
            Selected features for this bar, or None if insufficient data.
        """
        if hasattr(bar, "open"):
            row = {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "delta": bar.delta,
                "duration_seconds": bar.duration_seconds,
                "num_trades": bar.num_trades,
                "cvd": bar.cvd,
            }
            # Include ask/bid volume if available
            if hasattr(bar, "ask_volume"):
                row["ask_volume"] = bar.ask_volume
                row["bid_volume"] = bar.bid_volume
        else:
            row = dict(bar)

        self._bars.append(row)

        # Trim buffer
        if len(self._bars) > self.buffer_size:
            self._bars = self._bars[-self.buffer_size:]

        # Need at least a few bars for meaningful features
        if len(self._bars) < 5:
            return None

        return self._compute_latest()

    def _compute_latest(self) -> np.ndarray:
        """Recompute features on the full buffer and return the last row."""
        df = pd.DataFrame(self._bars)
        _build = build_all_features_legacy if self.legacy else build_all_features
        tech_ary, feature_names, _ = _build(
            df, reversal_points=self.reversal_points
        )
        # Select training features from last bar
        selected = tech_ary[-1, self.feature_indices]
        return selected.astype(np.float32)

    def get_full_tech_ary(self) -> Optional[np.ndarray]:
        """Return full tech_ary for all buffered bars (selected features)."""
        if len(self._bars) < 5:
            return None
        df = pd.DataFrame(self._bars)
        tech_ary, _, _ = build_all_features(df, reversal_points=self.reversal_points)
        return tech_ary[:, self.feature_indices].astype(np.float32)

    @property
    def n_bars(self) -> int:
        return len(self._bars)

    def reset(self):
        """Clear the bar buffer."""
        self._bars.clear()
