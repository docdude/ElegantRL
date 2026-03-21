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
_PIPELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                              "wyckoff_effort", "pipeline")
# Also try the /opt/finrl path
_PIPELINE_DIRS = [
    os.path.abspath(_PIPELINE_DIR),
    "/opt/finrl/wyckoff_effort/pipeline",
    os.path.expanduser("~/wyckoff_effort/pipeline"),
]
for d in _PIPELINE_DIRS:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

from wyckoff_features import build_all_features  # noqa: E402


# Feature indices used in training (36 of 58)
# From run_config.json of the CPCV run
TRAINING_FEATURE_INDICES = [
    0, 1, 4, 5, 8, 9, 10, 11, 13, 14,
    16, 17, 18, 19, 20, 21, 22, 23, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 41, 48, 49, 50,
]
N_TRAINING_FEATURES = len(TRAINING_FEATURE_INDICES)  # 36


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
        ZigZag reversal for Weis Wave (40 for NQ 40pt range bars).
    """

    def __init__(
        self,
        buffer_size: int = 200,
        feature_indices: Optional[list[int]] = None,
        reversal_points: float = 40.0,
    ):
        self.buffer_size = buffer_size
        self.feature_indices = feature_indices or TRAINING_FEATURE_INDICES
        self.reversal_points = reversal_points
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
        tech_ary, feature_names, _ = build_all_features(
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
