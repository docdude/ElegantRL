"""
Live Range Bar Builder — converts tick stream into 40pt NQ range bars.

Matches the bar construction logic in scid_parser.py:
  - Tracks running high/low from the bar's open
  - When high - low >= range_size, the bar closes
  - Bar close is pinned to open ± range_size in the direction of the breakout
  - New bar opens at the previous bar's close
  - Volume is split into ask_volume (uptick) and bid_volume (downtick)
  - Delta = ask_volume - bid_volume, CVD = cumulative delta
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class RangeBar:
    """A completed range bar."""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    ask_volume: float = 0.0
    bid_volume: float = 0.0
    delta: float = 0.0
    num_trades: int = 0
    duration_seconds: float = 0.0
    cvd: float = 0.0
    timestamp: float = 0.0  # unix timestamp of bar close


class RangeBarBuilder:
    """
    Accumulates ticks and emits completed range bars.

    Replicates the exact logic from _range_bar_boundaries_python() in
    scid_parser.py so that live bars match training data.

    Parameters
    ----------
    range_size : float
        Price range for bar completion (40.0 for NQ 40pt bars).
    on_bar : callable, optional
        Callback invoked with each completed RangeBar.
    """

    def __init__(self, range_size: float = 40.0, on_bar: Optional[Callable[[RangeBar], None]] = None):
        self.range_size = range_size
        self.on_bar = on_bar

        # Current bar state
        self._bar_open: Optional[float] = None
        self._bar_high: float = 0.0
        self._bar_low: float = 0.0
        self._volume: float = 0.0
        self._ask_volume: float = 0.0
        self._bid_volume: float = 0.0
        self._num_trades: int = 0
        self._bar_start_time: float = 0.0
        self._cvd: float = 0.0  # cumulative delta (persistent across bars)

        # Completed bars buffer (for polling mode)
        self.completed_bars: list[RangeBar] = []

    def on_tick(self, price: float, size: float, is_uptick: bool, timestamp: float = 0.0):
        """
        Process a single tick.

        Parameters
        ----------
        price : float
            Trade price.
        size : float
            Trade size (contracts).
        is_uptick : bool
            True if trade hit the ask (buyer-initiated), False if bid.
        timestamp : float
            Unix timestamp of the tick.
        """
        if timestamp == 0.0:
            timestamp = time.time()

        # First tick initializes the bar
        if self._bar_open is None:
            self._bar_open = price
            self._bar_high = price
            self._bar_low = price
            self._bar_start_time = timestamp

        # Update running high/low
        if price > self._bar_high:
            self._bar_high = price
        if price < self._bar_low:
            self._bar_low = price

        # Accumulate volume
        self._volume += size
        self._num_trades += 1
        if is_uptick:
            self._ask_volume += size
        else:
            self._bid_volume += size

        # Check if bar completes
        if self._bar_high - self._bar_low >= self.range_size:
            # Determine close direction (matches scid_parser logic)
            if price >= self._bar_open + self.range_size:
                bar_close = self._bar_open + self.range_size
            elif price <= self._bar_open - self.range_size:
                bar_close = self._bar_open - self.range_size
            else:
                bar_close = price

            delta = self._ask_volume - self._bid_volume
            self._cvd += delta

            bar = RangeBar(
                open=self._bar_open,
                high=self._bar_high,
                low=self._bar_low,
                close=bar_close,
                volume=self._volume,
                ask_volume=self._ask_volume,
                bid_volume=self._bid_volume,
                delta=delta,
                num_trades=self._num_trades,
                duration_seconds=timestamp - self._bar_start_time,
                cvd=self._cvd,
                timestamp=timestamp,
            )

            self.completed_bars.append(bar)
            if self.on_bar is not None:
                self.on_bar(bar)

            # New bar opens at previous close
            self._bar_open = bar_close
            self._bar_high = bar_close
            self._bar_low = bar_close
            self._volume = 0.0
            self._ask_volume = 0.0
            self._bid_volume = 0.0
            self._num_trades = 0
            self._bar_start_time = timestamp

    def reset_cvd(self):
        """Reset CVD at session boundaries."""
        self._cvd = 0.0

    @property
    def current_bar_open(self) -> Optional[float]:
        return self._bar_open

    @property
    def n_completed(self) -> int:
        return len(self.completed_bars)
