"""
Precomputed replay data — parse SCID → build bars → compute features ONCE.

This module eliminates the three major performance bottlenecks:
  1. Repeated SCID tick parsing per checkpoint (~5M ticks × 3+ checkpoints)
  2. Per-bar build_all_features() on 200-bar buffer (O(bars² × features))
  3. Python for-loop over millions of ticks

Usage:
    from wyckoff_rl.live.precompute import PrecomputedReplay

    # Build once (~30s)
    replay = PrecomputedReplay.from_scid('/path/to/file.scid', '2026-01-15', '2026-03-18')

    # Replay each checkpoint (~0.5s each)
    results = replay.run_checkpoint('path/to/actor.pt', continuous=True)
    capture = replay.run_capture('path/to/actor.pt', continuous=True)
"""

from __future__ import annotations

import logging
import os
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from wyckoff_rl.live.range_bar_builder import RangeBarBuilder
from wyckoff_rl.live.live_features import TRAINING_FEATURE_INDICES, N_TRAINING_FEATURES
from wyckoff_rl.live.inference import InferenceEngine
from wyckoff_rl.feature_config import ALL_FEATURES

logger = logging.getLogger(__name__)

# SCID constants — must match adapters.py exactly
OLE_TO_UNIX_US = 25569 * 86400 * 1_000_000
RECORD_DTYPE = np.dtype([
    ("datetime", "<i8"),
    ("open", "<f4"), ("high", "<f4"), ("low", "<f4"), ("close", "<f4"),
    ("num_trades", "<i4"), ("volume", "<u4"),
    ("bid_volume", "<u4"), ("ask_volume", "<u4"),
])

FEATURE_NAMES = [ALL_FEATURES[i] for i in TRAINING_FEATURE_INDICES]


class PrecomputedReplay:
    """
    Precomputed bar + feature data from a single SCID replay period.

    Holds:
      - features: (n_bars, 36) array of training features per bar
      - windows: (n_windows, 30, 36) sliding windows ready for inference
      - bar metadata: prices, timestamps

    Build with from_scid(), then call run_checkpoint() per model.
    """

    def __init__(
        self,
        features: np.ndarray,
        windows: np.ndarray,
        prices: np.ndarray,
        timestamps: np.ndarray,
        window_offset: int,
    ):
        self.features = features        # (n_bars, 36)
        self.windows = windows           # (n_windows, 30, 36)
        self.prices = prices             # (n_bars,)
        self.timestamps = timestamps     # (n_bars,)
        self.window_offset = window_offset  # first bar index with a full window
        self.n_bars = len(prices)
        self.n_windows = len(windows)

    @classmethod
    def from_scid(
        cls,
        scid_path: str,
        start_date: str,
        end_date: str,
        range_size: float = 40.0,
        window_size: int = 30,
        feature_buffer: int = 200,
        feature_indices: Optional[list[int]] = None,
    ) -> "PrecomputedReplay":
        """
        Parse SCID file → build range bars → compute all features → build windows.

        Parameters
        ----------
        scid_path : str
            Path to .scid file.
        start_date, end_date : str
            Date range (YYYY-MM-DD).
        range_size : float
            Range bar size in points (40.0 for NQ).
        window_size : int
            Sliding window length (30).
        feature_buffer : int
            Bars of context for feature computation warmup (200).
        feature_indices : list[int]
            Which of 58 features to select (default: TRAINING_FEATURE_INDICES).

        Returns
        -------
        PrecomputedReplay
        """
        feat_idx = feature_indices or TRAINING_FEATURE_INDICES
        t0 = time.time()

        # Step 1: Parse SCID exactly like adapters.SCIDReplayAdapter.start()
        # UTC timestamps, searchsorted date filter, same tick loop
        logger.info("Step 1: Loading SCID ticks (adapters-compatible) ...")
        path = Path(scid_path)
        if not path.exists():
            raise FileNotFoundError(f"SCID file not found: {path}")

        with open(path, "rb") as f:
            header = f.read(56)
        if header[:4] != b"SCID":
            raise ValueError(f"Invalid SCID magic: {header[:4]!r}")
        header_size = struct.unpack_from("<I", header, 4)[0]
        record_size = struct.unpack_from("<I", header, 8)[0]
        total = (path.stat().st_size - header_size) // record_size

        data = np.memmap(path, dtype=RECORD_DTYPE, mode="r",
                         offset=header_size, shape=(total,))
        ole_us = data["datetime"]
        unix_us = ole_us.astype(np.int64) - OLE_TO_UNIX_US

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
        end_ts = int((datetime.strptime(end_date, "%Y-%m-%d")
                      .replace(tzinfo=timezone.utc) + timedelta(days=1))
                     .timestamp() * 1_000_000)
        start_idx = int(np.searchsorted(unix_us, start_ts))
        end_idx = int(np.searchsorted(unix_us, end_ts))
        n = end_idx - start_idx
        logger.info(f"  SCID: {n:,} records ({start_date} -> {end_date})")

        closes = np.array(data["close"][start_idx:end_idx], dtype=np.float64)
        volumes = np.array(data["volume"][start_idx:end_idx], dtype=np.float64)
        ask_vols = np.array(data["ask_volume"][start_idx:end_idx], dtype=np.float64)
        timestamps_s = (unix_us[start_idx:end_idx].astype(np.float64)) / 1_000_000
        del data

        builder = RangeBarBuilder(range_size=range_size)
        last_price = 0.0
        for i in range(n):
            price = closes[i]
            vol = volumes[i]
            if vol <= 0 or price <= 0:
                continue
            ask_vol = ask_vols[i]
            is_uptick = ask_vol > (vol - ask_vol) if vol > 0 else (price >= last_price)
            last_price = price
            builder.on_tick(price, vol, is_uptick, timestamps_s[i])

        bars = builder.completed_bars
        t1 = time.time()
        logger.info(f"  Built {len(bars)} range bars in {t1-t0:.1f}s")

        if len(bars) < window_size + 5:
            raise ValueError(f"Only {len(bars)} bars -- need at least {window_size+5}")

        # Step 2: Build DataFrame, compute features ONCE on the full bar series
        logger.info(f"Step 2: Computing features on {len(bars)} bars...")
        bar_df = pd.DataFrame([{
            'open': b.open, 'high': b.high, 'low': b.low, 'close': b.close,
            'volume': b.volume, 'delta': b.delta,
            'duration_seconds': b.duration_seconds,
            'num_trades': b.num_trades, 'cvd': b.cvd,
            'ask_volume': b.ask_volume, 'bid_volume': b.bid_volume,
        } for b in bars])

        from wyckoff_effort.pipeline.wyckoff_features import build_all_features
        tech_ary, _, _ = build_all_features(bar_df, reversal_points=range_size)
        features = tech_ary[:, feat_idx].astype(np.float32)  # (n_bars, 36)
        t2 = time.time()
        logger.info(f"  Computed {features.shape} features in {t2-t1:.1f}s")

        # Step 3: Build sliding windows
        logger.info(f"Step 3: Building {window_size}-bar sliding windows...")
        n_bars = len(bars)
        n_windows = n_bars - window_size + 1
        stride_b, stride_f = features.strides
        windows = np.lib.stride_tricks.as_strided(
            features,
            shape=(n_windows, window_size, features.shape[1]),
            strides=(stride_b, stride_b, stride_f),
        ).copy()  # copy to own memory (contiguous)

        prices = np.array([b.close for b in bars], dtype=np.float64)
        timestamps = np.array([b.timestamp for b in bars], dtype=np.float64)

        window_offset = window_size - 1  # first bar index that has a full window

        t3 = time.time()
        logger.info(f"  Built {windows.shape} windows in {t3-t2:.1f}s")
        logger.info(f"Total precompute: {t3-t0:.1f}s")

        return cls(features, windows, prices, timestamps, window_offset)

    # ─────────────────────────────────────────────────────────────────
    # Checkpoint replay (fast — inference only)
    # ─────────────────────────────────────────────────────────────────

    def run_checkpoint(
        self,
        checkpoint_path: str,
        *,
        continuous: bool = True,
        initial_capital: float = 250_000.0,
        max_contracts: int = 1,
        veto_rules: list | None = None,
        log_path: str | None = None,
    ) -> dict:
        """
        Replay a checkpoint against precomputed data. Fast — no SCID/feature work.

        Parameters
        ----------
        log_path : str, optional
            If provided, writes a trade log CSV matching the old WyckoffTrader
            format (timestamp, bar_num, bar_close, raw_action, action, quantity,
            position_after, pnl_realized_usd, total_pnl_usd, equity).

        Returns dict with: label, bars, trades, round_trips, vetoed,
        total_pnl_usd, pnl_pts, cumR, win_rate, max_dd_usd, equity, elapsed_s
        """
        from wyckoff_rl.live.trader import VetoFilter

        t0 = time.time()

        # Load actor
        actor = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        actor.eval()

        initial_amount = 1000.0
        vf = VetoFilter(veto_rules) if veto_rules else None

        # State
        position = 0.0
        entry_price = 0.0
        unrealized_pnl = 0.0
        cash = 0.0
        total_pnl = 0.0
        n_trades = 0
        n_vetoed = 0
        equity_curve = []
        trade_pnls = []
        trade_rows = [] if log_path else None

        for wi in range(self.n_windows):
            bar_idx = wi + self.window_offset
            window = self.windows[wi]   # (30, 36)
            price = self.prices[bar_idx]

            # Update unrealized
            if position != 0:
                sign = 1.0 if position > 0 else -1.0
                unrealized_pnl = (price - entry_price) * sign

            # Build state vector
            agent_state = np.array([
                position,
                np.tanh(unrealized_pnl / initial_amount),
                np.tanh(cash / initial_amount),
            ], dtype=np.float32)
            state = np.concatenate([agent_state, window.flatten()])
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                raw_action = actor(state_t).squeeze().item()

            # Map to target position
            if continuous:
                target_pos = raw_action
            else:
                if raw_action > 0.33:
                    target_pos = 1.0
                elif raw_action < -0.33:
                    target_pos = -1.0
                else:
                    target_pos = 0.0

            target_pos = max(-max_contracts, min(max_contracts, target_pos))

            # Veto check
            if vf is not None and position == 0 and target_pos != 0:
                veto, _ = vf.should_veto(window[-1])
                if veto:
                    n_vetoed += 1
                    equity_curve.append(initial_capital + total_pnl)
                    continue

            # Execute
            delta = int(round(target_pos - position))
            if delta != 0:
                pnl = 0.0
                if position != 0:
                    if ((position > 0 and delta < 0) or
                            (position < 0 and delta > 0)):
                        closed = min(abs(position), abs(delta))
                        pnl_pts = (price - entry_price) * (1 if position > 0 else -1)
                        pnl = closed * pnl_pts * 20.0
                        trade_pnls.append(pnl)

                position = position + delta
                if position != 0:
                    entry_price = price
                n_trades += 1
                total_pnl += pnl
                cash += pnl / 20.0

                # Record trade row (matches old WyckoffTrader CSV format)
                if trade_rows is not None:
                    ts = datetime.fromtimestamp(
                        self.timestamps[bar_idx], tz=timezone.utc
                    ).isoformat()
                    action_str = "BUY" if delta > 0 else "SELL"
                    trade_rows.append({
                        'timestamp': ts,
                        'bar_num': bar_idx,
                        'bar_close': price,
                        'raw_action': raw_action,
                        'action': action_str,
                        'quantity': abs(delta),
                        'position_after': position,
                        'pnl_realized_usd': f"{pnl:+.2f}",
                        'total_pnl_usd': f"{total_pnl:+.2f}",
                        'equity': f"{initial_capital + total_pnl:.2f}",
                    })

            equity_curve.append(initial_capital + total_pnl)

        # Compute metrics
        equity_arr = np.array(equity_curve) if equity_curve else np.array([initial_capital])
        peak = np.maximum.accumulate(equity_arr)
        max_dd = float((equity_arr - peak).min())

        winners = sum(1 for p in trade_pnls if p > 0)
        losers = sum(1 for p in trade_pnls if p < 0)
        n_round_trips = winners + losers
        win_rate = winners / max(1, n_round_trips) * 100

        pnl_pts = total_pnl / 20.0
        cum_r = pnl_pts / 1000.0 * 256

        elapsed = time.time() - t0

        # Write trade log CSV if requested
        if trade_rows is not None and trade_rows:
            os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
            pd.DataFrame(trade_rows).to_csv(log_path, index=False)
            logger.info(f"Wrote {len(trade_rows)} trades to {log_path}")

        return {
            'bars': self.n_bars,
            'trades': n_trades,
            'round_trips': n_round_trips,
            'vetoed': n_vetoed,
            'total_pnl_usd': total_pnl,
            'pnl_pts': pnl_pts,
            'cumR': cum_r,
            'win_rate': win_rate,
            'max_dd_usd': max_dd,
            'equity': equity_arr[-1] if len(equity_arr) else initial_capital,
            'elapsed_s': elapsed,
            'log_path': log_path,
        }

    def run_capture(
        self,
        checkpoint_path: str,
        *,
        continuous: bool = True,
        initial_capital: float = 250_000.0,
        max_contracts: int = 1,
        critic_path: str | None = None,
    ) -> dict:
        """
        Replay and capture per-bar windows/actions/trades for analysis.

        Returns capture dict compatible with analyze_trades.py and discover_veto.py.
        """
        t0 = time.time()

        actor = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        actor.eval()

        critic = None
        if critic_path and os.path.exists(critic_path):
            critic = torch.load(critic_path, map_location='cpu', weights_only=False)
            critic.eval()

        initial_amount = 1000.0
        position = 0.0
        entry_price = 0.0
        unrealized_pnl = 0.0
        cash = 0.0
        total_pnl = 0.0
        n_trades = 0

        # Capture buffers
        cap_actions = np.zeros(self.n_windows, dtype=np.float32)
        cap_positions = np.zeros(self.n_windows, dtype=np.float32)
        cap_critic_values = np.zeros(self.n_windows, dtype=np.float32) if critic else None
        trade_events = []

        for wi in range(self.n_windows):
            bar_idx = wi + self.window_offset
            window = self.windows[wi]
            price = self.prices[bar_idx]

            if position != 0:
                sign = 1.0 if position > 0 else -1.0
                unrealized_pnl = (price - entry_price) * sign

            cap_positions[wi] = position

            # Build state
            agent_state = np.array([
                position,
                np.tanh(unrealized_pnl / initial_amount),
                np.tanh(cash / initial_amount),
            ], dtype=np.float32)
            state = np.concatenate([agent_state, window.flatten()])
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                raw_action = actor(state_t).squeeze().item()
            cap_actions[wi] = raw_action

            # Critic value
            if critic is not None:
                with torch.no_grad():
                    v = critic(state_t).squeeze().item()
                cap_critic_values[wi] = v

            # Position target
            if continuous:
                target_pos = raw_action
            else:
                if raw_action > 0.33:
                    target_pos = 1.0
                elif raw_action < -0.33:
                    target_pos = -1.0
                else:
                    target_pos = 0.0

            target_pos = max(-max_contracts, min(max_contracts, target_pos))
            delta = int(round(target_pos - position))

            if delta != 0:
                action_str = "BUY" if delta > 0 else "SELL"
                pnl = 0.0
                if position != 0:
                    if ((position > 0 and delta < 0) or
                            (position < 0 and delta > 0)):
                        closed = min(abs(position), abs(delta))
                        pnl_pts = (price - entry_price) * (1 if position > 0 else -1)
                        pnl = closed * pnl_pts * 20.0

                trade_events.append(
                    (wi, action_str, delta, price, pnl, position))

                position = position + delta
                if position != 0:
                    entry_price = price
                n_trades += 1
                total_pnl += pnl
                cash += pnl / 20.0

        elapsed = time.time() - t0

        # Build capture dict (compatible with analyze_trades + discover_veto)
        capture = {
            'windows': self.windows,                            # (n_windows, 30, 36)
            'actions': cap_actions,                             # (n_windows,)
            'prices': self.prices[self.window_offset:],         # (n_windows,)
            'timestamps': self.timestamps[self.window_offset:], # (n_windows,)
            'positions': cap_positions,                         # (n_windows,)
            'trade_events': trade_events,
            'n_bars': self.n_bars,
            'total_pnl': total_pnl,
            'n_trades': n_trades,
            'has_critic': critic is not None,
        }
        if cap_critic_values is not None:
            capture['critic_values'] = cap_critic_values

        return capture
