"""
Data & Execution Adapters — pluggable backends for the Wyckoff trader.

Data Adapters (tick sources):
  SCIDReplayAdapter  — replay historical ticks from Sierra Chart .scid file
  IBLiveAdapter      — real-time ticks via IB TWS/Gateway
  SCIDTailAdapter    — tail a live .scid file as SC writes new ticks

Order Adapters (execution):
  SimExecutor        — simulated fills (for replay / offline testing)
  IBExecutor         — real orders via IB TWS/Gateway
"""

from __future__ import annotations

import logging
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── SCID constants (from scid_parser.py) ─────────────────────────────────

SCID_HEADER_SIZE = 56
SCID_RECORD_SIZE = 40
SCID_MAGIC = b"SCID"
OLE_TO_UNIX_US = 25569 * 86400 * 1_000_000

RECORD_DTYPE = np.dtype([
    ("datetime",   "<i8"),
    ("open",       "<f4"),
    ("high",       "<f4"),
    ("low",        "<f4"),
    ("close",      "<f4"),
    ("num_trades", "<i4"),
    ("volume",     "<u4"),
    ("bid_volume", "<u4"),
    ("ask_volume", "<u4"),
])


# ═════════════════════════════════════════════════════════════════════════
# Base Classes
# ═════════════════════════════════════════════════════════════════════════

class DataAdapter(ABC):
    """
    Base class for tick data sources.

    Subclasses must implement start() which drives the event loop,
    calling on_tick(price, size, is_uptick, timestamp) for each tick.
    """

    def __init__(self):
        self.on_tick: Optional[Callable] = None  # set by trader

    @abstractmethod
    def start(self):
        """Begin streaming ticks. Blocks until done (replay) or stopped (live)."""

    @abstractmethod
    def stop(self):
        """Signal the adapter to stop."""

    def sleep(self, seconds: float):
        """Process events for a duration. Override for event-loop adapters."""
        time.sleep(seconds)


class OrderAdapter(ABC):
    """
    Base class for order execution backends.

    Provides the same interface regardless of whether orders go to IB,
    SC Sim, or a simulated fill engine.
    """

    @abstractmethod
    def place_order(self, quantity: int, price: float = 0.0) -> bool:
        """
        Submit a market order.

        Parameters
        ----------
        quantity : int
            Positive = buy, negative = sell.
        price : float
            Current market price (used by SimExecutor for fills).

        Returns
        -------
        True if order was accepted/filled.
        """

    @abstractmethod
    def get_position(self) -> float:
        """Get current position (signed contracts)."""

    @abstractmethod
    def get_account_value(self) -> float:
        """Get account equity in USD."""


# ═════════════════════════════════════════════════════════════════════════
# SCID Replay Adapter — replay historical ticks from .scid file
# ═════════════════════════════════════════════════════════════════════════

class SCIDReplayAdapter(DataAdapter):
    """
    Replay ticks from a Sierra Chart .scid file.

    Reads the binary SCID records directly (zero-copy memmap) and fires
    on_tick() for each record. Supports date filtering and speed control.

    Parameters
    ----------
    scid_path : str
        Path to .scid file (e.g., '/opt/SierraChart/Data/NQH26-CME.scid').
    start_date : str, optional
        Start date filter 'YYYY-MM-DD' (default: replay all).
    end_date : str, optional
        End date filter 'YYYY-MM-DD'.
    speed : float
        Replay speed multiplier. 0 = max speed, 1 = real-time, 10 = 10x.
    max_records : int, optional
        Limit number of records replayed.
    """

    def __init__(
        self,
        scid_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        speed: float = 0.0,
        max_records: Optional[int] = None,
    ):
        super().__init__()
        self.scid_path = Path(scid_path)
        self.start_date = start_date
        self.end_date = end_date
        self.speed = speed
        self.max_records = max_records
        self._running = False
        self._records_replayed = 0

        if not self.scid_path.exists():
            raise FileNotFoundError(f"SCID file not found: {self.scid_path}")

    def _read_header(self):
        with open(self.scid_path, "rb") as f:
            header = f.read(SCID_HEADER_SIZE)
        if header[:4] != SCID_MAGIC:
            raise ValueError(f"Invalid SCID magic: {header[:4]!r}")
        header_size = struct.unpack_from("<I", header, 4)[0]
        record_size = struct.unpack_from("<I", header, 8)[0]
        total = (self.scid_path.stat().st_size - header_size) // record_size
        return header_size, record_size, total

    def start(self):
        """Replay all ticks, blocking until complete or stopped."""
        header_size, record_size, total_records = self._read_header()

        if record_size != RECORD_DTYPE.itemsize:
            raise ValueError(f"Unexpected record size {record_size} "
                             f"(expected {RECORD_DTYPE.itemsize})")

        data = np.memmap(
            self.scid_path, dtype=RECORD_DTYPE, mode="r",
            offset=header_size, shape=(total_records,),
        )

        # — Date filtering via OLE timestamps ————————————————————————
        ole_us = data["datetime"]
        unix_us = ole_us.astype(np.int64) - OLE_TO_UNIX_US

        start_idx = 0
        end_idx = total_records
        if self.start_date:
            start_ts = int(datetime.strptime(self.start_date, "%Y-%m-%d")
                          .replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
            start_idx = int(np.searchsorted(unix_us, start_ts))
        if self.end_date:
            end_ts = int((datetime.strptime(self.end_date, "%Y-%m-%d")
                         .replace(tzinfo=timezone.utc) + timedelta(days=1))
                        .timestamp() * 1_000_000)
            end_idx = int(np.searchsorted(unix_us, end_ts))

        if self.max_records:
            end_idx = min(end_idx, start_idx + self.max_records)

        n = end_idx - start_idx
        logger.info(f"[SCIDReplay] Replaying {n:,} records from "
                    f"{self.scid_path.name} "
                    f"(indices {start_idx:,}–{end_idx:,} of {total_records:,})")

        # — Extract arrays ————————————————————————————————————————
        closes = np.array(data["close"][start_idx:end_idx], dtype=np.float64)
        volumes = np.array(data["volume"][start_idx:end_idx], dtype=np.float64)
        ask_vols = np.array(data["ask_volume"][start_idx:end_idx], dtype=np.float64)
        timestamps_us = unix_us[start_idx:end_idx]
        timestamps_s = timestamps_us.astype(np.float64) / 1_000_000

        del data  # release memmap

        # — Replay loop ———————————————————————————————————————————
        self._running = True
        self._records_replayed = 0
        prev_ts = timestamps_s[0] if n > 0 else 0
        last_price = 0.0

        for i in range(n):
            if not self._running:
                break

            price = closes[i]
            vol = volumes[i]
            ask_vol = ask_vols[i]
            ts = timestamps_s[i]

            # Skip zero-volume or zero-price records
            if vol <= 0 or price <= 0:
                continue

            # Uptick heuristic: majority ask volume → uptick
            is_uptick = ask_vol > (vol - ask_vol) if vol > 0 else (price >= last_price)
            last_price = price

            if self.on_tick:
                self.on_tick(price, vol, is_uptick, ts)

            self._records_replayed += 1

            # Speed control
            if self.speed > 0 and i < n - 1:
                dt = timestamps_s[i + 1] - ts
                if dt > 0:
                    time.sleep(dt / self.speed)

            # Progress
            if self._records_replayed % 1_000_000 == 0:
                logger.info(f"[SCIDReplay] {self._records_replayed:,}/{n:,} records "
                            f"({100*self._records_replayed/n:.0f}%)")

        self._running = False
        logger.info(f"[SCIDReplay] Done — {self._records_replayed:,} ticks replayed")

    def stop(self):
        self._running = False

    @property
    def records_replayed(self) -> int:
        return self._records_replayed


# ═════════════════════════════════════════════════════════════════════════
# IB Live Adapter — real-time ticks from IB TWS/Gateway
# ═════════════════════════════════════════════════════════════════════════

class IBLiveAdapter(DataAdapter):
    """
    Stream live ticks from Interactive Brokers.

    Wraps IBConnector for the DataAdapter interface.

    Parameters
    ----------
    host, port, client_id : IB connection params.
    nq_expiry : str
        NQ contract expiry (e.g., '20260618').
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        nq_expiry: str = "",
    ):
        super().__init__()
        from .ib_connector import IBConnector
        self._ib = IBConnector(
            host=host, port=port, client_id=client_id,
            nq_expiry=nq_expiry, on_tick=self._forward_tick,
        )
        self._running = False

    def _forward_tick(self, price, size, is_uptick, timestamp):
        if self.on_tick:
            self.on_tick(price, size, is_uptick, timestamp)

    def start(self):
        """Connect, subscribe, and enter the IB event loop."""
        self._ib.connect()
        self._ib.subscribe_ticks()
        self._running = True
        logger.info("[IBLive] Streaming ticks…")
        while self._running:
            self._ib.sleep(0.1)

    def stop(self):
        self._running = False
        self._ib.disconnect()

    def sleep(self, seconds: float):
        self._ib.sleep(seconds)

    @property
    def ib(self):
        """Access underlying IBConnector (for position sync, account queries)."""
        return self._ib


# ═════════════════════════════════════════════════════════════════════════
# SCID Tail Adapter — tail a live .scid file as SC appends new records
# ═════════════════════════════════════════════════════════════════════════

class SCIDTailAdapter(DataAdapter):
    """
    Tail a .scid file that Sierra Chart is actively writing to.

    Polls for new records appended to the file, emitting ticks as they
    appear. Use when SC handles the data feed and you want Python to
    process the same ticks in near-real-time.

    Parameters
    ----------
    scid_path : str
        Path to the active .scid file.
    poll_interval : float
        Seconds between polls for new data (default: 0.1).
    """

    def __init__(self, scid_path: str, poll_interval: float = 0.1):
        super().__init__()
        self.scid_path = Path(scid_path)
        self.poll_interval = poll_interval
        self._running = False
        self._last_size = 0
        self._last_price = 0.0

        if not self.scid_path.exists():
            raise FileNotFoundError(f"SCID file not found: {self.scid_path}")

    def start(self):
        """Begin tailing the SCID file, blocking until stopped."""
        header_size, record_size, _ = self._read_header()

        if record_size != RECORD_DTYPE.itemsize:
            raise ValueError(f"Unexpected record size: {record_size}")

        # Start from end of current file
        self._last_size = self.scid_path.stat().st_size
        self._running = True
        logger.info(f"[SCIDTail] Tailing {self.scid_path.name} "
                    f"(starting at byte {self._last_size:,})")

        while self._running:
            current_size = self.scid_path.stat().st_size
            new_bytes = current_size - self._last_size

            if new_bytes >= record_size:
                n_new = new_bytes // record_size
                offset = self._last_size
                data = np.memmap(
                    self.scid_path, dtype=RECORD_DTYPE, mode="r",
                    offset=offset, shape=(n_new,),
                )

                for i in range(n_new):
                    rec = data[i]
                    price = float(rec["close"])
                    vol = float(rec["volume"])
                    ask_vol = float(rec["ask_volume"])

                    if vol <= 0 or price <= 0:
                        continue

                    ole_us = int(rec["datetime"])
                    ts = (ole_us - OLE_TO_UNIX_US) / 1_000_000

                    is_uptick = ask_vol > (vol - ask_vol) if vol > 0 else (price >= self._last_price)
                    self._last_price = price

                    if self.on_tick:
                        self.on_tick(price, vol, is_uptick, ts)

                del data
                self._last_size = offset + n_new * record_size

            time.sleep(self.poll_interval)

        logger.info("[SCIDTail] Stopped")

    def stop(self):
        self._running = False

    def _read_header(self):
        with open(self.scid_path, "rb") as f:
            header = f.read(SCID_HEADER_SIZE)
        if header[:4] != SCID_MAGIC:
            raise ValueError(f"Invalid SCID magic: {header[:4]!r}")
        header_size = struct.unpack_from("<I", header, 4)[0]
        record_size = struct.unpack_from("<I", header, 8)[0]
        total = (self.scid_path.stat().st_size - header_size) // record_size
        return header_size, record_size, total


# ═════════════════════════════════════════════════════════════════════════
# Sim Executor — simulated fills for offline replay
# ═════════════════════════════════════════════════════════════════════════

class SimExecutor(OrderAdapter):
    """
    Simulated order execution for replay / offline testing.

    Fills at the given price immediately. Tracks position, PnL, and
    generates a full trade log.

    Parameters
    ----------
    initial_capital : float
        Starting account equity in USD.
    cost_per_side : float
        Commission + slippage per side per contract (USD).
    nq_multiplier : float
        Dollar value per point (NQ = $20/point).
    """

    def __init__(
        self,
        initial_capital: float = 250_000.0,
        cost_per_side: float = 5.0,
        nq_multiplier: float = 20.0,
    ):
        self.initial_capital = initial_capital
        self.nq_multiplier = nq_multiplier
        self.cost_per_side = cost_per_side

        self._position: float = 0.0
        self._entry_price: float = 0.0
        self._equity: float = initial_capital
        self._realized_pnl: float = 0.0
        self._trades: list[dict] = []

    def place_order(self, quantity: int, price: float = 0.0) -> bool:
        if quantity == 0:
            return False

        # Compute realized PnL on position close/reduction
        old_pos = self._position
        new_pos = old_pos + quantity

        # If reducing or flipping position, realize PnL on closed portion
        if old_pos != 0:
            # Contracts being closed
            if (old_pos > 0 and quantity < 0) or (old_pos < 0 and quantity > 0):
                closed = min(abs(old_pos), abs(quantity))
                pnl_pts = (price - self._entry_price) * (1 if old_pos > 0 else -1)
                pnl_usd = closed * pnl_pts * self.nq_multiplier
                cost = closed * self.cost_per_side * 2  # round trip
                net_pnl = pnl_usd - cost
                self._realized_pnl += net_pnl
                self._equity += net_pnl

        self._position = new_pos
        if new_pos != 0:
            self._entry_price = price

        self._trades.append({
            "time": time.time(),
            "action": "BUY" if quantity > 0 else "SELL",
            "quantity": abs(quantity),
            "price": price,
            "position_after": new_pos,
            "equity": self._equity,
            "realized_pnl": self._realized_pnl,
        })

        return True

    def get_position(self) -> float:
        return self._position

    def get_account_value(self) -> float:
        return self._equity

    @property
    def entry_price(self) -> float:
        return self._entry_price

    @property
    def trades(self) -> list[dict]:
        return self._trades

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl


# ═════════════════════════════════════════════════════════════════════════
# IB Executor — real orders via IB TWS/Gateway
# ═════════════════════════════════════════════════════════════════════════

class IBExecutor(OrderAdapter):
    """
    Execute orders through Interactive Brokers.

    Wraps IBConnector for the OrderAdapter interface.

    Parameters
    ----------
    ib_connector : IBConnector
        An already-connected IBConnector instance.
    """

    def __init__(self, ib_connector):
        self._ib = ib_connector

    def place_order(self, quantity: int, price: float = 0.0) -> bool:
        trade = self._ib.place_order(quantity)
        return trade is not None

    def get_position(self) -> float:
        return self._ib.get_position()

    def get_account_value(self) -> float:
        return self._ib.get_account_value()
