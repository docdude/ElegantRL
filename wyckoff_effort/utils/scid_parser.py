"""
Unified Sierra Chart SCID parser — best of deep_m_effort + Wyckoff chatbot.

Features:
  - np.memmap for zero-copy reads with partial-read support (tail-only)
  - 1-second pre-resample + Numba JIT for fast range bar construction
  - Cumulative sums for O(1) volume aggregation per bar
  - UTC → US/Mountain timezone conversion
  - RTH filtering, open=0 fix, padded-record handling
  - Proper logging (with print fallback)

Public API:
  SCIDReader         — class: read .scid files into DataFrames
  SierraChartDataLocator — class: find .scid files on disk
  resample_ticks()   — resample tick data to time-based OHLCV bars
  resample_range_bars() — build range bars from tick data
  load_nq_data()     — convenience: load NQ from SCID directory

SCID binary format:
  Header: 56 bytes (magic "SCID", header_size, record_size, version, etc.)
  Records: 40 bytes each
    DateTime:    int64  (microseconds since OLE epoch 1899-12-30)
    O/H/L/C:    float32 × 4
    NumTrades:   int32
    Volume:      uint32
    BidVolume:   uint32
    AskVolume:   uint32
"""

import os
import struct
import logging
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _log(msg: str):
    """Log via logging if a handler is configured, else print."""
    if logger.handlers or logging.root.handlers:
        logger.info(msg)
    else:
        print(msg)


# =============================================================================
# Constants
# =============================================================================

SCID_HEADER_SIZE = 56
SCID_RECORD_SIZE = 40
SCID_MAGIC = b"SCID"
SCID_RECORD_FORMAT = "<q4fi3I"
SCID_RECORD_STRUCT = struct.Struct(SCID_RECORD_FORMAT)
OLE_EPOCH = datetime(1899, 12, 30)
OLE_EPOCH_NP = np.datetime64("1899-12-30", "us")
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


# =============================================================================
# Header
# =============================================================================

@dataclass
class SCIDHeader:
    file_type: str
    header_size: int
    record_size: int
    version: int
    utc_start_index: int
    total_records: int


# =============================================================================
# SCID Reader  (memmap, partial reads, timezone, RTH)
# =============================================================================

class SCIDReader:
    """
    Read Sierra Chart .scid intraday data files.

    Uses numpy memmap for zero-copy reads — a 1.3 GB file doesn't need
    1.3 GB of RAM.  Supports tail-only reads via ``max_records``.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"SCID file not found: {self.filepath}")

        self.file_size = self.filepath.stat().st_size
        self.file_size_mb = self.file_size / (1024 * 1024)
        self.header = self._read_header()

        _log(f"[SCID] Opened: {self.filepath.name}")
        _log(f"[SCID] Size: {self.file_size_mb:.1f} MB, "
             f"Version: {self.header.version}, "
             f"Records: {self.header.total_records:,}, "
             f"Record size: {self.header.record_size} bytes")

    # -- header ---------------------------------------------------------------

    def _read_header(self) -> SCIDHeader:
        with open(self.filepath, "rb") as f:
            header_bytes = f.read(SCID_HEADER_SIZE)

        if len(header_bytes) < SCID_HEADER_SIZE:
            raise ValueError("File too small for valid SCID header")

        magic = header_bytes[0:4]
        if magic != SCID_MAGIC:
            raise ValueError(f"Invalid SCID magic: {magic!r}")

        header_size = struct.unpack_from("<I", header_bytes, 4)[0]
        record_size = struct.unpack_from("<I", header_bytes, 8)[0]
        version = struct.unpack_from("<H", header_bytes, 12)[0]
        utc_start_index = struct.unpack_from("<I", header_bytes, 16)[0]

        data_bytes = self.file_size - header_size
        total_records = data_bytes // record_size

        return SCIDHeader(
            file_type=magic.decode("ascii"),
            header_size=header_size,
            record_size=record_size,
            version=version,
            utc_start_index=utc_start_index,
            total_records=total_records,
        )

    # -- main read ------------------------------------------------------------

    def read(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_records: Optional[int] = None,
        trading_hours_only: bool = False,
        rth_start: str = "07:30",
        rth_end: str = "14:00",
    ) -> pd.DataFrame:
        """
        Read SCID file into a pandas DataFrame.

        Parameters
        ----------
        start_date, end_date : str, optional
            Filter by date, e.g. ``"2024-12-01"``.
        max_records : int, optional
            Read only the last *N* records (most recent).
        trading_hours_only : bool
            Keep only bars inside ``rth_start``–``rth_end``.
        rth_start, rth_end : str
            RTH window (US/Mountain time), default ``"07:30"``–``"14:00"``.
        """
        t0 = time_module.time()
        h = self.header
        total = h.total_records

        if max_records and max_records < total:
            n_read = max_records
            skip = total - n_read
        else:
            n_read = total
            skip = 0

        byte_offset = h.header_size + skip * h.record_size
        _log(f"[SCID] Reading {n_read:,} records "
             f"(skipping first {skip:,})…")

        # --- bulk read via memmap (or padded fallback) -----------------------
        if h.record_size == RECORD_DTYPE.itemsize:
            data = np.memmap(
                self.filepath, dtype=RECORD_DTYPE, mode="r",
                offset=byte_offset, shape=(n_read,),
            )
        else:
            _log(f"[SCID] Record size {h.record_size} != "
                 f"{RECORD_DTYPE.itemsize}, using padded read…")
            data = self._read_padded(byte_offset, n_read, h.record_size)

        _log(f"[SCID] Converting {n_read:,} records to DataFrame…")

        # --- timestamps: OLE µs → UTC → US/Mountain -------------------------
        ole_us = np.array(data["datetime"], dtype=np.int64)
        unix_us = ole_us - OLE_TO_UNIX_US
        ns = unix_us * 1000
        timestamps = pd.to_datetime(ns, unit="ns", utc=True, errors="coerce")
        timestamps = timestamps.tz_convert("US/Mountain").tz_localize(None)

        # --- OHLC + volume ---------------------------------------------------
        opens = np.array(data["open"], dtype=np.float64)
        highs = np.array(data["high"], dtype=np.float64)
        lows = np.array(data["low"], dtype=np.float64)
        closes = np.array(data["close"], dtype=np.float64)

        # Fix open=0 records (common in tick-level SCID data)
        zero_open = opens == 0.0
        if zero_open.any():
            opens[zero_open] = closes[zero_open]
            _log(f"[SCID] Fixed {zero_open.sum():,} records with open=0")

        df = pd.DataFrame(
            {
                "open":       opens,
                "high":       highs,
                "low":        lows,
                "close":      closes,
                "num_trades": np.array(data["num_trades"], dtype=np.int32),
                "volume":     np.array(data["volume"], dtype=np.float64),
                "bid_volume": np.array(data["bid_volume"], dtype=np.float64),
                "ask_volume": np.array(data["ask_volume"], dtype=np.float64),
            },
            index=timestamps,
        )
        df.index.name = "datetime"

        if isinstance(data, np.memmap):
            del data

        # --- cleanup ---------------------------------------------------------
        initial_len = len(df)
        df = df[df.index.notna() & (df["volume"] > 0)]
        if len(df) < initial_len:
            _log(f"[SCID] Dropped {initial_len - len(df):,} "
                 "empty/invalid records")

        df = df.sort_index()

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        if trading_hours_only:
            df = df.between_time(rth_start, rth_end)
            _log(f"[SCID] RTH filter ({rth_start}–{rth_end}): "
                 f"{len(df):,} records remain")

        df["delta"] = df["ask_volume"] - df["bid_volume"]

        elapsed = time_module.time() - t0
        _log(f"[SCID] Loaded {len(df):,} records in {elapsed:.1f}s")
        if len(df) > 0:
            _log(f"[SCID] Date range: {df.index[0]} → {df.index[-1]}")
            _log(f"[SCID] Price range: {df['low'].min():.2f} – "
                 f"{df['high'].max():.2f}")
        return df

    # -- padded-record fallback -----------------------------------------------

    def _read_padded(self, byte_offset: int, n_records: int,
                     actual_record_size: int) -> np.ndarray:
        our_size = RECORD_DTYPE.itemsize
        result = np.zeros(n_records, dtype=RECORD_DTYPE)
        with open(self.filepath, "rb") as f:
            f.seek(byte_offset)
            chunk = 100_000
            idx = 0
            while idx < n_records:
                batch = min(chunk, n_records - idx)
                raw = f.read(actual_record_size * batch)
                if not raw:
                    break
                n_got = len(raw) // actual_record_size
                for j in range(n_got):
                    start = j * actual_record_size
                    rec = raw[start : start + our_size]
                    if len(rec) == our_size:
                        result[idx] = np.frombuffer(
                            rec, dtype=RECORD_DTYPE
                        )[0]
                    idx += 1
                if idx % 1_000_000 == 0:
                    _log(f"  … {idx:,}/{n_records:,} records read")
        return result

    # -- quick file info ------------------------------------------------------

    def info(self) -> dict:
        """Return first/last record dates and prices without loading all data."""
        h = self.header
        with open(self.filepath, "rb") as f:
            f.seek(h.header_size)
            first_rec = f.read(min(h.record_size, SCID_RECORD_SIZE))
            f.seek(h.header_size + (h.total_records - 1) * h.record_size)
            last_rec = f.read(min(h.record_size, SCID_RECORD_SIZE))
        first = SCID_RECORD_STRUCT.unpack(first_rec)
        last = SCID_RECORD_STRUCT.unpack(last_rec)
        return {
            "file": self.filepath.name,
            "size_mb": self.file_size_mb,
            "total_records": h.total_records,
            "first_date": OLE_EPOCH + timedelta(microseconds=first[0]),
            "last_date": OLE_EPOCH + timedelta(microseconds=last[0]),
            "first_close": first[4],
            "last_close": last[4],
        }


# =============================================================================
# Sierra Chart Data Locator
# =============================================================================

class SierraChartDataLocator:
    """Find .scid files in common Sierra Chart install locations."""

    COMMON_PATHS = [
        "/opt/SierraChart/Data",
        os.path.expanduser("~/SierraChart/Data"),
        r"C:\SierraChart\Data",
        r"C:\SierraChart2\Data",
        r"D:\SierraChart\Data",
        os.path.expanduser(r"~\Documents\SierraChart\Data"),
        r"C:\Program Files\SierraChart\Data",
    ]

    @classmethod
    def find_data_dir(cls) -> Optional[Path]:
        for p in cls.COMMON_PATHS:
            path = Path(p)
            if path.is_dir():
                return path
        for p in [Path("."), Path(".."), Path("./data")]:
            if list(p.glob("*.scid")):
                return p.resolve()
        return None

    @classmethod
    def find_scid_file(cls, symbol: str = "NQZ25-CME") -> Optional[Path]:
        local = Path(f"{symbol}.scid")
        if local.exists():
            return local
        data_dir = cls.find_data_dir()
        if data_dir is None:
            return None
        exact = data_dir / f"{symbol}.scid"
        if exact.exists():
            return exact
        matches = list(data_dir.glob(f"*{symbol}*.scid"))
        if not matches:
            matches = list(data_dir.glob("*NQ*.scid"))
        return matches[0] if matches else None

    @classmethod
    def list_all_scid(cls) -> List[Path]:
        data_dir = cls.find_data_dir()
        if data_dir is None:
            return []
        return sorted(
            data_dir.glob("*.scid"),
            key=lambda f: f.stat().st_size,
            reverse=True,
        )


# =============================================================================
# Resample: ticks → time-based bars
# =============================================================================

def resample_ticks(tick_df: pd.DataFrame, timeframe: str = "1min") -> pd.DataFrame:
    """
    Resample tick-level data into OHLCV bars at the given timeframe,
    preserving Bid/Ask volume, Delta, and NumTrades.

    Parameters
    ----------
    tick_df : DataFrame from ``SCIDReader.read()``
    timeframe : pandas frequency string (``'1min'``, ``'5min'``, etc.)
    """
    if tick_df.empty:
        return tick_df

    resampled = tick_df.resample(timeframe).agg(
        {
            "open":       "first",
            "high":       "max",
            "low":        "min",
            "close":      "last",
            "volume":     "sum",
            "bid_volume": "sum",
            "ask_volume": "sum",
            "delta":      "sum",
            "num_trades": "sum",
        }
    )
    resampled = resampled.dropna(subset=["open"])
    resampled = resampled[resampled["volume"] > 0]
    resampled["cvd"] = resampled["delta"].cumsum()
    return resampled


# =============================================================================
# Resample: ticks → range bars  (1s pre-resample + Numba JIT + cumsum)
# =============================================================================

def resample_range_bars(
    tick_df: pd.DataFrame,
    range_size: float = 10.0,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """
    Build range bars from tick data.

    Deep-M Effort (NQ) uses a 40-range chart: 40 ticks × $0.25 = 10 points.

    Pipeline:
      1. Pre-resample ticks to 1-second bars (34 M → ~500 K rows)
      2. Numba-JIT inner loop finds bar boundaries
      3. Cumulative sums give O(1) volume aggregation per bar

    Parameters
    ----------
    tick_df : DataFrame from ``SCIDReader.read()``
    range_size : bar range in points (default 10.0 for NQ 40-range)
    tick_size : instrument tick size (NQ = 0.25)

    Returns
    -------
    DataFrame with open/high/low/close/volume/bid_volume/ask_volume/
    delta/num_trades/duration_seconds/cvd.
    Index is the timestamp of the last sub-bar in each range bar.
    """
    if tick_df.empty:
        return tick_df

    t0 = time_module.time()

    # -- Step 1: pre-resample to 1-second bars --------------------------------
    _log(f"[RangeBar] Pre-resampling {len(tick_df):,} ticks to 1s bars…")
    sec = tick_df.resample("1s").agg(
        {
            "close": ["first", "max", "min", "last"],
            "volume":     "sum",
            "bid_volume": "sum",
            "ask_volume": "sum",
            "delta":      "sum",
            "num_trades": "sum",
        }
    )
    sec.columns = [
        "open", "high", "low", "close",
        "volume", "bid_volume", "ask_volume", "delta", "num_trades",
    ]
    sec = sec.dropna(subset=["close"])
    sec = sec[sec["volume"] > 0]
    _log(f"[RangeBar] Reduced to {len(sec):,} 1s bars")

    # -- numpy arrays for speed -----------------------------------------------
    high       = sec["high"].values.astype(np.float64)
    low        = sec["low"].values.astype(np.float64)
    close      = sec["close"].values.astype(np.float64)
    total_vol  = sec["volume"].values.astype(np.int64)
    bid_vol    = sec["bid_volume"].values.astype(np.int64)
    ask_vol    = sec["ask_volume"].values.astype(np.int64)
    delta_arr  = sec["delta"].values.astype(np.int64)
    num_trades = sec["num_trades"].values.astype(np.int64)
    timestamps = sec.index.values                         # datetime64[ns]
    n = len(close)

    # -- Step 2: cumulative sums for O(1) aggregation -------------------------
    cum_vol   = _prepend_zero_cumsum(total_vol)
    cum_bid   = _prepend_zero_cumsum(bid_vol)
    cum_ask   = _prepend_zero_cumsum(ask_vol)
    cum_delta = _prepend_zero_cumsum(delta_arr)
    cum_nt    = _prepend_zero_cumsum(num_trades)

    # -- Step 3: find bar boundaries (Numba or Python) ------------------------
    bar_ends, bar_opens, bar_highs, bar_lows, bar_closes = (
        _range_bar_boundaries(high, low, close, n, range_size)
    )

    n_bars = len(bar_ends)
    if n_bars == 0:
        return pd.DataFrame()

    ends = np.array(bar_ends, dtype=np.int64)
    starts = np.empty(n_bars, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1

    # -- Step 4: vectorized aggregation via cumsum diffs ----------------------
    result = pd.DataFrame(
        {
            "open":       np.array(bar_opens),
            "high":       np.array(bar_highs),
            "low":        np.array(bar_lows),
            "close":      np.array(bar_closes),
            "volume":     cum_vol[ends + 1] - cum_vol[starts],
            "bid_volume": cum_bid[ends + 1] - cum_bid[starts],
            "ask_volume": cum_ask[ends + 1] - cum_ask[starts],
            "delta":      cum_delta[ends + 1] - cum_delta[starts],
            "num_trades": cum_nt[ends + 1] - cum_nt[starts],
        },
        index=pd.DatetimeIndex(timestamps[ends], name="datetime"),
    )

    # Duration per bar: time between start and end 1s-bucket
    start_ts = timestamps[starts].astype(np.int64)
    end_ts   = timestamps[ends].astype(np.int64)
    dur_ns   = end_ts - start_ts
    result["duration_seconds"] = np.maximum(dur_ns / 1e9, 0.1)

    result["cvd"] = result["delta"].cumsum()

    elapsed = time_module.time() - t0
    _log(
        f"[RangeBar] Built {n_bars:,} range bars "
        f"({range_size}-pt / {int(range_size / tick_size)}-tick range) "
        f"from {len(tick_df):,} ticks in {elapsed:.1f}s"
    )
    return result


# =============================================================================
# Range bar boundary detection  (Numba JIT with Python fallback)
# =============================================================================

def _range_bar_boundaries_python(high, low, close, n, range_size):
    """Pure-Python fallback for range bar boundary detection."""
    bar_ends = []
    bar_opens = []
    bar_highs = []
    bar_lows = []
    bar_closes = []

    bar_open = close[0]
    bar_high = close[0]
    bar_low = close[0]

    for i in range(n):
        if high[i] > bar_high:
            bar_high = high[i]
        if low[i] < bar_low:
            bar_low = low[i]

        if bar_high - bar_low >= range_size:
            # Direction: which boundary was exceeded?
            if high[i] >= bar_open + range_size:
                bar_close = bar_open + range_size
            elif low[i] <= bar_open - range_size:
                bar_close = bar_open - range_size
            else:
                bar_close = close[i]

            bar_ends.append(i)
            bar_opens.append(bar_open)
            bar_highs.append(bar_high)
            bar_lows.append(bar_low)
            bar_closes.append(bar_close)

            bar_open = bar_close
            bar_high = bar_close
            bar_low = bar_close

    return bar_ends, bar_opens, bar_highs, bar_lows, bar_closes


try:
    from numba import njit

    @njit(cache=True)
    def _range_bar_boundaries_numba(high, low, close, n, range_size):
        """Numba JIT-compiled range bar boundary detection."""
        bar_ends = np.empty(n, dtype=np.int64)
        bar_opens = np.empty(n, dtype=np.float64)
        bar_highs = np.empty(n, dtype=np.float64)
        bar_lows = np.empty(n, dtype=np.float64)
        bar_closes = np.empty(n, dtype=np.float64)
        count = 0

        bar_open = close[0]
        bar_high = close[0]
        bar_low = close[0]

        for i in range(n):
            if high[i] > bar_high:
                bar_high = high[i]
            if low[i] < bar_low:
                bar_low = low[i]

            if bar_high - bar_low >= range_size:
                if high[i] >= bar_open + range_size:
                    bar_close = bar_open + range_size
                elif low[i] <= bar_open - range_size:
                    bar_close = bar_open - range_size
                else:
                    bar_close = close[i]

                bar_ends[count] = i
                bar_opens[count] = bar_open
                bar_highs[count] = bar_high
                bar_lows[count] = bar_low
                bar_closes[count] = bar_close
                count += 1

                bar_open = bar_close
                bar_high = bar_close
                bar_low = bar_close

        return (
            bar_ends[:count], bar_opens[:count], bar_highs[:count],
            bar_lows[:count], bar_closes[:count],
        )

    _range_bar_boundaries = _range_bar_boundaries_numba
    _log("[RangeBar] Using Numba JIT for range bar construction")

except ImportError:
    _range_bar_boundaries = _range_bar_boundaries_python
    _log("[RangeBar] Numba not available — using pure Python")


# =============================================================================
# Convenience loader
# =============================================================================

def load_nq_data(
    scid_dir: str,
    symbol: str = "NQH26.CME",
    timeframe: str = "range",
    range_size: float = 10.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_records: Optional[int] = None,
    trading_hours_only: bool = False,
    rth_start: str = "07:30",
    rth_end: str = "14:00",
) -> pd.DataFrame:
    """
    Load NQ futures data from an SCID directory.

    Parameters
    ----------
    scid_dir : directory containing .scid files
    symbol : Sierra Chart symbol name (filename stem)
    timeframe : ``'range'`` for range bars, or a pandas freq (``'1min'``, …)
    range_size : for range bars — size in points (10.0 = NQ 40-range)
    max_records : read only the last N tick records
    trading_hours_only : filter to RTH window
    """
    filepath = os.path.join(scid_dir, f"{symbol}.scid")

    if not os.path.exists(filepath):
        for f in os.listdir(scid_dir):
            if f.lower().endswith(".scid") and "nq" in f.lower():
                filepath = os.path.join(scid_dir, f)
                _log(f"[SCID] Using: {filepath}")
                break

    reader = SCIDReader(filepath)
    tick_data = reader.read(
        start_date=start_date,
        end_date=end_date,
        max_records=max_records,
        trading_hours_only=trading_hours_only,
        rth_start=rth_start,
        rth_end=rth_end,
    )

    if tick_data.empty:
        return tick_data

    if timeframe == "range":
        return resample_range_bars(tick_data, range_size=range_size, tick_size=0.25)
    return resample_ticks(tick_data, timeframe)


# =============================================================================
# Helpers
# =============================================================================

def _prepend_zero_cumsum(arr: np.ndarray) -> np.ndarray:
    """Return cumulative sum with a leading zero: [0, cumsum(arr)]."""
    out = np.empty(len(arr) + 1, dtype=arr.dtype)
    out[0] = 0
    np.cumsum(arr, out=out[1:])
    return out
