"""
Deep-M Effort NQ — With Native Sierra Chart .scid File Reader (FIXED)
======================================================================
Reads NQZ25-CME.scid (or any SCID file) directly for real market data.
Unified reader interface for both standard and fast (memmap) readers.
"""

import struct
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import time as time_module

warnings.filterwarnings("ignore")


# =============================================================================
# SCID Binary Format Constants
# =============================================================================

SCID_HEADER_SIZE = 56
SCID_RECORD_SIZE = 40
SCID_MAGIC = b"SCID"
SCID_RECORD_FORMAT = "<d4fi3I"
SCID_RECORD_STRUCT = struct.Struct(SCID_RECORD_FORMAT)
OLE_EPOCH = datetime(1899, 12, 30)


@dataclass
class SCIDHeader:
    file_type: str
    header_size: int
    record_size: int
    version: int
    utc_start_index: int
    total_records: int


# =============================================================================
# Unified SCID Reader — Handles files of any size
# =============================================================================

class SCIDReader:
    """
    Unified reader for Sierra Chart .scid intraday data files.
    Automatically uses memory-mapped numpy for large files.
    
    Usage:
        reader = SCIDReader("C:/SierraChart/Data/NQZ25-CME.scid")
        df = reader.read(
            start_date="2024-12-01",
            end_date="2024-12-31",
            trading_hours_only=True,
            max_records=5_000_000,
        )
    """

    # numpy structured dtype matching the 40-byte SCID record layout
    RECORD_DTYPE = np.dtype([
        ("datetime", "<f8"),       # OLE date as float64
        ("open",     "<f4"),
        ("high",     "<f4"),
        ("low",      "<f4"),
        ("close",    "<f4"),
        ("num_trades", "<i4"),
        ("volume",   "<u4"),
        ("bid_volume", "<u4"),
        ("ask_volume", "<u4"),
    ])

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"SCID file not found: {self.filepath}")
        
        self.file_size = self.filepath.stat().st_size
        self.file_size_mb = self.file_size / (1024 * 1024)
        self.header = self._read_header()

        print(f"[SCID] Opened: {self.filepath.name}")
        print(f"[SCID] Size: {self.file_size_mb:.1f} MB, "
              f"Version: {self.header.version}, "
              f"Records: {self.header.total_records:,}, "
              f"Record size: {self.header.record_size} bytes")

    def _read_header(self) -> SCIDHeader:
        with open(self.filepath, "rb") as f:
            header_bytes = f.read(SCID_HEADER_SIZE)

        if len(header_bytes) < SCID_HEADER_SIZE:
            raise ValueError("File too small for valid SCID header")

        magic = header_bytes[0:4]
        if magic != SCID_MAGIC:
            raise ValueError(f"Invalid SCID magic: {magic!r} (expected {SCID_MAGIC!r})")

        header_size = struct.unpack_from("<I", header_bytes, 4)[0]
        record_size = struct.unpack_from("<I", header_bytes, 8)[0]
        version = struct.unpack_from("<H", header_bytes, 12)[0]
        utc_start_index = struct.unpack_from("<I", header_bytes, 16)[0]

        data_bytes = self.filepath.stat().st_size - header_size
        total_records = data_bytes // record_size

        return SCIDHeader(
            file_type=magic.decode("ascii"),
            header_size=header_size,
            record_size=record_size,
            version=version,
            utc_start_index=utc_start_index,
            total_records=total_records,
        )

    def read(self,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             max_records: Optional[int] = None,
             trading_hours_only: bool = False,
             rth_start: str = "09:30",
             rth_end: str = "16:00",
             progress: bool = True) -> pd.DataFrame:
        """
        Read SCID file into pandas DataFrame.

        Parameters
        ----------
        start_date : str, optional
            ISO date string e.g. "2024-12-01". Records before this are dropped.
        end_date : str, optional
            ISO date string. Records after this are dropped.
        max_records : int, optional
            Read at most this many records from file (from the END of file
            to get most recent data). None = read all.
        trading_hours_only : bool
            If True, keep only rows between rth_start and rth_end.
        rth_start : str
            RTH start time, default "09:30".
        rth_end : str
            RTH end time, default "16:00".
        progress : bool
            Print progress during loading.

        Returns
        -------
        pd.DataFrame with DatetimeIndex and columns:
            open, high, low, close, num_trades, volume,
            bid_volume, ask_volume, delta
        """
        t0 = time_module.time()
        header = self.header
        total = header.total_records

        # Determine how many records and from what offset
        if max_records and max_records < total:
            n_read = max_records
            # Read the LAST max_records (most recent data)
            skip_records = total - n_read
        else:
            n_read = total
            skip_records = 0

        byte_offset = header.header_size + skip_records * header.record_size

        if progress:
            print(f"[SCID] Reading {n_read:,} records "
                  f"(skipping first {skip_records:,})...")

        # -----------------------------------------------------------
        # Strategy: use numpy memmap for speed on large files
        # Handle record_size mismatch (newer SCID may have >40 bytes)
        # -----------------------------------------------------------
        if header.record_size == self.RECORD_DTYPE.itemsize:
            # Perfect match — direct memmap (fastest path)
            data = np.memmap(
                self.filepath, dtype=self.RECORD_DTYPE, mode="r",
                offset=byte_offset, shape=(n_read,)
            )
        else:
            # Record size differs — read raw bytes, extract our 40 bytes per record
            if progress:
                print(f"[SCID] Record size {header.record_size} != {self.RECORD_DTYPE.itemsize}, "
                      f"using padded read...")
            data = self._read_padded(byte_offset, n_read, header.record_size)

        # -----------------------------------------------------------
        # Convert to pandas (vectorized, no Python loops)
        # -----------------------------------------------------------
        if progress:
            print(f"[SCID] Converting {n_read:,} records to DataFrame...")

        ole_dates = np.array(data["datetime"], dtype=np.float64)

        # Vectorized OLE → datetime64 conversion
        # OLE date = days since 1899-12-30
        # pandas epoch = 1970-01-01 = OLE day 25569
        OLE_TO_UNIX_DAYS = 25569.0
        unix_days = ole_dates - OLE_TO_UNIX_DAYS
        # Convert to nanoseconds for datetime64[ns]
        ns = (unix_days * 86400 * 1e9).astype("int64")
        timestamps = pd.to_datetime(ns, unit="ns", errors="coerce")

        df = pd.DataFrame({
            "open":        np.array(data["open"], dtype=np.float64),
            "high":        np.array(data["high"], dtype=np.float64),
            "low":         np.array(data["low"], dtype=np.float64),
            "close":       np.array(data["close"], dtype=np.float64),
            "num_trades":  np.array(data["num_trades"], dtype=np.int32),
            "volume":      np.array(data["volume"], dtype=np.float64),
            "bid_volume":  np.array(data["bid_volume"], dtype=np.float64),
            "ask_volume":  np.array(data["ask_volume"], dtype=np.float64),
        }, index=timestamps)
        df.index.name = "datetime"

        # Release memmap
        if isinstance(data, np.memmap):
            del data

        # -----------------------------------------------------------
        # Filter
        # -----------------------------------------------------------
        initial_len = len(df)

        # Drop NaT and zero-volume
        df = df[df.index.notna() & (df["volume"] > 0)]
        if progress and len(df) < initial_len:
            print(f"[SCID] Dropped {initial_len - len(df):,} empty/invalid records")

        df = df.sort_index()

        # Remove duplicate timestamps (keep last)
        n_before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        if progress and len(df) < n_before:
            print(f"[SCID] Removed {n_before - len(df):,} duplicate timestamps")

        # Date range filter
        if start_date:
            ts = pd.Timestamp(start_date)
            df = df[df.index >= ts]
        if end_date:
            ts = pd.Timestamp(end_date)
            df = df[df.index <= ts]

        # Trading hours filter
        if trading_hours_only:
            df = df.between_time(rth_start, rth_end)
            if progress:
                print(f"[SCID] RTH filter ({rth_start}-{rth_end}): "
                      f"{len(df):,} records remain")

        # Compute delta
        df["delta"] = df["ask_volume"] - df["bid_volume"]

        elapsed = time_module.time() - t0
        if progress:
            print(f"[SCID] Loaded {len(df):,} records in {elapsed:.1f}s")
            if len(df) > 0:
                print(f"[SCID] Date range: {df.index[0]} → {df.index[-1]}")
                print(f"[SCID] Price range: {df['low'].min():.2f} — "
                      f"{df['high'].max():.2f}")
                print(f"[SCID] Total volume: {df['volume'].sum():,.0f}")

        return df

    def _read_padded(self, byte_offset: int, n_records: int,
                     actual_record_size: int) -> np.ndarray:
        """Read records when record_size > 40 bytes (newer SCID versions)."""
        our_size = self.RECORD_DTYPE.itemsize  # 40
        result = np.zeros(n_records, dtype=self.RECORD_DTYPE)

        with open(self.filepath, "rb") as f:
            f.seek(byte_offset)
            # Read in large chunks for I/O efficiency
            chunk_records = 100_000
            idx = 0
            while idx < n_records:
                batch = min(chunk_records, n_records - idx)
                raw = f.read(actual_record_size * batch)
                if not raw:
                    break
                n_got = len(raw) // actual_record_size
                for j in range(n_got):
                    start = j * actual_record_size
                    rec = raw[start:start + our_size]
                    if len(rec) == our_size:
                        result[idx] = np.frombuffer(rec, dtype=self.RECORD_DTYPE)[0]
                    idx += 1

                if idx % 1_000_000 == 0:
                    print(f"  ... {idx:,}/{n_records:,} records read")

        return result

    def info(self) -> dict:
        """Quick summary without loading all data."""
        # Read first and last record for date range
        h = self.header
        with open(self.filepath, "rb") as f:
            f.seek(h.header_size)
            first_rec = f.read(min(h.record_size, SCID_RECORD_SIZE))
            f.seek(h.header_size + (h.total_records - 1) * h.record_size)
            last_rec = f.read(min(h.record_size, SCID_RECORD_SIZE))

        first = SCID_RECORD_STRUCT.unpack(first_rec)
        last = SCID_RECORD_STRUCT.unpack(last_rec)

        first_dt = OLE_EPOCH + timedelta(days=first[0])
        last_dt = OLE_EPOCH + timedelta(days=last[0])

        return {
            "file": self.filepath.name,
            "size_mb": self.file_size_mb,
            "total_records": h.total_records,
            "first_date": first_dt,
            "last_date": last_dt,
            "first_close": first[4],
            "last_close": last[4],
        }


# =============================================================================
# Sierra Chart Data Locator
# =============================================================================

class SierraChartDataLocator:
    COMMON_PATHS = [
        r"C:\SierraChart\Data",
        r"C:\SierraChart2\Data",
        r"D:\SierraChart\Data",
        os.path.expanduser(r"~\Documents\SierraChart\Data"),
        os.path.expanduser(r"~\SierraChart\Data"),
        r"C:\Program Files\SierraChart\Data",
        r"C:\Program Files (x86)\SierraChart\Data",
        # Linux / WSL
        os.path.expanduser("~/SierraChart/Data"),
        "/opt/SierraChart/Data",
        "/opt/finrl/data",  # Custom paths
    ]

    @classmethod
    def find_data_dir(cls) -> Optional[Path]:
        for p in cls.COMMON_PATHS:
            path = Path(p)
            if path.is_dir():
                return path
        # Also search current directory and parent
        for p in [Path("."), Path(".."), Path("./data")]:
            if list(p.glob("*.scid")):
                return p.resolve()
        return None

    @classmethod
    def find_scid_file(cls, symbol: str = "NQZ25-CME") -> Optional[Path]:
        # Check current directory first
        local = Path(f"{symbol}.scid")
        if local.exists():
            return local

        data_dir = cls.find_data_dir()
        if data_dir is None:
            # Last resort: search recursively from common roots
            for root in [Path("."), Path("/opt"), Path.home()]:
                if root.exists():
                    matches = list(root.glob(f"**/{symbol}.scid"))
                    if matches:
                        return matches[0]
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
        return sorted(data_dir.glob("*.scid"),
                       key=lambda f: f.stat().st_size, reverse=True)


# =============================================================================
# Configuration & Core Types
# =============================================================================

@dataclass
class DeepMEffortConfig:
    range_size: float = 40.0
    tick_size: float = 0.25
    ema_period: int = 20
    zone_std_mult: float = 1.5
    delta_ema_period: int = 14
    time_ema_period: int = 14
    volume_ma_period: int = 20
    zone_lookback: int = 50
    absorption_threshold: float = 1.5
    vacuum_threshold: float = 0.6
    delta_filter: float = 0.20
    speed_filter: float = 1.1
    abs_continuation: float = 0.6
    vac_continuation: float = 1.3
    min_zone_bars: int = 2
    min_zone_width: float = 10.0
    max_zone_age: int = 200


class ZoneType(Enum):
    ABSORPTION_SUPPORT = "absorption_support"
    ABSORPTION_RESISTANCE = "absorption_resistance"
    VACUUM_BULLISH = "vacuum_bullish"
    VACUUM_BEARISH = "vacuum_bearish"


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
    touches: int = 0


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

    def from_scid_dataframe(self, df: pd.DataFrame,
                            progress_every: int = 500_000) -> List[RangeBar]:
        """Build range bars from SCID DataFrame. Handles millions of rows."""
        bars = []
        if df.empty:
            return bars

        # Work with numpy arrays for speed
        opens_arr = df["open"].values
        highs_arr = df["high"].values
        lows_arr = df["low"].values
        closes_arr = df["close"].values
        vol_arr = df["volume"].values
        ask_arr = df["ask_volume"].values
        bid_arr = df["bid_volume"].values
        trades_arr = df["num_trades"].values
        times = df.index

        n = len(df)
        rp = self.range_points

        current_open = opens_arr[0]
        current_high = current_open
        current_low = current_open
        cum_volume = 0.0
        cum_ask = 0.0
        cum_bid = 0.0
        cum_trades = 0
        start_idx = 0
        bar_idx = 0

        t0 = time_module.time()
        print(f"[RangeBarBuilder] Processing {n:,} records → "
              f"Range-{int(self.config.range_size)} bars "
              f"({rp:.2f} pts per bar)...")

        for i in range(n):
            current_high = max(current_high, highs_arr[i])
            current_low = min(current_low, lows_arr[i])
            cum_volume += vol_arr[i]
            cum_ask += ask_arr[i]
            cum_bid += bid_arr[i]
            cum_trades += trades_arr[i]

            while (current_high - current_low) >= rp:
                price = closes_arr[i]
                if price >= current_open:
                    bar_close = current_low + rp
                    bar_high = bar_close
                    bar_low = current_low
                else:
                    bar_close = current_high - rp
                    bar_low = bar_close
                    bar_high = current_high

                try:
                    duration = max(
                        (times[i] - times[start_idx]).total_seconds(), 0.1)
                except:
                    duration = 1.0

                bars.append(RangeBar(
                    open=round(current_open, 2),
                    high=round(bar_high, 2),
                    low=round(bar_low, 2),
                    close=round(bar_close, 2),
                    volume=max(cum_volume, 1),
                    delta=cum_ask - cum_bid,
                    duration_seconds=duration,
                    tick_count=int(cum_trades),
                    bar_index=bar_idx,
                    timestamp=times[i],
                ))

                bar_idx += 1
                current_open = bar_close
                current_high = max(closes_arr[i], bar_close)
                current_low = min(closes_arr[i], bar_close)
                cum_volume = 0
                cum_ask = 0
                cum_bid = 0
                cum_trades = 0
                start_idx = i

            if progress_every and (i + 1) % progress_every == 0:
                elapsed = time_module.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate
                print(f"  {i+1:>12,}/{n:,} records | "
                      f"{bar_idx:>8,} bars | "
                      f"{rate:,.0f} rec/s | "
                      f"ETA {eta:.0f}s")

        elapsed = time_module.time() - t0
        print(f"[RangeBarBuilder] Done: {len(bars):,} range bars "
              f"from {n:,} records in {elapsed:.1f}s")
        if bars:
            print(f"[RangeBarBuilder] Price range: "
                  f"{min(b.low for b in bars):.2f} — "
                  f"{max(b.high for b in bars):.2f}")
        return bars

    def from_synthetic(self, n_bars: int = 500, seed: int = 42) -> List[RangeBar]:
        rng = np.random.RandomState(seed)
        bars = []
        price = 15000.0
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


# =============================================================================
# EMA & Utilities
# =============================================================================

def ema(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    return result


def rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(data, np.nan, dtype=float)
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1:i + 1])
    return result


def normalize_series(data: np.ndarray) -> np.ndarray:
    mn, mx = np.nanmin(data), np.nanmax(data)
    if mx - mn < 1e-10:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn)


# =============================================================================
# Deep-M Effort Engine
# =============================================================================

class DeepMEffortEngine:
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

        vol_ema = ema(volumes, self.config.ema_period)
        vol_ratio = np.where(vol_ema > 0, volumes / vol_ema, 1.0)

        abs_delta = np.abs(deltas)
        delta_ema_line = ema(abs_delta, self.config.delta_ema_period)
        delta_pct = np.where(volumes > 0, deltas / volumes, 0.0)
        delta_pct_ema = ema(delta_pct, self.config.delta_ema_period)
        delta_divergence = np.where(
            abs_delta > 0, vol_ratio / (np.abs(delta_pct) + 0.01), 0.0)

        speed = np.where(durations > 0, 1.0 / durations, 0.0)
        speed_ema = ema(speed, self.config.time_ema_period)
        speed_ratio = np.where(speed_ema > 0, speed / speed_ema, 1.0)

        effort_index = (
            0.50 * normalize_series(vol_ratio) +
            0.25 * normalize_series(delta_divergence) +
            0.25 * normalize_series(speed_ratio)
        )
        effort_ema_arr = ema(effort_index, self.config.ema_period)
        effort_std = rolling_std(effort_index, self.config.ema_period)
        effort_upper = effort_ema_arr + self.config.zone_std_mult * effort_std
        effort_lower = effort_ema_arr - self.config.zone_std_mult * effort_std

        absorption_score = (vol_ratio *
                            (1.0 - np.abs(delta_pct)) *
                            (1.0 / (speed_ratio + 0.1)))
        absorption_ema_arr = ema(absorption_score, self.config.ema_period)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": volumes, "delta": deltas, "duration": durations,
            "direction": direction,
            "vol_ema": vol_ema, "vol_ratio": vol_ratio,
            "abs_delta": abs_delta, "delta_ema": delta_ema_line,
            "delta_pct": delta_pct, "delta_pct_ema": delta_pct_ema,
            "delta_divergence": delta_divergence,
            "speed": speed, "speed_ema": speed_ema, "speed_ratio": speed_ratio,
            "effort_index": effort_index,
            "effort_ema": effort_ema_arr,
            "effort_upper": effort_upper,
            "effort_lower": effort_lower,
            "absorption_score": absorption_score,
            "absorption_ema": absorption_ema_arr,
        })

        elapsed = time_module.time() - t0
        print(f"[Effort Engine] Metrics computed in {elapsed:.1f}s")

        print(f"[Effort Engine] Detecting zones...")
        self.zones = self._detect_zones(df, bars)
        print(f"[Effort Engine] Found {len(self.zones)} zones")
        return df

    def _detect_zones(self, df, bars):
        zones = []
        n = len(df)
        cfg = self.config
        i = cfg.ema_period

        while i < n:
            row = df.iloc[i]

            if (row["vol_ratio"] >= cfg.absorption_threshold and
                    abs(row["delta_pct"]) < cfg.delta_filter):
                j = i + 1
                while (j < n and
                       df.iloc[j]["vol_ratio"] >= cfg.absorption_threshold * cfg.abs_continuation):
                    j += 1
                cluster = bars[i:j]
                zh = max(b.high for b in cluster) + cfg.tick_size * 4
                zl = min(b.low for b in cluster) - cfg.tick_size * 4
                if (zh - zl) >= cfg.min_zone_width * cfg.tick_size:
                    avg_dir = np.mean(df.iloc[i:j]["direction"])
                    z_type = (ZoneType.ABSORPTION_SUPPORT if avg_dir < 0
                              else ZoneType.ABSORPTION_RESISTANCE)
                    strength = min(1.0,
                                   np.mean(df.iloc[i:j]["vol_ratio"]) /
                                   (cfg.absorption_threshold * 2))
                    zones.append(EffortZone(
                        zone_type=z_type, price_high=zh, price_low=zl,
                        bar_index_start=i, bar_index_end=j - 1,
                        strength=strength,
                        volume_effort=float(np.sum(df.iloc[i:j]["volume"])),
                        delta_effort=float(np.mean(df.iloc[i:j]["delta"])),
                    ))
                    i = j
                    continue

            if (row["vol_ratio"] <= cfg.vacuum_threshold and
                    row["speed_ratio"] > cfg.speed_filter):
                j = i + 1
                while (j < n and
                       df.iloc[j]["vol_ratio"] <= cfg.vacuum_threshold * cfg.vac_continuation):
                    j += 1
                if j - i >= cfg.min_zone_bars:
                    cluster = bars[i:j]
                    zh = max(b.high for b in cluster)
                    zl = min(b.low for b in cluster)
                    avg_dir = np.mean(df.iloc[i:j]["direction"])
                    z_type = (ZoneType.VACUUM_BULLISH if avg_dir > 0
                              else ZoneType.VACUUM_BEARISH)
                    strength = min(1.0,
                                   1.0 - np.mean(df.iloc[i:j]["vol_ratio"]))
                    zones.append(EffortZone(
                        zone_type=z_type, price_high=zh, price_low=zl,
                        bar_index_start=i, bar_index_end=j - 1,
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
    @staticmethod
    def generate(df: pd.DataFrame, zones: List[EffortZone]) -> pd.DataFrame:
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
            if zone.zone_type == ZoneType.ABSORPTION_SUPPORT:
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
                    signals.at[mid, "signal_type"] = "VACUUM_LONG"
                    signals.at[mid, "zone_strength"] = zone.strength
            elif zone.zone_type == ZoneType.VACUUM_BEARISH:
                mid = (start + end) // 2
                if mid < n:
                    signals.at[mid, "signal"] = -1
                    signals.at[mid, "signal_type"] = "VACUUM_SHORT"
                    signals.at[mid, "zone_strength"] = zone.strength
        return signals


# =============================================================================
# Chart
# =============================================================================

class DeepMEffortChart:
    ZONE_COLORS = {
        ZoneType.ABSORPTION_SUPPORT: ("#2196F3", 0.20),
        ZoneType.ABSORPTION_RESISTANCE: ("#F44336", 0.20),
        ZoneType.VACUUM_BULLISH: ("#4CAF50", 0.12),
        ZoneType.VACUUM_BEARISH: ("#FF9800", 0.12),
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
        x = np.arange(len(df))

        fig, axes = plt.subplots(
            4, 1, figsize=figsize, height_ratios=[4, 1.5, 1.2, 1.2],
            facecolor="#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#333355")

        ax_price = axes[0]

        # Draw candles (limit detail for large datasets)
        n_bars = len(df)
        bar_width = 0.7 if n_bars < 500 else 0.9
        for i in range(n_bars):
            o, h, l, c = df.iloc[i][["open", "high", "low", "close"]]
            color = "#26A69A" if c >= o else "#EF5350"
            ax_price.plot([x[i], x[i]], [l, h], color=color, linewidth=0.6)
            body_bottom = min(o, c)
            body_h = max(abs(c - o), self.config.tick_size * 0.5)
            rect = plt.Rectangle((x[i] - bar_width / 2, body_bottom),
                                 bar_width, body_h,
                                 facecolor=color, edgecolor=color, linewidth=0.3)
            ax_price.add_patch(rect)

        # Zones
        for zone in self.zones:
            if zone.bar_index_end < start or zone.bar_index_start > end:
                continue
            color, base_alpha = self.ZONE_COLORS.get(
                zone.zone_type, ("#888", 0.1))
            alpha = base_alpha + zone.strength * 0.15
            xs = max(zone.bar_index_start - start, 0)
            xe = min(end - start, zone.bar_index_end - start + 30)
            rect = plt.Rectangle(
                (xs, zone.price_low), xe - xs,
                zone.price_high - zone.price_low,
                facecolor=color, alpha=alpha,
                edgecolor=color, linewidth=0.5, linestyle="--")
            ax_price.add_patch(rect)

        # Signals
        sigs = self.signals.iloc[start:start + len(df)].reset_index(drop=True)
        tick_offset = self.config.range_points * 0.5
        for i in range(len(sigs)):
            s = sigs.iloc[i]
            if s["signal"] == 1:
                ax_price.scatter(x[i], df.iloc[i]["low"] - tick_offset,
                                 marker="^", color="#00E676", s=60, zorder=5)
            elif s["signal"] == -1:
                ax_price.scatter(x[i], df.iloc[i]["high"] + tick_offset,
                                 marker="v", color="#FF1744", s=60, zorder=5)

        price_ema_line = ema(df["close"].values, self.config.ema_period)
        ax_price.plot(x, price_ema_line, color="#FFD700", linewidth=1.2, alpha=0.8)
        ax_price.set_ylabel("Price", color="white", fontsize=10)

        # Title with date range if available
        title = "Deep-M Effort NQ — Range 40"
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            ts_first = df["timestamp"].dropna().iloc[0]
            ts_last = df["timestamp"].dropna().iloc[-1]
            title += f" | {ts_first} → {ts_last}"
        ax_price.set_title(title, color="white", fontsize=13, fontweight="bold")

        # Volume
        ax_vol = axes[1]
        colors_v = np.where(df["direction"] > 0, "#26A69A", "#EF5350")
        ax_vol.bar(x, df["volume"], color=colors_v, alpha=0.7, width=0.8)
        ax_vol.plot(x, df["vol_ema"], color="#FFD700", linewidth=1.0)
        ax_vol.set_ylabel("Volume", color="white", fontsize=10)

        # Effort
        ax_eff = axes[2]
        ax_eff.fill_between(x, df["effort_upper"], df["effort_lower"],
                            color="#7B1FA2", alpha=0.15)
        ax_eff.plot(x, df["effort_index"], color="#00BCD4", linewidth=1.0)
        ax_eff.plot(x, df["effort_ema"], color="#FFD700", linewidth=1.0)
        ax_eff.set_ylabel("Effort", color="white", fontsize=10)

        # Delta + Absorption
        ax_d = axes[3]
        colors_d = np.where(df["delta"] > 0, "#26A69A", "#EF5350")
        ax_d.bar(x, df["delta"], color=colors_d, alpha=0.6, width=0.8)
        ax_abs = ax_d.twinx()
        ax_abs.plot(x, df["absorption_score"], color="#AB47BC",
                    linewidth=1.0, alpha=0.8)
        ax_abs.plot(x, df["absorption_ema"], color="#FFD700",
                    linewidth=0.8, alpha=0.7)
        ax_d.set_ylabel("Delta", color="white", fontsize=10)
        ax_abs.set_ylabel("Absorption", color="#AB47BC", fontsize=10)
        ax_abs.tick_params(colors="#AB47BC", labelsize=8)

        plt.tight_layout()
        outfile = "deep_m_effort_nq_real.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.show()
        print(f"[Chart saved: {outfile}]")


# =============================================================================
# Performance
# =============================================================================

def performance_summary(df, signals, config):
    trades = []
    position = 0
    entry_price = entry_bar = 0
    for i in range(len(signals)):
        sig = signals.iloc[i]["signal"]
        if sig != 0 and position == 0:
            position = int(sig)
            entry_price = df.iloc[i]["close"]
            entry_bar = i
        elif position != 0 and i - entry_bar >= 10:
            exit_price = df.iloc[i]["close"]
            pnl = (exit_price - entry_price) * position
            trades.append({
                "entry_bar": entry_bar, "exit_bar": i,
                "direction": "LONG" if position > 0 else "SHORT",
                "entry": entry_price, "exit": exit_price,
                "pnl_points": round(pnl, 2),
                "pnl_ticks": round(pnl / config.tick_size, 1),
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
    print(f"  Total PnL:      {tdf['pnl_points'].sum():.2f} pts")
    print(f"  Avg PnL/Trade:  {tdf['pnl_points'].mean():.2f} pts")
    gl = abs(tdf[tdf["pnl_points"] < 0]["pnl_points"].sum())
    pf = wins["pnl_points"].sum() / gl if gl > 0 else float("inf")
    print(f"  Profit Factor:  {pf:.2f}")
    print(f"{'='*60}")
    print(tdf.groupby("signal_type")["pnl_points"].agg(
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
    builder = RangeBarBuilder(config)

    # -----------------------------------------------------------------
    # SET YOUR SCID FILE PATH HERE (or let auto-detect find it)
    # -----------------------------------------------------------------
    SCID_PATH = None  # e.g. r"C:\SierraChart\Data\NQZ25-CME.scid"

    # Auto-detect
    if SCID_PATH is None:
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

        # Quick file info before full load
        try:
            info = reader.info()
            print(f"  First record: {info['first_date']}")
            print(f"  Last record:  {info['last_date']}")
            print(f"  First close:  {info['first_close']:.2f}")
            print(f"  Last close:   {info['last_close']:.2f}")
            print()
        except Exception as e:
            print(f"  (Could not read file info: {e})")

        # For a 1.3GB file, you may want to limit records
        # or filter by date to keep processing time reasonable.
        # Set max_records=None to load everything.
        scid_df = reader.read(
            start_date=None,         # e.g. "2025-01-01"
            end_date=None,
            max_records=5_000_000,   # Last 5M records (~recent days)
            trading_hours_only=False,
        )

        if scid_df.empty:
            print("[WARNING] No data. Falling back to synthetic.")
            use_real = False
        else:
            bars = builder.from_scid_dataframe(scid_df)
            if not bars:
                print("[WARNING] No range bars built. Check data/range size.")
                use_real = False
    
    if not use_real:
        print("\n[Using synthetic data]")
        avail = SierraChartDataLocator.list_all_scid()
        if avail:
            print("Available .scid files:")
            for f in avail[:10]:
                print(f"  {f.name:40s} ({f.stat().st_size/(1024**2):.1f} MB)")
        bars = builder.from_synthetic(n_bars=500, seed=42)

    print(f"\nTotal range bars: {len(bars):,}")

    engine = DeepMEffortEngine(config)
    df = engine.compute(bars)
    zones = engine.zones

    print(f"\nDetected {len(zones)} effort zones:")
    for z in zones[:20]:  # Show first 20
        ts = ""
        if z.bar_index_start < len(bars) and bars[z.bar_index_start].timestamp:
            ts = f" @ {bars[z.bar_index_start].timestamp}"
        print(f"  {z.zone_type.value:30s} "
              f"[{z.price_low:.2f}—{z.price_high:.2f}] "
              f"str={z.strength:.2f}{ts}")
    if len(zones) > 20:
        print(f"  ... and {len(zones) - 20} more zones")

    signals = SignalGenerator.generate(df, zones)
    n_long = (signals["signal"] == 1).sum()
    n_short = (signals["signal"] == -1).sum()
    print(f"\nSignals: {n_long} long, {n_short} short")

    trades_df = performance_summary(df, signals, config)

    # Chart last N bars
    n_chart = min(len(df), 300)
    chart = DeepMEffortChart(df, zones, signals, config)
    chart.plot(start=max(0, len(df) - n_chart), end=len(df))


if __name__ == "__main__":
    main()
