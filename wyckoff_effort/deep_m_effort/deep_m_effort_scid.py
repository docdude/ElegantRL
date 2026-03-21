"""
Deep-M Effort NQ — With Native Sierra Chart .scid File Reader
===============================================================
Reads NQZ25-CME.scid (or any SCID file) directly for real market data.
"""

import struct
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# SCID File Reader — Sierra Chart Intraday Data Format
# =============================================================================

"""
Sierra Chart .scid Binary Format
=================================
Header: 56 bytes
  Bytes  0- 3: FileTypeUniqueID  (4 chars, "SCID")
  Bytes  4- 7: HeaderSize        (uint32, typically 56)
  Bytes  8-11: RecordSize        (uint32, typically 40)
  Bytes 12-13: Version           (uint16)
  Bytes 14-15: Unused
  Bytes 16-19: UTCStartIndex     (uint32)
  Bytes 20-55: Reserved          (36 bytes)

Each Intraday Record: 40 bytes
  Bytes  0- 7: SCDateTime        (double, OLE Automation Date)
  Bytes  8-11: Open              (float32)
  Bytes 12-15: High              (float32)
  Bytes 16-19: Low               (float32)
  Bytes 20-23: Close             (float32)
  Bytes 24-27: NumTrades         (int32)
  Bytes 28-31: TotalVolume       (uint32)
  Bytes 32-35: BidVolume         (uint32)
  Bytes 36-39: AskVolume         (uint32)

OLE Date: days since 1899-12-30 as a floating-point double.
"""

SCID_HEADER_SIZE = 56
SCID_RECORD_SIZE = 40
SCID_MAGIC = b"SCID"

# struct format: double + 4 floats + 1 int32 + 3 uint32
SCID_RECORD_FORMAT = "<d4fi3I"
SCID_RECORD_STRUCT = struct.Struct(SCID_RECORD_FORMAT)

# OLE epoch
OLE_EPOCH = datetime(1899, 12, 30)


@dataclass
class SCIDHeader:
    file_type: str
    header_size: int
    record_size: int
    version: int
    utc_start_index: int
    total_records: int


class SCIDReader:
    """
    Native reader for Sierra Chart .scid intraday data files.
    
    Usage:
        reader = SCIDReader("path/to/NQZ25-CME.scid")
        df = reader.read()
        # or with date filtering:
        df = reader.read(start_date="2024-12-01", end_date="2024-12-15")
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"SCID file not found: {self.filepath}")
        if not self.filepath.suffix.lower() == ".scid":
            raise ValueError(f"Expected .scid file, got: {self.filepath.suffix}")
        
        self.header = self._read_header()
        print(f"[SCID] Opened: {self.filepath.name}")
        print(f"[SCID] Version: {self.header.version}, "
              f"Records: {self.header.total_records:,}, "
              f"Record size: {self.header.record_size} bytes")

    def _read_header(self) -> SCIDHeader:
        """Parse the 56-byte SCID file header."""
        file_size = self.filepath.stat().st_size
        
        with open(self.filepath, "rb") as f:
            header_bytes = f.read(SCID_HEADER_SIZE)
        
        if len(header_bytes) < SCID_HEADER_SIZE:
            raise ValueError("File too small to contain valid SCID header")
        
        magic = header_bytes[0:4]
        if magic != SCID_MAGIC:
            raise ValueError(f"Invalid SCID magic bytes: {magic} (expected {SCID_MAGIC})")
        
        header_size = struct.unpack_from("<I", header_bytes, 4)[0]
        record_size = struct.unpack_from("<I", header_bytes, 8)[0]
        version = struct.unpack_from("<H", header_bytes, 12)[0]
        utc_start_index = struct.unpack_from("<I", header_bytes, 16)[0]
        
        data_size = file_size - header_size
        total_records = data_size // record_size
        
        return SCIDHeader(
            file_type=magic.decode("ascii"),
            header_size=header_size,
            record_size=record_size,
            version=version,
            utc_start_index=utc_start_index,
            total_records=total_records,
        )

    @staticmethod
    def _ole_to_datetime(ole_date: float) -> pd.Timestamp:
        """Convert OLE Automation Date (double) to pandas Timestamp."""
        if ole_date == 0 or np.isnan(ole_date):
            return pd.NaT
        return pd.Timestamp(OLE_EPOCH + timedelta(days=ole_date))

    def read(self, start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             max_records: Optional[int] = None,
             trading_hours_only: bool = False) -> pd.DataFrame:
        """
        Read SCID file into a pandas DataFrame.
        
        Parameters:
            start_date: Filter start (e.g. "2024-12-01")
            end_date:   Filter end (e.g. "2024-12-15")  
            max_records: Limit number of records read (None = all)
            trading_hours_only: If True, filter to RTH 9:30-16:00 ET
            
        Returns:
            DataFrame with columns: datetime, open, high, low, close,
                                    num_trades, volume, bid_volume, ask_volume
        """
        header = self.header
        n_records = header.total_records
        if max_records:
            n_records = min(n_records, max_records)

        # Pre-allocate arrays for speed
        datetimes = np.empty(n_records, dtype="datetime64[ns]")
        opens = np.empty(n_records, dtype=np.float32)
        highs = np.empty(n_records, dtype=np.float32)
        lows = np.empty(n_records, dtype=np.float32)
        closes = np.empty(n_records, dtype=np.float32)
        num_trades = np.empty(n_records, dtype=np.int32)
        volumes = np.empty(n_records, dtype=np.uint32)
        bid_volumes = np.empty(n_records, dtype=np.uint32)
        ask_volumes = np.empty(n_records, dtype=np.uint32)

        actual_count = 0

        with open(self.filepath, "rb") as f:
            f.seek(header.header_size)
            
            # Read in chunks for performance
            chunk_size = 10000
            records_read = 0
            
            while records_read < n_records:
                batch = min(chunk_size, n_records - records_read)
                raw = f.read(header.record_size * batch)
                
                if not raw:
                    break
                
                n_in_chunk = len(raw) // header.record_size
                
                for j in range(n_in_chunk):
                    offset = j * header.record_size
                    
                    # Handle potential record size mismatch
                    # (newer SCID versions may have larger records)
                    rec_bytes = raw[offset:offset + min(header.record_size, SCID_RECORD_SIZE)]
                    if len(rec_bytes) < SCID_RECORD_SIZE:
                        continue
                    
                    (ole_dt, o, h, l, c, nt, tv, bv, av) = \
                        SCID_RECORD_STRUCT.unpack(rec_bytes)
                    
                    if ole_dt == 0 or tv == 0:
                        continue  # Skip empty/zero-volume records
                    
                    ts = OLE_EPOCH + timedelta(days=ole_dt)
                    
                    datetimes[actual_count] = np.datetime64(ts)
                    opens[actual_count] = o
                    highs[actual_count] = h
                    lows[actual_count] = l
                    closes[actual_count] = c
                    num_trades[actual_count] = nt
                    volumes[actual_count] = tv
                    bid_volumes[actual_count] = bv
                    ask_volumes[actual_count] = av
                    actual_count += 1
                
                records_read += n_in_chunk

        # Trim to actual count
        df = pd.DataFrame({
            "datetime": pd.to_datetime(datetimes[:actual_count]),
            "open": opens[:actual_count].astype(float),
            "high": highs[:actual_count].astype(float),
            "low": lows[:actual_count].astype(float),
            "close": closes[:actual_count].astype(float),
            "num_trades": num_trades[:actual_count],
            "volume": volumes[:actual_count].astype(float),
            "bid_volume": bid_volumes[:actual_count].astype(float),
            "ask_volume": ask_volumes[:actual_count].astype(float),
        })

        df = df.set_index("datetime").sort_index()
        
        # Remove duplicate indices
        df = df[~df.index.duplicated(keep="last")]
        
        # Date filtering
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        # RTH filter (NQ regular trading hours: 9:30 AM - 4:00 PM Eastern)
        if trading_hours_only:
            df = df.between_time("09:30", "16:00")
        
        # Compute delta column for convenience
        df["delta"] = df["ask_volume"] - df["bid_volume"]
        
        print(f"[SCID] Loaded {len(df):,} records")
        if len(df) > 0:
            print(f"[SCID] Date range: {df.index[0]} → {df.index[-1]}")
            print(f"[SCID] Price range: {df['low'].min():.2f} — {df['high'].max():.2f}")
            print(f"[SCID] Total volume: {df['volume'].sum():,.0f}")
        
        return df


class SCIDFastReader:
    """
    Ultra-fast SCID reader using numpy structured arrays and memory mapping.
    Use this for very large files (>1GB).
    """

    DTYPE = np.dtype([
        ("datetime", "<f8"),      # OLE date as double
        ("open", "<f4"),
        ("high", "<f4"),
        ("low", "<f4"),
        ("close", "<f4"),
        ("num_trades", "<i4"),
        ("volume", "<u4"),
        ("bid_volume", "<u4"),
        ("ask_volume", "<u4"),
    ])

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"SCID file not found: {filepath}")

    def read(self, start_date: Optional[str] = None,
             end_date: Optional[str] = None,
             max_records: Optional[int] = None,
             trading_hours_only: bool = False) -> pd.DataFrame:
        """Memory-mapped read — extremely fast for large files."""
        
        # Read header
        with open(self.filepath, "rb") as f:
            magic = f.read(4)
            if magic != SCID_MAGIC:
                raise ValueError(f"Invalid SCID file: {magic}")
            header_size = struct.unpack("<I", f.read(4))[0]
            record_size = struct.unpack("<I", f.read(4))[0]

        # Memory-map the data portion
        file_size = self.filepath.stat().st_size
        n_records = (file_size - header_size) // record_size

        # If record size matches our dtype, use direct memory mapping
        if record_size == self.DTYPE.itemsize:
            data = np.memmap(
                self.filepath, dtype=self.DTYPE, mode="r",
                offset=header_size, shape=(n_records,)
            )
        else:
            # Record size mismatch — read manually with padding
            raw = np.memmap(self.filepath, dtype=np.uint8, mode="r",
                            offset=header_size)
            data = np.zeros(n_records, dtype=self.DTYPE)
            for i in range(n_records):
                start = i * record_size
                chunk = raw[start:start + self.DTYPE.itemsize]
                if len(chunk) == self.DTYPE.itemsize:
                    data[i] = np.frombuffer(chunk, dtype=self.DTYPE)[0]

        # Convert OLE dates to pandas timestamps
        ole_dates = data["datetime"]
        # Vectorized conversion: OLE epoch + days
        timestamps = pd.to_datetime(
            ole_dates, unit="D", origin=pd.Timestamp("1899-12-30"),
            errors="coerce"
        )

        df = pd.DataFrame({
            "open": data["open"].astype(float),
            "high": data["high"].astype(float),
            "low": data["low"].astype(float),
            "close": data["close"].astype(float),
            "num_trades": data["num_trades"],
            "volume": data["volume"].astype(float),
            "bid_volume": data["bid_volume"].astype(float),
            "ask_volume": data["ask_volume"].astype(float),
        }, index=timestamps)

        df.index.name = "datetime"
        
        # Filter zero volume and NaT
        df = df[df.index.notna() & (df["volume"] > 0)]
        df = df.sort_index()
        
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        df["delta"] = df["ask_volume"] - df["bid_volume"]

        # RTH filter (NQ regular trading hours: 9:30 AM - 4:00 PM Eastern)
        if trading_hours_only:
            df = df.between_time("09:30", "16:00")

        print(f"[SCID-Fast] Loaded {len(df):,} records from {self.filepath.name}")
        return df


# =============================================================================
# SCID Data Locator — Find Sierra Chart Data Directory
# =============================================================================

class SierraChartDataLocator:
    """Automatically locate Sierra Chart data files on the system."""

    COMMON_PATHS = [
        # Windows default installs
        r"C:\SierraChart\Data",
        r"C:\SierraChart2\Data",
        r"D:\SierraChart\Data",
        # User-specific
        os.path.expanduser(r"~\Documents\SierraChart\Data"),
        os.path.expanduser(r"~\SierraChart\Data"),
        # Program Files
        r"C:\Program Files\SierraChart\Data",
        r"C:\Program Files (x86)\SierraChart\Data",
    ]

    @classmethod
    def find_data_dir(cls) -> Optional[Path]:
        for p in cls.COMMON_PATHS:
            path = Path(p)
            if path.is_dir():
                print(f"[Locator] Found Sierra Chart data directory: {path}")
                return path
        return None

    @classmethod
    def find_scid_file(cls, symbol: str = "NQZ25-CME") -> Optional[Path]:
        """Search for a specific .scid file."""
        data_dir = cls.find_data_dir()
        if data_dir is None:
            print("[Locator] Sierra Chart data directory not found. "
                  "Please specify the path manually.")
            return None
        
        # Try exact match first
        exact = data_dir / f"{symbol}.scid"
        if exact.exists():
            return exact
        
        # Fuzzy match
        matches = list(data_dir.glob(f"*{symbol}*.scid"))
        if not matches:
            matches = list(data_dir.glob("*NQ*.scid"))
        
        if matches:
            print(f"[Locator] Found {len(matches)} matching files:")
            for m in matches[:10]:
                size_mb = m.stat().st_size / (1024 * 1024)
                print(f"  {m.name} ({size_mb:.1f} MB)")
            return matches[0]
        
        print(f"[Locator] No .scid files found matching '{symbol}'")
        return None

    @classmethod
    def list_all_scid(cls) -> List[Path]:
        data_dir = cls.find_data_dir()
        if data_dir is None:
            return []
        files = sorted(data_dir.glob("*.scid"), key=lambda f: f.stat().st_size, reverse=True)
        return files


# =============================================================================
# Configuration & Core Types (same as before, included for completeness)
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
# Range Bar Builder — From SCID Data
# =============================================================================

class RangeBarBuilder:
    """Builds range bars from SCID tick/intraday data."""

    def __init__(self, config: DeepMEffortConfig):
        self.config = config
        self.range_points = config.range_size * config.tick_size

    def from_scid_dataframe(self, df: pd.DataFrame) -> List[RangeBar]:
        """
        Build range bars from a DataFrame loaded via SCIDReader.
        The SCID data contains 1-tick or 1-second resolution records
        which we aggregate into range bars.
        
        This handles the real Sierra Chart data format where each row
        may represent a single trade or a small time bucket.
        """
        bars = []
        if df.empty:
            return bars

        current_open = float(df.iloc[0]["open"])
        current_high = current_open
        current_low = current_open
        cum_volume = 0.0
        cum_ask_vol = 0.0
        cum_bid_vol = 0.0
        cum_trades = 0
        start_time = df.index[0]
        bar_idx = 0

        opens_arr = df["open"].values.astype(float)
        highs_arr = df["high"].values.astype(float)
        lows_arr = df["low"].values.astype(float)
        closes_arr = df["close"].values.astype(float)
        vol_arr = df["volume"].values.astype(float)
        ask_arr = df["ask_volume"].values.astype(float)
        bid_arr = df["bid_volume"].values.astype(float)
        trades_arr = df["num_trades"].values
        times = df.index

        n = len(df)
        print(f"[RangeBarBuilder] Processing {n:,} intraday records into "
              f"Range-{int(self.config.range_size)} bars...")

        for i in range(n):
            row_high = highs_arr[i]
            row_low = lows_arr[i]
            row_close = closes_arr[i]
            row_vol = vol_arr[i]
            row_ask = ask_arr[i]
            row_bid = bid_arr[i]
            row_trades = trades_arr[i]

            current_high = max(current_high, row_high)
            current_low = min(current_low, row_low)
            cum_volume += row_vol
            cum_ask_vol += row_ask
            cum_bid_vol += row_bid
            cum_trades += row_trades

            # Check if range bar is complete
            while (current_high - current_low) >= self.range_points:
                if row_close >= current_open:
                    # Bullish bar: close at top of range
                    bar_close = current_low + self.range_points
                    bar_high = bar_close
                    bar_low = current_low
                else:
                    # Bearish bar: close at bottom of range
                    bar_close = current_high - self.range_points
                    bar_low = bar_close
                    bar_high = current_high

                end_time = times[i]
                duration = 0.0
                if hasattr(start_time, 'timestamp') and hasattr(end_time, 'timestamp'):
                    try:
                        duration = (end_time - start_time).total_seconds()
                    except:
                        duration = 1.0
                duration = max(duration, 0.1)

                delta = cum_ask_vol - cum_bid_vol

                bars.append(RangeBar(
                    open=round(current_open, 2),
                    high=round(bar_high, 2),
                    low=round(bar_low, 2),
                    close=round(bar_close, 2),
                    volume=max(cum_volume, 1),
                    delta=delta,
                    duration_seconds=duration,
                    tick_count=int(cum_trades),
                    bar_index=bar_idx,
                    timestamp=end_time,
                ))

                bar_idx += 1
                current_open = bar_close
                current_high = max(row_close, bar_close)
                current_low = min(row_close, bar_close)
                cum_volume = 0
                cum_ask_vol = 0
                cum_bid_vol = 0
                cum_trades = 0
                start_time = end_time

            if (i + 1) % 500000 == 0:
                print(f"  ... processed {i+1:,}/{n:,} records → "
                      f"{bar_idx:,} range bars so far")

        print(f"[RangeBarBuilder] Built {len(bars):,} range bars from "
              f"{n:,} intraday records")
        return bars

    def from_synthetic(self, n_bars: int = 500, seed: int = 42) -> List[RangeBar]:
        """Generate synthetic range bars for testing."""
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
# EMA & Statistical Utilities
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
        effort_ema = ema(effort_index, self.config.ema_period)
        effort_std = rolling_std(effort_index, self.config.ema_period)
        effort_upper = effort_ema + self.config.zone_std_mult * effort_std
        effort_lower = effort_ema - self.config.zone_std_mult * effort_std

        absorption_score = vol_ratio * (1.0 - np.abs(delta_pct)) * (1.0 / (speed_ratio + 0.1))
        absorption_ema = ema(absorption_score, self.config.ema_period)

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
            "effort_ema": effort_ema, "effort_upper": effort_upper,
            "effort_lower": effort_lower,
            "absorption_score": absorption_score,
            "absorption_ema": absorption_ema,
        })

        self.zones = self._detect_zones(df, bars)
        return df

    def _detect_zones(self, df, bars):
        zones = []
        n = len(df)
        cfg = self.config
        i = cfg.ema_period

        while i < n:
            row = df.iloc[i]

            if row["vol_ratio"] >= cfg.absorption_threshold and abs(row["delta_pct"]) < cfg.delta_filter:
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
                    zones.append(EffortZone(
                        zone_type=z_type, price_high=zone_high, price_low=zone_low,
                        bar_index_start=i, bar_index_end=j - 1, strength=strength,
                        volume_effort=float(np.sum(df.iloc[i:j]["volume"])),
                        delta_effort=float(np.mean(df.iloc[i:j]["delta"])),
                    ))
                    i = j
                    continue

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
                        zone_type=z_type, price_high=zone_high, price_low=zone_low,
                        bar_index_start=i, bar_index_end=j - 1, strength=strength,
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
# Visualization
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

        fig, axes = plt.subplots(4, 1, figsize=figsize,
                                 height_ratios=[4, 1.5, 1.2, 1.2],
                                 facecolor="#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333355")

        ax_price = axes[0]
        for i in range(len(df)):
            o, h, l, c = df.iloc[i][["open", "high", "low", "close"]]
            color = "#26A69A" if c >= o else "#EF5350"
            ax_price.plot([x[i], x[i]], [l, h], color=color, linewidth=0.8)
            body_bottom = min(o, c)
            body_height = max(abs(c - o), self.config.tick_size * 0.5)
            rect = plt.Rectangle((x[i] - 0.35, body_bottom), 0.7,
                                 body_height, facecolor=color, edgecolor=color)
            ax_price.add_patch(rect)

        for zone in self.zones:
            if zone.bar_index_end < start or zone.bar_index_start > end:
                continue
            color, base_alpha = self.ZONE_COLORS.get(zone.zone_type, ("#888", 0.1))
            alpha = base_alpha + zone.strength * 0.15
            x_start = max(zone.bar_index_start - start, 0)
            x_end = min(end - start, zone.bar_index_end - start + 30)
            rect = plt.Rectangle((x_start, zone.price_low), x_end - x_start,
                                 zone.price_high - zone.price_low,
                                 facecolor=color, alpha=alpha,
                                 edgecolor=color, linewidth=0.5, linestyle="--")
            ax_price.add_patch(rect)

        sigs = self.signals.iloc[start:start + len(df)].reset_index(drop=True)
        for i in range(len(sigs)):
            sig = sigs.iloc[i]
            if sig["signal"] == 1:
                ax_price.scatter(x[i], df.iloc[i]["low"] - 3, marker="^",
                                 color="#00E676", s=80, zorder=5)
            elif sig["signal"] == -1:
                ax_price.scatter(x[i], df.iloc[i]["high"] + 3, marker="v",
                                 color="#FF1744", s=80, zorder=5)

        price_ema_line = ema(df["close"].values, self.config.ema_period)
        ax_price.plot(x, price_ema_line, color="#FFD700", linewidth=1.2, alpha=0.8)
        ax_price.set_ylabel("Price", color="white", fontsize=10)
        ax_price.set_title("Deep-M Effort NQ — Real SCID Data | Range 40",
                           color="white", fontsize=14, fontweight="bold")

        ax_vol = axes[1]
        colors_vol = np.where(df["direction"] > 0, "#26A69A", "#EF5350")
        ax_vol.bar(x, df["volume"], color=colors_vol, alpha=0.7, width=0.8)
        ax_vol.plot(x, df["vol_ema"], color="#FFD700", linewidth=1.0)
        ax_vol.set_ylabel("Volume", color="white", fontsize=10)

        ax_effort = axes[2]
        ax_effort.fill_between(x, df["effort_upper"], df["effort_lower"],
                               color="#7B1FA2", alpha=0.15)
        ax_effort.plot(x, df["effort_index"], color="#00BCD4", linewidth=1.0)
        ax_effort.plot(x, df["effort_ema"], color="#FFD700", linewidth=1.0)
        ax_effort.set_ylabel("Effort", color="white", fontsize=10)

        ax_delta = axes[3]
        colors_d = np.where(df["delta"] > 0, "#26A69A", "#EF5350")
        ax_delta.bar(x, df["delta"], color=colors_d, alpha=0.6, width=0.8)
        ax_abs = ax_delta.twinx()
        ax_abs.plot(x, df["absorption_score"], color="#AB47BC", linewidth=1.0, alpha=0.8)
        ax_abs.plot(x, df["absorption_ema"], color="#FFD700", linewidth=0.8, alpha=0.7)
        ax_delta.set_ylabel("Delta", color="white", fontsize=10)
        ax_abs.set_ylabel("Absorption", color="#AB47BC", fontsize=10)
        ax_abs.tick_params(colors="#AB47BC", labelsize=8)

        plt.tight_layout()
        plt.savefig("deep_m_effort_nq_real.png", dpi=150, bbox_inches="tight",
                     facecolor="#1a1a2e")
        plt.show()
        print("[Chart saved: deep_m_effort_nq_real.png]")


# =============================================================================
# Performance Summary
# =============================================================================

def performance_summary(df, signals, config):
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
                "pnl_points": round(pnl, 2), "pnl_ticks": round(pnl_ticks, 1),
                "signal_type": signals.iloc[entry_bar]["signal_type"],
            })
            position = 0

    if not trades:
        print("No trades generated.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_points"] > 0]
    print(f"\n{'='*60}")
    print(f"  DEEP-M EFFORT NQ — PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Trades:   {len(trades_df)}")
    print(f"  Winners:        {len(wins)} ({100*len(wins)/len(trades_df):.1f}%)")
    print(f"  Total PnL:      {trades_df['pnl_points'].sum():.2f} pts")
    print(f"  Avg PnL/Trade:  {trades_df['pnl_points'].mean():.2f} pts")
    gross_loss = abs(trades_df[trades_df["pnl_points"] < 0]["pnl_points"].sum())
    pf = wins["pnl_points"].sum() / gross_loss if gross_loss > 0 else float("inf")
    print(f"  Profit Factor:  {pf:.2f}")
    print(f"{'='*60}")
    print(trades_df.groupby("signal_type")["pnl_points"].agg(["count", "sum", "mean"]).round(2))
    return trades_df


# =============================================================================
# Main — With SCID Support
# =============================================================================

def main():
    config = DeepMEffortConfig(
        range_size=40, tick_size=0.25, ema_period=20,
        zone_std_mult=1.5, absorption_threshold=1.5, vacuum_threshold=0.6,
    )
    builder = RangeBarBuilder(config)

    # =========================================================================
    # OPTION 1: Load real data from Sierra Chart .scid file
    # =========================================================================
    # 
    # Specify the path to your SCID file. Common locations:
    #   Windows: C:\SierraChart\Data\NQZ25-CME.scid
    #   Custom:  D:\Trading\SierraChart\Data\NQZ25-CME.scid
    #
    # Uncomment the block below and set your path:

    SCID_PATH = None  # <-- SET THIS TO YOUR FILE PATH

    # Auto-detect Sierra Chart data directory
    if SCID_PATH is None:
        found = SierraChartDataLocator.find_scid_file("NQZ25-CME")
        if found:
            SCID_PATH = str(found)

    # Explicit path examples (uncomment one):
    # SCID_PATH = r"C:\SierraChart\Data\NQZ25-CME.scid"
    # SCID_PATH = r"D:\SierraChart\Data\NQZ25-CME.scid"
    SCID_PATH = "/opt/SierraChart/Data/NQZ25-CME.scid"

    use_real_data = SCID_PATH is not None and Path(SCID_PATH).exists()

    if use_real_data:
        print(f"\n{'='*60}")
        print(f"  LOADING REAL DATA: {SCID_PATH}")
        print(f"{'='*60}\n")

        # Choose reader: SCIDReader (safe) or SCIDFastReader (fast for big files)
        file_size_mb = Path(SCID_PATH).stat().st_size / (1024 * 1024)

        if file_size_mb > 500:
            print(f"[Large file: {file_size_mb:.0f} MB — using fast reader]")
            reader = SCIDFastReader(SCID_PATH)
        else:
            reader = SCIDReader(SCID_PATH)

        # Load data with optional date filter
        scid_df = reader.read(
            start_date="2024-12-01",  # Adjust dates as needed
            end_date=None,            # None = latest
            trading_hours_only=False, # Set True for RTH only
        )

        if scid_df.empty:
            print("[WARNING] No data loaded from SCID file. Falling back to synthetic.")
            use_real_data = False
        else:
            # Show data summary
            print(f"\n  Intraday records:  {len(scid_df):,}")
            print(f"  Date range:        {scid_df.index[0]} → {scid_df.index[-1]}")
            print(f"  Price range:       {scid_df['low'].min():.2f} — "
                  f"{scid_df['high'].max():.2f}")
            print(f"  Total volume:      {scid_df['volume'].sum():,.0f}")
            print(f"  Total delta:       {scid_df['delta'].sum():,.0f}")
            print()

            # Build range bars from real data
            bars = builder.from_scid_dataframe(scid_df)

    if not use_real_data:
        print("\n[Using synthetic data for demonstration]")
        print("[To use real data, set SCID_PATH to your NQZ25-CME.scid file]\n")

        # List available SCID files if Sierra Chart directory found
        available = SierraChartDataLocator.list_all_scid()
        if available:
            print("Available .scid files found:")
            for f in available[:15]:
                sz = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name:40s} ({sz:.1f} MB)")
            print()

        bars = builder.from_synthetic(n_bars=500, seed=42)

    print(f"Total range bars: {len(bars):,}")

    # Compute effort metrics
    engine = DeepMEffortEngine(config)
    df = engine.compute(bars)
    zones = engine.zones

    print(f"\nDetected {len(zones)} effort zones:")
    for z in zones:
        ts_info = ""
        if z.bar_index_start < len(bars) and bars[z.bar_index_start].timestamp:
            ts_info = f" @ {bars[z.bar_index_start].timestamp}"
        print(f"  {z.zone_type.value:30s} [{z.price_low:.2f}—{z.price_high:.2f}] "
              f"str={z.strength:.2f}{ts_info}")

    # Signals & performance
    signals = SignalGenerator.generate(df, zones)
    n_long = (signals["signal"] == 1).sum()
    n_short = (signals["signal"] == -1).sum()
    print(f"\nSignals: {n_long} long, {n_short} short")

    trades_df = performance_summary(df, signals, config)

    # Chart
    n_bars_chart = min(len(df), 300)
    chart = DeepMEffortChart(df, zones, signals, config)
    chart.plot(start=max(0, len(df) - n_bars_chart), end=len(df))


if __name__ == "__main__":
    main()
