"""
Phase 1: Wyckoff Feature Extraction → NPZ

SCID ticks → range bars → analyze_wyckoff() → numeric feature columns → NPZ

Output format matches ElegantRL's StockTradingVecEnv:
    close_ary.shape = (n_bars, 1)       # single instrument (NQ)
    tech_ary.shape  = (n_bars, n_features)
    dates_ary.shape = (n_bars,)         # optional timestamps
"""

import os
import logging
import numpy as np
import pandas as pd

from .utils.scid_parser import SCIDReader, resample_range_bars
from .utils.wyckoff_analyzer import analyze_wyckoff
from .config import (
    DEFAULT_SCID_PATH, DEFAULT_NPZ_PATH, OUTPUT_DIR,
    RANGE_BAR_SIZE, RTH_ONLY, WYCKOFF_PARAMS,
    WYCKOFF_FEATURE_COLUMNS, N_WYCKOFF_FEATURES,
)

logger = logging.getLogger(__name__)


def load_scid_ticks(scid_path: str = None) -> pd.DataFrame:
    """Load raw ticks from an SCID file."""
    scid_path = scid_path or DEFAULT_SCID_PATH
    if not os.path.exists(scid_path):
        raise FileNotFoundError(f"SCID file not found: {scid_path}")

    reader = SCIDReader(scid_path)
    df = reader.read()
    logger.info(f"Loaded {len(df):,} ticks from {scid_path}")
    return df


def build_range_bars(
    ticks: pd.DataFrame,
    bar_size: float = None,
    rth_only: bool = None,
) -> pd.DataFrame:
    """Build range bars from tick data."""
    bar_size = bar_size or RANGE_BAR_SIZE
    rth_only = rth_only if rth_only is not None else RTH_ONLY

    bars = resample_range_bars(ticks, range_size=bar_size)
    logger.info(f"Built {len(bars):,} range bars (size={bar_size})")

    if rth_only and "datetime" in bars.columns:
        bars = bars[bars["datetime"].dt.hour.between(7, 14)]
        logger.info(f"RTH filter: {len(bars):,} bars remaining")

    return bars


def run_wyckoff_analysis(
    bars: pd.DataFrame,
    params: dict = None,
) -> pd.DataFrame:
    """Run analyze_wyckoff() on range bars."""
    params = {**WYCKOFF_PARAMS, **(params or {})}
    df = analyze_wyckoff(bars, **params)
    logger.info(f"Wyckoff analysis complete on {len(df):,} bars")
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features that aren't direct analyze_wyckoff() outputs.

    Adds: delta_norm, cvd_slope
    """
    # Normalized delta: bar delta / rolling avg volume
    avg_vol = df["volume"].rolling(window=20, min_periods=1).mean()
    df["delta_norm"] = (df["delta"] / avg_vol.replace(0, np.nan)).fillna(0).clip(-1, 1)

    # CVD slope: tanh-normalized slope of CVD over last 20 bars
    cvd = df["cvd"].values.astype(np.float64)
    slopes = np.zeros(len(cvd))
    for i in range(1, len(cvd)):
        lookback = min(20, i + 1)
        window = cvd[max(0, i - lookback + 1): i + 1]
        if len(window) > 1:
            x = np.arange(len(window))
            slope = np.polyfit(x, window, 1)[0]
            vol = avg_vol.iloc[i] if avg_vol.iloc[i] > 0 else 1.0
            slopes[i] = np.tanh(slope / vol)
    df["cvd_slope"] = slopes

    return df


def extract_feature_arrays(
    df: pd.DataFrame,
    feature_columns: list = None,
) -> tuple:
    """
    Extract close_ary and tech_ary from the analyzed DataFrame.

    Returns
    -------
    close_ary : np.ndarray, shape (n_bars, 1)
    tech_ary  : np.ndarray, shape (n_bars, n_features)
    dates_ary : np.ndarray, shape (n_bars,) — string timestamps
    """
    feature_columns = feature_columns or WYCKOFF_FEATURE_COLUMNS

    # Close prices — single instrument
    close_ary = df["close"].values.reshape(-1, 1).astype(np.float32)

    # Build tech_ary from feature columns
    tech_data = []
    missing = []
    for col in feature_columns:
        if col in df.columns:
            tech_data.append(df[col].values.astype(np.float32))
        else:
            missing.append(col)
            tech_data.append(np.zeros(len(df), dtype=np.float32))

    if missing:
        logger.warning(f"Missing feature columns (filled with 0): {missing}")

    tech_ary = np.column_stack(tech_data)

    # Replace NaN/inf with 0
    tech_ary = np.nan_to_num(tech_ary, nan=0.0, posinf=0.0, neginf=0.0)

    # Dates
    if "datetime" in df.columns:
        dates_ary = df["datetime"].astype(str).values
    elif isinstance(df.index, pd.DatetimeIndex):
        dates_ary = df.index.astype(str).values
    else:
        dates_ary = np.arange(len(df)).astype(str)

    logger.info(
        f"Extracted arrays: close_ary={close_ary.shape}, "
        f"tech_ary={tech_ary.shape} ({len(feature_columns)} features)"
    )
    return close_ary, tech_ary, dates_ary


def save_npz(
    close_ary: np.ndarray,
    tech_ary: np.ndarray,
    dates_ary: np.ndarray = None,
    output_path: str = None,
    feature_names: list = None,
) -> str:
    """Save arrays to NPZ format compatible with ElegantRL."""
    output_path = output_path or DEFAULT_NPZ_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        "close_ary": close_ary,
        "tech_ary": tech_ary,
    }
    if dates_ary is not None:
        data["dates_ary"] = dates_ary
    if feature_names is not None:
        data["feature_names"] = np.array(feature_names)

    np.savez_compressed(output_path, **data)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(f"Saved NPZ: {output_path} ({size_mb:.1f} MB)")
    return output_path


def run_full_pipeline(
    scid_path: str = None,
    output_path: str = None,
    bar_size: float = None,
    rth_only: bool = None,
    wyckoff_params: dict = None,
) -> dict:
    """
    Run the complete Phase 1 pipeline: SCID → NPZ.

    Returns dict with arrays and metadata.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Load ticks
    ticks = load_scid_ticks(scid_path)

    # 2. Build range bars
    bars = build_range_bars(ticks, bar_size=bar_size, rth_only=rth_only)

    # 3. Run Wyckoff analysis
    df = run_wyckoff_analysis(bars, params=wyckoff_params)

    # 4. Compute derived features
    df = compute_derived_features(df)

    # 5. Extract arrays
    close_ary, tech_ary, dates_ary = extract_feature_arrays(df)

    # 6. Save NPZ
    npz_path = save_npz(
        close_ary, tech_ary, dates_ary,
        output_path=output_path,
        feature_names=WYCKOFF_FEATURE_COLUMNS,
    )

    # 7. Save analyzed DataFrame for downstream use
    parquet_path = npz_path.replace(".npz", "_analyzed.parquet")
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved analyzed DataFrame: {parquet_path}")

    return {
        "close_ary": close_ary,
        "tech_ary": tech_ary,
        "dates_ary": dates_ary,
        "npz_path": npz_path,
        "parquet_path": parquet_path,
        "n_bars": len(df),
        "n_features": tech_ary.shape[1],
        "feature_names": WYCKOFF_FEATURE_COLUMNS,
    }


if __name__ == "__main__":
    result = run_full_pipeline()
    print(f"\nPipeline complete:")
    print(f"  Bars:     {result['n_bars']:,}")
    print(f"  Features: {result['n_features']}")
    print(f"  NPZ:      {result['npz_path']}")
    print(f"  Parquet:  {result['parquet_path']}")
