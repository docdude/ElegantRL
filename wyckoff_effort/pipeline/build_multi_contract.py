"""
Multi-Contract NQ Range Bar Pipeline
=====================================
Stitches multiple NQ quarterly contract SCID files into a single
deduplicated chronological bar series, then runs the full Wyckoff
feature pipeline and (optionally) generates transformer structural labels.

Rollover strategy: volume-based front-month windows (non-overlapping).
Each contract contributes only bars from its front-month period.
CVD continuity is preserved by offsetting each segment's cumsum.

Output files (in output_dir):
  wyckoff_nq_40pt_multi.npz              — combined feature NPZ
  wyckoff_nq_40pt_multi_bars.parquet     — raw bars for label generation
  structural_labels_nq_40pt_multi.npz    — transformer phase/event labels

Usage:
  # Full rebuild (default paths, all 5 contracts)
  python -m wyckoff_effort.pipeline.build_multi_contract

  # Custom scid directory or output
  python -m wyckoff_effort.pipeline.build_multi_contract \\
      --scid-dir /opt/SierraChart/Data \\
      --output-dir wyckoff_effort/pipeline_output \\
      --no-labels

  # Dry-run: just count bars per segment without building features
  python -m wyckoff_effort.pipeline.build_multi_contract --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Contract segment definitions
# Front-month date windows for NQ quarterly contracts.
# start_date=None → use full beginning of SCID file.
# end_date=None   → use through end of SCID file.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContractSegment:
    scid_filename: str
    start_date: Optional[str]  # inclusive, "YYYY-MM-DD"
    end_date: Optional[str]    # inclusive, "YYYY-MM-DD"
    label: str

# NQ quarterly front-month windows (volume rollover, 3rd Friday expiry):
#   Z24 expiry: Dec 20, 2024  |  H25 expiry: Mar 21, 2025
#   M25 expiry: Jun 20, 2025  |  U25 expiry: Sep 19, 2025
#   Z25 expiry: Dec 19, 2025 

NQ_SEGMENTS: List[ContractSegment] = [
    ContractSegment("NQZ24-CME.scid", None,         "2024-12-20", "Z24"),
    ContractSegment("NQH25-CME.scid", "2024-12-21", "2025-03-21", "H25"),
    ContractSegment("NQM25-CME.scid", "2025-03-22", "2025-06-20", "M25"),
    ContractSegment("NQU25-CME.scid", "2025-06-21", "2025-09-19", "U25"),
    ContractSegment("NQZ25-CME.scid", "2025-09-20", "2025-12-19", "Z25"),

]

# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────
SCID_DIR = "/opt/SierraChart/Data"
OUTPUT_DIR = "wyckoff_effort/pipeline_output"
RANGE_BAR_SIZE = 40.0
TICK_SIZE = 0.25
REVERSAL_POINTS = 120.0   # 3× bar size for structural wave detection


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _load_segment_bars(
    seg: ContractSegment,
    scid_dir: str,
    bar_size: float,
    tick_size: float,
    dry_run: bool = False,
) -> Optional[pd.DataFrame]:
    """Load one contract segment: SCID ticks → range bars."""
    from ..utils.scid_parser import SCIDReader, resample_range_bars

    scid_path = os.path.join(scid_dir, seg.scid_filename)
    if not os.path.exists(scid_path):
        logger.warning(f"SCID file not found, skipping: {scid_path}")
        return None

    logger.info(f"\n{'─'*55}")
    logger.info(f"[{seg.label}] Loading {seg.scid_filename}")
    logger.info(f"  Date window: {seg.start_date or 'start'} → {seg.end_date or 'end'}")

    reader = SCIDReader(scid_path)
    ticks = reader.read(start_date=seg.start_date, end_date=seg.end_date)

    if len(ticks) == 0:
        logger.warning(f"[{seg.label}] No ticks in date window — skipping")
        return None

    date_min = ticks.index.min()
    date_max = ticks.index.max()
    logger.info(f"  Ticks loaded: {len(ticks):,}  ({date_min.date()} → {date_max.date()})")

    if dry_run:
        # Estimate bar count without full processing
        estimated_bars = len(ticks) // 5000  # rough heuristic
        logger.info(f"  [dry-run] Estimated range bars: ~{estimated_bars:,}")
        # Return sentinel with count stored in attrs
        sentinel = pd.DataFrame()
        sentinel.attrs["dry_run_estimate"] = estimated_bars
        return sentinel

    bars = resample_range_bars(ticks, range_size=bar_size, tick_size=tick_size)
    logger.info(f"  Range bars built: {len(bars):,}")
    if len(bars) > 0:
        logger.info(f"  Bar range: {bars.index.min()} → {bars.index.max()}")
    return bars


def _stitch_segments(
    segments_bars: List[pd.DataFrame],
    back_adjust: bool = True,
) -> tuple[pd.DataFrame, list[float]]:
    """
    Concatenate segment bar DataFrames chronologically, optionally back-adjust
    prices at rollover boundaries, and fix CVD continuity.

    resample_range_bars() produces CVD = delta.cumsum() per segment,
    so each segment starts from 0. We offset each segment so CVD is
    continuous across the full series.
    """
    non_empty = [df for df in segments_bars if df is not None and len(df) > 0]
    if not non_empty:
        raise ValueError("No segment bars to stitch")

    # Compute per-segment price shifts for back-adjustment.
    # Convention: keep newest segment unchanged and shift older segments by
    # cumulative rollover close gaps onto the newest segment's price scale.
    shifts = [0.0 for _ in non_empty]
    if back_adjust and len(non_empty) > 1:
        cumulative = 0.0
        for i in range(len(non_empty) - 2, -1, -1):
            old_seg = non_empty[i]
            new_seg = non_empty[i + 1]
            gap = float(new_seg["close"].iloc[0] - old_seg["close"].iloc[-1])
            cumulative += gap
            shifts[i] = cumulative

    # Apply price shifts and fix CVD continuity: each segment starts at cvd_offset
    adjusted = []
    cvd_offset = 0.0
    price_cols = ("open", "high", "low", "close")
    for i, df in enumerate(non_empty):
        df = df.copy()
        shift = shifts[i]
        if back_adjust and abs(shift) > 0.0:
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col] + shift
        df["cvd"] = df["cvd"] + cvd_offset
        cvd_offset = float(df["cvd"].iloc[-1])
        adjusted.append(df)

    combined = pd.concat(adjusted, axis=0)
    combined = combined.sort_index()

    # Sanity check: remove any duplicate timestamps (shouldn't happen with
    # non-overlapping date windows, but protect against edge cases)
    n_before = len(combined)
    combined = combined[~combined.index.duplicated(keep="first")]
    n_removed = n_before - len(combined)
    if n_removed > 0:
        logger.warning(f"Removed {n_removed} duplicate-timestamp bars at rollover boundaries")

    return combined, shifts


def run_multi_contract_pipeline(
    scid_dir: str = SCID_DIR,
    output_dir: str = OUTPUT_DIR,
    bar_size: float = RANGE_BAR_SIZE,
    tick_size: float = TICK_SIZE,
    reversal_points: float = REVERSAL_POINTS,
    segments: Optional[List[ContractSegment]] = None,
    run_labels: bool = True,
    back_adjust: bool = True,
    run_importance: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Full multi-contract build: SCID segments → stitched bars → features → NPZ + labels.

    Parameters
    ----------
    scid_dir : str
        Directory containing SCID files.
    output_dir : str
        Output directory for NPZ, parquet, and label files.
    bar_size : float
        Range bar size in points (40.0 for NQ).
    tick_size : float
        Minimum price movement (0.25 for NQ).
    reversal_points : float
        ZigZag reversal for Weis Wave (120.0 = 3× bar size).
    segments : list of ContractSegment, optional
        Override the default NQ_SEGMENTS list.
    run_labels : bool
        If True, generate transformer structural labels after feature build.
    back_adjust : bool
        If True, apply cumulative rollover offsets so stitched prices are
        continuous (latest segment remains unadjusted).
    run_importance : bool
        If True, run feature importance evaluation (slow, skipped by default
        for large multi-contract datasets).
    dry_run : bool
        If True, estimate bar counts only — skip feature build and saving.

    Returns
    -------
    dict with keys: npz_path, bars_path, label_path, n_bars, n_features,
                    n_bars_per_contract, feature_names
    """
    from ..utils.scid_parser import SCIDReader, resample_range_bars
    from .wyckoff_features import build_all_features

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    segments = segments or NQ_SEGMENTS
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()
    logger.info("=" * 55)
    logger.info("MULTI-CONTRACT NQ RANGE BAR PIPELINE")
    logger.info("=" * 55)
    logger.info(f"SCID dir:        {scid_dir}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info(f"Bar size:        {bar_size}pt  (tick={tick_size})")
    logger.info(f"Reversal:        {reversal_points}pt")
    logger.info(f"Contracts:       {[s.label for s in segments]}")
    logger.info(f"Structural labels: {'yes' if run_labels else 'no'}")
    logger.info(f"Back-adjust:     {'yes' if back_adjust else 'no'}")

    # ── Step 1: Load each segment ─────────────────────────────────────────
    segment_bars: List[Optional[pd.DataFrame]] = []
    n_bars_per_contract: dict = {}

    for seg in segments:
        bars = _load_segment_bars(seg, scid_dir, bar_size, tick_size, dry_run=dry_run)
        segment_bars.append(bars)
        n_bars_per_contract[seg.label] = len(bars) if bars is not None else 0
        # In dry-run mode, use the pre-computed estimate from attrs
        if dry_run and bars is not None and hasattr(bars, 'attrs') and 'dry_run_estimate' in bars.attrs:
            n_bars_per_contract[seg.label] = bars.attrs["dry_run_estimate"]

    if dry_run:
        logger.info("\n[dry-run] Bar estimates per contract:")
        total_est = 0
        for seg, bars in zip(segments, segment_bars):
            count = n_bars_per_contract[seg.label]
            logger.info(f"  {seg.label}: ~{count:,} bars")
            total_est += count
        logger.info(f"  TOTAL: ~{total_est:,} bars")
        return {"dry_run": True, "n_bars_per_contract": n_bars_per_contract}

    # ── Step 2: Stitch segments ───────────────────────────────────────────
    logger.info(f"\n{'─'*55}")
    logger.info("Stitching segments + back-adjusting prices + fixing CVD continuity...")
    combined_bars, shifts = _stitch_segments(segment_bars, back_adjust=back_adjust)

    total_bars = len(combined_bars)
    date_min = combined_bars.index.min()
    date_max = combined_bars.index.max()
    logger.info(f"Combined bars: {total_bars:,}  ({date_min.date()} → {date_max.date()})")
    logger.info("")
    for seg in segments:
        nb = n_bars_per_contract.get(seg.label, 0)
        pct = 100 * nb / total_bars if total_bars else 0
        logger.info(f"  {seg.label}: {nb:,} bars  ({pct:.1f}%)")
    if back_adjust:
        logger.info("\n  Applied cumulative rollover offsets:")
        for seg, shift in zip(segments, shifts):
            logger.info(f"    {seg.label}: {shift:+.2f}")
    logger.info("")

    # Save raw bars parquet (needed for label generation)
    bars_path = os.path.join(output_dir, f"wyckoff_nq_{int(bar_size)}pt_multi_bars.parquet")
    combined_bars.to_parquet(bars_path)
    logger.info(f"Saved bars: {bars_path}")

    # ── Step 3: Build features ────────────────────────────────────────────
    logger.info(f"\n{'─'*55}")
    logger.info(f"Building Wyckoff features (reversal={reversal_points}pt)...")
    t_feat = time.time()
    tech_ary, feature_names, feat_df = build_all_features(
        combined_bars,
        reversal_points=reversal_points,
    )
    logger.info(f"Feature build: {time.time() - t_feat:.1f}s  →  {tech_ary.shape}")

    # ── Step 4: Save NPZ ──────────────────────────────────────────────────
    logger.info(f"\n{'─'*55}")
    close_ary = combined_bars["close"].values.reshape(-1, 1).astype(np.float32)

    if isinstance(combined_bars.index, pd.DatetimeIndex):
        dates_ary = combined_bars.index.astype(str).values
    else:
        dates_ary = np.arange(len(combined_bars)).astype(str)

    npz_path = os.path.join(output_dir, f"wyckoff_nq_{int(bar_size)}pt_multi.npz")
    np.savez_compressed(
        npz_path,
        close_ary=close_ary,
        tech_ary=tech_ary,
        dates_ary=dates_ary,
        feature_names=np.array(feature_names),
    )
    size_mb = os.path.getsize(npz_path) / 1024 / 1024
    logger.info(f"Saved NPZ: {npz_path} ({size_mb:.1f} MB, {tech_ary.shape[1]} features)")

    result = {
        "npz_path": npz_path,
        "bars_path": bars_path,
        "label_path": None,
        "n_bars": total_bars,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_bars_per_contract": n_bars_per_contract,
        "back_adjust": back_adjust,
        "segment_shifts": shifts,
    }

    # ── Step 5: Structural labels ─────────────────────────────────────────
    if run_labels:
        logger.info(f"\n{'─'*55}")
        logger.info("Generating transformer structural labels...")
        try:
            from wyckoff_rl.transformer.labels import generate_structural_labels, save_labels
            t_labels = time.time()
            labels = generate_structural_labels(bars_path)
            label_path = os.path.join(output_dir, f"structural_labels_nq_{int(bar_size)}pt_multi.npz")
            save_labels(labels, label_path)
            logger.info(f"Labels: {time.time() - t_labels:.1f}s  →  {label_path}")
            result["label_path"] = label_path
        except Exception as e:
            logger.error(f"Label generation failed: {e}")
            logger.info("Re-run with: python -m wyckoff_rl.transformer.labels <bars_path>")

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*55}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total bars:      {total_bars:,}")
    logger.info(f"  Features:        {len(feature_names)}")
    logger.info(f"  Date range:      {date_min.date()} → {date_max.date()}")
    logger.info(f"  NPZ:             {npz_path}")
    logger.info(f"  Bars parquet:    {bars_path}")
    if result["label_path"]:
        logger.info(f"  Labels:          {result['label_path']}")
    logger.info(f"  Elapsed:         {elapsed/60:.1f} min")
    logger.info(f"{'='*55}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build multi-contract NQ range bar + feature NPZ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full rebuild with default paths
  python -m wyckoff_effort.pipeline.build_multi_contract

  # Custom scid dir and output
  python -m wyckoff_effort.pipeline.build_multi_contract \\
      --scid-dir /opt/SierraChart/Data \\
      --output-dir wyckoff_effort/pipeline_output

  # Dry-run: just count bars per segment
  python -m wyckoff_effort.pipeline.build_multi_contract --dry-run

  # Skip structural label generation (faster)
  python -m wyckoff_effort.pipeline.build_multi_contract --no-labels
""",
    )
    parser.add_argument("--scid-dir", type=str, default=SCID_DIR,
                        help=f"Directory containing SCID files (default: {SCID_DIR})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--bar-size", type=float, default=RANGE_BAR_SIZE,
                        help=f"Range bar size in points (default: {RANGE_BAR_SIZE})")
    parser.add_argument("--reversal", type=float, default=REVERSAL_POINTS,
                        help=f"ZigZag reversal points (default: {REVERSAL_POINTS})")
    parser.add_argument("--no-labels", action="store_true",
                        help="Skip structural label generation")
    parser.add_argument("--no-back-adjust", action="store_true",
                        help="Disable rollover back-adjust offsets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate bar counts only, no feature build")
    parser.add_argument("--importance", action="store_true",
                        help="Run feature importance evaluation (slow)")

    args = parser.parse_args()

    result = run_multi_contract_pipeline(
        scid_dir=args.scid_dir,
        output_dir=args.output_dir,
        bar_size=args.bar_size,
        reversal_points=args.reversal,
        run_labels=not args.no_labels,
        back_adjust=not args.no_back_adjust,
        run_importance=args.importance,
        dry_run=args.dry_run,
    )
    return result


if __name__ == "__main__":
    main()
