#!/usr/bin/env python3
"""
Pre-normalize Wyckoff NPZ tech_ary: z-score → tanh per feature column.

Reads the raw selected NPZ, computes per-column mean/std, applies
z-score + tanh to produce features in [-1, 1], and saves:
  - wyckoff_nq_normalized.npz  (close_ary, tech_ary, feature_names, dates_ary,
                                 tech_mean, tech_std)

The mean/std are saved inside the NPZ so the same transform can be applied
to new data at inference time.

Usage:
    python -m wyckoff_rl.normalize_npz
    python -m wyckoff_rl.normalize_npz --input path/to/raw.npz --output path/to/norm.npz
"""

import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wyckoff_rl.config import DATA_DIR


def normalize_tech_ary(tech_ary: np.ndarray) -> tuple:
    """Z-score + tanh normalization per column.

    Returns: (normalized_tech, mean, std)
    """
    mean = tech_ary.mean(axis=0)
    std = tech_ary.std(axis=0)
    # Clamp std to avoid division by zero for constant columns
    std = np.maximum(std, 1e-8)
    z = (tech_ary - mean) / std
    normalized = np.tanh(z).astype(np.float32)
    return normalized, mean.astype(np.float32), std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Normalize Wyckoff NPZ tech_ary")
    parser.add_argument("--input", type=str,
                        default=os.path.join(DATA_DIR, "wyckoff_nq_selected.npz"))
    parser.add_argument("--output", type=str,
                        default=os.path.join(DATA_DIR, "wyckoff_nq_normalized.npz"))
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input NPZ not found: {args.input}")
        return 1

    print(f"Loading: {args.input}")
    data = np.load(args.input, allow_pickle=True)
    close_ary = data['close_ary']
    tech_ary = data['tech_ary'].astype(np.float32)
    feature_names = data['feature_names'] if 'feature_names' in data else np.array([])
    dates_ary = data['dates_ary'] if 'dates_ary' in data else None

    print(f"  Bars: {tech_ary.shape[0]:,}, Features: {tech_ary.shape[1]}")

    # Show before
    print(f"\n  BEFORE (raw):")
    for i, name in enumerate(feature_names):
        col = tech_ary[:, i]
        print(f"    {str(name):18s}: "
              f"mean={col.mean():10.4f}  std={col.std():10.4f}  "
              f"min={col.min():10.4f}  max={col.max():10.4f}")

    # Normalize
    normalized, tech_mean, tech_std = normalize_tech_ary(tech_ary)

    # Show after
    print(f"\n  AFTER (z-score + tanh):")
    for i, name in enumerate(feature_names):
        col = normalized[:, i]
        print(f"    {str(name):18s}: "
              f"mean={col.mean():10.4f}  std={col.std():10.4f}  "
              f"min={col.min():10.4f}  max={col.max():10.4f}")

    # Save
    save_dict = {
        'close_ary': close_ary,
        'tech_ary': normalized,
        'feature_names': feature_names,
        'tech_mean': tech_mean,
        'tech_std': tech_std,
    }
    if dates_ary is not None:
        save_dict['dates_ary'] = dates_ary

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **save_dict)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\n  Saved: {args.output} ({size_mb:.1f} MB)")
    print(f"  tech_mean saved: {tech_mean}")
    print(f"  tech_std saved:  {tech_std}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
