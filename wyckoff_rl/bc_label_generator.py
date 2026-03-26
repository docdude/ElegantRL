"""
Behavioral Cloning Label Generator
====================================

Reads a Wyckoff NPZ file and generates expert (state, action) pairs
by simulating trades using the CiB/Spring/Upthrust detectors as the
"expert trader."

Expert policy:
  - Enter long on spring or bullish CiB
  - Enter short on upthrust or bearish CiB
  - Exit on take-profit (2× bar_range), stop-loss (1.5× bar_range), or time (30 bars)
  - Hold otherwise

Output: .pt file with {states, actions, state_avg, state_std} ready for
supervised pre-training of ActorDiscretePPO.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch as th
from pathlib import Path

# ── Constants (must match NQWyckoffWeisEnv.py) ────────────────────────────

ACTION_HOLD = 0
ACTION_ENTER_LONG = 1
ACTION_ENTER_SHORT = 2
ACTION_ADD = 3
ACTION_REDUCE = 4
ACTION_EXIT = 5
N_ACTIONS = 6

# CiB thresholds
CIB_MIN_PB_VOL_RATIO = 0.05
CIB_MAX_PB_VOL_RATIO = 0.40
CIB_MAX_LARGE_WAVE = 0.50
EVENT_THRESHOLD = 0.3

# Feature column indices in 72-col tech_ary
COL_WAVE_DIR = 15
COL_WAVE_VOL_VS_PREV = 21
COL_LARGE_WAVE = 34
COL_SPRING = 35
COL_UPTHRUST = 36
COL_ABSORPTION = 39
COL_STOPPING = 41

# Selected features for state vector (must match ENV_FEATURE_INDICES)
ENV_FEATURE_INDICES = [
    0, 5, 7, 9, 11, 13, 14,
    16, 17, 18, 19, 20, 21, 23, 24, 34, 58, 59, 60,
    35, 36, 37, 39, 41, 42, 43,
    48, 49, 53, 55, 56, 47,
    61, 62, 63, 64, 65, 67, 68, 69,
]
N_ENV_FEATURES = len(ENV_FEATURE_INDICES)  # 44
N_POSITION_FEATURES = 8


def detect_entry_signal(tech_row: np.ndarray) -> int:
    """
    Check if this bar has a Wyckoff entry signal.

    Returns: ACTION_ENTER_LONG, ACTION_ENTER_SHORT, or ACTION_HOLD.
    """
    spring = tech_row[COL_SPRING]
    upthrust = tech_row[COL_UPTHRUST]
    wave_dir = tech_row[COL_WAVE_DIR]
    vol_vs_prev = tech_row[COL_WAVE_VOL_VS_PREV]
    large_wave = tech_row[COL_LARGE_WAVE]

    # CiB pullback conditions
    cib_pullback = (CIB_MIN_PB_VOL_RATIO < vol_vs_prev < CIB_MAX_PB_VOL_RATIO
                    and large_wave < CIB_MAX_LARGE_WAVE)

    # Long signals: spring OR bullish CiB (in DOWN pullback → long)
    if spring > EVENT_THRESHOLD:
        return ACTION_ENTER_LONG
    if cib_pullback and wave_dir < 0:
        return ACTION_ENTER_LONG

    # Short signals: upthrust OR bearish CiB (in UP pullback → short)
    if upthrust > EVENT_THRESHOLD:
        return ACTION_ENTER_SHORT
    if cib_pullback and wave_dir > 0:
        return ACTION_ENTER_SHORT

    return ACTION_HOLD


def build_state_vector(
    tech_row: np.ndarray,
    curr_price: float,
    pos_side: float,
    pos_size: float,
    entry_price: float,
    unrealized_pnl: float,
    realized_pnl: float,
    bars_in_trade: int,
    mfe: float,
    mae: float,
    bar_range: float,
    pnl_norm: float,
    max_position_size: int,
) -> np.ndarray:
    """Construct the 52D state vector matching NQWyckoffWeisVecEnv.get_state()."""
    # Market features (44)
    market_feats = tech_row[ENV_FEATURE_INDICES].astype(np.float32)

    # Position features (8)
    if pos_side != 0 and bar_range > 0:
        entry_dist = ((curr_price - entry_price) / bar_range) * pos_side
    else:
        entry_dist = 0.0

    pos_feats = np.array([
        float(pos_side),
        float(pos_size / max(max_position_size, 1.0)),
        float(entry_dist),
        float(unrealized_pnl / pnl_norm),
        float(np.clip(realized_pnl / (2.0 * pnl_norm), -1.0, 1.0)),
        float(min(bars_in_trade / 50.0, 1.0)),
        float(mfe / pnl_norm),
        float(mae / pnl_norm),
    ], dtype=np.float32)

    state = np.concatenate([market_feats, pos_feats])
    return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)


def generate_labels(
    npz_path: str,
    bar_range: float = 40.0,
    pnl_norm: float = 10000.0,
    commission: float = 1.50,
    slippage_ticks: float = 1.0,
    tick_size: float = 0.25,
    tick_value: float = 5.0,
    max_position_size: int = 2,
    tp_mult: float = 2.0,        # take-profit in multiples of bar_range
    sl_mult: float = 1.5,        # stop-loss in multiples of bar_range
    time_stop: int = 30,         # max bars to hold
    warmup: int = 50,            # skip first N bars for feature stability
) -> dict:
    """
    Generate behavioral cloning dataset from NPZ.

    Returns dict with:
      states:  (N, state_dim) float32 tensor
      actions: (N,) long tensor
      state_avg: (state_dim,) mean for normalization
      state_std: (state_dim,) std for normalization
      stats: dict with label counts
    """
    data = np.load(npz_path, allow_pickle=True)
    close_ary = data["close_ary"].astype(np.float32).flatten()
    tech_ary = data["tech_ary"].astype(np.float32)
    n_bars = len(close_ary)

    slip = slippage_ticks * tick_size
    point_value = tick_value / tick_size  # $ per point

    states_list = []
    actions_list = []

    # Simulate expert trading
    pos_side = 0.0      # -1, 0, +1
    pos_size = 0.0
    entry_price = 0.0
    unrealized_pnl = 0.0
    realized_pnl = 0.0
    bars_in_trade = 0
    mfe = 0.0
    mae = 0.0

    tp_pts = tp_mult * bar_range
    sl_pts = sl_mult * bar_range

    for t in range(warmup, n_bars):
        curr_price = close_ary[t]
        tech_row = tech_ary[t]

        # Update unrealized PnL, MFE, MAE if positioned
        if pos_side != 0:
            raw_pnl = (curr_price - entry_price) * pos_side * point_value * pos_size
            unrealized_pnl = raw_pnl
            if raw_pnl > mfe:
                mfe = raw_pnl
            if raw_pnl < mae:
                mae = raw_pnl
            bars_in_trade += 1

        # Build state
        state = build_state_vector(
            tech_row=tech_row,
            curr_price=curr_price,
            pos_side=pos_side,
            pos_size=pos_size,
            entry_price=entry_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            bars_in_trade=bars_in_trade,
            mfe=mfe,
            mae=mae,
            bar_range=bar_range,
            pnl_norm=pnl_norm,
            max_position_size=max_position_size,
        )

        # Determine expert action
        if pos_side == 0:
            # Flat — look for entry signals
            action = detect_entry_signal(tech_row)
        else:
            # Positioned — check exit conditions
            pnl_pts = (curr_price - entry_price) * pos_side

            if pnl_pts >= tp_pts:
                action = ACTION_EXIT  # take profit
            elif pnl_pts <= -sl_pts:
                action = ACTION_EXIT  # stop loss
            elif bars_in_trade >= time_stop:
                action = ACTION_EXIT  # time stop
            else:
                action = ACTION_HOLD

        states_list.append(state)
        actions_list.append(action)

        # Execute expert action (update position)
        if action == ACTION_ENTER_LONG and pos_side == 0:
            pos_side = 1.0
            pos_size = 1.0
            entry_price = curr_price + slip
            unrealized_pnl = 0.0
            realized_pnl -= commission
            bars_in_trade = 0
            mfe = 0.0
            mae = 0.0
        elif action == ACTION_ENTER_SHORT and pos_side == 0:
            pos_side = -1.0
            pos_size = 1.0
            entry_price = curr_price - slip
            unrealized_pnl = 0.0
            realized_pnl -= commission
            bars_in_trade = 0
            mfe = 0.0
            mae = 0.0
        elif action == ACTION_EXIT and pos_side != 0:
            exit_price = curr_price - slip if pos_side > 0 else curr_price + slip
            exit_pnl = (exit_price - entry_price) * pos_side * point_value * pos_size
            realized_pnl += exit_pnl - commission
            pos_side = 0.0
            pos_size = 0.0
            entry_price = 0.0
            unrealized_pnl = 0.0
            bars_in_trade = 0
            mfe = 0.0
            mae = 0.0

    states = np.stack(states_list)
    actions = np.array(actions_list, dtype=np.int64)

    # Compute normalization stats
    state_avg = states.mean(axis=0)
    state_std = states.std(axis=0)
    state_std[state_std < 1e-6] = 1.0  # avoid div-by-zero

    # Count labels
    unique, counts = np.unique(actions, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    print(f"\n{'='*60}")
    print(f"BC Label Generation: {npz_path}")
    print(f"  Total samples: {len(actions):,}")
    for a in range(N_ACTIONS):
        c = label_counts.get(a, 0)
        pct = 100.0 * c / len(actions) if len(actions) > 0 else 0
        names = ["HOLD", "ENTER_L", "ENTER_S", "ADD", "REDUCE", "EXIT"]
        print(f"  {names[a]:>8s}: {c:>7,}  ({pct:5.1f}%)")
    print(f"{'='*60}\n")

    return {
        "states": th.tensor(states, dtype=th.float32),
        "actions": th.tensor(actions, dtype=th.long),
        "state_avg": th.tensor(state_avg, dtype=th.float32),
        "state_std": th.tensor(state_std, dtype=th.float32),
        "stats": label_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate BC labels from Wyckoff NPZ")
    parser.add_argument("--npz", required=True, help="Path to NPZ file")
    parser.add_argument("--output", default=None, help="Output .pt path (default: bc_labels_<instrument>.pt)")
    parser.add_argument("--instrument", default="nq", choices=["nq", "us30"])
    parser.add_argument("--tp-mult", type=float, default=2.0, help="Take-profit multiple of bar_range")
    parser.add_argument("--sl-mult", type=float, default=1.5, help="Stop-loss multiple of bar_range")
    parser.add_argument("--time-stop", type=int, default=30, help="Max bars to hold")
    args = parser.parse_args()

    from wyckoff_rl.config import INSTRUMENT_PRESETS
    preset = INSTRUMENT_PRESETS.get(args.instrument, INSTRUMENT_PRESETS["nq"])

    result = generate_labels(
        npz_path=args.npz,
        bar_range=preset["bar_range"],
        pnl_norm=preset["pnl_norm"],
        commission=preset["commission"],
        slippage_ticks=preset["slippage_ticks"],
        tick_size=preset["tick_size"],
        tick_value=preset["tick_value"],
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        time_stop=args.time_stop,
    )

    output = args.output or f"bc_labels_{args.instrument}.pt"
    th.save(result, output)
    print(f"Saved BC dataset to {output}")
    print(f"  states:  {result['states'].shape}")
    print(f"  actions: {result['actions'].shape}")


if __name__ == "__main__":
    main()
