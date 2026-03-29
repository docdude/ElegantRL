"""
Wyckoff Learning Evaluation Harness
====================================
Loads a trained RL checkpoint, runs deterministic rollouts through the
full dataset, and reports whether the agent's entries correlate with
Wyckoff signals or it found a simpler shortcut.

Outputs:
  1. Action distribution (% HOLD, ENTER_LONG, ENTER_SHORT, etc.)
  2. Entry-signal correlation (% of entries on spring/upthrust/CiB bars)
  3. Trade-level stats (win rate, avg bars held, avg win vs avg loss)
  4. Phase alignment (entries in accum→long, distrib→short, etc.)
  5. Per-trade detail CSV for manual inspection

Usage:
  python -m wyckoff_rl.transformer.evaluate_wyckoff \
    --checkpoint wyckoff_rl/transformer/checkpoints/.../rl_step204800.pt \
    --npz wyckoff_effort/pipeline_output/wyckoff_nq_100pt_multi.npz \
    --seq-len 64 --d-model 32 --n-layers 2 --n-heads 4 \
    --bar-range 100.0 --tick-value 20.0 --commission 1.00
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .config_v2 import (
    TransformerConfig,
    TRANSFORMER_FEATURE_INDICES,
    PHASE_LABELS,
    EVENT_LABELS,
)
from .env import (
    ACTION_HOLD,
    ACTION_ENTER_LONG,
    ACTION_ENTER_SHORT,
    ACTION_ADD,
    ACTION_REDUCE,
    ACTION_EXIT,
    N_ACTIONS,
    N_POSITION_FEATURES,
)
from .actor_v2 import ActorDiscreteTransformer


ACTION_NAMES = {
    ACTION_HOLD: "HOLD",
    ACTION_ENTER_LONG: "ENTER_LONG",
    ACTION_ENTER_SHORT: "ENTER_SHORT",
    ACTION_ADD: "ADD",
    ACTION_REDUCE: "REDUCE",
    ACTION_EXIT: "EXIT",
}

# Full tech_ary column indices for Wyckoff signals
COL_SPRING = 35
COL_UPTHRUST = 36
COL_SC = 37
COL_BC = 38
COL_ABSORPTION = 39
COL_STOPPING = 41
COL_WAVE_DIR = 15
COL_WAVE_VOL_VS_PREV = 21
COL_LARGE_WAVE = 34
COL_PHASE_ACCUM = 53
COL_PHASE_MARKUP = 54
COL_PHASE_DISTRIB = 55
COL_PHASE_MARKDOWN = 56


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float
    pnl: float
    bars_held: int
    mfe: float
    mae: float
    # Wyckoff context at entry
    spring_score: float
    upthrust_score: float
    absorption_score: float
    stopping_score: float
    wave_direction: float
    cib_active: bool
    phase_accum: float
    phase_markup: float
    phase_distrib: float
    phase_markdown: float


def load_checkpoint(checkpoint_path: str, config: TransformerConfig, device: torch.device):
    actor = ActorDiscreteTransformer(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    elif "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    elif "model_state_dict" in ckpt:
        actor.load_state_dict(ckpt["model_state_dict"])
    else:
        actor.load_state_dict(ckpt)
    actor.eval()
    print(f"  Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'raw state_dict'}")
    if isinstance(ckpt, dict) and "steps" in ckpt:
        print(f"  Steps: {ckpt['steps']:,} | Episodes: {ckpt.get('episodes','?')} | avg_pnl: ${ckpt.get('avg_pnl', '?')}")
    return actor


def run_deterministic_eval(
    actor: ActorDiscreteTransformer,
    npz_path: str,
    config: TransformerConfig,
    commission: float,
    tick_size: float,
    tick_value: float,
    bar_range: float,
    slippage_ticks: float = 1.0,
    device: torch.device = torch.device("cpu"),
):
    """
    Run the actor deterministically (argmax) across the full dataset,
    one bar at a time with a single env (no staggering, no sign flip).
    """
    data = np.load(npz_path, allow_pickle=True)
    close_ary = data["close_ary"].astype(np.float32).flatten()
    tech_ary = data["tech_ary"].astype(np.float32)
    n_bars = len(close_ary)

    fi = list(config.feature_indices) if config.feature_indices else TRANSFORMER_FEATURE_INDICES
    fi_t = torch.tensor(fi, dtype=torch.long, device=device)

    close_t = torch.tensor(close_ary, device=device)
    tech_t = torch.tensor(tech_ary, device=device)
    selected = tech_t[:, fi_t]  # (n_bars, 41)

    sl = config.seq_len
    pad = torch.zeros(sl - 1, len(fi), dtype=torch.float32, device=device)
    padded = torch.cat([pad, selected], dim=0)

    slip = slippage_ticks * tick_size

    # Tracking
    actions_taken = []
    bar_features_at_entry = []  # (bar_idx, tech_ary row)
    trades: list[Trade] = []

    # Position state
    pos_side = 0.0   # -1, 0, +1
    pos_size = 0.0
    entry_price = 0.0
    entry_bar = 0
    bars_in_trade = 0
    realized_pnl = 0.0
    mfe = 0.0
    mae = 0.0
    unrealized_pnl = 0.0

    def compute_pnl_at(price):
        if pos_side == 0:
            return 0.0
        ticks = ((price - entry_price) / tick_size) * pos_side
        return ticks * tick_value * pos_size

    def close_trade(exit_price, exit_bar):
        nonlocal realized_pnl, pos_side, pos_size, entry_price, bars_in_trade, mfe, mae, unrealized_pnl
        pnl = compute_pnl_at(exit_price) - commission * pos_size
        feats_at_entry = tech_ary[entry_bar]
        vol_ratio = feats_at_entry[COL_WAVE_VOL_VS_PREV]
        lw = feats_at_entry[COL_LARGE_WAVE]
        cib = (0.3 < vol_ratio < 0.85) and (lw < 0.5)

        trades.append(Trade(
            entry_bar=entry_bar,
            exit_bar=exit_bar,
            side="long" if pos_side > 0 else "short",
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            bars_held=bars_in_trade,
            mfe=mfe,
            mae=mae,
            spring_score=float(feats_at_entry[COL_SPRING]),
            upthrust_score=float(feats_at_entry[COL_UPTHRUST]),
            absorption_score=float(feats_at_entry[COL_ABSORPTION]),
            stopping_score=float(feats_at_entry[COL_STOPPING]),
            wave_direction=float(feats_at_entry[COL_WAVE_DIR]),
            cib_active=cib,
            phase_accum=float(feats_at_entry[COL_PHASE_ACCUM]),
            phase_markup=float(feats_at_entry[COL_PHASE_MARKUP]),
            phase_distrib=float(feats_at_entry[COL_PHASE_DISTRIB]),
            phase_markdown=float(feats_at_entry[COL_PHASE_MARKDOWN]),
        ))
        realized_pnl += pnl
        pos_side = 0.0
        pos_size = 0.0
        entry_price = 0.0
        bars_in_trade = 0
        mfe = 0.0
        mae = 0.0
        unrealized_pnl = 0.0

    start_bar = sl  # need seq_len bars of history
    for bar in range(start_bar, n_bars - 1):
        # Build observation
        window = padded[bar: bar + sl]  # (seq_len, 41)
        window_flat = window.reshape(1, sl * len(fi))

        curr_price = close_ary[bar]

        if pos_side != 0:
            entry_dist = ((curr_price - entry_price) / bar_range) * pos_side
        else:
            entry_dist = 0.0

        pos_feats = torch.tensor([[
            pos_side,
            pos_size / 2.0,
            entry_dist,
            unrealized_pnl / 500.0,
            min(max(realized_pnl / 1000.0, -1.0), 1.0),
            min(bars_in_trade / 50.0, 1.0),
            mfe / 500.0,
            mae / 500.0,
        ]], dtype=torch.float32, device=device)

        state = torch.cat([window_flat, pos_feats], dim=1)
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            action = actor(state).item()

        actions_taken.append(action)

        # Execute action
        buy_price = curr_price + slip
        sell_price = curr_price - slip
        is_flat = pos_side == 0

        if action == ACTION_ENTER_LONG and is_flat:
            pos_side = 1.0
            pos_size = 1.0
            entry_price = buy_price
            entry_bar = bar
            bars_in_trade = 0
            mfe = 0.0
            mae = 0.0
            realized_pnl -= commission
            bar_features_at_entry.append((bar, tech_ary[bar].copy()))

        elif action == ACTION_ENTER_SHORT and is_flat:
            pos_side = -1.0
            pos_size = 1.0
            entry_price = sell_price
            entry_bar = bar
            bars_in_trade = 0
            mfe = 0.0
            mae = 0.0
            realized_pnl -= commission
            bar_features_at_entry.append((bar, tech_ary[bar].copy()))

        elif action == ACTION_EXIT and not is_flat:
            exit_px = sell_price if pos_side > 0 else buy_price
            close_trade(exit_px, bar)

        elif action == ACTION_REDUCE and not is_flat:
            if pos_size > 1:
                pos_size -= 1.0
                realized_pnl -= commission
            else:
                exit_px = sell_price if pos_side > 0 else buy_price
                close_trade(exit_px, bar)

        elif action == ACTION_ADD and not is_flat and pos_size < 2:
            add_px = buy_price if pos_side > 0 else sell_price
            new_size = pos_size + 1.0
            entry_price = (entry_price * pos_size + add_px) / new_size
            pos_size = new_size
            realized_pnl -= commission

        # Mark-to-market
        if pos_side != 0:
            bars_in_trade += 1
            next_price = close_ary[bar + 1]
            unrealized_pnl = compute_pnl_at(next_price)
            mfe = max(mfe, unrealized_pnl)
            mae = min(mae, unrealized_pnl)

    # Close any remaining position
    if pos_side != 0:
        final_px = (close_ary[-1] - slip) if pos_side > 0 else (close_ary[-1] + slip)
        close_trade(final_px, n_bars - 1)

    return actions_taken, trades, bar_features_at_entry, tech_ary, n_bars


def print_report(actions_taken, trades, bar_features_at_entry, tech_ary, n_bars, event_threshold=0.3):
    """Print comprehensive Wyckoff learning analysis."""
    total_bars = len(actions_taken)
    action_counts = defaultdict(int)
    for a in actions_taken:
        action_counts[a] += 1

    # ─── Action Distribution ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ACTION DISTRIBUTION")
    print("=" * 70)
    for a in range(N_ACTIONS):
        c = action_counts.get(a, 0)
        pct = 100.0 * c / total_bars if total_bars > 0 else 0
        print(f"  {ACTION_NAMES[a]:>12s}: {c:6d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':>12s}: {total_bars:6d}")

    # ─── Trade-Level Stats ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRADE-LEVEL STATISTICS")
    print("=" * 70)
    n_trades = len(trades)
    if n_trades == 0:
        print("  No trades taken!")
        return

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    longs = [t for t in trades if t.side == "long"]
    shorts = [t for t in trades if t.side == "short"]

    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / n_trades
    win_rate = 100.0 * len(wins) / n_trades
    avg_bars = np.mean([t.bars_held for t in trades])
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses and sum(t.pnl for t in losses) != 0 else float('inf')

    print(f"  Total trades:     {n_trades}")
    print(f"  Long/Short:       {len(longs)} / {len(shorts)}")
    print(f"  Win rate:         {win_rate:.1f}%")
    print(f"  Total PnL:        ${total_pnl:,.2f}")
    print(f"  Avg PnL/trade:    ${avg_pnl:,.2f}")
    print(f"  Avg winner:       ${avg_win:,.2f}")
    print(f"  Avg loser:        ${avg_loss:,.2f}")
    print(f"  Profit factor:    {profit_factor:.2f}")
    print(f"  Avg bars held:    {avg_bars:.1f}")
    print(f"  Avg MFE:          ${np.mean([t.mfe for t in trades]):,.2f}")
    print(f"  Avg MAE:          ${np.mean([t.mae for t in trades]):,.2f}")

    # ─── Entry Signal Correlation (THE KEY TEST) ──────────────────────
    print("\n" + "=" * 70)
    print("ENTRY-SIGNAL CORRELATION (Wyckoff Alignment Test)")
    print("=" * 70)

    thr = event_threshold

    # Count Wyckoff signals in the dataset (baseline)
    spring_bars = np.sum(tech_ary[:, COL_SPRING] > thr)
    upthrust_bars = np.sum(tech_ary[:, COL_UPTHRUST] > thr)
    absorption_bars = np.sum(tech_ary[:, COL_ABSORPTION] > thr)
    stopping_bars = np.sum(tech_ary[:, COL_STOPPING] > thr)

    vol_ratio = tech_ary[:, COL_WAVE_VOL_VS_PREV]
    lw = tech_ary[:, COL_LARGE_WAVE]
    cib_bars = np.sum((vol_ratio > 0.3) & (vol_ratio < 0.85) & (lw < 0.5))

    print(f"\n  Dataset signal frequency (threshold={thr}):")
    print(f"    Spring bars:     {spring_bars:4d} / {n_bars} ({100*spring_bars/n_bars:.1f}%)")
    print(f"    Upthrust bars:   {upthrust_bars:4d} / {n_bars} ({100*upthrust_bars/n_bars:.1f}%)")
    print(f"    Absorption bars: {absorption_bars:4d} / {n_bars} ({100*absorption_bars/n_bars:.1f}%)")
    print(f"    Stopping bars:   {stopping_bars:4d} / {n_bars} ({100*stopping_bars/n_bars:.1f}%)")
    print(f"    CiB bars:        {cib_bars:4d} / {n_bars} ({100*cib_bars/n_bars:.1f}%)")

    # Agent's entries on signal bars
    entries_on_spring = sum(1 for t in trades if t.side == "long" and t.spring_score > thr)
    entries_on_upthrust = sum(1 for t in trades if t.side == "short" and t.upthrust_score > thr)
    entries_on_absorption = sum(1 for t in trades if t.absorption_score > thr)
    entries_on_stopping = sum(1 for t in trades if t.stopping_score > thr)
    entries_on_cib = sum(1 for t in trades if t.cib_active)

    # Any Wyckoff signal at entry
    entries_on_any_signal = sum(1 for t in trades if (
        (t.side == "long" and t.spring_score > thr) or
        (t.side == "short" and t.upthrust_score > thr) or
        t.absorption_score > thr or
        t.stopping_score > thr or
        t.cib_active
    ))

    print(f"\n  Agent entry alignment ({n_trades} total entries):")
    print(f"    LONG on spring:        {entries_on_spring:3d} / {len(longs)} longs ({100*entries_on_spring/max(len(longs),1):.1f}%)")
    print(f"    SHORT on upthrust:     {entries_on_upthrust:3d} / {len(shorts)} shorts ({100*entries_on_upthrust/max(len(shorts),1):.1f}%)")
    print(f"    Entry on absorption:   {entries_on_absorption:3d} / {n_trades} ({100*entries_on_absorption/n_trades:.1f}%)")
    print(f"    Entry on stopping:     {entries_on_stopping:3d} / {n_trades} ({100*entries_on_stopping/n_trades:.1f}%)")
    print(f"    Entry on CiB:          {entries_on_cib:3d} / {n_trades} ({100*entries_on_cib/n_trades:.1f}%)")
    print(f"    ANY Wyckoff signal:    {entries_on_any_signal:3d} / {n_trades} ({100*entries_on_any_signal/n_trades:.1f}%)")

    # Random baseline for comparison
    random_pct_any = 100 * (spring_bars + upthrust_bars + absorption_bars + stopping_bars + cib_bars) / n_bars
    print(f"\n  Random baseline (any signal): {random_pct_any:.1f}% of bars")
    signal_ratio = (100 * entries_on_any_signal / n_trades) / max(random_pct_any, 0.1)
    print(f"  Agent/Random ratio:           {signal_ratio:.2f}x")
    if signal_ratio > 2.0:
        print("  >>> STRONG Wyckoff alignment — agent learned to time entries on signals")
    elif signal_ratio > 1.3:
        print("  >>> MODERATE Wyckoff alignment — agent somewhat prefers signal bars")
    else:
        print("  >>> WEAK/NO Wyckoff alignment — entries don't correlate with signals")

    # ─── Wave Direction Alignment ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("WAVE DIRECTION ALIGNMENT")
    print("=" * 70)
    longs_with_wave = [t for t in trades if t.side == "long"]
    shorts_with_wave = [t for t in trades if t.side == "short"]

    long_aligned = sum(1 for t in longs_with_wave if t.wave_direction > 0)  # with-trend
    long_counter = sum(1 for t in longs_with_wave if t.wave_direction < 0)  # counter-trend (CiB)
    short_aligned = sum(1 for t in shorts_with_wave if t.wave_direction < 0)
    short_counter = sum(1 for t in shorts_with_wave if t.wave_direction > 0)

    print(f"  LONG entries:  {long_aligned} with-trend, {long_counter} counter-trend (CiB), {len(longs_with_wave)-long_aligned-long_counter} neutral")
    print(f"  SHORT entries: {short_aligned} with-trend, {short_counter} counter-trend (CiB), {len(shorts_with_wave)-short_aligned-short_counter} neutral")

    # ─── Phase Alignment ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE ALIGNMENT")
    print("=" * 70)

    def dominant_phase(t):
        scores = {"accum": t.phase_accum, "markup": t.phase_markup,
                   "distrib": t.phase_distrib, "markdown": t.phase_markdown}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0.3 else "unclear"

    phase_entry_matrix = defaultdict(lambda: defaultdict(int))
    for t in trades:
        phase = dominant_phase(t)
        phase_entry_matrix[phase][t.side] += 1

    print(f"  {'Phase':<12s} {'Long':>6s} {'Short':>6s} {'Aligned?':>10s}")
    print(f"  {'-'*38}")
    for phase in ["accum", "markup", "distrib", "markdown", "unclear"]:
        l = phase_entry_matrix[phase]["long"]
        s = phase_entry_matrix[phase]["short"]
        if phase in ("accum", "markup"):
            aligned = "YES" if l > s else ("neutral" if l == s else "NO")
        elif phase in ("distrib", "markdown"):
            aligned = "YES" if s > l else ("neutral" if l == s else "NO")
        else:
            aligned = "-"
        print(f"  {phase:<12s} {l:>6d} {s:>6d} {aligned:>10s}")

    # ─── Signal-conditioned PnL ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("SIGNAL-CONDITIONED PnL")
    print("=" * 70)

    signal_trades = [t for t in trades if (
        (t.side == "long" and t.spring_score > thr) or
        (t.side == "short" and t.upthrust_score > thr) or
        t.absorption_score > thr or t.stopping_score > thr or t.cib_active
    )]
    no_signal_trades = [t for t in trades if t not in signal_trades]

    if signal_trades:
        sig_pnl = np.mean([t.pnl for t in signal_trades])
        sig_wr = 100 * sum(1 for t in signal_trades if t.pnl > 0) / len(signal_trades)
        print(f"  With Wyckoff signal:    {len(signal_trades):3d} trades | avg ${sig_pnl:,.2f} | WR {sig_wr:.1f}%")
    if no_signal_trades:
        no_pnl = np.mean([t.pnl for t in no_signal_trades])
        no_wr = 100 * sum(1 for t in no_signal_trades if t.pnl > 0) / len(no_signal_trades)
        print(f"  Without Wyckoff signal: {len(no_signal_trades):3d} trades | avg ${no_pnl:,.2f} | WR {no_wr:.1f}%")
    if signal_trades and no_signal_trades:
        sig_pnl = np.mean([t.pnl for t in signal_trades])
        no_pnl = np.mean([t.pnl for t in no_signal_trades])
        if sig_pnl > no_pnl:
            print("  >>> Signal trades outperform — agent benefits from Wyckoff timing")
        else:
            print("  >>> Non-signal trades outperform — Wyckoff signals not driving edge")

    # ─── Selectivity Score ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SELECTIVITY SCORE")
    print("=" * 70)
    entry_rate = 100 * n_trades / total_bars
    print(f"  Entry rate:     {entry_rate:.2f}% of bars (1 entry per {total_bars/max(n_trades,1):.0f} bars)")
    if entry_rate < 2:
        print("  >>> Highly selective — agent waits for specific setups")
    elif entry_rate < 10:
        print("  >>> Moderately selective")
    else:
        print("  >>> Frequent trader — may not be waiting for Wyckoff setups")


def save_trade_csv(trades: list[Trade], path: str):
    if not trades:
        return
    fieldnames = [
        "entry_bar", "exit_bar", "side", "entry_price", "exit_price",
        "pnl", "bars_held", "mfe", "mae",
        "spring_score", "upthrust_score", "absorption_score", "stopping_score",
        "wave_direction", "cib_active",
        "phase_accum", "phase_markup", "phase_distrib", "phase_markdown",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "entry_bar": t.entry_bar,
                "exit_bar": t.exit_bar,
                "side": t.side,
                "entry_price": f"{t.entry_price:.2f}",
                "exit_price": f"{t.exit_price:.2f}",
                "pnl": f"{t.pnl:.2f}",
                "bars_held": t.bars_held,
                "mfe": f"{t.mfe:.2f}",
                "mae": f"{t.mae:.2f}",
                "spring_score": f"{t.spring_score:.3f}",
                "upthrust_score": f"{t.upthrust_score:.3f}",
                "absorption_score": f"{t.absorption_score:.3f}",
                "stopping_score": f"{t.stopping_score:.3f}",
                "wave_direction": f"{t.wave_direction:.1f}",
                "cib_active": t.cib_active,
                "phase_accum": f"{t.phase_accum:.3f}",
                "phase_markup": f"{t.phase_markup:.3f}",
                "phase_distrib": f"{t.phase_distrib:.3f}",
                "phase_markdown": f"{t.phase_markdown:.3f}",
            })
    print(f"\n  Trade details saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wyckoff learning in transformer RL agent")
    parser.add_argument("--checkpoint", required=True, help="Path to RL checkpoint (.pt)")
    parser.add_argument("--npz", required=True, help="Path to feature NPZ")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.38)
    parser.add_argument("--d-ff", type=int, default=None, help="Feed-forward dim (default: from checkpoint or d_model*4)")
    parser.add_argument("--bar-range", type=float, default=100.0)
    parser.add_argument("--tick-value", type=float, default=20.0)
    parser.add_argument("--tick-size", type=float, default=1.0)
    parser.add_argument("--commission", type=float, default=1.00)
    parser.add_argument("--slippage", type=float, default=1.0, help="Slippage in ticks")
    parser.add_argument("--event-threshold", type=float, default=0.3)
    parser.add_argument("--gpu-id", type=int, default=-1, help="GPU id (-1 for CPU)")
    parser.add_argument("--output-csv", type=str, default="", help="Path for trade detail CSV")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu_id}" if (torch.cuda.is_available() and args.gpu_id >= 0) else "cpu"
    )

    # Try to extract config from checkpoint first
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt_peek = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved_config = ckpt_peek.get('config') if isinstance(ckpt_peek, dict) else None
    del ckpt_peek

    if saved_config is not None and isinstance(saved_config, TransformerConfig):
        config = saved_config
        print(f"  Using config from checkpoint: d_model={config.d_model}, d_ff={config.d_ff}, "
              f"n_layers={config.n_layers}, n_heads={config.n_heads}")
    else:
        d_ff = args.d_ff if args.d_ff is not None else args.d_model * 4
        config = TransformerConfig(
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            d_ff=d_ff,
        )

    actor = load_checkpoint(args.checkpoint, config, device)
    param_count = sum(p.numel() for p in actor.parameters())
    print(f"Actor params: {param_count:,}")

    print(f"Running deterministic eval on: {args.npz}")
    actions, trades, entry_feats, tech_ary, n_bars = run_deterministic_eval(
        actor=actor,
        npz_path=args.npz,
        config=config,
        commission=args.commission,
        tick_size=args.tick_size,
        tick_value=args.tick_value,
        bar_range=args.bar_range,
        slippage_ticks=args.slippage,
        device=device,
    )

    print_report(actions, trades, entry_feats, tech_ary, n_bars, args.event_threshold)

    csv_path = args.output_csv or os.path.splitext(args.checkpoint)[0] + "_trades.csv"
    save_trade_csv(trades, csv_path)


if __name__ == "__main__":
    main()
