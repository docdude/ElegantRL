#!/usr/bin/env python3
"""
Unified OOS replay runner for Wyckoff RL checkpoints.

Replaces: run_continuous_replay.py, run_filtered_replay.py,
          run_split8_replay.py, run_studio2_continuous_replay.py,
          run_studio2_replay.py

Major performance improvement: builds range bars + features ONCE from the
SCID file, then replays each checkpoint against precomputed data.  This
avoids re-parsing ~100M ticks and re-computing features per checkpoint.

Usage examples:
    # Single checkpoint
    python -m wyckoff_rl.tools.run_replay \\
        --checkpoints path/to/actor.pt

    # Multiple checkpoints (continuous sizing)
    python -m wyckoff_rl.tools.run_replay --continuous \\
        --checkpoints split3.pt split4.pt

    # With veto rules
    python -m wyckoff_rl.tools.run_replay --continuous \\
        --checkpoints split3.pt \\
        --veto "delta_ratio < -0.044, cvd_slope_fast > 0.037"

    # With named veto preset (from trader.py VETO_PRESETS)
    python -m wyckoff_rl.tools.run_replay --continuous \\
        --checkpoints split3.pt \\
        --veto-preset split3

    # Compare with and without veto
    python -m wyckoff_rl.tools.run_replay --continuous --compare-veto \\
        --checkpoints split3.pt split4.pt \\
        --veto-preset auto
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(message)s')
for name in ['wyckoff_effort.pipeline.wyckoff_features',
             'wyckoff_effort.pipeline', 'wyckoff_effort',
             'wyckoff_trader', 'wyckoff_rl.live.adapters',
             'wyckoff_rl', 'wyckoff_features']:
    logging.getLogger(name).setLevel(logging.ERROR)

sys.path.insert(0, '/opt/ElegantRL')

from wyckoff_rl.live.trader import VETO_PRESETS


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

SCID_PATH = '/opt/SierraChart/Data/NQH26-CME.scid'
START_DATE = '2026-01-15'
END_DATE = '2026-03-18'
INITIAL_CAPITAL = 250_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Veto rule parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_veto_string(veto_str: str) -> list[tuple[str, str, float]]:
    """Parse 'feature < thresh, feature > thresh' into rule tuples."""
    rules = []
    for part in veto_str.split(','):
        part = part.strip()
        if not part:
            continue
        for op in ('<', '>'):
            if op in part:
                feat, val = part.split(op, 1)
                rules.append((feat.strip(), op, float(val.strip())))
                break
    return rules


def get_veto_for_checkpoint(ckpt_path: str, preset_name: str) -> list | None:
    """Resolve veto rules from preset name.

    If preset_name == 'auto', tries to match checkpoint filename to a
    known preset key.
    """
    if preset_name == 'auto':
        basename = os.path.basename(ckpt_path).lower()
        for key in VETO_PRESETS:
            if key in basename:
                return VETO_PRESETS[key]
        return None
    return VETO_PRESETS.get(preset_name)


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], title: str):
    """Print a formatted comparison table."""
    print(f"\n\n{'='*110}", flush=True)
    print(title, flush=True)
    print(f"{'='*110}", flush=True)
    hdr = (f"{'Label':<24} {'Bars':>6} {'Trades':>7} {'RT':>5} {'Vetoed':>7} "
           f"{'PnL $':>12} {'PnL pts':>9} {'cumR':>7} {'WR%':>6} "
           f"{'MaxDD $':>12} {'Equity':>12} {'Time':>6}")
    print(hdr, flush=True)
    print("-" * 110, flush=True)
    for r in results:
        print(
            f"{r['label']:<24} {r['bars']:>6} {r['trades']:>7} "
            f"{r['round_trips']:>5} {r['vetoed']:>7} "
            f"{r['total_pnl_usd']:>+12,.0f} {r['pnl_pts']:>+9.1f} "
            f"{r['cumR']:>+7.1f} {r['win_rate']:>5.1f}% "
            f"{r['max_dd_usd']:>+12,.0f} {r['equity']:>12,.0f} "
            f"{r['elapsed_s']:>5.0f}s",
            flush=True)
    print("=" * 110, flush=True)


def print_veto_deltas(results: list[dict]):
    """Print improvement deltas for base/veto pairs."""
    # Find pairs by convention: 'label' vs 'label+veto'
    bases = {r['label']: r for r in results if '+veto' not in r['label']}
    for r in results:
        if '+veto' not in r['label']:
            continue
        base_label = r['label'].replace('+veto', '')
        rb = bases.get(base_label)
        if rb is None:
            continue
        delta_pnl = r['total_pnl_usd'] - rb['total_pnl_usd']
        delta_wr = r['win_rate'] - rb['win_rate']
        base_pnl = rb['total_pnl_usd']
        pct = (r['total_pnl_usd'] / base_pnl * 100) if base_pnl != 0 else 0
        print(
            f"  {r['label']}: PnL Δ={delta_pnl:+,.0f} "
            f"({pct:.0f}% of base), WR Δ={delta_wr:+.1f}pp, "
            f"vetoed {r['vetoed']} entries",
            flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Unified OOS replay runner for Wyckoff RL checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--checkpoints', '-c', nargs='+', required=True,
                   help='Checkpoint .pt file paths')
    p.add_argument('--labels', '-l', nargs='+', default=None,
                   help='Labels for each checkpoint (default: auto from filename)')
    p.add_argument('--continuous', action='store_true',
                   help='Use continuous position sizing (default: binary)')
    p.add_argument('--scid', default=SCID_PATH,
                   help=f'SCID file path (default: {SCID_PATH})')
    p.add_argument('--start-date', default=START_DATE)
    p.add_argument('--end-date', default=END_DATE)
    p.add_argument('--capital', type=float, default=INITIAL_CAPITAL)
    p.add_argument('--veto', type=str, default=None,
                   help='Inline veto rules: "feat < thresh, feat > thresh"')
    p.add_argument('--veto-preset', type=str, default=None,
                   help='Named veto preset from VETO_PRESETS (or "auto")')
    p.add_argument('--compare-veto', action='store_true',
                   help='Run each checkpoint twice: without and with veto')
    p.add_argument('--export-logs', action='store_true',
                   help='Write per-trade CSV logs (same format as old WyckoffTrader)')
    p.add_argument('--log-dir', default='wyckoff_rl/live_logs/replay')
    return p


def auto_label(ckpt_path: str) -> str:
    """Generate a short label from checkpoint path."""
    name = os.path.basename(ckpt_path)
    # Strip common prefixes/suffixes
    name = name.replace('actor__', '').replace('.pt', '').replace('.pth', '')
    # Try to extract split info from parent directory
    parent = os.path.basename(os.path.dirname(ckpt_path))
    if 'split' in name.lower():
        return name[:20]
    # Use parent dir + filename tail
    return f"{parent[-12:]}_{name[-12:]}" if len(name) > 12 else name


def main():
    args = build_parser().parse_args()

    # Validate checkpoints exist
    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"ERROR: Checkpoint not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Generate labels
    if args.labels:
        if len(args.labels) != len(args.checkpoints):
            print("ERROR: --labels count must match --checkpoints count",
                  file=sys.stderr)
            sys.exit(1)
        labels = args.labels
    else:
        labels = [auto_label(p) for p in args.checkpoints]

    # Parse veto rules
    inline_veto = parse_veto_string(args.veto) if args.veto else None

    sizing_str = "continuous" if args.continuous else "binary"
    print(f"\n{'='*60}", flush=True)
    print(f"WYCKOFF OOS REPLAY — {sizing_str} sizing", flush=True)
    print(f"SCID: {os.path.basename(args.scid)}", flush=True)
    print(f"Period: {args.start_date} → {args.end_date}", flush=True)
    print(f"Checkpoints: {len(args.checkpoints)}", flush=True)
    if args.compare_veto:
        print(f"Mode: compare with/without veto", flush=True)
    print(f"{'='*60}", flush=True)

    # Precompute bars + features ONCE (the expensive part)
    from wyckoff_rl.live.precompute import PrecomputedReplay
    print(f"\nPrecomputing bars + features...", flush=True)
    replay = PrecomputedReplay.from_scid(
        args.scid, args.start_date, args.end_date,
    )
    print(f"  → {replay.n_bars} bars, {replay.n_windows} windows\n", flush=True)

    results = []

    # Resolve veto rules once (apply to primary run unless --compare-veto)
    primary_veto = None
    if not args.compare_veto:
        if inline_veto:
            primary_veto = inline_veto
        elif args.veto_preset:
            primary_veto = VETO_PRESETS.get(args.veto_preset)

    for label, ckpt_path in zip(labels, args.checkpoints):
        # Resolve per-checkpoint veto for primary run
        ckpt_veto = primary_veto
        if ckpt_veto is None and not args.compare_veto and args.veto_preset == 'auto':
            ckpt_veto = get_veto_for_checkpoint(ckpt_path, 'auto')

        # --- Base run (with veto applied directly if not in compare mode) ---
        print(f"  RUNNING: {label}...", end='', flush=True)
        log_path = None
        if args.export_logs:
            log_path = os.path.join(
                args.log_dir, label,
                f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
            )
        r = replay.run_checkpoint(
            ckpt_path,
            continuous=args.continuous,
            initial_capital=args.capital,
            veto_rules=ckpt_veto,
            log_path=log_path,
        )
        r['label'] = label
        results.append(r)
        pnl_str = (f"  trades={r['trades']} RT={r['round_trips']} "
                   f"PnL=${r['total_pnl_usd']:+,.0f} WR={r['win_rate']:.0f}% "
                   f"MaxDD=${r['max_dd_usd']:+,.0f} ({r['elapsed_s']:.1f}s)")
        if r.get('vetoed', 0) > 0:
            pnl_str += f" vetoed={r['vetoed']}"
        if r.get('log_path'):
            pnl_str += f"\n    → log: {r['log_path']}"
        print(pnl_str, flush=True)

        # --- Veto comparison run (only with --compare-veto) ---
        if args.compare_veto:
            veto_rules = inline_veto
            if veto_rules is None and args.veto_preset:
                veto_rules = get_veto_for_checkpoint(ckpt_path, args.veto_preset)
            if veto_rules is None:
                for key in VETO_PRESETS:
                    if key in label.lower():
                        veto_rules = VETO_PRESETS[key]
                        break

            if veto_rules:
                veto_label = f"{label}+veto"
                print(f"  RUNNING: {veto_label}...", end='', flush=True)
                veto_log_path = None
                if args.export_logs:
                    veto_log_path = os.path.join(
                        args.log_dir, veto_label,
                        f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                    )
                rv = replay.run_checkpoint(
                    ckpt_path,
                    continuous=args.continuous,
                    initial_capital=args.capital,
                    veto_rules=veto_rules,
                    log_path=veto_log_path,
                )
                rv['label'] = veto_label
                results.append(rv)
                print(f"  trades={rv['trades']} RT={rv['round_trips']} "
                      f"vetoed={rv['vetoed']} "
                      f"PnL=${rv['total_pnl_usd']:+,.0f} "
                      f"WR={rv['win_rate']:.0f}% "
                      f"MaxDD=${rv['max_dd_usd']:+,.0f} ({rv['elapsed_s']:.1f}s)",
                      flush=True)

    # Summary
    title = (f"OOS Replay — {sizing_str} sizing "
             f"({args.start_date} → {args.end_date})")
    print_summary(results, title)

    if args.compare_veto or inline_veto or args.veto_preset:
        print_veto_deltas(results)


if __name__ == '__main__':
    main()
