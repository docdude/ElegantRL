#!/usr/bin/env python3
"""
Validate PrecomputedReplay against known trade logs.

Compares metrics from the precompute path against actual trade log CSVs
produced by the old WyckoffTrader replay (tick-by-tick, adapters.py path)
to ensure the two pipelines produce equivalent results.

Usage:
    # Validate a single checkpoint against its trade log
    python -m wyckoff_rl.tools.validate_replay \
        --checkpoint path/to/actor.pt \
        --trade-log path/to/trades_*.csv

    # Validate with tolerance thresholds
    python -m wyckoff_rl.tools.validate_replay \
        --checkpoint path/to/actor.pt \
        --trade-log path/to/trades_*.csv \
        --pnl-tol 0.10 --bar-tol 0.02

    # Run built-in regression suite (known good logs)
    python -m wyckoff_rl.tools.validate_replay --suite
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, '/opt/ElegantRL')

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

SCID_PATH = '/opt/SierraChart/Data/NQH26-CME.scid'
START_DATE = '2026-01-15'
END_DATE = '2026-03-18'

# Known-good checkpoints + trade logs for regression testing
REGRESSION_SUITE = [
    {
        'label': 'studio2c_split4',
        'checkpoint': '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous/split4_actor__000000874496_00444.927.pt',
        'trade_log': '/opt/ElegantRL/wyckoff_rl/live_logs/studio2c_s2c_split4_s2.37/trades_20260321_001132.csv',
        'continuous': True,
    },
    {
        'label': 'filtered_split4+veto',
        'checkpoint': '/opt/ElegantRL/wyckoff_rl/live/checkpoints/studio2_continuous/split4_actor__000000874496_00444.927.pt',
        'trade_log': '/opt/ElegantRL/wyckoff_rl/live_logs/filtered_split4+veto/trades_20260321_155538.csv',
        'continuous': True,
        'veto_preset': 'split4',
        'pnl_tol': 0.15,   # veto compounds bar-boundary diffs → wider tolerance
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction from trade log
# ─────────────────────────────────────────────────────────────────────────────

def metrics_from_trade_log(path: str) -> dict:
    """Extract summary metrics from a trade log CSV."""
    df = pd.read_csv(path)
    n_bars = int(df['bar_num'].max()) if 'bar_num' in df.columns else 0
    n_trades = len(df)

    realized = df['pnl_realized_usd'].astype(float)
    closes = realized[realized != 0]
    winners = int((closes > 0).sum())
    losers = int((closes < 0).sum())
    n_round_trips = winners + losers
    win_rate = winners / max(1, n_round_trips) * 100

    total_pnl = float(df['total_pnl_usd'].iloc[-1]) if len(df) > 0 else 0.0
    equity = df['equity'].astype(float)
    peak = equity.cummax()
    max_dd = float((equity - peak).min())

    return {
        'bars': n_bars,
        'trades': n_trades,
        'round_trips': n_round_trips,
        'win_rate': win_rate,
        'total_pnl_usd': total_pnl,
        'max_dd_usd': max_dd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_metrics(actual: dict, replay: dict, pnl_tol: float, bar_tol: float) -> List[dict]:
    """Compare actual vs replay metrics, return list of check results."""
    checks = []

    def check(name, actual_val, replay_val, tolerance, unit=''):
        if actual_val == 0 and replay_val == 0:
            pct_diff = 0.0
        elif actual_val == 0:
            pct_diff = float('inf')
        else:
            pct_diff = abs(replay_val - actual_val) / abs(actual_val)
        passed = pct_diff <= tolerance
        checks.append({
            'name': name,
            'actual': actual_val,
            'replay': replay_val,
            'diff_pct': pct_diff * 100,
            'tolerance_pct': tolerance * 100,
            'passed': passed,
            'unit': unit,
        })

    check('Bars',        actual['bars'],          replay['bars'],          bar_tol)
    check('Trades',      actual['trades'],        replay['trades'],        0.15)
    check('Round-trips', actual['round_trips'],   replay['round_trips'],   0.15)
    check('Win Rate',    actual['win_rate'],       replay['win_rate'],      0.10, unit='%')
    check('PnL $',       actual['total_pnl_usd'], replay['total_pnl_usd'], pnl_tol, unit='$')
    check('Max DD $',    actual['max_dd_usd'],     replay['max_dd_usd'],    0.30, unit='$')

    return checks


def print_comparison(label: str, actual: dict, replay: dict, checks: List[dict]):
    """Print formatted comparison table."""
    all_pass = all(c['passed'] for c in checks)
    status = 'PASS' if all_pass else 'FAIL'

    print(f"\n{'='*72}")
    print(f"  {label}  [{status}]")
    print(f"{'='*72}")
    print(f"  {'Metric':<14} {'Actual':>14} {'Replay':>14} {'Diff':>8} {'Tol':>8}  {'':>4}")
    print(f"  {'-'*62}")

    for c in checks:
        sym = 'OK' if c['passed'] else 'FAIL'
        actual_str = f"{c['actual']:>12,.1f}{c['unit']}"
        replay_str = f"{c['replay']:>12,.1f}{c['unit']}"
        print(f"  {c['name']:<14} {actual_str:>14} {replay_str:>14} "
              f"{c['diff_pct']:>6.1f}% {c['tolerance_pct']:>6.1f}%  [{sym}]")

    print(f"{'='*72}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Validation runners
# ─────────────────────────────────────────────────────────────────────────────

def validate_single(
    checkpoint: str,
    trade_log: str,
    *,
    continuous: bool = True,
    veto_rules: List | None = None,
    scid_path: str = SCID_PATH,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    pnl_tol: float = 0.10,
    bar_tol: float = 0.02,
    replay=None,
) -> bool:
    """Validate one checkpoint against its trade log. Returns True if all checks pass."""
    from wyckoff_rl.live.precompute import PrecomputedReplay

    # Extract actual metrics
    actual = metrics_from_trade_log(trade_log)

    # Run precompute replay (reuse if provided)
    if replay is None:
        print("Precomputing bars + features...", flush=True)
        replay = PrecomputedReplay.from_scid(scid_path, start_date, end_date)
        print(f"  -> {replay.n_bars} bars, {replay.n_windows} windows", flush=True)

    result = replay.run_checkpoint(
        checkpoint, continuous=continuous, veto_rules=veto_rules,
    )

    label = os.path.basename(os.path.dirname(trade_log))
    checks = compare_metrics(actual, result, pnl_tol, bar_tol)
    return print_comparison(label, actual, result, checks)


def run_suite(
    *,
    scid_path: str = SCID_PATH,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    pnl_tol: float = 0.10,
    bar_tol: float = 0.02,
) -> bool:
    """Run built-in regression suite. Returns True if all pass."""
    from wyckoff_rl.live.precompute import PrecomputedReplay

    # Filter to entries where files exist
    suite = [e for e in REGRESSION_SUITE
             if os.path.exists(e['checkpoint']) and os.path.exists(e['trade_log'])]

    if not suite:
        print("ERROR: No regression entries found (missing checkpoint/log files)")
        return False

    print(f"\nRunning regression suite: {len(suite)} entries")
    print(f"Tolerances: PnL={pnl_tol*100:.0f}%, Bars={bar_tol*100:.0f}%\n")

    # Precompute once
    print("Precomputing bars + features...", flush=True)
    replay = PrecomputedReplay.from_scid(scid_path, start_date, end_date)
    print(f"  -> {replay.n_bars} bars, {replay.n_windows} windows\n", flush=True)

    all_pass = True
    for entry in suite:
        # Resolve veto rules from preset if specified
        veto_rules = None
        if entry.get('veto_preset'):
            from wyckoff_rl.live.trader import VETO_PRESETS
            veto_rules = VETO_PRESETS.get(entry['veto_preset'])

        passed = validate_single(
            entry['checkpoint'],
            entry['trade_log'],
            continuous=entry.get('continuous', True),
            veto_rules=veto_rules,
            scid_path=scid_path,
            start_date=start_date,
            end_date=end_date,
            pnl_tol=entry.get('pnl_tol', pnl_tol),
            bar_tol=entry.get('bar_tol', bar_tol),
            replay=replay,
        )
        if not passed:
            all_pass = False

    print(f"\n{'='*72}")
    status = 'ALL PASS' if all_pass else 'SOME FAILED'
    print(f"  Suite result: {status} ({len(suite)} entries)")
    print(f"{'='*72}\n")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Validate PrecomputedReplay against known trade logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--suite', action='store_true',
                   help='Run built-in regression suite')
    p.add_argument('--checkpoint', '-c', type=str, default=None,
                   help='Checkpoint .pt file')
    p.add_argument('--trade-log', '-t', type=str, default=None,
                   help='Trade log CSV from the reference pipeline')
    p.add_argument('--continuous', action='store_true', default=True,
                   help='Use continuous sizing (default: True)')
    p.add_argument('--binary', action='store_true',
                   help='Use binary sizing')
    p.add_argument('--scid', default=SCID_PATH)
    p.add_argument('--start-date', default=START_DATE)
    p.add_argument('--end-date', default=END_DATE)
    p.add_argument('--pnl-tol', type=float, default=0.10,
                   help='PnL tolerance as fraction (default: 0.10 = 10%%)')
    p.add_argument('--bar-tol', type=float, default=0.02,
                   help='Bar count tolerance as fraction (default: 0.02 = 2%%)')
    return p


def main():
    args = build_parser().parse_args()

    if args.suite:
        success = run_suite(
            scid_path=args.scid,
            start_date=args.start_date,
            end_date=args.end_date,
            pnl_tol=args.pnl_tol,
            bar_tol=args.bar_tol,
        )
        sys.exit(0 if success else 1)

    if not args.checkpoint or not args.trade_log:
        print("ERROR: --checkpoint and --trade-log required (or use --suite)")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.trade_log):
        print(f"ERROR: Trade log not found: {args.trade_log}", file=sys.stderr)
        sys.exit(1)

    continuous = not args.binary
    success = validate_single(
        args.checkpoint,
        args.trade_log,
        continuous=continuous,
        scid_path=args.scid,
        start_date=args.start_date,
        end_date=args.end_date,
        pnl_tol=args.pnl_tol,
        bar_tol=args.bar_tol,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
