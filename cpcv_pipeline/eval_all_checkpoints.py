#!/usr/bin/env python3
"""
Evaluate ALL checkpoints in each CPCV split directory and compare their
OOS performance against DJIA baseline.

Best avgR during training often != best OOS Sharpe/return.

This script evaluates every actor__*.pt (plus act.pth) on the held-out test
set, computes Sharpe, max drawdown, final return, alpha vs DJIA, and generates
comparison plots per split plus aggregate summary.

Ported from examples/eval_all_checkpoints.py with CPCV split support.

Usage -- single split (while training is still running):
    python -m cpcv_pipeline.eval_all_checkpoints \
        --results-dir train_results/20260221_CPCV_PPO_N5K2_seed1943 \
        --split 0

Usage -- all splits (after full CPCV run):
    python -m cpcv_pipeline.eval_all_checkpoints \
        --results-dir train_results/20260221_CPCV_PPO_N5K2_seed1943

Options:
    --gpu 0             GPU device (default: 0)
    --force             Re-evaluate even if checkpoint_results.csv exists
    --top-k 5           Highlight top-k checkpoints in plots (default: 5)
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cpcv_pipeline.config import GPU_ID, DEFAULT_ERL_PARAMS, DEFAULT_ENV_PARAMS
from cpcv_pipeline.function_train_test import (
    load_full_data,
    load_dates_from_npz,
    save_sliced_data,
    get_agent_class,
    DATA_CACHE_DIR,
)
from elegantrl.envs.StockTradingEnv import StockTradingVecEnv

import torch as th


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dates() -> np.ndarray:
    """Load unique sorted trading dates.

    Prefers dates_ary embedded in the NPZ (approach 2).
    Falls back to parsing the cached Alpaca CSV.

    Returns array of date strings, index-aligned with close_ary rows.
    """
    # Try NPZ first (fast, no CSV dependency)
    dates = load_dates_from_npz()
    if dates is not None:
        return dates

    # Fallback: read CSV
    ALPACA_CSV_PATH = os.path.join(DATA_CACHE_DIR, "alpaca_processed.csv")
    if not os.path.exists(ALPACA_CSV_PATH):
        raise FileNotFoundError(
            f"No dates found in NPZ and cached CSV not found at {ALPACA_CSV_PATH}. "
            f"Run download_and_preprocess() or re-generate the NPZ with dates."
        )
    import pandas as _pd
    df = _pd.read_csv(ALPACA_CSV_PATH, usecols=['date'])
    dates = sorted(df['date'].unique())
    return np.array([str(d)[:10] for d in dates])


def fetch_djia_baseline(start_date: str, end_date: str, initial_amount: float = 1e6):
    """Fetch DJIA baseline via finrl.plot.get_baseline.

    Returns (baseline_values, baseline_return) or (None, None) on failure.
    baseline_values is normalised to initial_amount.
    """
    try:
        from finrl.plot import get_baseline
        baseline_df = get_baseline(ticker='^DJI', start=start_date, end=end_date)
        baseline_values = baseline_df['close'] / baseline_df['close'].iloc[0] * initial_amount
        baseline_return = (baseline_values.iloc[-1] / initial_amount - 1) * 100
        return baseline_values.values, baseline_return
    except Exception as e:
        print(f"    DJIA baseline not available: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Segment helpers for non-contiguous test sets
# ─────────────────────────────────────────────────────────────────────────────

def _split_contiguous_segments(indices: np.ndarray) -> list[np.ndarray]:
    """Split a sorted index array into contiguous segments.

    Non-contiguous test sets (K>=2 in CPCV) must be evaluated per-segment
    to avoid artificial price jumps at gap boundaries.

    Returns list of 1-D arrays, each a contiguous run of indices.
    """
    indices = np.sort(indices)
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0]
    if len(breaks) == 0:
        return [indices]
    segments = []
    start = 0
    for b in breaks:
        segments.append(indices[start:b + 1])
        start = b + 1
    segments.append(indices[start:])
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_checkpoints(split_dir: str) -> list[dict]:
    """Find all actor checkpoints in a split directory.

    Returns list of dicts with keys: path, filename, step, avgR (if in name).
    Sorted by step ascending.
    """
    pattern = re.compile(
        r'^actor__(\d+)(?:_(\d+\.\d+))?\.pt$'
    )
    checkpoints = []
    for f in os.listdir(split_dir):
        m = pattern.match(f)
        if m:
            step = int(m.group(1))
            avgR = float(m.group(2)) if m.group(2) else None
            checkpoints.append({
                'path': os.path.join(split_dir, f),
                'filename': f,
                'step': step,
                'avgR': avgR,
            })
    # Also include act.pth (latest) if present
    act_path = os.path.join(split_dir, 'act.pth')
    if os.path.exists(act_path):
        # Infer step from the highest-step actor__ file
        max_step = max((c['step'] for c in checkpoints), default=0)
        checkpoints.append({
            'path': act_path,
            'filename': 'act.pth',
            'step': max_step + 1,  # sort after last actor__
            'avgR': None,
        })

    checkpoints.sort(key=lambda c: c['step'])
    return checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# Single-checkpoint evaluation
# Evaluates each contiguous test segment independently, then chains daily
# returns.  This avoids artificial price jumps at non-contiguous boundaries
# (K >= 2 in CPCV).
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_single_checkpoint(
    checkpoint_path: str,
    segment_envs: list[dict],
    initial_amount: float,
    device: str,
    combined_baseline_values: np.ndarray | None = None,
    combined_baseline_return: float | None = None,
) -> dict | None:
    """Evaluate one checkpoint across one or more contiguous test segments.

    For non-contiguous test sets (K>=2), runs the actor through each segment
    independently, then chains daily returns for correct aggregate metrics.

    Parameters
    ----------
    segment_envs : list of dict
        Each dict has 'env' (StockTradingVecEnv) and 'num_days' (int).
    combined_baseline_values : optional
        DJIA baseline account values, already chained per-segment.
    combined_baseline_return : optional
        DJIA total return (%).
    """
    try:
        actor = th.load(checkpoint_path, map_location=device, weights_only=False)
        actor.eval()
    except Exception as e:
        print(f"    WARN: Failed to load {os.path.basename(checkpoint_path)}: {e}")
        return None

    all_daily_returns = []

    for seg in segment_envs:
        env = seg['env']
        num_days = seg['num_days']

        env.if_random_reset = False
        state, _ = env.reset()

        seg_values = [initial_amount]
        max_step = num_days - 1

        with th.no_grad():
            for t in range(max_step):
                action = actor(state.to(device) if hasattr(state, 'to') else state)
                state, reward, terminal, truncate, info = env.step(action)

                if hasattr(env, 'total_asset'):
                    if t < max_step - 1:
                        seg_values.append(env.total_asset[0].cpu().item())
                    else:
                        if hasattr(env, 'cumulative_returns') and env.cumulative_returns is not None:
                            cr = env.cumulative_returns
                            if isinstance(cr, list):
                                final_value = initial_amount * cr[0] / 100
                            elif hasattr(cr, 'cpu'):
                                final_value = initial_amount * cr[0].cpu().item() / 100
                            else:
                                final_value = initial_amount * float(cr) / 100
                            seg_values.append(final_value)
                        else:
                            seg_values.append(seg_values[-1])

        seg_values = np.array(seg_values)
        seg_returns = np.diff(seg_values) / (seg_values[:-1] + 1e-9)
        all_daily_returns.append(seg_returns)

    # Chain daily returns across segments
    daily_returns = np.concatenate(all_daily_returns)

    # Reconstruct combined account values by compounding
    account_values = initial_amount * np.concatenate(
        [[1.0], np.cumprod(1 + daily_returns)])

    final_return = (account_values[-1] / account_values[0]) - 1.0

    # Sharpe ratio (annualised)
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)) * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(account_values)
    drawdown = (account_values - peak) / peak
    max_drawdown = drawdown.min()

    # Sortino ratio (downside deviation only)
    neg_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 1e-9
    sortino = (np.mean(daily_returns) / (downside_std + 1e-9)) * np.sqrt(252)

    # Calmar ratio (annualised return / max drawdown)
    n_days = max(len(daily_returns), 1)
    ann_return = (1 + final_return) ** (252 / n_days) - 1
    calmar = ann_return / (abs(max_drawdown) + 1e-9)

    # Baseline comparison
    alpha = 0.0
    first_beat_day = -1
    days_beating = 0
    pct_beating = 0.0
    has_baseline = (combined_baseline_values is not None
                    and combined_baseline_return is not None)

    if has_baseline:
        alpha = final_return * 100 - combined_baseline_return

        strategy_pct = (account_values / account_values[0] - 1) * 100
        baseline_pct = (combined_baseline_values / combined_baseline_values[0] - 1) * 100

        # Both curves are constructed from the same segment structure so
        # lengths should match; handle minor mismatches gracefully.
        min_len = min(len(strategy_pct), len(baseline_pct))
        beating_mask = strategy_pct[:min_len] > baseline_pct[:min_len]
        if beating_mask.any():
            first_beat_day = int(np.argmax(beating_mask))
            days_beating = int(beating_mask.sum())
            pct_beating = days_beating / min_len * 100
        else:
            first_beat_day = -1
            days_beating = 0
            pct_beating = 0.0

    return {
        'final_return': float(final_return),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'calmar': float(calmar),
        'max_drawdown': float(max_drawdown),
        'ann_return': float(ann_return),
        'alpha': float(alpha),
        'first_beat_day': first_beat_day,
        'days_beating': days_beating,
        'pct_beating': float(pct_beating),
        'account_values': account_values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-split evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(
    split_dir: str,
    close_ary: np.ndarray,
    tech_ary: np.ndarray,
    all_dates: np.ndarray,
    gpu_id: int = 0,
    force: bool = False,
    top_k: int = 5,
) -> pd.DataFrame | None:
    """Evaluate all checkpoints in one split directory.

    For non-contiguous test sets (K>=2 in CPCV), evaluates each contiguous
    segment independently, chaining daily returns to avoid artificial price
    jumps at gap boundaries.
    """
    from elegantrl.envs.vec_normalize import VecNormalize

    csv_path = os.path.join(split_dir, 'checkpoint_results.csv')
    if os.path.exists(csv_path) and not force:
        print(f"  {os.path.basename(split_dir)}: checkpoint_results.csv exists, "
              f"skipping (use --force)")
        return pd.read_csv(csv_path)

    # Load metadata -- check cwd first, then parent dir (survives if_remove)
    split_name = os.path.basename(split_dir)
    meta_path = os.path.join(split_dir, 'split_meta.json')
    if not os.path.exists(meta_path):
        parent_meta = os.path.join(os.path.dirname(split_dir), f'{split_name}_meta.json')
        if os.path.exists(parent_meta):
            meta_path = parent_meta
        else:
            print(f"  {split_name}: no split_meta.json found (cwd or parent), skipping")
            return None

    with open(meta_path) as f:
        meta = json.load(f)

    test_indices = np.array(meta['test_indices'])
    model_name = meta.get('model_name', 'ppo')

    # Discover checkpoints
    checkpoints = discover_checkpoints(split_dir)
    if not checkpoints:
        print(f"  {split_name}: no checkpoints found, skipping")
        return None

    # ── Detect contiguous segments ────────────────────────────────────
    segments = _split_contiguous_segments(test_indices)
    is_multi_segment = len(segments) > 1

    print(f"\n  {split_name}: {len(checkpoints)} checkpoints, "
          f"test={len(test_indices)} days"
          + (f" ({len(segments)} segments)" if is_multi_segment else ""))

    initial_amount = DEFAULT_ENV_PARAMS.get('initial_amount', 1e6)
    gamma = DEFAULT_ERL_PARAMS.get('gamma', 0.99)
    device = f'cuda:{gpu_id}' if gpu_id >= 0 and th.cuda.is_available() else 'cpu'
    vec_norm_path = os.path.join(split_dir, 'vec_normalize.pt')
    has_vec_norm = os.path.exists(vec_norm_path)

    # ── Build per-segment envs + DJIA baselines ──────────────────────
    segment_envs = []       # list of {'env': ..., 'num_days': ...}
    all_bl_daily_rets = []  # per-segment DJIA daily returns
    baseline_ok = True
    seg_npz_paths = []

    for seg_idx, seg_indices in enumerate(segments):
        num_days = len(seg_indices)
        seg_start_date = all_dates[seg_indices[0]]
        seg_end_date = all_dates[seg_indices[-1]]

        if is_multi_segment:
            print(f"    Segment {seg_idx}: [{seg_indices[0]}..{seg_indices[-1]}] "
                  f"({seg_start_date} to {seg_end_date}, {num_days} days)")

        # Sliced NPZ for this segment
        seg_npz = os.path.join(DATA_CACHE_DIR, "cpcv_splits",
                               f"_eval_{split_name}_seg{seg_idx}.npz")
        save_sliced_data(seg_indices, seg_npz, close_ary, tech_ary, dates_ary=all_dates)
        seg_npz_paths.append(seg_npz)

        env = StockTradingVecEnv(
            npz_path=seg_npz,
            initial_amount=initial_amount,
            max_stock=DEFAULT_ENV_PARAMS.get('max_stock', 1e2),
            cost_pct=DEFAULT_ENV_PARAMS.get('cost_pct', 1e-3),
            gamma=gamma,
            beg_idx=0,
            end_idx=num_days,
            num_envs=1,
            gpu_id=gpu_id,
        )
        env.if_random_reset = False

        if has_vec_norm:
            env = VecNormalize(env, training=False, norm_reward=False)
            env.load(vec_norm_path, verbose=False)
            # load() restores norm_obs/norm_reward flags from saved state;
            # force training=False and norm_reward=False for eval
            # (we want raw rewards to compute accurate Sharpe/returns)
            env.training = False
            env.norm_reward = False

        segment_envs.append({'env': env, 'num_days': num_days})

        # DJIA baseline for this segment's date range
        bv, _ = fetch_djia_baseline(seg_start_date, seg_end_date, initial_amount)
        if bv is not None:
            # Resample DJIA to match segment day count
            bv_x = np.linspace(0, num_days - 1, len(bv))
            bv_resampled = np.interp(np.arange(num_days), bv_x, bv)
            seg_bl_rets = np.diff(bv_resampled) / (bv_resampled[:-1] + 1e-9)
            all_bl_daily_rets.append(seg_bl_rets)
        else:
            baseline_ok = False

    if has_vec_norm:
        # Report what normalization mode was used for training
        sample_env = segment_envs[0]['env'] if segment_envs else None
        norm_info = ""
        if sample_env and hasattr(sample_env, 'norm_obs'):
            norm_info = f" (norm_obs={sample_env.norm_obs}, norm_reward=False for eval)"
        print(f"    VecNormalize loaded{norm_info}"
              + (f" (applied to {len(segments)} segment envs)" if is_multi_segment else ""))

    # Construct combined DJIA baseline by chaining per-segment daily returns
    if baseline_ok and all_bl_daily_rets:
        combined_bl_rets = np.concatenate(all_bl_daily_rets)
        combined_baseline_values = initial_amount * np.concatenate(
            [[1.0], np.cumprod(1 + combined_bl_rets)])
        combined_baseline_return = (combined_baseline_values[-1] / initial_amount - 1) * 100
        has_baseline = True
        print(f"    DJIA baseline return: {combined_baseline_return:.2f}%"
              + (f" (chained across {len(segments)} segments)" if is_multi_segment else ""))
    else:
        combined_baseline_values = None
        combined_baseline_return = None
        has_baseline = False

    test_start_date = all_dates[test_indices[0]]
    test_end_date = all_dates[test_indices[-1]]
    print(f"    Test period: {test_start_date} to {test_end_date}")

    # Compute segment boundary x-positions for plotting
    seg_boundary_x = []
    seg_gap_labels = []
    if is_multi_segment:
        pos = len(segments[0])
        for i in range(1, len(segments)):
            seg_boundary_x.append(pos - 0.5)
            gap_start = all_dates[segments[i - 1][-1]]
            gap_end = all_dates[segments[i][0]]
            seg_gap_labels.append(f"{gap_start} \u2192 {gap_end}")
            pos += len(segments[i]) - 1

    # ── Evaluate each checkpoint ──────────────────────────────────────
    rows = []
    all_curves = {}
    for ckpt in checkpoints:
        result = evaluate_single_checkpoint(
            ckpt['path'], segment_envs, initial_amount, device,
            combined_baseline_values=combined_baseline_values,
            combined_baseline_return=combined_baseline_return,
        )
        if result is None:
            continue

        row = {
            'checkpoint': ckpt['filename'],
            'step': ckpt['step'],
            'train_avgR': ckpt['avgR'],
            'final_return': result['final_return'] * 100,
            'alpha': result['alpha'],
            'sharpe': result['sharpe'],
            'sortino': result['sortino'],
            'calmar': result['calmar'],
            'max_drawdown': result['max_drawdown'] * 100,
            'ann_return': result['ann_return'] * 100,
            'first_beat_day': result['first_beat_day'],
            'days_beating': result['days_beating'],
            'pct_beating': result['pct_beating'],
        }
        rows.append(row)
        all_curves[ckpt['filename']] = result['account_values']

        avgR_str = f"avgR={ckpt['avgR']:.1f}" if ckpt['avgR'] else "latest"
        beat_str = f"Beat%={result['pct_beating']:.0f}%" if has_baseline else ""
        print(f"    {ckpt['filename']:<45} "
              f"Return={result['final_return']*100:>+7.2f}%  "
              f"Alpha={result['alpha']:>+6.2f}%  "
              f"Sharpe={result['sharpe']:>6.2f}  "
              f"MaxDD={result['max_drawdown']*100:>6.1f}%  "
              f"{beat_str}  ({avgR_str})")

    # Clean up temp NPZ files
    for npz_path in seg_npz_paths:
        if os.path.exists(npz_path):
            os.remove(npz_path)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(['sharpe', 'final_return'], ascending=[False, False])

    # Print summary table
    print(f"\n    {'Checkpoint':<45} {'Return':>8} {'Alpha':>8} {'Sharpe':>7} "
          f"{'MaxDD':>8} {'Beat%':>7} {'FirstBeat':>10}")
    print(f"    {'-'*100}")
    for _, row in df.iterrows():
        print(f"    {row['checkpoint']:<45} {row['final_return']:>7.2f}% "
              f"{row['alpha']:>+7.2f}% {row['sharpe']:>7.2f} "
              f"{row['max_drawdown']:>7.2f}% {row['pct_beating']:>6.0f}% "
              f"{'Day '+str(int(row['first_beat_day'])) if row['first_beat_day'] >= 0 else 'Never':>10}")

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"\n    Saved: {csv_path}")

    # Plots
    _plot_split_comparison(
        df, all_curves, split_dir, initial_amount, top_k,
        baseline_values=combined_baseline_values,
        baseline_return=combined_baseline_return,
        segment_boundaries=seg_boundary_x if seg_boundary_x else None,
        segment_gap_labels=seg_gap_labels if seg_gap_labels else None,
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting -- mirrors examples/eval_all_checkpoints.py 4-panel layout
# ─────────────────────────────────────────────────────────────────────────────

def _plot_split_comparison(
    df: pd.DataFrame,
    all_curves: dict,
    split_dir: str,
    initial_amount: float,
    top_k: int,
    baseline_values: np.ndarray | None = None,
    baseline_return: float | None = None,
    segment_boundaries: list[float] | None = None,
    segment_gap_labels: list[str] | None = None,
):
    """Generate 4-panel comparison plots.

    When segment_boundaries is provided, draws vertical markers at the
    boundaries between non-contiguous test segments.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    split_name = os.path.basename(split_dir)
    has_baseline = baseline_values is not None and baseline_return is not None
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Panel 1: All curves + DJIA baseline ──────────────────────────
    ax1 = axes[0, 0]
    n = len(all_curves)
    colors = plt.cm.viridis(np.linspace(0, 1, max(n, 1)))
    for i, (ckpt, values) in enumerate(all_curves.items()):
        returns = (values / values[0] - 1) * 100
        label = ckpt.replace('actor__', '').replace('.pt', '').replace('.pth', '')
        ax1.plot(returns, label=label, alpha=0.7, linewidth=1.5, color=colors[i])

    if has_baseline:
        baseline_returns = (baseline_values / baseline_values[0] - 1) * 100
        ax1.plot(baseline_returns, 'k--', linewidth=2,
                 label=f'DJIA ({baseline_return:.1f}%)')

    # Draw segment boundaries (non-contiguous test sets)
    if segment_boundaries:
        for sb_x, sb_label in zip(segment_boundaries,
                                   segment_gap_labels or [''] * len(segment_boundaries)):
            ax1.axvline(sb_x, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
            if sb_label:
                ax1.annotate(f'gap\n{sb_label}', xy=(sb_x, 0),
                             xycoords=('data', 'axes fraction'),
                             fontsize=6, color='red', alpha=0.7, ha='center',
                             va='bottom', rotation=90)

    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title(f'{split_name}: All Checkpoints vs DJIA')
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Cumulative Return (%)')
    ncol = 3 if n > 20 else (2 if n > 10 else 1)
    ax1.legend(loc='upper left', fontsize=6, ncol=ncol, framealpha=0.7)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Best by different metrics + DJIA ────────────────────
    ax2 = axes[0, 1]
    best_return_ckpt = df.loc[df['final_return'].idxmax(), 'checkpoint']
    best_sharpe_ckpt = df.loc[df['sharpe'].idxmax(), 'checkpoint']
    best_consistency_ckpt = df.loc[df['pct_beating'].idxmax(), 'checkpoint']

    # Group metrics that resolve to the same checkpoint
    # Always plot all three metrics as separate legend entries, even if they
    # resolve to the same checkpoint.  Use distinct colors + line styles.
    for ckpt, label, color, ls, lw in [
        (best_return_ckpt, 'Best Return', 'blue', '-', 2.5),
        (best_sharpe_ckpt, 'Best Sharpe', 'green', '--', 2.0),
        (best_consistency_ckpt, 'Most Consistent', 'orange', ':', 2.0),
    ]:
        if ckpt in all_curves:
            values = all_curves[ckpt]
            returns = (values / values[0] - 1) * 100
            short = ckpt.replace('actor__', '').replace('.pt', '').replace('.pth', '')
            ax2.plot(returns, linewidth=lw, color=color, linestyle=ls,
                     label=f'{label}: {short[-20:]} ({returns[-1]:.1f}%)')

    if has_baseline:
        baseline_returns = (baseline_values / baseline_values[0] - 1) * 100
        ax2.plot(baseline_returns, 'k--', linewidth=2,
                 label=f'DJIA ({baseline_return:.1f}%)')

    # Segment boundaries on panel 2 as well
    if segment_boundaries:
        for sb_x in segment_boundaries:
            ax2.axvline(sb_x, color='red', linestyle=':', alpha=0.6, linewidth=1.5)

    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_title('Best Checkpoints by Metric')
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Return vs Consistency scatter (color = Sharpe) ──────
    # Matches original: x = % days beating baseline, y = final return
    ax3 = axes[1, 0]
    sc = ax3.scatter(df['pct_beating'], df['final_return'],
                     c=df['sharpe'], cmap='RdYlGn', s=100, alpha=0.7,
                     edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('% Days Beating DJIA')
    ax3.set_ylabel('Final Return (%)')
    ax3.set_title('Return vs Consistency (color = Sharpe)')
    if has_baseline:
        ax3.axhline(baseline_return, color='red', linestyle='--',
                     label=f'DJIA Return ({baseline_return:.1f}%)')
    ax3.axvline(50, color='gray', linestyle=':', alpha=0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax3, label='Sharpe Ratio')

    # Annotate top-k by pct_beating
    top_df = df.nlargest(min(top_k, len(df)), 'pct_beating')
    for _, row in top_df.iterrows():
        short = row['checkpoint'].replace('actor__', '').replace('.pt', '')[:12]
        ax3.annotate(short, (row['pct_beating'], row['final_return']),
                     fontsize=7, alpha=0.8,
                     xytext=(5, 5), textcoords='offset points')

    # ── Panel 4: Metrics bar chart (Return % vs Beat Baseline %) ─────
    # Matches original: grouped bars with baseline overlay
    ax4 = axes[1, 1]
    df_sorted = df.sort_values('pct_beating', ascending=False)
    x = range(len(df_sorted))
    width = 0.35
    ax4.bar([i - width / 2 for i in x], df_sorted['final_return'], width,
            label='Return %', color='blue', alpha=0.7)
    ax4.bar([i + width / 2 for i in x], df_sorted['pct_beating'], width,
            label='Beat DJIA %', color='green', alpha=0.7)
    if has_baseline:
        ax4.axhline(baseline_return, color='red', linestyle='--',
                     label=f'DJIA ({baseline_return:.1f}%)')
    ax4.set_xticks(list(x))
    ax4.set_xticklabels(
        [c[-15:] for c in df_sorted['checkpoint']],
        rotation=45, ha='right', fontsize=8,
    )
    ax4.set_ylabel('Percentage')
    ax4.set_title('Return vs Consistency by Checkpoint')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{split_name}: Checkpoint Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(split_dir, 'checkpoint_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {plot_path}")

    # Recommendations (matches original)
    print(f"\n    RECOMMENDATIONS:")
    print(f"      Best Return:     {best_return_ckpt}")
    print(f"      Best Sharpe:     {best_sharpe_ckpt}")
    print(f"      Most Consistent: {best_consistency_ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary across splits
# ─────────────────────────────────────────────────────────────────────────────

def summarise_all_splits(results_dir: str, split_dfs: dict):
    """Print aggregate summary and highlight avgR vs OOS Sharpe mismatches."""
    print(f"\n{'='*80}")
    print(f"AGGREGATE SUMMARY")
    print(f"{'='*80}")

    header = (f"  {'Split':<10} {'Checkpoint':<35} "
              f"{'Return%':>8} {'Alpha%':>8} {'Sharpe':>7} {'Beat%':>7} {'Step':>10}")

    for metric_name, col, ascending in [
        ('Best OOS Sharpe', 'sharpe', False),
        ('Best OOS Return', 'final_return', False),
        ('Most Consistent (Beat%)', 'pct_beating', False),
        ('Least Drawdown', 'max_drawdown', False),
    ]:
        print(f"\n  --- {metric_name} ---")
        print(header)
        print(f"  {'-'*88}")
        for split_name, df in sorted(split_dfs.items()):
            if df is None or len(df) == 0:
                continue
            if col == 'max_drawdown':
                best = df.loc[df[col].idxmax()]  # less negative = better
            else:
                best = df.loc[df[col].idxmax()]
            print(f"  {split_name:<10} {best['checkpoint']:<35} "
                  f"{best['final_return']:>+7.2f}% "
                  f"{best.get('alpha', 0):>+7.2f}% "
                  f"{best['sharpe']:>7.2f} "
                  f"{best.get('pct_beating', 0):>6.0f}% "
                  f"{int(best['step']):>10,}")

    # Compare: does best avgR = best OOS Sharpe?
    print(f"\n  --- Best avgR vs Best OOS Sharpe ---")
    mismatches = 0
    total = 0
    for split_name, df in sorted(split_dfs.items()):
        if df is None or len(df) == 0:
            continue
        df_with_avgr = df.dropna(subset=['train_avgR'])
        if len(df_with_avgr) == 0:
            continue
        total += 1
        best_avgr_ckpt = df_with_avgr.loc[df_with_avgr['train_avgR'].idxmax(), 'checkpoint']
        best_sharpe_ckpt = df.loc[df['sharpe'].idxmax(), 'checkpoint']
        match = best_avgr_ckpt == best_sharpe_ckpt
        marker = 'MATCH' if match else 'MISMATCH'
        if not match:
            mismatches += 1
        print(f"  {split_name:<10} [{marker:>8}] "
              f"avgR -> {best_avgr_ckpt:<30} Sharpe -> {best_sharpe_ckpt}")

    if mismatches > 0:
        print(f"\n  WARNING: {mismatches}/{total} splits: best-avgR != best-OOS-Sharpe")

    # Rebuild combined CSV from ALL per-split checkpoint_results.csv on disk.
    # This is authoritative -- no fragile merge with prior all_checkpoint state.
    combined_path = os.path.join(results_dir, 'all_checkpoint_results.csv')
    all_rows = []
    for d in sorted(os.listdir(results_dir)):
        split_csv = os.path.join(results_dir, d, 'checkpoint_results.csv')
        if d.startswith('split_') and os.path.isfile(split_csv):
            df_disk = pd.read_csv(split_csv)
            df_disk.insert(0, 'split', d)
            all_rows.append(df_disk)
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined CSV: {combined_path} "
              f"({len(all_rows)} split(s), {len(combined)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate ALL checkpoints in CPCV split directories'
    )
    parser.add_argument(
        '--results-dir', type=str, required=True,
        help='Path to CPCV results directory (contains split_0/, split_1/, ...)'
    )
    parser.add_argument('--split', type=int, default=None,
                        help='Evaluate only this split index (0-based)')
    parser.add_argument('--gpu', type=int, default=GPU_ID)
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate even if checkpoint_results.csv exists')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Highlight top-k checkpoints in plots')
    return parser.parse_args()


def main():
    args = parse_args()

    close_ary, tech_ary = load_full_data()
    all_dates = load_dates()
    print(f"Data: {close_ary.shape[0]} days x {close_ary.shape[1]} stocks, "
          f"dates: {all_dates[0]} to {all_dates[-1]}")

    assert len(all_dates) == close_ary.shape[0], (
        f"Date count ({len(all_dates)}) != data rows ({close_ary.shape[0]})"
    )

    # Find split directories
    if args.split is not None:
        split_dirs = [f'split_{args.split}']
    else:
        split_dirs = sorted([
            d for d in os.listdir(args.results_dir)
            if d.startswith('split_')
            and os.path.isdir(os.path.join(args.results_dir, d))
        ])

    if not split_dirs:
        print(f"No split_* directories found in {args.results_dir}")
        return 1

    print(f"Evaluating {len(split_dirs)} split(s): {', '.join(split_dirs)}")

    split_dfs = {}
    for split_dir_name in split_dirs:
        split_dir = os.path.join(args.results_dir, split_dir_name)
        if not os.path.isdir(split_dir):
            print(f"  {split_dir_name}: directory not found, skipping")
            continue

        df = evaluate_split(
            split_dir=split_dir,
            close_ary=close_ary,
            tech_ary=tech_ary,
            all_dates=all_dates,
            gpu_id=args.gpu,
            force=args.force,
            top_k=args.top_k,
        )
        split_dfs[split_dir_name] = df

    # Aggregate summary (if any valid results)
    valid_dfs = {k: v for k, v in split_dfs.items() if v is not None}
    if valid_dfs:
        summarise_all_splits(args.results_dir, valid_dfs)

    return 0


if __name__ == '__main__':
    sys.exit(main())
