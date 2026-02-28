#!/usr/bin/env python3
"""
Compare CPCV/ACPCV training configurations side-by-side.

Produces:
  1. Aggregate OOS metrics table (Sharpe, Return, Alpha, Beat%, MaxDD) per config
  2. Per-split comparison heatmap
  3. Training curve comparison (recorder.npy avgR over steps)
  4. Box/violin plots of per-split distributions

Usage:
    # Compare all configs in train_results/
    python -m cpcv_pipeline.compare_configs

    # Compare specific directories
    python -m cpcv_pipeline.compare_configs \
        --dirs train_results/20260224_* train_results/20260225_* train_results/20260227_*

    # Custom labels
    python -m cpcv_pipeline.compare_configs \
        --dirs train_results/20260224_* train_results/20260225_* \
        --labels "no-norm" "norm-obs-only"

    # Save to specific output directory
    python -m cpcv_pipeline.compare_configs --output comparison_results/
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Config discovery and labeling
# ─────────────────────────────────────────────────────────────────────────────

def auto_label(dirname: str) -> str:
    """Generate a short human-readable label from a results directory name."""
    base = os.path.basename(dirname.rstrip('/'))

    # Check for vec_normalize.pt in split_0 to detect normalization type
    split0 = os.path.join(dirname, 'split_0')
    has_vecnorm = os.path.exists(os.path.join(split0, 'vec_normalize.pt'))

    # Check if it's AdaptCPCV
    is_acpcv = 'AdaptCPCV' in base or 'Adapt' in base

    # Check split_meta.json for feature info
    feature = ''
    meta_path = os.path.join(split0, 'split_meta.json')
    if os.path.exists(meta_path):
        import json
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            feat = meta.get('feature', meta.get('features', ''))
            if feat and feat != 'default':
                feature = f'+{feat}'
        except (json.JSONDecodeError, KeyError):
            pass

    # Check norm type from vec_normalize
    norm_label = 'no-norm'
    if has_vecnorm:
        try:
            import torch
            vn = torch.load(os.path.join(split0, 'vec_normalize.pt'), map_location='cpu', weights_only=False)
            norm_obs = vn.get('norm_obs', False)
            norm_rew = vn.get('norm_reward', vn.get('norm_ret', False))
            if norm_obs and norm_rew:
                norm_label = 'full-norm'
            elif norm_obs:
                norm_label = 'norm-obs'
            elif norm_rew:
                norm_label = 'norm-rew'
        except Exception:
            norm_label = 'normed'

    method = 'ACPCV' if is_acpcv else 'CPCV'
    # Extract date prefix
    date_prefix = base[:8] if len(base) >= 8 and base[:8].isdigit() else ''

    label = f"{method} {norm_label}{feature}"
    if date_prefix:
        label = f"{label} ({date_prefix})"

    return label


def discover_configs(train_results_dir: str) -> List[str]:
    """Find all config directories under train_results/."""
    pattern = os.path.join(train_results_dir, '*')
    dirs = sorted(glob.glob(pattern))
    return [d for d in dirs if os.path.isdir(d) and
            os.path.exists(os.path.join(d, 'all_checkpoint_results.csv'))]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint_results(results_dir: str) -> pd.DataFrame:
    """Load all_checkpoint_results.csv for a config."""
    csv_path = os.path.join(results_dir, 'all_checkpoint_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No all_checkpoint_results.csv in {results_dir}")
    return pd.read_csv(csv_path)


def get_best_per_split(df: pd.DataFrame, metric: str = 'sharpe') -> pd.DataFrame:
    """Get the best checkpoint per split by a given metric."""
    # For each split, pick the row with the highest metric value
    idx = df.groupby('split')[metric].idxmax()
    return df.loc[idx].reset_index(drop=True)


def load_recorders(results_dir: str) -> Dict[int, np.ndarray]:
    """Load recorder.npy for all available splits."""
    recorders = {}
    for i in range(20):  # generous upper bound
        rec_path = os.path.join(results_dir, f'split_{i}', 'recorder.npy')
        if os.path.exists(rec_path):
            recorders[i] = np.load(rec_path)
    return recorders


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_aggregate(best_df: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregate metrics from best-per-split dataframe."""
    agg = {}
    for col in ['final_return', 'alpha', 'sharpe', 'sortino', 'max_drawdown',
                'ann_return', 'pct_beating']:
        if col in best_df.columns:
            vals = best_df[col].dropna()
            agg[f'{col}_mean'] = vals.mean()
            agg[f'{col}_std'] = vals.std()
            agg[f'{col}_median'] = vals.median()
            agg[f'{col}_min'] = vals.min()
            agg[f'{col}_max'] = vals.max()
    agg['n_splits'] = len(best_df)
    agg['positive_alpha_pct'] = (best_df['alpha'].dropna() > 0).mean() * 100 if 'alpha' in best_df.columns else np.nan
    agg['sharpe_above_1_pct'] = (best_df['sharpe'].dropna() > 1.0).mean() * 100 if 'sharpe' in best_df.columns else np.nan

    # Cross-split variability (CV = std/|mean|, lower = more stable)
    for col in ['final_return', 'alpha', 'sharpe', 'sortino']:
        if col in best_df.columns:
            vals = best_df[col].dropna()
            mean_val = vals.mean()
            if abs(mean_val) > 1e-8:
                agg[f'{col}_cv'] = vals.std() / abs(mean_val)
            else:
                agg[f'{col}_cv'] = np.nan
            agg[f'{col}_iqr'] = vals.quantile(0.75) - vals.quantile(0.25)
            agg[f'{col}_range'] = vals.max() - vals.min()

    return agg


def compute_training_stability(recorders: Dict[int, np.ndarray]) -> Dict[str, float]:
    """Compute training stability metrics from recorder.npy data.

    Recorder columns: [step, avgR, stdR, expR, objC, objA, objEntropy, exploreRate]

    Returns dict with:
      - final_avgR_mean/std: final avgR across splits
      - best_avgR_mean/std: peak avgR across splits
      - late_oscillation_mean: std of avgR in last 20% of training (lower = more stable)
      - overshoot_mean: (best - final) / best — how much avgR drops from peak (lower = better)
      - convergence_step_mean: step to reach 90% of final avgR (lower = faster)
      - objC_final_mean: final critic loss (lower = better fit)
      - expR_trend: whether exploration reward is still improving at end
    """
    if not recorders:
        return {}

    stability = {}
    per_split = {
        'final_avgR': [], 'best_avgR': [], 'late_oscillation': [],
        'overshoot': [], 'convergence_step_pct': [],
        'objC_final': [], 'objA_final': [],
        'objC_late_oscillation': [], 'objA_late_oscillation': [],
        'objC_reduction': [], 'objC_trend': [], 'objA_trend': [],
        'avgR_progression': [],
        'objC_progression': [], 'objA_progression': [],
    }

    for split_idx, rec in sorted(recorders.items()):
        steps = rec[:, 0]
        avgR = rec[:, 1]
        stdR = rec[:, 2]
        expR = rec[:, 3]
        objC = rec[:, 4]
        objA = rec[:, 5]

        n = len(avgR)
        if n < 5:
            continue

        # Final and best avgR
        final_r = avgR[-1]
        best_r = avgR.max()
        per_split['final_avgR'].append(final_r)
        per_split['best_avgR'].append(best_r)

        # Late-stage oscillation: std of avgR in last 20%
        tail_start = int(n * 0.8)
        late_std = np.std(avgR[tail_start:])
        per_split['late_oscillation'].append(late_std)

        # Overshoot: how much avgR drops from peak to final
        if best_r > 1e-8:
            overshoot = (best_r - final_r) / best_r
        else:
            overshoot = 0.0
        per_split['overshoot'].append(overshoot)

        # Convergence speed: step to reach 90% of best avgR
        target_r = avgR[0] + 0.9 * (best_r - avgR[0])
        reached_idx = np.where(avgR >= target_r)[0]
        if len(reached_idx) > 0:
            convergence_pct = steps[reached_idx[0]] / steps[-1]
        else:
            convergence_pct = 1.0
        per_split['convergence_step_pct'].append(convergence_pct)

        # Final critic/actor objectives
        per_split['objC_final'].append(objC[-1])
        per_split['objA_final'].append(objA[-1])

        # objC/objA late-stage oscillation (std of last 20%)
        objC_late_std = np.std(objC[tail_start:])
        objA_late_std = np.std(objA[tail_start:])
        per_split['objC_late_oscillation'].append(objC_late_std)
        per_split['objA_late_oscillation'].append(objA_late_std)

        # objC reduction ratio: how much critic loss decreased (early 10% mean vs final 10% mean)
        early_end = max(int(n * 0.1), 1)
        late_start_10 = int(n * 0.9)
        objC_early = np.mean(objC[:early_end])
        objC_late = np.mean(objC[late_start_10:])
        if abs(objC_early) > 1e-8:
            per_split['objC_reduction'].append(objC_late / objC_early)
        else:
            per_split['objC_reduction'].append(np.nan)

        # objC/objA late-stage trend (linear slope over last 20%, normalized)
        tail_steps = steps[tail_start:]
        if len(tail_steps) > 2:
            # Normalize steps to [0, 1] for comparable slope
            t_norm = (tail_steps - tail_steps[0]) / max(tail_steps[-1] - tail_steps[0], 1)
            objC_slope = np.polyfit(t_norm, objC[tail_start:], 1)[0]
            objA_slope = np.polyfit(t_norm, objA[tail_start:], 1)[0]
            per_split['objC_trend'].append(objC_slope)
            per_split['objA_trend'].append(objA_slope)
        else:
            per_split['objC_trend'].append(0.0)
            per_split['objA_trend'].append(0.0)

        # Progressions (8 evenly spaced samples)
        sample_idx = np.linspace(0, n - 1, 8, dtype=int)
        per_split['avgR_progression'].append(avgR[sample_idx])
        per_split['objC_progression'].append(objC[sample_idx])
        per_split['objA_progression'].append(objA[sample_idx])

    # Aggregate across splits
    for key in ['final_avgR', 'best_avgR', 'late_oscillation', 'overshoot',
                'convergence_step_pct', 'objC_final', 'objA_final',
                'objC_late_oscillation', 'objA_late_oscillation',
                'objC_reduction', 'objC_trend', 'objA_trend']:
        vals = np.array([v for v in per_split[key] if not (isinstance(v, float) and np.isnan(v))])
        if len(vals) > 0:
            stability[f'{key}_mean'] = vals.mean()
            stability[f'{key}_std'] = vals.std()
            stability[f'{key}_median'] = np.median(vals)

    # Cross-split consistency (CV = std/|mean|)
    for key in ['final_avgR', 'objC_final', 'objA_final']:
        vals = np.array(per_split[key])
        if len(vals) > 1 and abs(vals.mean()) > 1e-8:
            stability[f'{key}_cv'] = vals.std() / abs(vals.mean())
        else:
            stability[f'{key}_cv'] = np.nan

    stability['n_recorder_splits'] = len(per_split['final_avgR'])
    stability['avgR_progressions'] = per_split['avgR_progression']
    stability['objC_progressions'] = per_split['objC_progression']
    stability['objA_progressions'] = per_split['objA_progression']

    return stability


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(configs: List[Dict], selection: str = 'sharpe'):
    """Print a side-by-side comparison table."""
    print(f"\n{'='*100}")
    print(f"  CONFIGURATION COMPARISON — Best checkpoint selected by: {selection.upper()}")
    print(f"{'='*100}")

    # Header
    header = f"{'Metric':<25}"
    for cfg in configs:
        header += f"  {cfg['label']:>22}"
    print(header)
    print('-' * (25 + 24 * len(configs)))

    # Rows
    metrics = [
        ('Splits', 'n_splits', '{:.0f}'),
        ('Return (mean)', 'final_return_mean', '{:+.2f}%'),
        ('Return (median)', 'final_return_median', '{:+.2f}%'),
        ('Return (std)', 'final_return_std', '{:.2f}%'),
        ('Alpha (mean)', 'alpha_mean', '{:+.2f}%'),
        ('Alpha (median)', 'alpha_median', '{:+.2f}%'),
        ('Positive Alpha %', 'positive_alpha_pct', '{:.0f}%'),
        ('Sharpe (mean)', 'sharpe_mean', '{:.2f}'),
        ('Sharpe (median)', 'sharpe_median', '{:.2f}'),
        ('Sharpe > 1.0 %', 'sharpe_above_1_pct', '{:.0f}%'),
        ('Sortino (mean)', 'sortino_mean', '{:.2f}'),
        ('MaxDD (mean)', 'max_drawdown_mean', '{:.2f}%'),
        ('MaxDD (worst)', 'max_drawdown_min', '{:.2f}%'),
        ('Beat% (mean)', 'pct_beating_mean', '{:.1f}%'),
    ]

    for display_name, key, fmt in metrics:
        row = f"{display_name:<25}"
        for cfg in configs:
            val = cfg['agg'].get(key, np.nan)
            if np.isnan(val):
                row += f"  {'N/A':>22}"
            else:
                row += f"  {fmt.format(val):>22}"
        print(row)

    print()


def print_stability_table(configs: List[Dict]):
    """Print cross-split variability and training stability metrics."""
    print(f"\n{'='*100}")
    print(f"  METRIC STABILITY — Cross-Split Variability & Training Convergence")
    print(f"{'='*100}")

    # Header
    header = f"{'Metric':<30}"
    for cfg in configs:
        header += f"  {cfg['label']:>22}"
    print(header)
    print('-' * (30 + 24 * len(configs)))

    # Section 1: Cross-split variability (from OOS metrics)
    print(f"\n  {'--- Cross-Split Variability (lower = more consistent) ---':^{30 + 24*len(configs)}}")
    variability_metrics = [
        ('Sharpe CV', 'sharpe_cv', '{:.3f}'),
        ('Sharpe IQR', 'sharpe_iqr', '{:.3f}'),
        ('Sharpe range', 'sharpe_range', '{:.3f}'),
        ('Return CV', 'final_return_cv', '{:.3f}'),
        ('Return IQR', 'final_return_iqr', '{:.2f}%'),
        ('Return range', 'final_return_range', '{:.2f}%'),
        ('Alpha CV', 'alpha_cv', '{:.3f}'),
        ('Alpha IQR', 'alpha_iqr', '{:.2f}%'),
        ('Sortino CV', 'sortino_cv', '{:.3f}'),
    ]

    for display_name, key, fmt in variability_metrics:
        row = f"  {display_name:<28}"
        for cfg in configs:
            val = cfg['agg'].get(key, np.nan)
            if isinstance(val, float) and np.isnan(val):
                row += f"  {'N/A':>22}"
            else:
                row += f"  {fmt.format(val):>22}"
        print(row)

    # Section 2: Training stability (from recorder.npy)
    has_stability = any('stability' in cfg for cfg in configs)
    if has_stability:
        print(f"\n  {'--- Training Stability (from recorder.npy) ---':^{30 + 24*len(configs)}}")
        training_metrics = [
            ('Recorder splits', 'n_recorder_splits', '{:.0f}'),
            ('Final avgR (mean)', 'final_avgR_mean', '{:.2f}'),
            ('Final avgR (std)', 'final_avgR_std', '{:.2f}'),
            ('Final avgR CV', 'final_avgR_cv', '{:.3f}'),
            ('Best avgR (mean)', 'best_avgR_mean', '{:.2f}'),
            ('Late oscillation (mean)', 'late_oscillation_mean', '{:.2f}'),
            ('Late oscillation (std)', 'late_oscillation_std', '{:.2f}'),
            ('Overshoot % (mean)', 'overshoot_mean', '{:.3f}'),
            ('Convergence @ 90% (mean)', 'convergence_step_pct_mean', '{:.1%}'),
            ('Critic loss final', 'objC_final_mean', '{:.2f}'),
            ('Critic loss CV', 'objC_final_cv', '{:.3f}'),
            ('Critic late oscillation', 'objC_late_oscillation_mean', '{:.3f}'),
            ('Critic reduction (late/early)', 'objC_reduction_mean', '{:.3f}'),
            ('Critic late trend', 'objC_trend_mean', '{:+.3f}'),
            ('Actor obj final', 'objA_final_mean', '{:.4f}'),
            ('Actor obj CV', 'objA_final_cv', '{:.3f}'),
            ('Actor late oscillation', 'objA_late_oscillation_mean', '{:.4f}'),
            ('Actor late trend', 'objA_trend_mean', '{:+.4f}'),
        ]

        for display_name, key, fmt in training_metrics:
            row = f"  {display_name:<28}"
            for cfg in configs:
                stab = cfg.get('stability', {})
                val = stab.get(key, np.nan)
                if isinstance(val, float) and np.isnan(val):
                    row += f"  {'N/A':>22}"
                else:
                    row += f"  {fmt.format(val):>22}"
            print(row)

        # Per-split progressions
        progression_labels = ['start', '14%', '29%', '43%', '57%', '71%', '86%', 'end']

        for prog_name, prog_key, fmt in [('avgR', 'avgR_progressions', '{:.2f}'),
                                          ('objC', 'objC_progressions', '{:.2f}'),
                                          ('objA', 'objA_progressions', '{:.4f}')]:
            print(f"\n  {'--- ' + prog_name + ' Progression (mean across splits) ---':^{30 + 24*len(configs)}}")
            for p_idx, p_label in enumerate(progression_labels):
                row = f"  {prog_name + ' @ ' + p_label:<28}"
                for cfg in configs:
                    stab = cfg.get('stability', {})
                    progs = stab.get(prog_key, [])
                    if progs:
                        vals = [p[p_idx] for p in progs if len(p) > p_idx]
                        if vals:
                            mean_val = np.mean(vals)
                            row += f"  {fmt.format(mean_val):>22}"
                        else:
                            row += f"  {'N/A':>22}"
                    else:
                        row += f"  {'N/A':>22}"
                print(row)

    print()


def print_per_split_table(configs: List[Dict], metric: str = 'sharpe'):
    """Print per-split comparison for a given metric."""
    print(f"\n{'='*100}")
    print(f"  PER-SPLIT COMPARISON — {metric.upper()}")
    print(f"{'='*100}")

    header = f"{'Split':<10}"
    for cfg in configs:
        header += f"  {cfg['label']:>18}"
    header += f"  {'Best':>18}"
    print(header)
    print('-' * (10 + 20 * (len(configs) + 1)))

    all_splits = sorted(set().union(
        *(cfg['best_df']['split'].unique() for cfg in configs)
    ))

    wins = {cfg['label']: 0 for cfg in configs}

    for split in all_splits:
        row = f"{split:<10}"
        best_val = -np.inf
        best_label = ''
        for cfg in configs:
            split_data = cfg['best_df'][cfg['best_df']['split'] == split]
            if len(split_data) == 0:
                row += f"  {'---':>18}"
                continue
            val = split_data[metric].values[0]
            if metric in ['max_drawdown']:
                # For drawdown, less negative is better
                fmt_val = f"{val:.2f}%"
                if val > best_val:
                    best_val = val
                    best_label = cfg['label']
            elif metric in ['final_return', 'alpha', 'ann_return']:
                fmt_val = f"{val:+.2f}%"
                if val > best_val:
                    best_val = val
                    best_label = cfg['label']
            elif metric in ['pct_beating']:
                fmt_val = f"{val:.1f}%"
                if val > best_val:
                    best_val = val
                    best_label = cfg['label']
            else:
                fmt_val = f"{val:.3f}"
                if val > best_val:
                    best_val = val
                    best_label = cfg['label']
            row += f"  {fmt_val:>18}"
        row += f"  {best_label:>18}"
        if best_label:
            wins[best_label] = wins.get(best_label, 0) + 1
        print(row)

    # Win counts
    print('-' * (10 + 20 * (len(configs) + 1)))
    win_row = f"{'Wins':<10}"
    for cfg in configs:
        win_row += f"  {wins[cfg['label']]:>18}"
    print(win_row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(configs: List[Dict], output_dir: str):
    """Plot average training curves (avgR over steps) for each config."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available for training curve plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(configs), 10)))

    # Left: avgR over steps (mean ± std across splits)
    ax = axes[0]
    for i, cfg in enumerate(configs):
        recs = cfg.get('recorders', {})
        if not recs:
            continue

        # Interpolate all splits to common step axis
        all_steps = np.concatenate([r[:, 0] for r in recs.values()])
        max_step = np.percentile(all_steps, 50)  # use median max to avoid outlier splits
        common_steps = np.linspace(0, max_step, 200)

        interp_avgR = []
        for split_idx, rec in recs.items():
            steps = rec[:, 0]
            avgR = rec[:, 1]
            interp = np.interp(common_steps, steps, avgR, left=np.nan, right=np.nan)
            interp_avgR.append(interp)

        interp_avgR = np.array(interp_avgR)
        mean_avgR = np.nanmean(interp_avgR, axis=0)
        std_avgR = np.nanstd(interp_avgR, axis=0)

        valid = ~np.isnan(mean_avgR)
        ax.plot(common_steps[valid], mean_avgR[valid], label=cfg['label'],
                color=colors[i], linewidth=1.5)
        ax.fill_between(common_steps[valid],
                        (mean_avgR - std_avgR)[valid],
                        (mean_avgR + std_avgR)[valid],
                        color=colors[i], alpha=0.15)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('avgR (mean across splits)')
    ax.set_title('Training Curves: avgR over Steps')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

    # Right: Box plot of final avgR per split
    ax = axes[1]
    box_data = []
    box_labels = []
    for cfg in configs:
        recs = cfg.get('recorders', {})
        if recs:
            final_avgRs = [rec[-1, 1] for rec in recs.values()]
            box_data.append(final_avgRs)
            box_labels.append(cfg['label'])

    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.4)
        ax.set_ylabel('Final avgR')
        ax.set_title('Distribution of Final avgR Across Splits')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_oos_comparison(configs: List[Dict], output_dir: str):
    """Plot OOS metric comparison across configs."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available for OOS comparison plot")
        return

    metrics = ['sharpe', 'final_return', 'alpha', 'pct_beating']
    titles = ['OOS Sharpe Ratio', 'OOS Return (%)', 'OOS Alpha (%)', 'Beat DJIA (%)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(configs), 10)))

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        box_data = []
        box_labels = []
        for cfg in configs:
            vals = cfg['best_df'][metric].dropna().values
            if len(vals) > 0:
                box_data.append(vals)
                box_labels.append(cfg['label'])

        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
            for j, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[j])
                patch.set_alpha(0.4)

            # Add individual points
            for j, data in enumerate(box_data):
                x = np.random.normal(j + 1, 0.04, size=len(data))
                ax.scatter(x, data, alpha=0.5, s=20, color=colors[j], zorder=5)

        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

        # Add mean line
        for j, data in enumerate(box_data):
            ax.axhline(y=np.mean(data), xmin=(j + 0.6) / (len(box_data) + 1),
                       xmax=(j + 1.4) / (len(box_data) + 1),
                       color=colors[j], linestyle='--', alpha=0.5, linewidth=1)

    plt.suptitle('OOS Metrics Comparison (Best Sharpe per Split)', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'oos_metrics_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_per_split_heatmap(configs: List[Dict], output_dir: str, metric: str = 'sharpe'):
    """Plot a heatmap of per-split metric values across configs."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available for heatmap")
        return

    # Build matrix: rows=splits, cols=configs
    all_splits = sorted(set().union(
        *(cfg['best_df']['split'].unique() for cfg in configs)
    ))
    n_splits = len(all_splits)
    n_configs = len(configs)

    data = np.full((n_splits, n_configs), np.nan)
    for j, cfg in enumerate(configs):
        for i, split in enumerate(all_splits):
            row = cfg['best_df'][cfg['best_df']['split'] == split]
            if len(row) > 0 and metric in row.columns:
                data[i, j] = row[metric].values[0]

    fig, ax = plt.subplots(figsize=(max(8, n_configs * 2.5), max(6, n_splits * 0.6)))

    # Choose colormap
    if metric in ['max_drawdown']:
        cmap = 'RdYlGn'  # green=less negative=better
    else:
        cmap = 'RdYlGn'  # green=higher=better

    im = ax.imshow(data, cmap=cmap, aspect='auto')

    # Labels
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels([cfg['label'] for cfg in configs], rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels(all_splits, fontsize=9)
    ax.set_ylabel('Split')

    # Annotate cells
    for i in range(n_splits):
        for j in range(n_configs):
            val = data[i, j]
            if not np.isnan(val):
                if metric in ['final_return', 'alpha', 'ann_return', 'max_drawdown']:
                    text = f"{val:+.1f}%"
                elif metric in ['pct_beating']:
                    text = f"{val:.0f}%"
                else:
                    text = f"{val:.2f}"
                ax.text(j, i, text, ha='center', va='center', fontsize=8,
                        color='black' if abs(val) < abs(np.nanmax(data)) * 0.7 else 'white')

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f'Per-Split {metric.upper()} Comparison', fontsize=12)

    # Highlight best per row
    for i in range(n_splits):
        row_vals = data[i, :]
        valid = ~np.isnan(row_vals)
        if valid.any():
            if metric in ['max_drawdown']:
                best_j = np.nanargmax(row_vals)  # least negative
            else:
                best_j = np.nanargmax(row_vals)
            ax.add_patch(plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='gold', linewidth=2.5))

    plt.tight_layout()
    path = os.path.join(output_dir, f'heatmap_{metric}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_summary_csv(configs: List[Dict], output_dir: str):
    """Export a summary CSV with aggregate metrics per config."""
    rows = []
    for cfg in configs:
        row = {'config': cfg['label'], 'directory': cfg['dir']}
        row.update(cfg['agg'])
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'config_comparison_summary.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Also export per-split best values for all configs combined
    combined = []
    for cfg in configs:
        bdf = cfg['best_df'].copy()
        bdf['config'] = cfg['label']
        combined.append(bdf)
    combined_df = pd.concat(combined, ignore_index=True)
    path2 = os.path.join(output_dir, 'config_comparison_per_split.csv')
    combined_df.to_csv(path2, index=False)
    print(f"  Saved: {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare CPCV/ACPCV training configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--dirs', nargs='+',
                        help='Result directories to compare. If omitted, auto-discovers from train_results/')
    parser.add_argument('--labels', nargs='+',
                        help='Custom labels for each directory (must match --dirs count)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory for plots/CSVs (default: train_results/comparison_<timestamp>)')
    parser.add_argument('--metric', default='sharpe',
                        choices=['sharpe', 'final_return', 'alpha', 'sortino', 'pct_beating', 'max_drawdown'],
                        help='Primary metric for best-checkpoint selection (default: sharpe)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--heatmap-metrics', nargs='+',
                        default=['sharpe', 'final_return', 'alpha', 'max_drawdown'],
                        help='Metrics to generate heatmaps for')

    args = parser.parse_args()

    # Discover or validate directories
    if args.dirs:
        result_dirs = []
        for d in args.dirs:
            expanded = sorted(glob.glob(d))
            result_dirs.extend([x for x in expanded if os.path.isdir(x)])
        if not result_dirs:
            print(f"ERROR: No valid directories found from: {args.dirs}")
            sys.exit(1)
    else:
        train_results_dir = os.path.join(PROJECT_ROOT, 'train_results')
        result_dirs = discover_configs(train_results_dir)
        if not result_dirs:
            print(f"ERROR: No configs found in {train_results_dir}")
            sys.exit(1)

    # Assign labels
    if args.labels:
        if len(args.labels) != len(result_dirs):
            print(f"ERROR: {len(args.labels)} labels for {len(result_dirs)} directories")
            sys.exit(1)
        labels = args.labels
    else:
        labels = [auto_label(d) for d in result_dirs]

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(PROJECT_ROOT, 'train_results', f'comparison_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*100}")
    print(f"  CPCV CONFIGURATION COMPARISON")
    print(f"  Selection metric: {args.metric}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*100}\n")

    # Load data for each config
    configs = []
    for result_dir, label in zip(result_dirs, labels):
        print(f"  Loading: {label}")
        print(f"    Dir: {result_dir}")

        try:
            df = load_checkpoint_results(result_dir)
        except FileNotFoundError as e:
            print(f"    WARNING: {e} — skipping")
            continue

        best_df = get_best_per_split(df, metric=args.metric)
        agg = compute_aggregate(best_df)
        recorders = load_recorders(result_dir)

        n_splits = len(best_df)
        n_rec = len(recorders)
        print(f"    Splits: {n_splits}, Checkpoints: {len(df)}, Recorders: {n_rec}")

        configs.append({
            'label': label,
            'dir': result_dir,
            'df': df,
            'best_df': best_df,
            'agg': agg,
            'recorders': recorders,
            'stability': compute_training_stability(recorders),
        })

    if len(configs) < 2:
        print(f"\nERROR: Need at least 2 configs to compare, found {len(configs)}")
        sys.exit(1)

    print(f"\n  Loaded {len(configs)} configurations\n")

    # Print tables
    print_comparison_table(configs, selection=args.metric)
    print_stability_table(configs)
    print_per_split_table(configs, metric='sharpe')
    print_per_split_table(configs, metric='final_return')
    print_per_split_table(configs, metric='alpha')

    # Generate plots
    if not args.no_plots and HAS_MPL:
        print(f"\n  Generating plots...")
        plot_training_curves(configs, output_dir)
        plot_oos_comparison(configs, output_dir)
        for hm_metric in args.heatmap_metrics:
            plot_per_split_heatmap(configs, output_dir, metric=hm_metric)

    # Export CSVs
    export_summary_csv(configs, output_dir)

    print(f"\n{'='*100}")
    print(f"  COMPARISON COMPLETE — {len(configs)} configs, output: {output_dir}")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
