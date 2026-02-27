"""
CPCV Split Visualization — replicates Berend Gort & Bruce Yang's
``plot_cv_indices`` from the Combinatorial PurgedKFold CV notebook
(https://pub.towardsai.net/363eb378a9c5).

Original plot_cv_indices was based on sklearn's CV visualisation:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html

Layout matches the article's figure exactly:
  - Y-axis = Sample index (top=0, inverted)
  - X columns (right→left) = S1 .. Sn, "class", "group"
  - Blue (coolwarm 0)  = train
  - Red  (coolwarm 1)  = test
  - Dark red (clipped 2) = purge / embargo gap
  - seaborn style, serif font, compact figure, dpi=300
"""

from __future__ import annotations

import argparse
import itertools as itt
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── local imports ────────────────────────────────────────────────────────────
from cpcv_pipeline.function_CPCV import (
    CombPurgedKFoldCV,
    back_test_paths_generator,
)
from cpcv_pipeline.config import N_GROUPS, K_TEST_GROUPS, EMBARGO_DAYS


# ── colour maps — identical to the notebook ──────────────────────────────────
cmap_cv   = plt.cm.coolwarm   # 0 → blue (train), 1 → red (test), 2 → deep red
cmap_data = plt.cm.Paired     # for "group" / "class" columns


# ─────────────────────────────────────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────────────────────────────────────

def plot_cpcv_splits(
    total_samples: int,
    n_groups: int = N_GROUPS,
    n_test_groups: int = K_TEST_GROUPS,
    embargo_days: int = EMBARGO_DAYS,
    *,
    show_paths: bool = False,
    figsize: Optional[tuple] = None,
    dpi: int = 300,
    lw: float = 5,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    scale: float = 1.0,
) -> plt.Figure:
    """
    Visualise every CPCV split identically to the notebook's
    ``plot_cv_indices`` function.

    Uses the same seaborn style, coolwarm colourmap, compact figure size
    and ``marker='_'`` scatter approach so bars appear solid.

    Parameters
    ----------
    total_samples : int
        Number of time steps (days) in the dataset.
    n_groups, n_test_groups, embargo_days :
        CPCV hyper-parameters (N, K, embargo).
    show_paths : bool
        If True, also draw backtest-path assignment columns.
    figsize : tuple or None
        (width, height).  Auto-computed if None.
    dpi : int
        Resolution (default 300, matching notebook).
    lw : float
        Marker line-width for the scatter points.
    save_path : str or None
        If given, save the figure to this path.
    title : str or None
        Optional figure super-title.
    scale : float
        Scale factor applied to figsize and font sizes (default 1.0).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    cv = CombPurgedKFoldCV(
        n_splits=n_groups,
        n_test_splits=n_test_groups,
        embargo_days=embargo_days,
    )
    n_splits = cv.n_combinations
    n_paths  = cv.n_paths

    # ── assign each sample to a fold group ───────────────────────────────
    fold_bounds = cv.get_fold_bounds(total_samples)
    group = np.zeros(total_samples, dtype=int)
    for g, (s, e) in enumerate(fold_bounds):
        group[s:e] = g

    # ── build path matrix (optional) ─────────────────────────────────────
    if show_paths:
        _, paths, _ = back_test_paths_generator(
            total_samples, n_groups, n_test_groups,
        )

    # ── apply the notebook's rcParams inside a context ───────────────────
    with plt.style.context("seaborn-v0_8"):
        plt.rcParams.update({
            "figure.figsize": [5 * scale, 2 * scale],
            "figure.dpi": dpi,
            "font.size": 5 * scale,
            "axes.labelsize": 5 * scale,
            "axes.titlesize": 6 * scale,
            "xtick.labelsize": 4 * scale,
            "ytick.labelsize": 4 * scale,
            "font.family": "serif",
        })

        if figsize is None:
            base_w = 5 * scale
            if show_paths:
                base_w += n_paths * 0.3 * scale  # extra width for path columns
            figsize = (base_w, 2 * scale)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # ── draw each CV split column ────────────────────────────────────
        ii = -1  # in case n_splits == 0
        for ii, (tr, tt) in enumerate(cv.split(total_samples)):
            # identical to the notebook: nan → 2 (gap/purge/embargo)
            indices = np.array([np.nan] * total_samples)
            indices[tt] = 1    # test
            indices[tr] = 0    # train
            indices[np.isnan(indices)] = 2  # gap

            ax.scatter(
                [ii + 0.5] * len(indices),
                range(len(indices)),
                c=[indices],
                marker="_",
                lw=lw,
                cmap=cmap_cv,
                vmin=-0.2,
                vmax=1.2,
            )

        # ── "class" column (fold group ids — banded colors) ──────────
        ax.scatter(
            [ii + 1.5] * total_samples,
            range(total_samples),
            c=group,
            marker="_",
            lw=lw,
            cmap=cmap_data,
        )

        # ── "group" column (continuous sample index — smooth gradient) ───
        #    notebook passes list(range(X.shape[0])), Paired has 12 colours
        ax.scatter(
            [ii + 2.5] * total_samples,
            range(total_samples),
            c=list(range(total_samples)),
            marker="_",
            lw=lw,
            cmap=cmap_data,
        )

        # ── path columns (optional) ─────────────────────────────────────
        #    Each path column shows which split provides the OOS prediction
        #    for every sample. Colour = split index via tab20.
        n_extra = 2  # class + group
        if show_paths:
            for p in range(n_paths):
                ax.scatter(
                    [ii + 2.5 + (p + 1)] * total_samples,
                    range(total_samples),
                    c=paths[:, p],
                    marker="_",
                    lw=lw,
                    cmap="tab20",
                )
            n_extra += n_paths

        # ── formatting — exactly matches the notebook ────────────────────
        xlabelz = list(range(n_splits, 0, -1))
        xlabelz = ["S" + str(x) for x in xlabelz]
        xticklabels = xlabelz + ["class", "group"]
        if show_paths:
            xticklabels += [f"P{p + 1}" for p in range(n_paths)]

        ax.set(
            xticks=np.arange(n_splits + n_extra) + 0.45,
            xticklabels=xticklabels,
            ylabel="Sample index",
            xlabel="CV iteration",
            xlim=[n_splits + n_extra + 0.2, -0.2],
            ylim=[0, total_samples],
        )
        ax.set_title("{}".format(type(cv).__name__), fontsize=5 * scale)
        ax.xaxis.tick_top()
        plt.gca().invert_yaxis()

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
            print(f"Saved → {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Paths example — replicate the notebook's raw paths output
# ─────────────────────────────────────────────────────────────────────────────

def print_paths(
    total_samples: int = 30,
    n_groups: int = 6,
    n_test_groups: int = 2,
) -> np.ndarray:
    """
    Print the backtest paths matrix (1-indexed) like the notebook does::

        _, paths, _ = back_test_paths_generator(30, 6, k, ...)
        paths + 1

    Returns the raw ``paths`` array (0-indexed).
    """
    _, paths, path_folds = back_test_paths_generator(
        total_samples, n_groups, n_test_groups,
    )
    print(f"paths + 1  (shape {paths.shape}):\n")
    print(paths + 1)
    print(f"\npath_folds + 1  (which split serves each group per path):\n")
    print(path_folds + 1)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: plot from the actual Alpaca dataset dimensions
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpaca_cpcv(
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Plot CPCV splits for the Alpaca stock dataset (753 days)."""
    from cpcv_pipeline.config import ALPACA_NPZ_PATH
    data = np.load(ALPACA_NPZ_PATH)
    total_days = data["close_ary"].shape[0]
    return plot_cpcv_splits(total_days, save_path=save_path, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise CPCV train/test/embargo splits"
    )
    parser.add_argument(
        "--total-samples", "-T", type=int, default=None,
        help="Number of time steps. If omitted, uses the Alpaca dataset size.",
    )
    parser.add_argument("--n-groups", "-N", type=int, default=N_GROUPS)
    parser.add_argument("--n-test-groups", "-K", type=int, default=K_TEST_GROUPS)
    parser.add_argument("--embargo-days", "-E", type=int, default=EMBARGO_DAYS)
    parser.add_argument(
        "--no-paths", action="store_true",
        help="Hide the backtest-path subplot.",
    )
    parser.add_argument(
        "--save", "-o", type=str, default=None,
        help="Save figure to this path (e.g. cpcv_splits.png).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    if args.total_samples is not None:
        fig = plot_cpcv_splits(
            total_samples=args.total_samples,
            n_groups=args.n_groups,
            n_test_groups=args.n_test_groups,
            embargo_days=args.embargo_days,
            show_paths=not args.no_paths,
            dpi=args.dpi,
            save_path=args.save,
            title=args.title,
        )
    else:
        fig = plot_alpaca_cpcv(
            n_groups=args.n_groups,
            n_test_groups=args.n_test_groups,
            embargo_days=args.embargo_days,
            show_paths=not args.no_paths,
            dpi=args.dpi,
            save_path=args.save,
            title=args.title,
        )

    if args.save is None:
        plt.show()


if __name__ == "__main__":
    main()
