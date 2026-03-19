"""
Backtest Evaluation: PBO + PSR

Uses RiskLabAI for:
    - Probability of Backtest Overfitting (PBO)
    - Probabilistic Sharpe Ratio (PSR)
    - Backtest statistics (Sharpe, Sortino, drawdowns)

Input: strategy returns (from CPCV paths or meta-label backtest)
Output: PBO score, PSR, performance metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute annualized Sharpe ratio."""
    from RiskLabAI.backtest.backtest_statistics import sharpe_ratio
    sr = sharpe_ratio(returns, risk_free)
    return float(sr)


def compute_sortino(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute Sortino ratio (penalizes downside only)."""
    excess = returns - risk_free
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std())


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from returns."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min())


def compute_calmar(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Compute Calmar ratio (annualized return / max drawdown)."""
    mdd = compute_max_drawdown(returns)
    if mdd == 0:
        return 0.0
    ann_return = np.mean(returns) * 252  # approximate
    return float(ann_return / abs(mdd))


# ─────────────────────────────────────────────────────────────────────────────
# Probabilistic Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_psr(
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Compute Probabilistic Sharpe Ratio.

    PSR = P(true SR > benchmark SR | observed track record).
    Adjusts for track record length, skewness, and kurtosis.

    Parameters
    ----------
    returns : np.ndarray
        Strategy returns.
    benchmark_sharpe : float
        Benchmark Sharpe ratio to beat.

    Returns
    -------
    psr : float, range [0, 1]
    """
    from RiskLabAI.backtest.probabilistic_sharpe_ratio import probabilistic_sharpe_ratio
    from scipy.stats import skew, kurtosis

    observed_sr = compute_sharpe(returns)
    n = len(returns)
    sk = float(skew(returns))
    ku = float(kurtosis(returns, fisher=False))  # excess=False → raw kurtosis

    psr = probabilistic_sharpe_ratio(
        observed_sharpe_ratio=observed_sr,
        benchmark_sharpe_ratio=benchmark_sharpe,
        number_of_returns=n,
        skewness_of_returns=sk,
        kurtosis_of_returns=ku,
    )
    return float(psr)


# ─────────────────────────────────────────────────────────────────────────────
# Probability of Backtest Overfitting
# ─────────────────────────────────────────────────────────────────────────────

def compute_pbo(
    strategy_returns: np.ndarray,
    n_partitions: int = 16,
    risk_free: float = 0.0,
    n_jobs: int = -1,
) -> Tuple[float, np.ndarray]:
    """
    Compute Probability of Backtest Overfitting.

    PBO = frequency with which the best IS strategy underperforms
    median OOS. Lower PBO is better (< 0.3 target).

    Parameters
    ----------
    strategy_returns : np.ndarray, shape (T, N)
        Matrix: T time periods × N strategy variants.
        Each column is a different hyperparameter configuration
        or strategy variant.
    n_partitions : int
        Number of partitions (must be even).
    risk_free : float
        Risk-free return.
    n_jobs : int
        Parallel jobs (-1 = all cores).

    Returns
    -------
    pbo : float, range [0, 1]
    logit_values : np.ndarray — logit distribution
    """
    from RiskLabAI.backtest.probability_of_backtest_overfitting import (
        probability_of_backtest_overfitting,
    )

    pbo, logits = probability_of_backtest_overfitting(
        performances=strategy_returns,
        n_partitions=n_partitions,
        risk_free_return=risk_free,
        n_jobs=n_jobs,
    )

    logger.info(f"PBO = {pbo:.3f} ({n_partitions} partitions, {strategy_returns.shape[1]} strategies)")
    return float(pbo), logits


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation Report
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_strategy(
    returns: np.ndarray,
    benchmark_sharpe: float = 0.0,
    strategy_variants: np.ndarray = None,
    n_partitions: int = 16,
) -> dict:
    """
    Generate a full evaluation report for a strategy.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Primary strategy returns.
    benchmark_sharpe : float
        Benchmark Sharpe for PSR.
    strategy_variants : np.ndarray, shape (T, N), optional
        Multiple strategy variants for PBO calculation.
    n_partitions : int
        For PBO.

    Returns
    -------
    dict with all metrics
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = {
        "sharpe": compute_sharpe(returns),
        "sortino": compute_sortino(returns),
        "calmar": compute_calmar(returns),
        "max_drawdown": compute_max_drawdown(returns),
        "total_return": float(np.prod(1 + returns) - 1),
        "n_observations": len(returns),
        "psr": compute_psr(returns, benchmark_sharpe),
    }

    if strategy_variants is not None and strategy_variants.shape[1] > 1:
        pbo, logits = compute_pbo(strategy_variants, n_partitions)
        report["pbo"] = pbo
        report["logit_distribution"] = logits

    logger.info(
        f"Evaluation: Sharpe={report['sharpe']:.3f}, "
        f"Sortino={report['sortino']:.3f}, "
        f"MaxDD={report['max_drawdown']:.3%}, "
        f"PSR={report['psr']:.3f}"
        + (f", PBO={report.get('pbo', 'N/A')}" if "pbo" in report else "")
    )
    return report
