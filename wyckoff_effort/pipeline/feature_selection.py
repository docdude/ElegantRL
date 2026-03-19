"""
Phase 2: Feature Selection & Denoising

Uses RiskLabAI for:
    - Marcenko-Pastur covariance denoising (RMT)
    - PCA orthogonalization with variance threshold
    - MDI / MDA feature importance ranking
    - Clustered feature importance

Input: tech_ary from Phase 1
Output: denoised/selected tech_ary, feature importance rankings
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from .config import (
    WYCKOFF_FEATURE_COLUMNS,
    FEATURE_SELECTION_DEFAULTS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Covariance Denoising (RMT)
# ─────────────────────────────────────────────────────────────────────────────

def denoise_features(
    tech_ary: np.ndarray,
    bandwidth: float = None,
) -> np.ndarray:
    """
    Denoise the feature covariance matrix using Random Matrix Theory.

    Fits the Marcenko-Pastur distribution to identify noise eigenvalues,
    then reconstructs the covariance matrix using only signal components.

    Parameters
    ----------
    tech_ary : np.ndarray, shape (n_bars, n_features)
    bandwidth : float
        KDE bandwidth for Marcenko-Pastur fitting.

    Returns
    -------
    denoised_cov : np.ndarray, shape (n_features, n_features)
    """
    from RiskLabAI.data.denoise.denoising import denoise_cov

    bandwidth = bandwidth or FEATURE_SELECTION_DEFAULTS["kde_bandwidth"]

    n_bars, n_features = tech_ary.shape
    q = n_bars / n_features

    # Standardize before computing covariance
    means = tech_ary.mean(axis=0)
    stds = tech_ary.std(axis=0)
    stds[stds == 0] = 1.0
    standardized = (tech_ary - means) / stds

    cov_raw = np.cov(standardized, rowvar=False)
    cov_denoised = denoise_cov(cov_raw, q=q, bandwidth=bandwidth)

    logger.info(
        f"Denoised covariance: {n_features}×{n_features}, "
        f"q={q:.1f}, bandwidth={bandwidth}"
    )
    return cov_denoised


# ─────────────────────────────────────────────────────────────────────────────
# PCA Feature Orthogonalization
# ─────────────────────────────────────────────────────────────────────────────

def pca_features(
    tech_ary: np.ndarray,
    feature_names: list = None,
    variance_threshold: float = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Apply PCA to extract orthogonal feature components.

    Parameters
    ----------
    tech_ary : np.ndarray, shape (n_bars, n_features)
    feature_names : list of str
    variance_threshold : float
        Cumulative variance to retain (e.g. 0.95).

    Returns
    -------
    orthogonal_ary : np.ndarray, shape (n_bars, n_components)
    eigen_info : pd.DataFrame with eigenvalues and cumulative variance
    """
    from RiskLabAI.features.feature_importance.orthogonal_features import (
        orthogonal_features,
    )

    variance_threshold = variance_threshold or FEATURE_SELECTION_DEFAULTS["variance_threshold"]
    feature_names = feature_names or WYCKOFF_FEATURE_COLUMNS

    df = pd.DataFrame(tech_ary, columns=feature_names[:tech_ary.shape[1]])
    ortho_df, eigen_df = orthogonal_features(df, variance_threshold=variance_threshold)

    logger.info(
        f"PCA: {tech_ary.shape[1]} features → {ortho_df.shape[1]} components "
        f"({variance_threshold*100:.0f}% variance)"
    )
    return ortho_df.values.astype(np.float32), eigen_df


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance (MDI / MDA)
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_importance_mdi(
    tech_ary: np.ndarray,
    labels: np.ndarray,
    feature_names: list = None,
    sample_weights: np.ndarray = None,
    n_estimators: int = 500,
) -> pd.DataFrame:
    """
    Compute Mean Decrease Impurity feature importance.

    Parameters
    ----------
    tech_ary : np.ndarray, shape (n_bars, n_features)
    labels : np.ndarray, shape (n_bars,)
        Target labels (e.g. from triple barrier or sign of returns).
    feature_names : list of str
    sample_weights : np.ndarray, optional
    n_estimators : int

    Returns
    -------
    importance_df : pd.DataFrame with "Mean" and "StandardDeviation" columns
    """
    from sklearn.ensemble import RandomForestClassifier
    from RiskLabAI.features.feature_importance.feature_importance_mdi import (
        FeatureImportanceMDI,
    )

    feature_names = feature_names or WYCKOFF_FEATURE_COLUMNS
    X = pd.DataFrame(tech_ary, columns=feature_names[:tech_ary.shape[1]])
    y = pd.Series(labels)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=1,
        random_state=42,
        n_jobs=-1,
    )
    strategy = FeatureImportanceMDI(clf)

    kwargs = {}
    if sample_weights is not None:
        kwargs["sample_weight"] = sample_weights

    importance = strategy.compute(X, y, **kwargs)
    importance = importance.sort_values("Mean", ascending=False)

    logger.info(f"MDI top 5: {importance.head().index.tolist()}")
    return importance


def compute_feature_importance_mda(
    tech_ary: np.ndarray,
    labels: np.ndarray,
    feature_names: list = None,
    sample_weights: np.ndarray = None,
    n_estimators: int = 500,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Compute Mean Decrease Accuracy feature importance.

    Parameters
    ----------
    tech_ary : np.ndarray, shape (n_bars, n_features)
    labels : np.ndarray, shape (n_bars,)
    feature_names : list of str
    sample_weights : np.ndarray, optional
    n_estimators : int
    n_splits : int
        Number of CV folds for MDA.

    Returns
    -------
    importance_df : pd.DataFrame with "Mean" and "StandardDeviation" columns
    """
    from sklearn.ensemble import RandomForestClassifier
    from RiskLabAI.features.feature_importance.feature_importance_mda import (
        FeatureImportanceMDA,
    )

    feature_names = feature_names or WYCKOFF_FEATURE_COLUMNS
    X = pd.DataFrame(tech_ary, columns=feature_names[:tech_ary.shape[1]])
    y = pd.Series(labels)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=1,
        random_state=42,
        n_jobs=-1,
    )
    strategy = FeatureImportanceMDA(clf, n_splits=n_splits)

    kwargs = {}
    if sample_weights is not None:
        kwargs["train_sample_weights"] = sample_weights
        kwargs["score_sample_weights"] = sample_weights

    importance = strategy.compute(X, y, **kwargs)
    importance = importance.sort_values("Mean", ascending=False)

    logger.info(f"MDA top 5: {importance.head().index.tolist()}")
    return importance


# ─────────────────────────────────────────────────────────────────────────────
# Select Top-K Features
# ─────────────────────────────────────────────────────────────────────────────

def select_features(
    tech_ary: np.ndarray,
    importance_df: pd.DataFrame,
    top_k: int = None,
    min_importance: float = 0.0,
    feature_names: list = None,
) -> Tuple[np.ndarray, list]:
    """
    Select top features by importance score.

    Parameters
    ----------
    tech_ary : np.ndarray
    importance_df : pd.DataFrame — from MDI or MDA
    top_k : int, optional — keep top K features. If None, use all above min_importance.
    min_importance : float — minimum mean importance threshold.
    feature_names : list of str

    Returns
    -------
    selected_ary : np.ndarray, shape (n_bars, n_selected)
    selected_names : list of str
    """
    feature_names = feature_names or WYCKOFF_FEATURE_COLUMNS

    # Filter by importance
    selected = importance_df[importance_df["Mean"] > min_importance]
    if top_k is not None:
        selected = selected.head(top_k)

    selected_names = selected.index.tolist()

    # Map feature names to column indices
    name_to_idx = {name: i for i, name in enumerate(feature_names[:tech_ary.shape[1]])}
    indices = [name_to_idx[name] for name in selected_names if name in name_to_idx]

    selected_ary = tech_ary[:, indices]
    logger.info(f"Selected {len(indices)} features: {selected_names}")
    return selected_ary, selected_names


# ─────────────────────────────────────────────────────────────────────────────
# Full Phase 2 Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_selection(
    tech_ary: np.ndarray,
    labels: np.ndarray = None,
    feature_names: list = None,
    method: str = "pca",
    variance_threshold: float = None,
    top_k: int = None,
) -> dict:
    """
    Run Phase 2 feature selection pipeline.

    Parameters
    ----------
    tech_ary : np.ndarray
    labels : np.ndarray, optional — needed for MDI/MDA methods
    feature_names : list of str
    method : str — "pca", "mdi", "mda", "denoise", or "all"
    variance_threshold : float — for PCA
    top_k : int — for MDI/MDA selection

    Returns
    -------
    dict with selected arrays, importance rankings, eigen info
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    feature_names = feature_names or WYCKOFF_FEATURE_COLUMNS
    result = {"original_shape": tech_ary.shape}

    if method in ("denoise", "all"):
        result["denoised_cov"] = denoise_features(tech_ary)

    if method in ("pca", "all"):
        ortho_ary, eigen_info = pca_features(
            tech_ary, feature_names, variance_threshold
        )
        result["pca_ary"] = ortho_ary
        result["eigen_info"] = eigen_info

    if method in ("mdi", "all") and labels is not None:
        mdi = compute_feature_importance_mdi(tech_ary, labels, feature_names)
        result["mdi_importance"] = mdi
        if top_k:
            sel_ary, sel_names = select_features(
                tech_ary, mdi, top_k=top_k, feature_names=feature_names
            )
            result["mdi_selected_ary"] = sel_ary
            result["mdi_selected_names"] = sel_names

    if method in ("mda", "all") and labels is not None:
        mda = compute_feature_importance_mda(tech_ary, labels, feature_names)
        result["mda_importance"] = mda
        if top_k:
            sel_ary, sel_names = select_features(
                tech_ary, mda, top_k=top_k, feature_names=feature_names
            )
            result["mda_selected_ary"] = sel_ary
            result["mda_selected_names"] = sel_names

    return result
