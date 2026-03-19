"""
Phase 2: Feature Selection & Denoising

    - Marcenko-Pastur covariance denoising (RMT) via numpy eigendecomposition
    - PCA orthogonalization via sklearn
    - MDI / MDA feature importance via sklearn

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

    Uses Marcenko-Pastur upper bound to identify noise eigenvalues,
    then shrinks them to their mean (constant residual eigenvalue).

    Parameters
    ----------
    tech_ary : np.ndarray, shape (n_bars, n_features)
    bandwidth : float
        Unused (kept for API compatibility). MP bound is analytic.

    Returns
    -------
    denoised_cov : np.ndarray, shape (n_features, n_features)
    """
    n_bars, n_features = tech_ary.shape
    q = n_bars / n_features

    # Standardize before computing covariance
    means = tech_ary.mean(axis=0)
    stds = tech_ary.std(axis=0)
    stds[stds == 0] = 1.0
    standardized = (tech_ary - means) / stds

    cov_raw = np.cov(standardized, rowvar=False)

    # Eigendecompose
    eigenvalues, eigenvectors = np.linalg.eigh(cov_raw)
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Marcenko-Pastur upper bound: λ+ = σ²(1 + 1/√q)²
    sigma2 = 1.0  # standardized data
    lambda_plus = sigma2 * (1 + 1.0 / np.sqrt(q)) ** 2

    # Shrink noise eigenvalues (below MP bound) to their average
    noise_mask = eigenvalues < lambda_plus
    if noise_mask.any():
        noise_mean = eigenvalues[noise_mask].mean()
        eigenvalues[noise_mask] = noise_mean

    # Reconstruct
    cov_denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    n_signal = (~noise_mask).sum()
    logger.info(
        f"Denoised covariance: {n_features}×{n_features}, "
        f"q={q:.1f}, λ+={lambda_plus:.3f}, {n_signal} signal eigenvalues"
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
    from sklearn.decomposition import PCA

    variance_threshold = variance_threshold or FEATURE_SELECTION_DEFAULTS["variance_threshold"]

    pca = PCA(n_components=variance_threshold, svd_solver="full")
    orthogonal = pca.fit_transform(tech_ary).astype(np.float32)

    eigen_info = pd.DataFrame({
        "eigenvalue": pca.explained_variance_,
        "variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    })

    logger.info(
        f"PCA: {tech_ary.shape[1]} features → {orthogonal.shape[1]} components "
        f"({variance_threshold*100:.0f}% variance)"
    )
    return orthogonal, eigen_info


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

    feature_names = feature_names or [f"f{i}" for i in range(tech_ary.shape[1])]
    names = feature_names[:tech_ary.shape[1]]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=1,
        random_state=42,
        n_jobs=-1,
    )
    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights
    clf.fit(tech_ary, labels, **fit_kwargs)

    # Per-tree importances for std
    tree_importances = np.array([t.feature_importances_ for t in clf.estimators_])
    importance = pd.DataFrame({
        "Mean": tree_importances.mean(axis=0),
        "StandardDeviation": tree_importances.std(axis=0),
    }, index=names).sort_values("Mean", ascending=False)

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
    from sklearn.inspection import permutation_importance

    feature_names = feature_names or [f"f{i}" for i in range(tech_ary.shape[1])]
    names = feature_names[:tech_ary.shape[1]]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=1,
        random_state=42,
        n_jobs=-1,
    )
    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights
    clf.fit(tech_ary, labels, **fit_kwargs)

    perm = permutation_importance(
        clf, tech_ary, labels, n_repeats=n_splits,
        random_state=42, n_jobs=-1,
    )
    importance = pd.DataFrame({
        "Mean": perm.importances_mean,
        "StandardDeviation": perm.importances_std,
    }, index=names).sort_values("Mean", ascending=False)

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
