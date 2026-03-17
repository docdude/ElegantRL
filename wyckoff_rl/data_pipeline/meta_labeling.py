"""
Phase 4B: Meta-Labeling Classifier

Trains a secondary classifier (RF or XGBoost) that predicts P(success)
for each Wyckoff signal event. Uses RiskLabAI's CPCV for purged
cross-validation to prevent leakage.

Input: labeled signals from signal_extraction.py
Output: trained meta-label model, probability predictions, feature importance
"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from .config import (
    WYCKOFF_FEATURE_COLUMNS,
    META_LABEL_DEFAULTS,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


def _build_feature_matrix(
    labeled_df: pd.DataFrame,
    feature_columns: list = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build X (features) and y (labels) from labeled signal events.

    X = Wyckoff feature snapshot at each signal bar
    y = binary label (1 = TP hit, 0 = SL/vertical hit)
    """
    feature_columns = feature_columns or WYCKOFF_FEATURE_COLUMNS
    available = [c for c in feature_columns if c in labeled_df.columns]

    X = labeled_df[available].astype(np.float32).reset_index(drop=True)
    X = X.fillna(0)
    y = labeled_df["label"].astype(int).reset_index(drop=True)

    return X, y


def _build_times_series(labeled_df: pd.DataFrame) -> pd.Series:
    """
    Build a times Series for purged CV: index=event start (positional),
    value=event end (positional). Maps bar_idx/exit_bar into unique
    positional space [0..N-1] so CPCV can use get_indexer (requires unique index).
    """
    bar_idxs = labeled_df["bar_idx"].values
    exit_bars = labeled_df["exit_bar"].values
    # Map exit_bar to the positional index of the last event starting at or before it
    end_positions = np.searchsorted(bar_idxs, exit_bars, side="right") - 1
    end_positions = np.clip(end_positions, 0, len(bar_idxs) - 1)
    return pd.Series(end_positions, index=pd.RangeIndex(len(bar_idxs)))


# ─────────────────────────────────────────────────────────────────────────────
# CPCV-Purged Training
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_label_model(
    labeled_df: pd.DataFrame,
    feature_columns: list = None,
    sample_weights: np.ndarray = None,
    model_type: str = "rf",
    n_estimators: int = None,
    max_depth: int = None,
    min_samples_leaf: int = None,
    n_splits: int = None,
    n_test_groups: int = None,
    embargo_pct: float = None,
) -> dict:
    """
    Train meta-label classifier with CPCV cross-validation.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        From signal_extraction.apply_triple_barrier().
    feature_columns : list
    sample_weights : np.ndarray
    model_type : str — "rf" or "xgb"
    n_estimators, max_depth, min_samples_leaf : int
    n_splits, n_test_groups : int — CPCV params
    embargo_pct : float

    Returns
    -------
    dict with:
        - model: trained classifier
        - cv_results: per-fold metrics
        - oos_predictions: out-of-sample probabilities
        - feature_importance: MDI importance from all folds
    """
    defaults = META_LABEL_DEFAULTS
    n_estimators = n_estimators or defaults["n_estimators"]
    max_depth = max_depth or defaults["max_depth"]
    min_samples_leaf = min_samples_leaf or defaults["min_samples_leaf"]
    n_splits = n_splits or defaults["n_splits"]
    n_test_groups = n_test_groups or defaults["n_test_groups"]
    embargo_pct = embargo_pct or defaults["embargo_pct"]

    X, y = _build_feature_matrix(labeled_df, feature_columns)
    times = _build_times_series(labeled_df)

    logger.info(
        f"Training meta-label ({model_type}): "
        f"{len(X)} samples, {X.shape[1]} features, "
        f"pos_rate={y.mean():.3f}"
    )

    # Build classifier
    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    elif model_type == "xgb":
        from xgboost import XGBClassifier
        scale = (y == 0).sum() / max((y == 1).sum(), 1)
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_samples_leaf,
            scale_pos_weight=scale,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # CPCV splits
    from RiskLabAI.backtest.validation.combinatorial_purged import CombinatorialPurged
    cpcv = CombinatorialPurged(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        times=times,
        embargo=embargo_pct,
    )

    cv_results = []
    oos_predictions = np.full(len(X), np.nan)
    feature_importances = []

    for fold_idx, (train_idx, test_idx) in enumerate(cpcv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights[train_idx]

        clf_fold = _clone_classifier(clf)
        clf_fold.fit(X_train, y_train, **fit_kwargs)

        # OOS predictions
        proba = clf_fold.predict_proba(X_test)[:, 1]
        oos_predictions[test_idx] = proba

        # Fold metrics
        from sklearn.metrics import log_loss, precision_score, recall_score, f1_score
        preds = (proba > 0.5).astype(int)
        fold_result = {
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "log_loss": log_loss(y_test, proba, labels=[0, 1]),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "pos_rate_train": y_train.mean(),
            "pos_rate_test": y_test.mean(),
        }
        cv_results.append(fold_result)

        # Feature importance
        if hasattr(clf_fold, "feature_importances_"):
            fi = pd.Series(clf_fold.feature_importances_, index=X.columns)
            feature_importances.append(fi)

        logger.info(
            f"  Fold {fold_idx}: F1={fold_result['f1']:.3f}, "
            f"Prec={fold_result['precision']:.3f}, "
            f"Rec={fold_result['recall']:.3f}"
        )

    # Train final model on all data
    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights
    clf.fit(X, y, **fit_kwargs)

    # Aggregate feature importance
    if feature_importances:
        fi_df = pd.DataFrame(feature_importances)
        fi_agg = pd.DataFrame({
            "Mean": fi_df.mean(),
            "Std": fi_df.std(),
        }).sort_values("Mean", ascending=False)
    else:
        fi_agg = pd.DataFrame()

    cv_df = pd.DataFrame(cv_results)
    logger.info(
        f"Meta-label CV summary: "
        f"F1={cv_df['f1'].mean():.3f}±{cv_df['f1'].std():.3f}, "
        f"LogLoss={cv_df['log_loss'].mean():.3f}"
    )

    return {
        "model": clf,
        "cv_results": cv_df,
        "oos_predictions": oos_predictions,
        "feature_importance": fi_agg,
        "feature_columns": X.columns.tolist(),
    }


def _clone_classifier(clf):
    """Clone a classifier for per-fold training."""
    from sklearn.base import clone
    return clone(clf)


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_meta_label_model(result: dict, path: str = None) -> str:
    """Save trained meta-label model and results."""
    path = path or os.path.join(OUTPUT_DIR, "meta_label_model.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_data = {
        "model": result["model"],
        "feature_columns": result["feature_columns"],
        "cv_results": result["cv_results"],
        "feature_importance": result["feature_importance"],
    }

    with open(path, "wb") as f:
        pickle.dump(save_data, f)

    logger.info(f"Saved meta-label model: {path}")
    return path


def load_meta_label_model(path: str) -> dict:
    """Load a saved meta-label model."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict_signal_probability(
    model_result: dict,
    signals_df: pd.DataFrame,
) -> np.ndarray:
    """
    Predict P(success) for new signal events.

    Parameters
    ----------
    model_result : dict — from train_meta_label_model() or load_meta_label_model()
    signals_df : pd.DataFrame — new signals with feature columns

    Returns
    -------
    probabilities : np.ndarray, shape (n_signals,)
    """
    model = model_result["model"]
    feature_cols = model_result["feature_columns"]

    X = signals_df[feature_cols].fillna(0).astype(np.float32)
    probabilities = model.predict_proba(X)[:, 1]

    return probabilities
