"""Leak-free feature preparation and scikit-learn preprocessing pipelines.

All stateful transformations (imputation, scaling, one-hot encoding, and feature
selection) are built so they are fitted exclusively on training data.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
    VALIDATION_SIZE,
)


def split_features_target(
    df: pd.DataFrame,
    selected_features: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features X from binary target y, dropping attack_type and difficulty.

    Args:
        df: Loaded NSL-KDD DataFrame containing feature columns and `label`.
        selected_features: Optional subset of feature columns to keep.

    Returns:
        Tuple `(X, y)` containing only permitted predictive features and binary target.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    cols = list(selected_features) if selected_features is not None else list(FEATURE_COLUMNS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Selected feature(s) missing from DataFrame: {missing}")

    X = df[cols].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def make_train_val_split(
    train_df: pd.DataFrame,
    val_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_SEED,
    selected_features: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/validation split from KDDTrain+ only.

    Args:
        train_df: Official NSL-KDD training DataFrame.
        val_size: Fraction of training rows reserved for validation (default 0.20).
        random_state: Fixed random seed for reproducibility (default 42).
        selected_features: Optional subset of feature columns to retain.

    Returns:
        `X_train, X_val, y_train, y_val`.
    """
    X, y = split_features_target(train_df, selected_features=selected_features)
    return train_test_split(
        X,
        y,
        test_size=val_size,
        stratify=y,
        random_state=random_state,
    )


def find_redundant_features(
    X_train: pd.DataFrame,
    corr_threshold: float = 0.90,
) -> dict[str, list[str]]:
    """Identify constant and highly correlated numeric columns using training data only.

    Args:
        X_train: Training feature DataFrame (never pass validation or test data).
        corr_threshold: Absolute Pearson correlation threshold above which one column
            in each correlated pair is flagged for removal.

    Returns:
        Dictionary with keys:
            - `constant_cols`: Columns with <= 1 unique non-null value.
            - `correlated_cols`: Redundant numeric columns exceeding `corr_threshold`.
            - `drop_cols`: Deduplicated combined list of columns to drop.
            - `kept_features`: Remaining feature columns preserving original order.
    """
    constant_cols = [col for col in X_train.columns if X_train[col].nunique(dropna=True) <= 1]

    numeric_candidates = [
        col for col in X_train.columns if col in NUMERIC_COLUMNS and col not in constant_cols
    ]
    corr_matrix = X_train[numeric_candidates].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    correlated_cols = [
        col for col in upper.columns if (upper[col] > corr_threshold).any()
    ]

    drop_set = set(constant_cols) | set(correlated_cols)
    drop_cols = [col for col in X_train.columns if col in drop_set]
    kept_features = [col for col in X_train.columns if col not in drop_set]

    return {
        "constant_cols": constant_cols,
        "correlated_cols": correlated_cols,
        "drop_cols": drop_cols,
        "kept_features": kept_features,
    }


def build_preprocessor(
    feature_columns: Optional[list[str]] = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for NSL-KDD features.

    Numeric columns receive median imputation (and optional StandardScaler).
    Categorical columns receive most-frequent imputation and OneHotEncoder with
    `handle_unknown="ignore"` so unseen service/flag categories at test or inference
    time never raise errors.

    Args:
        feature_columns: List of active feature names; defaults to all 41 features.
        scale_numeric: Whether to apply StandardScaler to numeric features.

    Returns:
        Unfitted `ColumnTransformer` ready to place inside a modeling `Pipeline`.
    """
    active_cols = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    cat_cols = [c for c in active_cols if c in CATEGORICAL_COLUMNS]
    num_cols = [c for c in active_cols if c in NUMERIC_COLUMNS]

    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
