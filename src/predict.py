"""Inference and input-validation utilities for the Network Intrusion Detection Prototype.

Provides a clean, reusable inference layer over the persisted scikit-learn Pipeline
(`models/model.joblib`) for single connection records, pandas DataFrames, and CSV files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, IO, Union

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ATTACK_TYPE_COLUMN,
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DIFFICULTY_COLUMN,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    RAW_COLUMNS,
    TARGET_COLUMN,
)
from src.load_data import validate_feature_dataframe

OPTIONAL_METADATA_COLUMNS: frozenset[str] = frozenset(
    {ATTACK_TYPE_COLUMN, DIFFICULTY_COLUMN, TARGET_COLUMN, "attack_family"}
)

LABEL_NORMAL: str = "Normal"
LABEL_ATTACK: str = "Suspicious (Attack)"


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Any:
    """Load the serialized scikit-learn Pipeline from disk.

    Args:
        model_path: Path to the saved `.joblib` model artifact.

    Returns:
        Fitted scikit-learn estimator exposing `predict_proba`.

    Raises:
        FileNotFoundError: If `model_path` does not exist.
        RuntimeError: If the model cannot be deserialized or does not support
            probability predictions.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at '{path}'. "
            "Run `python -m src.train` to generate models/model.joblib."
        )

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifact from '{path}': {exc}") from exc

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"Loaded artifact from '{path}' does not expose `predict_proba`."
        )

    return model


def validate_threshold(threshold: float) -> float:
    """Validate that the classification decision threshold lies within [0.0, 1.0].

    Args:
        threshold: Probability cutoff for flagging a connection as suspicious.

    Returns:
        Validated threshold as a float.

    Raises:
        ValueError: If `threshold` is boolean, non-numeric, NaN, or outside [0.0, 1.0].
    """
    if isinstance(threshold, bool):
        raise ValueError("Decision threshold must be a float between 0.0 and 1.0, not a boolean.")

    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Decision threshold must be numeric between 0.0 and 1.0; got {threshold!r}."
        ) from exc

    if np.isnan(value) or not (0.0 <= value <= 1.0):
        raise ValueError(
            f"Decision threshold must be between 0.0 and 1.0 (inclusive); got {threshold!r}."
        )

    return value


def prepare_inference_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and align an input DataFrame to the exact 41-column NSL-KDD schema.

    Known optional NSL-KDD metadata columns (`attack_type`, `difficulty`, `label`,
    `attack_family`) are stripped automatically if a user supplies a labeled dataset.
    Any missing required columns, unknown extra columns, duplicate column names,
    non-numeric text in numeric columns, or infinite numeric values raise a clear
    `ValueError`. Missing (`NaN`) values are preserved so the fitted pipeline's
    `SimpleImputer` handles them cleanly.

    Args:
        df: Incoming pandas DataFrame of connection records.

    Returns:
        Validated copy of the DataFrame containing the exact 41 `FEATURE_COLUMNS`
        in canonical order.

    Raises:
        ValueError: If the DataFrame is None, empty, malformed, or fails schema checks.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame containing connection records.")

    if df.empty:
        raise ValueError("Input dataset is empty; at least one connection record is required.")

    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
        raise ValueError(
            f"Input dataset contains duplicate column name(s): {dup_cols}. "
            f"Expected exactly {len(FEATURE_COLUMNS)} unique NSL-KDD feature columns: "
            f"{FEATURE_COLUMNS}"
        )

    allowed_columns = set(FEATURE_COLUMNS) | OPTIONAL_METADATA_COLUMNS
    unexpected_cols = [col for col in df.columns if col not in allowed_columns]
    if unexpected_cols:
        raise ValueError(
            f"Unexpected column(s) found ({len(unexpected_cols)}): {unexpected_cols[:10]}. "
            f"Expected {len(FEATURE_COLUMNS)} NSL-KDD feature columns: {FEATURE_COLUMNS}"
        )

    # Strip known optional metadata/target columns if present (e.g., labeled NSL-KDD export)
    cols_to_drop = [col for col in df.columns if col in OPTIONAL_METADATA_COLUMNS]
    working = df.drop(columns=cols_to_drop) if cols_to_drop else df.copy()

    missing_cols = [col for col in FEATURE_COLUMNS if col not in working.columns]
    if missing_cols or working.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Column mismatch: received {working.shape[1]} feature column(s), "
            f"missing {len(missing_cols)} required column(s) {missing_cols[:10]}. "
            f"Expected all {len(FEATURE_COLUMNS)} NSL-KDD feature columns: {FEATURE_COLUMNS}"
        )

    validated = validate_feature_dataframe(working)

    # Guard against +/-inf values that scikit-learn estimators reject
    for col in NUMERIC_COLUMNS:
        if np.isinf(validated[col].to_numpy(dtype=float, na_value=0.0)).any():
            raise ValueError(
                f"Infinite value(s) detected in numeric column '{col}'. "
                "Replace infinities with finite values or NaN."
            )

    return validated[FEATURE_COLUMNS].copy()


def predict_dataframe(
    df: pd.DataFrame,
    model: Any = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> pd.DataFrame:
    """Score a DataFrame of network connection records at the given decision threshold.

    Args:
        df: DataFrame containing the 41 NSL-KDD feature columns (optional NSL-KDD
            metadata columns are ignored safely).
        model: Optional pre-loaded scikit-learn pipeline. Loads `DEFAULT_MODEL_PATH`
            if omitted.
        threshold: Probability cutoff in `[0.0, 1.0]` for classifying a connection
            as `"Suspicious (Attack)"` (default `0.50`).

    Returns:
        DataFrame with prediction columns (`prediction`, `predicted_label`,
        `attack_probability`, `decision_threshold`) followed by the validated
        41 feature columns, preserving the input index.
    """
    thr = validate_threshold(threshold)
    features = prepare_inference_dataframe(df)
    active_model = load_model() if model is None else model

    probas = active_model.predict_proba(features)[:, 1].astype(float)
    predicted_labels = (probas >= thr).astype(int)
    predictions = np.where(predicted_labels == 1, LABEL_ATTACK, LABEL_NORMAL)

    result = pd.DataFrame(
        {
            "prediction": predictions,
            "predicted_label": predicted_labels,
            "attack_probability": probas,
            "decision_threshold": thr,
        },
        index=features.index,
    )
    return pd.concat([result, features], axis=1)


def predict_single_record(
    record: dict[str, Any],
    model: Any = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Score a single network connection record provided as a dictionary.

    Args:
        record: Mapping of the 41 NSL-KDD feature names to values.
        model: Optional pre-loaded scikit-learn pipeline.
        threshold: Probability cutoff in `[0.0, 1.0]` (default `0.50`).

    Returns:
        Dictionary containing `prediction`, `predicted_label`, `attack_probability`,
        and `threshold`.

    Raises:
        ValueError: If `record` is not a non-empty dictionary or fails validation.
    """
    if not isinstance(record, dict) or len(record) == 0:
        raise ValueError(
            "Single record must be a non-empty dictionary mapping NSL-KDD feature names to values."
        )

    single_df = pd.DataFrame([record])
    scored = predict_dataframe(single_df, model=model, threshold=threshold)
    row = scored.iloc[0]
    return {
        "prediction": str(row["prediction"]),
        "predicted_label": int(row["predicted_label"]),
        "attack_probability": float(row["attack_probability"]),
        "threshold": float(row["decision_threshold"]),
    }


def read_inference_csv(
    source: Union[str, Path, IO[Any]],
    has_header: bool = True,
) -> pd.DataFrame:
    """Safely read a CSV file path or uploaded file-like stream for inference.

    Args:
        source: Path to a CSV/TXT file or a binary/text file-like stream.
        has_header: Whether the CSV includes a header row (default `True`).

    Returns:
        Raw pandas DataFrame ready for `prepare_inference_dataframe`.

    Raises:
        FileNotFoundError: If a path-like `source` does not exist.
        ValueError: If the file is empty, unreadable, or has an unsupported
            headerless column count.
    """
    source_label = str(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: '{path}'.")
        if path.stat().st_size == 0:
            raise ValueError(f"CSV file is empty (0 bytes): '{path}'.")
        read_target: Any = path
    else:
        source_label = getattr(source, "name", "uploaded file")
        if hasattr(source, "seek"):
            source.seek(0)
        read_target = source

    try:
        df = pd.read_csv(read_target, header=0 if has_header else None)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"CSV file '{source_label}' is empty or contains no parseable data."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file '{source_label}': {exc}") from exc

    if df.empty:
        raise ValueError(f"CSV file '{source_label}' contains 0 connection records.")

    if not has_header:
        if df.shape[1] == len(FEATURE_COLUMNS):
            df.columns = FEATURE_COLUMNS
        elif df.shape[1] == len(RAW_COLUMNS):
            df.columns = RAW_COLUMNS
        else:
            raise ValueError(
                f"Headerless file '{source_label}' has {df.shape[1]} column(s); expected "
                f"{len(FEATURE_COLUMNS)} feature columns or {len(RAW_COLUMNS)} raw NSL-KDD columns: "
                f"{FEATURE_COLUMNS}"
            )

    return df


def predict_csv(
    filepath: Union[str, Path, IO[Any]],
    model: Any = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    has_header: bool = True,
) -> pd.DataFrame:
    """Load and score a CSV file (or uploaded stream) of network connection records.

    Args:
        filepath: Path to a CSV/TXT file or a file-like object.
        model: Optional pre-loaded scikit-learn pipeline.
        threshold: Probability cutoff in `[0.0, 1.0]` (default `0.50`).
        has_header: Whether the file includes a header row (default `True`).

    Returns:
        Scored results DataFrame from `predict_dataframe`.
    """
    raw_df = read_inference_csv(filepath, has_header=has_header)
    return predict_dataframe(raw_df, model=model, threshold=threshold)
