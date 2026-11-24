"""Data loading and schema validation utilities for NSL-KDD network connection logs."""

from pathlib import Path
from typing import Union

import pandas as pd

from src.config import (
    ATTACK_CATEGORY_MAP,
    ATTACK_TYPE_COLUMN,
    CATEGORICAL_COLUMNS,
    DIFFICULTY_COLUMN,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    RAW_COLUMNS,
    TARGET_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
)


def validate_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a feature DataFrame matches the 41-column NSL-KDD schema.

    Args:
        df: Input DataFrame containing network connection feature columns.

    Returns:
        A validated copy of the DataFrame with numeric columns coerced to numeric
        dtypes and categorical columns stripped of surrounding whitespace.

    Raises:
        ValueError: If the DataFrame is empty, missing required columns, has an
            unexpected column count, or contains non-numeric text in numeric columns.
    """
    if df is None or df.empty:
        raise ValueError("Input dataset is empty; at least one connection record is required.")

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing {len(missing_cols)} required feature column(s): {missing_cols[:10]}. "
            f"Expected all {len(FEATURE_COLUMNS)} NSL-KDD feature columns: {FEATURE_COLUMNS}"
        )

    validated = df.copy()

    # Validate and coerce numeric columns explicitly so text values fail with the column name
    for col in NUMERIC_COLUMNS:
        series = validated[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        non_null_mask = series.notna() & (series.astype(str).str.strip() != "")
        coerced = pd.to_numeric(series, errors="coerce")
        invalid_mask = non_null_mask & coerced.isna()
        if invalid_mask.any():
            bad_examples = series[invalid_mask].astype(str).unique()[:3].tolist()
            raise ValueError(
                f"Invalid non-numeric value(s) in numeric column '{col}': {bad_examples}"
            )
        validated[col] = coerced

    # Normalize categorical string columns while preserving NaN values
    for col in CATEGORICAL_COLUMNS:
        validated[col] = validated[col].where(
            validated[col].isna(), validated[col].astype(str).str.strip()
        )

    return validated


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target column `label` (0 = normal, 1 = any attack) and `attack_family`.

    Args:
        df: Raw DataFrame containing `attack_type`.

    Returns:
        Copy of DataFrame with integer `label` and string `attack_family` columns added.

    Raises:
        ValueError: If `attack_type` is missing.
    """
    if ATTACK_TYPE_COLUMN not in df.columns:
        raise ValueError(
            f"Cannot create binary label: required column '{ATTACK_TYPE_COLUMN}' is missing."
        )

    out = df.copy()
    cleaned_attack = out[ATTACK_TYPE_COLUMN].astype(str).str.strip().str.lower()
    out[ATTACK_TYPE_COLUMN] = cleaned_attack
    out[TARGET_COLUMN] = (cleaned_attack != "normal").astype(int)
    out["attack_family"] = cleaned_attack.map(ATTACK_CATEGORY_MAP).fillna("Other")
    return out


def load_nsl_kdd(
    filepath: Union[str, Path],
    has_header: bool = False,
    expect_labels: bool = True,
) -> pd.DataFrame:
    """Load an NSL-KDD TXT/CSV file with strict file, shape, and dtype checks.

    Args:
        filepath: Path to KDDTrain+.TXT, KDDTest+.TXT, or a CSV file.
        has_header: Whether the input file includes a header row.
        expect_labels: If True, expects 43 raw columns (41 features + attack_type +
            difficulty). If False, expects 41 feature columns only.

    Returns:
        Validated pandas DataFrame. If `expect_labels=True`, also adds binary `label`
        (0 = normal, 1 = attack) and `attack_family`.

    Raises:
        FileNotFoundError: If `filepath` does not exist.
        ValueError: If the file is empty, malformed, has wrong column count, or
            contains non-numeric values inside numeric columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: '{path}'. "
            "Download NSL-KDD (KDDTrain+.TXT and KDDTest+.TXT) into data/raw/."
        )

    if path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty (0 bytes): '{path}'.")

    expected_columns = RAW_COLUMNS if expect_labels else FEATURE_COLUMNS

    try:
        if has_header:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file contains no parseable records: '{path}'.") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read dataset file '{path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset file '{path}' loaded 0 rows.")

    if not has_header:
        if df.shape[1] != len(expected_columns):
            raise ValueError(
                f"Column count mismatch in '{path}': found {df.shape[1]} columns, "
                f"expected {len(expected_columns)} columns ({expected_columns})."
            )
        df.columns = expected_columns
    else:
        if expect_labels:
            missing = [c for c in expected_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing required column(s) in '{path}': {missing}. "
                    f"Expected columns: {expected_columns}"
                )
        else:
            if df.shape[1] != len(FEATURE_COLUMNS) or any(
                c not in df.columns for c in FEATURE_COLUMNS
            ):
                raise ValueError(
                    f"Column mismatch in '{path}': found {df.shape[1]} columns "
                    f"{list(df.columns)[:10]}; expected {len(FEATURE_COLUMNS)} columns: "
                    f"{FEATURE_COLUMNS}"
                )

    df = validate_feature_dataframe(df)

    if expect_labels:
        # Validate difficulty column is numeric
        diff_coerced = pd.to_numeric(df[DIFFICULTY_COLUMN], errors="coerce")
        if diff_coerced.isna().any():
            raise ValueError(
                f"Invalid non-numeric values found in metadata column '{DIFFICULTY_COLUMN}'."
            )
        df[DIFFICULTY_COLUMN] = diff_coerced.astype(int)
        df = add_binary_label(df)

    return df


def load_train_test_splits(
    train_path: Union[str, Path] = TRAIN_FILE,
    test_path: Union[str, Path] = TEST_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official NSL-KDD training and test datasets.

    Args:
        train_path: Path to KDDTrain+.TXT.
        test_path: Path to KDDTest+.TXT.

    Returns:
        Tuple of `(train_df, test_df)` with validated schema and binary `label`.
    """
    train_df = load_nsl_kdd(train_path, has_header=False, expect_labels=True)
    test_df = load_nsl_kdd(test_path, has_header=False, expect_labels=True)
    return train_df, test_df
