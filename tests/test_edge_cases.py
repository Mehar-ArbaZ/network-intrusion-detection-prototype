"""Automated edge-case and reliability test suite for the NIDS prototype.

Covers all five mandatory blueprint edge cases plus threshold sensitivity,
sample-batch scoring, metadata-column handling, and optional KDDTest+ unseen-data
verification.

Run:
    pytest -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_MODEL_PATH, FEATURE_COLUMNS, SAMPLE_FILE, TEST_FILE
from src.load_data import load_nsl_kdd
from src.predict import (
    LABEL_ATTACK,
    LABEL_NORMAL,
    load_model,
    predict_csv,
    predict_dataframe,
    predict_single_record,
)


@pytest.fixture(scope="module")
def trained_model():
    """Load the persisted Stage 2 model artifact once for the test module."""
    assert DEFAULT_MODEL_PATH.exists(), f"Missing trained model at {DEFAULT_MODEL_PATH}"
    return load_model(DEFAULT_MODEL_PATH)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Load the 100-row verified NSL-KDD sample dataset."""
    assert SAMPLE_FILE.exists(), f"Missing sample CSV at {SAMPLE_FILE}"
    return pd.read_csv(SAMPLE_FILE)


# ---------------------------------------------------------------------------
# Mandatory Blueprint Edge Cases (1 - 5)
# ---------------------------------------------------------------------------


def test_edge_case_1_missing_numeric_nan_handled(trained_model, sample_df):
    """Edge Case 1: NaN in numeric column is imputed cleanly without crashing."""
    df_with_nan = sample_df.head(5).copy()
    df_with_nan.loc[df_with_nan.index[0], "src_bytes"] = np.nan
    df_with_nan.loc[df_with_nan.index[1], "serror_rate"] = np.nan

    results = predict_dataframe(df_with_nan, model=trained_model, threshold=0.50)

    assert len(results) == 5
    assert results["attack_probability"].notna().all()
    assert ((results["attack_probability"] >= 0.0) & (results["attack_probability"] <= 1.0)).all()
    assert set(results["prediction"]).issubset({LABEL_NORMAL, LABEL_ATTACK})


def test_edge_case_2_unseen_service_category_handled(trained_model, sample_df):
    """Edge Case 2: Unseen service category passes OneHotEncoder(handle_unknown='ignore')."""
    record = sample_df.iloc[0].to_dict()
    record["service"] = "brand_new_unseen_service_xyz"

    outcome = predict_single_record(record, model=trained_model, threshold=0.50)

    assert outcome["prediction"] in {LABEL_NORMAL, LABEL_ATTACK}
    assert outcome["predicted_label"] in {0, 1}
    assert 0.0 <= outcome["attack_probability"] <= 1.0


def test_edge_case_3_wrong_column_count_raises_readable_error(trained_model, sample_df, tmp_path):
    """Edge Case 3: CSV with missing or unexpected extra columns raises clear ValueError."""
    # Too few columns (dropped 3 required columns)
    too_few_path = tmp_path / "too_few_cols.csv"
    sample_df.head(3).drop(columns=["duration", "src_bytes", "service"]).to_csv(
        too_few_path, index=False
    )

    with pytest.raises(ValueError) as exc_few:
        predict_csv(too_few_path, model=trained_model)
    msg_few = str(exc_few.value)
    assert "41" in msg_few
    assert "duration" in msg_few

    # Unexpected extra column
    extra_col_path = tmp_path / "extra_col.csv"
    df_extra = sample_df.head(3).copy()
    df_extra["unexpected_custom_col"] = 123
    df_extra.to_csv(extra_col_path, index=False)

    with pytest.raises(ValueError) as exc_extra:
        predict_csv(extra_col_path, model=trained_model)
    msg_extra = str(exc_extra.value)
    assert "unexpected_custom_col" in msg_extra
    assert "41" in msg_extra


def test_edge_case_4_empty_and_nonexistent_file_raise_clear_errors(trained_model, tmp_path):
    """Edge Case 4: Empty (0-byte) file raises ValueError; missing file raises FileNotFoundError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        predict_csv(empty_file, model=trained_model)

    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        predict_csv(missing_file, model=trained_model)

    # Also verify load_nsl_kdd directly for empty and missing files
    with pytest.raises(ValueError, match="empty"):
        load_nsl_kdd(empty_file, has_header=True, expect_labels=False)
    with pytest.raises(FileNotFoundError, match="not found"):
        load_nsl_kdd(missing_file, has_header=True, expect_labels=False)


def test_edge_case_5_non_numeric_text_in_numeric_column_rejected(trained_model, sample_df, tmp_path):
    """Edge Case 5: Text inside a numeric column fails with ValueError naming the column."""
    bad_df = sample_df.head(3).copy()
    bad_df["src_bytes"] = bad_df["src_bytes"].astype(object)
    bad_df.loc[bad_df.index[0], "src_bytes"] = "not_a_number"

    bad_csv = tmp_path / "bad_numeric.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError) as exc_info:
        predict_csv(bad_csv, model=trained_model)

    assert "src_bytes" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Additional Reliability & Unseen-Data Verification Tests
# ---------------------------------------------------------------------------


def test_sample_batch_scoring_returns_100_valid_predictions(trained_model):
    """Scoring the 100-row sample CSV yields 100 valid probabilities and labels."""
    results = predict_csv(SAMPLE_FILE, model=trained_model, threshold=0.50)

    assert len(results) == 100
    assert list(results.columns[:4]) == [
        "prediction",
        "predicted_label",
        "attack_probability",
        "decision_threshold",
    ]
    assert list(results.columns[4:]) == FEATURE_COLUMNS
    assert results["attack_probability"].notna().all()
    assert ((results["attack_probability"] >= 0.0) & (results["attack_probability"] <= 1.0)).all()
    assert set(results["predicted_label"]).issubset({0, 1})


def test_threshold_sensitivity_and_bounds_validation(trained_model, sample_df):
    """Lowering threshold never decreases flagged count; out-of-range thresholds fail."""
    low_thr = predict_dataframe(sample_df, model=trained_model, threshold=0.10)
    high_thr = predict_dataframe(sample_df, model=trained_model, threshold=0.80)

    assert int(low_thr["predicted_label"].sum()) >= int(high_thr["predicted_label"].sum())

    for invalid_thr in (-0.01, 1.01, float("nan"), True):
        with pytest.raises(ValueError):
            predict_dataframe(sample_df.head(2), model=trained_model, threshold=invalid_thr)


def test_optional_nsl_kdd_metadata_columns_stripped_cleanly(trained_model, sample_df):
    """Labeled NSL-KDD CSVs carrying attack_type/difficulty are accepted and stripped."""
    labeled_df = sample_df.head(5).copy()
    labeled_df["attack_type"] = "normal"
    labeled_df["difficulty"] = 21

    results = predict_dataframe(labeled_df, model=trained_model, threshold=0.50)

    assert len(results) == 5
    assert "attack_type" not in results.columns
    assert "difficulty" not in results.columns
    assert list(results.columns[4:]) == FEATURE_COLUMNS


@pytest.mark.skipif(
    not TEST_FILE.exists(),
    reason="Official KDDTest+.TXT not present locally in data/raw/.",
)
def test_unseen_official_kddtest_scoring(trained_model):
    """Verify saved model artifact scores the held-out KDDTest+ benchmark cleanly."""
    test_df = load_nsl_kdd(TEST_FILE, has_header=False, expect_labels=True)
    results = predict_dataframe(test_df, model=trained_model, threshold=0.50)

    assert len(results) == len(test_df)
    assert results["attack_probability"].notna().all()
    assert ((results["attack_probability"] >= 0.0) & (results["attack_probability"] <= 1.0)).all()
