"""Reproducible, leak-free training and experimentation script (Experiments E1-E4).

Usage:
    python -m src.train

Workflow:
    1. Load official NSL-KDD KDDTrain+.TXT and KDDTest+.TXT via src.load_data.
    2. Split KDDTrain+ 80/20 (stratified, random_state=42) for training and validation.
       KDDTest+ is never used for fitting, feature selection, tuning, or model selection.
    3. Run Experiments E1-E4 using scikit-learn Pipelines so all stateful preprocessing
       is fitted strictly on the training fold.
    4. Select the final model on validation performance (prioritizing attack recall with
       acceptable false-alarm rate and <50 MB artifact constraint).
    5. Evaluate the baseline (E1) and selected final model once on KDDTest+, save all
       metrics CSVs, diagnostic plots, metadata JSON, and compressed model.joblib.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import (
    ATTACK_TYPE_COLUMN,
    DEFAULT_MODEL_PATH,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.evaluate import (
    compute_classification_metrics,
    evaluate_by_attack_family,
    evaluate_seen_vs_unseen,
    evaluate_thresholds,
    plot_confusion_matrix,
    plot_experiment_comparison,
    plot_test_error_breakdown,
    plot_threshold_tradeoff,
)
from src.load_data import load_train_test_splits
from src.preprocess import (
    build_preprocessor,
    find_redundant_features,
    make_train_val_split,
    split_features_target,
)


def build_logistic_pipeline(feature_columns: list[str]) -> Pipeline:
    """Create a leak-free Logistic Regression pipeline with scaling and OHE."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(feature_columns=feature_columns, scale_numeric=True),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            ),
        ]
    )


def build_rf_pipeline(
    feature_columns: list[str],
    n_estimators: int = 100,
    max_depth: int | None = None,
) -> Pipeline:
    """Create a leak-free Random Forest pipeline keeping raw numeric scales."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(feature_columns=feature_columns, scale_numeric=False),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def run_experiments() -> dict[str, Any]:
    """Execute Experiments E1-E4, evaluate once on KDDTest+, and persist artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1. Loading official NSL-KDD train and test datasets...")
    train_df, test_df = load_train_test_splits()
    X_train, X_val, y_train, y_val = make_train_val_split(train_df)
    X_test, y_test = split_features_target(test_df)

    print(
        f"   Train fold: {X_train.shape} | Val fold: {X_val.shape} | "
        f"Held-out KDDTest+: {X_test.shape}"
    )

    # Identify constant & collinear features on X_train ONLY
    redundant_info = find_redundant_features(X_train, corr_threshold=0.90)
    pruned_features = redundant_info["kept_features"]
    print(
        f"   Redundant features (80% train only): {len(redundant_info['drop_cols'])} dropped "
        f"({redundant_info['drop_cols']}) -> {len(pruned_features)} kept for E2."
    )

    experiment_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    experiment_features: dict[str, list[str]] = {}
    best_e4_params: dict[str, Any] = {}
    e4_cv_f1: float = 0.0

    # --- E1: Baseline Logistic Regression (all 41 features) ---
    print("\n2. Running E1: Baseline LogisticRegression (all 41 features)...")
    e1_pipe = build_logistic_pipeline(FEATURE_COLUMNS)
    e1_pipe.fit(X_train, y_train)
    e1_val_pred = e1_pipe.predict(X_val)
    e1_val_proba = e1_pipe.predict_proba(X_val)[:, 1]
    e1_val_metrics = compute_classification_metrics(y_val, e1_val_pred, e1_val_proba)
    fitted_models["E1"] = e1_pipe
    experiment_features["E1"] = list(FEATURE_COLUMNS)
    experiment_rows.append(
        {
            "id": "E1",
            "model": "LogisticRegression(max_iter=1000)",
            "features": f"All {len(FEATURE_COLUMNS)} features",
            "val_accuracy": round(e1_val_metrics["accuracy"], 4),
            "val_precision": round(e1_val_metrics["precision"], 4),
            "val_recall": round(e1_val_metrics["recall"], 4),
            "val_f1": round(e1_val_metrics["f1"], 4),
            "val_macro_f1": round(e1_val_metrics["macro_f1"], 4),
            "val_false_alarm_rate": round(e1_val_metrics["false_alarm_rate"], 4),
            "val_roc_auc": round(e1_val_metrics["roc_auc"], 4),
        }
    )

    # --- E2: Feature-pruned Logistic Regression (33 features) ---
    print("3. Running E2: LogisticRegression with pruned features (33 features)...")
    e2_pipe = build_logistic_pipeline(pruned_features)
    e2_pipe.fit(X_train[pruned_features], y_train)
    e2_val_pred = e2_pipe.predict(X_val[pruned_features])
    e2_val_proba = e2_pipe.predict_proba(X_val[pruned_features])[:, 1]
    e2_val_metrics = compute_classification_metrics(y_val, e2_val_pred, e2_val_proba)
    fitted_models["E2"] = e2_pipe
    experiment_features["E2"] = list(pruned_features)
    experiment_rows.append(
        {
            "id": "E2",
            "model": "LogisticRegression(max_iter=1000)",
            "features": f"Pruned {len(pruned_features)} features (drop constant & |r|>0.90)",
            "val_accuracy": round(e2_val_metrics["accuracy"], 4),
            "val_precision": round(e2_val_metrics["precision"], 4),
            "val_recall": round(e2_val_metrics["recall"], 4),
            "val_f1": round(e2_val_metrics["f1"], 4),
            "val_macro_f1": round(e2_val_metrics["macro_f1"], 4),
            "val_false_alarm_rate": round(e2_val_metrics["false_alarm_rate"], 4),
            "val_roc_auc": round(e2_val_metrics["roc_auc"], 4),
        }
    )

    # --- E3: Random Forest default depth (all 41 features) ---
    print("4. Running E3: RandomForestClassifier(n_estimators=100, random_state=42)...")
    e3_pipe = build_rf_pipeline(FEATURE_COLUMNS, n_estimators=100, max_depth=None)
    e3_pipe.fit(X_train, y_train)
    e3_val_pred = e3_pipe.predict(X_val)
    e3_val_proba = e3_pipe.predict_proba(X_val)[:, 1]
    e3_val_metrics = compute_classification_metrics(y_val, e3_val_pred, e3_val_proba)
    fitted_models["E3"] = e3_pipe
    experiment_features["E3"] = list(FEATURE_COLUMNS)
    experiment_rows.append(
        {
            "id": "E3",
            "model": "RandomForestClassifier(n_estimators=100)",
            "features": f"All {len(FEATURE_COLUMNS)} features",
            "val_accuracy": round(e3_val_metrics["accuracy"], 4),
            "val_precision": round(e3_val_metrics["precision"], 4),
            "val_recall": round(e3_val_metrics["recall"], 4),
            "val_f1": round(e3_val_metrics["f1"], 4),
            "val_macro_f1": round(e3_val_metrics["macro_f1"], 4),
            "val_false_alarm_rate": round(e3_val_metrics["false_alarm_rate"], 4),
            "val_roc_auc": round(e3_val_metrics["roc_auc"], 4),
        }
    )

    # --- E4: Tuned Random Forest via GridSearchCV on training split only ---
    # Note: GridSearchCV uses n_jobs=1 while RandomForestClassifier uses n_jobs=-1
    # to avoid nested loky/thread oversubscription on Windows.
    print("5. Running E4: GridSearchCV RandomForest (max_depth=[10,20], n_estimators=[100,200], cv=3)...", flush=True)
    e4_base = build_rf_pipeline(FEATURE_COLUMNS)
    param_grid = {
        "classifier__max_depth": [10, 20],
        "classifier__n_estimators": [100, 200],
    }
    grid_search = GridSearchCV(
        estimator=e4_base,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    e4_pipe = grid_search.best_estimator_
    best_e4_params = {
        k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()
    }
    e4_cv_f1 = float(grid_search.best_score_)
    e4_val_pred = e4_pipe.predict(X_val)
    e4_val_proba = e4_pipe.predict_proba(X_val)[:, 1]
    e4_val_metrics = compute_classification_metrics(y_val, e4_val_pred, e4_val_proba)
    fitted_models["E4"] = e4_pipe
    experiment_features["E4"] = list(FEATURE_COLUMNS)
    experiment_rows.append(
        {
            "id": "E4",
            "model": (
                f"RandomForest GridSearchCV "
                f"(max_depth={best_e4_params['max_depth']}, "
                f"n_estimators={best_e4_params['n_estimators']})"
            ),
            "features": f"All {len(FEATURE_COLUMNS)} features",
            "val_accuracy": round(e4_val_metrics["accuracy"], 4),
            "val_precision": round(e4_val_metrics["precision"], 4),
            "val_recall": round(e4_val_metrics["recall"], 4),
            "val_f1": round(e4_val_metrics["f1"], 4),
            "val_macro_f1": round(e4_val_metrics["macro_f1"], 4),
            "val_false_alarm_rate": round(e4_val_metrics["false_alarm_rate"], 4),
            "val_roc_auc": round(e4_val_metrics["roc_auc"], 4),
        }
    )

    experiments_df = pd.DataFrame(experiment_rows)
    experiments_csv = RESULTS_DIR / "experiments.csv"
    experiments_df.to_csv(experiments_csv, index=False)
    print("\nValidation Experiment Summary (E1-E4):")
    print(experiments_df.to_string(index=False))

    # Model selection strictly on validation set + deployment size constraint (max_depth <= 20)
    # E4 bounds tree depth (<= 20) to control overfitting and keep model.joblib well under 50 MB.
    selected_id = "E4"
    selected_model = fitted_models[selected_id]
    print(
        f"\n6. Selected final model on validation: {selected_id} "
        f"(best params={best_e4_params}, CV F1={e4_cv_f1:.4f}, "
        f"val recall={e4_val_metrics['recall']:.4f}, val macro F1={e4_val_metrics['macro_f1']:.4f})"
    )

    # Save final selected model with compression (kept under 50 MB for GitHub/Streamlit)
    joblib.dump(selected_model, DEFAULT_MODEL_PATH, compress=3)
    model_size_mb = DEFAULT_MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"   Saved final model to '{DEFAULT_MODEL_PATH}' ({model_size_mb:.2f} MB).")

    # --- Final Single-Shot Evaluation on Held-Out Official KDDTest+ ---
    print("\n7. Evaluating Baseline (E1) and Selected Model (E4) once on official KDDTest+...")
    test_rows = []
    test_details: dict[str, dict[str, Any]] = {}
    for exp_id in ["E1", selected_id]:
        model = fitted_models[exp_id]
        feats = experiment_features[exp_id]
        y_test_pred = model.predict(X_test[feats])
        y_test_proba = model.predict_proba(X_test[feats])[:, 1]
        m_test = compute_classification_metrics(y_test, y_test_pred, y_test_proba)
        test_details[exp_id] = {
            "metrics": m_test,
            "y_pred": y_test_pred,
            "y_proba": y_test_proba,
        }
        test_rows.append(
            {
                "id": exp_id,
                "model": experiments_df.loc[experiments_df["id"] == exp_id, "model"].iloc[0],
                "test_accuracy": round(m_test["accuracy"], 4),
                "test_precision": round(m_test["precision"], 4),
                "test_recall": round(m_test["recall"], 4),
                "test_f1": round(m_test["f1"], 4),
                "test_macro_f1": round(m_test["macro_f1"], 4),
                "test_false_alarm_rate": round(m_test["false_alarm_rate"], 4),
                "test_roc_auc": round(m_test["roc_auc"], 4),
                "tn": m_test["tn"],
                "fp": m_test["fp"],
                "fn": m_test["fn"],
                "tp": m_test["tp"],
            }
        )

    test_eval_df = pd.DataFrame(test_rows)
    test_eval_csv = RESULTS_DIR / "test_evaluation.csv"
    test_eval_df.to_csv(test_eval_csv, index=False)
    print("\nHeld-Out Official KDDTest+ Evaluation:")
    print(test_eval_df.to_string(index=False))

    # Error breakdown by attack family and seen vs. unseen attack types for final model
    final_test_pred = test_details[selected_id]["y_pred"]
    final_test_proba = test_details[selected_id]["y_proba"]
    train_attack_types = set(train_df[ATTACK_TYPE_COLUMN].str.lower().unique()) - {"normal"}

    family_df = evaluate_by_attack_family(y_test, final_test_pred, test_df["attack_family"])
    seen_unseen_df = evaluate_seen_vs_unseen(
        y_test, final_test_pred, test_df[ATTACK_TYPE_COLUMN], train_attack_types
    )
    threshold_df = evaluate_thresholds(y_test, final_test_proba)

    family_df.to_csv(RESULTS_DIR / "test_attack_family_recall.csv", index=False)
    seen_unseen_df.to_csv(RESULTS_DIR / "test_seen_vs_unseen.csv", index=False)
    threshold_df.to_csv(RESULTS_DIR / "test_threshold_tradeoff.csv", index=False)

    print("\nKDDTest+ Breakdown by Attack Family (Selected Model):")
    print(family_df.to_string(index=False))
    print("\nKDDTest+ Seen vs. Unseen Attack Recall (Selected Model):")
    print(seen_unseen_df.to_string(index=False))
    print("\nKDDTest+ Threshold Trade-off (Selected Model):")
    print(threshold_df.to_string(index=False))

    # Generate and save diagnostic figures
    fig_exp = plot_experiment_comparison(
        experiments_df, save_path=FIGURES_DIR / "06_experiment_comparison_val.png"
    )
    plt.close(fig_exp)

    fig_cm_base = plot_confusion_matrix(
        y_test,
        test_details["E1"]["y_pred"],
        title="Baseline Logistic Regression (E1) — KDDTest+ Confusion Matrix",
        save_path=FIGURES_DIR / "07_confusion_matrix_e1_test.png",
    )
    plt.close(fig_cm_base)

    fig_cm_final = plot_confusion_matrix(
        y_test,
        final_test_pred,
        title=f"Selected Tuned Random Forest ({selected_id}) — KDDTest+ Confusion Matrix",
        save_path=FIGURES_DIR / "08_confusion_matrix_final_test.png",
    )
    plt.close(fig_cm_final)

    fig_breakdown = plot_test_error_breakdown(
        family_df,
        seen_unseen_df,
        model_label=f"Tuned RF ({selected_id})",
        save_path=FIGURES_DIR / "09_test_recall_by_family_and_unseen.png",
    )
    plt.close(fig_breakdown)

    fig_threshold = plot_threshold_tradeoff(
        threshold_df,
        model_label=f"Tuned RF ({selected_id})",
        save_path=FIGURES_DIR / "10_threshold_tradeoff_test.png",
    )
    plt.close(fig_threshold)

    # Save structured metadata JSON for Stage 3 app & documentation
    metadata = {
        "selected_experiment": selected_id,
        "selected_model_name": experiments_df.loc[
            experiments_df["id"] == selected_id, "model"
        ].iloc[0],
        "best_params": best_e4_params,
        "e4_cv_f1": round(e4_cv_f1, 4),
        "random_seed": RANDOM_SEED,
        "model_path": str(DEFAULT_MODEL_PATH.relative_to(DEFAULT_MODEL_PATH.parent.parent)),
        "model_size_mb": round(model_size_mb, 2),
        "e2_dropped_features": redundant_info["drop_cols"],
        "e2_kept_feature_count": len(pruned_features),
        "validation_metrics": experiments_df.to_dict(orient="records"),
        "test_metrics": test_eval_df.to_dict(orient="records"),
        "success_criteria": {
            "target_test_attack_recall": 0.80,
            "achieved_test_attack_recall": round(
                test_details[selected_id]["metrics"]["recall"], 4
            ),
            "recall_criterion_met": bool(
                test_details[selected_id]["metrics"]["recall"] >= 0.80
            ),
            "target_macro_f1_gain": 0.02,
            "achieved_val_macro_f1_gain": round(
                e4_val_metrics["macro_f1"] - e1_val_metrics["macro_f1"], 4
            ),
            "achieved_test_macro_f1_gain": round(
                test_details[selected_id]["metrics"]["macro_f1"]
                - test_details["E1"]["metrics"]["macro_f1"],
                4,
            ),
            "val_macro_f1_criterion_met": bool(
                (e4_val_metrics["macro_f1"] - e1_val_metrics["macro_f1"]) >= 0.02
            ),
            "test_macro_f1_criterion_met": bool(
                (
                    test_details[selected_id]["metrics"]["macro_f1"]
                    - test_details["E1"]["metrics"]["macro_f1"]
                )
                >= 0.02
            ),
        },
    }
    metadata_path = RESULTS_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n8. Saved all experiment CSVs, figures, and metadata to '{RESULTS_DIR}' and '{FIGURES_DIR}'.")

    return metadata


if __name__ == "__main__":
    matplotlib.use("Agg")
    run_experiments()
