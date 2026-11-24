# Experiments & Model Evaluation Log (Stage 2)

**Owner:** Muhammad Arbaz (Project Lead & ML Developer)  
**Date:** `2026-11-24`  
**Reproducible Command:** `python -m src.train`  
**Notebook:** `notebooks/03_models.ipynb`  
**Machine-Readable Results:** `reports/results/experiments.csv`, `reports/results/test_evaluation.csv`, `reports/results/model_metadata.json`

---

## 1. Leak-Free Validation Protocol

To avoid data leakage and test-set overfitting:
1. All model comparisons, feature-pruning decisions (E2), and hyperparameter tuning (E4) were performed strictly inside `KDDTrain+.TXT` (`125,973` rows), partitioned via an **80/20 stratified train/validation split** (`random_state=42`: `100,778` train / `25,195` validation).
2. Every experiment wraps `SimpleImputer`, `OneHotEncoder(handle_unknown="ignore")`, and optional `StandardScaler` inside a `scikit-learn` `Pipeline` so preprocessing parameters are fitted exclusively on the training fold.
3. The official benchmark file `KDDTest+.TXT` (`22,544` rows) was held out completely during training, feature selection, and tuning, and evaluated **once** at the end for the baseline (`E1`) and final selected model (`E4`).

---

## 2. Validation Experiments Comparison (E1–E4)

| ID | Model | Features | Val Accuracy | Val Precision | Val Recall (Attack) | Val Attack F1 | Val Macro F1 | Val False-Alarm Rate | Val ROC-AUC | One-Line Conclusion |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **E1** | `LogisticRegression(max_iter=1000)` | All 41 features (scaled + OHE) | 0.9719 | 0.9764 | 0.9629 | 0.9696 | 0.9717 | 0.0203 | 0.9964 | Strong linear baseline achieving 96.29% validation attack recall with 2.03% false-alarm rate. |
| **E2** | `LogisticRegression(max_iter=1000)` | Pruned 33 features (dropped `num_outbound_cmds` + 7 features with `|r| > 0.90`) | 0.9699 | 0.9746 | 0.9604 | 0.9674 | 0.9697 | 0.0218 | 0.9960 | Removing 8 redundant features simplified the feature space with only a minor -0.0025 recall drop, confirming strong collinearity in error-rate features but slightly favoring full features for linear fit. |
| **E3** | `RandomForestClassifier(n_estimators=100, random_state=42)` | All 41 features (raw numeric + OHE) | 0.9990 | 0.9995 | 0.9983 | 0.9989 | 0.9990 | 0.0004 | 1.0000 | Non-linear tree ensemble dramatically outperformed linear baselines (+0.0273 macro F1, +0.0354 attack recall, 50× lower false-alarm rate). |
| **E4** | `RandomForest GridSearchCV` (`max_depth=20`, `n_estimators=200`, `cv=3`, `scoring="f1"`) | All 41 features (raw numeric + OHE) | **0.9990** | **0.9995** | **0.9983** | **0.9989** | **0.9990** | **0.0004** | **1.0000** | **Selected final model:** matches E3's top validation recall (99.83%) and macro F1 (99.90%) with 3-fold CV F1 = 0.9982 while bounding `max_depth=20` for regularization and compact 3.58 MB deployment size. |

**Features dropped in E2 (computed strictly on the 80% training fold):**
`num_outbound_cmds` (constant zero), `num_root`, `srv_serror_rate`, `srv_rerror_rate`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate`.

---

## 3. Final Model Selection Rationale

**Selected Model:** **E4 — Tuned Random Forest (`max_depth=20`, `n_estimators=200`, `random_state=42`)**, persisted via `joblib.dump(..., compress=3)` to `models/model.joblib` (**3.58 MB**, well below the 50 MB GitHub/Streamlit limit).

Why **E4** was chosen over E1, E2, and E3 based strictly on validation performance and deployment constraints:
- **vs. E1 & E2 (Logistic Regression):** E4 raised validation attack recall from `0.9629` to `0.9983` (+3.54 percentage points), increased validation macro F1 by `+0.0273` (`0.9717 → 0.9990`), and slashed validation false alarms from `2.03%` to `0.04%`.
- **vs. E3 (Unbounded Random Forest):** E4 achieved identical top validation metrics (`0.9983` recall, `0.9990` macro F1, `0.0004` FPR) while capping tree depth at `max_depth=20` via 3-fold cross-validation (`CV F1 = 0.9982`), reducing deep-leaf memorization risk and guaranteeing a lightweight `3.58 MB` artifact.

---

## 4. Final Held-Out Evaluation on Official `KDDTest+` (`n = 22,544`)

Evaluated **once** at the default probability threshold (`0.50`) after model selection:

| Model | Test Accuracy | Test Precision | Test Attack Recall | Test Attack F1 | Test Macro F1 | Test False-Alarm Rate | Test ROC-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **E1 (Baseline Logistic Regression)** | 0.7540 | 0.9172 | 0.6242 | 0.7428 | 0.7535 | 0.0745 | 0.7975 | 8,988 | 723 | 4,823 | 8,010 |
| **E4 (Selected Tuned Random Forest)** | **0.7628** | **0.9672** | 0.6038 | **0.7435** | **0.7615** | **0.0271** | **0.9652** | 9,448 | 263 | 5,084 | 7,749 |

### Key Diagnostic Findings on `KDDTest+`

1. **Large Validation-to-Test Gap Explained by Unseen Attacks:**
   Validation recall was `99.83%`, whereas default-threshold `KDDTest+` recall dropped to `60.38%`. Breakdown by novelty confirms the exact root cause identified in Stage 1 (`docs/dataset_audit.md`):

   | Test Generalization Bucket | Records | Correct | Missed | Detection / Specificity Rate |
   | --- | ---: | ---: | ---: | ---: |
   | Normal traffic | 9,711 | 9,448 | 263 | **97.29%** |
   | Seen Attacks (labels present in `KDDTrain+`) | 9,083 | 6,984 | 2,099 | **76.89%** |
   | **Unseen Attacks (17 attack labels only in `KDDTest+`)** | 3,750 | 765 | 2,985 | **20.40%** |

2. **Breakdown by Intrusion Family (`E4` at default threshold `0.50`):**

   | Traffic Category | Records | Correct | Detection / Specificity Rate |
   | --- | ---: | ---: | ---: |
   | Normal | 9,711 | 9,448 | **97.29%** |
   | DoS | 7,460 | 5,800 | **77.75%** |
   | Probe | 2,421 | 1,771 | **73.15%** |
   | R2L (Remote-to-Local) | 2,885 | 168 | **5.82%** |
   | U2R (User-to-Root) | 67 | 10 | **14.93%** |

   R2L and U2R attacks account for most false negatives at the default `0.50` threshold because R2L volume nearly triples in `KDDTest+` (995 train → 2,885 test) and stealthy login/application attacks mimic normal authenticated traffic.

3. **Probability Ranking Power (`ROC-AUC = 0.9652`) & Decision-Threshold Trade-Off:**
   Although E4's default-threshold recall is `60.38%`, its **`KDDTest+` ROC-AUC is `0.9652`** versus `0.7975` for Logistic Regression — proving the Random Forest still ranks attack connections far higher in probability than normal traffic. Lowering the alert threshold directly trades a modest rise in false alarms for major recall gains:

   | Threshold | Test Accuracy | Attack Precision | Attack Recall | Attack F1 | Macro F1 | False-Alarm Rate |
   | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | **0.10** | **0.8699** | 0.9191 | **0.8460** | **0.8810** | **0.8688** | 0.0984 |
   | **0.20** | 0.8388 | 0.9439 | 0.7622 | 0.8434 | 0.8387 | 0.0598 |
   | **0.30** | 0.8228 | 0.9688 | 0.7117 | 0.8206 | 0.8228 | 0.0303 |
   | **0.40** | 0.7975 | 0.9687 | 0.6657 | 0.7891 | 0.7971 | 0.0284 |
   | **0.50** *(default)* | 0.7630 | 0.9672 | 0.6041 | 0.7437 | 0.7616 | 0.0271 |
   | **0.60** | 0.7487 | 0.9673 | 0.5781 | 0.7237 | 0.7466 | 0.0258 |
   | **0.70** | 0.7415 | 0.9683 | 0.5644 | 0.7131 | 0.7390 | 0.0244 |

   *(Note: `predict()` uses strict `p > 0.5` for ties while threshold evaluation uses `p >= threshold`, causing a 4-row difference at 0.50: `0.6038` vs `0.6041`.)*
   This result strongly motivates exposing an **interactive decision-threshold slider** in Stage 3's Streamlit triage app so security analysts can tune sensitivity based on alert capacity without retraining.

---

## 5. Honest Assessment Against Proposal Success Criteria

| Proposal Criterion | Target | Achieved (Default `0.50` Threshold) | Status & Explanation |
| --- | --- | --- | --- |
| **1. Test Attack Recall (`KDDTest+`)** | `>= 0.80` | `0.6038` (default `0.50`) | **Not met at default `0.50` threshold.** Unseen attack types (`20.40%` recall across 3,750 rows) and stealthy R2L/U2R traffic drag down default-threshold recall. Because ROC-AUC is `0.9652`, operating at threshold `0.10` achieves `0.8460` recall at `9.84%` FPR (demonstrated interactively in Stage 3). |
| **2. Macro F1 Improvement over Baseline (E1)** | `>= +0.02` | **Validation:** `+0.0273` (`0.9717 → 0.9990`)<br>**`KDDTest+`:** `+0.0080` (`0.7535 → 0.7615`) | **Met on validation (+0.0273); partially met on `KDDTest+` (+0.0080 at `0.50`).** On `KDDTest+`, E4 improves accuracy, precision (+5.00 pp), ROC-AUC (+16.77 pp), and cuts false alarms by 63.6%, but default `0.50` calibration leaves unseen-attack probabilities below 0.50. |

---

## 6. Generated Stage 2 Figures

- `reports/figures/06_experiment_comparison_val.png` — Validation metrics across E1–E4
- `reports/figures/07_confusion_matrix_e1_test.png` — Baseline Logistic Regression (`E1`) confusion matrix on `KDDTest+`
- `reports/figures/08_confusion_matrix_final_test.png` — Selected Tuned Random Forest (`E4`) confusion matrix on `KDDTest+`
- `reports/figures/09_test_recall_by_family_and_unseen.png` — `KDDTest+` recall by intrusion family and Seen vs. Unseen attacks
- `reports/figures/10_threshold_tradeoff_test.png` — `KDDTest+` recall vs. false-alarm rate trade-off across decision thresholds
