# Engineering Error & QA Log

**Owner:** Muhammad Umar Siddiqi (UI, Deployment & QA Lead)  
**Scope:** End-to-end issues encountered, root causes, implemented fixes, and edge-case verification across Stages 1–3 of the Network Intrusion Detection Prototype.  
**Verification Suite:** `pytest -v` (`tests/test_edge_cases.py`)

---

## 1. Logged Engineering Issues & Resolutions

| ID | Stage | Component | Issue / Failure Mode | Root Cause | Resolution | Verification |
| --- | --- | --- | --- | --- | --- | --- |
| **ERR-01** | Stage 1 | Data Acquisition (`docs/dataset_audit.md`) | Original UNB direct download link for NSL-KDD was no longer hosted. | Upstream archive link deprecated on the UNB dataset page. | Acquired `KDDTrain+.txt` (`125,973` rows) and `KDDTest+.txt` (`22,544` rows) from the public GitHub mirror (`jmnwong/NSL-KDD-Dataset`) and verified all 41 feature names + 2 metadata columns against official `KDDTrain+.arff`. | `notebooks/01_audit.ipynb` shape & schema assertions passed. |
| **ERR-02** | Stage 1 / 3 | Schema & Leakage (`src/config.py`, `src/load_data.py`, `src/predict.py`) | Raw `.TXT` files have no header row and append `attack_type` and `difficulty` (`0–21`), risking label-hardness leakage or column mismatch during CSV upload. | NSL-KDD distributes 43 headerless columns where `difficulty` is a post-hoc benchmark score (not an observable network attribute). | Defined canonical 41 `FEATURE_COLUMNS` and 43 `RAW_COLUMNS` in `src/config.py`; enforced strict validation in `src/load_data.py`; stripped `OPTIONAL_METADATA_COLUMNS` (`attack_type`, `difficulty`, `label`, `attack_family`) automatically in `src/preprocess.py` and `src/predict.py`. | `test_optional_nsl_kdd_metadata_columns_stripped_cleanly` passed. |
| **ERR-03** | Stage 1 / 3 | Preprocessing (`src/preprocess.py`) | Missing numeric values (`NaN`) or unseen categorical strings (e.g., new `service` names in external uploads or CV folds) crash unguarded `scikit-learn` estimators. | `KDDTrain+` contains 6 train-only `service` values, and production/user CSVs may contain unseen categories or missing cells. | Wrapped `SimpleImputer` (`median` for numeric, `most_frequent` for categorical) and `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` inside `ColumnTransformer` inside every modeling `Pipeline`. | `test_edge_case_1_missing_numeric_nan_handled` and `test_edge_case_2_unseen_service_category_handled` passed. |
| **ERR-04** | Stage 2 | Training (`src/train.py`) | Nested parallelism (`GridSearchCV(n_jobs=-1)` wrapping `RandomForestClassifier(n_jobs=-1)`) caused loky worker/thread oversubscription and hangs on Windows. | Multiple joblib worker pools competing for cores under Windows process spawning. | Configured `GridSearchCV(..., n_jobs=1)` while keeping parallelism inside `RandomForestClassifier(n_jobs=-1)`. | Full `python -m src.train` pipeline completed cleanly and reproducibly with `random_state=42`. |
| **ERR-05** | Stage 2 | Model Persistence (`models/model.joblib`) | Unbounded Random Forest trees (`max_depth=None`) risk deep-leaf overfitting and oversized `.joblib` artifacts exceeding GitHub/Streamlit Cloud limits (`<50 MB`). | Deep fully-grown trees memorize rare training patterns and inflate serialized tree arrays. | Selected tuned `E4` (`max_depth=20`, `n_estimators=200`) via 3-fold CV (`CV F1 = 0.9982`) and serialized with `joblib.dump(..., compress=3)`. | `models/model.joblib` verified at `3.58 MB` with identical top validation metrics (`0.9990` Macro F1). |
| **ERR-06** | Stage 2 / 3 | Generalization & Thresholding (`docs/experiments.md`, `src/predict.py`, `app/app.py`) | Default-threshold (`0.50`) attack recall dropped from `99.83%` (validation) to `60.38%` (`KDDTest+`), and `predict()` (`p > 0.5`) differed by 4 rows from `p >= 0.50` (`0.6038` vs. `0.6041`). | `KDDTest+` adds 17 unseen attack types (`3,750` rows, `20.40%` recall at `0.50`) and nearly triples R2L traffic (`995 → 2,885`), while `predict()` uses strict `>` on ties. | Reported the unseen-attack gap honestly without test-set tuning; standardized inference on `probas >= threshold` in `src/predict.py`; exposed an interactive threshold slider (`0.05–0.95`) in `app/app.py` leveraging `E4`'s `0.9652` `KDDTest+` ROC-AUC (`84.60%` recall at threshold `0.10`). | `test_threshold_sensitivity_and_bounds_validation` passed; verified against `reports/results/test_threshold_tradeoff.csv`. |
| **ERR-07** | Stage 3 | Streamlit Packaging (`app/app.py`) | Running `streamlit run app/app.py` raises `ModuleNotFoundError: No module named 'src'` when `app/` is placed first on `sys.path`. | Streamlit sets the script directory (`app/`) as `sys.path[0]` rather than the repository root. | Inserted `ROOT_DIR = Path(__file__).resolve().parent.parent` into `sys.path` at module startup in `app/app.py`. | Verified clean headless startup and imports of `app/app.py` via `streamlit run app/app.py`. |
| **ERR-08** | Stage 3 | Input Validation & UI Error Handling (`src/predict.py`, `app/app.py`) | Uploading empty (0-byte) CSVs, wrong column counts, duplicate/unexpected columns, non-numeric text in numeric fields, `±inf` values, or out-of-range thresholds caused raw Python tracebacks. | Unvalidated user input passed directly into `pandas` / `scikit-learn` `predict_proba`. | Added strict validation helpers (`validate_threshold`, `prepare_inference_dataframe`, `read_inference_csv`) with readable `ValueError`/`FileNotFoundError` messages naming the offending column or schema mismatch, and caught them in `app/app.py` via `st.error(...)`. | All 5 mandatory edge-case tests (`test_edge_case_1`–`5`) and reliability tests passed in `pytest -v`. |

---

## 2. Automated Edge-Case & Reliability Test Matrix

All automated checks live in `tests/test_edge_cases.py` and exercise the persisted Stage 2 pipeline (`models/model.joblib`) through `src/predict.py` and `src/load_data.py`.

| Test Function | Scenario Covered | Expected Behavior | Status |
| --- | --- | --- | --- |
| `test_edge_case_1_missing_numeric_nan_handled` | `NaN` injected into `src_bytes` and `serror_rate` | Pipeline `SimpleImputer` imputes cleanly; returns finite probabilities in `[0, 1]` and valid labels without crashing | Passed |
| `test_edge_case_2_unseen_service_category_handled` | Unseen string `"brand_new_unseen_service_xyz"` in `service` | `OneHotEncoder(handle_unknown="ignore")` encodes cleanly; returns valid single-record prediction dict | Passed |
| `test_edge_case_3_wrong_column_count_raises_readable_error` | CSV missing 3 columns (`duration`, `src_bytes`, `service`) and CSV with 1 unexpected extra column | Raises `ValueError` naming expected count (`41`) and offending column names | Passed |
| `test_edge_case_4_empty_and_nonexistent_file_raise_clear_errors` | 0-byte empty CSV and nonexistent file path via both `predict_csv` and `load_nsl_kdd` | Raises `ValueError` matching `"empty"` for 0-byte file and `FileNotFoundError` matching `"not found"` for missing path | Passed |
| `test_edge_case_5_non_numeric_text_in_numeric_column_rejected` | Text string `"not_a_number"` placed inside numeric `src_bytes` column | Raises `ValueError` explicitly naming `"src_bytes"` | Passed |
| `test_sample_batch_scoring_returns_100_valid_predictions` | Batch scoring `data/sample/sample_connections.csv` (100 rows) | Returns 100 rows with 4 prediction columns + 41 canonical features and valid probabilities | Passed |
| `test_threshold_sensitivity_and_bounds_validation` | Threshold `0.10` vs. `0.80` monotonicity; invalid thresholds (`-0.01`, `1.01`, `NaN`, `True`) | Lower threshold flags `>=` rows than higher threshold; invalid thresholds raise `ValueError` | Passed |
| `test_optional_nsl_kdd_metadata_columns_stripped_cleanly` | Labeled CSV containing `attack_type` and `difficulty` alongside 41 features | Strips optional metadata safely before inference without error or leakage | Passed |
| `test_unseen_official_kddtest_scoring` | Optional full `KDDTest+.TXT` (`22,544` rows) scoring when raw file is present locally | Scores all `22,544` rows cleanly with valid probabilities (auto-skipped cleanly if `data/raw/KDDTest+.TXT` is absent in CI/clone) | Passed / Auto-skipped when raw file omitted |

---

## 3. Manual UI & Deployment Smoke Checks (`app/app.py`)

- **Model caching:** `@st.cache_resource` loads `models/model.joblib` (`3.58 MB`) once per app session; zero model retraining occurs inside the app.
- **Safety banner:** Prominent research-prototype disclaimer displayed at the top of every view (`render_disclaimer()`).
- **Interactive threshold slider:** Sidebar slider (`0.05`–`0.95`, step `0.05`, default `0.50`) dynamically updates both single-record decisions and batch CSV triage tables.
- **Error resilience:** Malformed uploads (empty file, missing columns, invalid numeric values) display readable `st.error(...)` alerts instead of crashing the Streamlit session.
- **CSV export:** Batch triage results download cleanly as `nids_predictions_thr_<threshold>.csv` with `prediction`, `predicted_label`, `attack_probability`, and all 41 validated feature columns.
