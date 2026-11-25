# Team Contribution Matrix

**Project:** Network Intrusion Detection Prototype (CUSIT — *Programming for Artificial Intelligence, Fall 2026*)  
**Repository:** `https://github.com/Mehar-ArbaZ/network-intrusion-detection-prototype`

---

## 1. Team Roster & Primary Ownership

| Team Member | GitHub / Git Identity | Role | Primary Stage & Scope |
| --- | --- | --- | --- |
| **Muhammad Arbaz** | `Mehar-ArbaZ` (`arbazfayyaz999@gmail.com`) | Project Lead & ML Developer | Repository setup, Stage 2 leak-free modeling & evaluation pipeline (`E1`–`E4`), PR review & `main` integration |
| **Muhammad Huzaifa Sohail** | `iamhuzaifa` (`imhuxaifa@gmail.com`) | Data Engineer / Analyst | Stage 1 problem proposal, NSL-KDD vs. KDD Cup 99 dataset audit, data dictionary, schema/loader/preprocessing modules, EDA notebook & figures |
| **Muhammad Umar Siddiqi** | `Muhammad Umar Siddiqi` (`m.umar1siddiqui@gmail.com`) | UI, Deployment & QA Lead | Stage 3 inference module, interactive Streamlit triage prototype, automated `pytest` edge-case suite, error log, contribution matrix, README & deployment guide |

---

## 2. Stage-by-Stage Deliverable & Git Traceability Matrix

| Stage | Owner | Feature Branch / PR | Key Files & Artifacts Delivered | Verifiable Git Commits |
| --- | --- | --- | --- | --- |
| **Setup** | Muhammad Arbaz | `main` | Project folder scaffolding, `.gitignore`, `requirements.txt`, initial `README.md` | `3fb681d` |
| **Stage 1: Data Audit & EDA** | Muhammad Huzaifa Sohail | `feature/huzaifa-stage1-audit` (PR #1 → `50a9576`) | `docs/proposal.md`, `docs/project_log.md`, `docs/dataset_audit.md`, `docs/data_dictionary.md`, `src/config.py`, `src/load_data.py`, `src/preprocess.py`, `data/processed/feature_summary.csv`, `data/sample/sample_connections.csv`, `notebooks/01_audit.ipynb`, `notebooks/02_eda.ipynb`, `reports/figures/01–05` | `21ddd9f`, `3bf20de`, `99bcf58`, `d94ad3e`, `b4fe742` |
| **Stage 2: Modeling & Evaluation** | Muhammad Arbaz | `feature/arbaz-stage2-models` (PR #2 → `789c7ad`) | `src/evaluate.py`, `src/train.py`, `models/model.joblib` (`3.58 MB`), `notebooks/03_models.ipynb`, `reports/figures/06–10`, `reports/results/*.csv`, `reports/results/model_metadata.json`, `docs/experiments.md`, `docs/project_log.md` | `0c6bdee`, `424053d`, `15e2f92`, `f673f6c` |
| **Stage 3: App, QA & Documentation** | Muhammad Umar Siddiqi | `feature/umar-stage3-app-qa` (PR #3) | `src/predict.py`, `app/app.py`, `tests/test_edge_cases.py`, `docs/error_log.md`, `docs/contribution_matrix.md`, updated `docs/project_log.md`, final `README.md` | Stage 3 commits on `feature/umar-stage3-app-qa` |

---

## 3. Detailed Individual Contribution Breakdown

### Muhammad Arbaz (Project Lead & ML Developer)
- Initialized the public repository structure, dependency manifest (`requirements.txt`), and `.gitignore` rules keeping raw NSL-KDD files out of version control.
- Built `src/evaluate.py` with binary classification metrics, false-alarm rate, subgroup recall by intrusion family (`DoS`, `Probe`, `R2L`, `U2R`), seen-vs.-unseen attack analysis, threshold trade-off tables, and diagnostic matplotlib/seaborn plotting utilities.
- Built `src/train.py` executing leak-free `scikit-learn` `Pipeline` experiments `E1`–`E4` on an 80/20 stratified `KDDTrain+` validation split (`random_state=42`), tuned Random Forest via 3-fold `GridSearchCV`, exported compressed `models/model.joblib` (`3.58 MB`), and ran single-shot evaluation on held-out `KDDTest+`.
- Executed `notebooks/03_models.ipynb`, generated Figures `06`–`10` and `reports/results/*`, documented `docs/experiments.md`, and coordinated branch integration into `main`.

### Muhammad Huzaifa Sohail (Data Engineer / Analyst)
- Authored `docs/proposal.md` with stakeholder definition, 6 cited academic references, target/feature specification, and measurable success criteria.
- Acquired NSL-KDD (`KDDTrain+.TXT`, `KDDTest+.TXT`), empirically compared it against KDD Cup 99 (`70.53%` duplicates in the 10% subset), verified all 41 feature names against official `KDDTrain+.arff`, and documented `docs/dataset_audit.md` and `docs/data_dictionary.md`.
- Implemented `src/config.py`, `src/load_data.py` (strict schema, file, and numeric validation), and `src/preprocess.py` (leak-free `ColumnTransformer` with imputation, `OneHotEncoder(handle_unknown="ignore")`, optional `StandardScaler`, and train-only redundancy detection).
- Generated `data/processed/feature_summary.csv`, `data/sample/sample_connections.csv` (100 verified rows), executed `notebooks/01_audit.ipynb` and `notebooks/02_eda.ipynb`, and saved EDA Figures `01`–`05`.

### Muhammad Umar Siddiqi (UI, Deployment & QA Lead)
- Implemented `src/predict.py` providing validated model loading, threshold validation, schema alignment, metadata stripping, single-record scoring, DataFrame batch scoring, and headered/headerless CSV ingestion.
- Built the interactive Streamlit triage application (`app/app.py`) featuring the mandatory safety disclaimer, sidebar decision-threshold slider (`0.05–0.95`), Sample Record Explorer, Batch CSV Upload with protocol/service filters and downloadable predictions, and Stage 2 Benchmark Summary tab.
- Authored and verified the automated `pytest` edge-case and reliability suite (`tests/test_edge_cases.py`) covering missing numeric `NaN`s, unseen `service` categories, wrong column counts, empty/nonexistent files, non-numeric text in numeric columns, threshold monotonicity, and metadata stripping.
- Authored `docs/error_log.md`, `docs/contribution_matrix.md`, updated `docs/project_log.md`, and finalized `README.md` with reproducible instructions, Stage 2 results, Streamlit Community Cloud deployment steps, limitations, and AI-assistance disclosure.

---

## 4. Cross-Team Review & Defence Readiness Sign-Off

Every team member has reviewed the integrated repository across all three stages and is prepared to explain the following during the project defence and individual viva:

- [x] **Dataset choice & leakage prevention:** Why NSL-KDD replaced KDD Cup 99, why `difficulty` and `attack_type` are excluded, and why preprocessing fits strictly on the training fold.
- [x] **Experiment design & unseen-attack gap:** Why `E4` Tuned Random Forest (`max_depth=20`, `n_estimators=200`) was selected over `E1`–`E3`, why default `0.50` test recall drops on 17 unseen attack types, and how threshold adjustment leverages `0.9652` ROC-AUC.
- [x] **Inference validation & QA reliability:** How `src/predict.py`, `app/app.py`, and `tests/test_edge_cases.py` handle malformed inputs, missing values, and unseen categories gracefully without retraining inside the UI.
- [x] **Ethical & scope boundaries:** Why this 1998-era benchmark prototype is strictly a human-in-the-loop analyst triage aid and must never automate traffic blocking or disciplinary actions.
